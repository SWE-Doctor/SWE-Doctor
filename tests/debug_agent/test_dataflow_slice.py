from debug_agent.traceback_parser import TracebackFrame
from debug_agent.dataflow_slice import label_frames


def _frame(file, lineno, qualname="", is_test=False):
    return TracebackFrame(file, lineno, qualname, "", is_test)


SRC_SOURCE = """\
def make_bad():
    ebooks = [1, 2, 3, 3]
    return ebooks
"""

SRC_TRANSFORM = """\
def transform(ebooks):
    cleaned = [e for e in ebooks if e]
    return cleaned
"""

SRC_PASS = """\
def forward(ebooks):
    result = ebooks
    return result
"""

SRC_SINK = """\
def test_dedup():
    result = forward(make_bad())
    assert len(result) == 2
"""


def _resolver(files: dict[str, str]):
    return lambda p: files.get(p)


def test_labels_source_transform_passthrough_sink():
    files = {
        "mk.py": SRC_SOURCE, "tx.py": SRC_TRANSFORM,
        "pt.py": SRC_PASS,   "tests/test_s.py": SRC_SINK,
    }
    frames = [
        _frame("tests/test_s.py", 3, "test_dedup", is_test=True),
        _frame("pt.py", 3, "forward"),
        _frame("tx.py", 3, "transform"),
        _frame("mk.py", 2, "make_bad"),
    ]
    labeled = label_frames(frames, _resolver(files))
    roles = {l["file"]: l["role"] for l in labeled}
    assert roles["tests/test_s.py"] == "sink"
    assert roles["mk.py"] == "source"
    assert roles["tx.py"] == "transform"
    assert roles["pt.py"] == "pass_through"


def test_unreadable_source_yields_unknown():
    frames = [_frame("missing.py", 10, "f")]
    labeled = label_frames(frames, _resolver({}))
    assert labeled[0]["role"] == "unknown"
    assert "source-unreadable" in labeled[0]["reason"]


def test_ast_parse_error_yields_unknown():
    frames = [_frame("bad.py", 1, "f")]
    labeled = label_frames(frames, _resolver({"bad.py": "def :::: broken"}))
    assert labeled[0]["role"] == "unknown"
    assert "ast-parse-error" in labeled[0]["reason"]
