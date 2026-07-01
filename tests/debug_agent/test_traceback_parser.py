from debug_agent.traceback_parser import parse_failing_traceback, parse_text


def _turn(name: str, output: str, i_marker: str = "") -> dict:
    return {"kind": "action", "action_name": name, "action_payload": "",
            "tool_output": output, "llm_response": ""}


PYTEST_LONG = """
============================= test session starts ==============================
collected 1 item

tests/test_foo.py F                                                       [100%]

=================================== FAILURES ===================================
__________________________________ test_bar ___________________________________

    def test_bar():
        x = compute(3)
>       assert x == 5
E       assert 6 == 5

tests/test_foo.py:5: AssertionError
=========================== short test summary info ============================
FAILED tests/test_foo.py::test_bar - assert 6 == 5
"""


def test_parse_long_pytest_picks_symptom_and_refs():
    traj = [
        _turn("pytest", "ERROR: import failure"),        # must be skipped
        _turn("read", "def compute(x): return x*2"),
        _turn("pytest", PYTEST_LONG),                    # target
        _turn("pdb", "(Pdb) p x\n6"),                    # later turn, irrelevant
    ]
    out = parse_failing_traceback(traj)
    assert out["status"] == "ok"
    assert out["frames"], "expected at least one frame"
    assert out["frames"][-1].file.endswith("test_foo.py")
    assert out["frames"][-1].lineno == 5
    assert out["symptom"] is not None
    assert out["symptom"]["lineno"] == 5
    assert out["exc_type"] == "AssertionError"
    assert "6 == 5" in out["exc_msg"]
    assert out["evidence_refs"] == ["trajectory[2].tool_output"]


def test_no_pytest_turn_returns_unavailable():
    traj = [_turn("read", "hello"), _turn("bash", "ls")]
    out = parse_failing_traceback(traj)
    assert out["status"] == "unavailable"
    assert "no-failing-traceback" in out["reason"]
    assert out["frames"] == []


def test_all_frames_are_test_code_symptom_flagged():
    only_tests = """
tests/test_only.py:10: in test_a
    assert False
E   assert False
tests/test_only.py:10: AssertionError
"""
    traj = [_turn("pytest", only_tests)]
    out = parse_failing_traceback(traj)
    assert out["status"] == "ok"
    assert out["symptom"] is not None
    assert out["symptom"]["is_test_code"] is True
    assert all(f.is_test_code for f in out["frames"])


def test_chained_exception_picks_last_traceback():
    chained = """
Traceback (most recent call last):
  File "pkg/a.py", line 3, in foo
    raise ValueError("inner")
ValueError: inner

During handling of the above exception, another exception occurred:

  File "pkg/b.py", line 7, in bar
    raise RuntimeError("outer")
RuntimeError: outer
pkg/b.py:7: RuntimeError
"""
    out = parse_failing_traceback([_turn("pytest", chained)])
    assert out["exc_type"] == "RuntimeError"
    assert out["symptom"]["file"] == "pkg/b.py"
    assert out["symptom"]["lineno"] == 7


PYTEST_WITH_DIFF_HINTS = """
=================================== FAILURES ===================================
_ test_thing _
_repro/test_a0.py:85: in test_thing
    assert doc["k"] == "OL2M"
E   AssertionError: assert 'OL1M' == 'OL2M'
E     - OL2M
E     ?   ^
E     + OL1M
E     ?   ^
=========================== short test summary info ============================
FAILED _repro/test_a0.py::test_thing
"""


def test_long_form_frame_header_is_not_mistaken_for_exc_type():
    """`path:ln: in <funcname>` is the long-form frame header, NOT a short
    summary line. The parser must not set exc_type='in'."""
    out = parse_text(PYTEST_WITH_DIFF_HINTS, "repro_test_run")
    assert out["exc_type"] == "AssertionError"
    assert out["exc_type"] != "in"


def test_exc_msg_ignores_diff_hint_lines():
    """Pytest expands `assert a == b` into multi-line E lines with diff hints
    starting with -/+/?. Those must not pollute exc_msg."""
    out = parse_text(PYTEST_WITH_DIFF_HINTS, "repro_test_run")
    assert out["exc_msg"].startswith("assert 'OL1M'")
    assert "^" not in out["exc_msg"]
    assert out["evidence_refs"] == ["repro_test_run"]
