"""Language registry for Stage-1 reproduction generation (python + go).

Go support mirrors the JS adaptation's langpack approach: a single dataclass
row drives ~90% of per-language Stage-1 differences. Default 'python'
preserves Verified behavior exactly.
"""
import pytest

from reproduction_test_agent.langpack import get_langpack, parse_test_output


def test_go_langpack_uses_go_test_file_and_fence():
    lp = get_langpack("go")
    assert lp.name == "go"
    assert lp.test_filename == "repro_test.go"  # go tests must end in _test.go
    assert lp.code_fence == "go"


def test_go_langpack_globs_and_test_substrings():
    lp = get_langpack("go")
    assert lp.source_globs == ["*.go"]
    assert "_test.go" in lp.test_path_substrings
    assert lp.grep_type_args == "--type go"


def test_go_func_def_grep_matches_funcs_and_methods():
    lp = get_langpack("go")
    pat = lp.func_def_grep.format(name="Evaluate")
    import re
    assert re.search(pat, "func Evaluate(ctx context.Context) error {")
    assert re.search(pat, "func (s *Server) Evaluate(r *Req) (*Resp, error) {")


def test_python_langpack_preserves_verified_defaults():
    lp = get_langpack("python")
    assert lp.test_filename == "repro_test.py"
    assert lp.code_fence == "python"
    assert lp.source_globs == ["*.py"]


def test_get_langpack_unknown_language_raises():
    with pytest.raises(KeyError):
        get_langpack("rust")


def test_parse_test_output_go_extracts_failed_test_name():
    out = (
        "=== RUN   TestBatchEvaluate\n"
        "--- FAIL: TestBatchEvaluate (0.00s)\n"
        "    evaluator_test.go:123: unexpected error\n"
        "FAIL\n"
        "FAIL\tgithub.com/foo/bar/server\t0.012s\n"
    )
    etype, msg = parse_test_output("go", out)
    assert etype != ""
    assert "TestBatchEvaluate" in (etype + " " + msg)
