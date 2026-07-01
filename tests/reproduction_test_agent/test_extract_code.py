"""extract_code must be language-agnostic: strip the fence-language tag line
for go (and others) the same way it already does for python."""
from reproduction_test_agent.utils import extract_code


def test_extract_code_strips_go_fence_tag():
    resp = "Here is the repro:\n```go\npackage server\n\nfunc TestX(t *testing.T) {}\n```\n"
    code = extract_code(resp)
    assert code.startswith("package server")
    assert "func TestX" in code
    assert "```" not in code
    assert not code.lstrip().startswith("go")


def test_extract_code_preserves_python_fence_behavior():
    resp = "```python\ndef test_x():\n    assert True\n```"
    code = extract_code(resp)
    assert code == "def test_x():\n    assert True"


def test_extract_code_bare_go_fence_label_line():
    resp = "```go\npackage foo\nfunc TestZ(t *testing.T) {}\n```"
    code = extract_code(resp)
    assert code.startswith("package foo")
