"""critic prompts must fence the test code with the langpack code_fence.
language='go' => ```go; default python => ```python (byte-identical)."""
from reproduction_test_agent import critic


def test_critique_test_go_uses_go_fence(monkeypatch):
    captured = {}

    def fake(**k):
        captured["prompt"] = k["messages"][0]["content"]
        return "FAILS_FOR_RIGHT_REASON: yes\nFAILURE_CATEGORY: assertion_failure"

    monkeypatch.setattr(critic, "llm_call", fake)
    critic.critique_test("issue", "func TestX(t *testing.T) {}", "log", "m", language="go")
    assert "```go" in captured["prompt"]
    assert "```python" not in captured["prompt"]


def test_critique_test_default_python_fence(monkeypatch):
    captured = {}

    def fake(**k):
        captured["prompt"] = k["messages"][0]["content"]
        return "FAILS_FOR_RIGHT_REASON: yes"

    monkeypatch.setattr(critic, "llm_call", fake)
    critic.critique_test("issue", "def test_x(): pass", "log", "m")
    assert "```python" in captured["prompt"]


def test_critique_passing_test_go_uses_go_fence(monkeypatch):
    captured = {}

    def fake(**k):
        captured["prompt"] = k["messages"][0]["content"]
        return "FAILS_FOR_RIGHT_REASON: no"

    monkeypatch.setattr(critic, "llm_call", fake)
    critic.critique_passing_test("issue", "func TestX(t *testing.T) {}", "log", "m", language="go")
    assert "```go" in captured["prompt"]
