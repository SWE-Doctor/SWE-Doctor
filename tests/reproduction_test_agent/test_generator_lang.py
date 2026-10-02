"""generator must select the system prompt + code fence by `language`.
Default 'python' stays byte-identical to the original Verified prompt
(including the CRITICAL code-path section); 'go' uses the Go langpack."""
from reproduction_test_agent import generator


def test_generate_test_uses_go_system_prompt(monkeypatch):
    captured = {}

    def fake_llm_call(messages, model, temperature=0.0, max_tokens=4096, caller=""):
        captured["system"] = messages[0]["content"]
        return "```go\npackage server\n\nfunc TestX(t *testing.T) {}\n```"

    monkeypatch.setattr(generator, "llm_call", fake_llm_call)
    code = generator.generate_test("issue", "none", {}, "m", language="go")
    assert "Go test engineer" in captured["system"]
    assert code.startswith("package server")


def test_generate_test_default_is_python(monkeypatch):
    captured = {}

    def fake_llm_call(messages, model, temperature=0.0, max_tokens=4096, caller=""):
        captured["system"] = messages[0]["content"]
        return "```python\nimport x\n```"

    monkeypatch.setattr(generator, "llm_call", fake_llm_call)
    generator.generate_test("issue", "none", {}, "m")
    assert "Python test engineer" in captured["system"]
    # the original BASE_SYSTEM_PROMPT carried this exact guidance — preserve it
    assert "CRITICAL — choosing the right code path" in captured["system"]


def test_generate_test_for_aspect_uses_go_system_prompt(monkeypatch):
    captured = {}

    def fake_llm_call(messages, model, temperature=0.0, max_tokens=4096, caller=""):
        captured["system"] = messages[0]["content"]
        return "```go\npackage server\nfunc TestY(t *testing.T) {}\n```"

    monkeypatch.setattr(generator, "llm_call", fake_llm_call)
    generator.generate_test_for_aspect("issue", "none", {}, "asp", "m", language="go")
    assert "Go test engineer" in captured["system"]


def test_build_context_uses_go_code_fence():
    loc = {"relevant_files": ["server/eval.go"],
           "file_contents": {"server/eval.go": "package server\n"}}
    prompt = generator.build_prompt("issue", "patchLoc", loc, language="go")
    assert "```go" in prompt
    assert "```python" not in prompt


def test_build_context_default_python_fence():
    loc = {"relevant_files": ["a.py"], "file_contents": {"a.py": "x = 1\n"}}
    prompt = generator.build_prompt("issue", "patchLoc", loc)
    assert "```python" in prompt
