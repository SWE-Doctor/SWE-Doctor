"""localizer must drive repo structure + keyword grep off the langpack:
language='go' uses *.go / --type go and excludes vendor dirs; python keeps
its original *.py / --type py and test/site-packages exclusions verbatim."""
from reproduction_test_agent import localizer


class RecEnv:
    def __init__(self):
        self.commands = []

    def execute(self, action, cwd="", *, timeout=None):
        self.commands.append(action["command"])
        return {"output": "", "returncode": 0}


def test_localize_go_uses_go_globs_and_vendor_exclusion(monkeypatch):
    monkeypatch.setattr(localizer, "llm_call", lambda **k: "RELEVANT_FILES:\nserver/eval.go\n")
    env = RecEnv()
    localizer.localize("issue mentions someEvaluator", env, "/app", "m", language="go")
    joined = "\n".join(env.commands)
    assert "--type go" in joined
    assert "*.go" in joined
    assert "vendor" in joined          # go excludes vendor dirs from grep
    assert "--type py" not in joined


def test_localize_python_unchanged(monkeypatch):
    monkeypatch.setattr(localizer, "llm_call", lambda **k: "RELEVANT_FILES:\nsrc/a.py\n")
    env = RecEnv()
    localizer.localize("issue mentions some_helper", env, "/app", "m")
    joined = "\n".join(env.commands)
    assert "--type py" in joined
    assert "*.py" in joined
    assert "--type go" not in joined
    assert "vendor" not in joined
    assert "site-packages" in joined   # original python grep exclusion preserved
