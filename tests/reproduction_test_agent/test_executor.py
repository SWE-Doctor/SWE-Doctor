"""Tests for reproduction_test_agent.executor helpers."""
from reproduction_test_agent.executor import _ensure_pytest


class FakeEnv:
    """Minimal Environment stub: records commands, fakes the pytest import probe."""

    def __init__(self, pytest_present: bool):
        self._pytest_present = pytest_present
        self.commands: list[str] = []

    def execute(self, action, cwd="", *, timeout=None):
        cmd = action["command"]
        self.commands.append(cmd)
        if "import pytest" in cmd:
            return {"output": "", "returncode": 0 if self._pytest_present else 1}
        return {"output": "", "returncode": 0}


def test_ensure_pytest_installs_when_missing():
    """django/sympy Verified images lack pytest; the BRT can't run without it,
    so _ensure_pytest must install pytest when the import probe fails."""
    env = FakeEnv(pytest_present=False)
    _ensure_pytest(env, cwd="/testbed")
    assert any("pip install" in c and "pytest" in c for c in env.commands)


def test_ensure_pytest_noop_when_present():
    """When pytest is already importable, do not reinstall it."""
    env = FakeEnv(pytest_present=True)
    _ensure_pytest(env, cwd="/testbed")
    assert not any("pip install" in c for c in env.commands)
