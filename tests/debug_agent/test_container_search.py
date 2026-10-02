"""Stub-Container unit tests for debug_agent.container_search."""
from __future__ import annotations

from dataclasses import dataclass

from debug_agent.container_search import (
    container_rg, container_walk_py, container_read_text,
)


@dataclass
class _R:
    returncode: int
    stdout: str
    stderr: str = ""


class _StubContainer:
    """Captures the last cmd; returns canned stdout per substring trigger."""
    def __init__(self, responses: dict[str, _R]):
        self.responses = responses
        self.calls: list[str] = []

    def exec_bash(self, cmd: str, timeout: int = 120) -> _R:
        self.calls.append(cmd)
        for needle, r in self.responses.items():
            if needle in cmd:
                return r
        return _R(0, "")


def test_container_rg_parses_path_lineno_via_grep_fallback():
    """SWE-bench Pro images don't have rg, so the grep fallback is the
    primary path."""
    stub = _StubContainer({
        "command -v rg": _R(1, ""),  # rg not present
        "grep -rnw": _R(0, "/app/lib/x.py:42:    target()\n/app/lib/y.py:10:    target(x)\n"),
    })
    hits, truncated = container_rg(stub, "target", root="/app", time_budget_s=5.0)
    assert truncated is False
    assert hits == [("lib/x.py", 42), ("lib/y.py", 10)]


def test_container_rg_via_rg_when_available():
    stub = _StubContainer({
        "command -v rg": _R(0, ""),  # rg present
        "rg -n": _R(0, "/app/lib/x.py:42:    target()\n"),
    })
    hits, truncated = container_rg(stub, "target", root="/app", time_budget_s=5.0)
    assert truncated is False
    assert hits == [("lib/x.py", 42)]


def test_container_rg_timeout_marker():
    """When the search tool returns rc not in {0,1}, mark truncated=True."""
    stub = _StubContainer({
        "command -v rg": _R(1, ""),
        "grep -rnw": _R(124, "", "killed"),
    })
    _, truncated = container_rg(stub, "target", root="/app", time_budget_s=5.0)
    assert truncated is True


def test_container_walk_py_filters_tests():
    stub = _StubContainer({
        "find ": _R(0,
                    "/app/lib/x.py\n"
                    "/app/tests/test_x.py\n"
                    "/app/lib/test_y.py\n"
                    "/app/lib/z.py\n"),
    })
    out = list(container_walk_py(stub, root="/app"))
    assert out == ["lib/x.py", "lib/z.py"]


def test_container_read_text_returns_stdout():
    stub = _StubContainer({"cat ": _R(0, "hello")})
    assert container_read_text(stub, "/app/x.py") == "hello"
