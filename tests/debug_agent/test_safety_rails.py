"""Verify that run_one_instance always terminates the container,
even when the analyzer raises, and that budget/timeout safety rails
short-circuit long runs cleanly."""
from __future__ import annotations

from pathlib import Path

import pytest

from debug_agent import run_debug
from debug_agent.analyzer import DebugAnalyzer


class _RecordingContainer:
    def __init__(self):
        self.container_id = "rec"
        self.workdir = "/work"
        self.terminated = False

    def exec_bash(self, cmd, timeout=120):
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
            combined = ""
        return R()

    def read_file(self, path): return ""
    def write_file(self, path, content): pass
    def snapshot(self, path): pass
    def restore(self, path): pass
    def restore_all(self): pass

    def terminate(self):
        self.terminated = True


def test_run_one_instance_terminates_container_on_analyzer_exception(tmp_path, monkeypatch):
    instance_dir = tmp_path / "instance_x"
    instance_dir.mkdir()
    (instance_dir / "problem_statement.txt").write_text("issue")

    rec = _RecordingContainer()
    monkeypatch.setattr(run_debug, "_copy_into_container", lambda c, s, d: None)

    def _boom_llm_factory(_model):
        def _boom(_prompt):
            raise RuntimeError("llm explosion")
        return _boom

    with pytest.raises(RuntimeError, match="llm explosion"):
        run_debug.run_one_instance(
            instance_dir=instance_dir,
            output_dir=tmp_path / "out",
            image="ignored",
            model="ignored",
            container_factory=lambda: rec,
            llm_factory=_boom_llm_factory,
            max_rounds=5,
            wall_timeout=30,
        )
    assert rec.terminated is True, "Container.terminate() must run even when analyzer raises"


def test_max_history_chars_terminates_analyzer():
    """Mirror of the analyzer-level test but phrased as a safety-rail assertion:
    a runaway LLM spewing huge action payloads must halt before max_rounds."""
    huge = "x" * 20_000
    ana = DebugAnalyzer(
        llm=lambda _p: f'<action name="bash">{huge}</action>',
        dispatch=lambda a, c, ctx: f"OK({a.name})",
        container=None,
        ctx={"issue": "x", "repro_nodeid": "n"},
        max_rounds=1000,
        max_history_chars=30_000,
    )
    report = ana.run()
    assert report.timed_out is True
    assert len(report.transcript) < 1000
