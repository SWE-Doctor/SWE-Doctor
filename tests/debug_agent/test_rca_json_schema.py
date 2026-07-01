import json
from pathlib import Path

from debug_agent.run_debug import run_one_instance


class _FakeReport:
    def __init__(self):
        self.root_cause_files = ["pkg/mod.py"]
        self.root_cause_functions = ["pkg.mod.C.add"]
        self.reasoning = "because"
        self.suggested_fix = ""
        self.transcript = []
        self.trajectory = []
        self.timed_out = False

    def to_json(self):
        return {
            "root_cause_files": self.root_cause_files,
            "root_cause_functions": self.root_cause_functions,
            "reasoning": self.reasoning,
            "suggested_fix": self.suggested_fix,
            "transcript": self.transcript,
            "trajectory": self.trajectory,
            "timed_out": self.timed_out,
        }


class _FakeAnalyzer:
    def __init__(self, *a, **kw): pass
    def run(self): return _FakeReport()


class _FakeContainer:
    container_id = "x"
    def exec_bash(self, *a, **kw):
        class R: returncode=0; stdout=""; stderr=""
        return R()
    def read_file(self, p): return ""
    def terminate(self): pass


def _run(tmp_path, monkeypatch, enrich_on):
    monkeypatch.setenv("DEBUG_AGENT_ENRICH", "1" if enrich_on else "")
    instance_dir = tmp_path / "instance_x"
    instance_dir.mkdir()
    (instance_dir / "problem_statement.txt").write_text("issue")
    out_dir = tmp_path / "rca_out"

    import debug_agent.run_debug as rd
    monkeypatch.setattr(rd, "DebugAnalyzer", _FakeAnalyzer)
    # Bypass the preflight module since we have no real container.
    class _Pre:
        env_prefix = ""
        def as_transcript_seed(self): return ""
    import debug_agent.preflight as pf
    monkeypatch.setattr(pf, "bootstrap", lambda *a, **kw: _Pre())
    path = run_one_instance(
        instance_dir=instance_dir, output_dir=out_dir,
        image=None, model="fake", attach_container=None,
        llm_factory=lambda m: (lambda p: ""),
        container_factory=lambda: _FakeContainer(),
    )
    return json.loads(path.read_text())


def test_legacy_keys_unchanged_when_enrich_off(tmp_path, monkeypatch):
    payload = _run(tmp_path, monkeypatch, enrich_on=False)
    assert set(payload.keys()) == {"source", "candidates", "debug_report", "meta"}
    assert payload["source"] == "debug_agent"
    assert "rich_rca" not in payload


def test_rich_rca_appended_when_enrich_on(tmp_path, monkeypatch):
    payload = _run(tmp_path, monkeypatch, enrich_on=True)
    assert set(payload.keys()) >= {"source", "candidates", "debug_report", "meta", "rich_rca"}
    assert payload["rich_rca"]["schema_version"] == 2
    assert payload["source"] == "debug_agent"
    assert isinstance(payload["candidates"], list)
    assert "root_cause_files" in payload["debug_report"]


def test_enricher_exception_does_not_break_write(tmp_path, monkeypatch):
    monkeypatch.setenv("DEBUG_AGENT_ENRICH", "1")
    import debug_agent.run_debug as rd

    def _boom(*a, **kw): raise RuntimeError("enrich-boom")
    monkeypatch.setattr(rd, "_run_enricher", _boom, raising=False)

    payload = _run(tmp_path, monkeypatch, enrich_on=True)
    assert payload["rich_rca"]["status"] == "enricher-error"
    assert "enrich-boom" in payload["rich_rca"]["error"]
