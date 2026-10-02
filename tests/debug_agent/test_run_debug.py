"""Happy-path end-to-end test for debug_agent.run_debug.run_one_instance
with a stub Container and a scripted LLM. No docker required."""
from __future__ import annotations

import json
from pathlib import Path

from debug_agent import run_debug


class _StubContainer:
    container_id = "stub"
    workdir = "/work"

    def exec_bash(self, cmd, timeout=120):
        class R:
            returncode = 0
            stdout = "ok"
            stderr = ""
            combined = "ok"
        return R()

    def read_file(self, path):
        return ""

    def write_file(self, path, content):
        pass

    def snapshot(self, path):
        pass

    def restore(self, path):
        pass

    def restore_all(self):
        pass

    def terminate(self):
        pass


def test_run_one_instance_writes_unified_rca_json(tmp_path, monkeypatch):
    instance_dir = tmp_path / "instance_demo"
    instance_dir.mkdir()
    (instance_dir / "problem_statement.txt").write_text("thing breaks when x is None")
    # Simulate a Phase-A accepted repro
    repro_dir = instance_dir / "stage1_reproduction" / "accepted"
    repro_dir.mkdir(parents=True)
    (repro_dir / "test_a0.py").write_text("def test_case():\n    assert False\n")

    out_dir = tmp_path / "rca_out"

    # Skip docker cp by monkeypatching the helper.
    monkeypatch.setattr(run_debug, "_copy_into_container", lambda c, s, d: None)

    llm_responses = iter([
        '<reason>inspect</reason><action name="bash">ls /work</action>',
        '<reason>repro</reason><action name="pytest">-x</action>',
        ('<conclusion>'
         '<root_cause_files>pkg/mod.py, pkg/util.py</root_cause_files>'
         '<root_cause_functions>pkg.mod.foo, pkg.util.bar</root_cause_functions>'
         '<reasoning>foo() returns None when it should raise.</reasoning>'
         '<suggested_fix>raise ValueError in pkg.mod.foo</suggested_fix>'
         '</conclusion>'),
    ])
    llm_factory = lambda _model: (lambda _p: next(llm_responses))

    # The new conclusion gate requires a real pdb stop + 3 cmds. Bypass it
    # for this happy-path test by monkeypatching can_conclude.
    import debug_agent.analyzer as _ana
    monkeypatch.setattr(_ana, "can_conclude", lambda *_a, **_k: True)

    out_path = run_debug.run_one_instance(
        instance_dir=instance_dir,
        output_dir=out_dir,
        image="python:3.11-slim",  # ignored thanks to container_factory
        model="stub/model",
        container_factory=lambda: _StubContainer(),
        llm_factory=llm_factory,
        max_rounds=5,
        wall_timeout=30,
    )
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["source"] == "debug_agent"
    assert data["meta"]["container_mode"] == "launch"
    assert data["debug_report"]["reasoning"].startswith("foo()")
    assert [c["file"] for c in data["candidates"]] == ["pkg/mod.py", "pkg/util.py"]
    assert data["candidates"][0]["score"] > data["candidates"][1]["score"]


def test_run_debug_writes_rich_branch_observations_and_anomalies(tmp_path, monkeypatch):
    """Patch enrich() to return synthetic rich keys; verify they hit _rca.json."""
    from debug_agent import run_debug
    import json as _json

    fake_rca = {
        "schema_version": 2,
        "root_cause": {"file": "m.py", "qualname": "f", "lineno": 0,
                       "provenance": {"source": "llm", "evidence_refs": []}},
        "symptom": {"status": "unavailable", "reason": "test",
                    "provenance": {"source": "x", "evidence_refs": []}},
        "propagation_path": {"frames": [],
                             "provenance": {"source": "x", "evidence_refs": []}},
        "contract_impact": {"changed": False, "kind": "none", "summary": "",
                            "callers": [], "callers_truncated": False,
                            "provenance": {"source": "x", "evidence_refs": []}},
        "related_non_code": {"hits": [], "note": "skipped",
                             "provenance": {"source": "x", "evidence_refs": []}},
        "rich_branch_observations": [{"file": "m.py", "lineno": 5,
                                      "cond_text": "if x", "arm_taken": "then",
                                      "locals_at_stop": {}, "evidence_refs": []}],
        "rich_pdb_anomalies": [{"expr": "x", "file": "m.py", "lineno": 5,
                                "value_repr": "None", "tags": ["none"],
                                "evidence_refs": []}],
    }

    monkeypatch.setenv("DEBUG_AGENT_ENRICH", "1")
    monkeypatch.setattr(run_debug, "_run_enricher", lambda *a, **k: fake_rca)

    instance_dir = tmp_path / "instance_demo"
    instance_dir.mkdir()
    (instance_dir / "problem_statement.txt").write_text("x")
    repro_dir = instance_dir / "stage1_reproduction" / "accepted"
    repro_dir.mkdir(parents=True)
    (repro_dir / "test_a0.py").write_text("def test(): assert False\n")
    out_dir = tmp_path / "rca_out"

    monkeypatch.setattr(run_debug, "_copy_into_container", lambda c, s, d: None)
    llm_responses = iter([
        '<reason>repro</reason><action name="pytest">-x</action>',
        ('<conclusion><root_cause_files>m.py</root_cause_files>'
         '<root_cause_functions>m.f</root_cause_functions>'
         '<reasoning>r</reasoning><suggested_fix>s</suggested_fix></conclusion>'),
        '<reason>retry</reason><action name="bash">noop</action>',
    ])

    out_path = run_debug.run_one_instance(
        instance_dir=instance_dir,
        output_dir=out_dir,
        image="python:3.11-slim",
        model="stub",
        container_factory=lambda: _StubContainer(),
        llm_factory=lambda _m: (lambda _p: next(llm_responses, '<reason>x</reason><action name="bash">noop</action>')),
        max_rounds=4,
        wall_timeout=30,
    )
    data = _json.loads(out_path.read_text())
    rca = data["rich_rca"]
    assert rca["rich_branch_observations"][0]["arm_taken"] == "then"
    assert "none" in rca["rich_pdb_anomalies"][0]["tags"]
