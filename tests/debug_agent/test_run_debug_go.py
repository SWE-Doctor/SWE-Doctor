"""Go run_debug wiring (no docker): --language go is parsed, _find_accepted_repro
prefers *_test.go, and run_one_instance(language='go') injects a GoDlvSession
factory + language='go' into ctx while the LLM still drives pdb_start/pdb_cmd.
Python behavior must be untouched."""
from __future__ import annotations

import json

from debug_agent import run_debug


# ---- _find_accepted_repro -------------------------------------------------

def test_find_accepted_repro_go_prefers_test_go(tmp_path):
    d = tmp_path / "instance_x" / "workspace" / "_repro_tests"
    d.mkdir(parents=True)
    (d / "helper.go").write_text("package x\n")
    (d / "repro_test.go").write_text("package x\nimport \"testing\"\nfunc TestRepro(t *testing.T){}\n")
    found = run_debug._find_accepted_repro(tmp_path / "instance_x", language="go")
    assert found is not None and found.name == "repro_test.go"


def test_find_accepted_repro_python_default_unchanged(tmp_path):
    d = tmp_path / "instance_y" / "stage1_reproduction" / "accepted"
    d.mkdir(parents=True)
    (d / "test_a0.py").write_text("def test(): assert False\n")
    found = run_debug._find_accepted_repro(tmp_path / "instance_y")
    assert found is not None and found.suffix == ".py"


# ---- parse_args -----------------------------------------------------------

def test_parse_args_language_default_python():
    ns = run_debug.parse_args(["--instance-dir", "/x"])
    assert ns.language == "python"


def test_parse_args_language_go():
    ns = run_debug.parse_args(["--instance-dir", "/x", "--language", "go"])
    assert ns.language == "go"


# ---- run_one_instance(language='go') --------------------------------------

class _StubContainer:
    container_id = "stub"
    workdir = "/app"

    def __init__(self):
        self.dlv_ensured = False

    def ensure_dlv(self):
        self.dlv_ensured = True

    def exec_bash(self, cmd, timeout=120, stdin_data=None):
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

    def restore_all(self):
        pass

    def terminate(self):
        pass


def test_run_one_instance_go_injects_dlv_factory_and_language(tmp_path, monkeypatch):
    instance_dir = tmp_path / "instance_demo"
    instance_dir.mkdir()
    (instance_dir / "problem_statement.txt").write_text("nil deref when store missing")
    repro_dir = instance_dir / "stage1_reproduction" / "accepted"
    repro_dir.mkdir(parents=True)
    (repro_dir / "repro_test.go").write_text(
        "package server\nimport \"testing\"\nfunc TestRepro(t *testing.T){ t.Fatal(\"boom\") }\n")
    out_dir = tmp_path / "rca_out"

    monkeypatch.setattr(run_debug, "_copy_into_container", lambda c, s, d: None)

    captured = {}

    # Capture the ctx the analyzer was built with, and avoid running the real loop.
    import debug_agent.run_debug as RD

    class _FakeAnalyzer:
        def __init__(self, **kw):
            captured["ctx"] = kw["ctx"]
        def run(self):
            from debug_agent.analyzer import DebugReport
            return DebugReport(root_cause_files=["server/evaluator.go"],
                               root_cause_functions=["server.(*Server).Evaluate"],
                               reasoning="nil store", suggested_fix="init store")

    monkeypatch.setattr(RD, "DebugAnalyzer", _FakeAnalyzer)

    stub = _StubContainer()
    out_path = run_debug.run_one_instance(
        instance_dir=instance_dir,
        output_dir=out_dir,
        image="golang:1.24",
        model="stub/model",
        container_factory=lambda: stub,
        llm_factory=lambda _m: (lambda _p: ""),
        max_rounds=5,
        wall_timeout=30,
        language="go",
    )
    assert stub.dlv_ensured is True
    ctx = captured["ctx"]
    assert ctx["language"] == "go"
    assert callable(ctx["_pdb_session_factory"])
    # The factory must build a GoDlvSession bound to our container.
    from debug_agent.go_session import GoDlvSession
    sess = ctx["_pdb_session_factory"](["./server", "-run", "TestRepro"])
    assert isinstance(sess, GoDlvSession)
    assert sess.run_args == ["./server", "-run", "TestRepro"]

    data = json.loads(out_path.read_text())
    assert data["source"] == "debug_agent"
    assert [c["file"] for c in data["candidates"]] == ["server/evaluator.go"]


def test_run_go_instance_places_repro_in_package_dir_via_relpath(tmp_path, monkeypatch):
    # Go tests must land in their package dir (not _repro/) so they compile and
    # reach same-package helpers. The .relpath sidecar carries that path.
    instance_dir = tmp_path / "instance_pkg"
    instance_dir.mkdir()
    (instance_dir / "problem_statement.txt").write_text("batch eval bug")
    repro_dir = instance_dir / "stage1_reproduction" / "accepted"
    repro_dir.mkdir(parents=True)
    (repro_dir / "zzz_repro_test.go").write_text(
        "package server\nimport \"testing\"\nfunc TestBatchRepro(t *testing.T){ t.Fatal(\"x\") }\n")
    (repro_dir / "zzz_repro_test.relpath").write_text("server/zzz_repro_test.go")
    out_dir = tmp_path / "rca_out"

    copied = {}
    monkeypatch.setattr(run_debug, "_copy_into_container",
                        lambda c, s, d: copied.update(dest=d))
    captured = {}
    import debug_agent.run_debug as RD

    class _FakeAnalyzer:
        def __init__(self, **kw):
            captured["ctx"] = kw["ctx"]
        def run(self):
            from debug_agent.analyzer import DebugReport
            return DebugReport(root_cause_files=[], root_cause_functions=[],
                               reasoning="", suggested_fix="")

    monkeypatch.setattr(RD, "DebugAnalyzer", _FakeAnalyzer)
    run_debug.run_one_instance(
        instance_dir=instance_dir, output_dir=out_dir, image="golang:1.24",
        model="stub/model", container_factory=lambda: _StubContainer(),
        llm_factory=lambda _m: (lambda _p: ""), max_rounds=5, wall_timeout=30,
        language="go")

    assert copied["dest"] == "/app/server/zzz_repro_test.go"   # package dir, NOT _repro/
    ctx = captured["ctx"]
    assert ctx["repro_nodeid"] == "server/zzz_repro_test.go"
    assert ctx["go_pkg"] == "./server"
    assert ctx["go_test_name"] == "TestBatchRepro"
