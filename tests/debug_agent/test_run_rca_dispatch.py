"""run_rca dispatcher: backend selection + auto-fallback.

No docker / no real RCA runs — we patch the subprocess-executing helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_RCA_PATH = Path(__file__).resolve().parents[2] / "run_pro_test" / "run_rca.py"
_spec = importlib.util.spec_from_file_location("run_pro_test_run_rca", _RCA_PATH)
run_rca = importlib.util.module_from_spec(_spec)
sys.modules["run_pro_test_run_rca"] = run_rca
_spec.loader.exec_module(run_rca)


def _mk_instance(root: Path, name: str, with_repro: bool) -> Path:
    d = root / name
    (d / "stage1_reproduction" / "accepted").mkdir(parents=True) if with_repro else d.mkdir()
    if with_repro:
        (d / "stage1_reproduction" / "accepted" / "test_a0.py").write_text("def test_x(): pass\n")
    return d


def test_dispatcher_routes_to_debug_agent_when_repro_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("RCA_BACKEND", "debug_agent")
    inst = _mk_instance(tmp_path, "instance_with_repro", with_repro=True)

    called = {"dbg": [], "static": []}
    monkeypatch.setattr(run_rca, "_run_debug_agent",
                        lambda dirs, args: (called["dbg"].append([d.name for d in dirs]) or 0))
    monkeypatch.setattr(run_rca, "_run_static",
                        lambda tail: (called["static"].append(tail) or 0))

    rc = run_rca.main(["--instance-dir", str(inst)])
    assert rc == 0
    assert called["dbg"] == [["instance_with_repro"]]
    assert called["static"] == []


def test_dispatcher_falls_back_to_static_when_no_repro(tmp_path, monkeypatch):
    monkeypatch.setenv("RCA_BACKEND", "debug_agent")
    inst = _mk_instance(tmp_path, "instance_no_repro", with_repro=False)

    called = {"dbg": [], "static": []}
    monkeypatch.setattr(run_rca, "_run_debug_agent",
                        lambda dirs, args: (called["dbg"].append([d.name for d in dirs]) or 0))
    monkeypatch.setattr(run_rca, "_run_static",
                        lambda tail: (called["static"].append(tail) or 0))

    rc = run_rca.main(["--instance-dir", str(inst)])
    assert rc == 0
    assert called["dbg"] == []
    assert called["static"] and called["static"][0][:2] == ["--instance-dir", str(inst)]


def test_dispatcher_static_backend_forces_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("RCA_BACKEND", "static")
    inst = _mk_instance(tmp_path, "instance_with_repro", with_repro=True)

    called = {"dbg": [], "static": []}
    monkeypatch.setattr(run_rca, "_run_debug_agent",
                        lambda dirs, args: (called["dbg"].append([d.name for d in dirs]) or 0))
    monkeypatch.setattr(run_rca, "_run_static",
                        lambda tail: (called["static"].append(tail) or 0))

    rc = run_rca.main(["--instance-dir", str(inst)])
    assert rc == 0
    assert called["dbg"] == []
    assert called["static"]


def test_dispatcher_splits_batch_by_repro_presence(tmp_path, monkeypatch):
    monkeypatch.setenv("RCA_BACKEND", "debug_agent")
    results = tmp_path / "results"
    results.mkdir()
    _mk_instance(results, "instance_a", with_repro=True)
    _mk_instance(results, "instance_b", with_repro=False)
    _mk_instance(results, "instance_c", with_repro=True)

    called = {"dbg": [], "static": []}
    monkeypatch.setattr(run_rca, "_run_debug_agent",
                        lambda dirs, args: (called["dbg"].append(sorted(d.name for d in dirs)) or 0))
    monkeypatch.setattr(run_rca, "_run_static",
                        lambda tail: (called["static"].append(tail) or 0))

    rc = run_rca.main(["--results-dir", str(results)])
    assert rc == 0
    assert called["dbg"] == [["instance_a", "instance_c"]]
    assert len(called["static"]) == 1
    assert "instance_b" in called["static"][0][1]
