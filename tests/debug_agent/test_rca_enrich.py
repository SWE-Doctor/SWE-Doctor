from dataclasses import dataclass, field
from debug_agent.rca_enrich import enrich


@dataclass
class _FakeTurn:
    kind: str = "action"
    action_name: str = ""
    action_payload: str = ""
    tool_output: str = ""
    llm_response: str = ""

    def to_json(self):
        return self.__dict__


@dataclass
class _FakeReport:
    root_cause_files: list = field(default_factory=list)
    root_cause_functions: list = field(default_factory=list)
    suggested_fix: str = ""
    reasoning: str = ""
    trajectory: list = field(default_factory=list)
    timed_out: bool = False

    @property
    def trajectory_as_dicts(self):
        return [t.to_json() for t in self.trajectory]


PYTEST_FAIL = """
tests/test_m.py:3: in test_add
    assert C().add([1,1]) == [1]
E   AssertionError: lists differ
pkg/mod.py:3: AssertionError
"""


def _layout(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg/mod.py").write_text(
        "class C:\n"
        "    def add(self, xs):\n"
        "        return xs\n"
    )
    (tmp_path / "tests/test_m.py").write_text(
        "from pkg.mod import C\n"
        "def test_add():\n"
        "    assert C().add([1,1]) == [1]\n"
    )
    (tmp_path / "docs.md").write_text("add: returns xs\n")


def test_happy_path_produces_all_sections(tmp_path):
    _layout(tmp_path)
    report = _FakeReport(
        root_cause_files=["pkg/mod.py"],
        root_cause_functions=["pkg.mod.C.add"],
        suggested_fix="```python\nclass C:\n    def add(self, xs):\n        return sorted(set(xs))\n```",
        trajectory=[_FakeTurn(action_name="pytest", tool_output=PYTEST_FAIL)],
    )
    out = enrich(report, container=None, ctx={"repo_root": tmp_path})
    assert out["schema_version"] == 2
    assert out["root_cause"]["file"] == "pkg/mod.py"
    assert out["root_cause"]["provenance"]["source"] == "llm_conclusion"
    assert out["symptom"]["provenance"]["source"] == "traceback_parse"
    assert out["symptom"]["exc_type"] == "AssertionError"
    assert out["propagation_path"]["provenance"]["source"] == "traceback_parse+ast_slice"
    assert out["propagation_path"]["frames"], "expected at least one labeled frame"
    assert out["contract_impact"]["changed"] is True
    assert out["related_non_code"]["provenance"]["source"] == "rg_nontext"


def test_debug_loop_not_converged_short_circuits(tmp_path):
    report = _FakeReport(timed_out=True, root_cause_files=[], root_cause_functions=[])
    out = enrich(report, container=None, ctx={"repo_root": tmp_path})
    assert out == {"schema_version": 2, "status": "debug-loop-did-not-converge"}


def test_no_pytest_turn_yields_unavailable_symptom(tmp_path):
    _layout(tmp_path)
    report = _FakeReport(
        root_cause_files=["pkg/mod.py"],
        root_cause_functions=["pkg.mod.C.add"],
        suggested_fix="",
        trajectory=[_FakeTurn(action_name="read", tool_output="def add: ...")],
    )
    out = enrich(report, container=None, ctx={"repo_root": tmp_path})
    assert out["symptom"]["status"] == "unavailable"
    assert out["propagation_path"]["frames"] == []
    assert out["contract_impact"]["changed"] is False


def test_container_absolute_paths_are_normalized(tmp_path):
    """Root-cause paths like /app/pkg/mod.py must be stripped of the workdir
    prefix before being concatenated with repo_root — otherwise pathlib drops
    the left operand and the file is never found."""
    _layout(tmp_path)
    report = _FakeReport(
        root_cause_files=["/app/pkg/mod.py"],
        root_cause_functions=["pkg.mod.C.add"],
        suggested_fix="```python\nclass C:\n    def add(self, xs):\n        return sorted(set(xs))\n```",
        trajectory=[_FakeTurn(action_name="pytest", tool_output=PYTEST_FAIL)],
    )
    out = enrich(report, container=None,
                 ctx={"repo_root": tmp_path, "workdir": "/app"})
    # With the workdir stripped, the enricher should find pkg/mod.py and detect the diff.
    assert out["contract_impact"]["changed"] is True


class _FakeContainer:
    """Minimal container: records the last bash cmd and returns scripted output."""
    def __init__(self, stdout: str = "", stderr: str = "", rc: int = 1):
        self._stdout, self._stderr, self._rc = stdout, stderr, rc
        self.last_cmd = None

    def exec_bash(self, cmd, timeout=120):
        self.last_cmd = cmd
        class _R:
            pass
        r = _R()
        r.returncode, r.stdout, r.stderr = self._rc, self._stdout, self._stderr
        return r


def test_symptom_prefers_reproduction_test_over_trajectory(tmp_path):
    """When ctx carries a repro_nodeid and a container is attached, the enricher
    must run the repro test and parse its output — not rely on whatever pytest
    residue happened to surface in the debug trajectory."""
    _layout(tmp_path)
    # Trajectory has NO pytest action — mimics the real smoke where debug_agent
    # only ran `read`/`probe` actions.
    report = _FakeReport(
        root_cause_files=["pkg/mod.py"],
        root_cause_functions=["pkg.mod.C.add"],
        suggested_fix="",
        trajectory=[_FakeTurn(action_name="read", tool_output="irrelevant")],
    )
    c = _FakeContainer(stdout=PYTEST_FAIL, rc=1)
    out = enrich(report, container=c, ctx={
        "repo_root": tmp_path, "workdir": "/app",
        "repro_nodeid": "_repro/test_m.py", "env_prefix": "PYTHONPATH=/app",
    })
    assert out["symptom"]["exc_type"] == "AssertionError"
    assert out["symptom"]["provenance"]["evidence_refs"] == ["repro_test_run"]
    assert "pytest -x _repro/test_m.py" in (c.last_cmd or "")
    assert "PYTHONPATH=/app" in (c.last_cmd or "")


def test_missing_repo_root_skips_caller_and_nontext(tmp_path):
    report = _FakeReport(
        root_cause_files=["pkg/mod.py"],
        root_cause_functions=["pkg.mod.C.add"],
        suggested_fix="```python\ndef add(xs): return sorted(xs)\n```",
        trajectory=[_FakeTurn(action_name="pytest", tool_output=PYTEST_FAIL)],
    )
    out = enrich(report, container=None, ctx={})
    assert out["contract_impact"]["changed"] is False
    assert out["related_non_code"]["hits"] == []


def test_enrich_populates_callers_via_container_without_repo_root():
    """Lever 3: with no DEBUG_AGENT_REPO_ROOT but a live container, callers
    should still be populated by routing through container_search."""
    class _R:
        def __init__(self, rc, stdout): self.returncode = rc; self.stdout = stdout; self.stderr = ""

    class _Container:
        def exec_bash(self, cmd, timeout=120):
            if cmd.startswith("cat "):
                if "lib.py" in cmd:
                    return _R(0, "def target(x):\n    return x\n")
                if "user.py" in cmd:
                    return _R(0, "from lib import target\n\ndef caller():\n    return target(1)\n")
                return _R(1, "")
            if cmd.startswith("rg "):
                # No nontext-grep glob hint here, so handle both flavours by
                # returning a python-ish hit that still routes through the
                # parser; container_rg only emits one row.
                return _R(0, "/app/user.py:4:    return target(1)\n")
            if cmd.startswith("find "):
                return _R(0, "/app/lib.py\n/app/user.py\n")
            return _R(0, "")
        def read_file(self, path):
            return self.exec_bash(f"cat {path}").stdout

    report = _FakeReport(
        root_cause_files=["lib.py"],
        root_cause_functions=["target"],
        suggested_fix="Tweak target's return value.",
        trajectory=[],
    )
    out = enrich(report, container=_Container(),
                 ctx={"workdir": "/app", "_pdb_session_log": []})
    callers = out["contract_impact"]["callers"]
    assert any(c["file"] == "user.py" for c in callers)
    assert out["contract_impact"]["changed"] is False


def test_enrich_includes_branch_observations_and_anomalies(tmp_path, monkeypatch):
    """Synthetic trajectory + a synthetic _pdb_session_log; verify the
    new keys appear in the enriched dict."""
    from debug_agent.analyzer import DebugReport
    from debug_agent import rca_enrich

    src = (
        "def f(x):\n"
        "    if x is None:\n"
        "        return -1\n"
        "    return x\n"
    )
    src_file = tmp_path / "m.py"
    src_file.write_text(src)

    report = DebugReport()
    report.root_cause_files = [str(src_file)]
    report.root_cause_functions = ["m.f"]
    report.timed_out = False

    pdb_log = [
        {"kind": "start", "session_id": 1, "ok": True,
         "initial_frame": {"file": str(src_file), "lineno": 2, "qualname": "f"}},
        {"kind": "cmd", "session_id": 1, "cmd": "p x", "last_eval": "None",
         "current_frame": {"file": str(src_file), "lineno": 2, "qualname": "f"},
         "ended": False},
        {"kind": "cmd", "session_id": 1, "cmd": "n", "last_eval": "",
         "current_frame": {"file": str(src_file), "lineno": 3, "qualname": "f"},
         "ended": False},
        {"kind": "cmd", "session_id": 1, "cmd": "p x", "last_eval": "None",
         "current_frame": {"file": str(src_file), "lineno": 3, "qualname": "f"},
         "ended": False},
    ]
    ctx = {
        "issue": "x", "repro_nodeid": "", "workdir": str(tmp_path),
        "repo_root": tmp_path,
        "_pdb_session_log": pdb_log,
    }

    out = rca_enrich.enrich(report, container=None, ctx=ctx, wall_budget_s=10.0)
    assert "rich_branch_observations" in out
    assert any(b["arm_taken"] == "then" for b in out["rich_branch_observations"])
    assert "rich_pdb_anomalies" in out
    assert any("none" in a["tags"] for a in out["rich_pdb_anomalies"])


def test_enrich_merges_pdb_visited_frames_into_propagation_path(tmp_path):
    from debug_agent.analyzer import DebugReport
    from debug_agent import rca_enrich

    src_file = tmp_path / "m.py"
    src_file.write_text("def f():\n    return 1\n")

    report = DebugReport()
    report.root_cause_files = [str(src_file)]
    report.root_cause_functions = ["m.f"]
    report.transcript = []
    report.trajectory = []
    report.timed_out = False

    pdb_log = [
        {"kind": "start", "session_id": 1, "ok": True,
         "initial_frame": {"file": str(src_file), "lineno": 1, "qualname": "f"}},
        {"kind": "cmd", "session_id": 1, "cmd": "n",
         "current_frame": {"file": str(src_file), "lineno": 2, "qualname": "f"},
         "ended": False},
        {"kind": "cmd", "session_id": 1, "cmd": "n",
         "current_frame": {"file": str(src_file), "lineno": 2, "qualname": "f"},
         "ended": False},
    ]
    ctx = {"workdir": str(tmp_path), "repo_root": tmp_path,
           "_pdb_session_log": pdb_log}
    out = rca_enrich.enrich(report, container=None, ctx=ctx, wall_budget_s=10.0)
    frames = out["propagation_path"]["frames"]
    visited = [(f["file"], f["lineno"]) for f in frames]
    assert (str(src_file), 2) in visited
