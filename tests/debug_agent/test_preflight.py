from debug_agent.container import ExecResult
from debug_agent.preflight import bootstrap


class _C:
    """Fake container that scripts exec_bash responses in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def exec_bash(self, cmd, timeout=120):
        self.calls.append(cmd)
        # `_ensure_required_plugins` probes pytest config files via `cat`;
        # existing scripts don't enumerate those probes, so treat any
        # `cat <path>` call as "file not present" without consuming a script
        # slot.
        if cmd.startswith("cat "):
            return ExecResult(1, "", "")
        rc, out, err = self.script.pop(0)
        return ExecResult(rc, out, err)


def test_bootstrap_noop_when_baseline_run_already_hits_real_failure():
    # Baseline run reaches an AssertionError — no env fixups needed.
    c = _C([
        (1, "FAILED _repro/test_a0.py::x - AssertionError: foo", ""),
    ])
    rep = bootstrap(c, workdir="/app", repro_nodeid="_repro/test_a0.py")
    assert rep.ready is True
    assert rep.actions == []
    assert "AssertionError" in rep.initial_output
    # No collect-only call; readiness is decided by the real run.
    pytest_calls = [c_ for c_ in c.calls if not c_.startswith("cat ")]
    assert len(pytest_calls) == 1
    assert "pytest -x" in pytest_calls[0]
    assert "--collect-only" not in pytest_calls[0]


def test_bootstrap_sets_pythonpath_on_module_not_found():
    # Baseline run → MNFE. PYTHONPATH run → AssertionError (ready).
    c = _C([
        (1, "", "ModuleNotFoundError: No module named 'openlibrary'"),
        (1, "FAILED _repro/test_a0.py::x - AssertionError: foo", ""),
    ])
    rep = bootstrap(c, workdir="/app", repro_nodeid="_repro/test_a0.py")
    assert rep.ready is True
    assert "set PYTHONPATH=/app" in rep.actions
    assert "PYTHONPATH=/app" in rep.env_prefix
    assert "AssertionError" in rep.initial_output
    assert "ModuleNotFoundError" not in rep.initial_output


def test_bootstrap_detects_function_body_import_failures():
    # Guards the original Phase-C smoke bug: a test whose real import happens
    # inside the function body would pass --collect-only yet fail at runtime.
    # The baseline run must observe the runtime MNFE and escalate.
    c = _C([
        (1, "", "ModuleNotFoundError: No module named 'openlibrary'"),   # baseline run
        (1, "FAILED ... AssertionError: x != y", ""),                    # PYTHONPATH run
    ])
    rep = bootstrap(c, workdir="/app", repro_nodeid="_repro/test_a0.py")
    assert rep.ready is True
    assert rep.actions == ["set PYTHONPATH=/app"]


def test_bootstrap_reports_not_ready_if_all_fixes_fail():
    c = _C([
        (1, "", "ModuleNotFoundError: No module named 'openlibrary'"),   # baseline
        (1, "", "ModuleNotFoundError: No module named 'openlibrary'"),   # PYTHONPATH
        (0, "", ""),                                                      # setup.py detected
        (1, "ERROR: could not install", "pip install failed"),            # pip install
        (1, "", "ModuleNotFoundError: No module named 'openlibrary'"),   # after pip
    ])
    rep = bootstrap(c, workdir="/app", repro_nodeid="_repro/test_a0.py")
    assert rep.ready is False
    assert "set PYTHONPATH=/app" in rep.actions
    assert "pip install -e ." in rep.actions
    assert "ModuleNotFoundError" in rep.initial_output


def test_bootstrap_empty_repro_nodeid_returns_not_ready():
    c = _C([])
    rep = bootstrap(c, workdir="/app", repro_nodeid="")
    assert rep.ready is False
    assert "no repro" in rep.initial_output.lower()


def test_bootstrap_uses_pipefail_so_broken_pipeline_does_not_mask_rc():
    # Guards the latent bug where `... | tail -n 80` made pipeline rc always 0.
    c = _C([(1, "FAILED ... AssertionError", "")])
    rep = bootstrap(c, workdir="/app", repro_nodeid="_repro/t.py")
    assert rep.ready is True
    pytest_calls = [c_ for c_ in c.calls if not c_.startswith("cat ")]
    assert "pipefail" in pytest_calls[0]
