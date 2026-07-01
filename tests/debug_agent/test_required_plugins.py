from unittest.mock import MagicMock

from debug_agent.required_plugins import (
    detect_required_plugins, required_plugin_short_names,
)


def _container_with(files):
    c = MagicMock()
    def exec_bash(cmd, *a, **kw):
        if cmd.startswith("cat "):
            path = cmd[4:].strip()
            if path in files:
                r = MagicMock(); r.returncode = 0; r.stdout = files[path]; r.stderr = ""
                return r
            r = MagicMock(); r.returncode = 1; r.stdout = ""; r.stderr = ""
            return r
        r = MagicMock(); r.returncode = 0; r.stdout = ""; r.stderr = ""
        return r
    c.exec_bash.side_effect = exec_bash
    return c


def test_detect_required_plugins_setup_cfg():
    c = _container_with({
        "/app/setup.cfg": "[tool:pytest]\nrequiredplugins = pytest-rerunfailures pytest-mock\n",
    })
    assert detect_required_plugins(c, "/app") == ["pytest-rerunfailures", "pytest-mock"]


def test_detect_required_plugins_with_version():
    c = _container_with({
        "/app/pytest.ini": "[pytest]\nrequired_plugins = pytest-rerunfailures>=9.0\n",
    })
    assert detect_required_plugins(c, "/app") == ["pytest-rerunfailures"]


def test_detect_required_plugins_none():
    c = _container_with({"/app/setup.cfg": "[tool:pytest]\naddopts = -ra\n"})
    assert detect_required_plugins(c, "/app") == []


def test_short_names_strips_pytest_prefix():
    assert required_plugin_short_names(["pytest-rerunfailures", "pytest-mock"]) == {
        "rerunfailures", "mock",
    }


def test_short_names_empty():
    assert required_plugin_short_names([]) == set()


def test_pdb_session_cmd_skips_required_plugin():
    """The disable loop should NOT add `-p no:rerunfailures` if the project
    declares pytest-rerunfailures as required — that's what was breaking
    qutebrowser pdb_start."""
    from debug_agent.pdb_session import _build_session_cmd
    cmd = _build_session_cmd(
        "fake_cid", ["-m", "pytest", "_repro/x.py"],
        required_short_names={"rerunfailures"},
    )
    flat = " ".join(cmd)
    assert "no:rerunfailures" not in flat
    assert "no:xdist" in flat  # xdist isn't required → still disabled
    assert "no:forked" in flat


def test_pdb_session_cmd_strips_agent_injected_disable_for_required():
    """If the LLM (or addopts) put `-p no:rerunfailures` in run_args, AND
    the project requires rerunfailures, we must strip it — else pytest
    aborts with `Missing required plugins`."""
    from debug_agent.pdb_session import _build_session_cmd
    cmd = _build_session_cmd(
        "fake_cid",
        ["-m", "pytest", "-p", "no:rerunfailures", "_repro/x.py"],
        required_short_names={"rerunfailures"},
    )
    flat = " ".join(cmd)
    assert "no:rerunfailures" not in flat


def test_pdb_session_cmd_disables_when_no_required():
    from debug_agent.pdb_session import _build_session_cmd
    cmd = _build_session_cmd(
        "fake_cid", ["-m", "pytest", "_repro/x.py"],
        required_short_names=set(),
    )
    flat = " ".join(cmd)
    assert "no:rerunfailures" in flat
    assert "no:xdist" in flat
    assert "no:forked" in flat
