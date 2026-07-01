from unittest.mock import MagicMock

from debug_agent.preflight import _ensure_required_plugins


def _container_with_files(file_contents: dict[str, str]):
    """Mock container whose exec_bash returns file contents for `cat <path>`."""
    c = MagicMock()
    calls = []

    def exec_bash(cmd, *args, **kwargs):
        calls.append(cmd)
        if cmd.startswith("cat "):
            path = cmd[4:].strip()
            if path in file_contents:
                r = MagicMock()
                r.returncode = 0
                r.stdout = file_contents[path]
                r.stderr = ""
                return r
            r = MagicMock()
            r.returncode = 1
            r.stdout = ""
            r.stderr = ""
            return r
        # any other shell op (pip install, etc.)
        r = MagicMock()
        r.returncode = 0
        r.stdout = ""
        r.stderr = ""
        return r

    c.exec_bash.side_effect = exec_bash
    c._calls = calls
    return c


def test_installs_rerunfailures_when_required_in_setup_cfg():
    c = _container_with_files({
        "/app/setup.cfg":
            "[tool:pytest]\nrequiredplugins = pytest-rerunfailures pytest-mock\n",
    })
    installed = _ensure_required_plugins(c, "/app")
    assert "pytest-rerunfailures" in installed
    assert "pytest-mock" in installed
    assert any("pip install" in call and "pytest-rerunfailures" in call
               for call in c._calls)


def test_installs_from_pytest_ini():
    c = _container_with_files({
        "/app/pytest.ini": "[pytest]\nrequired_plugins = pytest-rerunfailures>=9.0\n",
    })
    installed = _ensure_required_plugins(c, "/app")
    assert "pytest-rerunfailures" in installed


def test_no_install_when_not_required():
    c = _container_with_files({
        "/app/setup.cfg": "[tool:pytest]\naddopts = -ra\n",
    })
    installed = _ensure_required_plugins(c, "/app")
    assert installed == []
    assert not any("pip install" in call for call in c._calls)


def test_no_files_returns_empty():
    c = _container_with_files({})
    installed = _ensure_required_plugins(c, "/app")
    assert installed == []
