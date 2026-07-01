import subprocess

import pytest

from debug_agent.container import Container
from debug_agent.pytest_runner import run_pytest


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="docker not available")

_FAILING_TEST = """\
def test_ok():
    assert 1 == 1


def test_bad():
    assert 2 + 2 == 5, "math is broken"
"""


@pytest.mark.docker
def test_run_pytest_parses_failed_tests():
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        c.exec_bash("pip install --quiet pytest")
        c.write_file("/work/test_sample.py", _FAILING_TEST)
        result = run_pytest(c, "-v test_sample.py")
        assert result.returncode != 0
        assert len(result.failed_tests) == 1
        ft = result.failed_tests[0]
        assert ft.nodeid.endswith("test_sample.py::test_bad")
        assert "AssertionError" in ft.exc_type
    finally:
        c.terminate()


@pytest.mark.docker
def test_run_pytest_returns_zero_on_pass():
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        c.exec_bash("pip install --quiet pytest")
        c.write_file("/work/test_green.py", "def test_ok():\n    assert True\n")
        result = run_pytest(c, "-v test_green.py")
        assert result.returncode == 0
        assert result.failed_tests == []
    finally:
        c.terminate()
