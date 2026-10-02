import subprocess

import pytest

from debug_agent.container import Container
from debug_agent.pdb_runner import run_pdb_script


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="docker not available")


@pytest.mark.docker
def test_pdb_runner_prints_variable_at_breakpoint():
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        c.write_file(
            "/work/t.py",
            "def f():\n"
            "    x = 42\n"
            "    return x\n"
            "f()\n",
        )
        tr = run_pdb_script(
            c,
            script_path="/work/t.py",
            commands=["b /work/t.py:3", "c", "p x", "q"],
        )
        assert "42" in tr
    finally:
        c.terminate()
