import subprocess

import pytest

from debug_agent.container import Container
from debug_agent.probe import apply_probe


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="docker not available")


@pytest.mark.docker
def test_apply_probe_prints_expr_then_reverts():
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        c.write_file(
            "/work/m.py",
            "def f(x):\n"
            "    y = x + 1\n"
            "    return y\n"
            "print(f(10))\n",
        )
        original = c.read_file("/work/m.py")
        result = apply_probe(
            c,
            file="/work/m.py",
            before_line=3,
            expr="y",
            run_cmd="python /work/m.py",
        )
        assert "PROBE" in result.stdout
        assert "11" in result.stdout
        assert c.read_file("/work/m.py") == original
    finally:
        c.terminate()


@pytest.mark.docker
def test_apply_probe_reverts_on_subprocess_crash():
    """Even if the injected probe causes a NameError in the target, the file
    must be reverted."""
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        c.write_file(
            "/work/m.py",
            "def f():\n    return 1\nf()\n",
        )
        original = c.read_file("/work/m.py")
        result = apply_probe(
            c,
            file="/work/m.py",
            before_line=2,
            expr="nonexistent_var",
            run_cmd="python /work/m.py",
        )
        # Probe itself raised a NameError at runtime (nonzero rc), but revert still happened.
        assert result.returncode != 0
        assert c.read_file("/work/m.py") == original
    finally:
        c.terminate()
