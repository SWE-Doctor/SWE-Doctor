import subprocess

import pytest

from debug_agent.container import Container
from debug_agent.source_reader import grep_src, read


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="docker not available")


@pytest.mark.docker
def test_read_returns_requested_range():
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        body = "\n".join(f"line{i}" for i in range(1, 201)) + "\n"
        c.write_file("/work/big.txt", body)
        out = read(c, "/work/big.txt", start=50, n=10)
        lines = out.strip().splitlines()
        assert len(lines) == 10
        assert lines[0] == "line50"
        assert lines[-1] == "line59"
    finally:
        c.terminate()


@pytest.mark.docker
def test_grep_src_returns_lineno_prefixed_hits():
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        c.write_file(
            "/work/m.py",
            "x = 1\ndef foo():\n    pass\ndef bar():\n    pass\n",
        )
        out = grep_src(c, r"def foo|def bar", "/work/m.py")
        assert "2:def foo" in out
        assert "4:def bar" in out
    finally:
        c.terminate()
