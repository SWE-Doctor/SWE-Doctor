import subprocess

import pytest

from debug_agent.container import Container


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="docker not available")


@pytest.mark.docker
def test_container_write_snapshot_restore():
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        c.exec_bash("echo 'original' > /work/a.txt")
        c.snapshot("/work/a.txt")
        c.write_file("/work/a.txt", "modified\n")
        assert c.read_file("/work/a.txt").strip() == "modified"
        c.restore("/work/a.txt")
        assert c.read_file("/work/a.txt").strip() == "original"
    finally:
        c.terminate()


@pytest.mark.docker
def test_container_exec_bash_returns_rc_stdout_stderr():
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        r = c.exec_bash("echo hi; echo err 1>&2; exit 3")
        assert r.returncode == 3
        assert "hi" in r.stdout
        assert "err" in r.stderr
    finally:
        c.terminate()


@pytest.mark.docker
def test_container_write_large_file_exceeds_arg_max():
    """write_file must handle content whose base64 exceeds the single-argument
    limit (Linux MAX_ARG_STRLEN, 128 KiB). Previously the base64 was inlined
    into the command string, so a large source file raised
    OSError(7, 'Argument list too long') and aborted RCA for that instance."""
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        big = "x" * 300_000  # base64 ~400 KiB, well over the 128 KiB single-arg cap
        c.write_file("/work/big.txt", big)
        assert c.read_file("/work/big.txt") == big
    finally:
        c.terminate()


@pytest.mark.docker
def test_container_attach_terminate_does_not_stop():
    """Attach-mode must NOT docker stop the container (someone else owns it)."""
    owned = Container.launch(image="python:3.11-slim", workdir="/work")
    try:
        attached = Container.attach(owned.container_id, "/work")
        attached.exec_bash("echo hi > /work/x.txt")
        attached.terminate()  # no-op for docker stop
        # owned container should still be alive
        r = owned.exec_bash("cat /work/x.txt")
        assert "hi" in r.stdout
    finally:
        owned.terminate()
