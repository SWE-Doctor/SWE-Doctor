"""R2: pdb prints '*** Blank or comment' / '*** End of file' / 'Breakpoint
... was not set' on bad `b` targets but reports no error code. The agent
then `c`s past and the program exits without ever stopping. Tag the
warning on CmdResult so the dispatcher can prepend a [WARNING] line."""
import subprocess
import textwrap

import pytest

from debug_agent.container import Container
from debug_agent.pdb_session import PdbSession


def test_cmdresult_has_last_eval_warning_field():
    """Pure unit test — no docker. Just verify the dataclass field exists."""
    from debug_agent.pdb_session import CmdResult
    r = CmdResult()
    assert hasattr(r, "last_eval_warning")
    assert r.last_eval_warning is None


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="docker not available")


@pytest.fixture
def slim_container():
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    yield c
    c.terminate()


@pytest.mark.docker
def test_blank_line_breakpoint_surfaces_warning(slim_container):
    slim_container.write_file(
        "/work/prog.py",
        textwrap.dedent("""\
            # comment line 1
            # comment line 2

            def main():
                return 1
            main()
        """),
    )
    s = PdbSession(slim_container, run_args=["/work/prog.py"])
    res = s.start()
    try:
        assert res.ok is True
        out = s.cmd("b /work/prog.py:3")  # line 3 is blank
        assert out.last_eval_warning == "blank_or_comment_breakpoint", out.transcript
    finally:
        s.stop()


@pytest.mark.docker
def test_eof_breakpoint_surfaces_warning(slim_container):
    slim_container.write_file(
        "/work/prog.py",
        "def main():\n    return 1\nmain()\n",
    )
    s = PdbSession(slim_container, run_args=["/work/prog.py"])
    res = s.start()
    try:
        assert res.ok is True
        out = s.cmd("b /work/prog.py:9999")
        assert out.last_eval_warning == "breakpoint_past_eof", out.transcript
    finally:
        s.stop()
