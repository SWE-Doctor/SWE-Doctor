import subprocess
import pytest
from debug_agent.container import Container
from debug_agent.pdb_session import PdbSession, _normalize_pdb_args


def test_normalize_pdb_args_strips_python_and_adds_dash_m():
    assert _normalize_pdb_args(["pytest", "-x", "/app/t.py"]) == ["-m", "pytest", "-x", "/app/t.py"]
    assert _normalize_pdb_args(["python", "-m", "pytest", "-x", "/app/t.py"]) == [
        "-m", "pytest", "-x", "/app/t.py",
    ]
    assert _normalize_pdb_args(["/usr/local/bin/python", "-m", "pytest", "-x", "/app/t.py"]) == [
        "-m", "pytest", "-x", "/app/t.py",
    ]
    assert _normalize_pdb_args(["python3.10", "-u", "-m", "unittest", "discover"]) == [
        "-m", "unittest", "discover",
    ]
    assert _normalize_pdb_args(["unittest", "discover", "-s", "/app"]) == [
        "-m", "unittest", "discover", "-s", "/app",
    ]
    # Already a -m form, leave alone.
    assert _normalize_pdb_args(["-m", "pytest", "-x"]) == ["-m", "pytest", "-x"]
    # Plain script path passes through.
    assert _normalize_pdb_args(["/app/repro.py", "arg"]) == ["/app/repro.py", "arg"]
    assert _normalize_pdb_args([]) == []


def test_is_pytest_invocation_recognizes_normalized_pytest_forms():
    from debug_agent.pdb_session import _is_pytest_invocation
    # After _normalize_pdb_args, pytest payloads look like ["-m", "pytest", ...]
    assert _is_pytest_invocation(["-m", "pytest", "-x", "/app/t.py"]) is True
    assert _is_pytest_invocation(["-m", "pytest"]) is True


def test_is_pytest_invocation_rejects_other_modules_and_scripts():
    from debug_agent.pdb_session import _is_pytest_invocation
    assert _is_pytest_invocation(["-m", "unittest", "discover"]) is False
    assert _is_pytest_invocation(["/app/repro.py"]) is False
    assert _is_pytest_invocation([]) is False


def test_build_session_cmd_pytest_mode_uses_pytest_pdb_no_outer_pdb():
    """When run_args invoke pytest, we drop the outer `python -m pdb` and
    use `python -m pytest --pdb …` instead. pytest's --pdb auto-drops into
    pdb.post_mortem() at the failing assertion's frame, which is what we
    actually want."""
    from debug_agent.pdb_session import _build_session_cmd
    cmd = _build_session_cmd(container_id="cid", run_args=["pytest", "-x", "/app/t.py"])
    # `docker exec -i [-e PYTEST_ADDOPTS=] cid python -u …` — the -e flag is
    # required to clear PYTEST_ADDOPTS that swebench-pro images preset.
    assert cmd[:3] == ["docker", "exec", "-i"]
    assert "cid" in cmd
    py_idx = cmd.index("python")
    assert cmd[py_idx:py_idx + 2] == ["python", "-u"]
    # Outer "-m pdb" must NOT be present immediately after python.
    assert cmd[py_idx + 2:py_idx + 4] != ["-m", "pdb"], cmd
    # Must invoke pytest as a module with --pdb.
    assert "-m" in cmd and "pytest" in cmd and "--pdb" in cmd, cmd
    # Original args (-x /app/t.py) preserved.
    assert "/app/t.py" in cmd and "-x" in cmd


def test_build_session_cmd_pytest_does_not_double_inject_pdb_flag():
    """If the agent already passed --pdb, don't add a second one."""
    from debug_agent.pdb_session import _build_session_cmd
    cmd = _build_session_cmd(container_id="cid",
                             run_args=["pytest", "--pdb", "-x", "/app/t.py"])
    assert cmd.count("--pdb") == 1, cmd


def test_build_session_cmd_non_pytest_keeps_outer_pdb_wrapper():
    """Module / script invocations stay on the legacy `python -m pdb` path."""
    from debug_agent.pdb_session import _build_session_cmd
    # Module form
    cmd = _build_session_cmd(container_id="cid",
                             run_args=["unittest", "discover", "-s", "/app"])
    assert cmd[4:8] == ["python", "-u", "-m", "pdb"]
    assert cmd[8:11] == ["-m", "unittest", "discover"]
    # Script form
    cmd = _build_session_cmd(container_id="cid", run_args=["/app/repro.py", "arg"])
    assert cmd[4:8] == ["python", "-u", "-m", "pdb"]
    assert cmd[8:10] == ["/app/repro.py", "arg"]


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
def test_pdb_session_program_exits_without_stop(slim_container):
    # `python -m pdb /nope.py` errors out before installing the trace hook —
    # the subprocess prints a usage / FileNotFoundError and exits with no Pdb
    # prompt. We treat that as "program_exited_without_stop".
    s = PdbSession(slim_container, run_args=["/work/does_not_exist_xyz.py"])
    res = s.start()
    assert res.ok is False
    assert res.reason == "program_exited_without_stop"
    s.stop()


@pytest.mark.docker
def test_pdb_session_start_pauses_on_first_line(slim_container):
    slim_container.write_file(
        "/work/t.py",
        "def f():\n    x = 42\n    return x\n\nf()\n",
    )
    s = PdbSession(slim_container, run_args=["/work/t.py"])
    res = s.start()
    try:
        assert res.ok is True
        assert res.initial_frame is not None
        assert res.initial_frame["file"].endswith("/work/t.py")
        assert "(Pdb)" in res.banner or "Pdb" in res.banner
    finally:
        s.stop()


@pytest.mark.docker
def test_pdb_session_cmd_breakpoint_and_print(slim_container):
    slim_container.write_file(
        "/work/t.py",
        "def f():\n    x = 42\n    return x\n\nf()\n",
    )
    s = PdbSession(slim_container, run_args=["/work/t.py"])
    s.start()
    try:
        s.cmd("b /work/t.py:3")
        r = s.cmd("c")
        assert r.ended is False
        assert r.current_frame is not None
        assert r.current_frame["lineno"] == 3
        r2 = s.cmd("p x")
        assert "42" in r2.transcript
        assert r2.last_eval.strip() == "42"
    finally:
        s.stop()


@pytest.mark.docker
def test_pdb_session_cmd_program_finishes(slim_container):
    slim_container.write_file(
        "/work/t.py",
        "def f():\n    x = 42\n    return x\n\nf()\n",
    )
    s = PdbSession(slim_container, run_args=["/work/t.py"])
    s.start()
    try:
        r = s.cmd("c")
        assert r.ended is True
        assert r.end_reason == "program_exited"
    finally:
        s.stop()


@pytest.mark.docker
def test_pdb_session_cmd_exception_kills_session(slim_container):
    slim_container.write_file(
        "/work/boom.py",
        "def f():\n    raise RuntimeError('boom')\n\nf()\n",
    )
    s = PdbSession(slim_container, run_args=["/work/boom.py"])
    s.start()
    try:
        r = s.cmd("c")
        assert r.ended is True
        assert r.end_reason in ("exception", "program_exited")
    finally:
        s.stop()


@pytest.mark.docker
def test_pdb_session_cmd_timeout_keeps_session_alive(slim_container):
    slim_container.write_file(
        "/work/sleep.py",
        "import time\n"
        "def f():\n    time.sleep(60)\n\n"
        "f()\n",
    )
    s = PdbSession(slim_container, run_args=["/work/sleep.py"], timeout_per_cmd=2.0)
    s.start()
    try:
        r = s.cmd("c")
        assert r.ended is False
        assert r.end_reason == "timeout"
        assert "[timeout]" in r.transcript
    finally:
        s.stop()


@pytest.mark.docker
def test_pdb_session_restart_idempotent(slim_container):
    slim_container.write_file("/work/t.py", "x = 1\n")
    s = PdbSession(slim_container, run_args=["/work/t.py"])
    s.start()
    s.stop()
    s.stop()  # idempotent
    r = s.restart()
    assert r.ok is True
    s.stop()


def test_pdb_session_truncates_large_transcript(tmp_path, monkeypatch):
    # Pure unit test for the truncation helper — no docker.
    from debug_agent.pdb_session import _maybe_truncate
    long = "X" * 20_000
    truncated, sidecar = _maybe_truncate(long, log_dir=tmp_path, session_id=1, turn=2)
    assert len(truncated) <= 8_500  # 8 KB + small marker
    assert "[truncated" in truncated
    assert sidecar is not None and sidecar.exists()
    assert sidecar.read_text() == long
