"""R1: pytest --pdb mode must work in containers shipping pytest plugins
that conflict with --pdb (rerunfailures, xdist, forked). Without these
disable flags pytest aborts before installing post_mortem and pdb_start
silently returns program_exited_without_stop."""
from debug_agent.pdb_session import _build_session_cmd


def test_pytest_pdb_disables_rerunfailures_xdist_forked_by_default():
    cmd = _build_session_cmd(container_id="cid", run_args=["pytest", "-x", "/app/t.py"])
    flat = " ".join(cmd)
    assert "--pdb" in cmd
    assert "-p no:rerunfailures" in flat, cmd
    assert "-p no:xdist" in flat, cmd
    assert "-p no:forked" in flat, cmd
    # SWE-bench-pro images preset PYTEST_ADDOPTS=--reruns=3 in the container
    # env; clear it for this invocation via docker exec -e.
    assert "-e" in cmd and "PYTEST_ADDOPTS=" in cmd, cmd
    # Belt-and-suspenders: also clear ini-file addopts.
    assert "-o addopts=" in flat, cmd
    # User-supplied flags preserved
    assert "-x" in cmd
    assert "/app/t.py" in cmd


def test_pytest_pdb_does_not_double_inject_addopts_override():
    cmd = _build_session_cmd(
        container_id="cid",
        run_args=["pytest", "-o", "addopts=-q", "/app/t.py"],
    )
    flat = " ".join(cmd)
    # User passed their own -o addopts=...; don't add another empty override
    assert flat.count("-o addopts=") == 1, cmd
    assert "addopts=-q" in flat


def test_pytest_pdb_does_not_double_inject_disable_flags():
    cmd = _build_session_cmd(
        container_id="cid",
        run_args=["pytest", "-p", "no:rerunfailures", "/app/t.py"],
    )
    assert " ".join(cmd).count("-p no:rerunfailures") == 1, cmd


def test_non_pytest_payload_unchanged_by_compat_logic():
    cmd = _build_session_cmd(container_id="cid", run_args=["/app/repro.py"])
    flat = " ".join(cmd)
    assert "rerunfailures" not in flat
    assert "--pdb" not in flat  # plain script keeps outer `python -m pdb`
