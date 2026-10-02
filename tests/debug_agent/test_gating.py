"""Pure tests for the can_conclude predicate driven by ctx['_pdb_session_log']."""
from debug_agent.gating import can_conclude, MIN_PDB_CMDS


def _pytest_turn(ok=True):
    return {"action_name": "pytest", "tool_output_ok": ok}


def _start(sid, frame=None):
    return {"kind": "start", "session_id": sid, "ok": True, "initial_frame": frame}


def _cmd(sid, frame=None, ended=False):
    return {"kind": "cmd", "session_id": sid, "current_frame": frame,
            "ended": ended, "end_reason": ""}


def test_no_pytest_evidence_blocks():
    history = [{"action_name": "bash", "tool_output_ok": True}]
    log = []
    assert can_conclude(history, log) is False


def test_pytest_only_blocks():
    history = [_pytest_turn()]
    log = []
    assert can_conclude(history, log) is False


def test_pytest_plus_pdb_start_no_cmds_blocks():
    history = [_pytest_turn()]
    log = [_start(1, frame={"file": "t.py", "lineno": 5, "qualname": "f"})]
    assert can_conclude(history, log) is False


def test_pytest_plus_continue_to_end_blocks():
    history = [_pytest_turn()]
    real = {"file": "t.py", "lineno": 5, "qualname": "f"}
    log = [_start(1, frame=real)] + [_cmd(1, frame=None, ended=True)] * 5
    assert can_conclude(history, log) is False


def test_pytest_plus_real_stop_plus_min_cmds_unblocks():
    history = [_pytest_turn()]
    real = {"file": "t.py", "lineno": 5, "qualname": "f"}
    log = [_start(1, frame=real)] + [_cmd(1, frame=real)] * MIN_PDB_CMDS
    assert can_conclude(history, log) is True


def test_pdb_internal_frame_does_not_count_as_real_stop():
    history = [_pytest_turn()]
    fake = {"file": "/usr/lib/python3.11/pdb.py", "lineno": 100, "qualname": "Pdb.user_line"}
    log = [_start(1, frame=fake)] + [_cmd(1, frame=fake)] * MIN_PDB_CMDS
    assert can_conclude(history, log) is False


def test_pytest_under_pdb_start_counts_as_pytest_evidence():
    """Agent ran `pdb_start run_args=[pytest, -x, t.py]` — that IS a pytest run.
    No separate <action name="pytest"> should be required."""
    history: list[dict] = []
    real = {"file": "t.py", "lineno": 5, "qualname": "f"}
    log = [
        {"kind": "start", "session_id": 1, "ok": True,
         "initial_frame": real, "run_args": ["pytest", "-x", "/app/t.py"]},
    ] + [_cmd(1, frame=real)] * MIN_PDB_CMDS
    assert can_conclude(history, log) is True


def test_pytest_under_pdb_with_dash_m_form_counts():
    history: list[dict] = []
    real = {"file": "t.py", "lineno": 5, "qualname": "f"}
    log = [
        {"kind": "start", "session_id": 1, "ok": True,
         "initial_frame": real, "run_args": ["-m", "pytest", "-x", "/app/t.py"]},
    ] + [_cmd(1, frame=real)] * MIN_PDB_CMDS
    assert can_conclude(history, log) is True


def test_pdb_start_with_plain_script_is_not_pytest_evidence():
    history: list[dict] = []
    real = {"file": "t.py", "lineno": 5, "qualname": "f"}
    log = [
        {"kind": "start", "session_id": 1, "ok": True,
         "initial_frame": real, "run_args": ["/app/repro.py"]},
    ] + [_cmd(1, frame=real)] * MIN_PDB_CMDS
    assert can_conclude(history, log) is False
