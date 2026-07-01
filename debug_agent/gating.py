"""Conclusion gate for the debug analyzer.

A conclusion is allowed only when:
  1) at least one pytest action succeeded (we saw a real failure signal), AND
  2) the agent drove a stateful pdb session past a real (non-pdb-internal)
     stop frame for at least MIN_PDB_CMDS commands."""
from __future__ import annotations

MIN_PDB_CMDS = 3


def _is_real_frame(frame: dict | None) -> bool:
    if not frame:
        return False
    f = (frame.get("file") or "").lower()
    if "/pdb.py" in f or f.endswith("pdb.py") or "/bdb.py" in f:
        return False
    # go stdlib / module cache / vendored deps are not the project's own code,
    # so a dlv stop there (e.g. testing.tRunner) must not satisfy the gate.
    if "/usr/local/go/src/" in f or "/go/pkg/mod/" in f:
        return False
    if f.startswith("vendor/") or "/vendor/" in f:
        return False
    return True


def _had_pytest_evidence(history: list[dict], pdb_log: list[dict] | None = None) -> bool:
    if any(t.get("action_name") in ("pytest", "gotest") and t.get("tool_output_ok") for t in history):
        return True
    # Running the test *under* a debug session counts as evidence: if the agent
    # gets to a real frame, the test ran for real.
    for t in pdb_log or []:
        if t.get("kind") != "start" or not t.get("ok"):
            continue
        run_args = t.get("run_args") or []
        if any(a in {"pytest", "-m"} for a in run_args[:2]) and "pytest" in run_args:
            return True
        # go: a go-test debug session (factory builds `dlv test <pkg>`); run_args
        # carry a package path (./pkg) or a -run/-test.run test selector.
        if any(str(a).startswith("./") or str(a) in ("-run", "-test.run") for a in run_args):
            return True
    return False


def can_conclude(history: list[dict], pdb_log: list[dict]) -> bool:
    if not _had_pytest_evidence(history, pdb_log):
        return False
    by_sid: dict[int, list[dict]] = {}
    for t in pdb_log:
        sid = t.get("session_id")
        if sid is None:
            continue
        by_sid.setdefault(sid, []).append(t)
    for entries in by_sid.values():
        first_real_stop = None
        for i, t in enumerate(entries):
            frame = t.get("initial_frame") if t.get("kind") == "start" else t.get("current_frame")
            if _is_real_frame(frame):
                first_real_stop = i
                break
        if first_real_stop is None:
            continue
        cmds_after = sum(
            1 for t in entries[first_real_stop + 1:]
            if t.get("kind") == "cmd" and not t.get("ended")
        )
        if cmds_after >= MIN_PDB_CMDS:
            return True
    return False
