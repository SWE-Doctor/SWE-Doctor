from debug_agent.branch_observation import collect_branch_observations


def _src_resolver(files: dict[str, str]):
    return lambda path: files.get(path)


def test_then_arm_taken(tmp_path):
    src = (
        "def f(x):\n"
        "    if x is None:\n"            # line 2 — IF
        "        return -1\n"             # line 3 — then arm
        "    return x + 1\n"              # line 4 — else arm
    )
    log = [
        {"kind": "cmd", "session_id": 1, "cmd": "b mod.py:2",
         "current_frame": {"file": "/repo/mod.py", "lineno": 2, "qualname": "f"},
         "ended": False},
        {"kind": "cmd", "session_id": 1, "cmd": "p x",
         "current_frame": {"file": "/repo/mod.py", "lineno": 2, "qualname": "f"},
         "last_eval": "None", "ended": False},
        {"kind": "cmd", "session_id": 1, "cmd": "n",
         "current_frame": {"file": "/repo/mod.py", "lineno": 3, "qualname": "f"},
         "ended": False},
    ]
    obs = collect_branch_observations(log, _src_resolver({"/repo/mod.py": src}))
    armed = [o for o in obs if o["provenance"]["kind"] == "arm_inferred"]
    assert len(armed) == 1
    o = armed[0]
    assert o["file"] == "/repo/mod.py"
    assert o["lineno"] == 2
    assert o["arm_taken"] == "then"
    assert "if x is None" in o["cond_text"]
    assert o["locals_at_stop"].get("x") == "None"


def test_else_arm_taken():
    src = (
        "def f(x):\n"
        "    if x > 0:\n"
        "        y = 1\n"
        "    else:\n"
        "        y = 2\n"
        "    return y\n"
    )
    log = [
        {"kind": "cmd", "session_id": 1, "cmd": "b m.py:2",
         "current_frame": {"file": "/m.py", "lineno": 2, "qualname": "f"},
         "ended": False},
        {"kind": "cmd", "session_id": 1, "cmd": "n",
         "current_frame": {"file": "/m.py", "lineno": 5, "qualname": "f"},
         "ended": False},
    ]
    obs = collect_branch_observations(log, _src_resolver({"/m.py": src}))
    assert obs[0]["arm_taken"] == "else"


def test_loop_skipped():
    src = (
        "def f(items):\n"
        "    total = 0\n"
        "    for x in items:\n"           # line 3
        "        total += x\n"             # line 4
        "    return total\n"               # line 5 (loop skipped → here next)
    )
    log = [
        {"kind": "cmd", "session_id": 1, "cmd": "b m.py:3",
         "current_frame": {"file": "/m.py", "lineno": 3, "qualname": "f"},
         "ended": False},
        {"kind": "cmd", "session_id": 1, "cmd": "n",
         "current_frame": {"file": "/m.py", "lineno": 5, "qualname": "f"},
         "ended": False},
    ]
    obs = collect_branch_observations(log, _src_resolver({"/m.py": src}))
    assert obs[0]["arm_taken"] == "loop_skipped"


def test_branch_observation_fires_when_stop_is_inside_if_body_not_on_header():
    """Real trajectories almost never stop exactly on the `if` line — they
    stop at a breakpoint inside the body. Match the enclosing If when the
    stop line is within its body."""
    src = (
        "def f(x):\n"           # 1
        "    if x is None:\n"   # 2  <- header
        "        y = 0\n"       # 3  <- body (then-arm)
        "        return y\n"    # 4  <- body (then-arm)
        "    else:\n"           # 5
        "        return -1\n"   # 6  <- body (else-arm)
    )
    log = [
        {"kind": "cmd", "session_id": 1,
         "current_frame": {"file": "m.py", "lineno": 3, "qualname": "f"},
         "ended": False},
        {"kind": "cmd", "session_id": 1,
         "current_frame": {"file": "m.py", "lineno": 4, "qualname": "f"},
         "ended": False},
    ]
    obs = collect_branch_observations(log, source_resolver=lambda _f: src)
    armed = [o for o in obs if o["provenance"]["kind"] == "arm_inferred"]
    assert len(armed) == 1, obs
    o = armed[0]
    assert o["arm_taken"] == "then"
    assert "if x is None" in o["cond_text"]


def test_no_control_flow_at_stop_yields_nothing():
    src = "def f():\n    return 1\n"
    log = [
        {"kind": "cmd", "session_id": 1, "cmd": "b m.py:2",
         "current_frame": {"file": "/m.py", "lineno": 2, "qualname": "f"},
         "ended": False},
    ]
    obs = collect_branch_observations(log, _src_resolver({"/m.py": src}))
    assert obs == []


# --- Lever 4 frame_only fallback ----------------------------------------------

def test_frame_only_emitted_when_no_step_pair():
    """A single 'pp x' stop inside an `if` body must yield one observation
    tagged provenance.kind='frame_only'."""
    src = (
        "def f(x):\n"
        "    if x > 0:\n"
        "        a = 1\n"
        "        return a\n"
        "    else:\n"
        "        b = 2\n"
        "        return b\n"
    )
    log = [
        {"kind": "cmd", "cmd": "pp x", "last_eval": "5",
         "current_frame": {"file": "/m.py", "lineno": 3, "qualname": "f"}},
    ]
    obs = collect_branch_observations(log, source_resolver=lambda _p: src)
    assert len(obs) == 1
    o = obs[0]
    assert o["file"] == "/m.py" and o["lineno"] == 3
    assert "if x > 0" in o["cond_text"]
    assert o["provenance"]["kind"] == "frame_only"


def test_arm_inferred_kept_when_step_pair_present():
    """Step pair (n→) into the then-body keeps the strong arm_inferred case."""
    src = "def f(x):\n    if x:\n        a = 1\n        return a\n    return 0\n"
    log = [
        {"kind": "cmd", "cmd": "pp x", "last_eval": "1",
         "current_frame": {"file": "/m.py", "lineno": 2, "qualname": "f"}},
        {"kind": "cmd", "cmd": "n", "last_eval": "",
         "current_frame": {"file": "/m.py", "lineno": 3, "qualname": "f"}},
    ]
    obs = collect_branch_observations(log, source_resolver=lambda _p: src)
    assert any(o["provenance"]["kind"] == "arm_inferred" and o["arm_taken"] == "then"
               for o in obs)


def test_no_emission_for_non_control_frames():
    src = "def f():\n    a = 1\n    return a\n"
    log = [
        {"kind": "cmd", "cmd": "pp a", "last_eval": "1",
         "current_frame": {"file": "/m.py", "lineno": 2, "qualname": "f"}},
    ]
    obs = collect_branch_observations(log, source_resolver=lambda _p: src)
    assert obs == []


def test_locals_at_stop_harvested_from_prior_pp():
    src = "def f():\n    if x:\n        a = 1\n        return a\n"
    log = [
        {"kind": "cmd", "cmd": "p x", "last_eval": "7",
         "current_frame": {"file": "/m.py", "lineno": 2, "qualname": "f"}},
        {"kind": "cmd", "cmd": "p y", "last_eval": "'hi'",
         "current_frame": {"file": "/m.py", "lineno": 3, "qualname": "f"}},
    ]
    obs = collect_branch_observations(log, source_resolver=lambda _p: src)
    by_line = {o["lineno"]: o for o in obs}
    assert "x" in by_line[3]["locals_at_stop"]
