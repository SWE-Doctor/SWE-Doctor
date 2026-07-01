from debug_agent.anomaly_snapshot import collect_anomalies


def _cmd(sid, file, lineno, cmd, last_eval, idx):
    return {"kind": "cmd", "session_id": sid,
            "cmd": cmd, "last_eval": last_eval,
            "current_frame": {"file": file, "lineno": lineno, "qualname": "f"},
            "ended": False, "_idx": idx}


def test_none_and_empty_and_zero_tags():
    log = [
        _cmd(1, "/m.py", 5, "p x", "None", 0),
        _cmd(1, "/m.py", 5, "p y", "[]", 1),
        _cmd(1, "/m.py", 6, "p z", "0", 2),
    ]
    out = collect_anomalies(log)
    by = {a["expr"]: a for a in out}
    assert "none" in by["x"]["tags"]
    assert "empty" in by["y"]["tags"]
    assert "zero" in by["z"]["tags"]


def test_str_bytes_type_drift_tagged():
    log = [
        _cmd(1, "/m.py", 5, "p x", "'hello'", 0),
        _cmd(1, "/m.py", 6, "p x", "b'hello'", 1),
    ]
    out = collect_anomalies(log)
    drift = [a for a in out if "type_str_bytes" in a["tags"]]
    assert len(drift) >= 1


def test_large_int_tag():
    log = [_cmd(1, "/m.py", 5, "p n", str(2**33), 0)]
    out = collect_anomalies(log)
    assert "large_int" in out[0]["tags"]


def test_value_repr_truncated_to_200():
    long = "'" + "X" * 1000 + "'"
    log = [_cmd(1, "/m.py", 5, "p s", long, 0)]
    out = collect_anomalies(log)
    assert len(out[0]["value_repr"]) <= 200


def test_non_p_commands_ignored():
    log = [_cmd(1, "/m.py", 5, "n", "", 0),
           _cmd(1, "/m.py", 5, "b /m.py:5", "", 1)]
    out = collect_anomalies(log)
    assert out == []


def test_bundled_pp_atomizes_into_separate_entries():
    log = [_cmd(1, "/m.py", 5,
                "pp constructed_file, inventory_file, loader",
                "('a.py', 'b.ini', <Loader at 0x1>)", 0)]
    out = collect_anomalies(log)
    assert [a["expr"] for a in out] == ["constructed_file", "inventory_file", "loader"]
    assert [a["value_repr"] for a in out] == ["'a.py'", "'b.ini'", "<Loader at 0x1>"]
    assert all(a["evidence_refs"] == [0] for a in out)


def test_atomization_skipped_when_repr_not_tuple_shaped():
    log = [_cmd(1, "/m.py", 5, "pp x, y", "[1, 2]", 0)]
    out = collect_anomalies(log)
    assert len(out) == 1
    assert out[0]["expr"] == "x, y"


def test_atomization_skipped_when_segment_count_mismatches():
    log = [_cmd(1, "/m.py", 5, "pp x, y, z", "(1, 2)", 0)]
    out = collect_anomalies(log)
    assert len(out) == 1
    assert out[0]["expr"] == "x, y, z"


def test_single_value_with_internal_comma_not_split():
    log = [_cmd(1, "/m.py", 5, "p t", "(1, 2, 3)", 0)]
    out = collect_anomalies(log)
    assert len(out) == 1
    assert out[0]["expr"] == "t"
