from debug_agent.actions import ActionCall, parse_action


def test_parse_action_extracts_name_and_payload():
    resp = '<action name="bash">ls -la /work</action>'
    a = parse_action(resp)
    assert a == ActionCall(name="bash", payload="ls -la /work", kwargs={})


def test_parse_action_reads_json_payload_for_probe():
    resp = (
        '<action name="probe">'
        '{"file": "/work/m.py", "before_line": 3, "expr": "y"}'
        '</action>'
    )
    a = parse_action(resp)
    assert a.name == "probe"
    assert a.kwargs == {"file": "/work/m.py", "before_line": 3, "expr": "y"}


def test_parse_action_reads_json_payload_for_read():
    resp = '<action name="read">{"path": "/work/m.py", "start": 5, "n": 10}</action>'
    a = parse_action(resp)
    assert a.name == "read"
    assert a.kwargs == {"path": "/work/m.py", "start": 5, "n": 10}


def test_parse_action_returns_none_when_tag_absent():
    assert parse_action("just thinking out loud") is None


def test_parse_action_picks_first_action_when_multiple():
    resp = '<action name="bash">one</action> blah <action name="bash">two</action>'
    assert parse_action(resp).payload == "one"


def test_parse_action_structured_fallback_when_invalid_json():
    """Probe with non-JSON body: fall through to payload so dispatcher can error cleanly."""
    resp = '<action name="probe">not-json-at-all</action>'
    a = parse_action(resp)
    assert a.name == "probe"
    assert a.payload == "not-json-at-all"
    assert a.kwargs == {}
