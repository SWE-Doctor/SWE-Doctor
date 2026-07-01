from debug_agent.analyzer import DebugAnalyzer, DebugReport


def _mock_dispatch(action, _container, _ctx):
    return f"[ran {action.name}] {action.payload or action.kwargs}"


def test_analyzer_emits_conclusion_and_exits():
    scripted = iter([
        '<reason>inspect</reason><action name="bash">ls /app</action>',
        '<reason>repro</reason><action name="pytest">-x tests/test_repro_a0.py</action>',
        ('<conclusion>\n'
         '<root_cause_files>pkg/mod.py</root_cause_files>\n'
         '<root_cause_functions>pkg.mod.do_thing</root_cause_functions>\n'
         '<reasoning>uninit state</reasoning>\n'
         '<suggested_fix>initialize in __init__</suggested_fix>\n'
         '</conclusion>'),
    ])
    real_frame = {"file": "m.py", "lineno": 5, "qualname": "f"}
    ana = DebugAnalyzer(
        llm=lambda _p: next(scripted),
        dispatch=_mock_dispatch,
        container=None,
        ctx={
            "issue": "x", "repro_nodeid": "tests/test_repro_a0.py::test_case",
            "_pdb_session_log": [
                {"kind": "start", "session_id": 1, "ok": True,
                 "initial_frame": real_frame},
                {"kind": "cmd", "session_id": 1, "current_frame": real_frame, "ended": False},
                {"kind": "cmd", "session_id": 1, "current_frame": real_frame, "ended": False},
                {"kind": "cmd", "session_id": 1, "current_frame": real_frame, "ended": False},
            ],
        },
        max_rounds=10,
    )
    report = ana.run()
    assert isinstance(report, DebugReport)
    assert report.root_cause_files == ["pkg/mod.py"]
    assert report.root_cause_functions == ["pkg.mod.do_thing"]
    assert report.reasoning == "uninit state"
    assert report.suggested_fix == "initialize in __init__"
    assert len(report.transcript) == 2
    assert report.transcript[0][0] == "bash"
    assert report.transcript[1][0] == "pytest"
    assert report.timed_out is False

    assert [t.kind for t in report.trajectory] == ["action", "action", "conclusion"]
    assert "<action name=\"bash\"" in report.trajectory[0].llm_response
    assert report.trajectory[2].llm_response.startswith("<conclusion>")
    assert report.trajectory[0].tool_output.startswith("[ran bash]")


def test_analyzer_hits_max_rounds():
    ana = DebugAnalyzer(
        llm=lambda _p: '<reason>loop</reason><action name="bash">echo keep going</action>',
        dispatch=_mock_dispatch,
        container=None,
        ctx={"issue": "x", "repro_nodeid": "n"},
        max_rounds=3,
    )
    report = ana.run()
    assert report.timed_out is True
    assert len(report.transcript) == 3


def test_analyzer_feeds_back_reminder_when_no_tag():
    responses = iter([
        'thinking out loud…',
        '<reason>ok</reason><action name="pytest">-x</action>',
        ('<conclusion>'
         '<root_cause_files></root_cause_files>'
         '<root_cause_functions></root_cause_functions>'
         '<reasoning>r</reasoning><suggested_fix>s</suggested_fix>'
         '</conclusion>'),
    ])
    real_frame = {"file": "m.py", "lineno": 5, "qualname": "f"}
    ana = DebugAnalyzer(
        llm=lambda _p: next(responses),
        dispatch=_mock_dispatch,
        container=None,
        ctx={
            "issue": "x", "repro_nodeid": "n",
            "_pdb_session_log": [
                {"kind": "start", "session_id": 1, "ok": True,
                 "initial_frame": real_frame},
                {"kind": "cmd", "session_id": 1, "current_frame": real_frame, "ended": False},
                {"kind": "cmd", "session_id": 1, "current_frame": real_frame, "ended": False},
                {"kind": "cmd", "session_id": 1, "current_frame": real_frame, "ended": False},
            ],
        },
        max_rounds=10,
    )
    report = ana.run()
    assert len(report.transcript) == 1
    assert report.transcript[0][0] == "pytest"
    assert report.timed_out is False

    kinds = [t.kind for t in report.trajectory]
    assert kinds == ["invalid", "action", "conclusion"]
    assert report.trajectory[0].llm_response == "thinking out loud…"


def test_analyzer_rejects_direct_conclusion_without_evidence():
    """New contract: conclusion before any runtime evidence is rejected."""
    resp = (
        "<conclusion><root_cause_files>a.py</root_cause_files>"
        "<root_cause_functions></root_cause_functions>"
        "<reasoning>speculation</reasoning><suggested_fix>s</suggested_fix></conclusion>"
    )
    responses = iter([resp, resp, resp])  # keeps retrying; exhaust max_rounds
    ana = DebugAnalyzer(
        llm=lambda _p: next(responses),
        dispatch=lambda a, c, ctx: "unused",
        container=None,
        ctx={"issue": "x", "repro_nodeid": "n"},
        max_rounds=3,
    )
    report = ana.run()
    assert report.transcript == []
    assert all(t.kind == "invalid" for t in report.trajectory)
    assert report.timed_out is True


def test_analyzer_terminates_on_history_budget():
    big = "x" * 40_000
    ana = DebugAnalyzer(
        llm=lambda _p: f'<reason>r</reason><action name="bash">{big}</action>',
        dispatch=_mock_dispatch,
        container=None,
        ctx={"issue": "x", "repro_nodeid": "n"},
        max_rounds=100,
        max_history_chars=50_000,
    )
    report = ana.run()
    assert report.timed_out is True
    assert len(report.transcript) < 100


def test_analyzer_rejects_conclusion_without_pdb_session():
    """A conclusion right after pytest is now rejected — the gate
    requires a real pdb stop + 3 commands."""
    scripted = iter([
        '<reason>repro</reason><action name="pytest">-x t</action>',
        ('<conclusion>'
         '<root_cause_files>m.py</root_cause_files>'
         '<root_cause_functions>m.f</root_cause_functions>'
         '<reasoning>r</reasoning><suggested_fix>s</suggested_fix>'
         '</conclusion>'),
        '<reason>retry</reason><action name="bash">echo retry</action>',
        '<reason>retry</reason><action name="bash">echo retry</action>',
        '<reason>retry</reason><action name="bash">echo retry</action>',
    ])
    ana = DebugAnalyzer(
        llm=lambda _p: next(scripted, '<reason>x</reason><action name="bash">noop</action>'),
        dispatch=_mock_dispatch,
        container=None,
        ctx={"issue": "x", "repro_nodeid": "n"},
        max_rounds=5,
    )
    report = ana.run()
    assert report.timed_out is True
    kinds = [t.kind for t in report.trajectory]
    assert "invalid" in kinds
