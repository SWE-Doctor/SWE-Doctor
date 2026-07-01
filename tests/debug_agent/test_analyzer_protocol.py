from debug_agent.analyzer import DebugAnalyzer
from debug_agent.container import ExecResult


class FakeContainer:
    workdir = "/app"

    def exec_bash(self, cmd, timeout=120):
        return ExecResult(0, "ok", "")


def _dispatch_echo(action, container, ctx):
    return f"TOOL_OK({action.name}:{action.payload})"


def _make(llm_responses):
    it = iter(llm_responses)
    real_frame = {"file": "m.py", "lineno": 5, "qualname": "f"}
    return DebugAnalyzer(
        llm=lambda _p: next(it),
        dispatch=_dispatch_echo,
        container=FakeContainer(),
        ctx={
            "issue": "x",
            "workdir": "/app",
            "repro_path": "/app/_repro/t.py",
            "repro_nodeid": "_repro/t.py",
            "preflight_seed": "(ok)",
            "_pdb_session_log": [
                {"kind": "start", "session_id": 1, "ok": True,
                 "initial_frame": real_frame},
                {"kind": "cmd", "session_id": 1, "current_frame": real_frame, "ended": False},
                {"kind": "cmd", "session_id": 1, "current_frame": real_frame, "ended": False},
                {"kind": "cmd", "session_id": 1, "current_frame": real_frame, "ended": False},
            ],
        },
        max_rounds=6,
    )


def test_multi_action_turn_executes_first_and_warns():
    responses = [
        '<reason>look</reason><action name="pytest">-x</action><action name="read">{"path":"/app/a.py"}</action>',
        '<conclusion><root_cause_files>a.py</root_cause_files><root_cause_functions>a.f</root_cause_functions><reasoning>r</reasoning><suggested_fix>s</suggested_fix></conclusion>',
    ]
    a = _make(responses)
    rep = a.run()
    assert rep.root_cause_files == ["a.py"]
    kinds = [t.kind for t in rep.trajectory]
    assert kinds == ["action", "conclusion"]
    first = next(t for t in rep.trajectory if t.kind == "action")
    assert first.action_name == "pytest"
    assert "multiple <action>" in first.tool_output
    assert "TOOL_OK(pytest:-x)" in first.tool_output


def test_action_without_reason_is_rejected():
    responses = [
        '<action name="pytest">-x</action>',
        '<reason>ok now</reason><action name="pytest">-x</action>',
        '<conclusion><root_cause_files>a.py</root_cause_files><root_cause_functions>a.f</root_cause_functions><reasoning>r</reasoning><suggested_fix>s</suggested_fix></conclusion>',
    ]
    a = _make(responses)
    rep = a.run()
    assert [t.kind for t in rep.trajectory] == ["invalid", "action", "conclusion"]


def test_conclusion_without_runtime_evidence_is_rejected():
    responses = [
        '<conclusion><root_cause_files>a.py</root_cause_files><root_cause_functions>a.f</root_cause_functions><reasoning>speculation</reasoning><suggested_fix>s</suggested_fix></conclusion>',
        '<reason>gather evidence first</reason><action name="pytest">-x</action>',
        '<conclusion><root_cause_files>a.py</root_cause_files><root_cause_functions>a.f</root_cause_functions><reasoning>now grounded</reasoning><suggested_fix>s</suggested_fix></conclusion>',
    ]
    a = _make(responses)
    rep = a.run()
    kinds = [t.kind for t in rep.trajectory]
    assert kinds[0] == "invalid"
    assert kinds[-1] == "conclusion"
    assert rep.reasoning == "now grounded"


def test_system_prompt_includes_pdb_checklist_and_new_actions():
    ana = DebugAnalyzer(
        llm=lambda _p: "",
        dispatch=lambda *a, **kw: "",
        container=None,
        ctx={"issue": "x", "repro_nodeid": "n", "workdir": "/app", "repro_path": "/app/t.py"},
    )
    sp = ana._system_prompt()
    assert "pdb_start" in sp
    assert "pdb_cmd" in sp
    assert "pdb_script" in sp
    assert "Abnormal values" in sp
    assert "Branch / loop took the wrong arm" in sp


def test_max_rounds_and_history_defaults_bumped():
    from debug_agent import analyzer as A
    assert A._MAX_HISTORY_CHARS == 160_000
    a = A.DebugAnalyzer(llm=lambda _: "", dispatch=lambda *_a, **_k: "",
                        container=None, ctx={})
    assert a.max_rounds == 40
