"""Go analyzer wiring: language=='go' selects the Go/dlv prompt triple, the
action regexes accept `gotest`, and the gate-diagnostic talks dlv (not pdb).
Python defaults must stay exactly as before."""
from debug_agent import analyzer as A
from debug_agent.analyzer import DebugAnalyzer, _diagnose_gate


def _ana(ctx):
    return DebugAnalyzer(llm=lambda _p: "", dispatch=lambda *a, **k: "",
                         container=None, ctx=ctx)


def test_system_prompt_go_uses_dlv_prompt_triple():
    ana = _ana({"language": "go", "issue": "x", "workdir": "/app",
                "repro_path": "/app/_repro/x_test.go", "repro_nodeid": "_repro/x_test.go"})
    sp = ana._system_prompt()
    assert "Go debug agent" in sp
    assert "dlv" in sp.lower()
    assert "break " in sp                 # dlv break command from the tutorial
    assert "nil pointer deref" in sp.lower() or "nil pointer" in sp.lower()
    assert "stepout" in sp                # dlv-specific verb, not pdb
    # Go prompt speaks pure dlv: the model emits dlv_* verbs and sees ZERO "pdb".
    assert "dlv_start" in sp and "dlv_cmd" in sp
    assert "pdb" not in sp.lower()


def test_system_prompt_python_default_unchanged():
    ana = _ana({"issue": "x", "workdir": "/app", "repro_path": "/app/t.py",
                "repro_nodeid": "n"})
    sp = ana._system_prompt()
    assert "Python debug agent" in sp
    assert "Abnormal values" in sp
    assert "dlv" not in sp.lower()


def test_action_regexes_accept_gotest():
    assert A._ACTION_OPEN.search('<action name="gotest">TestX</action>') is not None
    assert A._ALL_ACTIONS.search('<action name="gotest">TestX</action>') is not None
    # pdb_* still matched (Python path); dlv_* aliases also matched (Go path).
    assert A._ALL_ACTIONS.search('<action name="pdb_start">{}</action>') is not None
    assert A._ALL_ACTIONS.search('<action name="dlv_start">{}</action>') is not None
    assert A._ACTION_OPEN.search('<action name="dlv_cmd">{}</action>') is not None


def test_diagnose_gate_go_mentions_gotest_and_dlv():
    msg = _diagnose_gate([], [], language="go")
    assert "gotest" in msg
    assert "dlv" in msg
    assert "pytest" not in msg
    assert "pdb" not in msg     # Go gate hints speak pure dlv


def test_diagnose_gate_python_default_mentions_pytest():
    msg = _diagnose_gate([], [])
    assert "pytest" in msg
    assert "gotest" not in msg
