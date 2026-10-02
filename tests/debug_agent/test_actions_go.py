"""Go action wiring: _TAG must accept the new `gotest` verb, and dispatch must
route it to run_gotest. The pdb_* actions are REUSED for Go (dlv backend via the
injected _pdb_session_factory) — they are not re-added here."""
from debug_agent import actions
from debug_agent.actions import ActionCall


class _FakeContainer:
    workdir = "/app"
    container_id = "cid"

    def exec_bash(self, cmd, timeout=120, stdin_data=None):
        from debug_agent.container import ExecResult
        # Simulate a failing go test run.
        return ExecResult(1, "--- FAIL: TestRepro (0.00s)\nFAIL\n", "")


def test_tag_regex_accepts_gotest():
    m = actions._TAG.search('<action name="gotest">TestRepro</action>')
    assert m is not None and m.group(1) == "gotest"


def test_gotest_action_routes_to_run_gotest():
    ctx = {"language": "go", "cwd": "/app", "env_prefix": "cd /app &&"}
    out = actions.dispatch(ActionCall(name="gotest", payload="TestRepro"), _FakeContainer(), ctx)
    assert "rc=1" in out
    assert "FAILED TestRepro" in out
    assert "--- OUTPUT ---" in out


def test_gotest_action_no_failures_message():
    class _Pass(_FakeContainer):
        def exec_bash(self, cmd, timeout=120, stdin_data=None):
            from debug_agent.container import ExecResult
            return ExecResult(0, "ok\nPASS\n", "")
    ctx = {"language": "go", "cwd": "/app", "env_prefix": "cd /app &&"}
    out = actions.dispatch(ActionCall(name="gotest", payload=""), _Pass(), ctx)
    assert "rc=0" in out
    assert "(no failed tests parsed)" in out


def test_pdb_start_uses_injected_go_factory():
    """Go reuses pdb_start: when ctx supplies a _pdb_session_factory, dispatch
    must build the session from it (no pdb fallback)."""
    built = {}

    class _FakeSession:
        def __init__(self, run_args):
            built["run_args"] = run_args

        def start(self):
            from debug_agent.pdb_session import StartResult
            return StartResult(ok=True, banner="dlv session", initial_frame=None)

    ctx = {"language": "go",
           "_pdb_session_factory": lambda ra: _FakeSession(ra)}
    out = actions.dispatch(
        ActionCall(name="pdb_start", kwargs={"run_args": ["./server", "-run", "TestX"]}),
        _FakeContainer(), ctx)
    assert built["run_args"] == ["./server", "-run", "TestX"]
    assert ctx["_pdb_session"] is not None
    assert "session started" in out.lower() or "dlv session" in out


def test_tag_regex_accepts_dlv_aliases():
    for verb in ("dlv_start", "dlv_cmd", "dlv_script", "dlv_stop"):
        m = actions._TAG.search(f'<action name="{verb}">{{}}</action>')
        assert m is not None and m.group(1) == verb


def test_parse_action_normalizes_dlv_to_pdb():
    """The Go model emits dlv_* verbs; parse_action maps them onto the canonical
    pdb_* handler so dispatch/gate stay single-named."""
    act = actions.parse_action('<action name="dlv_cmd">{"cmd":"continue"}</action>')
    assert act.name == "pdb_cmd"
    assert act.kwargs == {"cmd": "continue"}
    act2 = actions.parse_action('<action name="dlv_start">{"run_args":["./server"]}</action>')
    assert act2.name == "pdb_start"


def test_go_dispatch_error_hints_use_dlv_verbs():
    """Protocol-error hints echo dlv_* for Go (never pdb_*), matching the prompt."""
    ctx = {"language": "go"}
    no_sess = actions.dispatch(ActionCall(name="pdb_cmd", kwargs={"cmd": "next"}),
                               _FakeContainer(), ctx)
    assert "dlv_start" in no_sess and "pdb" not in no_sess
    bad_start = actions.dispatch(ActionCall(name="pdb_start", kwargs={}),
                                 _FakeContainer(), ctx)
    assert "dlv_start" in bad_start and "pdb" not in bad_start


def test_python_dispatch_error_hints_keep_pdb_verbs():
    """Python path is untouched: hints still name pdb_*."""
    ctx = {"language": "python"}
    out = actions.dispatch(ActionCall(name="pdb_cmd", kwargs={"cmd": "n"}),
                           _FakeContainer(), ctx)
    assert "pdb_start" in out
