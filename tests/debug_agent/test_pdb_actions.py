"""Pure tests for pdb_start/cmd/script/stop dispatch, using a fake
PdbSession injected through ctx. No docker required."""
from dataclasses import dataclass, field
from debug_agent.actions import parse_action, dispatch


@dataclass
class _FakeStart:
    ok: bool = True
    banner: str = "fake banner"
    reason: str = ""
    initial_frame: dict | None = None


@dataclass
class _FakeCmd:
    transcript: str = "ok"
    ended: bool = False
    end_reason: str = ""
    current_frame: dict | None = None
    last_eval: str = ""


@dataclass
class _FakeSession:
    started: bool = False
    log: list = field(default_factory=list)

    def start(self):
        self.started = True
        self.log.append(("start",))
        return _FakeStart(ok=True, initial_frame={"file": "t.py", "lineno": 1, "qualname": "f"})

    def cmd(self, raw):
        self.log.append(("cmd", raw))
        return _FakeCmd(transcript=f"[cmd] {raw}",
                        current_frame={"file": "t.py", "lineno": 5, "qualname": "f"})

    def stop(self):
        self.log.append(("stop",))


def test_parse_pdb_start():
    a = parse_action('<reason>x</reason><action name="pdb_start">{"run_args": ["pytest", "-x"]}</action>')
    assert a is not None
    assert a.name == "pdb_start"
    assert a.kwargs == {"run_args": ["pytest", "-x"]}


def test_parse_pdb_cmd():
    a = parse_action('<reason>x</reason><action name="pdb_cmd">{"cmd": "p x"}</action>')
    assert a is not None
    assert a.name == "pdb_cmd"
    assert a.kwargs == {"cmd": "p x"}


def test_dispatch_pdb_start_creates_session_and_logs():
    fake = _FakeSession()
    ctx = {"_pdb_session_factory": lambda run_args: fake,
           "_pdb_session_log": []}
    a = parse_action('<action name="pdb_start">{"run_args":["/work/t.py"]}</action>')
    out = dispatch(a, container=None, ctx=ctx)
    assert "started" in out.lower() or "ok" in out.lower()
    assert ctx["_pdb_session"] is fake
    assert fake.started is True
    assert ctx["_pdb_session_log"][-1]["kind"] == "start"


def test_dispatch_pdb_cmd_uses_existing_session_and_records_turn():
    fake = _FakeSession()
    fake.started = True
    ctx = {"_pdb_session": fake, "_pdb_session_log": []}
    a = parse_action('<action name="pdb_cmd">{"cmd":"b t.py:5"}</action>')
    out = dispatch(a, container=None, ctx=ctx)
    assert "[cmd] b t.py:5" in out
    turn = ctx["_pdb_session_log"][-1]
    assert turn["kind"] == "cmd"
    assert turn["cmd"] == "b t.py:5"
    assert turn["current_frame"]["lineno"] == 5


def test_dispatch_pdb_script_runs_each_line_and_concatenates():
    fake = _FakeSession()
    fake.started = True
    ctx = {"_pdb_session": fake, "_pdb_session_log": []}
    a = parse_action(
        '<action name="pdb_script">{"commands":["b t.py:5","c","p x"]}</action>'
    )
    out = dispatch(a, container=None, ctx=ctx)
    assert out.count("[cmd]") == 3
    assert sum(1 for r in ctx["_pdb_session_log"] if r["kind"] == "cmd") == 3


def test_dispatch_pdb_start_replaces_existing_session():
    old = _FakeSession()
    old.started = True
    new = _FakeSession()
    ctx = {"_pdb_session": old,
           "_pdb_session_factory": lambda run_args: new,
           "_pdb_session_log": []}
    a = parse_action('<action name="pdb_start">{"run_args":["/work/t.py"]}</action>')
    dispatch(a, container=None, ctx=ctx)
    assert ("stop",) in old.log
    assert ctx["_pdb_session"] is new


def test_dispatch_pdb_stop_is_safe_when_no_session():
    ctx = {"_pdb_session_log": []}
    a = parse_action('<action name="pdb_stop">{}</action>')
    out = dispatch(a, container=None, ctx=ctx)
    assert "no session" in out.lower() or "stopped" in out.lower()
