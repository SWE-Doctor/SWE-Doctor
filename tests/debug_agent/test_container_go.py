"""Go container support: launch() must request ptrace (dlv needs it), and
ensure_dlv() must install Delve only when it's missing. subprocess.run is
monkeypatched so no real docker is needed."""
from debug_agent import container as C
from debug_agent.container import Container, ExecResult


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_launch_adds_ptrace_caps(monkeypatch):
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        return _FakeProc(0, "cid\n", "")

    monkeypatch.setattr(C.subprocess, "run", fake_run)
    Container.launch(image="golang:1.24", workdir="/app")
    run_args = calls[0]
    assert "--cap-add" in run_args
    assert "SYS_PTRACE" in run_args
    assert "--security-opt" in run_args
    assert "seccomp=unconfined" in run_args


def test_ensure_dlv_copies_precompiled_when_missing(monkeypatch):
    c = Container("cid", "/app")
    seen = []
    cp_calls = []

    def fake_exec(cmd, timeout=120, stdin_data=None):
        seen.append(cmd)
        if cmd.startswith("test -x /go/bin/dlv"):
            return ExecResult(0, "N\n", "")            # not present
        if cmd.startswith("/go/bin/dlv version"):
            return ExecResult(0, "Delve Debugger\n", "")  # copy verified
        return ExecResult(0, "", "")

    def fake_run(args, **kw):
        cp_calls.append(args)
        return _FakeProc(0, "", "")                    # docker cp succeeds

    monkeypatch.setattr(c, "exec_bash", fake_exec)
    monkeypatch.setattr(C.subprocess, "run", fake_run)
    c.ensure_dlv()
    assert any(a[:2] == ["docker", "cp"] for a in cp_calls)
    assert not any("go install" in s for s in seen)   # no compile when copy works


def test_ensure_dlv_falls_back_to_install_when_copy_fails(monkeypatch):
    c = Container("cid", "/app")
    seen = []

    def fake_exec(cmd, timeout=120, stdin_data=None):
        seen.append(cmd)
        if cmd.startswith("test -x /go/bin/dlv"):
            return ExecResult(0, "N\n", "")   # not present
        return ExecResult(0, "", "")

    def fake_run(args, **kw):
        return _FakeProc(1, "", "no such container")   # docker cp fails

    monkeypatch.setattr(c, "exec_bash", fake_exec)
    monkeypatch.setattr(C.subprocess, "run", fake_run)
    c.ensure_dlv()
    assert any("go install" in s and "delve" in s for s in seen)


def test_ensure_dlv_skips_when_present(monkeypatch):
    c = Container("cid", "/app")
    seen = []

    def fake_exec(cmd, timeout=120, stdin_data=None):
        seen.append(cmd)
        if cmd.startswith("test -x /go/bin/dlv"):
            return ExecResult(0, "Y\n", "")   # already present
        return ExecResult(0, "", "")

    monkeypatch.setattr(c, "exec_bash", fake_exec)
    c.ensure_dlv()
    assert not any("go install" in s for s in seen)
