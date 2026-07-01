"""GoDlvSession pure-logic tests: it must reuse the pdb_session StartResult/
CmdResult contract, parse dlv's REPL frame lines, and build a correct
`dlv test` docker-exec command. Frame formats are taken verbatim from a
real dlv spike on the flipt go1.24.3 image."""
from debug_agent.go_session import (
    GoDlvSession,
    _parse_frame_from_transcript,
    _normalize_go_test_args,
)


class _FakeContainer:
    def __init__(self, cid="cid", workdir="/app"):
        self.container_id = cid
        self.workdir = workdir


def test_go_session_reuses_pdb_contract_dataclasses():
    # Must import-reuse, NOT redefine, the debugging contract.
    import debug_agent.go_session as gs
    import debug_agent.pdb_session as ps
    assert gs.StartResult is ps.StartResult
    assert gs.CmdResult is ps.CmdResult


def test_parse_dlv_breakpoint_stop_frame():
    line = ("> [Breakpoint 1] github.com/markphelps/flipt/server.(*Server).Evaluate() "
            "./server/evaluator.go:20 (hits goroutine(34):1 total:1) (PC: 0xb8cdf6)")
    f = _parse_frame_from_transcript(line)
    assert f["file"] == "server/evaluator.go"   # normalized: leading ./ stripped
    assert f["lineno"] == 20
    assert "Evaluate" in f["qualname"]


def test_parse_dlv_step_frame_without_breakpoint_tag():
    line = ("> github.com/markphelps/flipt/server.(*Server).Evaluate() "
            "./server/evaluator.go:21 (PC: 0xb8ce2c)")
    f = _parse_frame_from_transcript(line)
    assert f["lineno"] == 21
    assert f["file"] == "server/evaluator.go"


def test_parse_dlv_frame_returns_none_on_banner():
    assert _parse_frame_from_transcript("Type 'help' for list of commands.") is None


def test_parse_dlv_frame_takes_last_match():
    text = ("> a.B() ./a.go:1 (PC: 0x1)\n"
            "> c.D() ./c.go:9 (PC: 0x2)\n")
    f = _parse_frame_from_transcript(text)
    assert f["file"] == "c.go" and f["lineno"] == 9


def test_normalize_go_test_args_extracts_pkg_and_run():
    assert _normalize_go_test_args(["go", "test", "./server", "-run", "TestX"]) == ("./server", "TestX")
    assert _normalize_go_test_args(["./server", "-run", "TestX"]) == ("./server", "TestX")
    assert _normalize_go_test_args(["-run", "TestX", "./server"]) == ("./server", "TestX")
    assert _normalize_go_test_args(["./server"]) == ("./server", "")
    assert _normalize_go_test_args(["-run", "^TestX$", "./server"]) == ("./server", "TestX")


def test_build_dlv_cmd_has_all_key_flags():
    s = GoDlvSession(_FakeContainer("cid123"),
                     run_args=["./server", "-run", "TestEvaluate_FlagDisabled"])
    cmd = s._build_dlv_cmd()
    assert cmd[:3] == ["docker", "exec", "-i"]
    assert "cid123" in cmd
    joined = " ".join(cmd)
    assert "dlv" in joined and " test " in f" {joined} "
    assert "./server" in joined
    assert "--allow-non-terminal-interactive=true" in joined
    assert "-test.run" in joined and "TestEvaluate_FlagDisabled" in joined
    assert "NO_COLOR=1" in joined        # de-color dlv REPL output
    assert "-mod=mod" in joined          # GOFLAGS so go1.24 backfills go.sum
    assert "/usr/local/go/bin" in joined  # PATH fix (login shell drops go)
