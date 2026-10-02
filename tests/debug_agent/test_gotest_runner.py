"""gotest_runner pure-logic tests: it reuses go_runner.extract_failed_tests to
pull failed test names from `go test` output, and builds a PATH/GOFLAGS-correct
`go test ./... -run <name> -v` command. No docker — a fake container records
the command and returns canned output."""
from debug_agent.gotest_runner import GoTestResult, parse_go_failures, run_gotest
from debug_agent.pytest_runner import FailedTest


_GO_FAIL_OUTPUT = """\
=== RUN   TestEvaluate_FlagDisabled
    evaluator_test.go:42: expected enabled=false, got true
--- FAIL: TestEvaluate_FlagDisabled (0.00s)
=== RUN   TestEvaluate_Ok
--- PASS: TestEvaluate_Ok (0.00s)
FAIL
FAIL    github.com/markphelps/flipt/server      0.012s
"""


class _FakeContainer:
    def __init__(self, rc=1, out="", err="", workdir="/app"):
        self.container_id = "cid"
        self.workdir = workdir
        self._rc, self._out, self._err = rc, out, err
        self.last_cmd = None
        self.last_timeout = None

    def exec_bash(self, cmd, timeout=120, stdin_data=None):
        from debug_agent.container import ExecResult
        self.last_cmd = cmd
        self.last_timeout = timeout
        return ExecResult(self._rc, self._out, self._err)


def test_parse_go_failures_extracts_failed_test_names():
    fails = parse_go_failures(_GO_FAIL_OUTPUT)
    assert any(f.nodeid == "TestEvaluate_FlagDisabled" for f in fails)
    assert all(isinstance(f, FailedTest) for f in fails)


def test_parse_go_failures_empty_on_clean_pass():
    assert parse_go_failures("--- PASS: TestOk (0.00s)\nPASS\nok  pkg 0.01s\n") == []


def test_run_gotest_builds_path_goflags_run_command():
    c = _FakeContainer(rc=1, out=_GO_FAIL_OUTPUT)
    res = run_gotest(c, args="TestEvaluate_FlagDisabled", cwd="/app")
    assert isinstance(res, GoTestResult)
    assert res.returncode == 1
    cmd = c.last_cmd
    assert "go test ./..." in cmd
    assert "-run 'TestEvaluate_FlagDisabled'" in cmd or "-run TestEvaluate_FlagDisabled" in cmd
    assert "GOFLAGS=-mod=mod" in cmd
    assert "/usr/local/go/bin:/go/bin" in cmd
    assert "cd /app" in cmd
    assert any(f.nodeid == "TestEvaluate_FlagDisabled" for f in res.failed_tests)


def test_run_gotest_no_args_runs_all_packages():
    c = _FakeContainer(rc=0, out="ok\n")
    run_gotest(c, args="", cwd="/app")
    assert "go test ./..." in c.last_cmd
    assert "-run" not in c.last_cmd


def test_run_gotest_uses_env_prefix_when_given():
    c = _FakeContainer(rc=0, out="ok\n")
    run_gotest(c, args="", cwd="/app", env_prefix="cd /app &&")
    # env_prefix supplies the cd; we must not double it.
    assert c.last_cmd.startswith("cd /app && export PATH=")
