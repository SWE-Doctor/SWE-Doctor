"""executor.run_test must grow a `language` param. language='go' writes the
test (mkdir-ing its package dir first), runs `go test` scoped to the test func
with GOFLAGS=-mod=mod, and derives passed/error via langpack.parse_test_output.
The python path must stay byte-identical (no mkdir, pytest cmd, passed=rc==0)."""
from reproduction_test_agent import executor


class FakeEnv:
    def __init__(self, run_output="", run_rc=0):
        self.commands = []
        self._run_output = run_output
        self._run_rc = run_rc

    def execute(self, action, cwd="", *, timeout=None):
        cmd = action["command"]
        self.commands.append(cmd)
        if "go test" in cmd:
            return {"output": self._run_output, "returncode": self._run_rc}
        return {"output": "", "returncode": 0}


def test_run_test_go_builds_command_and_mkdir():
    fail_out = (
        "=== RUN   TestEval\n--- FAIL: TestEval (0.00s)\n"
        "    eval_test.go:10: boom\nFAIL\nFAIL\tgithub.com/x/y\t0.01s\n"
    )
    env = FakeEnv(run_output=fail_out, run_rc=1)
    res = executor.run_test(
        'package server\n\nfunc TestEval(t *testing.T) { t.Fatalf("boom") }\n',
        env, cwd="/app", language="go", test_filename="server/repro_test.go",
    )
    joined = "\n".join(env.commands)
    assert "go test" in joined
    assert "-run TestEval" in joined          # scoped to the generated test func
    assert "GOFLAGS=-mod=mod" in joined       # let go1.24 patch go.sum
    assert any("mkdir -p" in c for c in env.commands)   # package dir created first
    assert res["passed"] is False
    assert res["returncode"] == 1
    assert res["error_type"]                  # non-empty


def test_run_test_go_passes_when_rc0_and_no_error():
    env = FakeEnv(run_output="ok\nPASS\nok\tgithub.com/x/y\t0.01s\n", run_rc=0)
    res = executor.run_test(
        "package server\nfunc TestEval(t *testing.T) {}\n",
        env, cwd="/app", language="go", test_filename="server/repro_test.go",
    )
    assert res["passed"] is True
    assert res["error_type"] == ""


def test_run_test_python_unchanged():
    class PyEnv:
        def __init__(self):
            self.commands = []

        def execute(self, action, cwd="", *, timeout=None):
            self.commands.append(action["command"])
            if "pytest" in action["command"]:
                return {"output": "1 passed", "returncode": 0}
            return {"output": "", "returncode": 0}

    env = PyEnv()
    res = executor.run_test("def test_x():\n    assert True\n", env, cwd="/app")
    joined = "\n".join(env.commands)
    assert "python -m pytest" in joined
    assert "go test" not in joined
    assert "GOFLAGS" not in joined
    assert not any("mkdir" in c for c in env.commands)  # python sequence unchanged
    assert res["passed"] is True
