"""Go gating: a dlv stop frame in the go stdlib/deps (e.g. testing.tRunner at
/usr/local/go/src/testing/testing.go) must NOT count as a real frame, and a
`gotest` action must count as test evidence — mirroring pytest for python."""
from debug_agent.gating import _is_real_frame, _had_pytest_evidence


def test_is_real_frame_rejects_go_stdlib_deps_vendor():
    assert _is_real_frame({"file": "/usr/local/go/src/testing/testing.go", "lineno": 1792}) is False
    assert _is_real_frame({"file": "/go/pkg/mod/github.com/x/y@v1.0.0/z.go", "lineno": 5}) is False
    assert _is_real_frame({"file": "vendor/github.com/x/y.go", "lineno": 5}) is False
    assert _is_real_frame({"file": "app/vendor/github.com/x/y.go", "lineno": 5}) is False


def test_is_real_frame_accepts_go_app_code():
    assert _is_real_frame({"file": "server/evaluator.go", "lineno": 20}) is True
    assert _is_real_frame({"file": "/app/internal/store/store.go", "lineno": 7}) is True


def test_had_pytest_evidence_accepts_gotest_action():
    history = [{"action_name": "gotest", "tool_output_ok": True}]
    assert _had_pytest_evidence(history, []) is True


def test_had_pytest_evidence_accepts_go_dlv_session():
    # pdb_start with go test run_args (factory builds `dlv test ./pkg`) is evidence.
    pdb_log = [{"kind": "start", "ok": True, "run_args": ["./server", "-run", "TestX"]}]
    assert _had_pytest_evidence([], pdb_log) is True


def test_had_pytest_evidence_still_false_without_any_test():
    assert _had_pytest_evidence([{"action_name": "bash", "tool_output_ok": True}], []) is False
