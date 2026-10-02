"""execution_augmented_repair must thread language + test_filename through to
run_test/critique so go repros run as `*_test.go` in the right package, while
python defaults stay byte-identical."""
import reproduction_test_agent.repair as rp


def _fake_critique_pass(**kw):
    _fake_critique_pass.seen = kw
    return {"fails_for_right_reason": True, "functions_to_read": []}


def test_repair_threads_go_language_and_filename(monkeypatch):
    seen = {}

    def fake_run_test(test_code, env, cwd, timeout, test_filename, language):
        seen["test_filename"] = test_filename
        seen["language"] = language
        return {"passed": False, "returncode": 1, "output": "--- FAIL: TestX",
                "error_type": "TestFailure", "error_message": "x"}

    def fake_critique(**kw):
        seen["critique_language"] = kw.get("language")
        return {"fails_for_right_reason": True, "functions_to_read": []}

    monkeypatch.setattr(rp, "run_test", fake_run_test)
    monkeypatch.setattr(rp, "critique_test", fake_critique)
    rp.execution_augmented_repair(
        "issue", "package server\nfunc TestX(t *testing.T){}",
        env=None, cwd="/app", model="m", max_attempts=1,
        language="go", test_filename="server/zzz_repro_test.go")
    assert seen["test_filename"] == "server/zzz_repro_test.go"
    assert seen["language"] == "go"
    assert seen["critique_language"] == "go"


def test_repair_python_defaults_unchanged(monkeypatch):
    seen = {}

    def fake_run_test(test_code, env, cwd, timeout, test_filename, language):
        seen["test_filename"] = test_filename
        seen["language"] = language
        return {"passed": False, "returncode": 1, "output": "x",
                "error_type": "", "error_message": ""}

    monkeypatch.setattr(rp, "run_test", fake_run_test)
    monkeypatch.setattr(rp, "critique_test",
                        lambda **kw: {"fails_for_right_reason": True, "functions_to_read": []})
    rp.execution_augmented_repair("issue", "def test_x(): pass",
                                  env=None, cwd="/app", model="m", max_attempts=1)
    assert seen["test_filename"] == "repro_test.py"
    assert seen["language"] == "python"


def test_go_error_symbols_extracts_undefined_and_methods():
    """The go build-error symbol extractor feeds _gather_extra_context so the
    repair sees real definitions (the LLM critic leaves functions_to_read empty
    for go compile failures)."""
    log = (
        "internal/config/zzz_repro_test.go:18:26: jwtInfo.RequiresDatabase undefined "
        "(type AuthenticationMethodInfo has no field or method RequiresDatabase)\n"
        "lib/backend/zzz_repro_test.go:15:15: undefined: wal2jsonMessage\n"
        "UserV2 does not implement types.User (missing method AddRole)\n"
    )
    syms = rp._go_error_symbols(log)
    assert "RequiresDatabase" in syms          # has-no-field-or-method
    assert "AuthenticationMethodInfo" in syms  # owning type
    assert "wal2jsonMessage" in syms           # undefined: X
    assert "AddRole" in syms                    # missing method M
    # package qualifiers dropped so the func/type grep can find the def
    assert "User" in syms and "types.User" not in syms


def test_go_error_symbols_empty_on_clean_log():
    assert rp._go_error_symbols("ok\nPASS\n") == []
