"""Go test files (*_test.go) must be recognized as test files so they are
never reported as bug candidates (bugs live in production code)."""
from debug_agent.path_norm import is_test_file


def test_is_test_file_recognizes_go_test_suffix():
    assert is_test_file("server/evaluator_test.go") is True
    assert is_test_file("internal/store/store_test.go") is True


def test_is_test_file_go_production_code_is_not_test():
    assert is_test_file("server/evaluator.go") is False
    assert is_test_file("errors/errors.go") is False
