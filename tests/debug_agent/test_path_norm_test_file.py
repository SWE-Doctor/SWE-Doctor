from debug_agent.path_norm import is_test_file


def test_is_test_file_repro():
    assert is_test_file("_repro/repro_0.py")
    assert is_test_file("/app/_repro/repro_0.py")


def test_is_test_file_tests_dir():
    assert is_test_file("tests/unit/test_foo.py")
    assert is_test_file("/app/tests/unit/test_foo.py")


def test_is_test_file_test_prefix():
    assert is_test_file("test_foo.py")
    assert is_test_file("/app/some/path/test_x.py")


def test_is_test_file_test_suffix():
    assert is_test_file("foo_test.py")


def test_is_test_file_conftest():
    assert is_test_file("conftest.py")
    assert is_test_file("/app/conftest.py")


def test_production_code_is_not_test_file():
    assert not is_test_file("lib/ansible/utils/vars.py")
    assert not is_test_file("/app/qutebrowser/browser/browsertab.py")
    assert not is_test_file("openlibrary/catalog/add_book/__init__.py")
