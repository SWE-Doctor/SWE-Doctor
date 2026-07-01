from debug_agent.path_norm import is_framework_file, is_noise_file, is_test_file


def test_pytest_internal_is_framework():
    assert is_framework_file("/usr/local/lib/python3.11/site-packages/_pytest/python.py")
    assert is_framework_file("_pytest/python.py")  # already-normalized


def test_pluggy_is_framework():
    assert is_framework_file("/usr/local/lib/python3.11/site-packages/pluggy/_callers.py")
    assert is_framework_file("pluggy/_callers.py")


def test_dist_packages_project_code_is_NOT_framework():
    """Critical: ansible installed via pip lives in dist-packages but is
    project code, not framework. Must not be filtered."""
    assert not is_framework_file(
        "/usr/local/lib/python3.9/dist-packages/ansible/plugins/lookup/password.py"
    )
    assert not is_framework_file(
        "/usr/local/lib/python3.11/dist-packages/openlibrary/catalog/add_book.py"
    )


def test_production_is_not_framework():
    assert not is_framework_file("lib/ansible/parsing/dataloader.py")
    assert not is_framework_file("/app/qutebrowser/browser/commands.py")


def test_is_noise_combines_both():
    assert is_noise_file("_pytest/python.py")
    assert is_noise_file("_repro/repro_0.py")
    assert not is_noise_file("lib/ansible/parsing/dataloader.py")


def test_is_test_file_unchanged():
    """sanity: existing test-file detection still works"""
    assert is_test_file("_repro/repro_0.py")
    assert is_test_file("conftest.py")
    assert not is_test_file("lib/ansible/parsing/dataloader.py")
    # framework files are NOT test files (they're framework files)
    assert not is_test_file("_pytest/python.py")
