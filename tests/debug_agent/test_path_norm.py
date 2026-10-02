from debug_agent.path_norm import normalize_repo_path, paths_match


def test_strip_app_prefix():
    assert normalize_repo_path("/app/lib/ansible/utils/vars.py") == "lib/ansible/utils/vars.py"


def test_strip_dist_packages_py39():
    assert normalize_repo_path(
        "/usr/local/lib/python3.9/dist-packages/ansible/plugins/lookup/password.py"
    ) == "ansible/plugins/lookup/password.py"


def test_strip_dist_packages_py311():
    assert normalize_repo_path(
        "/usr/local/lib/python3.11/dist-packages/ansible/cli/galaxy.py"
    ) == "ansible/cli/galaxy.py"


def test_strip_site_packages():
    assert normalize_repo_path(
        "/usr/lib/python3.10/site-packages/foo/bar.py"
    ) == "foo/bar.py"


def test_collapse_doubled_pkg_dir():
    assert normalize_repo_path(
        "/app/openlibrary/openlibrary/catalog/add_book.py"
    ) == "openlibrary/catalog/add_book.py"


def test_already_relative():
    assert normalize_repo_path("ansible/utils/vars.py") == "ansible/utils/vars.py"


def test_empty_returns_empty():
    assert normalize_repo_path("") == ""


def test_paths_match_collapses_lib_ansible_to_ansible():
    assert paths_match("lib/ansible/utils/vars.py", "ansible/utils/vars.py")
    assert paths_match("/app/lib/ansible/utils/vars.py", "lib/ansible/utils/vars.py")


def test_paths_match_negative():
    assert not paths_match("lib/ansible/utils/vars.py", "lib/ansible/utils/encrypt.py")
