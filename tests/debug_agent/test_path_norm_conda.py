"""Regression test for SWE-bench *Verified* conda-env site-packages paths.

Verified images install the test env under /opt/miniconda3/envs/testbed/, so
framework frames look like
  /opt/miniconda3/envs/testbed/lib/python3.9/site-packages/_pytest/python.py
The site-packages stripper must recognize this (it previously only matched
/usr and /usr/local), otherwise pytest framework frames leak into RCA
candidates. See TRAJECTORY_ANALYSIS F2.
"""
from debug_agent.path_norm import (
    normalize_repo_path,
    is_framework_file,
    is_noise_file,
)

CONDA_PYTEST = (
    "/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/_pytest/python.py"
)
CONDA_PROJECT = (
    "/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/somepkg/core.py"
)


def test_conda_sitepackages_prefix_is_stripped():
    assert normalize_repo_path(CONDA_PYTEST) == "_pytest/python.py"


def test_conda_pytest_frame_is_framework_noise():
    assert is_framework_file(CONDA_PYTEST) is True
    assert is_noise_file(CONDA_PYTEST) is True


def test_conda_project_code_is_not_over_filtered():
    # An editable project installed into the conda env must normalize to its
    # package-relative path and must NOT be treated as framework noise.
    assert normalize_repo_path(CONDA_PROJECT) == "somepkg/core.py"
    assert is_framework_file(CONDA_PROJECT) is False


def test_pro_usr_paths_still_work():
    # Regression: the broadened regex must keep handling the old /usr layouts.
    dist = "/usr/local/lib/python3.9/dist-packages/_pytest/python.py"
    site = "/usr/lib/python3.9/site-packages/_pytest/python.py"
    assert normalize_repo_path(dist) == "_pytest/python.py"
    assert normalize_repo_path(site) == "_pytest/python.py"
    assert is_framework_file(dist) is True
    assert is_framework_file(site) is True
