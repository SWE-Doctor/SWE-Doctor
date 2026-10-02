from verified_common import (
    VERIFIED_DATASET,
    VERIFIED_CWD,
    VERIFIED_CONDA_ACTIVATE,
    TESTBED_PYTHON,
    verified_image_name,
)


def test_dataset_constant():
    assert VERIFIED_DATASET == "SWE-bench/SWE-bench_Verified"


def test_workdir_is_testbed():
    assert VERIFIED_CWD == "/testbed"


def test_conda_activate_snippet():
    assert "conda activate testbed" in VERIFIED_CONDA_ACTIVATE
    assert "/opt/miniconda3/etc/profile.d/conda.sh" in VERIFIED_CONDA_ACTIVATE


def test_testbed_python_path():
    assert TESTBED_PYTHON == "/opt/miniconda3/envs/testbed/bin/python"


def test_image_name_double_underscore_to_1776():
    assert (
        verified_image_name("psf__requests-1142")
        == "docker.io/swebench/sweb.eval.x86_64.psf_1776_requests-1142:latest"
    )


def test_image_name_uppercase_id_is_lowercased():
    assert (
        verified_image_name("Django__Django-12345")
        == "docker.io/swebench/sweb.eval.x86_64.django_1776_django-12345:latest"
    )
