from run_verified_test.entryscript_verified import build_entryscript_verified


def test_entryscript_activates_conda_and_cds_testbed():
    script = build_entryscript_verified(
        base_commit="deadbeef",
        repro_rel_paths=["_repro_tests/repro_0.py"],
    )
    assert "conda activate testbed" in script
    assert "cd /testbed" in script
    assert "/workspace/_repro_tests" in script
    assert "_repro_tests/repro_0.py" in script
    assert "pytest" in script
    assert "/workspace/pytest-report.xml" in script
    assert "run_script.sh" not in script
    assert "/app" not in script


def test_entryscript_no_repro_paths_still_valid():
    script = build_entryscript_verified(base_commit="abc123", repro_rel_paths=[])
    assert "conda activate testbed" in script
    assert "git checkout" in script
