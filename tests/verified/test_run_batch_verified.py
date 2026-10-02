from reproduction_test_agent.run_batch_verified import build_verified_instance


def test_build_verified_instance_image_and_cwd():
    row = {
        "instance_id": "psf__requests-1142",
        "repo": "psf/requests",
        "base_commit": "deadbeef",
        "problem_statement": "Boom happens when ...",
        "version": "1.1",
    }
    inst = build_verified_instance(row)
    assert inst["instance_id"] == "psf__requests-1142"
    assert inst["image_name"] == (
        "docker.io/swebench/sweb.eval.x86_64.psf_1776_requests-1142:latest"
    )
    assert "Boom happens" in inst["problem_statement"]
