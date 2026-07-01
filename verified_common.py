"""Shared constants + helpers for the SWE-bench *Verified* pipeline path.

Pro lives in run_batch.py / run_pro_test / swebench_pro_baseline and is left
untouched. Everything Verified-specific that more than one stage needs lives
here so the per-stage entry scripts stay thin.
"""
from __future__ import annotations

# HuggingFace dataset id for SWE-bench Verified (split "test", 500 rows).
VERIFIED_DATASET = "SWE-bench/SWE-bench_Verified"

# Repo root / working directory inside the official SWE-bench eval images.
VERIFIED_CWD = "/testbed"

# Verified images layer the project deps into a conda env named "testbed".
# A *non-login* shell (`bash -c`) resolves `python` to base conda (wrong env),
# so every Python-running command must activate this first (or use a login
# shell `bash -lc`, which sources .bashrc and activates it automatically).
VERIFIED_CONDA_ACTIVATE = (
    "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
)

# Absolute path to the testbed interpreter — for callers that invoke python
# directly via `docker exec` (no shell), e.g. the PDB session.
TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"


def verified_image_name(instance_id: str) -> str:
    """Official SWE-bench eval image for a Verified instance.

    Docker tags cannot contain "__", so the harness encodes it as "_1776_".
    The id body is lowercased (Docker tags are case-sensitive but the images
    are published lowercase).
    """
    safe = instance_id.replace("__", "_1776_").lower()
    return f"docker.io/swebench/sweb.eval.x86_64.{safe}:latest"
