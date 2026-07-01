import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_pro_test.score_localization import score_run


def _make_instance(root: Path, name: str, cands_files: list[str], patched_files: list[str]):
    inst_dir = root / name
    inst_dir.mkdir(parents=True)
    (inst_dir / "patch.diff").write_text(
        "\n".join(
            f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-x\n+y\n"
            for p in patched_files
        )
    )
    out = root / "rca_output"
    out.mkdir(exist_ok=True)
    (out / f"{name}_rca.json").write_text(json.dumps({
        "candidates": [{"file": f, "func": "", "score": 1.0} for f in cands_files]
    }))


def test_score_handles_app_prefix(tmp_path):
    _make_instance(tmp_path, "instance_a",
                   cands_files=["/app/lib/ansible/utils/vars.py"],
                   patched_files=["lib/ansible/utils/vars.py"])
    res = score_run(tmp_path)
    assert res["total"] == 1
    assert res["hits"] == 1
    assert res["misses"] == []


def test_score_misses(tmp_path):
    _make_instance(tmp_path, "instance_b",
                   cands_files=[],
                   patched_files=["qutebrowser/config/qtargs.py"])
    res = score_run(tmp_path)
    assert res["hits"] == 0
    assert len(res["misses"]) == 1
    assert res["misses"][0]["instance"] == "instance_b"


def test_score_dist_packages_collapse(tmp_path):
    _make_instance(tmp_path, "instance_c",
                   cands_files=["/usr/local/lib/python3.9/dist-packages/ansible/plugins/lookup/password.py"],
                   patched_files=["lib/ansible/plugins/lookup/password.py"])
    res = score_run(tmp_path)
    assert res["hits"] == 1
