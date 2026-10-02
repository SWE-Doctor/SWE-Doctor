"""stage_go_repro.collect_repro_tests must pair each accepted *_test.go with its
sidecar .relpath (package-relative dest), so Stage-2 can place it in the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_pro_test"))
from stage_go_repro import collect_repro_tests


def test_collect_pairs_go_test_with_relpath(tmp_path):
    iid = "instance_foo"
    d = tmp_path / iid
    d.mkdir()
    (d / f"{iid}_test_0.go").write_text("package server\nfunc TestX(t *testing.T){}\n")
    (d / f"{iid}_test_0.relpath").write_text("server/zzz_repro_test.go")
    out = collect_repro_tests(tmp_path, iid)
    assert len(out) == 1
    name, content, relpath = out[0]
    assert name == f"{iid}_test_0.go"
    assert "package server" in content
    assert relpath == "server/zzz_repro_test.go"


def test_collect_missing_relpath_is_empty_string(tmp_path):
    iid = "instance_bar"
    d = tmp_path / iid
    d.mkdir()
    (d / f"{iid}_test_0.go").write_text("package x\n")
    out = collect_repro_tests(tmp_path, iid)
    assert out and out[0][2] == ""


def test_collect_empty_when_no_go_tests(tmp_path):
    iid = "instance_baz"
    (tmp_path / iid).mkdir()
    assert collect_repro_tests(tmp_path, iid) == []
