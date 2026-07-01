"""Hit-rate scorer for stage2_rca outputs.

Given a stage2_rca directory containing instance_*/patch.diff and
rca_output/<instance>_rca.json, report the fraction where any
candidate file matches any patched file under canonical paths.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debug_agent.path_norm import paths_match  # noqa: E402

_DIFF_FILE_RE = re.compile(r"^diff --git a/\S+ b/(\S+)", re.M)


def _patched_files(diff_path: Path) -> list[str]:
    return _DIFF_FILE_RE.findall(diff_path.read_text(errors="replace"))


def score_run(stage2_dir: Path) -> dict:
    stage2_dir = Path(stage2_dir)
    rca_dir = stage2_dir / "rca_output"
    misses, hits = [], 0
    instances = []
    for d in sorted(stage2_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("instance_"):
            continue
        rca = rca_dir / f"{d.name}_rca.json"
        patch = d / "patch.diff"
        if not rca.exists() or not patch.exists():
            continue
        instances.append(d.name)
        try:
            cands = [c.get("file", "") for c in json.loads(rca.read_text()).get("candidates", [])]
        except json.JSONDecodeError:
            cands = []
        patched = _patched_files(patch)
        is_hit = any(paths_match(c, p) for c in cands for p in patched)
        if is_hit:
            hits += 1
        else:
            misses.append({"instance": d.name, "candidates": cands, "patched": patched})
    return {"total": len(instances), "hits": hits, "misses": misses}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage2_dir")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = p.parse_args(argv)
    res = score_run(Path(args.stage2_dir))
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Total: {res['total']}  Hits: {res['hits']}  Misses: {len(res['misses'])}")
        for m in res["misses"]:
            print(f"  MISS {m['instance']}")
            print(f"    candidates: {m['candidates']}")
            print(f"    patched:    {m['patched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
