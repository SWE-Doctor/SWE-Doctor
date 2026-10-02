"""Stage Stage-1 accepted GO repro tests for the debug_agent (Stage 2b).

Go uses debug_agent with a dlv debug session inside the container. This stages
each instance's accepted repro test(s) into <output>/<id>/workspace/_repro_tests/
where run_debug._find_accepted_repro looks, plus problem_statement.txt and
image.txt. Each .go test ships with a sidecar .relpath holding its package-
relative path (e.g. server/zzz_repro_test.go) so Stage-2 can drop it back into
the package directory it belongs to (go tests must live in their package dir).
Stage-only: no docker run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def collect_repro_tests(repro_dir: Path, instance_id: str) -> list[tuple[str, str, str]]:
    """Return [(filename, content, relpath)] for the instance's accepted go repro
    test files (<id>_test_*.go) written by Stage 1's save_results, pairing each
    with its sidecar <id>_test_*.relpath (package-relative dest path)."""
    inst_dir = repro_dir / instance_id
    out: list[tuple[str, str, str]] = []
    for p in sorted(inst_dir.glob(f"{instance_id}_test_*.go")):
        try:
            content = p.read_text(errors="replace")
        except Exception:
            continue
        relpath_file = p.with_suffix(".relpath")
        relpath = relpath_file.read_text().strip() if relpath_file.exists() else ""
        out.append((p.name, content, relpath))
    return out


def _load_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage GO repro tests for debug_agent RCA.")
    ap.add_argument("--repro-dir", required=True, type=Path, help="Stage-1 output dir (<id>/<id>_test_*.go)")
    ap.add_argument("--output-dir", required=True, type=Path, help="Stage-2 dir to populate")
    ap.add_argument("--issue-ids-file", required=True, type=Path)
    ap.add_argument("--dataset", default="ScaleAI/SWE-bench_Pro")
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)

    from datasets import load_dataset
    by_id = {r["instance_id"]: r for r in load_dataset(args.dataset, split=args.split)}

    staged, skipped = [], []
    for iid in _load_ids(args.issue_ids_file):
        tests = collect_repro_tests(args.repro_dir, iid)
        if not tests:
            skipped.append({"instance_id": iid, "reason": "no_accepted_repro_test"})
            continue
        row = by_id.get(iid)
        if row is None:
            skipped.append({"instance_id": iid, "reason": "not_in_dataset"})
            continue
        inst_out = args.output_dir / iid
        repro_out = inst_out / "workspace" / "_repro_tests"
        repro_out.mkdir(parents=True, exist_ok=True)
        for i, (name, content, relpath) in enumerate(tests):
            (repro_out / f"repro_{i}.go").write_text(content)
            if relpath:
                (repro_out / f"repro_{i}.relpath").write_text(relpath)
        (inst_out / "problem_statement.txt").write_text(row.get("problem_statement", "") or "")
        (inst_out / "image.txt").write_text(f"jefzda/sweap-images:{row['dockerhub_tag']}")
        staged.append({"instance_id": iid, "num_tests": len(tests)})
        print(f"[stage-go] staged {iid}: {len(tests)} repro test(s)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "staged.json").write_text(json.dumps(staged, indent=2))
    (args.output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2))
    print(f"[stage-go] staged={len(staged)} skipped={len(skipped)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
