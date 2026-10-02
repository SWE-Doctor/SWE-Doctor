#!/usr/bin/env python3
"""Execute reproduction_test_agent outputs inside each instance's docker
container, producing the standard RCA workspace layout.

Bridges Stage 1 (reproduction_test_agent) and Stage 2 (run_statement_rca):

  - Reads <repro-dir>/<instance_id>/<instance_id>.json
  - Collects ALL accepted[*].final_test strings (or best_test as fallback)
  - Stages them as _repro_tests/repro_<i>.py inside the instance workspace
  - Calls python_runner.run_one_instance, which launches the SWE-bench Pro
    docker image, runs all tests in ONE pytest invocation under coverage.py +
    focused_trace_plugin, writing phase2_coverage.json / focused_*.log /
    stdout.log / stderr.log to the workspace. run_statement_rca.py then
    consumes that workspace directly with zero changes.

  - Instances whose `accepted` list is empty are skipped and recorded to
    <output-dir>/skipped.json with reason='no_accepted_repro_test'. RCA stage
    naturally produces no file for them; the repair agent already tolerates
    missing RCA input and will run on the plain problem statement.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import InstanceResult, make_error_result  # noqa: E402
from python_runner import run_one_instance  # noqa: E402


REPRO_TEST_SUBDIR = "_repro_tests"  # inside both workspace and /app


def load_issue_ids(path: Path) -> list[str]:
    return [
        l.strip()
        for l in path.read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]


def load_accepted_tests(repro_dir: Path, instance_id: str) -> tuple[list[str], str]:
    """Return (list_of_test_sources, reason_if_empty)."""
    p = repro_dir / instance_id / f"{instance_id}.json"
    if not p.exists():
        return [], "repro_output_missing"
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        return [], f"repro_output_unreadable: {exc}"
    accepted = data.get("accepted") or []
    tests: list[str] = []
    for cand in accepted:
        code = (cand.get("final_test") or cand.get("initial_test") or "").strip()
        if code:
            tests.append(code)
    if tests:
        return tests, ""
    best = (data.get("best_test") or "").strip()
    if best:
        return [best], ""
    return [], "no_accepted_repro_test"


def build_repro_sample(row: dict, tests: list[str]) -> tuple[dict, dict[str, str], str]:
    """Return (sample_for_run_one_instance, extra_workspace_files, preamble)."""
    # Relative to /app and to workspace; pytest will be cd'd to /app by run_script.sh.
    rel_paths = [f"{REPRO_TEST_SUBDIR}/repro_{i}.py" for i in range(len(tests))]

    # Files go under workspace/_repro_tests/ → preamble copies them to /app/_repro_tests/
    extra_files: dict[str, str] = {
        rel: code for rel, code in zip(rel_paths, tests)
    }
    # Ensure each test file is pytest-discoverable (no conftest needed — pytest
    # picks up any *.py passed on the command line).
    preamble = (
        f"mkdir -p /app/{REPRO_TEST_SUBDIR}\n"
        f"cp /workspace/{REPRO_TEST_SUBDIR}/*.py /app/{REPRO_TEST_SUBDIR}/ 2>/dev/null || true\n"
        f"touch /app/{REPRO_TEST_SUBDIR}/__init__.py || true\n"
    )

    sample = {
        **row,
        # Override whatever the dataset had — we run our generated tests only.
        "selected_test_files_to_run": rel_paths,
    }
    return sample, extra_files, preamble


def process_one(
    row: dict,
    repro_dir: Path,
    scripts_dir: Path,
    output_dir: Path,
    timeout_seconds: int,
) -> dict:
    instance_id = row["instance_id"]
    tests, reason = load_accepted_tests(repro_dir, instance_id)
    if not tests:
        return {"instance_id": instance_id, "status": "skipped", "reason": reason}

    sample, extra_files, preamble = build_repro_sample(row, tests)
    try:
        result: InstanceResult = run_one_instance(
            sample=sample,
            scripts_dir=scripts_dir,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
            entryscript_preamble=preamble,
            extra_workspace_files=extra_files,
        )
    except Exception as exc:
        return {"instance_id": instance_id, "status": "error", "reason": repr(exc)}

    return {
        "instance_id": instance_id,
        "status": "ok",
        "num_tests": len(tests),
        "result": asdict(result) if hasattr(result, "__dataclass_fields__") else None,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute repro tests for RCA ingestion")
    ap.add_argument("--repro-dir", type=Path, required=True,
                    help="reproduction_test_agent output dir (contains <id>/<id>.json)")
    ap.add_argument("--issue-ids-file", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Stage 2 trace dir — <id>/workspace/ will be written here")
    ap.add_argument("--scripts-dir", type=Path, required=True,
                    help="SWE-bench Pro run_scripts dir")
    ap.add_argument("--dataset", type=str, default="ScaleAI/SWE-bench_Pro")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--timeout-seconds", type=int, default=2400)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.repro_dir.exists():
        print(f"Repro dir does not exist: {args.repro_dir}", file=sys.stderr)
        sys.exit(2)

    issue_ids = load_issue_ids(args.issue_ids_file)
    print(f"Loading dataset {args.dataset}/{args.split}...")
    rows = {r["instance_id"]: dict(r) for r in load_dataset(args.dataset, split=args.split)}

    todo = [rows[i] for i in issue_ids if i in rows]
    missing = [i for i in issue_ids if i not in rows]
    print(f"Selected {len(todo)} instances, missing {len(missing)}.")

    summaries: list[dict] = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futures = {
            ex.submit(
                process_one, row, args.repro_dir, args.scripts_dir,
                args.output_dir, args.timeout_seconds,
            ): row["instance_id"]
            for row in todo
        }
        for fut in concurrent.futures.as_completed(futures):
            iid = futures[fut]
            try:
                summary = fut.result()
            except Exception as exc:
                summary = {"instance_id": iid, "status": "error", "reason": repr(exc)}
            summaries.append(summary)
            print(f"[{summary['status']}] {iid}"
                  + (f" — {summary.get('reason','')}" if summary['status'] != "ok" else ""))

    # Write top-level manifests
    by_status: dict[str, list[dict]] = {}
    for s in summaries:
        by_status.setdefault(s["status"], []).append(s)

    (args.output_dir / "repro_trace_summary.json").write_text(
        json.dumps({
            "total": len(summaries),
            "ok": len(by_status.get("ok", [])),
            "skipped": len(by_status.get("skipped", [])),
            "error": len(by_status.get("error", [])),
            "missing_from_dataset": missing,
            "elapsed_seconds": round(time.time() - start, 2),
        }, indent=2)
    )
    (args.output_dir / "skipped.json").write_text(
        json.dumps(by_status.get("skipped", []), indent=2)
    )
    if by_status.get("error"):
        (args.output_dir / "errors.json").write_text(
            json.dumps(by_status["error"], indent=2)
        )

    print(
        f"Done. ok={len(by_status.get('ok', []))} "
        f"skipped={len(by_status.get('skipped', []))} "
        f"error={len(by_status.get('error', []))}"
    )


if __name__ == "__main__":
    main()
