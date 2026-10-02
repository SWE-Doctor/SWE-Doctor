"""Stage 2a (repro-trace) for SWE-bench Verified."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

from datasets import load_dataset

from run_verified_test.entryscript_verified import build_entryscript_verified
from verified_common import VERIFIED_DATASET, verified_image_name

# Pro modules in run_pro_test/ use bare sibling imports (from common import ...)
# so they can't be imported as package sub-modules. We add run_pro_test/ to
# sys.path once at module load time — this mirrors how the Pro scripts run.
_PRO_DIR = str(Path(__file__).resolve().parents[1] / "run_pro_test")
if _PRO_DIR not in sys.path:
    sys.path.insert(0, _PRO_DIR)

_python_runner = importlib.import_module("python_runner")
_run_repro_trace = importlib.import_module("run_repro_trace")

extract_failed_tests = _python_runner.extract_failed_tests
has_pytest_failures = _python_runner.has_pytest_failures
load_phase2_coverage_files = _python_runner.load_phase2_coverage_files
load_accepted_tests = _run_repro_trace.load_accepted_tests


def _prepare_workspace(ws: Path, tests: list[str], base_commit: str) -> None:
    (ws / "_repro_tests").mkdir(parents=True, exist_ok=True)
    rel_paths = []
    for i, code in enumerate(tests):
        rel = f"_repro_tests/repro_{i}.py"
        (ws / rel).write_text(code)
        rel_paths.append(rel)
    (ws / "entryscript.sh").write_text(
        build_entryscript_verified(base_commit=base_commit, repro_rel_paths=rel_paths)
    )


def _run_container(image: str, ws: Path, timeout: int) -> int:
    name = f"verif_{uuid.uuid4().hex[:12]}"
    cmd = [
        "docker", "run", "--name", name, "--entrypoint", "/bin/bash",
        "-v", f"{ws.resolve()}:/workspace:rw", image,
        "-c", "bash /workspace/entryscript.sh",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode
    finally:
        subprocess.run(["docker", "stop", "-t", "10", name], capture_output=True, text=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)


def process_one(row: dict, repro_dir: Path, out_dir: Path, timeout: int) -> dict:
    iid = row["instance_id"]
    tests, reason = load_accepted_tests(repro_dir, iid)
    if not tests:
        return {"instance_id": iid, "status": "skipped", "reason": reason}
    inst_out = out_dir / iid
    ws = inst_out / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    image = verified_image_name(iid)
    _prepare_workspace(ws, tests, row["base_commit"])
    rc = _run_container(image, ws, timeout)
    (inst_out / "docker_image.txt").write_text(image + "\n")
    (ws / "docker_image.txt").write_text(image + "\n")
    stdout = (ws / "stdout.log").read_text(errors="replace") if (ws / "stdout.log").exists() else ""
    stderr = (ws / "stderr.log").read_text(errors="replace") if (ws / "stderr.log").exists() else ""
    failed = extract_failed_tests(stdout, stderr)
    # load_phase2_coverage_files (Pro) strips a leading "/app/" prefix, so
    # Verified paths under /testbed come back as "testbed/django/x.py" instead
    # of the repo-relative "django/x.py". Strip the extra "testbed/" prefix.
    cov_files = [
        c[len("testbed/"):] if c.startswith("testbed/") else c
        for c in load_phase2_coverage_files(ws)
    ]
    return {
        "instance_id": iid,
        "status": "ok",
        "container_rc": rc,
        "had_failures": has_pytest_failures(stdout, stderr),
        "failed_nodeids": failed,
        "coverage_files": cov_files[:50],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="V2 Stage-2a repro-trace (Verified)")
    p.add_argument("--repro-dir", type=Path, required=True)
    p.add_argument("--issue-ids-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", type=str, default=VERIFIED_DATASET)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = [x.strip() for x in args.issue_ids_file.read_text().splitlines() if x.strip()]
    by_id = {r["instance_id"]: dict(r) for r in load_dataset(args.dataset, split=args.split)}
    todo = [by_id[i] for i in ids if i in by_id]

    summaries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futs = {ex.submit(process_one, row, args.repro_dir, args.output_dir,
                          args.timeout_seconds): row["instance_id"] for row in todo}
        for fut in concurrent.futures.as_completed(futs):
            iid = futs[fut]
            try:
                summaries.append(fut.result())
            except Exception as exc:
                summaries.append({"instance_id": iid, "status": "error", "reason": repr(exc)})

    (args.output_dir / "repro_trace_summary.json").write_text(json.dumps(summaries, indent=2))
    (args.output_dir / "skipped.json").write_text(
        json.dumps([s for s in summaries if s.get("status") == "skipped"], indent=2)
    )


if __name__ == "__main__":
    main()
