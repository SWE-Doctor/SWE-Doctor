#!/usr/bin/env python3
"""Run SWE-bench Pro tests for a list of instance IDs and inspect stack traces."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

import go_runner
import js_runner
import python_runner
from common import (
    InstanceResult,
    extract_patch_files,
    extract_stack_files,
    get_error_excerpt,
    load_execution_files,
    make_error_result,
    match_patch_files,
)

_DONE_MARKER_FILE = "_run_completed.marker"


# ── Language detection ────────────────────────────────────────────────────────

_GO_EXTS = {".go"}
_JS_EXTS = {".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"}


def detect_language(sample: dict) -> str:
    """Infer repo language from patch file extensions."""
    patch_files = extract_patch_files(sample.get("patch", ""))
    exts = {Path(f).suffix.lower() for f in patch_files}
    if exts & _GO_EXTS:
        return "go"
    if exts & _JS_EXTS:
        return "js"
    return "python"


_RUNNERS = {
    "python": python_runner,
    "go": go_runner,
    "js": js_runner,
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    # External datasets live outside the repo; resolve from env vars so the
    # pipeline is portable, with the repo's parent directory as the fallback.
    data_root = Path(os.environ.get("UTA_DATA_ROOT", str(repo_root.parent)))
    pro_os_root = os.environ.get("SWEBENCH_PRO_OS_ROOT", str(data_root / "SWE-bench_Pro-os"))

    parser = argparse.ArgumentParser(
        description="Run SWE-bench Pro tests and check whether stack trace hits patch files."
    )
    parser.add_argument(
        "--issue-ids-file",
        default=os.environ.get(
            "SWEBENCH_PRO_ISSUE_IDS", str(data_root / "swebench_pro_issue_ids_100.txt")
        ),
        help="Text file with one instance_id per line.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "run_pro_test" / "results"),
        help="Output directory for logs and summaries.",
    )
    parser.add_argument(
        "--scripts-dir",
        default=str(Path(pro_os_root) / "run_scripts"),
        help="Directory containing per-instance run_script.sh and parser.py.",
    )
    parser.add_argument("--dataset", default="ScaleAI/SWE-bench_Pro", help="HuggingFace dataset name.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--timeout-seconds", type=int, default=2400, help="Per-instance timeout.")
    parser.add_argument("--max-instances", type=int, default=0, help="Only run first N instances, 0 means all.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel worker threads.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output dir: skip instance folders that look already completed.",
    )
    return parser.parse_args()


def load_issue_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


# ── Output helpers ────────────────────────────────────────────────────────────

def write_summary(results: list[InstanceResult], out_dir: Path) -> None:
    jsonl_path = out_dir / "results.jsonl"
    with jsonl_path.open("w") as f:
        for item in results:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    summary = {
        "total": len(results),
        "has_traceback_hit_patch_file": sum(1 for r in results if r.traceback_matched_patch_files),
        "has_execution_hit_patch_file": sum(1 for r in results if r.execution_matched_patch_files),
        "has_any_hit_patch_file": sum(1 for r in results if r.stack_contains_patch_file),
        "has_stack_and_hits_patch_file": sum(1 for r in results if r.stack_contains_patch_file),
        "has_stack_but_no_patch_file_match": sum(
            1 for r in results if r.stack_files and not r.stack_contains_patch_file
        ),
        "no_stack_found": sum(1 for r in results if not r.stack_files),
        "has_failed_tests_detected": sum(1 for r in results if r.failed_tests),
        "output_incomplete": sum(1 for r in results if r.output_incomplete),
        "timeout": sum(1 for r in results if r.timeout),
        "missing_assets": sum(1 for r in results if r.missing_assets),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [
        "# SWE-bench Pro test stack analysis",
        "",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "",
        "## Per instance quick view",
        "",
        "| instance_id | traceback_hit | execution_hit | any_hit | timeout | missing_assets |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.instance_id} | {int(bool(r.traceback_matched_patch_files))} "
            f"| {int(bool(r.execution_matched_patch_files))} | {int(r.stack_contains_patch_file)} "
            f"| {int(r.timeout)} | {int(bool(r.missing_assets))} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines))


def append_result_jsonl(path: Path, result: InstanceResult) -> None:
    with path.open("a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def load_results_jsonl(path: Path) -> dict[str, InstanceResult]:
    if not path.exists():
        return {}
    out: dict[str, InstanceResult] = {}
    field_names = set(InstanceResult.__dataclass_fields__.keys())
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if not field_names.issubset(payload.keys()):
            continue
        try:
            item = InstanceResult(**{k: payload[k] for k in field_names})
        except Exception:
            continue
        out[item.instance_id] = item
    return out


def is_instance_completed_by_folder(output_dir: Path, instance_id: str) -> bool:
    """
    Heuristic completion detection without results.jsonl.
    A completed instance should have final analysis artifacts persisted in sample_dir.
    """
    sample_dir = output_dir / instance_id
    if not sample_dir.exists():
        return False
    if (sample_dir / _DONE_MARKER_FILE).exists():
        return True

    has_failure_trace = (sample_dir / "failure_trace.log").exists()
    has_main_log = (sample_dir / "stdout.log").exists() or (sample_dir / "stderr.log").exists()
    has_exit_code = (sample_dir / "pytest_exit_code.txt").exists()
    return has_failure_trace and (has_main_log or has_exit_code)


def mark_instance_completed(output_dir: Path, result: InstanceResult) -> None:
    sample_dir = output_dir / result.instance_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "instance_id": result.instance_id,
        "completed_at": int(time.time()),
        "timeout": bool(result.timeout),
        "has_missing_assets": bool(result.missing_assets),
    }
    (sample_dir / _DONE_MARKER_FILE).write_text(json.dumps(marker_payload, ensure_ascii=False, indent=2))


def _parse_optional_int(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(errors="replace").strip())
    except Exception:
        return None


def build_result_from_instance_folder(instance_dir: Path) -> InstanceResult | None:
    instance_id = instance_dir.name
    if not instance_id.startswith("instance_"):
        return None

    stdout_path = instance_dir / "stdout.log"
    stderr_path = instance_dir / "stderr.log"
    failure_trace_path = instance_dir / "failure_trace.log"
    patch_path = instance_dir / "patch.diff"

    if not stdout_path.exists() and not stderr_path.exists() and not failure_trace_path.exists():
        return None

    stdout_text = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
    failure_trace_excerpt = failure_trace_path.read_text(errors="replace") if failure_trace_path.exists() else ""
    patch_text = patch_path.read_text(errors="replace") if patch_path.exists() else ""
    patch_files = extract_patch_files(patch_text)
    stack_files = sorted(set(extract_stack_files(stdout_text + "\n" + failure_trace_excerpt, stderr_text)))
    workspace = instance_dir / "workspace"
    execution_files = load_execution_files(workspace) if workspace.exists() else []
    traceback_matched = match_patch_files(patch_files, stack_files)
    execution_matched = match_patch_files(patch_files, execution_files)
    matched = sorted(set(traceback_matched + execution_matched))

    marker_timeout = False
    marker_missing_assets = False
    marker_path = instance_dir / _DONE_MARKER_FILE
    if marker_path.exists():
        try:
            payload = json.loads(marker_path.read_text(errors="replace"))
            marker_timeout = bool(payload.get("timeout"))
            marker_missing_assets = bool(payload.get("has_missing_assets"))
        except Exception:
            pass

    missing_assets = ["unknown_missing_assets"] if marker_missing_assets else []

    return InstanceResult(
        instance_id=instance_id,
        repo="",
        docker_image="",
        base_commit="",
        timeout=marker_timeout,
        container_status_code=None,
        missing_assets=missing_assets,
        patch_files=patch_files,
        stack_files=stack_files,
        execution_files=execution_files,
        traceback_matched_patch_files=traceback_matched,
        execution_matched_patch_files=execution_matched,
        matched_patch_files=matched,
        stack_contains_patch_file=bool(matched),
        error_excerpt=get_error_excerpt(stdout_text, stderr_text),
        failure_trace_excerpt=failure_trace_excerpt,
        failure_trace_log=str(failure_trace_path.resolve()) if failure_trace_path.exists() else "",
        failed_tests=[],
        stdout_log=str(stdout_path.resolve()) if stdout_path.exists() else "",
        stderr_log=str(stderr_path.resolve()) if stderr_path.exists() else "",
        pytest_exit_code=_parse_optional_int(instance_dir / "pytest_exit_code.txt"),
        pytest_report_xml=str((instance_dir / "pytest-report.xml").resolve()) if (instance_dir / "pytest-report.xml").exists() else "",
        output_incomplete=False,
        workspace=str(workspace.resolve()) if workspace.exists() else "",
        duration_seconds=0.0,
    )


def collect_results_from_output_dir(output_dir: Path, in_memory_results: list[InstanceResult]) -> list[InstanceResult]:
    by_id = load_results_jsonl(output_dir / "results.jsonl")
    for item in in_memory_results:
        by_id[item.instance_id] = item

    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("instance_"):
            continue
        if child.name in by_id:
            continue
        folder_result = build_result_from_instance_folder(child)
        if folder_result is not None:
            by_id[folder_result.instance_id] = folder_result

    return sorted(by_id.values(), key=lambda x: x.instance_id)


def prepare_output_dir(output_dir: Path) -> Path:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    try:
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except PermissionError as exc:
        suffix = time.strftime("%Y%m%d_%H%M%S")
        fallback_dir = output_dir.with_name(f"{output_dir.name}_{suffix}")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Warning: cannot clean output dir '{output_dir}' ({exc}). "
            f"Using fallback: '{fallback_dir}'."
        )
        return fallback_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    if args.resume:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = prepare_output_dir(output_dir)
    incremental_jsonl_path = output_dir / "results.jsonl"
    scripts_dir = Path(args.scripts_dir)
    issue_ids_path = Path(args.issue_ids_file)

    issue_ids = load_issue_ids(issue_ids_path)
    if args.max_instances > 0:
        issue_ids = issue_ids[: args.max_instances]

    dataset = load_dataset(args.dataset, split=args.split)
    by_id = {row["instance_id"]: row for row in dataset}
    existing_results = load_results_jsonl(output_dir / "results.jsonl") if args.resume else {}

    def _log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    results: list[InstanceResult] = []
    workers = max(1, int(args.num_workers))
    _log(f"Running {len(issue_ids)} instances with num_workers={workers}")
    if args.resume:
        _log("Resume mode: skip issue folders with completed artifacts.")

    future_to_meta: dict[concurrent.futures.Future[InstanceResult], tuple[int, str]] = {}
    indexed_results: dict[int, InstanceResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, iid in enumerate(issue_ids, start=1):
            if args.resume and is_instance_completed_by_folder(output_dir, iid):
                old = existing_results.get(iid)
                if old is not None:
                    indexed_results[idx] = old
                _log(f"[{idx}/{len(issue_ids)}] Reused {iid} (already completed by folder)")
                continue

            row = by_id.get(iid)
            if row is None:
                indexed_results[idx] = make_error_result(
                    iid,
                    "Instance id not found in dataset",
                    missing_assets=["instance_not_found_in_dataset"],
                )
                mark_instance_completed(output_dir, indexed_results[idx])
                append_result_jsonl(incremental_jsonl_path, indexed_results[idx])
                _log(f"[{idx}/{len(issue_ids)}] Skipped {iid} (not found in dataset)")
                continue

            lang = detect_language(row)
            runner = _RUNNERS[lang]
            _log(f"[{idx}/{len(issue_ids)}] Submitted {iid} ({lang})")
            fut = executor.submit(
                runner.run_one_instance,
                sample=row,
                scripts_dir=scripts_dir,
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
            future_to_meta[fut] = (idx, iid)

        completed = 0
        total_submitted = len(future_to_meta)
        for fut in concurrent.futures.as_completed(future_to_meta):
            idx, iid = future_to_meta[fut]
            completed += 1
            try:
                indexed_results[idx] = fut.result()
                dur = indexed_results[idx].duration_seconds
                mark_instance_completed(output_dir, indexed_results[idx])
                append_result_jsonl(incremental_jsonl_path, indexed_results[idx])
                _log(f"[done {completed}/{total_submitted}] {iid} ({dur:.0f}s)")
            except Exception as exc:
                indexed_results[idx] = make_error_result(
                    iid, f"Unhandled worker exception: {repr(exc)}"
                )
                mark_instance_completed(output_dir, indexed_results[idx])
                append_result_jsonl(incremental_jsonl_path, indexed_results[idx])
                _log(f"[done {completed}/{total_submitted}] {iid} failed: {exc}")

    for idx in range(1, len(issue_ids) + 1):
        result = indexed_results.get(idx)
        if result is not None:
            results.append(result)

    all_results = collect_results_from_output_dir(output_dir, results)
    write_summary(all_results, output_dir)
    _log(f"Done. Results written to: {output_dir}")


if __name__ == "__main__":
    # Ensure the runner directory is on the path when executed directly
    sys.path.insert(0, str(Path(__file__).parent))
    main()
