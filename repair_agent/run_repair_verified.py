"""Stage 3 (repair) runner for SWE-bench Verified.

Imports benchmark-neutral helpers from run_repair.py (RCA loading, repro
localizer loading, task rendering, the per-instance agent run) and overrides
only: image-name, dataset default, and config file. Pro's run_repair.py is
left untouched. The shared task_template.j2 is reused (its requirements/
interface block is Jinja-guarded and won't render for Verified).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import threading
import time
from pathlib import Path

import yaml
from datasets import load_dataset
from jinja2 import Template
from rich.live import Live

from minisweagent.utils.log import add_file_handler, logger

try:
    from minisweagent.run.benchmarks.swebench import process_instance
    from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
except Exception:
    from minisweagent.run.extra.swebench import process_instance
    from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager

import repair_agent.run_repair as rr
from verified_common import VERIFIED_DATASET, verified_image_name

DEFAULT_CONFIG = Path(__file__).with_name("repair_config_verified.yaml")
DEFAULT_TASK_TEMPLATE = Path(__file__).with_name("task_template.j2")

_LOCK = threading.Lock()


def _build_instance_verified(
    row: dict,
    rca_dir,
    task_template: Template,
    repro_dir=None,
) -> dict:
    """Mirror run_repair._build_instance but with the Verified image.

    Reuses rr._load_rca, rr._load_repro_localizer, and rr._render_task
    for the RCA loading and task rendering logic. Overrides only image_name
    to use verified_image_name(iid) and repo_name to 'testbed'.
    """
    iid = row["instance_id"]
    rca_data = rr._load_rca(rca_dir, iid) if rca_dir else None
    repro_files, repro_funcs = rr._load_repro_localizer(repro_dir, iid)
    return {
        **row,
        "instance_id": iid,
        "base_commit": row.get("base_commit", ""),
        "problem_statement": rr._render_task(
            task_template, row, rca_data, repro_files, repro_funcs,
        ),
        "image_name": verified_image_name(iid),
        "repo_name": "testbed",
        "_rca_available": rca_data is not None,
    }


def _filter_existing(instances: list[dict], output_dir: Path, redo_existing: bool) -> list[dict]:
    if redo_existing:
        return instances
    preds = output_dir / "preds.json"
    if not preds.exists():
        return instances
    with _LOCK:
        existing = set(json.loads(preds.read_text()).keys())
    if existing:
        logger.info(f"Skipping {len(existing)} instances already in preds.json")
    return [i for i in instances if i["instance_id"] not in existing]


def main() -> None:
    p = argparse.ArgumentParser(description="V2 Stage-3 repair runner (Verified)")
    p.add_argument("--issue-ids-file", type=Path, required=True)
    p.add_argument("--rca-dir", type=Path, required=True)
    p.add_argument("--repro-dir", type=Path, default=None)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--dataset", type=str, default=VERIFIED_DATASET)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("-w", "--workers", type=int, default=4)
    p.add_argument("-m", "--model", type=str, default="anthropic/claude-sonnet-4-6")
    p.add_argument("--max-instances", type=int, default=0)
    p.add_argument("--redo-existing", action="store_true", default=False)
    p.add_argument("--require-rca", action="store_true", default=False,
                   help="Skip instances without RCA output.")
    p.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--task-template", type=Path, default=DEFAULT_TASK_TEMPLATE)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    add_file_handler(args.output / "repair_agent.log")

    if not args.rca_dir.exists():
        p.error(f"RCA dir not found: {args.rca_dir}")
    if not args.config.exists():
        p.error(f"Config not found: {args.config}")
    if not args.task_template.exists():
        p.error(f"Task template not found: {args.task_template}")

    task_template = Template(args.task_template.read_text())

    issue_ids = rr._load_issue_ids(args.issue_ids_file)
    if args.max_instances > 0:
        issue_ids = issue_ids[: args.max_instances]
    if not issue_ids:
        p.error(f"No instance IDs in {args.issue_ids_file}")

    logger.info(f"Loading dataset {args.dataset}/{args.split}...")
    rows = {r["instance_id"]: dict(r) for r in load_dataset(args.dataset, split=args.split)}

    selected = [i for i in issue_ids if i in rows]
    missing = [i for i in issue_ids if i not in rows]
    instances = [
        _build_instance_verified(rows[i], args.rca_dir, task_template, args.repro_dir)
        for i in selected
    ]

    if args.require_rca:
        before = len(instances)
        instances = [i for i in instances if i.get("_rca_available")]
        logger.info("require_rca: kept %d/%d", len(instances), before)

    instances = _filter_existing(instances, args.output, args.redo_existing)
    with_rca = sum(1 for i in instances if i.get("_rca_available"))
    logger.info(
        "Requested %d, selected %d, missing %d, with_rca %d, to_run %d",
        len(issue_ids), len(selected), len(missing), with_rca, len(instances),
    )

    (args.output / "selected_ids.json").write_text(
        json.dumps([i["instance_id"] for i in instances], indent=2)
    )
    (args.output / "missing_ids.json").write_text(json.dumps(missing, indent=2))

    if not instances:
        logger.info("Nothing to run.")
        return

    # Dump one rendered task per instance for offline inspection/debugging.
    debug_dir = args.output / "rendered_tasks"
    debug_dir.mkdir(exist_ok=True)
    for inst in instances:
        (debug_dir / f"{inst['instance_id']}.txt").write_text(inst["problem_statement"])

    for inst in instances:
        inst.pop("_rca_available", None)

    logger.info(f"Loading repair-agent config from {args.config}")
    config = yaml.safe_load(args.config.read_text()) or {}
    if args.model is not None:
        config.setdefault("model", {})["model_name"] = args.model
    # Inherit reasoning_effort from env when set (same pattern as Pro runner).
    _re = os.environ.get("REASONING_EFFORT", "").strip()
    if _re:
        config.setdefault("model", {}).setdefault("model_kwargs", {})["reasoning_effort"] = _re

    progress_manager = RunBatchProgressManager(
        len(instances), args.output / f"exit_statuses_{time.time()}.yaml"
    )

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {
                ex.submit(process_instance, inst, args.output, config, progress_manager): inst["instance_id"]
                for inst in instances
            }
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except concurrent.futures.CancelledError:
                    pass
                except Exception as exc:
                    iid = futures[fut]
                    logger.error(f"Error in future for instance {iid}: {exc}", exc_info=True)
                    progress_manager.on_uncaught_exception(iid, exc)


if __name__ == "__main__":
    main()
