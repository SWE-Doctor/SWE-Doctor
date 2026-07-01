"""Stage 1 (reproduction-test generation) batch runner for SWE-bench *Verified*.

Mirrors reproduction_test_agent/run_batch.py but targets the official Verified
images (/testbed, conda env `testbed`, login shell). Pro's run_batch.py is left
untouched; we import its neutral helpers to stay DRY.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
from pathlib import Path

from datasets import load_dataset

from minisweagent.environments.docker import DockerEnvironment

from reproduction_test_agent.config import ReproTestConfig
from reproduction_test_agent.executor import _ensure_pytest
from reproduction_test_agent.llm import set_trace_file
from reproduction_test_agent.pipeline import run_pipeline, save_results
from reproduction_test_agent.run_batch import (
    _build_problem_statement,
    _append_prediction,
    _load_issue_ids,
)
from verified_common import VERIFIED_DATASET, VERIFIED_CWD, verified_image_name

logger = logging.getLogger("repro_batch_verified")


def build_verified_instance(row: dict) -> dict:
    """Normalize a Verified dataset row into the instance dict the pipeline wants."""
    iid = row["instance_id"]
    return {
        **row,
        "instance_id": iid,
        "problem_statement": _build_problem_statement(row),
        "image_name": verified_image_name(iid),
    }


def _process_instance_verified(inst: dict, output_dir: Path, config: ReproTestConfig) -> dict | None:
    instance_id = inst["instance_id"]
    image_name = inst["image_name"]
    problem_statement = inst["problem_statement"]

    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] launching %s", instance_id, image_name)
    set_trace_file(instance_dir / f"{instance_id}_trace.jsonl")
    env = None
    try:
        env = DockerEnvironment(
            image=image_name,
            cwd=VERIFIED_CWD,
            timeout=config.test_timeout,
            interpreter=["bash", "-lc"],  # login shell -> conda activate testbed
            env={
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        )

        # django/sympy Verified images ship their own test runner but not
        # pytest; install it once up front so the generated BRT (and the
        # repair-loop reflection) fail on the bug, not on a missing runner.
        _ensure_pytest(env, cwd=VERIFIED_CWD)

        results = run_pipeline(
            instance_id=instance_id,
            problem_statement=problem_statement,
            env=env,
            cwd=VERIFIED_CWD,
            config=config,
        )

        save_results(results, str(instance_dir))
        _append_prediction(output_dir, instance_id, results)
        return results

    except Exception as e:
        logger.error("[%s] Failed: %s", instance_id, e, exc_info=True)
        err_path = instance_dir / "error.json"
        err_path.write_text(json.dumps({
            "instance_id": instance_id,
            "error": str(e),
            "error_type": type(e).__name__,
        }, indent=2))
        return None
    finally:
        set_trace_file(None)
        if env is not None:
            env.cleanup()


def main() -> None:
    p = argparse.ArgumentParser(description="V2 Stage-1 reproduction-test batch runner (Verified)")
    p.add_argument("--issue-ids-file", type=str, required=True)
    p.add_argument("--dataset", type=str, default=VERIFIED_DATASET)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("-o", "--output", type=str, default="./repro_test_outputs_verified")
    p.add_argument("-w", "--workers", type=int, default=4)
    p.add_argument("-m", "--model", type=str, default="anthropic/claude-sonnet-4-6")
    p.add_argument("--max-repair-attempts", type=int, default=8)
    p.add_argument("--test-timeout", type=int, default=120)
    p.add_argument("--redo-existing", action="store_true", default=False)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    selected = _load_issue_ids(Path(args.issue_ids_file))
    by_id = {r["instance_id"]: dict(r) for r in load_dataset(args.dataset, split=args.split)}
    missing = [i for i in selected if i not in by_id]
    if missing:
        logger.warning("IDs not in dataset (skipped): %s", missing)
    instances = [build_verified_instance(by_id[i]) for i in selected if i in by_id]

    # Skip already-done (resume after a monitor pause). Mirrors Pro run_batch.py
    # so a relaunch jumps over instances already in preds.json instead of
    # re-running all 500 from scratch each time the relay frees up.
    if not args.redo_existing:
        preds_path = output / "preds.json"
        if preds_path.exists():
            done = set(json.loads(preds_path.read_text()).keys())
            before = len(instances)
            instances = [i for i in instances if i["instance_id"] not in done]
            logger.info("Skipping %d already-completed instances (resume)", before - len(instances))

    config = ReproTestConfig(
        model_name=args.model,
        max_repair_attempts=args.max_repair_attempts,
        test_timeout=args.test_timeout,
    )

    if args.workers <= 1:
        for inst in instances:
            _process_instance_verified(inst, output, config)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_process_instance_verified, inst, output, config): inst["instance_id"]
                    for inst in instances}
            for fut in concurrent.futures.as_completed(futs):
                iid = futs[fut]
                try:
                    fut.result()
                    logger.info("[%s] done", iid)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[%s] FAILED: %s", iid, exc)


if __name__ == "__main__":
    main()
