"""Batch runner for reproduction test generation on SWE-bench Pro.

Mirrors swebench_pro_baseline/run_subset.py:
- Loads dataset from HuggingFace
- Spins up Docker containers per instance
- Runs the e-Otter++ pipeline inside each container
- Saves results per instance
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from datasets import load_dataset

# Add mini-swe-agent src to path
_MINI_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_MINI_SRC) not in sys.path:
    sys.path.insert(0, str(_MINI_SRC))

from minisweagent.environments.docker import DockerEnvironment

from .config import ReproTestConfig
from .llm import set_trace_file
from .pipeline import run_pipeline, save_results

logger = logging.getLogger("repro_test.runner")

_OUTPUT_LOCK = threading.Lock()

# External issue-id list lives outside the repo; resolve from an env var, falling
# back to the repo's parent directory so the default is portable.
DEFAULT_ISSUE_IDS_FILE = Path(
    os.environ.get(
        "SWEBENCH_PRO_ISSUE_IDS",
        str(Path(__file__).resolve().parents[1].parent / "swebench_pro_issue_ids.txt"),
    )
)


def _load_issue_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def _build_problem_statement(row: dict) -> str:
    ps = (row.get("problem_statement") or "").strip()
    req = (row.get("requirements") or "").strip()
    iface = (row.get("interface") or "").strip()
    if req or iface:
        return f"{ps}\n\nRequirements:\n{req}\n\nNew interfaces introduced:\n{iface}".strip()
    return ps


def _fallback_dockerhub_tag(instance_id: str, repo_name: str) -> str:
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = instance_id.replace("instance_", "")
    if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]
    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    return tag[:128]


def _build_image_name(row: dict, dockerhub_username: str) -> str:
    tag = (row.get("dockerhub_tag") or "").strip()
    if not tag:
        repo = (row.get("repo") or "").strip()
        if not repo or "/" not in repo:
            raise ValueError(f"Missing dockerhub_tag/repo for {row['instance_id']}")
        tag = _fallback_dockerhub_tag(row["instance_id"], repo)
    return f"docker.io/{dockerhub_username}/sweap-images:{tag}"


def _process_instance(
    instance: dict,
    output_dir: Path,
    config: ReproTestConfig,
) -> dict | None:
    """Process a single SWE-bench instance: start Docker, run pipeline, save."""
    instance_id = instance["instance_id"]
    image_name = instance["image_name"]
    problem_statement = instance["problem_statement"]

    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Starting Docker container from %s", instance_id, image_name)
    set_trace_file(instance_dir / f"{instance_id}_trace.jsonl")
    env = None
    try:
        env = DockerEnvironment(
            image=image_name,
            cwd="/app",
            timeout=config.test_timeout,
            interpreter=["bash", "-c"],
            env={
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        )

        results = run_pipeline(
            instance_id=instance_id,
            problem_statement=problem_statement,
            env=env,
            cwd="/app",
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


def _append_prediction(output_dir: Path, instance_id: str, results: dict):
    """Thread-safe append to preds.json."""
    preds_path = output_dir / "preds.json"
    with _OUTPUT_LOCK:
        preds = {}
        if preds_path.exists():
            preds = json.loads(preds_path.read_text())
        preds[instance_id] = {
            "instance_id": instance_id,
            "best_test": results.get("best_test"),
            "n_accepted": len(results.get("accepted", [])),
            "n_candidates": len(results.get("candidates", [])),
            "elapsed_seconds": results.get("elapsed_seconds"),
        }
        preds_path.write_text(json.dumps(preds, indent=2))


def main():
    parser = argparse.ArgumentParser(description="e-Otter++ batch runner for SWE-bench Pro")
    parser.add_argument("--issue-ids-file", type=str, default=str(DEFAULT_ISSUE_IDS_FILE))
    parser.add_argument("--dataset", type=str, default="ScaleAI/SWE-bench_Pro")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("-o", "--output", type=str, default="./repro_test_outputs")
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("-m", "--model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--language", type=str, default="python", choices=["python", "go"])
    parser.add_argument("--dockerhub-username", type=str, default="jefzda")
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--max-repair-attempts", type=int, default=10)
    parser.add_argument("--repair-temperature", type=float, default=0.8)
    parser.add_argument("--test-timeout", type=int, default=60)
    parser.add_argument("--morphs", type=str, default="standard,simple,dropCode,initTest,initPatch")
    parser.add_argument("--masks", type=str, default="planner,full,testLoc,patchLoc,none")
    parser.add_argument("--redo-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(output / "repro_test.log"), mode="a"),
        ],
    )

    config = ReproTestConfig(
        model_name=args.model,
        repair_temperature=args.repair_temperature,
        max_repair_attempts=args.max_repair_attempts,
        test_timeout=args.test_timeout,
        output_dir=args.output,
        morphs=args.morphs.split(","),
        masks=args.masks.split(","),
        language=args.language,
    )

    # Load issue IDs
    issue_ids = _load_issue_ids(Path(args.issue_ids_file))
    if args.max_instances > 0:
        issue_ids = issue_ids[:args.max_instances]
    if not issue_ids:
        logger.error("No instance IDs in %s", args.issue_ids_file)
        sys.exit(1)

    # Load dataset
    logger.info("Loading dataset %s split=%s ...", args.dataset, args.split)
    rows = list(load_dataset(args.dataset, split=args.split))
    by_id = {r["instance_id"]: dict(r) for r in rows}

    missing = [iid for iid in issue_ids if iid not in by_id]
    selected = [iid for iid in issue_ids if iid in by_id]

    # Build instance dicts
    instances = []
    for iid in selected:
        row = by_id[iid]
        instances.append({
            **row,
            "instance_id": iid,
            "problem_statement": _build_problem_statement(row),
            "image_name": _build_image_name(row, args.dockerhub_username),
        })

    # Skip already-done
    if not args.redo_existing:
        preds_path = output / "preds.json"
        if preds_path.exists():
            done = set(json.loads(preds_path.read_text()).keys())
            instances = [i for i in instances if i["instance_id"] not in done]
            logger.info("Skipping %d already-completed instances", len(done))

    logger.info(
        "Requested %d, selected %d, missing %d, to_run %d",
        len(issue_ids), len(selected), len(missing), len(instances),
    )

    if args.dry_run:
        for inst in instances:
            print(f"  {inst['instance_id']}  ->  {inst['image_name']}")
        return

    if not instances:
        logger.info("Nothing to run.")
        return

    # Run
    if args.workers <= 1:
        for inst in instances:
            _process_instance(inst, output, config)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_process_instance, inst, output, config): inst["instance_id"]
                for inst in instances
            }
            for fut in concurrent.futures.as_completed(futs):
                iid = futs[fut]
                try:
                    result = fut.result()
                    if result:
                        n = len(result.get("accepted", []))
                        logger.info("[%s] Done — %d accepted tests", iid, n)
                    else:
                        logger.warning("[%s] Failed (see error log)", iid)
                except Exception as e:
                    logger.error("[%s] Uncaught: %s", iid, e, exc_info=True)


if __name__ == "__main__":
    main()
