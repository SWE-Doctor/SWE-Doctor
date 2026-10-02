"""Evaluate generated reproduction tests against golden patches (fail-to-pass).

For each instance in preds.json:
  1. Start Docker container from the SWE-bench Pro image (base commit)
  2. Run the generated test → expect FAIL (detects the bug)
  3. Apply the golden patch
  4. Run the generated test → expect PASS (validates the fix)

A test is "f2p" (fail-to-pass) if it fails before and passes after the patch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import threading
import time
from pathlib import Path

from datasets import load_dataset

_MINI_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_MINI_SRC) not in sys.path:
    sys.path.insert(0, str(_MINI_SRC))

from minisweagent.environments.docker import DockerEnvironment

logger = logging.getLogger("repro_test.evaluate")

_RESULT_LOCK = threading.Lock()


def _build_image_name(row: dict, dockerhub_username: str) -> str:
    """Same logic as run_batch.py."""
    tag = (row.get("dockerhub_tag") or "").strip()
    if not tag:
        repo = (row.get("repo") or "").strip()
        if not repo or "/" not in repo:
            raise ValueError(f"Missing dockerhub_tag/repo for {row['instance_id']}")
        repo_base, repo_name_only = repo.lower().split("/")
        hsh = row["instance_id"].replace("instance_", "")
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
        tag = f"{repo_base}.{repo_name_only}-{hsh}"
        tag = tag[:128]
    return f"docker.io/{dockerhub_username}/sweap-images:{tag}"


def _run_test_in_env(
    test_code: str,
    env: DockerEnvironment,
    cwd: str = "/app",
    timeout: int = 60,
    test_filename: str = "_eval_repro_test.py",
) -> dict:
    """Write and run a test, return {passed, returncode, output}."""
    write_cmd = f"cat > {cwd}/{test_filename} << 'EVAL_TEST_EOF'\n{test_code}\nEVAL_TEST_EOF"
    env.execute({"command": write_cmd}, cwd=cwd, timeout=10)

    run_cmd = f"cd {cwd} && python -m pytest {test_filename} -xvs --tb=long --no-header 2>&1"
    result = env.execute({"command": run_cmd}, cwd=cwd, timeout=timeout)

    output = result.get("output", "")
    returncode = result.get("returncode", -1)

    env.execute({"command": f"rm -f {cwd}/{test_filename}"}, cwd=cwd, timeout=5)

    return {
        "passed": returncode == 0,
        "returncode": returncode,
        "output": output,
    }


def _apply_patch(patch: str, env: DockerEnvironment, cwd: str = "/app") -> dict:
    """Apply a git diff patch inside the container."""
    # Write patch to a temp file and apply
    write_cmd = f"cat > /tmp/_eval_patch.diff << 'EVAL_PATCH_EOF'\n{patch}\nEVAL_PATCH_EOF"
    env.execute({"command": write_cmd}, cwd=cwd, timeout=10)

    apply_cmd = f"cd {cwd} && git apply /tmp/_eval_patch.diff 2>&1"
    result = env.execute({"command": apply_cmd}, cwd=cwd, timeout=30)
    return {
        "success": result.get("returncode", -1) == 0,
        "output": result.get("output", ""),
    }


def _revert_patch(patch: str, env: DockerEnvironment, cwd: str = "/app") -> dict:
    """Revert a previously applied patch."""
    revert_cmd = f"cd {cwd} && git apply -R /tmp/_eval_patch.diff 2>&1"
    result = env.execute({"command": revert_cmd}, cwd=cwd, timeout=30)
    return {
        "success": result.get("returncode", -1) == 0,
        "output": result.get("output", ""),
    }


def evaluate_instance(
    instance_id: str,
    test_code: str,
    golden_patch: str,
    image_name: str,
    timeout: int = 60,
) -> dict:
    """Evaluate a single generated test against the golden patch.

    Returns dict with:
        instance_id, fail_before_patch, pass_after_patch, f2p,
        before_output, after_output, patch_apply_output, error
    """
    result = {
        "instance_id": instance_id,
        "fail_before_patch": None,
        "pass_after_patch": None,
        "f2p": False,
        "before_output": "",
        "after_output": "",
        "patch_apply_output": "",
        "error": None,
    }

    if not test_code:
        result["error"] = "no_test_code"
        return result

    env = None
    try:
        env = DockerEnvironment(
            image=image_name,
            cwd="/app",
            timeout=timeout + 30,
            interpreter=["bash", "-c"],
            env={
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        )

        # Step 1: Run test BEFORE patch (expect FAIL)
        logger.info("[%s] Running test before patch...", instance_id)
        before = _run_test_in_env(test_code, env, timeout=timeout)
        result["fail_before_patch"] = not before["passed"]
        result["before_output"] = before["output"][-2000:]  # truncate

        # Step 2: Apply golden patch
        logger.info("[%s] Applying golden patch...", instance_id)
        patch_result = _apply_patch(golden_patch, env)
        result["patch_apply_output"] = patch_result["output"]
        if not patch_result["success"]:
            result["error"] = f"patch_apply_failed: {patch_result['output'][:500]}"
            return result

        # Step 3: Run test AFTER patch (expect PASS)
        logger.info("[%s] Running test after patch...", instance_id)
        after = _run_test_in_env(test_code, env, timeout=timeout)
        result["pass_after_patch"] = after["passed"]
        result["after_output"] = after["output"][-2000:]

        # f2p = fail before AND pass after
        result["f2p"] = bool(result["fail_before_patch"] and result["pass_after_patch"])

    except Exception as e:
        logger.error("[%s] Error: %s", instance_id, e, exc_info=True)
        result["error"] = str(e)
    finally:
        if env is not None:
            env.cleanup()

    return result


def _save_results(results: list[dict], output_path: Path):
    """Save evaluation results and print summary."""
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)

    # Summary
    total = len(results)
    has_test = [r for r in results if r["error"] != "no_test_code"]
    f2p = [r for r in results if r["f2p"]]
    fail_before = [r for r in results if r["fail_before_patch"]]
    pass_after = [r for r in results if r["pass_after_patch"]]
    errors = [r for r in results if r["error"] and r["error"] != "no_test_code"]
    fail_only = [r for r in results if r["fail_before_patch"] and not r.get("pass_after_patch")]
    pass_only = [r for r in results if r["pass_after_patch"] and not r.get("fail_before_patch")]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total instances:         {total}")
    print(f"With test code:          {len(has_test)}")
    print(f"Errors (Docker/patch):   {len(errors)}")
    print(f"---")
    print(f"Fail before patch:       {len(fail_before)}/{total} ({100*len(fail_before)/total:.1f}%)")
    print(f"Pass after patch:        {len(pass_after)}/{total} ({100*len(pass_after)/total:.1f}%)")
    print(f"F2P (fail→pass):         {len(f2p)}/{total} ({100*len(f2p)/total:.1f}%)")
    print(f"---")
    print(f"Fail-only (not fixed):   {len(fail_only)}")
    print(f"Pass-only (trivial):     {len(pass_only)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated tests with golden patches")
    parser.add_argument("--preds", type=str, required=True, help="Path to preds.json")
    parser.add_argument("--dataset", type=str, default="ScaleAI/SWE-bench_Pro")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("-o", "--output", type=str, default="eval_results.json")
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("--dockerhub-username", type=str, default="jefzda")
    parser.add_argument("--test-timeout", type=int, default=60)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load predictions
    preds_path = Path(args.preds)
    preds = json.loads(preds_path.read_text())
    logger.info("Loaded %d predictions from %s", len(preds), preds_path)

    # Load dataset
    logger.info("Loading dataset %s split=%s ...", args.dataset, args.split)
    hf_cache = Path(__file__).resolve().parent.parent.parent / ".hf_cache"
    cache_kwargs = {"cache_dir": str(hf_cache)} if hf_cache.exists() else {}
    rows = list(load_dataset(args.dataset, split=args.split, **cache_kwargs))
    by_id = {r["instance_id"]: dict(r) for r in rows}

    # Match predictions with dataset
    instance_ids = list(preds.keys())
    if args.max_instances > 0:
        instance_ids = instance_ids[:args.max_instances]

    tasks = []
    for iid in instance_ids:
        if iid not in by_id:
            logger.warning("Instance %s not found in dataset, skipping", iid)
            continue
        row = by_id[iid]
        test_code = preds[iid].get("best_test") or ""
        golden_patch = row.get("patch") or ""
        if not golden_patch:
            logger.warning("Instance %s has no golden patch, skipping", iid)
            continue
        try:
            image_name = _build_image_name(row, args.dockerhub_username)
        except ValueError as e:
            logger.warning("Skipping %s: %s", iid, e)
            continue
        tasks.append({
            "instance_id": iid,
            "test_code": test_code,
            "golden_patch": golden_patch,
            "image_name": image_name,
        })

    logger.info("Evaluating %d instances...", len(tasks))

    results = []
    output_path = Path(args.output)

    def _run_one(task):
        r = evaluate_instance(
            instance_id=task["instance_id"],
            test_code=task["test_code"],
            golden_patch=task["golden_patch"],
            image_name=task["image_name"],
            timeout=args.test_timeout,
        )
        status = "F2P" if r["f2p"] else ("FAIL_ONLY" if r["fail_before_patch"] else "MISS")
        logger.info("[%s] %s (fail_before=%s, pass_after=%s)",
                    r["instance_id"], status, r["fail_before_patch"], r["pass_after_patch"])
        return r

    if args.workers <= 1:
        for task in tasks:
            results.append(_run_one(task))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_run_one, t): t["instance_id"] for t in tasks}
            for fut in concurrent.futures.as_completed(futs):
                iid = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.error("[%s] Uncaught: %s", iid, e, exc_info=True)
                    results.append({
                        "instance_id": iid,
                        "fail_before_patch": None,
                        "pass_after_patch": None,
                        "f2p": False,
                        "error": str(e),
                    })

    # Sort by instance_id for consistent output
    results.sort(key=lambda r: r["instance_id"])
    _save_results(results, output_path)


if __name__ == "__main__":
    main()
