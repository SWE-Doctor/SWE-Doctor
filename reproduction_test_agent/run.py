"""CLI entry point for reproduction test generation.

Usage:
    # Single instance
    python -m reproduction_test_agent.run \
        --instance-id "django__django-12345" \
        --problem-statement "Bug: ..." \
        --repo-path /path/to/django \
        --model "anthropic/claude-sonnet-4-6"

    # From SWE-bench dataset JSON
    python -m reproduction_test_agent.run \
        --dataset swebench_pro.json \
        --repo-base /path/to/repos \
        --model "anthropic/claude-sonnet-4-6"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add mini-swe-agent src to path
_MINI_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_MINI_SRC) not in sys.path:
    sys.path.insert(0, str(_MINI_SRC))

from minisweagent.environments.local import LocalEnvironment

from .config import ReproTestConfig
from .llm import set_trace_file
from .pipeline import run_pipeline, save_results


def main():
    parser = argparse.ArgumentParser(description="e-Otter++ reproduction test generator")

    # Input: single instance
    parser.add_argument("--instance-id", type=str, help="SWE-bench instance ID")
    parser.add_argument("--problem-statement", type=str, help="Issue description text")
    parser.add_argument("--problem-statement-file", type=str, help="File containing issue description")
    parser.add_argument("--repo-path", type=str, help="Path to repo checkout (c_old)")

    # Input: batch from dataset
    parser.add_argument("--dataset", type=str, help="Path to SWE-bench dataset JSON")
    parser.add_argument("--repo-base", type=str, help="Base directory containing repo checkouts")
    parser.add_argument("--instance-filter", type=str, help="Comma-separated instance IDs to process")

    # Config
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-6")
    parser.add_argument("--repair-temperature", type=float, default=0.8)
    parser.add_argument("--max-repair-attempts", type=int, default=10)
    parser.add_argument("--test-timeout", type=int, default=60)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--morphs", type=str, default="standard,simple,dropCode,initTest,initPatch")
    parser.add_argument("--masks", type=str, default="planner,full,testLoc,patchLoc,none")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = ReproTestConfig(
        model_name=args.model,
        repair_temperature=args.repair_temperature,
        max_repair_attempts=args.max_repair_attempts,
        test_timeout=args.test_timeout,
        output_dir=args.output_dir,
        morphs=args.morphs.split(","),
        masks=args.masks.split(","),
    )

    if args.dataset:
        _run_batch(args, config)
    elif args.instance_id and (args.problem_statement or args.problem_statement_file) and args.repo_path:
        _run_single(args, config)
    else:
        parser.error("Provide either --dataset or (--instance-id + --problem-statement + --repo-path)")


def _run_single(args, config: ReproTestConfig):
    problem = args.problem_statement
    if args.problem_statement_file:
        problem = Path(args.problem_statement_file).read_text()

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_trace_file(out / f"{args.instance_id}_trace.jsonl")
    env = LocalEnvironment(cwd=args.repo_path, timeout=config.test_timeout)
    results = run_pipeline(
        instance_id=args.instance_id,
        problem_statement=problem,
        env=env,
        cwd=args.repo_path,
        config=config,
    )
    save_results(results, config.output_dir)
    set_trace_file(None)
    _print_summary(results)


def _run_batch(args, config: ReproTestConfig):
    dataset = json.loads(Path(args.dataset).read_text())
    if isinstance(dataset, dict):
        dataset = list(dataset.values())

    # Filter instances if requested
    if args.instance_filter:
        ids = set(args.instance_filter.split(","))
        dataset = [d for d in dataset if d["instance_id"] in ids]

    logging.getLogger("repro_test").info("Processing %d instances", len(dataset))

    for instance in dataset:
        instance_id = instance["instance_id"]

        # Determine repo path
        if args.repo_base:
            # Convention: repo_base/owner__repo/instance_id
            repo_path = str(Path(args.repo_base) / instance_id)
            if not Path(repo_path).exists():
                # Try: repo_base/repo_name
                repo_name = instance.get("repo", "").replace("/", "__")
                repo_path = str(Path(args.repo_base) / repo_name)
        else:
            repo_path = instance.get("repo_path", "")

        if not Path(repo_path).is_dir():
            logging.getLogger("repro_test").warning(
                "Skipping %s: repo not found at %s", instance_id, repo_path
            )
            continue

        env = LocalEnvironment(cwd=repo_path, timeout=config.test_timeout)
        results = run_pipeline(
            instance_id=instance_id,
            problem_statement=instance["problem_statement"],
            env=env,
            cwd=repo_path,
            config=config,
        )
        save_results(results, config.output_dir)
        _print_summary(results)


def _print_summary(results: dict):
    n_total = len(results["candidates"])
    n_accepted = len(results["accepted"])
    print(f"\n{'='*60}")
    print(f"Instance: {results['instance_id']}")
    print(f"Candidates: {n_total}, Accepted: {n_accepted}")
    print(f"Time: {results['elapsed_seconds']:.1f}s")

    if results["best_test"]:
        print(f"\nBest test (first accepted):")
        print("-" * 40)
        # Print first 30 lines
        lines = results["best_test"].splitlines()
        for line in lines[:30]:
            print(f"  {line}")
        if len(lines) > 30:
            print(f"  ... ({len(lines) - 30} more lines)")
    else:
        print("\nNo accepted tests (none failed for the right reason)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
