#!/usr/bin/env python3
"""Run mini-swe-agent on a SWE-bench Pro subset defined by instance_id list."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.utils.log import add_file_handler, logger

try:
    from minisweagent.run.benchmarks.swebench import process_instance
    from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
except Exception:
    from minisweagent.run.extra.swebench import process_instance
    from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager

DEFAULT_CONFIG_FILE = (
    builtin_config_dir / "benchmarks" / "swebench_pro_textbased.yaml"
    if (builtin_config_dir / "benchmarks" / "swebench_pro_textbased.yaml").exists()
    else builtin_config_dir / "benchmarks" / "swebench_pro.yaml"
)
# External issue-id list lives outside the repo; resolve from an env var, falling
# back to the repo's parent directory so the default is portable.
DEFAULT_ISSUE_IDS_FILE = Path(
    os.environ.get(
        "SWEBENCH_PRO_ISSUE_IDS",
        str(Path(__file__).resolve().parents[1].parent / "swebench_pro_issue_ids.txt"),
    )
)

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()


def _load_issue_ids(path: Path) -> list[str]:
    issue_ids: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        issue_ids.append(line)
    return issue_ids


def _build_problem_statement(row: dict) -> str:
    problem_statement = (row.get("problem_statement") or "").strip()
    requirements = (row.get("requirements") or "").strip()
    interface = (row.get("interface") or "").strip()
    if requirements or interface:
        return (
            f"{problem_statement}\n\n"
            f"Requirements:\n{requirements}\n\n"
            f"New interfaces introduced:\n{interface}"
        ).strip()
    return problem_statement


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
    dockerhub_tag = (row.get("dockerhub_tag") or "").strip()
    if not dockerhub_tag:
        repo_name = (row.get("repo") or "").strip()
        instance_id = row["instance_id"]
        if not repo_name or "/" not in repo_name:
            raise ValueError(f"Missing dockerhub_tag/repo for instance {instance_id}")
        dockerhub_tag = _fallback_dockerhub_tag(instance_id, repo_name)
    return f"docker.io/{dockerhub_username}/sweap-images:{dockerhub_tag}"


def _build_instance(row: dict, dockerhub_username: str) -> dict:
    return {
        **row,
        "instance_id": row["instance_id"],
        "base_commit": row.get("base_commit", ""),
        "problem_statement": _build_problem_statement(row),
        "image_name": _build_image_name(row, dockerhub_username),
        "repo_name": "app",
    }


def _filter_existing_instances(instances: list[dict], output_dir: Path, redo_existing: bool) -> list[dict]:
    if redo_existing:
        return instances
    preds_path = output_dir / "preds.json"
    if not preds_path.exists():
        return instances
    with _OUTPUT_FILE_LOCK:
        existing = set(json.loads(preds_path.read_text()).keys())
    if existing:
        logger.info(f"Skipping {len(existing)} existing instances from preds.json")
    return [inst for inst in instances if inst["instance_id"] not in existing]


def _validate_issue_id(issue_id: str) -> bool:
    return bool(re.match(r"^instance_[A-Za-z0-9_.-]+$", issue_id))


def _docker_image_preflight(images: list[str], output_dir: Path) -> None:
    """Preflight-check docker images before spending model budget."""
    unique_images = sorted(set(images))
    failures: list[dict[str, str]] = []

    def _smoke_check(image: str) -> tuple[bool, str]:
        smoke = subprocess.run(
            ["docker", "run", "--rm", "--pull=never", "--entrypoint", "/bin/echo", image, "preflight-ok"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        ok = smoke.returncode == 0 and (smoke.stdout or "").strip() == "preflight-ok"
        reason = (smoke.stderr or smoke.stdout or "").strip()[:400]
        return ok, reason

    for idx, image in enumerate(unique_images, start=1):
        logger.info("Preflight [%d/%d] checking image %s", idx, len(unique_images), image)
        try:
            inspect = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if inspect.returncode != 0:
                pull = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                    text=True,
                    timeout=1800,  # large images may take a while
                )
                if pull.returncode != 0:
                    failures.append({
                        "image": image,
                        "reason": (pull.stderr or pull.stdout or "").strip()[:400],
                    })
                    continue
            # Override entrypoint to avoid false negatives on ENTRYPOINT=/bin/bash images.
            ok, reason = _smoke_check(image)
            if not ok:
                # Cached image could be stale/corrupted; force refresh once and retry smoke check.
                logger.warning("Preflight smoke failed, trying re-pull once: %s", image)
                repull = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                if repull.returncode != 0:
                    failures.append({
                        "image": image,
                        "reason": f"smoke failed then repull failed: {(repull.stderr or repull.stdout or '').strip()[:300]}",
                    })
                    continue
                ok, reason = _smoke_check(image)
            if not ok:
                failures.append({
                    "image": image,
                    "reason": reason,
                })
        except FileNotFoundError:
            failures.append({"image": image, "reason": "docker not found in PATH"})
        except subprocess.TimeoutExpired as exc:
            failures.append({"image": image, "reason": f"timeout: {exc}"})
        except Exception as exc:  # defensive: keep preflight robust
            failures.append({"image": image, "reason": repr(exc)})

    if not failures:
        logger.info("Docker preflight passed for %d images", len(unique_images))
        return

    preflight_path = output_dir / "preflight_failures.json"
    preflight_path.write_text(json.dumps(failures, indent=2))
    logger.error(
        "Docker preflight failed for %d/%d images. See %s",
        len(failures),
        len(unique_images),
        preflight_path,
    )
    raise typer.Exit(code=2)


# fmt: off
@app.command()
def main(
    issue_ids_file: Path = typer.Option(DEFAULT_ISSUE_IDS_FILE, "--issue-ids-file", help="One instance_id per line."),
    dataset: str = typer.Option("ScaleAI/SWE-bench_Pro", "--dataset", help="HuggingFace dataset name."),
    split: str = typer.Option("test", "--split", help="Dataset split."),
    output: Path = typer.Option(Path("./outputs/swebench_pro_python49"), "-o", "--output", help="Output directory."),
    workers: int = typer.Option(4, "-w", "--workers", help="Parallel worker count."),
    model: Optional[str] = typer.Option(None, "-m", "--model", help="Model name."),
    model_class: Optional[str] = typer.Option(None, "--model-class", help="Model class."),
    environment_class: Optional[str] = typer.Option("docker", "--environment-class", help="Environment class."),
    dockerhub_username: str = typer.Option("jefzda", "--dockerhub-username", help="Docker Hub username for sweap-images."),
    max_instances: int = typer.Option(0, "--max-instances", help="Only run first N issue IDs. 0 means all."),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Re-run instances already in preds.json."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only validate + print selected instances."),
    preflight: bool = typer.Option(True, "--preflight/--no-preflight", help="Run docker image preflight before evaluation."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_FILE, "-c", "--config", help="Path to swebench config yaml."),
) -> None:
    # fmt: on
    output.mkdir(parents=True, exist_ok=True)
    add_file_handler(output / "minisweagent.log")

    issue_ids = _load_issue_ids(issue_ids_file)
    if max_instances > 0:
        issue_ids = issue_ids[:max_instances]
    if not issue_ids:
        raise typer.BadParameter(f"No instance IDs found in {issue_ids_file}")
    invalid_ids = [iid for iid in issue_ids if not _validate_issue_id(iid)]
    if invalid_ids:
        raise typer.BadParameter(f"Invalid instance_id format, first bad one: {invalid_ids[0]}")

    logger.info(f"Loading dataset {dataset}, split {split}...")
    dataset_rows = list(load_dataset(dataset, split=split))
    by_id = {row["instance_id"]: dict(row) for row in dataset_rows}

    missing_ids = [iid for iid in issue_ids if iid not in by_id]
    selected_ids = [iid for iid in issue_ids if iid in by_id]
    instances = [_build_instance(by_id[iid], dockerhub_username) for iid in selected_ids]
    instances = _filter_existing_instances(instances, output, redo_existing)

    (output / "selected_ids.json").write_text(json.dumps([x["instance_id"] for x in instances], indent=2))
    (output / "missing_ids.json").write_text(json.dumps(missing_ids, indent=2))

    logger.info(
        "Requested %d IDs, selected %d, missing %d, to_run %d",
        len(issue_ids),
        len(selected_ids),
        len(missing_ids),
        len(instances),
    )
    if missing_ids:
        logger.warning("First 5 missing IDs: %s", missing_ids[:5])

    if dry_run:
        logger.info("Dry-run enabled, exiting without running agent.")
        return
    if not instances:
        logger.info("No instances left to run.")
        return
    if preflight and (environment_class or "").lower() == "docker":
        _docker_image_preflight([x["image_name"] for x in instances], output)

    resolved_config_path = get_config_path(config_path)
    logger.info(f"Loading config from '{resolved_config_path}'")
    config = yaml.safe_load(resolved_config_path.read_text()) or {}
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    # Inject reasoning_effort (e.g. "high") from env into model_kwargs, which
    # litellm_textbased forwards to litellm.completion. Empty leaves it unset.
    _re = os.environ.get("REASONING_EFFORT", "").strip()
    if _re:
        config.setdefault("model", {}).setdefault("model_kwargs", {})["reasoning_effort"] = _re
        logger.info(f"Injecting reasoning_effort={_re} into model_kwargs")

    progress_manager = RunBatchProgressManager(len(instances), output / f"exit_statuses_{time.time()}.yaml")

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(process_instance, instance, output, config, progress_manager): instance["instance_id"]
                for instance in instances
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except concurrent.futures.CancelledError:
                    pass
                except Exception as exc:
                    instance_id = futures[future]
                    logger.error(f"Error in future for instance {instance_id}: {exc}", exc_info=True)
                    progress_manager.on_uncaught_exception(instance_id, exc)


if __name__ == "__main__":
    app()
