#!/usr/bin/env python3
"""Repair agent runner.

Independent from ``swebench_pro_baseline``: uses its own prompt template
(``repair_config.yaml``) and its own task template (``task_template.j2``),
consuming the RCA agent's per-instance top-1/top-5 suspicious files as
hints injected into the task string.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import typer
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

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "repair_config.yaml"
DEFAULT_TASK_TEMPLATE = HERE / "task_template.j2"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_LOCK = threading.Lock()


# ── helpers ─────────────────────────────────────────────────────────────────
def _load_issue_ids(path: Path) -> list[str]:
    return [
        l.strip()
        for l in path.read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]


_TEST_PATH_RE = re.compile(
    r"(^|/)(_repro_tests/|tests?/|conftest\.py$|test_[^/]*\.py$|[^/]*_test\.py$)"
)


def _is_test_file(path: str) -> bool:
    """True if the path looks like a test file (and so is a poisoned RCA candidate).

    Coverage-based RCA frequently ranks the repro test itself at top-1 because
    the failing assertion fires inside the test. Those entries actively
    mislead the repair agent, so we drop them before any downstream use.
    """
    if not path:
        return True
    return bool(_TEST_PATH_RE.search(path.replace("\\", "/")))


def _filter_rca_candidates(rca_data: list) -> tuple[list, bool]:
    """Drop test-file candidates from every entry's candidate list.

    Returns (filtered_data, any_top1_dropped). ``any_top1_dropped`` means at
    least one entry originally had a test file at rank 1 — a signal that the
    remaining candidates are lower-confidence even after filtering.
    """
    filtered: list = []
    any_top1_dropped = False
    for entry in rca_data or []:
        cands = entry.get("candidates") or []
        if cands and _is_test_file(cands[0].get("file") or ""):
            any_top1_dropped = True
        kept = [c for c in cands if not _is_test_file(c.get("file") or "")]
        if not kept:
            continue
        new_entry = dict(entry)
        new_entry["candidates"] = kept
        filtered.append(new_entry)
    return filtered, any_top1_dropped


def _classify_rca_confidence(
    rca_data: Optional[list], any_top1_dropped: bool
) -> str:
    """Bucket RCA quality into high / low / missing for the task template.

    - ``missing``: no RCA at all, or everything was filtered out as test files.
    - ``low``:     at least one entry originally had a test-file top-1, or the
                   remaining top candidate has a weak score.
    - ``high``:    otherwise.
    """
    if not rca_data:
        return "missing"
    if any_top1_dropped:
        return "low"
    top_score = 0.0
    for entry in rca_data:
        for c in entry.get("candidates") or []:
            try:
                top_score = max(top_score, float(c.get("score") or 0))
            except (TypeError, ValueError):
                pass
            break
    if top_score < 0.3:
        return "low"
    return "high"


def _load_rca(rca_dir: Path, instance_id: str):
    """Return the parsed RCA JSON.

    Two schemas co-exist:
      - Legacy static RCA — list[ {test_nodeid, candidates, …} ]
      - debug_agent — dict {source: "debug_agent", candidates, debug_report, rich_rca?}
    Callers must branch on type.
    """
    p = rca_dir / f"{instance_id}_rca.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logger.warning("Failed to parse RCA file %s: %s", p, exc)
        return None


def _is_debug_agent_rca(rca_json) -> bool:
    return isinstance(rca_json, dict) and rca_json.get("source") == "debug_agent"


def _debug_transcript_tail(transcript, n: int = 6, tail_chars: int = 800) -> str:
    if not transcript:
        return ""
    out: list[str] = []
    for item in list(transcript)[-n:]:
        try:
            name, payload, output = item[0], item[1], item[2]
        except Exception:
            continue
        payload_s = str(payload) if payload is not None else ""
        output_s = (str(output) if output is not None else "")[-tail_chars:]
        out.append(f"[{name}] {payload_s}\n{output_s}")
    return "\n".join(out)


def _extract_debug_context(rca_json: dict) -> dict:
    """Flatten the debug_agent _rca.json payload into template ctx fields."""
    dbg = rca_json.get("debug_report") or {}
    ctx: dict = {
        "debug_reasoning": (dbg.get("reasoning") or "").strip(),
        "debug_suggested_fix": (dbg.get("suggested_fix") or "").strip(),
        "debug_root_cause_files": list(dbg.get("root_cause_files") or []),
        "debug_root_cause_functions": list(dbg.get("root_cause_functions") or []),
        "debug_transcript_tail": _debug_transcript_tail(dbg.get("transcript") or []),
        "debug_timed_out": bool(dbg.get("timed_out")),
    }
    rich = rca_json.get("rich_rca") or {}
    if rich and isinstance(rich, dict):
        ctx["rich_rca"] = rich
        symptom = rich.get("symptom") or {}
        if symptom.get("status") != "unavailable":
            ctx["rich_symptom"] = {
                "file": symptom.get("file", ""),
                "lineno": symptom.get("lineno", 0),
                "exc_type": symptom.get("exc_type", ""),
                "exc_msg": symptom.get("exc_msg", ""),
            }
        contract = rich.get("contract_impact") or {}
        if contract.get("changed"):
            ctx["rich_contract_impact"] = {
                "kind": contract.get("kind", ""),
                "summary": contract.get("summary", ""),
                "callers": list(contract.get("callers") or [])[:15],
                "callers_truncated": bool(contract.get("callers_truncated")),
            }
        prop = (rich.get("propagation_path") or {}).get("frames") or []
        if prop:
            trimmed = []
            for fr in prop[-6:]:
                trimmed.append({
                    "file": fr.get("file", ""),
                    "lineno": fr.get("lineno", 0),
                    "qualname": fr.get("qualname", ""),
                    "role": fr.get("role", ""),
                })
            ctx["rich_propagation_frames"] = trimmed
        # NOTE: rich_rca.related_non_code is intentionally NOT lifted into ctx.
        # The 2026-04-30 lever ablation confirmed it is net-negative on the
        # 49-instance benchmark (Δ=+4 instances when dropped). Stage2 keeps
        # computing it in rca.json so future work can re-enable with stricter
        # filtering (path-aware / recently-modified / cap).
        # See plans_20260430/2026-04-30-non-code-lever-ablation-findings.md
        # Lift PDB-evidence sections to top-level ctx — the template references
        # `rich_branch_observations` / `rich_pdb_anomalies` directly, and Jinja
        # treats undefined names as falsy, so missing this lift silently skips
        # the entire "── PDB session evidence ──" block.
        branch_obs = rich.get("rich_branch_observations") or []
        if branch_obs:
            ctx["rich_branch_observations"] = list(branch_obs)
        pdb_anom = rich.get("rich_pdb_anomalies") or []
        if pdb_anom:
            ctx["rich_pdb_anomalies"] = list(pdb_anom)
    else:
        ctx["rich_rca"] = None
    return ctx


def _debug_agent_is_empty(debug_ctx: dict) -> bool:
    """Debug-agent timed out without producing anything actionable.

    Matches plan X2: render baseline-style prompt (problem statement only).
    """
    if not debug_ctx.get("debug_timed_out"):
        return False
    return not (
        debug_ctx.get("debug_reasoning")
        or debug_ctx.get("debug_suggested_fix")
        or debug_ctx.get("debug_root_cause_files")
    )


def _load_repro_localizer(
    repro_dir: Optional[Path], instance_id: str,
) -> tuple[list[str], list[str]]:
    """Load (relevant_files, focal_functions) produced by stage1's localizer.

    These are an INDEPENDENT signal from the coverage-based RCA top-K. We
    propagate them so the repair agent can spot disagreements between the two
    sources and search beyond what the failing test happened to execute.
    """
    if repro_dir is None:
        return [], []
    p = repro_dir / instance_id / f"{instance_id}.json"
    if not p.exists():
        return [], []
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        logger.warning("Failed to parse stage1 file %s: %s", p, exc)
        return [], []
    loc = data.get("localization") or {}
    files = [f for f in (loc.get("relevant_files") or []) if isinstance(f, str)]
    funcs = [f for f in (loc.get("focal_functions") or []) if isinstance(f, str)]
    return files, funcs


def _extract_top_files(rca_data: list) -> tuple[list[str], list[str]]:
    """Return (top1_files, top5_files) — unique, rank-ordered, aggregated
    across all failing-test entries for the instance."""
    top1: list[str] = []
    top5: list[str] = []
    for entry in rca_data or []:
        cands = entry.get("candidates") or []
        if not cands:
            continue
        f0 = cands[0].get("file")
        if f0 and f0 not in top1:
            top1.append(f0)
        seen = set()
        for c in cands:
            f = c.get("file")
            if not f or f in seen:
                continue
            seen.add(f)
            if f not in top5:
                top5.append(f)
            if len(seen) >= 5:
                break
    return top1, top5


def _format_rca_evidence(rca_data: list, max_entries: int = 1, max_cands: int = 3) -> str:
    """Compact RCA evidence. Kept intentionally small to avoid anchoring the
    repair agent on a single candidate — we drop source snippets and cap
    entries/candidates hard."""
    lines: list[str] = []
    for entry in (rca_data or [])[:max_entries]:
        test_id = entry.get("test_nodeid", "")
        err = f'{entry.get("error_type", "")}: {entry.get("error_message", "")}'.strip(": ")
        lines.append(f"- Failing test: {test_id}")
        if err:
            lines.append(f"  Error: {err}")
        for c in (entry.get("candidates") or [])[:max_cands]:
            lines.append(
                f"  * {c.get('file')}:{c.get('line')} "
                f"[{c.get('func_name', '')}] score={c.get('score', 0):.2f}"
            )
    return "\n".join(lines)


def _render_task(
    task_template: Template,
    row: dict,
    rca_data,
    repro_files: Optional[list[str]] = None,
    repro_funcs: Optional[list[str]] = None,
) -> str:
    repro_files = repro_files or []
    repro_funcs = repro_funcs or []

    base_ctx = dict(
        problem_statement=(row.get("problem_statement") or "").strip(),
        requirements=(row.get("requirements") or "").strip(),
        interface=(row.get("interface") or "").strip(),
        repro_localizer_files=repro_files,
        repro_localizer_funcs=repro_funcs,
        # Legacy static-RCA template fields — keyed off rca_confidence below.
        rca_top1_files=[], rca_top5_files=[], rca_evidence="",
        rca_confidence="missing", repro_only_files=[],
        # Debug-agent fields — empty by default; filled only on debug_agent RCA.
        debug_reasoning="", debug_suggested_fix="",
        debug_root_cause_files=[], debug_root_cause_functions=[],
        debug_transcript_tail="", debug_timed_out=False,
        rich_symptom=None, rich_contract_impact=None,
        rich_propagation_frames=None, rich_related_non_code=None,
        rich_branch_observations=None, rich_pdb_anomalies=None,
    )

    if _is_debug_agent_rca(rca_data):
        debug_ctx = _extract_debug_context(rca_data)
        if _debug_agent_is_empty(debug_ctx):
            # X2: degrade to baseline prompt on debug-agent timeout w/ no signal.
            return task_template.render(**base_ctx).strip()
        base_ctx.update(debug_ctx)
        # repro_only_files is only meaningful against legacy top5 — leave empty.
        return task_template.render(**base_ctx).strip()

    # Legacy static-RCA path (list of per-test entries).
    filtered_rca, any_top1_dropped = _filter_rca_candidates(rca_data or [])
    rca_confidence = _classify_rca_confidence(filtered_rca, any_top1_dropped)
    top1, top5 = _extract_top_files(filtered_rca)
    evidence = _format_rca_evidence(filtered_rca)
    rca_set = {f for f in top5 if f}
    repro_only = [f for f in repro_files if f and f not in rca_set]

    base_ctx.update(
        rca_top1_files=top1,
        rca_top5_files=top5,
        rca_evidence=evidence,
        rca_confidence=rca_confidence,
        repro_only_files=repro_only,
    )
    return task_template.render(**base_ctx).strip()


# ── image name helpers (same convention as baseline) ───────────────────────
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
    return f"{repo_base}.{repo_name_only}-{hsh}"[:128]


def _build_image_name(row: dict, dockerhub_username: str) -> str:
    tag = (row.get("dockerhub_tag") or "").strip()
    if not tag:
        repo_name = (row.get("repo") or "").strip()
        if not repo_name or "/" not in repo_name:
            raise ValueError(f"Missing dockerhub_tag/repo for instance {row['instance_id']}")
        tag = _fallback_dockerhub_tag(row["instance_id"], repo_name)
    return f"docker.io/{dockerhub_username}/sweap-images:{tag}"


def _build_instance(
    row: dict,
    dockerhub_username: str,
    rca_dir: Optional[Path],
    task_template: Template,
    repro_dir: Optional[Path] = None,
) -> dict:
    iid = row["instance_id"]
    rca_data = _load_rca(rca_dir, iid) if rca_dir else None
    repro_files, repro_funcs = _load_repro_localizer(repro_dir, iid)
    return {
        **row,
        "instance_id": iid,
        "base_commit": row.get("base_commit", ""),
        "problem_statement": _render_task(
            task_template, row, rca_data, repro_files, repro_funcs,
        ),
        "image_name": _build_image_name(row, dockerhub_username),
        "repo_name": "app",
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


def _valid_id(iid: str) -> bool:
    return bool(re.match(r"^instance_[A-Za-z0-9_.-]+$", iid))


# ── CLI ─────────────────────────────────────────────────────────────────────
# fmt: off
@app.command()
def main(
    issue_ids_file: Path = typer.Option(..., "--issue-ids-file"),
    rca_dir: Path = typer.Option(..., "--rca-dir", help="Dir with instance_<id>_rca.json files."),
    repro_dir: Optional[Path] = typer.Option(None, "--repro-dir", help="Stage1 reproduction output dir; used to read localizer hints (relevant_files) for cross-checking RCA."),
    output: Path = typer.Option(..., "-o", "--output"),
    dataset: str = typer.Option("ScaleAI/SWE-bench_Pro", "--dataset"),
    split: str = typer.Option("test", "--split"),
    workers: int = typer.Option(4, "-w", "--workers"),
    model: Optional[str] = typer.Option(None, "-m", "--model"),
    dockerhub_username: str = typer.Option("jefzda", "--dockerhub-username"),
    max_instances: int = typer.Option(0, "--max-instances"),
    redo_existing: bool = typer.Option(False, "--redo-existing"),
    require_rca: bool = typer.Option(False, "--require-rca", help="Skip instances without RCA output."),
    config_path: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config"),
    task_template_path: Path = typer.Option(DEFAULT_TASK_TEMPLATE, "--task-template"),
) -> None:
    # fmt: on
    output.mkdir(parents=True, exist_ok=True)
    add_file_handler(output / "repair_agent.log")

    if not rca_dir.exists():
        raise typer.BadParameter(f"RCA dir not found: {rca_dir}")
    if not config_path.exists():
        raise typer.BadParameter(f"Config not found: {config_path}")
    if not task_template_path.exists():
        raise typer.BadParameter(f"Task template not found: {task_template_path}")

    task_template = Template(task_template_path.read_text())

    issue_ids = _load_issue_ids(issue_ids_file)
    if max_instances > 0:
        issue_ids = issue_ids[:max_instances]
    if not issue_ids:
        raise typer.BadParameter(f"No instance IDs in {issue_ids_file}")
    bad = [i for i in issue_ids if not _valid_id(i)]
    if bad:
        raise typer.BadParameter(f"Invalid instance_id format: {bad[0]}")

    logger.info(f"Loading dataset {dataset}/{split}...")
    rows = {r["instance_id"]: dict(r) for r in load_dataset(dataset, split=split)}

    selected = [i for i in issue_ids if i in rows]
    missing = [i for i in issue_ids if i not in rows]
    instances = [
        _build_instance(rows[i], dockerhub_username, rca_dir, task_template, repro_dir)
        for i in selected
    ]

    if require_rca:
        before = len(instances)
        instances = [i for i in instances if i.get("_rca_available")]
        logger.info("require_rca: kept %d/%d", len(instances), before)

    instances = _filter_existing(instances, output, redo_existing)
    with_rca = sum(1 for i in instances if i.get("_rca_available"))
    logger.info(
        "Requested %d, selected %d, missing %d, with_rca %d, to_run %d",
        len(issue_ids), len(selected), len(missing), with_rca, len(instances),
    )

    (output / "selected_ids.json").write_text(json.dumps([i["instance_id"] for i in instances], indent=2))
    (output / "missing_ids.json").write_text(json.dumps(missing, indent=2))

    if not instances:
        logger.info("Nothing to run.")
        return

    # Dump one rendered task per instance for offline inspection/debugging.
    debug_dir = output / "rendered_tasks"
    debug_dir.mkdir(exist_ok=True)
    for inst in instances:
        (debug_dir / f"{inst['instance_id']}.txt").write_text(inst["problem_statement"])

    for inst in instances:
        inst.pop("_rca_available", None)

    logger.info(f"Loading repair-agent config from {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    # Inject reasoning_effort (e.g. "high") from env when set; merges into the
    # model_kwargs that litellm_textbased forwards to litellm.completion.
    _re = os.environ.get("REASONING_EFFORT", "").strip()
    if _re:
        config.setdefault("model", {}).setdefault("model_kwargs", {})["reasoning_effort"] = _re

    progress_manager = RunBatchProgressManager(len(instances), output / f"exit_statuses_{time.time()}.yaml")

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {
                ex.submit(process_instance, inst, output, config, progress_manager): inst["instance_id"]
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
    app()
