#!/usr/bin/env python3
"""Standalone CLI to run statement-level root cause analysis on existing results.

Usage:
    # Analyze all Python instances in a results directory
    python run_statement_rca.py --results-dir results/results_xxx

    # Analyze a single instance
    python run_statement_rca.py --instance-dir results/results_xxx/instance_foo

    # With source code (enables AST slicing — Step 3)
    python run_statement_rca.py --results-dir results/results_xxx --repos-dir /tmp/repos

    # With dataset for patch info (enables ground-truth evaluation)
    python run_statement_rca.py --results-dir results/results_xxx --dataset ScaleAI/SWE-bench_Pro
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

# Ensure run_pro_test directory is on path
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    extract_patch_files,
    load_execution_files,
    normalize_path,
    strip_ansi,
)
from statement_tracer import (
    StatementTrace,
    build_statement_traces,
    load_focused_lines_per_test,
    parse_coverage_json,
)
from context_extractor import (
    FailureContext,
    SourceReader,
    build_failure_contexts,
    extract_context_from_coverage,
    make_repo_source_reader,
)
from root_cause_analyzer import (
    RCAResult,
    RootCauseCandidate,
    aggregate_parametrized,
    extract_patch_changed_lines,
    run_rca_pipeline,
    save_rca_results,
)


# ── Detect language from patch ───────────────────────────────────────────────

_GO_EXTS = {".go"}
_JS_EXTS = {".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"}


def _is_python_instance(patch_text: str, instance_id: str) -> bool:
    """Return True if this looks like a Python instance (not Go/JS)."""
    files = extract_patch_files(patch_text)
    exts = {Path(f).suffix.lower() for f in files}
    if exts & _GO_EXTS or exts & _JS_EXTS:
        return False
    return True


# ── Failed test extraction (from existing artifacts) ─────────────────────────

def _load_failed_tests(instance_dir: Path, workspace: Path) -> list[str]:
    """Load failed test node IDs from available artifacts."""
    # Try failed_nodeids.json first (written by Phase 2 in entryscript)
    nodeids_path = workspace / "failed_nodeids.json"
    if nodeids_path.exists():
        try:
            nodeids = json.loads(nodeids_path.read_text(errors="replace"))
            if isinstance(nodeids, list) and nodeids:
                return nodeids
        except Exception:
            pass

    # Parse from stdout/stderr using python_runner's extract_failed_tests
    try:
        from python_runner import extract_failed_tests
        stdout = ""
        stderr = ""
        for name in ("stdout.log", "rerun_stdout.log"):
            p = workspace / name
            if p.exists():
                stdout += p.read_text(errors="replace") + "\n"
            p2 = instance_dir / name
            if p2.exists():
                stdout += p2.read_text(errors="replace") + "\n"
        for name in ("stderr.log", "rerun_stderr.log"):
            p = workspace / name
            if p.exists():
                stderr += p.read_text(errors="replace") + "\n"
            p2 = instance_dir / name
            if p2.exists():
                stderr += p2.read_text(errors="replace") + "\n"
        return extract_failed_tests(stdout, stderr)
    except Exception:
        return []


def _load_failure_text(instance_dir: Path, workspace: Path) -> str:
    """Load combined failure text from all available sources."""
    parts: list[str] = []
    for name in ("failure_trace.log", "stdout.log", "stderr.log",
                  "rerun_stdout.log", "rerun_stderr.log",
                  "focused_stdout.log", "focused_stderr.log"):
        for base in (instance_dir, workspace):
            p = base / name
            if p.exists():
                try:
                    parts.append(p.read_text(errors="replace"))
                except Exception:
                    pass
                break
    return "\n".join(parts)


# ── Source code access ───────────────────────────────────────────────────────

def _checkout_repo(repo: str, base_commit: str, repos_dir: Path) -> Path | None:
    """Clone/checkout a repo at a specific commit for source access."""
    safe_name = repo.replace("/", "__")
    repo_path = repos_dir / f"{safe_name}_{base_commit[:12]}"
    if repo_path.exists() and (repo_path / ".git").exists():
        return repo_path

    repo_path.mkdir(parents=True, exist_ok=True)
    try:
        url = f"https://github.com/{repo}.git"
        subprocess.run(
            ["git", "clone", "--depth=1", "--single-branch", url, str(repo_path)],
            capture_output=True, timeout=120,
        )
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", base_commit],
            cwd=str(repo_path), capture_output=True, timeout=60,
        )
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=str(repo_path), capture_output=True, timeout=30,
        )
        return repo_path
    except Exception:
        return None


def extract_source_from_docker(
    instance_dir: Path,
    image: str,
    base_commit: str,
    files_to_extract: list[str],
    timeout: int = 120,
) -> Path | None:
    """Extract source files from a Docker image into a source_snapshot directory.

    Used for offline RCA on existing results that don't have a snapshot yet.
    Spins up a temporary container, does git checkout, copies files out.
    """
    snapshot_dir = instance_dir / "source_snapshot"
    if snapshot_dir.exists() and any(snapshot_dir.rglob("*.py")):
        return snapshot_dir  # already extracted

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Build a minimal script that checks out the commit and cats each file
    # We use docker cp instead to be more efficient
    import uuid
    container_name = f"rca_src_{uuid.uuid4().hex[:8]}"

    try:
        # Start container in background
        proc = subprocess.run(
            ["docker", "run", "-d", "--name", container_name,
             "--entrypoint", "/bin/bash", image,
             "-c", f"cd /app && git checkout -f {base_commit} 2>/dev/null; sleep 300"],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            return None

        # Wait a moment for checkout to complete
        subprocess.run(
            ["docker", "exec", container_name, "bash", "-c",
             f"cd /app && git checkout -f {base_commit} 2>/dev/null || true"],
            capture_output=True, timeout=30,
        )

        # Copy each file from container
        for rel_path in files_to_extract:
            container_path = f"/app/{rel_path}"
            local_path = snapshot_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["docker", "cp", f"{container_name}:{container_path}", str(local_path)],
                capture_output=True, timeout=10,
            )

        # Write manifest
        manifest = sorted(
            str(p.relative_to(snapshot_dir))
            for p in snapshot_dir.rglob("*.py")
        )
        (snapshot_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))

        return snapshot_dir if manifest else None

    except Exception:
        return None
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True, timeout=15,
            )
        except Exception:
            pass


def extract_source_for_instance(
    instance_dir: Path,
    sample: dict | None = None,
) -> Path | None:
    """Extract source snapshot for an instance using Docker, if not already present.

    Determines which files to extract from coverage JSON + traceback data.
    """
    workspace = instance_dir / "workspace"
    if not workspace.exists():
        workspace = instance_dir

    # Already have snapshot?
    for base in (workspace, instance_dir):
        snapshot = base / "source_snapshot"
        if snapshot.exists() and any(snapshot.rglob("*.py")):
            return snapshot

    # Need image + base_commit
    if not sample:
        return None
    image = f"jefzda/sweap-images:{sample.get('dockerhub_tag', '')}"
    base_commit = sample.get("base_commit", "")
    if not base_commit:
        return None

    # Collect files to extract from coverage + tracebacks
    files_needed: set[str] = set()

    # From coverage JSON
    cov_path = workspace / "phase2_coverage.json"
    if not cov_path.exists():
        cov_path = instance_dir / "phase2_coverage.json"
    if cov_path.exists():
        try:
            data = json.loads(cov_path.read_text(errors="replace"))
            for abs_path in data.get("files", {}).keys():
                abs_path = abs_path.replace("\\", "/")
                if "/app/" in abs_path:
                    low = abs_path.lower()
                    if "/site-packages/" not in low and "/dist-packages/" not in low:
                        rel = abs_path[abs_path.find("/app/") + 5:].lstrip("./")
                        if rel.endswith(".py"):
                            files_needed.add(rel)
        except Exception:
            pass

    # From focused execution files
    for log_name in ("focused_execution_files.log",):
        for base in (instance_dir, workspace):
            log_path = base / log_name
            if log_path.exists():
                for line in log_path.read_text(errors="replace").splitlines():
                    line = line.strip()
                    if line.endswith(".py"):
                        files_needed.add(line)

    # From traceback in failure_trace.log
    import re
    _FILE_RE = re.compile(r'File "(/app/[^"]+\.py)"')
    _LOC_RE = re.compile(r'^(\S+\.py):\d+:', re.MULTILINE)
    for log_name in ("failure_trace.log", "stdout.log"):
        for base in (instance_dir, workspace):
            log_path = base / log_name
            if log_path.exists():
                text = log_path.read_text(errors="replace")
                for m in _FILE_RE.finditer(text):
                    rel = m.group(1)[5:]  # strip /app/
                    files_needed.add(rel)
                for m in _LOC_RE.finditer(text):
                    files_needed.add(m.group(1))

    if not files_needed:
        return None

    return extract_source_from_docker(
        instance_dir, image, base_commit,
        sorted(files_needed), timeout=120,
    )


# ── Single instance analysis ─────────────────────────────────────────────────

def _find_source_reader(
    instance_dir: Path,
    workspace: Path,
    repo: str = "",
    base_commit: str = "",
    repos_dir: Path | None = None,
    explicit_reader: SourceReader | None = None,
    sample: dict | None = None,
) -> SourceReader | None:
    """Find the best available source reader for an instance.

    Priority:
    1. Explicitly provided reader
    2. Source snapshot from Docker (workspace/source_snapshot or instance_dir/source_snapshot)
    3. Docker extraction (spin up container, copy files out)
    4. Local repo clone from GitHub (if repos_dir provided)
    """
    if explicit_reader:
        return explicit_reader

    # Check for source snapshot (extracted by entryscript from Docker)
    for base in (workspace, instance_dir):
        snapshot = base / "source_snapshot"
        if snapshot.exists() and any(snapshot.rglob("*.py")):
            return make_repo_source_reader(snapshot)

    # Try Docker extraction for existing results
    if sample:
        snapshot = extract_source_for_instance(instance_dir, sample)
        if snapshot:
            return make_repo_source_reader(snapshot)

    # Fall back to cloning repo from GitHub
    if repos_dir and repo and base_commit:
        repo_path = _checkout_repo(repo, base_commit, repos_dir)
        if repo_path:
            return make_repo_source_reader(repo_path)

    return None


def analyze_instance(
    instance_dir: Path,
    patch_text: str = "",
    repo: str = "",
    base_commit: str = "",
    source_reader: SourceReader | None = None,
    repos_dir: Path | None = None,
    sample: dict | None = None,
    enable_phase3: bool = False,
    phase3_model: str | None = None,
) -> list[RCAResult] | None:
    """Run RCA on a single instance directory. Returns None if not applicable."""
    workspace = instance_dir / "workspace"
    if not workspace.exists():
        workspace = instance_dir  # some layouts put files directly in instance_dir

    # Check for coverage data (Python indicator)
    has_coverage = (workspace / "phase2_coverage.json").exists()
    has_stdout = (workspace / "stdout.log").exists() or (instance_dir / "stdout.log").exists()
    if not has_coverage and not has_stdout:
        return None

    # Load patch files
    patch_files = extract_patch_files(patch_text)
    if not patch_files and (instance_dir / "patch.diff").exists():
        patch_text = (instance_dir / "patch.diff").read_text(errors="replace")
        patch_files = extract_patch_files(patch_text)

    # Skip non-Python
    if patch_files and not _is_python_instance(patch_text, instance_dir.name):
        return None

    # Load failed tests
    failed_tests = _load_failed_tests(instance_dir, workspace)
    if not failed_tests:
        return None

    # Load failure text
    failure_text = _load_failure_text(instance_dir, workspace)

    # JUnit XML
    junit_path = workspace / "pytest-report.xml"
    if not junit_path.exists():
        junit_path = instance_dir / "pytest-report.xml"

    # Resolve source reader: snapshot > docker extraction > clone > None
    resolved_reader = _find_source_reader(
        instance_dir, workspace,
        repo=repo, base_commit=base_commit,
        repos_dir=repos_dir, explicit_reader=source_reader,
        sample=sample,
    )

    # Build statement traces (Step 1)
    traces = build_statement_traces(
        workspace=workspace,
        failure_text=failure_text,
        failed_tests=failed_tests,
        junit_xml_path=junit_path if junit_path.exists() else None,
    )
    if not traces:
        return None

    # Build failure contexts (Step 3)
    contexts = build_failure_contexts(traces, resolved_reader)

    # Run RCA pipeline (Step 4)
    results = run_rca_pipeline(
        traces=traces,
        contexts=contexts,
        patch_files=patch_files,
        patch_text=patch_text,
        source_reader=resolved_reader,
    )

    # Phase 3: LLM RCA agent refinement (conditional)
    if enable_phase3 and results:
        from root_cause_analysis_agent_integration import run_phase3_refinement

        # Find source snapshot path
        snapshot_path = None
        for base in (workspace, instance_dir):
            snap = base / "source_snapshot"
            if snap.exists() and any(snap.rglob("*.py")):
                snapshot_path = snap
                break

        if snapshot_path:
            trajectory_dir = instance_dir / "phase3_trajectories"
            results = run_phase3_refinement(
                results=results,
                traces=traces,
                contexts=contexts,
                snapshot_path=snapshot_path,
                instance_dir=instance_dir,
                workspace=workspace,
                model_name=phase3_model,
                trajectory_dir=trajectory_dir,
                patch_files=patch_files,
                patch_text=patch_text,
            )

    return results


def _upgrade_traces_with_line_data(
    traces: list[StatementTrace],
    focused_lines: dict[str, dict[str, list[int]]],
) -> None:
    """Upgrade traces with per-test line-level data from Step 2 settrace output."""
    focused_keys = list(focused_lines.keys())
    for trace in traces:
        safe = "".join(
            c if c.isalnum() or c in "._-" else "_" for c in trace.test_nodeid
        )[:120]
        for key in focused_keys:
            if safe == key or safe in key or key in safe:
                trace.per_test_executed_lines = focused_lines[key]
                break


# ── Batch analysis ───────────────────────────────────────────────────────────

def _analyze_one_instance(
    instance_dir: Path,
    dataset_samples: dict[str, dict] | None,
    repos_dir: Path | None,
    enable_phase3: bool,
    phase3_model: str | None,
) -> tuple[str, list[RCAResult] | None]:
    """Analyze a single instance. Returns (instance_id, results or None)."""
    instance_id = instance_dir.name
    patch_text = ""
    repo = ""
    base_commit = ""
    matched_sample = None
    if dataset_samples:
        for sid, sample in dataset_samples.items():
            if instance_id.endswith(sid) or sid in instance_id:
                patch_text = sample.get("patch", "")
                repo = sample.get("repo", "")
                base_commit = sample.get("base_commit", "")
                matched_sample = sample
                break

    results = analyze_instance(
        instance_dir,
        patch_text=patch_text,
        repo=repo,
        base_commit=base_commit,
        repos_dir=repos_dir,
        sample=matched_sample,
        enable_phase3=enable_phase3,
        phase3_model=phase3_model,
    )
    return instance_id, results


def _load_cached_rca(output_dir: Path, instance_id: str) -> list[RCAResult] | None:
    """Load previously saved RCA results for resume support."""
    rca_file = output_dir / f"{instance_id}_rca.json"
    if not rca_file.exists():
        return None
    try:
        import json as _json
        data = _json.loads(rca_file.read_text())
        results = []
        for entry in data:
            results.append(RCAResult(
                test_nodeid=entry["test_nodeid"],
                error_type=entry.get("error_type", ""),
                error_message=entry.get("error_message", ""),
                candidates=[RootCauseCandidate(**c) for c in entry.get("candidates", [])],
                patch_files=entry.get("patch_files", []),
                top1_hit=entry.get("top1_hit", False),
                top3_hit=entry.get("top3_hit", False),
                top5_hit=entry.get("top5_hit", False),
                top1_line_hit=entry.get("top1_line_hit", False),
                top5_line_hit=entry.get("top5_line_hit", False),
            ))
        return results
    except Exception:
        return None


def analyze_results_dir(
    results_dir: Path,
    dataset_samples: dict[str, dict] | None = None,
    repos_dir: Path | None = None,
    enable_phase3: bool = False,
    phase3_model: str | None = None,
    num_workers: int = 1,
    output_dir: Path | None = None,
) -> dict[str, list[RCAResult]]:
    """Analyze all Python instances in a results directory."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_results: dict[str, list[RCAResult]] = {}
    instance_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("instance_")
    )

    total = len(instance_dirs)
    analyzed = 0
    skipped = 0
    cached = 0

    if num_workers <= 1:
        # Serial path
        for i, instance_dir in enumerate(instance_dirs, 1):
            instance_id = instance_dir.name
            if output_dir:
                prev = _load_cached_rca(output_dir, instance_id)
                if prev is not None:
                    cached += 1
                    all_results[instance_id] = prev
                    n_candidates = sum(len(r.candidates) for r in prev)
                    any_hit = any(r.top1_hit for r in prev)
                    print(
                        f"[{i}/{total}] {instance_id}: "
                        f"{len(prev)} tests, {n_candidates} candidates, "
                        f"top1_hit={any_hit} (cached)"
                    )
                    continue

            instance_id, results = _analyze_one_instance(
                instance_dir, dataset_samples, repos_dir,
                enable_phase3, phase3_model,
            )
            if results is None:
                skipped += 1
                continue
            analyzed += 1
            all_results[instance_id] = results
            n_candidates = sum(len(r.candidates) for r in results)
            any_hit = any(r.top1_hit for r in results)
            print(
                f"[{i}/{total}] {instance_id}: "
                f"{len(results)} tests, {n_candidates} candidates, "
                f"top1_hit={any_hit}"
            )
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                save_rca_results(results, output_dir / f"{instance_id}_rca.json")
    else:
        print(f"Running with {num_workers} parallel workers")

        # Pre-load cached results
        pending_dirs = []
        for instance_dir in instance_dirs:
            instance_id = instance_dir.name
            if output_dir:
                prev = _load_cached_rca(output_dir, instance_id)
                if prev is not None:
                    cached += 1
                    all_results[instance_id] = prev
                    continue
            pending_dirs.append(instance_dir)

        if cached:
            print(f"Resumed {cached} cached results, {len(pending_dirs)} remaining")

        futures = {}
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for instance_dir in pending_dirs:
                fut = pool.submit(
                    _analyze_one_instance,
                    instance_dir, dataset_samples, repos_dir,
                    enable_phase3, phase3_model,
                )
                futures[fut] = instance_dir

            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                instance_id, results = fut.result()
                if results is None:
                    skipped += 1
                    continue
                analyzed += 1
                all_results[instance_id] = results
                n_candidates = sum(len(r.candidates) for r in results)
                any_hit = any(r.top1_hit for r in results)
                print(
                    f"[{done_count}/{len(pending_dirs)}] {instance_id}: "
                    f"{len(results)} tests, {n_candidates} candidates, "
                    f"top1_hit={any_hit}"
                )
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    save_rca_results(results, output_dir / f"{instance_id}_rca.json")

    print(f"\nDone: {analyzed} analyzed, {cached} cached, {skipped} skipped (non-Python or no data)")
    return all_results


# ── Summary reporting ────────────────────────────────────────────────────────

def write_rca_summary(
    all_results: dict[str, list[RCAResult]],
    output_dir: Path,
) -> None:
    """Write summary statistics and per-instance RCA results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-instance results
    for instance_id, results in all_results.items():
        save_rca_results(results, output_dir / f"{instance_id}_rca.json")

    # Aggregate statistics
    total_tests = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top1_line_hits = 0
    top5_line_hits = 0
    instances_with_hit = 0

    for instance_id, results in all_results.items():
        instance_hit = False
        for r in results:
            total_tests += 1
            if r.top1_hit:
                top1_hits += 1
            if r.top3_hit:
                top3_hits += 1
            if r.top5_hit:
                top5_hits += 1
                instance_hit = True
            if r.top1_line_hit:
                top1_line_hits += 1
            if r.top5_line_hit:
                top5_line_hits += 1
        if instance_hit:
            instances_with_hit += 1

    # Aggregated metrics (parametrized tests collapsed)
    agg_total = 0
    agg_top1 = 0
    agg_top5 = 0
    agg_instances_with_hit = 0
    for instance_id, results in all_results.items():
        agg = aggregate_parametrized(results)
        agg_total += len(agg)
        agg_instance_hit = False
        for r in agg:
            if r.top1_hit:
                agg_top1 += 1
            if r.top5_hit:
                agg_top5 += 1
                agg_instance_hit = True
        if agg_instance_hit:
            agg_instances_with_hit += 1

    summary = {
        "total_instances": len(all_results),
        "total_failing_tests": total_tests,
        "top1_file_hit": top1_hits,
        "top3_file_hit": top3_hits,
        "top5_file_hit": top5_hits,
        "top1_line_hit": top1_line_hits,
        "top5_line_hit": top5_line_hits,
        "top1_file_hit_rate": round(top1_hits / total_tests, 4) if total_tests else 0,
        "top3_file_hit_rate": round(top3_hits / total_tests, 4) if total_tests else 0,
        "top5_file_hit_rate": round(top5_hits / total_tests, 4) if total_tests else 0,
        "top1_line_hit_rate": round(top1_line_hits / total_tests, 4) if total_tests else 0,
        "top5_line_hit_rate": round(top5_line_hits / total_tests, 4) if total_tests else 0,
        "instance_hit_rate": round(instances_with_hit / len(all_results), 4) if all_results else 0,
        # Aggregated (parametrized tests collapsed to unique base tests)
        "agg_unique_tests": agg_total,
        "agg_top1_file_hit": agg_top1,
        "agg_top5_file_hit": agg_top5,
        "agg_top1_file_hit_rate": round(agg_top1 / agg_total, 4) if agg_total else 0,
        "agg_top5_file_hit_rate": round(agg_top5 / agg_total, 4) if agg_total else 0,
        "agg_instance_hit_rate": round(agg_instances_with_hit / len(all_results), 4) if all_results else 0,
    }

    (output_dir / "rca_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # Markdown report
    lines = [
        "# Statement-Level Root Cause Analysis Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Per-Instance Results",
        "",
        "| Instance | Tests | Top-1 Hit | Top-3 Hit | Top-5 Hit | Top-1 (line) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for iid, results in sorted(all_results.items()):
        n = len(results)
        t1 = sum(1 for r in results if r.top1_hit)
        t3 = sum(1 for r in results if r.top3_hit)
        t5 = sum(1 for r in results if r.top5_hit)
        t1l = sum(1 for r in results if r.top1_line_hit)
        short_id = iid[:80] + "..." if len(iid) > 80 else iid
        lines.append(f"| {short_id} | {n} | {t1} | {t3} | {t5} | {t1l} |")

    lines.append("")
    lines.append("## Top Candidates per Instance")
    lines.append("")
    for iid, results in sorted(all_results.items()):
        lines.append(f"### {iid}")
        for r in results:
            lines.append(f"\n**{r.test_nodeid}** — {r.error_type}: {r.error_message}")
            for i, c in enumerate(r.candidates[:5], 1):
                hit_marker = ""
                if r.patch_files:
                    for pf in r.patch_files:
                        if (c.file == pf or c.file.endswith("/" + pf)
                                or pf.endswith("/" + c.file)):
                            hit_marker = " **[PATCH FILE]**"
                            break
                lines.append(
                    f"  {i}. `{c.file}:{c.line}` (score={c.score:.3f}) "
                    f"{' '.join(c.signals)}{hit_marker}"
                )
                if c.code_snippet:
                    lines.append(f"     `{c.code_snippet}`")
        lines.append("")

    (output_dir / "rca_summary.md").write_text("\n".join(lines))
    print(f"RCA summary written to {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run statement-level root cause analysis on existing test results."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results-dir",
        help="Directory containing instance_* subdirectories (batch mode).",
    )
    group.add_argument(
        "--instance-dir",
        help="Single instance directory to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for RCA results (default: <results-dir>/rca_output).",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="HuggingFace dataset name for patch info (e.g. ScaleAI/SWE-bench_Pro).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split.",
    )
    parser.add_argument(
        "--repos-dir",
        help="Directory for cloning repos (enables AST slicing, Step 3).",
    )
    parser.add_argument(
        "--enable-phase3",
        action="store_true",
        help="Enable Phase 3 LLM RCA agent for weak-signal tests.",
    )
    parser.add_argument(
        "--phase3-model",
        default=None,
        help="Model name for Phase 3 agent (default: anthropic/claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for batch analysis (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load dataset if specified
    dataset_samples: dict[str, dict] | None = None
    if args.dataset:
        try:
            from datasets import load_dataset
            ds = load_dataset(args.dataset, split=args.split)
            dataset_samples = {row["instance_id"]: row for row in ds}
            print(f"Loaded {len(dataset_samples)} samples from {args.dataset}")
        except Exception as e:
            print(f"Warning: could not load dataset: {e}")

    repos_dir = Path(args.repos_dir) if args.repos_dir else None

    if args.instance_dir:
        instance_dir = Path(args.instance_dir)
        patch_text = ""
        repo = ""
        base_commit = ""
        matched_sample = None
        if dataset_samples:
            for sid, sample in dataset_samples.items():
                if instance_dir.name.endswith(sid) or sid in instance_dir.name:
                    patch_text = sample.get("patch", "")
                    repo = sample.get("repo", "")
                    base_commit = sample.get("base_commit", "")
                    matched_sample = sample
                    break

        results = analyze_instance(
            instance_dir,
            patch_text=patch_text,
            repo=repo,
            base_commit=base_commit,
            repos_dir=repos_dir,
            sample=matched_sample,
            enable_phase3=args.enable_phase3,
            phase3_model=args.phase3_model,
        )
        if results is None:
            print("No analysis results (non-Python or no data).")
            return

        output_dir = Path(args.output_dir) if args.output_dir else instance_dir / "rca_output"
        write_rca_summary({instance_dir.name: results}, output_dir)

    else:
        results_dir = Path(args.results_dir)
        output_dir = Path(args.output_dir) if args.output_dir else results_dir / "rca_output"

        all_results = analyze_results_dir(
            results_dir,
            dataset_samples=dataset_samples,
            repos_dir=repos_dir,
            enable_phase3=args.enable_phase3,
            phase3_model=args.phase3_model,
            num_workers=args.num_workers,
            output_dir=output_dir,
        )
        write_rca_summary(all_results, output_dir)


if __name__ == "__main__":
    main()
