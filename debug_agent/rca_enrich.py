"""RCA Enricher — post-loop pass that turns DebugReport into structured rich_rca."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

from . import traceback_parser, dataflow_slice, caller_search, nontext_grep


def _trajectory_as_dicts(report) -> list[dict]:
    if hasattr(report, "trajectory_as_dicts"):
        return report.trajectory_as_dicts
    return [t.to_json() for t in getattr(report, "trajectory", [])]


def _last_read_for_path(trajectory: list[dict], path: str) -> str | None:
    for t in reversed(trajectory):
        if t.get("action_name") == "read" and path in (t.get("action_payload") or ""):
            return t.get("tool_output") or None
    return None


def _source_resolver(trajectory: list[dict], container, repo_root: Path | None,
                     workdir: str | None) -> Callable[[str], str | None]:
    def resolve(path: str) -> str | None:
        cached = _last_read_for_path(trajectory, path)
        if cached:
            return cached
        if container is not None:
            try:
                text = container.read_file(path)
                if text:
                    return text
            except Exception:
                pass
        if repo_root is not None:
            rel = _normalize_repo_path(path, workdir)
            p = repo_root / rel
            try:
                return p.read_text()
            except OSError:
                return None
        return None
    return resolve


_PDB_P_RE = re.compile(r"\(Pdb\)\s*p\s+(\w+)\s*\n([^\n]+)")
_PROBE_ASSIGN_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+)$", re.MULTILINE)


def _collect_observed_values(trajectory: list[dict]) -> dict[str, str]:
    obs: dict[str, str] = {}
    for t in trajectory:
        if t.get("action_name") not in ("pdb", "probe"):
            continue
        out = t.get("tool_output") or ""
        for m in _PDB_P_RE.finditer(out):
            obs.setdefault(m.group(1), m.group(2).strip())
        for m in _PROBE_ASSIGN_RE.finditer(out):
            obs.setdefault(m.group(1), m.group(2).strip())
    return obs


def _attach_observed(frames_labeled: list[dict], observed: dict[str, str]) -> None:
    for f in frames_labeled:
        if observed and not f.get("observed_values"):
            f["observed_values"] = dict(observed)


def _normalize_repo_path(path: str, workdir: str | None) -> str:
    """Convert a container-absolute path into a repo-relative one.

    Debug reports often carry paths like `/app/openlibrary/solr/update_work.py`;
    `repo_root / "/app/..."` collapses to `/app/...` because pathlib drops the
    left operand when the right is absolute. Strip the workdir (or any leading
    `/`) so the result concatenates correctly.
    """
    if not path:
        return path
    if workdir:
        prefix = workdir.rstrip("/") + "/"
        if path.startswith(prefix):
            return path[len(prefix):]
    if path.startswith("/"):
        return path.lstrip("/")
    return path


def _symptom_from_repro(container, ctx: dict) -> dict | None:
    """Run the reproduction test inside the container and parse its output.

    Returns the parser result dict, or None when this path is not viable
    (no container, no repro nodeid, exec raised, etc.) — caller falls back.
    """
    if container is None:
        return None
    nodeid = ctx.get("repro_nodeid")
    if not nodeid:
        return None
    workdir = ctx.get("workdir") or "/app"
    env_prefix = ctx.get("env_prefix") or ""
    cmd = f"cd {workdir} && {env_prefix} pytest -x {nodeid}".strip()
    try:
        r = container.exec_bash(cmd, timeout=120)
    except Exception:
        return None
    raw = (getattr(r, "stdout", "") or "") + "\n" + (getattr(r, "stderr", "") or "")
    return traceback_parser.parse_text(raw, evidence_ref="repro_test_run")


def _short_ident(rc_funcs: list[str], rc_files: list[str]) -> str:
    if rc_funcs and rc_funcs[0]:
        return rc_funcs[0].split(".")[-1]
    if rc_files and rc_files[0]:
        return Path(rc_files[0]).stem
    return ""


def _frames_from_pdb(pdb_log: list[dict]) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for t in pdb_log:
        if t.get("kind") != "cmd":
            continue
        cmd = (t.get("cmd") or "").strip().split(None, 1)
        is_step = bool(cmd) and cmd[0] in {"n", "s", "next", "step"}
        f = t.get("current_frame")
        if not f:
            continue
        key = (f.get("file"), f.get("lineno"), f.get("qualname"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "file": f.get("file", ""), "lineno": f.get("lineno", 0),
            "qualname": f.get("qualname", ""),
            "role": "stepped" if is_step else "visited",
        })
    return out


def enrich(report, container, ctx: dict, wall_budget_s: float = 45.0) -> dict:
    if getattr(report, "timed_out", False) and not getattr(report, "root_cause_files", []):
        return {"schema_version": 2, "status": "debug-loop-did-not-converge"}

    start = time.monotonic()
    def over_budget() -> bool:
        return (time.monotonic() - start) > wall_budget_s

    rc_files = list(getattr(report, "root_cause_files", []))
    rc_funcs = list(getattr(report, "root_cause_functions", []))
    rc_file = rc_files[0] if rc_files else ""
    rc_func = rc_funcs[0] if rc_funcs else ""
    repo_root = ctx.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(repo_root)
    workdir = ctx.get("workdir")
    rc_file_rel = _normalize_repo_path(rc_file, workdir)

    root_cause = {
        "file": rc_file, "qualname": rc_func, "lineno": 0,
        "provenance": {"source": "llm_conclusion", "evidence_refs": []},
    }

    trajectory = _trajectory_as_dicts(report)

    # Pass 1: symptom + frames — prefer running the upstream reproduction test
    # (deterministic, authoritative). Fall back to scanning the debug_agent
    # trajectory for any pytest output it happened to capture.
    if over_budget():
        sym_res = {"status": "unavailable", "reason": "timeout", "frames": [],
                   "symptom": None, "evidence_refs": []}
    else:
        sym_res = _symptom_from_repro(container, ctx)
        if sym_res is None:
            sym_res = traceback_parser.parse_failing_traceback(trajectory)

    if sym_res.get("status") == "unavailable":
        symptom = {"status": "unavailable", "reason": sym_res.get("reason", "unknown"),
                   "provenance": {"source": "traceback_parse", "evidence_refs": []}}
        frames: list = []
    else:
        sym = sym_res["symptom"]
        if sym is not None:
            symptom = {"file": sym["file"], "lineno": sym["lineno"],
                       "exc_type": sym.get("exc_type", ""), "exc_msg": sym.get("exc_msg", ""),
                       "provenance": {"source": "traceback_parse",
                                      "evidence_refs": sym_res["evidence_refs"]}}
        else:
            symptom = {"status": "unavailable", "reason": "no-frames-parsed",
                       "provenance": {"source": "traceback_parse",
                                      "evidence_refs": sym_res["evidence_refs"]}}
        frames = sym_res["frames"]

    # Pass 2: role labels
    if over_budget() or not frames:
        labeled: list[dict] = []
        pp_prov_refs = sym_res.get("evidence_refs", [])
    else:
        resolver = _source_resolver(trajectory, container, repo_root, workdir)
        labeled = dataflow_slice.label_frames(frames, resolver)
        observed = _collect_observed_values(trajectory)
        _attach_observed(labeled, observed)
        pp_prov_refs = list(sym_res["evidence_refs"])

    pdb_frames = _frames_from_pdb(ctx.get("_pdb_session_log") or [])
    if pdb_frames:
        existing = {(f.get("file"), f.get("lineno"), f.get("qualname")) for f in labeled}
        for pf in pdb_frames:
            key = (pf["file"], pf["lineno"], pf["qualname"])
            if key not in existing:
                labeled.append(pf)
                existing.add(key)

    propagation_path = {
        "frames": labeled,
        "provenance": {"source": "traceback_parse+ast_slice",
                       "evidence_refs": pp_prov_refs},
    }

    # Pass 3: contract impact — runs whenever we have a function name AND
    # at least one of (host repo_root, live container).
    if over_budget() or not rc_file or not rc_func or (repo_root is None and container is None):
        contract_impact = {"changed": False, "kind": "none", "summary": "",
                           "callers": [], "callers_truncated": False,
                           "provenance": {"source": "ast_caller_scan", "evidence_refs": []}}
    else:
        if repo_root is not None:
            rc_target = rc_file_rel
        else:
            rc_target = (rc_file if rc_file.startswith("/")
                         else f"{(workdir or '/app').rstrip('/')}/{rc_file_rel}")
        ci = caller_search.compute_contract_impact(
            rc_target, rc_func, getattr(report, "suggested_fix", "") or "",
            repo_root=repo_root, container=container,
        )
        contract_impact = {
            "changed": ci["changed"], "kind": ci["kind"], "summary": ci["summary"],
            "callers": ci["callers"], "callers_truncated": ci["callers_truncated"],
            "provenance": {"source": "ast_caller_scan",
                           "evidence_refs": ci["evidence_refs"]},
        }

    # Pass 4: related non-code
    if over_budget() or (repo_root is None and container is None):
        related = {"hits": [], "note": "skipped",
                   "provenance": {"source": "rg_nontext", "evidence_refs": []}}
    else:
        r = nontext_grep.find_related_non_code(
            _short_ident(rc_funcs, rc_files),
            repo_root=repo_root,
            container=container,
        )
        related = {"hits": r["hits"], "note": r["note"],
                   "provenance": {"source": "rg_nontext", "evidence_refs": []}}

    pdb_log = ctx.get("_pdb_session_log") or []
    if over_budget() or not pdb_log:
        rich_branch_observations: list = []
        rich_pdb_anomalies: list = []
    else:
        from .branch_observation import collect_branch_observations
        from .anomaly_snapshot import collect_anomalies
        resolver = _source_resolver(trajectory, container, repo_root, workdir)
        try:
            rich_branch_observations = collect_branch_observations(pdb_log, resolver)
        except Exception:
            rich_branch_observations = []
        try:
            rich_pdb_anomalies = collect_anomalies(pdb_log)
        except Exception:
            rich_pdb_anomalies = []
        if len(rich_pdb_anomalies) > 12:
            rich_pdb_anomalies = rich_pdb_anomalies[:12]

    return {
        "schema_version": 2,
        "root_cause": root_cause,
        "symptom": symptom,
        "propagation_path": propagation_path,
        "contract_impact": contract_impact,
        "related_non_code": related,
        "rich_branch_observations": rich_branch_observations,
        "rich_pdb_anomalies": rich_pdb_anomalies,
    }
