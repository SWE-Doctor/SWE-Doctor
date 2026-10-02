#!/usr/bin/env python3
"""Unified RCA entrypoint.

Dispatches to debug_agent.run_debug (default) or run_statement_rca (fallback)
based on:
  * RCA_BACKEND env var ("debug_agent" | "static"), default "debug_agent"
  * auto-fallback to static when no accepted Phase-A repro bundle exists
    for the target instance(s) — because debug_agent has no test to drive.

CLI is a superset of run_statement_rca.py so shell drivers can be swapped
without touching their arg lists.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _has_accepted_repro(instance_dir: Path) -> bool:
    # Must stay in sync with debug_agent.run_debug._find_accepted_repro.
    for candidate in [
        instance_dir / "stage1_reproduction" / "accepted",
        instance_dir / "stage1_reproduction",
        instance_dir / "workspace" / "_repro_tests",
    ]:
        if candidate.is_dir() and any(candidate.glob("*.py")):
            return True
    return False


def _iter_instance_dirs(args: argparse.Namespace) -> list[Path]:
    if args.instance_dir:
        return [Path(args.instance_dir)]
    root = Path(args.results_dir)
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("instance_"))


def _split_instances_by_backend(
    args: argparse.Namespace, backend: str
) -> tuple[list[Path], list[Path]]:
    """Return (debug_agent_targets, static_targets)."""
    if backend != "debug_agent":
        return [], _iter_instance_dirs(args)
    dbg, static = [], []
    for d in _iter_instance_dirs(args):
        (dbg if _has_accepted_repro(d) else static).append(d)
    return dbg, static


def _run_static(argv_tail: list[str]) -> int:
    cmd = [sys.executable, str(_HERE / "run_statement_rca.py"), *argv_tail]
    return subprocess.run(cmd).returncode


def _resolve_image(instance_dir: Path, fallback: str | None) -> str | None:
    """Per-instance image resolution.

    Prefers ``docker_image.txt`` written by run_repro_trace/python_runner,
    because each instance needs a different container tag. ``--image`` is a
    last-resort fallback (useful only for single-instance smoke).
    """
    pinned = instance_dir / "docker_image.txt"
    if pinned.exists():
        tag = pinned.read_text().strip()
        if tag:
            return tag
    return fallback


def _run_debug_agent(instance_dirs: list[Path], args: argparse.Namespace) -> int:
    # Forward per-instance to keep output paths aligned with instance_dir.
    # Each subprocess spawns its own docker container (unique name + per-instance
    # image), writes its own _rca.json, and the LLM endpoint is shared via tenacity
    # backoff — so plain ThreadPoolExecutor parallelism is safe.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(d: Path) -> int:
        cmd = [
            sys.executable, "-m", "debug_agent.run_debug",
            "--instance-dir", str(d),
        ]
        if args.output_dir:
            cmd += ["--output-dir", args.output_dir]
        image = _resolve_image(d, args.image)
        if image:
            cmd += ["--image", image]
        if args.model:
            cmd += ["--model", args.model]
        r = subprocess.run(cmd, cwd=str(_ROOT))
        return r.returncode

    workers = max(1, int(getattr(args, "num_workers", 1) or 1))
    rc = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, d): d for d in instance_dirs}
        for f in as_completed(futures):
            r = f.result()
            if r != 0:
                rc = r
    return rc


def _build_static_argv(instance_dirs: list[Path], args: argparse.Namespace) -> list[list[str]]:
    """For fallback instances, call run_statement_rca once per instance-dir."""
    out: list[list[str]] = []
    for d in instance_dirs:
        tail = ["--instance-dir", str(d)]
        if args.output_dir:
            tail += ["--output-dir", args.output_dir]
        if args.dataset:
            tail += ["--dataset", args.dataset]
        out.append(tail)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified RCA dispatcher (debug-agent | static).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--instance-dir")
    g.add_argument("--results-dir")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--model", default=os.environ.get("DEBUG_AGENT_MODEL"))
    p.add_argument("--dataset", default="")
    # Accept legacy flags silently for call-site compatibility.
    p.add_argument("--split", default="test")
    p.add_argument("--repos-dir", default=None)
    p.add_argument("--enable-phase3", action="store_true")
    p.add_argument("--phase3-model", default=None)
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backend = os.environ.get("RCA_BACKEND", "debug_agent")

    dbg_targets, static_targets = _split_instances_by_backend(args, backend)

    rc = 0
    if dbg_targets:
        rc |= _run_debug_agent(dbg_targets, args)
    for tail in _build_static_argv(static_targets, args):
        rc |= _run_static(tail)
    return rc


if __name__ == "__main__":
    sys.exit(main())
