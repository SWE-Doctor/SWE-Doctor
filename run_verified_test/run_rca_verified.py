"""Stage 2b RCA dispatcher for Verified — spawns debug_agent.run_debug per
instance with --workdir /testbed and the testbed python."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path

from verified_common import TESTBED_PYTHON, verified_image_name

_ROOT = Path(__file__).resolve().parents[1]


def _resolve_image(instance_dir: Path, iid: str) -> str:
    pinned = instance_dir / "docker_image.txt"
    if pinned.exists():
        text = pinned.read_text().strip()
        if text:
            return text
    return verified_image_name(iid)


def _run_one(instance_dir: Path, out_dir: Path, model: str) -> int:
    iid = instance_dir.name
    # Resume after a monitor pause: skip instances whose RCA is already written
    # (debug_agent emits <out_dir>/<iid>_rca.json) instead of re-spending relay.
    if (out_dir / f"{iid}_rca.json").exists():
        return 0
    image = _resolve_image(instance_dir, iid)
    cmd = [
        sys.executable, "-m", "debug_agent.run_debug",
        "--instance-dir", str(instance_dir),
        "--output-dir", str(out_dir),
        "--image", image,
        "--workdir", "/testbed",
        "--python-exe", TESTBED_PYTHON,
    ]
    if model:
        cmd += ["--model", model]
    return subprocess.run(cmd, cwd=str(_ROOT)).returncode


def main() -> None:
    p = argparse.ArgumentParser(description="V2 Stage-2b RCA dispatcher (Verified, debug_agent)")
    p.add_argument("--results-dir", type=Path, required=True, help="Stage-2a output dir")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inst_dirs = [d for d in sorted(args.results_dir.iterdir())
                 if d.is_dir() and (d / "workspace").exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futs = {ex.submit(_run_one, d, args.output_dir, args.model): d.name for d in inst_dirs}
        for fut in concurrent.futures.as_completed(futs):
            iid = futs[fut]
            try:
                rc = fut.result()
                print(f"[{iid}] debug_agent rc={rc}")
            except Exception as exc:  # noqa: BLE001 - keep batch going
                print(f"[{iid}] ERROR: {exc!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
