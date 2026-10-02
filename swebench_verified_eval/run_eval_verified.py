"""Stage 4 (grading) for Verified — thin wrapper over the official harness.

Reads Stage-3 preds.json, runs `python -m swebench.harness.run_evaluation`
against SWE-bench Verified, then normalizes the harness report into
eval_summary.json alongside the predictions.

Note on report file location: make_run_report() in the official harness
writes <model_name_or_path.replace("/","__")>.<run_id>.json to the CWD of the
process — NOT to --report_dir (which only creates the directory but is not
forwarded to make_run_report). We work around this by running the subprocess
with cwd=out_dir so the summary JSON lands alongside our other outputs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from verified_common import VERIFIED_DATASET


def _normalize_preds(preds_path: Path, model_name: str) -> Path:
    """Ensure each entry has instance_id/model_name_or_path/model_patch.

    Stage-3 writes a dict {iid: {model_patch, ...}}; the harness accepts that,
    but we make model_name_or_path consistent so the report filename is stable.
    """
    data = json.loads(preds_path.read_text())
    if isinstance(data, dict):
        for iid, entry in data.items():
            entry.setdefault("instance_id", iid)
            entry["model_name_or_path"] = model_name
        normalized = data
    else:
        normalized = data
    out = preds_path.resolve().with_name("preds_for_eval.json")
    out.write_text(json.dumps(normalized, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="V2 Stage-4 eval (Verified, official harness)")
    p.add_argument("--preds", type=Path, required=True, help="Stage-3 preds.json")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--model-name", type=str, default="anthropic__claude-sonnet-4-6")
    p.add_argument("--dataset", type=str, default=VERIFIED_DATASET)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write eval_summary.json (default: alongside preds)")
    args = p.parse_args()

    out_dir = (args.output_dir or args.preds.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_for_eval = _normalize_preds(args.preds, args.model_name)

    # model_name_or_path with "/" replaced by "__" — matches reporting.py logic
    model_slug = args.model_name.replace("/", "__")

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", args.dataset,
        "--split", args.split,
        "--predictions_path", str(preds_for_eval),
        "--run_id", args.run_id,
        "--max_workers", str(args.max_workers),
        "--timeout", str(args.timeout),
        "--namespace", "swebench",
        "--report_dir", str(out_dir),
    ]
    print("RUN:", " ".join(cmd))
    # Run with cwd=out_dir so make_run_report writes the summary JSON here.
    # (The harness writes <model_slug>.<run_id>.json relative to CWD, not report_dir.)
    rc = subprocess.run(cmd, cwd=str(out_dir)).returncode

    # The harness writes: <model_slug>.<run_id>.json in cwd (=out_dir)
    report_file = out_dir / f"{model_slug}.{args.run_id}.json"
    summary = {"harness_rc": rc, "report_file": str(report_file)}
    if report_file.exists():
        rep = json.loads(report_file.read_text())
        # Keys from reporting.py make_run_report():
        #   submitted_instances, completed_instances, resolved_instances,
        #   unresolved_instances, error_instances  (counts)
        #   resolved_ids, unresolved_ids, error_ids  (lists)
        summary.update({
            "submitted": rep.get("submitted_instances"),
            "completed": rep.get("completed_instances"),
            "resolved": rep.get("resolved_instances"),
            "resolved_ids": rep.get("resolved_ids", []),
            "unresolved_ids": rep.get("unresolved_ids", []),
            "error_ids": rep.get("error_ids", []),
        })
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print("eval_summary.json ->", out_dir / "eval_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
