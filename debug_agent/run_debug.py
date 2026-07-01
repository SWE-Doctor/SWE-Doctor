"""Per-instance debug-agent runner. CLI mirrors run_statement_rca.py.

Emits <output-dir>/<instance_id>_rca.json with the unified schema:
    {"source": "debug_agent",
     "candidates": [{"file", "func", "score"}...],
     "debug_report": {...DebugReport.to_json()...},
     "meta": {"container_mode": "launch"|"attach", "image": ..., "model": ...}}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .actions import dispatch as default_dispatch
from .analyzer import DebugAnalyzer
from .container import Container


def _run_enricher(report, container, workdir: str,
                  repro_nodeid: str = "", env_prefix: str = "",
                  pdb_session_log: list | None = None) -> dict:
    from .rca_enrich import enrich
    repo_root = os.environ.get("DEBUG_AGENT_REPO_ROOT")
    ctx = {"workdir": workdir, "repro_nodeid": repro_nodeid, "env_prefix": env_prefix}
    if repo_root:
        ctx["repo_root"] = Path(repo_root)
    if pdb_session_log is not None:
        ctx["_pdb_session_log"] = pdb_session_log
    return enrich(report, container=container, ctx=ctx)


def _load_issue_text(instance_dir: Path) -> str:
    # Prefer a cached problem_statement.txt if the repro stage wrote one.
    for name in ("problem_statement.txt", "issue.txt"):
        p = instance_dir / name
        if p.exists():
            return p.read_text(errors="replace")
    # Fall back to the Stage-2a stdout (it usually includes the issue header).
    stdout = instance_dir / "stdout.log"
    return stdout.read_text(errors="replace") if stdout.exists() else ""


def _find_accepted_repro(instance_dir: Path, language: str = "python") -> Path | None:
    # Layouts, in preference order:
    #   - Phase A smoke:           stage1_reproduction/accepted/*.py / *_test.go
    #   - Phase A output (flat):   stage1_reproduction/*.py / *_test.go
    #   - Full pipeline (stage2):  workspace/_repro_tests/*.py / *_test.go  ← written by
    #                              run_repro_trace / python_runner
    # Go prefers a *_test.go (the runnable target) but falls back to any *.go.
    globs = ["*_test.go", "*.go"] if language == "go" else ["*.py"]
    for candidate in [
        instance_dir / "stage1_reproduction" / "accepted",
        instance_dir / "stage1_reproduction",
        instance_dir / "workspace" / "_repro_tests",
    ]:
        if candidate.is_dir():
            for pattern in globs:
                files = sorted(candidate.glob(pattern))
                if files:
                    return files[0]
    return None


def _copy_into_container(c: Container, src: Path, dest_in_container: str) -> None:
    subprocess.run(
        ["docker", "cp", str(src), f"{c.container_id}:{dest_in_container}"],
        check=True, capture_output=True,
    )


def _build_llm(model: str) -> Callable[[list], str]:
    """Return a callable(messages) -> str backed by litellm.

    Takes a role-separated chat `messages` list (system/user/assistant/...).
    A single flat user prompt let gpt-5.4 continue the transcript and fabricate
    the tool's turns in one runaway response; the tu-zi gateway ignores
    max_tokens/stop so role boundaries are the only lever (see analyzer.run)."""
    import litellm
    # gpt-5 / gpt-5-codex reject temperature!=1 with UnsupportedParamsError;
    # telling litellm to drop unsupported params keeps the call generic across
    # model families without a per-model switch.
    litellm.drop_params = True

    # Inject reasoning_effort (e.g. "high") from env when set; empty leaves unset.
    extra = {}
    _re = os.environ.get("REASONING_EFFORT", "").strip()
    if _re:
        extra["reasoning_effort"] = _re

    # Streaming toggle (default off). Some gateways (tu-zi) serve gpt-5.x ONLY as
    # a stream; a non-stream call returns an empty SSE. When LLM_STREAM is set we
    # collect chunks and rebuild an equivalent ModelResponse via stream_chunk_builder.
    _stream = os.environ.get("LLM_STREAM", "").strip().lower() in ("1", "true", "yes", "on")

    # DeepSeek thinking-mode toggle (external param, default on). Stage 2b runs
    # with reasoning ON by default; gate on model name so non-DeepSeek is unaffected.
    if "deepseek" in model.lower():
        _think = os.environ.get("DEEPSEEK_THINKING", "enabled").strip().lower()
        _off = _think in ("disabled", "off", "0", "false", "no")
        extra["extra_body"] = {"thinking": {"type": "disabled" if _off else "enabled"}}

    def call(messages: list) -> str:
        # Pre-flight: refuse over-context-window input. tu-zi does NOT raise
        # ContextWindowExceededError for over-window input — it hangs ~273s then
        # times out, so the 8-retry loop below would burn ~30+ min per round and
        # the SIGALRM wall_timeout does not reliably interrupt the blocked read.
        # Raising ContextWindowExceededError lets analyzer.run terminate the
        # instance immediately. Unknown window / token count -> skip the check.
        try:
            _info = litellm.get_model_info(model)
            _limit = _info.get("max_input_tokens") or _info.get("max_tokens")
        except Exception:
            _limit = None
        if _limit:
            try:
                _ntok = litellm.token_counter(model=model, messages=messages)
            except Exception:
                _ntok = None
            if _ntok and _ntok > _limit:
                raise litellm.exceptions.ContextWindowExceededError(
                    message=f"debug-agent input ~{_ntok} tokens exceeds context window ({_limit}) for {model}",
                    model=model,
                    llm_provider=model.split("/")[0] if "/" in model else "openai",
                )
        # Retry transient tu-zi failures: 403 "groups not available" flapping,
        # rate limits, and None/empty content (gpt-5.4 returns content=None
        # intermittently). Exhausting retries returns "" so the analyzer
        # degrades (re-prompt / timeout) instead of crashing the instance.
        last = None
        for attempt in range(1, 9):
            try:
                if _stream:
                    _chunks = list(litellm.completion(
                        model=model, messages=messages, temperature=0.0,
                        stream=True, stream_options={"include_usage": True}, **extra,
                    ))
                    resp = litellm.stream_chunk_builder(_chunks, messages=messages)
                else:
                    resp = litellm.completion(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        **extra,
                    )
                content = resp["choices"][0]["message"]["content"]
                if content:
                    return content
                last = "empty/None content"
            except Exception as e:  # transient gateway error
                last = repr(e)[:200]
            if attempt < 8:
                time.sleep(min(3 * attempt, 30))
        print(f"[debug-agent] llm call exhausted 8 retries; last={last}", flush=True)
        return ""

    return call


def _build_candidates(report) -> list[dict]:
    """Build the candidates[] list for the unified _rca.json, normalizing
    container-specific path prefixes (/app/, dist-packages, doubled package dirs)
    to canonical repo-relative paths. Test files / repro scripts are filtered
    out — bugs live in production code, never in the test that observes them.
    The original string is preserved as `raw_file` whenever normalization
    changes it."""
    from .path_norm import normalize_repo_path, is_noise_file
    out = []
    skipped = 0
    for f in report.root_cause_files:
        if is_noise_file(f):
            skipped += 1
            continue
        i = len(out) + skipped  # preserve original score ordering
        func = report.root_cause_functions[i] if i < len(report.root_cause_functions) else ""
        normalized = normalize_repo_path(f)
        entry = {"file": normalized, "func": func, "score": max(0.0, 1.0 - 0.1 * len(out))}
        if normalized != f:
            entry["raw_file"] = f
        out.append(entry)
    return out


def _run_go_instance(
    *, c, instance_id: str, issue_text: str, repro_test: Path | None,
    output_dir: Path, workdir: str, image: str | None, model: str, mode: str,
    max_rounds: int, wall_timeout: int,
    llm_factory: Callable[[str], Callable[[str], str]] | None,
) -> Path:
    """Go RCA path: dlv backend reused through pdb_start/pdb_cmd. Mirrors the
    python path but (a) installs dlv, (b) injects a GoDlvSession factory so the
    LLM's pdb_start drives `dlv test`, and (c) skips python-only steps (pytest
    preflight, setprofile focused-trace). Does NOT terminate the container — the
    caller's finally owns that."""
    from .go_session import GoDlvSession

    c.ensure_dlv()

    repro_nodeid = ""
    go_pkg = "./..."
    go_test_name = ""
    if repro_test is not None:
        # Go tests MUST live in their package directory (same `package`, so they
        # reach unexported symbols and same-package test helpers) — copying to
        # _repro/ would not compile. Honor the .relpath sidecar (e.g.
        # server/zzz_repro_test.go) written by Stage-1/2a; fall back to the repo
        # root package when absent.
        relpath_file = repro_test.with_suffix(".relpath")
        dest_rel = (relpath_file.read_text().strip()
                    if relpath_file.exists() else repro_test.name)
        dest_dir = os.path.dirname(dest_rel)
        if dest_dir:
            c.exec_bash(f"mkdir -p {workdir}/{dest_dir}")
            go_pkg = "./" + dest_dir
        _copy_into_container(c, repro_test, f"{workdir}/{dest_rel}")
        repro_nodeid = dest_rel
        _m = re.search(r"func\s+(Test\w+)", repro_test.read_text(errors="replace"))
        go_test_name = _m.group(1) if _m else ""

    ctx = {
        "language": "go",
        "issue": issue_text,
        "workdir": workdir,
        "repro_path": f"{workdir}/{repro_nodeid}" if repro_nodeid else "",
        "repro_nodeid": repro_nodeid,
        "go_pkg": go_pkg,                 # e.g. ./server — for dlv test / go test
        "go_test_name": go_test_name,     # e.g. TestBatchEvaluate...
        "cwd": workdir,
        "probe_run_cmd": None,            # probe is python/js-only; dlv covers Go
        "env_prefix": f"cd {workdir} &&",
        "preflight_seed": "(none)",
        # The multi-backend injection point (actions.pdb_start): the LLM keeps
        # emitting pdb_start/pdb_cmd, but the session is dlv, not pdb.
        "_pdb_session_factory": (lambda ra, _c=c: GoDlvSession(_c, run_args=ra)),
    }

    llm = (llm_factory or _build_llm)(model)
    analyzer = DebugAnalyzer(
        llm=llm, dispatch=default_dispatch, container=c, ctx=ctx,
        max_rounds=max_rounds,
    )

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"debug-agent wall-clock timeout after {wall_timeout}s")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler) if hasattr(signal, "SIGALRM") else None
    if hasattr(signal, "SIGALRM"):
        signal.alarm(wall_timeout)
    try:
        report = analyzer.run()
    except TimeoutError:
        report = analyzer._partial_report() if hasattr(analyzer, "_partial_report") else None
        if report is None:
            from .analyzer import DebugReport
            report = DebugReport(timed_out=True)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    # Cross-check LLM conclusion files against the dlv frame log (same shape as
    # the python pdb log). Replaces hallucinated paths with executed ones.
    from .conclusion_validator import validate_against_pdb_log
    _pdb_log = ctx.get("_pdb_session_log", []) or []
    if report.root_cause_files:
        new_files, new_funcs = validate_against_pdb_log(
            report.root_cause_files, report.root_cause_functions or [], _pdb_log,
        )
        report.root_cause_files = new_files
        report.root_cause_functions = new_funcs

    candidates = _build_candidates(report)
    payload = {
        "source": "debug_agent",
        "candidates": candidates,
        "debug_report": report.to_json(),
        "meta": {
            "container_mode": mode,
            "image": image,
            "model": model,
            "instance_id": instance_id,
        },
    }
    out_path = output_dir / f"{instance_id}_rca.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def run_one_instance(
    instance_dir: Path,
    output_dir: Path,
    image: str | None,
    model: str,
    attach_container: str | None = None,
    max_rounds: int = 30,
    wall_timeout: int = 600,
    workdir: str = "/app",
    python_exe: str = "python",
    llm_factory: Callable[[str], Callable[[str], str]] | None = None,
    container_factory: Callable | None = None,
    language: str = "python",
) -> Path:
    """Run the debug agent on one instance. Returns the path to the written _rca.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_id = instance_dir.name

    issue_text = _load_issue_text(instance_dir)
    repro_test = _find_accepted_repro(instance_dir, language=language)

    mode = "attach" if attach_container else "launch"
    if container_factory is not None:
        c = container_factory()
    elif attach_container:
        c = Container.attach(attach_container, workdir=workdir)
    else:
        if not image:
            raise SystemExit("--image required unless --attach provided")
        c = Container.launch(image=image, workdir=workdir)

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"debug-agent wall-clock timeout after {wall_timeout}s")

    report = None
    try:
        if language == "go":
            out_path = _run_go_instance(
                c=c, instance_id=instance_id, issue_text=issue_text,
                repro_test=repro_test, output_dir=output_dir, workdir=workdir,
                image=image, model=model, mode=mode, max_rounds=max_rounds,
                wall_timeout=wall_timeout, llm_factory=llm_factory,
            )
            return out_path

        # Copy the repro test into <workdir>/_repro/ so the agent can run it.
        repro_nodeid = ""
        if repro_test is not None:
            c.exec_bash(f"mkdir -p {workdir}/_repro")
            _copy_into_container(c, repro_test, f"{workdir}/_repro/{repro_test.name}")
            repro_nodeid = f"_repro/{repro_test.name}"

        from .preflight import bootstrap
        pre = bootstrap(c, workdir=workdir, repro_nodeid=repro_nodeid)
        ctx = {
            "issue": issue_text,
            "workdir": workdir,
            "repro_path": f"{workdir}/{repro_nodeid}" if repro_nodeid else "",
            "repro_nodeid": repro_nodeid,
            "cwd": workdir,
            "probe_run_cmd": f"{pre.env_prefix} pytest -x {repro_nodeid}" if repro_nodeid else None,
            "pdb_target": None,
            "env_prefix": pre.env_prefix,
            "preflight_seed": pre.as_transcript_seed(),
            "python_exe": python_exe,
        }

        llm = (llm_factory or _build_llm)(model)
        analyzer = DebugAnalyzer(
            llm=llm,
            dispatch=default_dispatch,
            container=c,
            ctx=ctx,
            max_rounds=max_rounds,
        )

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler) if hasattr(signal, "SIGALRM") else None
        if hasattr(signal, "SIGALRM"):
            signal.alarm(wall_timeout)
        try:
            report = analyzer.run()
        except TimeoutError:
            # Best-effort: surface whatever partial report the analyzer built.
            report = analyzer._partial_report() if hasattr(analyzer, "_partial_report") else None
            if report is None:
                from .analyzer import DebugReport
                report = DebugReport(timed_out=True)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        # Cross-check LLM conclusion files against PDB frame log; replace
        # hallucinated paths with what was actually executed. Validator keeps
        # files and functions aligned by index — filtering a file also drops
        # its function; the PDB-frames replacement path emits empty function
        # names since PDB tells us the file but not the bug function.
        from .conclusion_validator import validate_against_pdb_log
        _pdb_log = ctx.get("_pdb_session_log", []) or []
        if report.root_cause_files:
            new_files, new_funcs = validate_against_pdb_log(
                report.root_cause_files,
                report.root_cause_functions or [],
                _pdb_log,
            )
            report.root_cause_files = new_files
            report.root_cause_functions = new_funcs

        # If the analyzer never produced root_cause_files (typically because
        # pdb_start could not post_mortem an assert-style failure), fall back
        # to a setprofile run that records every production file the failing
        # repro touched. The deepest call site is the most likely bug site.
        if not report.root_cause_files and repro_nodeid:
            from .focused_trace_fallback import run_focused_trace
            from .path_norm import normalize_repo_path
            try:
                files = run_focused_trace(c, workdir=workdir, repro_nodeid=repro_nodeid)
            except Exception as e:
                files = []
                report.reasoning = (report.reasoning or "") + f"\n[focused_trace_fallback failed: {e!r}]"
            if files:
                report.root_cause_files = [normalize_repo_path(f) for f in files[:5]]
                report.reasoning = (report.reasoning or "") + (
                    "\n[fallback] PDB could not post-mortem (program exited without stop). "
                    "Above files come from setprofile during the failing repro, "
                    "ranked by last-seen-first."
                )

        # Build unified _rca.json — run BEFORE c.terminate() so the enricher
        # can still access the container via read_file().
        candidates = _build_candidates(report)

        payload = {
            "source": "debug_agent",
            "candidates": candidates,
            "debug_report": report.to_json(),
            "meta": {
                "container_mode": mode,
                "image": image,
                "model": model,
                "instance_id": instance_id,
            },
        }

        if os.environ.get("DEBUG_AGENT_ENRICH") == "1":
            import sys as _sys
            enricher = getattr(_sys.modules[__name__], "_run_enricher")
            try:
                payload["rich_rca"] = enricher(
                    report, c, workdir,
                    repro_nodeid=repro_nodeid, env_prefix=pre.env_prefix,
                    pdb_session_log=ctx.get("_pdb_session_log"),
                )
            except Exception as e:
                payload["rich_rca"] = {
                    "schema_version": 2, "status": "enricher-error", "error": repr(e),
                }

        out_path = output_dir / f"{instance_id}_rca.json"
        out_path.write_text(json.dumps(payload, indent=2))
    finally:
        c.terminate()

    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic debug-agent RCA runner.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--instance-dir", help="Single instance directory.")
    g.add_argument("--results-dir", help="Directory containing instance_* subdirectories.")
    p.add_argument("--output-dir", help="Output dir (default: <results-dir>/rca_output).")
    p.add_argument("--image", help="Docker image to launch (required unless --attach).")
    p.add_argument("--model", default=os.environ.get("DEBUG_AGENT_MODEL", "openai/gpt-4o-mini"))
    p.add_argument("--attach", help="Attach to an existing container id instead of launching.")
    p.add_argument("--max-rounds", type=int, default=30)
    p.add_argument("--wall-timeout", type=int, default=600)
    p.add_argument("--workdir", default="/app",
                   help="In-container workdir (SWE-bench Pro images use /app).")
    p.add_argument("--python-exe", default="python",
                   help="In-container python (Verified: /opt/miniconda3/envs/testbed/bin/python).")
    p.add_argument("--language", default="python", choices=["python", "go"],
                   help="Language of the repro test (default: python). 'go' drives dlv via pdb_start.")
    # Accept and ignore legacy flags so run_rca dispatcher can forward verbatim.
    p.add_argument("--dataset", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--repos-dir", default=None)
    p.add_argument("--enable-phase3", action="store_true")
    p.add_argument("--phase3-model", default=None)
    p.add_argument("--num-workers", type=int, default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.instance_dir:
        instance_dirs = [Path(args.instance_dir)]
        out_dir = Path(args.output_dir) if args.output_dir else instance_dirs[0].parent / "rca_output"
    else:
        root = Path(args.results_dir)
        instance_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("instance_"))
        out_dir = Path(args.output_dir) if args.output_dir else root / "rca_output"

    rc = 0
    for d in instance_dirs:
        try:
            path = run_one_instance(
                instance_dir=d,
                output_dir=out_dir,
                image=args.image,
                model=args.model,
                attach_container=args.attach,
                max_rounds=args.max_rounds,
                wall_timeout=args.wall_timeout,
                workdir=args.workdir,
                python_exe=args.python_exe,
                language=args.language,
            )
            print(f"[debug-agent] wrote {path}", flush=True)
        except Exception as e:  # keep batch going
            import traceback as _tb
            print(f"[debug-agent] ERROR on {d.name}: {e!r}", file=sys.stderr, flush=True)
            _tb.print_exc(file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
