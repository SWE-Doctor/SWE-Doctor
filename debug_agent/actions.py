"""Action parsing and dispatch for the debug agent.

The LLM emits exactly one <action name="..."> per turn. We parse, route
to the matching runner, and return a string the LLM can consume next turn."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_TAG = re.compile(
    r'<action\s+name="(bash|pytest|gotest|pdb|pdb_start|pdb_cmd|pdb_script|pdb_stop'
    r'|dlv_start|dlv_cmd|dlv_script|dlv_stop|probe|read)"\s*>(.*?)</action>',
    re.DOTALL,
)
# The Go prompts speak pure dlv, so the model emits dlv_* verbs. They map onto
# the shared pdb_* handlers (the dlv backend is driven through the same session
# plumbing), so the parser/gate/dispatch keep ONE canonical name. Python keeps
# pdb_* untouched.
_ALIAS = {"dlv_start": "pdb_start", "dlv_cmd": "pdb_cmd",
          "dlv_script": "pdb_script", "dlv_stop": "pdb_stop"}
_STRUCTURED = {"probe", "read", "pdb_start", "pdb_cmd", "pdb_script", "pdb_stop"}


@dataclass
class ActionCall:
    name: str
    payload: str = ""
    kwargs: dict = field(default_factory=dict)


def parse_action(response: str) -> ActionCall | None:
    m = _TAG.search(response)
    if not m:
        return None
    name, body = m.group(1), m.group(2).strip()
    name = _ALIAS.get(name, name)   # dlv_* (Go) -> canonical pdb_* handler
    if name in _STRUCTURED:
        try:
            return ActionCall(name=name, kwargs=json.loads(body))
        except json.JSONDecodeError:
            return ActionCall(name=name, payload=body)
    return ActionCall(name=name, payload=body)


def _truncate(s: str, n: int = 8000) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n… [truncated {len(s) - n} chars]"


def dispatch(action: ActionCall, container: Any, ctx: dict) -> str:
    """Execute the action against `container` and return text for the LLM.

    ctx provides per-instance paths like `repro_nodeid` and `cwd`."""
    # Local imports so pure parser tests don't require docker modules.
    from .pdb_runner import run_pdb_script
    from .probe import apply_probe
    from .pytest_runner import run_pytest
    from .source_reader import grep_src, read

    # Echo the verb family the model actually drives: dlv_* for Go, pdb_* for
    # Python. Keeps protocol-error hints consistent with the prompt the model saw.
    _go = ctx.get("language") == "go"
    _start = "dlv_start" if _go else "pdb_start"
    _cmd = "dlv_cmd" if _go else "pdb_cmd"
    _script = "dlv_script" if _go else "pdb_script"

    if action.name == "bash":
        r = container.exec_bash(action.payload)
        return _truncate(f"rc={r.returncode}\n{r.combined}")

    if action.name == "pytest":
        args = action.payload or ctx.get("repro_nodeid", "")
        cwd = ctx.get("cwd")
        env_prefix = ctx.get("env_prefix") or ""
        r = run_pytest(container, args=args, cwd=cwd, env_prefix=env_prefix)
        summary = "\n".join(
            f"FAILED {ft.nodeid} — {ft.exc_type}: {ft.exc_msg}" for ft in r.failed_tests
        ) or "(no failed tests parsed)"
        return _truncate(f"rc={r.returncode}\n{summary}\n--- OUTPUT ---\n{r.combined}")

    if action.name == "gotest":
        # Non-debug test evidence for the gate (the Go analog of pytest/jest).
        # Local import so pure parser tests don't pull in run_pro_test/go_runner.
        from .gotest_runner import run_gotest
        args = action.payload or ctx.get("gotest_run", "")
        cwd = ctx.get("cwd")
        env_prefix = ctx.get("env_prefix") or ""
        r = run_gotest(container, args=args, cwd=cwd, env_prefix=env_prefix)
        summary = "\n".join(f"FAILED {ft.nodeid}" for ft in r.failed_tests) or "(no failed tests parsed)"
        return _truncate(f"rc={r.returncode}\n{summary}\n--- OUTPUT ---\n{r.combined}")

    if action.name == "pdb":
        # payload is a newline-separated script of pdb commands
        commands = [line for line in action.payload.splitlines() if line.strip()]
        script_path = ctx.get("pdb_target") or ctx.get("repro_nodeid", "").split("::", 1)[0]
        if not script_path:
            return "ERROR: no pdb_target in ctx and repro_nodeid missing"
        tr = run_pdb_script(container, script_path=script_path, commands=commands)
        return _truncate(tr)

    if action.name == "probe":
        try:
            file = action.kwargs["file"]
            before_line = int(action.kwargs["before_line"])
            expr = action.kwargs["expr"]
        except (KeyError, TypeError, ValueError) as e:
            return f"ERROR: probe kwargs invalid ({e}); need {{file, before_line, expr}}"
        run_cmd = action.kwargs.get("run_cmd") or ctx.get("probe_run_cmd")
        if not run_cmd:
            return "ERROR: probe needs run_cmd (kwarg or ctx['probe_run_cmd'])"
        try:
            pr = apply_probe(container, file=file, before_line=before_line,
                             expr=expr, run_cmd=run_cmd)
        except ValueError as e:
            return f"ERROR: {e}"
        return _truncate(f"rc={pr.returncode}\nSTDOUT:\n{pr.stdout}\nSTDERR:\n{pr.stderr}")

    if action.name == "pdb_start":
        run_args = action.kwargs.get("run_args") or []
        if not isinstance(run_args, list) or not run_args:
            return f"ERROR: {_start} requires run_args=[...]"
        old = ctx.get("_pdb_session")
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
        factory = ctx.get("_pdb_session_factory")
        if factory is None:
            from .pdb_session import PdbSession
            _py_exe = ctx.get("python_exe", "python")
            factory = lambda ra, _c=container, _pe=_py_exe: PdbSession(_c, run_args=ra, python_exe=_pe)
        sess = factory(run_args)
        res = sess.start()
        ctx["_pdb_session"] = sess
        ctx["_pdb_session_id"] = ctx.get("_pdb_session_id", 0) + 1
        _record_session_turn(ctx, kind="start", session_id=ctx["_pdb_session_id"],
                             ok=res.ok, reason=res.reason,
                             initial_frame=res.initial_frame,
                             run_args=list(run_args))
        return _truncate(_format_start(res))

    if action.name == "pdb_cmd":
        sess = ctx.get("_pdb_session")
        if sess is None:
            return f"ERROR: no active debug session — call {_start} first"
        cmd_text = action.kwargs.get("cmd", "")
        if not cmd_text:
            return f"ERROR: {_cmd} requires {{cmd: '...'}}"
        res = sess.cmd(cmd_text)
        _record_session_turn(ctx, kind="cmd", session_id=ctx.get("_pdb_session_id", 0),
                             cmd=cmd_text, transcript=res.transcript,
                             current_frame=res.current_frame, ended=res.ended,
                             end_reason=res.end_reason, last_eval=res.last_eval,
                             last_eval_warning=getattr(res, "last_eval_warning", None))
        return _truncate(_format_cmd(cmd_text, res))

    if action.name == "pdb_script":
        sess = ctx.get("_pdb_session")
        if sess is None:
            return f"ERROR: no active debug session — call {_start} first"
        commands = action.kwargs.get("commands") or []
        if not isinstance(commands, list) or not commands:
            return f"ERROR: {_script} requires commands=[...]"
        chunks = []
        for c in commands:
            res = sess.cmd(c)
            _record_session_turn(ctx, kind="cmd",
                                 session_id=ctx.get("_pdb_session_id", 0),
                                 cmd=c, transcript=res.transcript,
                                 current_frame=res.current_frame, ended=res.ended,
                                 end_reason=res.end_reason, last_eval=res.last_eval,
                                 last_eval_warning=getattr(res, "last_eval_warning", None))
            chunks.append(_format_cmd(c, res))
            if res.ended:
                break
        return _truncate("\n\n".join(chunks))

    if action.name == "pdb_stop":
        sess = ctx.get("_pdb_session")
        if sess is None:
            return "no session to stop"
        try:
            sess.stop()
        finally:
            ctx["_pdb_session"] = None
        _record_session_turn(ctx, kind="stop",
                             session_id=ctx.get("_pdb_session_id", 0))
        return "stopped"

    if action.name == "read":
        if "grep" in action.kwargs and "path" in action.kwargs:
            return _truncate(grep_src(container, action.kwargs["grep"], action.kwargs["path"]))
        path = action.kwargs.get("path")
        if not path:
            return "ERROR: read needs {path, start?, n?} or {grep, path}"
        # Clamp to the valid 1-indexed range: the model sometimes emits start=0
        # (0-indexing) or n=0, which read() would otherwise reject by raising.
        start = max(1, int(action.kwargs.get("start", 1)))
        n = max(1, int(action.kwargs.get("n", 40)))
        return _truncate(read(container, path, start=start, n=n))

    return f"ERROR: unknown action '{action.name}'"


def _record_session_turn(ctx: dict, **fields) -> None:
    ctx.setdefault("_pdb_session_log", []).append(fields)


def _format_start(res) -> str:
    if not res.ok:
        return f"ERROR: debug session failed to start — {res.reason}\n{res.banner[:1000]}"
    frame = res.initial_frame or {}
    return (f"debug session started.\n"
            f"initial_frame: {frame.get('file','?')}:{frame.get('lineno','?')} "
            f"in {frame.get('qualname','?')}\n--- BANNER ---\n{res.banner[:2000]}")


_BP_WARNING_HINT = {
    "blank_or_comment_breakpoint": (
        "[WARNING] pdb did NOT set the breakpoint — the line is blank or a comment. "
        "Pick an executable line and re-issue `b <file>:<line>` BEFORE `c`."
    ),
    "breakpoint_past_eof": (
        "[WARNING] pdb did NOT set the breakpoint — line is past end of file. "
        "Re-`read` the target file to find a valid line, then re-issue `b`."
    ),
    "breakpoint_not_set": (
        "[WARNING] pdb refused to set the breakpoint. Verify the in-container "
        "absolute path and that the line is executable."
    ),
}


def _format_cmd(cmd_text: str, res) -> str:
    frame = res.current_frame or {}
    head = (f"current_frame: {frame.get('file','?')}:{frame.get('lineno','?')} "
            f"in {frame.get('qualname','?')}") if res.current_frame else "current_frame: (none)"
    end = f"  ended={res.ended}{(' reason='+res.end_reason) if res.end_reason else ''}"
    body = f"$ {cmd_text}\n{head}{end}\n--- TRANSCRIPT ---\n{res.transcript}"
    hint = _BP_WARNING_HINT.get(getattr(res, "last_eval_warning", None) or "")
    if hint:
        body = f"{hint}\n{body}"
    return body
