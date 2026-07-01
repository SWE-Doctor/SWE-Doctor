"""Stateful long-lived pdb session over `docker exec -i`.

The LLM drives debugging reactively: set a breakpoint, look at locals,
choose the next step. That requires a single pdb subprocess that survives
across turns. We track turns by reading until the next `(Pdb) ` prompt.

Use `python -u -m pdb` to disable buffering — line buffering alone is not
enough because pdb writes prompts without a trailing newline."""
from __future__ import annotations

import os
import re
import select
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

_PROMPT = "(Pdb) "
_END_MARKERS = (
    "The program finished and will be restarted",
    "--Return--",
    "The program exited via sys.exit",
)
_WHERE_FRAME_RE = re.compile(
    r"> ?(?P<file>[^(\n]+?)\((?P<lineno>\d+)\)(?P<qualname>[<A-Za-z_][\w\.<>]*)\(\)",
)
_POST_MORTEM_MARKER = "Entering post mortem debugging"

_PER_TURN_CAP = 8 * 1024


@dataclass
class StartResult:
    ok: bool
    banner: str = ""
    reason: str = ""                # only when ok=False
    initial_frame: dict | None = None


@dataclass
class CmdResult:
    transcript: str = ""
    ended: bool = False
    end_reason: str = ""            # "program_exited" | "exception" | "timeout" | ""
    current_frame: dict | None = None
    last_eval: str = ""
    # pdb prints these warnings to stdout but reports no error: a `b` on a
    # blank/comment/EOF line is silently dropped, so the next `c` runs the
    # program to completion. Surface the warning so the LLM picks an
    # executable line on the next turn.
    last_eval_warning: str | None = None


class PdbSession:
    def __init__(self, container, run_args: list[str], timeout_per_cmd: float = 10.0,
                 python_exe: str = "python"):
        self.container = container
        self.run_args = list(run_args)
        self.timeout_per_cmd = timeout_per_cmd
        self.python_exe = python_exe
        self._proc: subprocess.Popen | None = None
        self._dead = True

    def start(self) -> StartResult:
        if self._proc is not None:
            self.stop()
        from .required_plugins import detect_required_plugins, required_plugin_short_names
        try:
            workdir = getattr(self.container, "workdir", "/app") or "/app"
            required_pkgs = detect_required_plugins(self.container, workdir)
            required_short = required_plugin_short_names(required_pkgs)
        except Exception:
            required_short = set()
        cmd = _build_session_cmd(
            self.container.container_id, self.run_args,
            required_short_names=required_short,
            python_exe=self.python_exe,
        )
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=0,
        )
        # Make stdout non-blocking so select-based reads can time out.
        try:
            os.set_blocking(self._proc.stdout.fileno(), False)
        except (AttributeError, OSError):
            pass
        self._dead = False
        banner, ended, end_reason = self._read_until_prompt(self.timeout_per_cmd)
        if ended:
            self._dead = True
            return StartResult(ok=False, banner=banner,
                               reason=("program_exited_without_stop"
                                       if end_reason == "program_exited" else end_reason))
        frame = _parse_frame_from_transcript(banner)
        return StartResult(ok=True, banner=banner, initial_frame=frame)

    def cmd(self, raw: str) -> CmdResult:
        if self._dead or self._proc is None:
            return CmdResult(transcript="[session_dead]", ended=True, end_reason="program_exited")
        line = raw if raw.endswith("\n") else raw + "\n"
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._dead = True
            return CmdResult(transcript="[broken_pipe]", ended=True, end_reason="program_exited")

        transcript, ended, end_reason = self._read_until_prompt(self.timeout_per_cmd)
        if end_reason == "timeout":
            transcript += "\n[timeout]"
            return CmdResult(transcript=transcript, ended=False, end_reason="timeout",
                             current_frame=_parse_frame_from_transcript(transcript))
        if ended:
            self._dead = True
            reason = "exception" if "Traceback" in transcript else end_reason
            return CmdResult(transcript=transcript, ended=True, end_reason=reason,
                             current_frame=_parse_frame_from_transcript(transcript))
        # Post-mortem: pdb stays alive at a (Pdb) prompt but the program is dead.
        if _POST_MORTEM_MARKER in transcript:
            return CmdResult(transcript=transcript, ended=True, end_reason="exception",
                             current_frame=_parse_frame_from_transcript(transcript))
        frame = _parse_frame_from_transcript(transcript)
        last_eval = _extract_last_eval(raw, transcript)
        warning = _detect_breakpoint_warning(transcript)
        return CmdResult(transcript=transcript, ended=False, end_reason="",
                         current_frame=frame, last_eval=last_eval,
                         last_eval_warning=warning)

    def restart(self) -> StartResult:
        self.stop()
        return self.start()

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                try:
                    self._proc.stdin.write("q\n")
                    self._proc.stdin.flush()
                except (BrokenPipeError, ValueError):
                    pass
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        finally:
            self._proc = None
            self._dead = True

    def _read_until_prompt(self, timeout: float) -> tuple[str, bool, str]:
        """Read self._proc.stdout until next `(Pdb) ` prompt or EOF/timeout.

        Returns (transcript, ended, end_reason)."""
        assert self._proc is not None and self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        buf: list[str] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ("".join(buf), False, "timeout")
            r, _, _ = select.select([fd], [], [], min(remaining, 0.5))
            if not r:
                continue
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            if chunk == b"":
                return ("".join(buf), True, "program_exited")
            buf.append(chunk.decode("utf-8", errors="replace"))
            tail = "".join(buf)
            if tail.endswith(_PROMPT) or _PROMPT in tail[-len(_PROMPT) - 32:]:
                if tail.endswith(_PROMPT):
                    if any(m in tail for m in _END_MARKERS):
                        return (tail, True, "program_exited")
                    return (tail, False, "")


_PY_BIN_RE = re.compile(r"(?:^|/)python(?:[0-9.]*)$")


def _normalize_pdb_args(run_args: list[str]) -> list[str]:
    """Rewrite agent-supplied run_args into a form `python -m pdb` accepts.

    The agent commonly supplies one of:
      ["pytest", "-x", "tests/foo.py::bar"]      → -m pytest -x …
      ["python", "-m", "pytest", "-x", …]        → -m pytest -x …
      ["/usr/local/bin/python", "-m", "pytest", …] → -m pytest -x …
      ["unittest", "discover", …]                → -m unittest discover …
      ["script.py", "arg"]                       → script.py arg (unchanged)

    pdb treats the first positional as a script path unless `-m <module>` is
    used, so without rewriting these shapes either fail outright ("pytest does
    not exist") or land in post-mortem at <string>:1.
    """
    if not run_args:
        return run_args
    # Strip a leading python interpreter (with or without -u flags).
    args = list(run_args)
    if _PY_BIN_RE.search(args[0]):
        args = args[1:]
        while args and args[0].startswith("-") and args[0] != "-m":
            args = args[1:]
    if args and args[0] == "-m" and len(args) >= 2:
        return args  # already a `-m module …` form pdb accepts directly.
    if args and args[0] in {"pytest", "unittest", "nose2"}:
        return ["-m", args[0], *args[1:]]
    return args  # assume it's a script path; pass through.


def _is_pytest_invocation(pdb_args: list[str]) -> bool:
    """True iff `pdb_args` (already passed through _normalize_pdb_args) invoke
    pytest as a module — i.e. starts with `["-m", "pytest"]`."""
    return len(pdb_args) >= 2 and pdb_args[0] == "-m" and pdb_args[1] == "pytest"


# Plugins pytest auto-loads from the container that abort on `--pdb` if
# active: `pytest-rerunfailures` errors out with "--reruns incompatible
# with --pdb"; `pytest-xdist` and `pytest-forked` likewise refuse. Disable
# them at the front of the args (before plugin auto-load) so --pdb fires.
_PDB_INCOMPATIBLE_PYTEST_PLUGINS = ("rerunfailures", "xdist", "forked")


def _build_session_cmd(
    container_id: str,
    run_args: list[str],
    required_short_names: set[str] | None = None,
    python_exe: str = "python",
) -> list[str]:
    """Return the full `docker exec -i …` command list for a debug session.

    pytest payloads route through `python -u -m pytest --pdb …` because
    pytest catches AssertionError internally; an outer `python -m pdb`
    wrapper never sees the failure and exits cleanly when `c` is issued.
    Non-pytest payloads keep the legacy `python -u -m pdb …` wrapper.
    """
    pdb_args = _normalize_pdb_args(run_args)
    if _is_pytest_invocation(pdb_args):
        # SWE-bench-pro images preset PYTEST_ADDOPTS=--reruns=3 ... in the
        # container env; that bypasses both -o addopts= and -c <ini> because
        # pytest concatenates PYTEST_ADDOPTS onto argv before parsing config.
        # `docker exec -e PYTEST_ADDOPTS=` clears it for this invocation.
        base = ["docker", "exec", "-i", "-e", "PYTEST_ADDOPTS=",
                container_id, python_exe, "-u"]
        rest = list(pdb_args[2:])
        if "--pdb" not in rest:
            rest = ["--pdb", *rest]
        # Strip any `-p no:<required>` the agent (or addopts/env) injected for
        # plugins the project declares as required — disabling them aborts
        # pytest before --pdb can fire (qutebrowser bug).
        required = required_short_names or set()
        if required:
            cleaned: list[str] = []
            i = 0
            while i < len(rest):
                tok = rest[i]
                if (
                    tok == "-p"
                    and i + 1 < len(rest)
                    and rest[i + 1].startswith("no:")
                    and rest[i + 1][len("no:"):] in required
                ):
                    i += 2
                    continue
                cleaned.append(tok)
                i += 1
            rest = cleaned

        flat = " ".join(rest)
        prefix: list[str] = []
        for plug in _PDB_INCOMPATIBLE_PYTEST_PLUGINS:
            # Skip disabling plugins the project requires (loaded but inactive
            # is fine because addopts is cleared below + PYTEST_ADDOPTS empty,
            # so no --reruns conflict fires).
            if plug in required:
                continue
            token = f"no:{plug}"
            if token not in flat:
                prefix += ["-p", token]
        # Defensive: also clear any addopts coming from pyproject.toml / setup.cfg
        if "-o" not in flat or "addopts=" not in flat:
            prefix += ["-o", "addopts="]
        return [*base, "-m", "pytest", *prefix, *rest]
    base = ["docker", "exec", "-i", container_id, python_exe, "-u"]
    return [*base, "-m", "pdb", *pdb_args]


def _parse_frame_from_transcript(text: str) -> dict | None:
    matches = list(_WHERE_FRAME_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {
        "file": m.group("file").strip(),
        "lineno": int(m.group("lineno")),
        "qualname": m.group("qualname"),
    }


def _extract_last_eval(cmd_text: str, transcript: str) -> str:
    """For `p <expr>` / `pp <expr>` only — return the line(s) printed before
    the next prompt. Returns "" otherwise."""
    head = cmd_text.strip().split(None, 1)
    if not head or head[0] not in ("p", "pp"):
        return ""
    body = transcript
    if body.endswith(_PROMPT):
        body = body[: -len(_PROMPT)]
    return body.strip().splitlines()[-1] if body.strip() else ""


def _detect_breakpoint_warning(transcript: str) -> str | None:
    """pdb silently rejects `b` on blank/comment/EOF lines: it prints a
    one-line warning and returns to the prompt without installing the
    breakpoint. Detect that so the dispatcher can surface a [WARNING] to
    the LLM and the agent picks an executable line next turn."""
    if "Blank or comment" in transcript:
        return "blank_or_comment_breakpoint"
    if "End of file" in transcript:
        return "breakpoint_past_eof"
    if "Breakpoint" in transcript and "was not set" in transcript:
        return "breakpoint_not_set"
    return None


def _maybe_truncate(text: str, log_dir: "Path | None", session_id: int,
                    turn: int) -> tuple[str, "Path | None"]:
    if len(text) <= _PER_TURN_CAP:
        return text, None
    sidecar: Path | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        sidecar = log_dir / f"pdb_session_{session_id}_turn_{turn}.log"
        sidecar.write_text(text)
    head = text[:_PER_TURN_CAP]
    note = f"\n[truncated {len(text) - _PER_TURN_CAP} chars; full at {sidecar}]"
    return head + note, sidecar
