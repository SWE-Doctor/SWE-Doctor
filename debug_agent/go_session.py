"""Stateful long-lived Delve (dlv) debug session over `docker exec -i`.

The Go equivalent of pdb_session.PdbSession: the LLM drives debugging
reactively (set a breakpoint, look at locals, step), so we keep a single
`dlv test` subprocess alive across turns and read until the next `(dlv) `
prompt. We deliberately REUSE the StartResult/CmdResult contract and the
`{file, lineno, qualname}` frame dict from pdb_session — downstream gating /
conclusion_validator / rca_enrich consume those shapes and stay untouched.

dlv wiring (validated by spike on the flipt go1.24.3 image):
- login shell (`bash -lc`) drops go from PATH → export it explicitly.
- NO_COLOR=1 TERM=dumb → strip dlv's ANSI colour codes from the transcript.
- --allow-non-terminal-interactive=true → dlv refuses a piped stdin otherwise.
- GOFLAGS=-mod=mod → go1.24 is -mod=readonly by default and won't backfill go.sum.
"""
from __future__ import annotations

import os
import re
import select
import subprocess
import time

from .pdb_session import StartResult, CmdResult  # contract reuse — do NOT redefine

_PROMPT = "(dlv) "
_END_MARKERS = (
    "has exited with status",
    "Process restarted",
)
# Breakpoint/step stop line, e.g.:
#   > [Breakpoint 1] pkg.path.(*T).M() ./a/b.go:20 (hits ...) (PC: 0x..)
#   > pkg.path.(*T).M() ./a/b.go:21 (PC: 0x..)
_DLV_FRAME_RE = re.compile(
    r"> (?:\[Breakpoint \d+\] )?(?P<qualname>\S+?)\(\) (?P<file>[^\s:]+):(?P<lineno>\d+)"
)

_PER_TURN_CAP = 8 * 1024


def _normalize_go_test_args(run_args: list[str]) -> tuple[str, str]:
    """Reduce an agent-supplied run_args list to (package, test_name).

    Accepts the common shapes the LLM emits:
      ["go","test","./server","-run","TestX"]   → ("./server", "TestX")
      ["./server","-run","TestX"]               → ("./server", "TestX")
      ["-run","TestX","./server"]               → ("./server", "TestX")
      ["./server"]                              → ("./server", "")
    The test name is stripped of any ^...$ anchors (we re-anchor on build)."""
    args = [a for a in (run_args or []) if a not in ("go", "test", "dlv")]
    pkg, run = "", ""
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-run", "-test.run") and i + 1 < len(args):
            run = args[i + 1].strip("^$")
            i += 2
            continue
        if a.startswith("-run=") or a.startswith("-test.run="):
            run = a.split("=", 1)[1].strip("^$")
        elif not a.startswith("-"):
            pkg = a
        i += 1
    return (pkg or "./...", run)


def _parse_frame_from_transcript(text: str) -> dict | None:
    matches = list(_DLV_FRAME_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    f = m.group("file")
    if f.startswith("./"):
        f = f[2:]
    return {"file": f, "lineno": int(m.group("lineno")), "qualname": m.group("qualname")}


def _extract_last_eval(cmd_text: str, transcript: str) -> str:
    """For `print`/`p` only — return the last non-prompt line printed."""
    head = cmd_text.strip().split(None, 1)
    if not head or head[0] not in ("print", "p"):
        return ""
    body = transcript
    if body.endswith(_PROMPT):
        body = body[: -len(_PROMPT)]
    lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _detect_breakpoint_warning(transcript: str) -> str | None:
    """dlv prints a notice and stays at the prompt when a breakpoint can't be
    placed; surface it so the LLM picks a real line next turn."""
    if "could not find statement" in transcript or "could not find file" in transcript:
        return "breakpoint_not_set"
    if "Command failed: location" in transcript:
        return "breakpoint_not_set"
    return None


class GoDlvSession:
    def __init__(self, container, run_args: list[str], timeout_per_cmd: float = 160.0,
                 dlv_path: str = "/go/bin/dlv"):
        self.container = container
        self.run_args = list(run_args)
        self.timeout_per_cmd = timeout_per_cmd
        self.dlv_path = dlv_path
        self._proc: subprocess.Popen | None = None
        self._dead = True

    def _build_dlv_cmd(self) -> list[str]:
        pkg, run = _normalize_go_test_args(self.run_args)
        workdir = getattr(self.container, "workdir", "/app") or "/app"
        run_flag = f" -test.run '^{run}$'" if run else ""
        inner = (
            "export PATH=/usr/local/go/bin:/go/bin:$PATH; "
            f"cd {workdir}; "
            f"exec env NO_COLOR=1 TERM=dumb GOFLAGS=-mod=mod "
            f"{self.dlv_path} test {pkg} --allow-non-terminal-interactive=true --{run_flag}"
        )
        return ["docker", "exec", "-i", self.container.container_id, "bash", "-c", inner]

    def start(self) -> StartResult:
        if self._proc is not None:
            self.stop()
        cmd = self._build_dlv_cmd()
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=0,
        )
        try:
            os.set_blocking(self._proc.stdout.fileno(), False)
        except (AttributeError, OSError):
            pass
        self._dead = False
        banner, ended, end_reason = self._read_until_prompt(self.timeout_per_cmd)
        if ended:
            self._dead = True
            return StartResult(ok=False, banner=banner,
                               reason=("compile_or_start_failed"
                                       if end_reason == "program_exited" else end_reason))
        # dlv test halts at the REPL with the program not yet running → no frame.
        return StartResult(ok=True, banner=banner, initial_frame=None)

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
            reason = "exception" if "panic:" in transcript else end_reason
            return CmdResult(transcript=transcript, ended=True, end_reason=reason,
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
                    self._proc.stdin.write("quit\n")
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
        """Read stdout until the next `(dlv) ` prompt or EOF/timeout.
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
            if tail.endswith(_PROMPT):
                if any(m in tail for m in _END_MARKERS):
                    return (tail, True, "program_exited")
                return (tail, False, "")
