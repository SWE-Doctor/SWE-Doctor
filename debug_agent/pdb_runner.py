"""One-shot pdb: compose -c flags, run, return transcript.

pdb's -c flag accepts any pdb command; when the last command is 'q' (or
the program exits naturally) the subprocess terminates. Avoids long-lived
screen sessions and TTY handling."""
from __future__ import annotations

import shlex

from .container import Container


def run_pdb_script(c: Container, script_path: str, commands: list[str], timeout: int = 60) -> str:
    flags = " ".join(f"-c {shlex.quote(cmd)}" for cmd in commands)
    r = c.exec_bash(
        f"python -m pdb {flags} {shlex.quote(script_path)}",
        timeout=timeout,
    )
    # pdb transcripts are mostly on stderr when running under -c;
    # keep both and trim to 8 KB so the LLM context stays manageable.
    return r.combined[-8000:]
