"""Read and grep source files inside a Container, with explicit errors."""
from __future__ import annotations

import shlex

from .container import Container


def read(c: Container, path: str, start: int = 1, n: int = 40) -> str:
    if start < 1 or n < 1:
        raise ValueError("start>=1 and n>=1 required")
    end = start + n - 1
    q = shlex.quote(path)
    r = c.exec_bash(f"test -f {q} && sed -n {start},{end}p {q}")
    if r.returncode != 0:
        stderr = (r.stderr or "").strip() or "file not found or unreadable"
        return f"ERROR reading {path}: {stderr}"
    if not r.stdout.strip():
        return f"(empty — lines {start}..{end} past end of {path})"
    return r.stdout


def grep_src(c: Container, pattern: str, path: str, max_hits: int = 40) -> str:
    q_pat = shlex.quote(pattern)
    q_path = shlex.quote(path)
    cmd = (
        f"test -e {q_path} && grep -nE {q_pat} -r {q_path} 2>/dev/null "
        f"| head -n {int(max_hits)}"
    )
    r = c.exec_bash(cmd)
    if r.returncode == 1 or not r.stdout.strip():
        return f"(0 hits for /{pattern}/ in {path})"
    if r.returncode not in (0, 1):
        return f"ERROR grep {path}: rc={r.returncode} {r.stderr.strip()}"
    return r.stdout
