"""Transactional print-probe: inject `print("PROBE", repr(<expr>))`
before a given line, run a command, and revert on any path.

Always goes through Container.snapshot/restore so probes can never
leak between rounds — even on exception."""
from __future__ import annotations

from dataclasses import dataclass

from .container import Container


@dataclass
class ProbeResult:
    stdout: str
    stderr: str
    returncode: int


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def apply_probe(
    c: Container,
    file: str,
    before_line: int,
    expr: str,
    run_cmd: str,
    timeout: int = 180,
) -> ProbeResult:
    src = c.read_file(file)
    c.snapshot(file)
    lines = src.splitlines(keepends=True)
    if before_line < 1 or before_line > len(lines) + 1:
        raise ValueError(f"before_line {before_line} out of range (1..{len(lines) + 1})")
    # Indent matches the target line's indent (or empty at EOF).
    anchor = lines[before_line - 1] if before_line <= len(lines) else ""
    indent = _indent_of(anchor)
    inject = f'{indent}print("PROBE", repr({expr}))\n'
    lines.insert(before_line - 1, inject)
    c.write_file(file, "".join(lines))
    try:
        r = c.exec_bash(run_cmd, timeout=timeout)
        return ProbeResult(stdout=r.stdout, stderr=r.stderr, returncode=r.returncode)
    finally:
        c.restore(file)
