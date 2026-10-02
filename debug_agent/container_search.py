"""Run rg / grep / walk / read-text inside a Container, mirroring host helpers.

SWE-bench Pro images do NOT have ripgrep installed. Default to `grep -rnw`
which is universally available; rg is tried opportunistically when present.

Output normalization: paths are returned relative to `root` (default /app).
"""
from __future__ import annotations

import shlex
from typing import Iterator


def _strip_root(path: str, root: str) -> str:
    prefix = root.rstrip("/") + "/"
    if path.startswith(prefix):
        return path[len(prefix):]
    return path.lstrip("/")


def _has_rg(container) -> bool:
    r = container.exec_bash("command -v rg >/dev/null 2>&1", timeout=5)
    return r.returncode == 0


def _parse_lines(stdout: str, root: str) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for line in (stdout or "").splitlines():
        try:
            path_part, ln_part, _ = line.split(":", 2)
        except ValueError:
            continue
        try:
            ln = int(ln_part)
        except ValueError:
            continue
        hits.append((_strip_root(path_part, root), ln))
    return hits


def container_rg(container, pattern: str, root: str = "/app",
                 time_budget_s: float = 10.0,
                 max_hits: int = 1000,
                 ) -> tuple[list[tuple[str, int]], bool]:
    """Word-boundary search inside `container`. Returns (hits, truncated).

    Uses ripgrep if available; otherwise falls back to `grep -rnw`. tests/
    directories and `test_*.py` files are excluded in both paths."""
    timeout_s = max(2, int(time_budget_s) + 1)
    if _has_rg(container):
        cmd = (
            f"rg -n --no-heading -w {shlex.quote(pattern)} "
            f"--glob '!tests/**' --glob '!**/test_*.py' "
            f"{shlex.quote(root)} 2>/dev/null | head -n {int(max_hits)}"
        )
    else:
        # grep -rnw: -r recursive, -n line numbers, -w word match
        # Exclude tests/ and test_*.py via --exclude-dir / --exclude.
        cmd = (
            f"grep -rnw --include='*.py' "
            f"--exclude-dir=tests --exclude='test_*.py' "
            f"{shlex.quote(pattern)} {shlex.quote(root)} 2>/dev/null "
            f"| head -n {int(max_hits)}"
        )
    try:
        r = container.exec_bash(cmd, timeout=timeout_s)
    except Exception:
        return [], True
    if r.returncode not in (0, 1):
        return [], True
    return _parse_lines(r.stdout or "", root), False


def container_grep_nontext(container, pattern: str, root: str = "/app",
                           time_budget_s: float = 10.0,
                           max_hits: int = 1000,
                           ) -> list[str] | None:
    """Search non-Python text files (md/rst/yaml/yml/json/toml/ini) for a
    word. Returns raw `path:lineno:match` lines (matching the host helper's
    contract), or None on error/timeout."""
    timeout_s = max(2, int(time_budget_s) + 1)
    if _has_rg(container):
        cmd = (
            f"rg -n --no-heading -w {shlex.quote(pattern)} "
            "--glob '!*.py' --glob '!tests/**' "
            "--glob '*.{md,rst,yaml,yml,json,toml,ini}' "
            f"{shlex.quote(root)} 2>/dev/null | head -n {int(max_hits)}"
        )
    else:
        # grep equivalent: --include for each extension; --exclude-dir for tests.
        includes = " ".join(f"--include='*.{ext}'" for ext in
                            ("md", "rst", "yaml", "yml", "json", "toml", "ini"))
        cmd = (
            f"grep -rnw {includes} --exclude-dir=tests "
            f"{shlex.quote(pattern)} {shlex.quote(root)} 2>/dev/null "
            f"| head -n {int(max_hits)}"
        )
    try:
        r = container.exec_bash(cmd, timeout=timeout_s)
    except Exception:
        return None
    if r.returncode not in (0, 1):
        return None
    return [ln for ln in (r.stdout or "").splitlines() if ln.strip()]


def container_walk_py(container, root: str = "/app") -> Iterator[str]:
    """Yield production-Python files under `root`, relative to `root`.
    Filters out tests/ directories and basenames starting with `test_`."""
    cmd = f"find {shlex.quote(root)} -type f -name '*.py'"
    r = container.exec_bash(cmd, timeout=30)
    for ln in (r.stdout or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rel = _strip_root(ln, root)
        parts = rel.split("/")
        if "tests" in parts:
            continue
        if parts and parts[-1].startswith("test_"):
            continue
        yield rel


def container_read_text(container, path: str) -> str:
    """Best-effort `cat`. Returns empty string on non-zero rc."""
    r = container.exec_bash(f"cat {shlex.quote(path)}", timeout=10)
    if r.returncode != 0:
        return ""
    return r.stdout or ""
