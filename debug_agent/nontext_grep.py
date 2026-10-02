"""Ripgrep wrapper (with pure-Python fallback): identifier matches in non-code non-test files."""
from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

_EXT_GLOB = "*.{md,rst,yaml,yml,json,toml,ini}"
_EXTS = {".md", ".rst", ".yaml", ".yml", ".json", ".toml", ".ini"}


def _rg_search(identifier: str, repo_root: Path, time_budget_s: float) -> list[str] | None:
    """Return raw rg output lines, or None if rg unavailable / failed."""
    if shutil.which("rg") is None:
        return None
    try:
        proc = subprocess.run(
            ["rg", "-n", "--no-heading", "-w", identifier,
             "--glob", "!*.py", "--glob", "!tests/**",
             "--glob", _EXT_GLOB, str(repo_root)],
            capture_output=True, text=True, timeout=time_budget_s,
        )
    except subprocess.TimeoutExpired:
        return None
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def _container_search(container, identifier: str, time_budget_s: float,
                      root: str = "/app") -> list[str] | None:
    """Run the search inside `container`. Returns raw lines or None on error.
    SWE-bench Pro images lack rg, so this delegates to a grep fallback."""
    from .container_search import container_grep_nontext
    return container_grep_nontext(container, identifier, root=root,
                                  time_budget_s=time_budget_s)


def _py_search(identifier: str, repo_root: Path) -> list[str]:
    word_re = re.compile(rf"\b{re.escape(identifier)}\b")
    lines: list[str] = []
    for p in repo_root.rglob("*"):
        if not p.is_file() or p.suffix not in _EXTS:
            continue
        try:
            rel = p.relative_to(repo_root)
        except ValueError:
            continue
        if "tests" in rel.parts:
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if word_re.search(line):
                lines.append(f"{p}:{i}:{line}")
    return lines


def find_related_non_code(identifier: str, repo_root: Path | None,
                          cap: int = 20, too_common_threshold: int = 200,
                          time_budget_s: float = 10.0,
                          container=None) -> dict:
    if not identifier or len(identifier) < 3:
        return {"hits": [], "note": "identifier too short — skipped", "evidence_refs": []}

    if repo_root is not None:
        rg_out = _rg_search(identifier, repo_root, time_budget_s)
        if rg_out is None:
            rg_out = _py_search(identifier, repo_root)
    elif container is not None:
        rg_out = _container_search(container, identifier, time_budget_s)
        if rg_out is None:
            return {"hits": [], "note": "container rg failed", "evidence_refs": []}
    else:
        return {"hits": [], "note": "no repo_root or container", "evidence_refs": []}

    if not rg_out:
        return {"hits": [], "note": "no identifier matches", "evidence_refs": []}
    if len(rg_out) > too_common_threshold:
        return {"hits": [], "note": "identifier too common — skipped", "evidence_refs": []}

    seen: set[tuple[str, int]] = set()
    hits: list[dict] = []
    for ln in rg_out:
        try:
            path_part, ln_num_part, match = ln.split(":", 2)
        except ValueError:
            continue
        try:
            ln_num = int(ln_num_part)
        except ValueError:
            continue
        if repo_root is not None:
            try:
                rel = str(Path(path_part).relative_to(repo_root))
            except ValueError:
                rel = path_part
        else:
            rel = path_part.lstrip("/").removeprefix("app/")
        key = (rel, ln_num)
        if key in seen:
            continue
        seen.add(key)
        hits.append({"file": rel, "lineno": ln_num, "match": match.strip()})
        if len(hits) >= cap:
            break
    return {"hits": hits, "note": "", "evidence_refs": []}
