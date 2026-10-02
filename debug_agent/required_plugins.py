"""Detect pytest plugins declared as required by a project.

Used by:
  - preflight: install missing required plugins (defense in depth)
  - pdb_session: skip disabling plugins the project requires (or pytest
    aborts with `Missing required plugins` before --pdb can fire).
"""
from __future__ import annotations
import re
from typing import Any

# Match `required_plugins = <value>` where <value> is either inline
#   required_plugins = pytest-foo pytest-bar
# or a multi-line indented block (the qutebrowser style):
#   required_plugins =
#       pytest-bdd >= 5.0.0
#       pytest-rerunfailures >= 9.1.1
# The block ends at the next non-indented line (next key or [section]) or EOF.
# The earlier single-line form `(.+)$` only captured the first plugin.
_REQUIRED_PLUGINS_RE = re.compile(
    r"^\s*required[_-]?plugins\s*=\s*(.*?)(?=^\S|\Z)",
    re.M | re.I | re.S,
)
_CONFIG_FILE_NAMES = ("setup.cfg", "pytest.ini", "pyproject.toml", "tox.ini")


def _read_file(c: Any, path: str) -> str:
    r = c.exec_bash(f"cat {path}")
    if getattr(r, "returncode", 1) != 0:
        return ""
    return getattr(r, "stdout", "") or ""


def detect_required_plugins(c: Any, workdir: str) -> list[str]:
    """Return canonical plugin package names (e.g. ['pytest-rerunfailures'])
    declared in any pytest config file under <workdir>. Version specifiers
    are stripped."""
    declared: list[str] = []
    for name in _CONFIG_FILE_NAMES:
        body = _read_file(c, f"{workdir}/{name}")
        if not body:
            continue
        for m in _REQUIRED_PLUGINS_RE.finditer(body):
            line = m.group(1).strip().strip('"').strip("'")
            for tok in re.split(r"[\s,]+", line):
                tok = tok.strip()
                if tok.startswith("pytest-"):
                    pkg = re.split(r"[<>=!~]", tok, maxsplit=1)[0]
                    if pkg not in declared:
                        declared.append(pkg)
    return declared


def required_plugin_short_names(declared: list[str]) -> set[str]:
    """Map full package names to the short plugin names pytest uses with
    `-p no:<short>`. e.g. 'pytest-rerunfailures' → 'rerunfailures'."""
    out = set()
    for pkg in declared:
        if pkg.startswith("pytest-"):
            out.add(pkg[len("pytest-"):])
    return out
