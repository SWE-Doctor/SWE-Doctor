"""Canonicalize Python file paths observed inside SWE-bench containers.

Three transformations, applied in order:
  1. Strip leading container roots (/app/, /testbed/, dist-packages, site-packages)
  2. Collapse doubled top-level package directories (foo/foo/...)
  3. Treat 'lib/<pkg>/' and '<pkg>/' as equivalent for matching only

`normalize_repo_path` returns a canonical repo-relative form.
`paths_match` is the comparison helper used by hit-rate scoring.
"""
from __future__ import annotations
import re

_PREFIXES = (
    "/app/",
    "/testbed/",
)
_SITE_OR_DIST_RE = re.compile(
    # Match site/dist-packages under ANY root, not just /usr[/local]: SWE-bench
    # Verified images put the test env under a conda path
    # (/opt/miniconda3/envs/testbed/lib/pythonX.Y/site-packages/). Superset of
    # the old /usr-only pattern, so Pro paths normalize identically.
    r"^.*?/lib/python\d+(?:\.\d+)?/(?:site|dist)-packages/"
)


def normalize_repo_path(p: str) -> str:
    if not p:
        return p
    p = _SITE_OR_DIST_RE.sub("", p)
    for pre in _PREFIXES:
        if p.startswith(pre):
            p = p[len(pre):]
            break
    p = p.lstrip("/")
    parts = p.split("/")
    if len(parts) >= 2 and parts[0] == parts[1]:
        p = "/".join(parts[1:])
    return p


def _match_key(p: str) -> str:
    """Form used for equality only. Drops a leading 'lib/' so
    'lib/ansible/x.py' compares equal to 'ansible/x.py'."""
    p = normalize_repo_path(p)
    if p.startswith("lib/"):
        p = p[len("lib/"):]
    return p


def paths_match(a: str, b: str) -> bool:
    return _match_key(a) == _match_key(b)


_FRAMEWORK_PKG_PREFIXES = (
    "_pytest/", "pytest/", "pluggy/", "py/", "_distutils_hack/",
    "setuptools/", "pip/", "importlib_metadata/",
)

# Stdlib lives at /usr[/local]/lib/pythonX.Y/<module>, NOT under
# site-packages/dist-packages. After normalize_repo_path strips the leading
# slash but leaves stdlib paths intact, they show up as
# `usr/local/lib/python3.10/asyncio/base_events.py`. Bugs in these projects
# are never in stdlib, so treat the whole tree as noise.
_STDLIB_PATH_RE = re.compile(r"^usr/(?:local/)?lib/python\d+(?:\.\d+)?/")


def is_framework_file(p: str) -> bool:
    """True for pytest/pluggy/setuptools-style paths AND stdlib paths that
    should never be reported as bug sites. Critically, project code installed
    via `pip install -e .` lives in site-packages too (e.g. ansible inside
    `/usr/local/lib/python3.9/dist-packages/ansible/`), so we must NOT
    blanket-exclude site-packages — only specific framework package names.
    Stdlib (no site/dist-packages segment) is safe to blanket-exclude."""
    if not p:
        return False
    norm = normalize_repo_path(p)  # strips site-packages prefix
    if any(norm.startswith(pre) for pre in _FRAMEWORK_PKG_PREFIXES):
        return True
    if _STDLIB_PATH_RE.match(norm):
        return True
    return False


def is_test_file(p: str) -> bool:
    """True for test files / repro scripts that should NEVER be reported as
    bug candidates. Bugs live in production code; the test is what observes
    the bug, not what causes it."""
    if not p:
        return False
    p = normalize_repo_path(p)
    base = p.rsplit("/", 1)[-1]
    if (base.startswith("test_") or base.endswith("_test.py")
            or base.endswith("_test.go") or base in ("conftest.py",)):
        return True
    if "/_repro/" in p or p.startswith("_repro/") or p.endswith("/_repro"):
        return True
    if "/tests/" in p or p.startswith("tests/"):
        return True
    return False


def is_noise_file(p: str) -> bool:
    """Combined predicate: test file OR framework file. Use this to filter
    candidate files that should never be reported as bug locations."""
    return is_test_file(p) or is_framework_file(p)
