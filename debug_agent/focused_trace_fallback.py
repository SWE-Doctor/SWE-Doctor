"""Fallback when pdb_start cannot post_mortem (assert-style failures).

Runs the failing repro once with sys.setprofile installed, prints
TRACE_ENTER lines for every Python file the test touches, and returns
production files in last-seen-first order — so the deepest call site
(usually the bug site) ranks highest.
"""
from __future__ import annotations
from typing import Any

# This script is dropped into the container and executed under `python`.
# It drives pytest itself (via runpy) with setprofile installed, so every
# function call inside the failing test prints a TRACE_ENTER line.
PROFILER_SCRIPT = r"""
import sys, runpy
def _profile(frame, event, arg):
    if event == "call":
        f = frame.f_code.co_filename or ""
        if f.endswith(".py"):
            print("TRACE_ENTER " + f, flush=True)
    return _profile
sys.setprofile(_profile)
try:
    runpy.run_module("pytest", run_name="__main__")
except SystemExit:
    pass
finally:
    sys.setprofile(None)
"""


def parse_profile_output(raw: str, workdir: str) -> list[str]:
    """Return production files, deduped, in last-seen-first order.

    Excludes:
      - files outside <workdir>
      - the repro test itself (under <workdir>/_repro/)
      - conftest.py (test plumbing, not production code)
    """
    seen: dict[str, int] = {}
    for i, line in enumerate(raw.splitlines()):
        if not line.startswith("TRACE_ENTER "):
            continue
        f = line[len("TRACE_ENTER "):].strip()
        if not f.startswith(workdir.rstrip("/") + "/"):
            continue
        if "/_repro/" in f or f.endswith("/conftest.py"):
            continue
        seen[f] = i  # last occurrence wins
    return [f for f, _ in sorted(seen.items(), key=lambda kv: -kv[1])]


def run_focused_trace(container: Any, workdir: str, repro_nodeid: str) -> list[str]:
    """Run pytest under setprofile inside the container; return ranked files."""
    shim_path = f"{workdir}/_repro/_profile_runner.py"
    container.exec_bash(f"mkdir -p {workdir}/_repro")
    # Heredoc: argv assignment + the shared PROFILER_SCRIPT body
    # IMPORTANT: do NOT pass `-p no:rerunfailures` or `-p no:xdist` here.
    # We're running pytest without --pdb, so neither plugin is harmful.
    # Disabling rerunfailures interacts with `requiredplugins =
    # pytest-rerunfailures` in conftests (qutebrowser) and makes pytest abort
    # with "Missing required plugins". Disabling xdist breaks
    # pytest-rerunfailures>=10 (`unknown hook 'pytest_configure_node'`).
    # `-p no:cacheprovider` is the only safe one — it just prevents lastfailed
    # cache writes.
    container.exec_bash(
        f"cat > {shim_path} <<'PROFILE_RUNNER'\n"
        f"import sys\n"
        f"sys.argv = ['pytest', '-x', '-p', 'no:cacheprovider', "
        f"'{repro_nodeid}']\n"
        f"{PROFILER_SCRIPT}\n"
        f"PROFILE_RUNNER"
    )
    res = container.exec_bash(f"cd {workdir} && python {shim_path} 2>&1")
    raw = getattr(res, "stdout", "") or ""
    return parse_profile_output(raw, workdir)
