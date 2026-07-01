"""Deterministic environment fixups before the LLM debug loop.

Goal: by the time the analyzer starts, the repro test actually runs and
produces a *useful* failure (AssertionError / exception inside target code),
not an environment error like ModuleNotFoundError.

Fixups attempted in order, each additive:
  1. PYTHONPATH=<workdir>
  2. pip install -e <workdir>   (only if setup.py or pyproject.toml present)

Readiness is decided by **actually running the repro** and checking the
combined stdout+stderr for env-error tokens. `--collect-only` is NOT a
reliable gate — tests commonly defer their imports into the function body
(e.g. `def test_x(): from pkg import mod`), which collection never triggers."""
from __future__ import annotations

import re as _re
import shlex
from dataclasses import dataclass, field
from typing import Any


_ENV_ERROR_TOKENS = ("ModuleNotFoundError", "ImportError", "collection error")

def _ensure_required_plugins(c: Any, workdir: str) -> list[str]:
    """If pytest config declares required[_]plugins, pip install each.
    Returns the list of plugin spec strings installed.

    Belt-and-suspenders: if the image is missing a plugin the project
    requires, install it so loading succeeds. The actual qutebrowser pdb
    timeout was NOT caused by missing installs (the plugin was already
    there) — it was caused by `pdb_session._build_session_cmd` disabling
    the plugin via `-p no:`, which then trips the project's
    `required_plugins` check. That fix lives in pdb_session.py."""
    from .required_plugins import detect_required_plugins
    declared = detect_required_plugins(c, workdir)
    installed = []
    for pkg in declared:
        c.exec_bash(f"pip install --quiet {pkg} 2>&1 | tail -n 5")
        installed.append(pkg)
    return installed


@dataclass
class BootstrapReport:
    ready: bool
    env_prefix: str = ""
    actions: list[str] = field(default_factory=list)
    initial_output: str = ""

    def as_transcript_seed(self) -> str:
        header = "PRE-FLIGHT BOOTSTRAP:\n"
        header += "  fixups applied: " + ("; ".join(self.actions) if self.actions else "none") + "\n"
        header += f"  ready={self.ready}\n"
        header += "--- initial pytest output ---\n"
        return header + self.initial_output


def _looks_like_env_error(out: str) -> bool:
    return any(t in out for t in _ENV_ERROR_TOKENS)


def _exec(c: Any, cmd: str) -> tuple[int, str]:
    r = c.exec_bash(cmd)
    combined = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    return r.returncode, combined


def _run_repro(c: Any, prefix: str, repro_nodeid: str) -> tuple[bool, str]:
    """Run the repro for real and return (ready, output).

    ready = output has no env-error tokens. A non-zero rc from an actual
    AssertionError / business-logic failure is fine — that's what we *want*
    the analyzer to see."""
    inner = f"{prefix} pytest -x -s --tb=long {repro_nodeid} 2>&1 | tail -n 200"
    # wrap in `bash -c` + `pipefail` so a broken pipeline doesn't silently rc=0.
    cmd = f"bash -c {shlex.quote('set -o pipefail; ' + inner)}"
    _rc, out = _exec(c, cmd)
    return (not _looks_like_env_error(out)), out


def bootstrap(c: Any, workdir: str, repro_nodeid: str) -> BootstrapReport:
    rep = BootstrapReport(ready=False)
    if not repro_nodeid:
        rep.initial_output = "(no repro test available)"
        return rep

    # Step 0: install pytest plugins declared as required by the project
    rep.actions.extend(f"pip install {p}" for p in _ensure_required_plugins(c, workdir))

    # Step 1: baseline
    prefix = f"cd {workdir} &&"
    ready, out = _run_repro(c, prefix, repro_nodeid)
    rep.initial_output = out
    if ready:
        rep.ready = True
        rep.env_prefix = prefix
        return rep

    # Step 2: PYTHONPATH
    rep.actions.append(f"set PYTHONPATH={workdir}")
    prefix = f"cd {workdir} && PYTHONPATH={workdir}:$PYTHONPATH"
    ready, out = _run_repro(c, prefix, repro_nodeid)
    rep.initial_output = out
    if ready:
        rep.ready = True
        rep.env_prefix = prefix
        return rep

    # Step 3: pip install -e . (only if package metadata exists)
    rc_has_pkg, _ = _exec(c, f"test -f {workdir}/setup.py -o -f {workdir}/pyproject.toml")
    if rc_has_pkg == 0:
        rep.actions.append("pip install -e .")
        _exec(c, f"cd {workdir} && pip install -e . 2>&1 | tail -n 40")

    ready, out = _run_repro(c, prefix, repro_nodeid)
    rep.initial_output = out
    rep.env_prefix = prefix
    rep.ready = ready
    return rep
