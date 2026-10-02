"""Build the in-container entryscript for the Verified repro-trace stage.

Differs from the Pro entryscript (run_pro_test/python_runner.build_entryscript):
  - activates the `testbed` conda env (Verified images need this)
  - works in /testbed (not /app)
  - no run_script.sh / before_repo_set_cmd; runs plain pytest on the staged tests
  - reuses the focused-trace plugin codegen from statement_tracer so phase-2
    coverage (phase2_coverage.json) is produced the same way for downstream RCA.
"""
from __future__ import annotations

import base64
import importlib
import sys
from pathlib import Path

from verified_common import VERIFIED_CWD, VERIFIED_CONDA_ACTIVATE

# statement_tracer uses bare imports (from common import ...) and lives in
# run_pro_test/ which is NOT a regular package — its sibling modules are
# imported as top-level names. We temporarily add run_pro_test/ to sys.path
# so the import resolves correctly, then restore it.
_PRO_DIR = str(Path(__file__).resolve().parents[1] / "run_pro_test")


def _get_generate_settrace_plugin_code():
    """Import generate_settrace_plugin_code from run_pro_test/statement_tracer.py.

    statement_tracer uses bare sibling imports so we must add run_pro_test/ to
    sys.path for the duration of the import, then remove it.
    """
    added = _PRO_DIR not in sys.path
    if added:
        sys.path.insert(0, _PRO_DIR)
    try:
        mod = importlib.import_module("statement_tracer")
        return mod.generate_settrace_plugin_code
    finally:
        if added and _PRO_DIR in sys.path:
            sys.path.remove(_PRO_DIR)


WORKDIR = VERIFIED_CWD
CONDA = VERIFIED_CONDA_ACTIVATE


def _conftest_loader() -> str:
    return (
        "try:\n"
        "    import sys as _sys\n"
        "    if '/workspace' not in _sys.path:\n"
        "        _sys.path.insert(0, '/workspace')\n"
        "    import focused_trace_plugin  # noqa: F401\n"
        "except Exception:\n"
        "    pass\n"
    )


def build_entryscript_verified(base_commit: str, repro_rel_paths: list[str]) -> str:
    generate_settrace_plugin_code = _get_generate_settrace_plugin_code()
    settrace_b64 = base64.b64encode(generate_settrace_plugin_code().encode()).decode()
    tests_csv = " ".join(repro_rel_paths)
    conftest = _conftest_loader().replace("'", "'\\''")
    lines = [
        "set -e",
        CONDA,
        f"cd {WORKDIR}",
        f"git checkout -f {base_commit} 2>/dev/null || true",
        f"mkdir -p {WORKDIR}/_repro_tests",
        f"cp /workspace/_repro_tests/*.py {WORKDIR}/_repro_tests/ 2>/dev/null || true",
        f"touch {WORKDIR}/_repro_tests/__init__.py || true",
        f"echo '{settrace_b64}' | base64 -d > /workspace/focused_trace_plugin.py 2>/dev/null || true",
        f"printf '%s' '{conftest}' >> {WORKDIR}/conftest.py",
        "export PYTHONPATH=/workspace:${PYTHONPATH:-}",
        # No --junitxml here so the coverage-run pytest cannot overwrite pytest-report.xml.
        'export PYTEST_ADDOPTS="--tb=long -rA"',
        # Capture real pytest exit code without letting set -e abort on test failures.
        "set +e",
        f"python -m pytest {tests_csv} --junitxml=/workspace/pytest-report.xml"
        " > /workspace/stdout.log 2> /workspace/stderr.log",
        "echo $? > /workspace/pytest_exit_code.txt",
        "set -e",
        "python -m pip install --quiet 'coverage>=7.0' 2>/dev/null || true",
        f"python -m coverage run --source={WORKDIR} -m pytest {tests_csv}"
        " --junitxml=/workspace/focused-report.xml"
        " > /workspace/focused_stdout.log 2> /workspace/focused_stderr.log || true",
        "python -m coverage json -o /workspace/phase2_coverage.json --pretty-print 2>/dev/null || true",
    ]
    return "\n".join(lines) + "\n"
