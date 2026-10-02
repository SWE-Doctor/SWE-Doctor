"""Execute tests and commands on a repository (c_old).

Supports both local subprocess execution and Docker containers
via a unified interface wrapping mini-swe-agent's environments.
"""

import logging
import re
from typing import Any, Protocol

logger = logging.getLogger("repro_test.executor")


class Environment(Protocol):
    """Minimal environment interface — execute a shell command, get output."""
    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]: ...


def _go_test_name(test_code: str) -> str:
    """Extract the first `func TestXxx(` name from generated Go test code."""
    m = re.search(r"func\s+(Test\w+)\s*\(", test_code)
    return m.group(1) if m else ""


def run_test(
    test_code: str,
    env: Environment,
    cwd: str = "/app",
    timeout: int = 60,
    test_filename: str = "repro_test.py",
    language: str = "python",
) -> dict:
    """Write test_code to cwd/test_filename and run it.

    python: runs `python -m pytest`, parses pytest output, passed = rc==0.
    go:     mkdir-s the test's package dir, runs `go test ./<pkg>` scoped to that
            package and the generated test func (GOFLAGS=-mod=mod patches go.sum
            on go1.24, but is dropped in workspace-mode repos that reject it),
            parses output via langpack.parse_test_output, passed = rc==0 and no
            error_type.

    Returns dict with keys: passed, returncode, output, error_type, error_message.
    """
    # Write the test file (go test files may live in a nested package dir).
    if language == "go":
        env.execute({"command": f"mkdir -p $(dirname {cwd}/{test_filename})"}, cwd=cwd, timeout=10)
    write_cmd = f"cat > {cwd}/{test_filename} << 'REPRO_TEST_EOF'\n{test_code}\nREPRO_TEST_EOF"
    env.execute({"command": write_cmd}, cwd=cwd, timeout=10)

    # Run the test
    if language == "go":
        test_name = _go_test_name(test_code)
        run_part = f"-run {test_name} " if test_name else ""
        # Scope the build to the test's own package — `./...` would compile the
        # whole repo and fail on unrelated/cgo packages (missing C headers) in
        # big repos like teleport. Workspace-mode repos (go.work, e.g. flipt)
        # reject `-mod=mod` ("may only be set to readonly or vendor"), so only
        # set it outside a workspace (where it patches go.sum on go1.24).
        pkg_dir = test_filename.rsplit("/", 1)[0] if "/" in test_filename else "."
        run_cmd = (f"cd {cwd} && export PATH=/usr/local/go/bin:/go/bin:$PATH && "
                   f"if [ -f go.work ]; then export GOFLAGS=; else export GOFLAGS=-mod=mod; fi && "
                   f"go test ./{pkg_dir} {run_part}-v 2>&1")
    else:
        run_cmd = f"cd {cwd} && python -m pytest {test_filename} -xvs --tb=long --no-header 2>&1"
    result = env.execute({"command": run_cmd}, cwd=cwd, timeout=timeout)

    output = result.get("output", "")
    returncode = result.get("returncode", -1)
    if language == "go":
        from .langpack import parse_test_output
        error_type, error_message = parse_test_output(language, output)
        passed = returncode == 0 and not error_type
    else:
        passed = returncode == 0
        error_type, error_message = _parse_error(output)

    # Cleanup
    env.execute({"command": f"rm -f {cwd}/{test_filename}"}, cwd=cwd, timeout=5)

    return {
        "passed": passed,
        "returncode": returncode,
        "output": output,
        "error_type": error_type,
        "error_message": error_message,
    }


def run_command(command: str, env: Environment, cwd: str = "/app", timeout: int = 30) -> dict:
    """Run a shell command. Returns dict with output, returncode."""
    result = env.execute({"command": command}, cwd=cwd, timeout=timeout)
    return {"output": result.get("output", ""), "returncode": result.get("returncode", -1)}


def _ensure_pytest(env: Environment, cwd: str = "/testbed", timeout: int = 120) -> None:
    """Install pytest in the environment if it is not importable.

    Some SWE-bench Verified images (notably django and sympy) ship the
    project's own test runner but not pytest, so a generated pytest-style BRT
    fails at collection with "No module named pytest" — a spurious failure
    unrelated to the bug, which also poisons the repair-loop reflection. Install
    pytest on demand so the BRT's pass/fail reflects the reproduction logic.
    """
    probe = env.execute({"command": "python -c 'import pytest'"}, cwd=cwd, timeout=timeout)
    if probe.get("returncode", 1) == 0:
        return
    logger.info("pytest missing in environment; installing on demand")
    env.execute({"command": "pip install -q pytest"}, cwd=cwd, timeout=timeout)


def _parse_error(output: str) -> tuple[str, str]:
    """Extract error type and message from pytest output.

    Parse strategy (in priority order):
    1. pytest short test summary info section (most reliable)
    2. Traceback last-line pattern (e.g. "ModuleNotFoundError: No module named 'x'")
    3. FAILED / ERROR lines as fallback
    """
    lines = output.splitlines()

    # Strategy 1: parse "=== short test summary info ===" section
    in_summary = False
    for line in lines:
        stripped = line.strip()
        if "short test summary info" in stripped:
            in_summary = True
            continue
        if in_summary:
            if stripped.startswith("="):
                break
            if stripped.startswith("FAILED") or stripped.startswith("ERROR"):
                # e.g. "FAILED test.py::test_foo - AssertionError: ..."
                if " - " in stripped:
                    err_part = stripped.split(" - ", 1)[1]
                    if ": " in err_part:
                        etype, emsg = err_part.split(": ", 1)
                        return etype.strip(), emsg.strip()
                    return err_part.strip(), ""
                return "TestFailure" if stripped.startswith("FAILED") else "TestError", stripped

    # Strategy 2: traceback last line — scan from bottom for "ErrorType: message"
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("=") or stripped.startswith("-"):
            continue
        if ": " in stripped and not stripped.startswith(" "):
            parts = stripped.split(": ", 1)
            candidate = parts[0].split()[-1] if parts[0] else ""
            if candidate.endswith("Error") or candidate.endswith("Exception"):
                return candidate, parts[1] if len(parts) > 1 else ""

    # Strategy 3: fallback — FAILED / ERROR keywords
    for line in lines:
        stripped = line.strip()
        if "FAILED" in stripped:
            return "TestFailure", stripped
        if "ERROR" in stripped:
            return "TestError", stripped

    return "", ""
