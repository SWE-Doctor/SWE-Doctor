"""Run pytest inside a Container and parse the failure summary.

Reuses regexes from run_pro_test.statement_tracer rather than duplicating."""
from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .container import Container

# Import regex from the existing statement_tracer so we don't drift.
_PRO_TEST_DIR = Path(__file__).resolve().parent.parent / "run_pro_test"
if str(_PRO_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_PRO_TEST_DIR))

try:
    from statement_tracer import _PYTEST_SHORT_RE, _TRACEBACK_FRAME_RE  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    import re

    _PYTEST_SHORT_RE = re.compile(
        r"^(\S+\.py):(\d+): (\w+(?:Error|Exception|Warning|Failure))\s*$",
        re.MULTILINE,
    )
    _TRACEBACK_FRAME_RE = re.compile(
        r'^\s*File "([^"]+)", line (\d+)(?:, in (.+))?',
        re.MULTILINE,
    )


@dataclass
class FailedTest:
    nodeid: str
    file: str
    lineno: int
    exc_type: str
    exc_msg: str = ""


@dataclass
class PytestResult:
    returncode: int
    stdout: str
    stderr: str
    failed_tests: list[FailedTest] = field(default_factory=list)

    @property
    def combined(self) -> str:
        return self.stdout + (f"\n--- STDERR ---\n{self.stderr}" if self.stderr else "")


def _parse_short_summary(text: str) -> list[FailedTest]:
    """Parse pytest's short test summary table.

    Pytest emits 'FAILED path::nodeid - ExcType: msg' lines near the end;
    we also catch the '<file>:<lineno>: ExcType' short-line form."""
    import re

    out: list[FailedTest] = []
    summary_re = re.compile(
        r"^FAILED\s+(\S+?::[^\s]+?)(?:\s+-\s+(\w+(?:Error|Exception|Warning|Failure))\s*:?\s*(.*))?$",
        re.MULTILINE,
    )
    shorts = {(m.group(1), int(m.group(2))): m.group(3) for m in _PYTEST_SHORT_RE.finditer(text)}
    for m in summary_re.finditer(text):
        nodeid = m.group(1)
        exc_type = (m.group(2) or "").strip()
        exc_msg = (m.group(3) or "").strip()
        file = nodeid.split("::", 1)[0]
        lineno = 0
        for (f, ln), et in shorts.items():
            if f.endswith(file) or file.endswith(f):
                lineno = ln
                if not exc_type:
                    exc_type = et
                break
        out.append(FailedTest(nodeid=nodeid, file=file, lineno=lineno,
                              exc_type=exc_type, exc_msg=exc_msg))
    return out


def run_pytest(
    c: Container, args: str, cwd: str | None = None,
    env_prefix: str = "", timeout: int = 300,
) -> PytestResult:
    cwd = cwd or c.workdir
    head = env_prefix.strip() if env_prefix.strip() else f"cd {shlex.quote(cwd)} &&"
    cmd = f"{head} pytest {args}"
    r = c.exec_bash(cmd, timeout=timeout)
    failed = _parse_short_summary(r.stdout + "\n" + r.stderr)
    return PytestResult(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr, failed_tests=failed)
