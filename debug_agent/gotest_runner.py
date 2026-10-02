"""Run `go test` inside a Container and parse the failure summary.

Mirrors pytest_runner / jest_runner; reuses run_pro_test/go_runner's mature
extract_failed_tests (handles `--- FAIL`, `go test -json`, and build errors)
rather than re-deriving the regexes here. The `gotest` action is non-debug test
evidence for the gate — the Go analog of pytest/jest."""
from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .container import Container
from .pytest_runner import FailedTest

# Import the extractor from the existing go_runner so we don't drift. Mirrors
# langpack.parse_test_output's go branch: insert run_pro_test on sys.path, import.
_PRO = Path(__file__).resolve().parent.parent / "run_pro_test"
if str(_PRO) not in sys.path:
    sys.path.insert(0, str(_PRO))
from go_runner import extract_failed_tests  # type: ignore


@dataclass
class GoTestResult:
    returncode: int
    stdout: str
    stderr: str
    failed_tests: list[FailedTest] = field(default_factory=list)

    @property
    def combined(self) -> str:
        return self.stdout + (f"\n--- STDERR ---\n{self.stderr}" if self.stderr else "")


def parse_go_failures(text: str) -> list[FailedTest]:
    """Failed go test names (or [build-error]/[build-fail] markers) as FailedTest.
    Go output gives us the test name but not a single source file/line, so file
    and lineno stay empty — downstream consumers tolerate that."""
    names = extract_failed_tests(text or "", "")
    return [FailedTest(nodeid=n, file="", lineno=0, exc_type="TestFailure", exc_msg=n)
            for n in names]


def run_gotest(c: Container, args: str = "", cwd: str | None = None,
               env_prefix: str = "", timeout: int = 300) -> GoTestResult:
    """Run `go test ./...` in the container; `args` (when set) is the -run pattern.

    Command: cd <cwd> && export PATH=/usr/local/go/bin:/go/bin:$PATH &&
             GOFLAGS=-mod=mod go test ./... -run <name> -v
    The PATH export is required because the login shell (`bash -lc`) drops go from
    PATH; GOFLAGS=-mod=mod lets go1.24 backfill go.sum instead of failing readonly."""
    cwd = cwd or c.workdir
    run_sel = f" -run {shlex.quote(args.strip())}" if args and args.strip() else ""
    head = env_prefix.strip() if env_prefix.strip() else f"cd {shlex.quote(cwd)} &&"
    cmd = (
        f"{head} export PATH=/usr/local/go/bin:/go/bin:$PATH && "
        f"GOFLAGS=-mod=mod go test ./...{run_sel} -v"
    )
    r = c.exec_bash(cmd, timeout=timeout)
    return GoTestResult(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr,
                        failed_tests=parse_go_failures(r.stdout + "\n" + r.stderr))
