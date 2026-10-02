"""Python / pytest runner: entryscript generation and output analysis."""

from __future__ import annotations

import json
import re
import textwrap
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from common import (
    InstanceResult,
    extract_failure_trace_excerpt,
    extract_patch_files,
    extract_stack_files,
    extract_stack_files_from_text,
    get_error_excerpt,
    load_execution_files,
    make_error_result,
    match_patch_files,
    normalize_path,
    prepare_workspace,
    run_container,
    strip_ansi,
)
from statement_tracer import generate_settrace_plugin_code



PYTEST_PROGRESS_FAILED_RE = re.compile(
    r"^(?P<nodeid>\S+::\S+)\s+(?:FAILED|ERROR)\s+\[\s*\d+%\s*\]\s*$"
)
PYTEST_SESSION_END_RE = re.compile(
    r"=+\s+.*\bin\s+[0-9.]+s(?:\s+\([0-9:]+\))?\s+=+"
)
PYTEST_FAILURE_BANNER_RE = re.compile(r"^=+\s+FAILURES\s+=+\s*$", re.MULTILINE)
PYTEST_ERROR_BANNER_RE = re.compile(r"^=+\s+ERRORS\s+=+\s*$", re.MULTILINE)
PYTEST_SHORT_SUMMARY_FAILED_RE = re.compile(r"^FAILED\s+\S+::\S+", re.MULTILINE)
PYTEST_SHORT_SUMMARY_ERROR_RE = re.compile(r"^ERROR\s+\S+::\S+", re.MULTILINE)
PYTEST_COUNT_FAILED_RE = re.compile(r"=+\s+\d+\s+failed(?:,\s+|\s+in\s+)", re.IGNORECASE)
PYTEST_COUNT_ERROR_RE = re.compile(r"=+\s+\d+\s+error(?:s)?(?:,\s+|\s+in\s+)", re.IGNORECASE)



# ── Pytest output analysis ────────────────────────────────────────────────────

def extract_failed_tests(stdout_text: str, stderr_text: str) -> list[str]:
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = PYTEST_PROGRESS_FAILED_RE.match(line)
        if match:
            nodeid = match.group("nodeid")
            if nodeid not in seen:
                seen.add(nodeid)
                out.append(nodeid)
            continue
        for prefix in ("FAILED ", "ERROR "):
            if line.startswith(prefix):
                maybe_nodeid = line.split(" - ", 1)[0].replace(prefix, "", 1).strip()
                if maybe_nodeid and maybe_nodeid not in seen:
                    seen.add(maybe_nodeid)
                    out.append(maybe_nodeid)
                break
    return out


def has_pytest_failures(stdout_text: str, stderr_text: str) -> bool:
    if extract_failed_tests(stdout_text, stderr_text):
        return True
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))
    return bool(
        PYTEST_FAILURE_BANNER_RE.search(text)
        or PYTEST_ERROR_BANNER_RE.search(text)
        or PYTEST_SHORT_SUMMARY_FAILED_RE.search(text)
        or PYTEST_SHORT_SUMMARY_ERROR_RE.search(text)
        or PYTEST_COUNT_FAILED_RE.search(text)
        or PYTEST_COUNT_ERROR_RE.search(text)
    )


def is_pytest_output_incomplete(stdout_text: str, stderr_text: str) -> bool:
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))
    low = text.lower()
    has_session_start = "test session starts" in low
    has_progress = bool(
        re.search(
            r"^\S+::\S+\s+(PASSED|FAILED|SKIPPED|ERROR|RERUN)\s+\[\s*\d+%\s*\]\s*$",
            text,
            re.MULTILINE,
        )
    )
    has_session_end = bool(PYTEST_SESSION_END_RE.search(text))
    return bool(has_session_start and has_progress and not has_session_end)


# ── JUnit XML analysis ────────────────────────────────────────────────────────

def extract_junit_failure_excerpt(
    junit_xml_text: str, max_cases: int = 20, max_case_lines: int = 40
) -> str:
    if not junit_xml_text.strip():
        return ""
    try:
        root = ET.fromstring(junit_xml_text)
    except ET.ParseError:
        return ""

    entries: list[str] = []
    for testcase in root.iter("testcase"):
        classname = testcase.attrib.get("classname", "").strip()
        name = testcase.attrib.get("name", "").strip()
        nodeid = "::".join([item for item in [classname, name] if item])
        if not nodeid:
            nodeid = testcase.attrib.get("name", "").strip() or "<unknown-testcase>"
        failure = testcase.find("failure") or testcase.find("error")
        if failure is None:
            continue
        message = (failure.attrib.get("message") or "").strip()
        content = (failure.text or "").strip()
        content_lines = [line for line in content.splitlines() if line.strip()][:max_case_lines]
        block_lines = [f"{nodeid}"]
        if message:
            block_lines.append(f"message: {message}")
        if content_lines:
            block_lines.extend(content_lines)
        entries.append("\n".join(block_lines))
        if len(entries) >= max_cases:
            break
    return "\n\n---\n\n".join(entries).strip()


def extract_stack_files_from_junit_xml(junit_xml_text: str) -> list[str]:
    if not junit_xml_text.strip():
        return []
    try:
        root = ET.fromstring(junit_xml_text)
    except ET.ParseError:
        return []

    snippets: list[str] = []
    for testcase in root.iter("testcase"):
        failure = testcase.find("failure")
        error = testcase.find("error")
        if failure is None and error is None:
            continue

        file_attr = normalize_path(testcase.attrib.get("file", ""))
        if file_attr:
            snippets.append(file_attr)

        classname = testcase.attrib.get("classname", "").strip()
        if classname and "." in classname:
            dotted = classname.strip(".")
            snippets.append(dotted.replace(".", "/") + ".py")
            snippets.append(dotted.replace(".", "/") + "/__init__.py")

        for tag in ("failure", "error", "system-out", "system-err"):
            node = testcase.find(tag)
            if node is None:
                continue
            if node.text:
                snippets.append(node.text)
            message = (node.attrib.get("message") or "").strip()
            if message:
                snippets.append(message)

    return extract_stack_files_from_text("\n".join(snippets))


_FRAMEWORK_ANALYSIS_RE = re.compile(
    r"^(?:_pytest|pluggy|pytest|py)/|"
    r"^(?:jinja2|yaml|coverage|pip|setuptools|pkg_resources|importlib_metadata|"
    r"exceptiongroup|tomli|iniconfig|anyio|filelock|platformdirs|packaging|"
    r"pytest_rerunfailures|pytest_mock|xdist|execnet|apipkg|passlib|cryptography)/"
)


def load_focused_execution_files(workspace: Path) -> list[str]:
    """Load per-test focused execution traces from focused_<pid>_<nodeid>.log files.

    These are written by focused_trace_plugin only for FAILING tests (production
    code only, already filtered of test files). Files are returned in call order
    across all failing tests — earlier entries = called closer to the test entry
    point = more likely to be the buggy module.
    """
    from common import normalize_path
    seen: set[str] = set()
    ordered: list[str] = []
    for path in sorted(workspace.glob("focused_*.log")):
        # Skip Phase 2 stdout/stderr files that also match focused_*.log
        if path.name in ("focused_stdout.log", "focused_stderr.log"):
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            item = normalize_path(line)
            if item and item not in seen and not _FRAMEWORK_ANALYSIS_RE.search(item):
                seen.add(item)
                ordered.append(item)
    return ordered


def load_extra_junit_reports(workspace: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    junit_dir = workspace / "extra_artifacts" / "junit"
    if not junit_dir.exists():
        return out
    for xml_path in sorted(junit_dir.glob("*.xml")):
        try:
            text = xml_path.read_text(errors="replace")
        except Exception:
            continue
        if text.strip():
            out.append((xml_path.name, text))
    return out


def load_phase2_coverage_files(workspace: Path) -> list[str]:
    """Parse coverage.py JSON from Phase 2 rerun to get executed production files."""
    import json as _json
    cov_path = workspace / "phase2_coverage.json"
    if not cov_path.exists():
        return []
    try:
        data = _json.loads(cov_path.read_text(errors="replace"))
    except Exception:
        return []
    _TEST_RE = re.compile(
        r"(?:^|/)(?:tests?|test_[^/]+|[^/]+_test|conftest)/|"
        r"(?:^|/)test_[^/]+\.py$|[^/]+_test\.py$",
        re.IGNORECASE,
    )
    files = []
    for abs_path in data.get("files", {}).keys():
        abs_path = abs_path.replace("\\", "/")
        if "/app/" in abs_path:
            rel = abs_path[abs_path.find("/app/") + 5:].lstrip("./")
        else:
            # coverage.py sometimes uses relative paths (e.g. "lib/ansible/foo.py")
            rel = abs_path.lstrip("./")
        if not rel or not rel.endswith(".py"):
            continue
        low = rel.lower()
        if "/site-packages/" in low or "/dist-packages/" in low:
            continue
        if _TEST_RE.search(rel):
            continue
        f = normalize_path(rel)
        if f:
            files.append(f)
    return sorted(set(files))


def extract_junit_failure_excerpt_many(
    junit_reports: list[tuple[str, str]], max_reports: int = 5
) -> str:
    entries: list[str] = []
    for report_name, report_text in junit_reports[:max_reports]:
        excerpt = extract_junit_failure_excerpt(report_text)
        if excerpt:
            entries.append(f"[{report_name}]\n{excerpt}")
    return "\n\n===\n\n".join(entries).strip()


# ── Entryscript ───────────────────────────────────────────────────────────────

def build_entryscript(sample: dict, extra_preamble: str = "") -> str:
    selected_tests = sample.get("selected_test_files_to_run", "")
    if isinstance(selected_tests, str):
        try:
            selected_tests = json.loads(selected_tests.replace("'", '"'))
        except Exception:
            try:
                selected_tests = eval(selected_tests)  # noqa: S307
            except Exception:
                selected_tests = []
    if not isinstance(selected_tests, list):
        selected_tests = []
    selected_tests_csv = ",".join(selected_tests)

    before_repo_set_cmd = (sample.get("before_repo_set_cmd") or "").strip()
    base_commit = sample.get("base_commit")

    # Pre-encode the settrace plugin for Phase 2 (line-level tracing).
    # We base64-encode it to avoid f-string / heredoc escaping issues.
    import base64 as _b64
    _settrace_code = generate_settrace_plugin_code()
    _settrace_b64 = _b64.b64encode(_settrace_code.encode()).decode()

    script = f"""\
set -e
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
{before_repo_set_cmd}
# ── Optional extra preamble (e.g. staging repro test files) ──────────────────
{extra_preamble}
# ── Phase 1: per-test focused tracer (works in xdist workers via pytest hooks) ─
cat > /workspace/focused_trace_plugin.py <<'PY'
import atexit
import os
import re
import sys
from pathlib import Path

# _order: pid -> {{nodeid -> list[str]}}  (insertion order = call order)
# _seen_set: pid -> {{nodeid -> set[str]}}  (fast dedup)
_order = {{}}
_seen_set = {{}}
_current = {{}}  # pid -> current nodeid

# Test-file patterns to exclude from focused traces (they are not buggy production code)
_TEST_RE = re.compile(
    r"(?:^|/)(?:tests?|testing|test_[^/]+|[^/]+_test|conftest|__fixtures__)/"
    r"|(?:^|/)test_[^/]+\\.py$|[^/]+_test\\.py$",
    re.IGNORECASE,
)
# Framework files to exclude (pytest internals, third-party libs not part of the project)
_FRAMEWORK_RE = re.compile(
    r"^(?:_pytest|pluggy|pytest|py)/|"
    r"^(?:jinja2|yaml|coverage|pip|setuptools|pkg_resources|importlib_metadata|"
    r"exceptiongroup|tomli|iniconfig|anyio|filelock|platformdirs|packaging|"
    r"pytest_rerunfailures|pytest_mock|xdist|execnet|apipkg)/|"
    r"^(?:_pytest|pluggy|pytest|py)/",
)

def _norm(filename):
    filename = (filename or "").replace("\\\\", "/")
    # Prefer /app/ source path
    if "/app/" in filename:
        low = filename.lower()
        if "/site-packages/" in low or "/dist-packages/" in low:
            # Production code installed in site-packages: extract package-relative path
            for marker in ("/site-packages/", "/dist-packages/"):
                pos = low.find(marker)
                if pos >= 0:
                    return filename[pos + len(marker):].lstrip("./")
            return ""
        idx = filename.find("/app/")
        rel = filename[idx + len("/app/"):]
        return rel.lstrip("./")
    # Fallback: site-packages outside /app/ (e.g. system Python)
    low = filename.lower()
    for marker in ("/site-packages/", "/dist-packages/"):
        pos = low.find(marker)
        if pos >= 0:
            rel = filename[pos + len(marker):].lstrip("./")
            # Only keep project-like paths (not third-party libs like yaml/, jinja2/ etc)
            # Heuristic: project code has at least one directory level
            if "/" in rel:
                return rel
    return ""

def _make_profile(pid):
    def _profile(frame, event, arg):
        if event != "call":
            return _profile
        fn = frame.f_code.co_filename or ""
        if fn.endswith(".py"):
            rel = _norm(fn)
            if rel:
                tid = _current.get(pid)
                if tid:
                    ss = _seen_set.setdefault(pid, {{}}).setdefault(tid, set())
                    if rel not in ss:
                        ss.add(rel)
                        _order.setdefault(pid, {{}}).setdefault(tid, []).append(rel)
        return _profile
    return _profile

def _write_focused(pid, tid):
    files = [f for f in _order.get(pid, {{}}).get(tid, []) if not _TEST_RE.search(f) and not _FRAMEWORK_RE.search(f)]
    if not files:
        return
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in tid)[:120]
    out = Path("/workspace") / f"focused_{{os.getpid()}}_{{safe}}.log"
    if not out.exists():
        out.write_text("\\n".join(files) + "\\n")

def _dump_on_exit():
    # Forked children (--forked) collect trace data but their pytest hooks never
    # fire in the parent. Write out whatever was collected when this process exits.
    for pid, tests in _order.items():
        for tid in tests:
            _write_focused(pid, tid)

atexit.register(_dump_on_exit)

def pytest_runtest_setup(item):
    pid = os.getpid()
    _current[pid] = item.nodeid
    sys.setprofile(_make_profile(pid))

def pytest_runtest_teardown(item, nextitem):
    pid = os.getpid()
    sys.setprofile(None)
    _current.pop(pid, None)

def pytest_runtest_logreport(report):
    if report.when == "call" and report.failed:
        pid = os.getpid()
        _write_focused(pid, report.nodeid)
PY
# ── Phase 1: session-level tracer (fallback for non-xdist runs) ────────────────
cat > /workspace/exec_trace_plugin.py <<'PY'
import atexit
import os
import sys
from pathlib import Path

_seen = set()

def _norm(filename: str) -> str:
    filename = (filename or "").replace("\\\\", "/")
    if "/app/" in filename:
        low = filename.lower()
        if "/site-packages/" in low or "/dist-packages/" in low:
            for marker in ("/site-packages/", "/dist-packages/"):
                pos = low.find(marker)
                if pos >= 0:
                    return filename[pos + len(marker):].lstrip("./")
            return ""
        idx = filename.find("/app/")
        rel = filename[idx + len("/app/"):]
        return rel.lstrip("./")
    low = filename.lower()
    for marker in ("/site-packages/", "/dist-packages/"):
        pos = low.find(marker)
        if pos >= 0:
            rel = filename[pos + len(marker):].lstrip("./")
            if "/" in rel:
                return rel
    return ""

def _profile(frame, event, arg):
    if event != "call":
        return _profile
    filename = frame.f_code.co_filename or ""
    if not filename.endswith(".py"):
        return _profile
    rel = _norm(filename)
    if rel:
        _seen.add(rel)
    return _profile

def _dump():
    if not _seen:
        return
    out = Path("/workspace") / f"executed_files.{{os.getpid()}}.log"
    out.write_text("\\n".join(sorted(_seen)) + "\\n")

def pytest_sessionstart(session):
    sys.setprofile(_profile)

def pytest_sessionfinish(session, exitstatus):
    sys.setprofile(None)
    _dump()

atexit.register(_dump)
PY
# ── Inject plugins via conftest.py (works with ansible-test, openlibrary, etc.) ──
# Instead of relying on PYTEST_ADDOPTS -p (which can be ignored by test wrappers
# like ansible-test, or fail with ModuleNotFoundError), we inject a conftest.py
# at /app root that conditionally registers our trace plugins.
export PYTHONPATH="/workspace:${{PYTHONPATH:-}}"
# Only add basic pytest options (no -p plugin), plugins loaded via conftest
export PYTEST_ADDOPTS="${{PYTEST_ADDOPTS:-}} --tb=long -rA --showlocals --junitxml=/workspace/pytest-report.xml"
# Create a conftest-based loader at /app root (append to existing conftest.py if present)
python3 - <<'INJECT_PY'
from pathlib import Path
loader = '''
# === Injected by execution engine: trace plugin loader ===
try:
    import sys as _sys
    if "/workspace" not in _sys.path:
        _sys.path.insert(0, "/workspace")
    import focused_trace_plugin  # noqa: F401
    import exec_trace_plugin  # noqa: F401
except ImportError:
    pass
# === End injected loader ===
'''
conftest = Path("/app/conftest.py")
existing = conftest.read_text(errors="replace") if conftest.exists() else ""
if "trace plugin loader" not in existing:
    conftest.write_text(existing + "\\n" + loader)
INJECT_PY
# ── Inject sitecustomize.py for subprocess tracing (ansible-test, tox, etc.) ──
# ansible-test spawns a separate Python subprocess with its own pytest config,
# bypassing conftest.py injection. sitecustomize.py runs at Python startup in
# EVERY subprocess, so we get session-level tracing even in ansible-test workers.
cat > /workspace/sitecustomize.py <<'PY'
import atexit as _atexit
import os as _os
import sys as _sys
from pathlib import Path as _Path

_seen = set()
_MARKER = "_SITECUSTOMIZE_TRACE_ACTIVE"

def _norm(filename):
    filename = (filename or "").replace("\\\\", "/")
    if "/app/" in filename:
        low = filename.lower()
        if "/site-packages/" in low or "/dist-packages/" in low:
            for marker in ("/site-packages/", "/dist-packages/"):
                pos = low.find(marker)
                if pos >= 0:
                    return filename[pos + len(marker):].lstrip("./")
            return ""
        idx = filename.find("/app/")
        return filename[idx + 5:].lstrip("./")
    low = filename.lower()
    for marker in ("/site-packages/", "/dist-packages/"):
        pos = low.find(marker)
        if pos >= 0:
            rel = filename[pos + len(marker):].lstrip("./")
            if "/" in rel:
                return rel
    return ""

def _profile(frame, event, arg):
    if event != "call":
        return _profile
    fn = frame.f_code.co_filename or ""
    if fn.endswith(".py"):
        rel = _norm(fn)
        if rel:
            _seen.add(rel)
    return _profile

def _dump():
    if not _seen:
        return
    out = _Path("/workspace") / f"executed_files.site.{{_os.getpid()}}.log"
    try:
        out.write_text("\\n".join(sorted(_seen)) + "\\n")
    except Exception:
        pass

if not _os.environ.get(_MARKER):
    _os.environ[_MARKER] = "1"
    _sys.setprofile(_profile)
    _atexit.register(_dump)

# Chain-load existing sitecustomize if present
import importlib as _importlib
_sys.modules.pop("sitecustomize", None)
# Temporarily remove /workspace from path to avoid infinite recursion
_orig_path = list(_sys.path)
_sys.path = [p for p in _sys.path if p != "/workspace"]
try:
    _importlib.import_module("sitecustomize")
except ImportError:
    pass
finally:
    _sys.path = _orig_path
PY
set +e
bash /workspace/run_script.sh "{selected_tests_csv}" > /workspace/stdout.log 2> /workspace/stderr.log
pytest_exit_code=$?
needs_rerun=0
if [ "$pytest_exit_code" -ne 0 ]; then
  needs_rerun=1
else
  python - <<'PY'
import re
from pathlib import Path

stdout = Path("/workspace/stdout.log").read_text(errors="replace")
stderr = Path("/workspace/stderr.log").read_text(errors="replace")
text = "\\n".join([stdout, stderr])
patterns = [
    re.compile(r"^\\S+::\\S+\\s+FAILED\\s+\\[\\s*\\d+%\\s*\\]\\s*$", re.MULTILINE),
    re.compile(r"^FAILED\\s+\\S+::\\S+", re.MULTILINE),
    re.compile(r"^=+\\s+FAILURES\\s+=+\\s*$", re.MULTILINE),
    re.compile(r"=+\\s+\\d+\\s+failed(?:,\\s+|\\s+in\\s+)", re.IGNORECASE),
    re.compile(r"^\\S+::\\S+\\s+ERROR\\s+\\[\\s*\\d+%\\s*\\]\\s*$", re.MULTILINE),
    re.compile(r"=+\\s+\\d+\\s+error(?:s)?(?:,\\s+|\\s+in\\s+)", re.IGNORECASE),
]
has_failure = any(p.search(text) for p in patterns)
raise SystemExit(0 if has_failure else 1)
PY
  if [ "$?" -eq 0 ]; then
    needs_rerun=1
  fi
fi
if [ "$needs_rerun" -eq 1 ]; then
  # Verbose rerun for better traceback detail
  export PYTEST_ADDOPTS="${{PYTEST_ADDOPTS:-}} --full-trace -vv -s"
  bash /workspace/run_script.sh "{selected_tests_csv}" > /workspace/rerun_stdout.log 2> /workspace/rerun_stderr.log
  printf "%s\\n" "$?" > /workspace/rerun_pytest_exit_code.txt
  # ── Phase 2: focused rerun of ONLY failing tests, single-threaded ────────────
  # Extract failing test node IDs from Phase 1 output
  python - <<'PY'
import re, json
from pathlib import Path

stdout = Path("/workspace/stdout.log").read_text(errors="replace")
stderr = Path("/workspace/stderr.log").read_text(errors="replace")
rerun_stdout = ""
rerun_stderr = ""
try:
    rerun_stdout = Path("/workspace/rerun_stdout.log").read_text(errors="replace")
except Exception:
    pass
try:
    rerun_stderr = Path("/workspace/rerun_stderr.log").read_text(errors="replace")
except Exception:
    pass
_raw = "\\n".join([stdout, stderr, rerun_stdout, rerun_stderr])
# Drop pathologically long lines (e.g. --showlocals dumps of zipdata/base64 blobs)
# to avoid catastrophic regex backtracking on the greedy \\S+ patterns below.
text = "\\n".join(l for l in _raw.splitlines() if len(l) <= 4096)
seen = set()
failed = []
for pat in [
    re.compile(r"^FAILED\\s+(\\S+::\\S+)", re.MULTILINE),
    re.compile(r"^(\\S+::\\S+)\\s+FAILED\\s+\\[\\s*\\d+%\\s*\\]\\s*$", re.MULTILINE),
    re.compile(r"^(\\S+::\\S+)\\s+ERROR\\s+\\[\\s*\\d+%\\s*\\]\\s*$", re.MULTILINE),
]:
    for m in pat.finditer(text):
        nid = m.group(1).strip()
        if nid and nid not in seen:
            seen.add(nid)
            failed.append(nid)

# If no pytest-format node IDs found, try to extract test file paths from
# ansible-test or other wrapper output (e.g. "FAIL test/units/foo/test_bar.py")
if not failed:
    test_file_pats = [
        re.compile(r"FAIL[ED]*\\s+(test\\S+\\.py(?:::\\S+)?)", re.MULTILINE),
        re.compile(r"(test/units/\\S+\\.py)\\s", re.MULTILINE),
    ]
    test_files_seen = set()
    for pat in test_file_pats:
        for m in pat.finditer(text):
            tf = m.group(1).strip()
            if tf and tf not in test_files_seen:
                test_files_seen.add(tf)
                failed.append(tf)

# Sanitize node IDs: pytest can't parse parametrize IDs with spaces/colons/quotes.
# Strip the parametrize suffix [xxx] if it contains problematic characters so that
# pytest runs the entire test function instead of one parametrize variant.
sanitized = []
for nid in failed:
    bracket = nid.find("[")
    if bracket >= 0:
        param_part = nid[bracket:]
        if any(c in param_part for c in [" ", ":", '"', "'"]):
            nid = nid[:bracket]
    if nid not in seen or nid in failed:
        sanitized.append(nid)
# Deduplicate after sanitization (multiple parametrize variants -> same function)
failed = list(dict.fromkeys(sanitized))

Path("/workspace/failed_nodeids.json").write_text(json.dumps(failed))
PY
  # ── Phase 2: rerun only failing tests with coverage.py for reliable execution tracing ──
  # Write settrace-based plugin for line-level tracing (fallback when coverage.py unavailable)
  echo '{_settrace_b64}' | base64 -d > /workspace/focused_trace_plugin_phase2.py 2>/dev/null || true
  # Ensure coverage.py is available (some Docker images lack it or have old versions)
  python3 -m pip install --quiet --disable-pip-version-check "coverage>=7.0" 2>/dev/null || true
  python - <<'PY'
import json, os, re as _re, subprocess, sys
from pathlib import Path

failed = json.loads(Path("/workspace/failed_nodeids.json").read_text())
if not failed:
    sys.exit(0)

env = {{**os.environ, "PYTHONPATH": "/workspace:" + os.environ.get("PYTHONPATH", ""), "PYTEST_ADDOPTS": ""}}
# Disable sitecustomize trace in Phase 2 (focused_trace_plugin handles it)
env["_SITECUSTOMIZE_TRACE_ACTIVE"] = "1"

# Inherit critical environment variables from run_script.sh so Phase 2
# has the same PYTHONPATH / QT / framework vars as Phase 1.
_run_script = Path("/workspace/run_script.sh")
if _run_script.exists():
    for _m in _re.finditer(
        r'^\\s*export\\s+(\\w+)=([^\\n]+)', _run_script.read_text(), _re.MULTILINE
    ):
        _key, _val = _m.group(1), _m.group(2).strip().strip('"').strip("'")
        if _key == "PYTEST_ADDOPTS":
            continue  # keep Phase 2 PYTEST_ADDOPTS clean
        # Expand $VAR and ${{VAR}} references using current env
        _expanded = _re.sub(
            r'\\$\\{{(\\w+)\\}}|\\$(\\w+)',
            lambda m: env.get(m.group(1) or m.group(2), os.environ.get(m.group(1) or m.group(2), "")),
            _val,
        )
        if _key == "PYTHONPATH":
            # Merge: run_script paths + existing Phase 2 paths
            env["PYTHONPATH"] = _expanded.rstrip(":") + ":" + env.get("PYTHONPATH", "")
        elif _key not in env or not env[_key]:
            env[_key] = _expanded

# Write coveragerc: track /app source, exclude test files and site-packages
cov_rc = Path("/workspace/.coveragerc_phase2")
cov_rc.write_text(
    "[run]\\n"
    "parallel = false\\n"
    "source = /app\\n"
    "omit =\\n"
    "    */test/*\\n"
    "    */tests/*\\n"
    "    */test_*.py\\n"
    "    *_test.py\\n"
    "    */conftest.py\\n"
    "    */site-packages/*\\n"
    "    */dist-packages/*\\n"
)

# Check coverage.py version: --data-file requires coverage >= 7.0
cov_check = subprocess.run([sys.executable, "-m", "coverage", "--version"], capture_output=True, text=True)
has_coverage = cov_check.returncode == 0
supports_data_file = False
if has_coverage:
    _ver_m = _re.search(r'Coverage(?:\\.py)?[,\\s]+(?:version\\s+)?([\\d.]+)', cov_check.stdout)
    if _ver_m:
        try:
            major = int(_ver_m.group(1).split(".")[0])
            supports_data_file = major >= 7
        except Exception:
            pass

# Common pytest args for Phase 2
_pytest_args = [
    "-p", "focused_trace_plugin",
    "-p", "no:xdist",
    "-p", "no:forked",
    "-p", "no:rerunfailures",
    "--override-ini=reruns=0",
    "-o", "addopts=",
    "--override-ini=addopts=",
    "--tb=long", "--showlocals", "--full-trace",
    "-q",
]

if has_coverage:
    cov_cmd = [sys.executable, "-m", "coverage", "run", f"--rcfile={{cov_rc}}"]
    if supports_data_file:
        cov_cmd.append("--data-file=/workspace/.coverage_phase2")
    else:
        env["COVERAGE_FILE"] = "/workspace/.coverage_phase2"
    cmd = cov_cmd + ["-m", "pytest", *failed] + _pytest_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app", env=env)
    # Export as JSON so we can parse executed files
    json_cmd = [sys.executable, "-m", "coverage", "json", f"--rcfile={{cov_rc}}",
                "-o", "/workspace/phase2_coverage.json", "--pretty-print"]
    if supports_data_file:
        json_cmd.append("--data-file=/workspace/.coverage_phase2")
    subprocess.run(json_cmd, capture_output=True, cwd="/app", env=env)
else:
    # coverage not available — fall back to settrace plugin for line-level tracing
    _phase2_plugin = "focused_trace_plugin_phase2" if Path("/workspace/focused_trace_plugin_phase2.py").exists() else "focused_trace_plugin"
    _fallback_args = [a if a != "focused_trace_plugin" else _phase2_plugin for a in _pytest_args]
    cmd = [sys.executable, "-m", "pytest", *failed] + _fallback_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app", env=env)

Path("/workspace/focused_stdout.log").write_text(result.stdout)
Path("/workspace/focused_stderr.log").write_text(result.stderr)
PY
fi
set -e
mkdir -p /workspace/extra_artifacts/junit
if [ -d /app/test/results/junit ]; then
  cp -f /app/test/results/junit/*.xml /workspace/extra_artifacts/junit/ 2>/dev/null || true
fi
if [ -f /app/.pytest_cache/v/cache/lastfailed ]; then
  cp -f /app/.pytest_cache/v/cache/lastfailed /workspace/extra_artifacts/pytest_lastfailed.json || true
fi
if [ -f /app/.pytest_cache/v/cache/nodeids ]; then
  cp -f /app/.pytest_cache/v/cache/nodeids /workspace/extra_artifacts/pytest_nodeids.json || true
fi
printf "%s\\n" "$pytest_exit_code" > /workspace/pytest_exit_code.txt
# ── Source snapshot: extract source files referenced in coverage/tracebacks ──
# Reads phase2_coverage.json + stdout/stderr to find which production files
# were involved, then copies them to /workspace/source_snapshot/ so the RCA
# pipeline can do AST slicing offline (without re-launching the container).
python3 - <<'PY'
import json, re, shutil
from pathlib import Path

snapshot_dir = Path("/workspace/source_snapshot")
snapshot_dir.mkdir(parents=True, exist_ok=True)

files_to_copy = set()

# Source 1: coverage.py JSON — all executed production files
cov_path = Path("/workspace/phase2_coverage.json")
if cov_path.exists():
    try:
        data = json.loads(cov_path.read_text(errors="replace"))
        for abs_path in data.get("files", dict()).keys():
            abs_path = abs_path.replace("\\\\", "/")
            if "/app/" in abs_path:
                low = abs_path.lower()
                if "/site-packages/" not in low and "/dist-packages/" not in low:
                    files_to_copy.add(abs_path)
    except Exception:
        pass

# Source 2: traceback frames from stdout/stderr/focused logs
_FILE_RE = re.compile(r'File "([^"]+\\.py)", line \\d+')
_PYTEST_LOC_RE = re.compile(r'^(\\S+\\.py):\\d+:', re.MULTILINE)
for log_name in ("stdout.log", "stderr.log", "rerun_stdout.log", "rerun_stderr.log",
                 "focused_stdout.log", "focused_stderr.log"):
    log_path = Path("/workspace") / log_name
    if not log_path.exists():
        continue
    text = log_path.read_text(errors="replace")
    for m in _FILE_RE.finditer(text):
        p = m.group(1)
        if p.startswith("/app/"):
            files_to_copy.add(p)
    for m in _PYTEST_LOC_RE.finditer(text):
        p = m.group(1)
        if not p.startswith("/"):
            files_to_copy.add("/app/" + p)
        elif p.startswith("/app/"):
            files_to_copy.add(p)

# Source 3: focused execution logs (file-level)
for log_path in sorted(Path("/workspace").glob("focused_*.log")):
    if log_path.name in ("focused_stdout.log", "focused_stderr.log",
                         "focused_execution_files.log"):
        continue
    for line in log_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line and line.endswith(".py"):
            if not line.startswith("/"):
                files_to_copy.add("/app/" + line)
            elif line.startswith("/app/"):
                files_to_copy.add(line)

# Copy files preserving directory structure under source_snapshot/
copied = 0
for abs_path in sorted(files_to_copy):
    src = Path(abs_path)
    if not src.exists() or not src.is_file():
        continue
    if abs_path.startswith("/app/"):
        rel = abs_path[5:]
    else:
        rel = abs_path.lstrip("/")
    dst = snapshot_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(src), str(dst))
        copied += 1
    except Exception:
        pass

manifest = sorted(str(p.relative_to(snapshot_dir)) for p in snapshot_dir.rglob("*.py"))
(snapshot_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
PY
exit 0
"""
    return textwrap.dedent(script)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_one_instance(
    sample: dict,
    scripts_dir: Path,
    output_dir: Path,
    timeout_seconds: int,
    entryscript_preamble: str = "",
    extra_workspace_files: dict[str, str] | None = None,
) -> InstanceResult:
    instance_id = sample["instance_id"]
    start = time.time()

    sample_dir = output_dir / instance_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    run_script_path = scripts_dir / instance_id / "run_script.sh"
    parser_path = scripts_dir / instance_id / "parser.py"
    missing_assets = [
        str(p) for p in (run_script_path, parser_path) if not p.exists()
    ]
    patch_files = extract_patch_files(sample.get("patch", ""))

    if missing_assets:
        return make_error_result(
            instance_id,
            "Missing run_script.sh or parser.py",
            repo=sample.get("repo", ""),
            docker_image=f"jefzda/sweap-images:{sample.get('dockerhub_tag', '')}",
            base_commit=sample.get("base_commit", ""),
            missing_assets=missing_assets,
            patch_files=patch_files,
            duration_seconds=round(time.time() - start, 3),
        )

    entryscript = build_entryscript(sample, extra_preamble=entryscript_preamble)
    workspace = prepare_workspace(
        output_dir, instance_id, entryscript, run_script_path.read_text(), parser_path.read_text()
    )
    if extra_workspace_files:
        for rel_path, content in extra_workspace_files.items():
            target = workspace / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

    image = f"jefzda/sweap-images:{sample['dockerhub_tag']}"
    (sample_dir / "entryscript.sh").write_text(entryscript)
    (sample_dir / "patch.diff").write_text(sample.get("patch", ""))
    # Pin the exact image used so downstream stages (e.g. debug_agent/run_rca)
    # can attach or relaunch the same container without re-deriving the tag.
    (sample_dir / "docker_image.txt").write_text(image + "\n")

    timeout_hit, status_code, startup_error = run_container(image, workspace, timeout_seconds)

    stdout_log = (workspace / "stdout.log").read_text(errors="replace") if (workspace / "stdout.log").exists() else ""
    stderr_log = (workspace / "stderr.log").read_text(errors="replace") if (workspace / "stderr.log").exists() else startup_error
    rerun_stdout_log = (workspace / "rerun_stdout.log").read_text(errors="replace") if (workspace / "rerun_stdout.log").exists() else ""
    rerun_stderr_log = (workspace / "rerun_stderr.log").read_text(errors="replace") if (workspace / "rerun_stderr.log").exists() else ""

    pytest_exit_code: int | None = None
    pytest_exit_code_path = workspace / "pytest_exit_code.txt"
    if pytest_exit_code_path.exists():
        try:
            pytest_exit_code = int(pytest_exit_code_path.read_text().strip())
        except Exception:
            pass

    pytest_report_xml = ""
    pytest_report_path = workspace / "pytest-report.xml"
    if pytest_report_path.exists():
        pytest_report_xml = pytest_report_path.read_text(errors="replace")

    extra_junit_reports = load_extra_junit_reports(workspace)

    # Persist logs to sample_dir
    (sample_dir / "stdout.log").write_text(stdout_log)
    (sample_dir / "stderr.log").write_text(stderr_log)
    if rerun_stdout_log:
        (sample_dir / "rerun_stdout.log").write_text(rerun_stdout_log)
    if rerun_stderr_log:
        (sample_dir / "rerun_stderr.log").write_text(rerun_stderr_log)
    if pytest_exit_code is not None:
        (sample_dir / "pytest_exit_code.txt").write_text(f"{pytest_exit_code}\n")
    if pytest_report_xml:
        (sample_dir / "pytest-report.xml").write_text(pytest_report_xml)
    if extra_junit_reports:
        extra_junit_dir = sample_dir / "extra_junit_reports"
        extra_junit_dir.mkdir(parents=True, exist_ok=True)
        for report_name, report_text in extra_junit_reports:
            (extra_junit_dir / report_name).write_text(report_text)

    # Persist focused rerun logs if present
    focused_stdout_log = (workspace / "focused_stdout.log").read_text(errors="replace") if (workspace / "focused_stdout.log").exists() else ""
    focused_stderr_log = (workspace / "focused_stderr.log").read_text(errors="replace") if (workspace / "focused_stderr.log").exists() else ""
    if focused_stdout_log:
        (sample_dir / "focused_stdout.log").write_text(focused_stdout_log)
    if focused_stderr_log:
        (sample_dir / "focused_stderr.log").write_text(focused_stderr_log)

    analysis_stdout = "\n".join([stdout_log, rerun_stdout_log]).strip()
    analysis_stderr = "\n".join([stderr_log, rerun_stderr_log]).strip()
    # Session-level execution files (broad, from exec_trace_plugin)
    execution_files = load_execution_files(workspace)
    # Per-test focused execution files: setprofile traces (atexit-based, works with --forked)
    # + coverage.py JSON from Phase 2 direct rerun (more reliable, handles any subprocess)
    focused_execution_files = list(dict.fromkeys(
        load_focused_execution_files(workspace) + load_phase2_coverage_files(workspace)
    ))

    stack_file_set = set(extract_stack_files(analysis_stdout, analysis_stderr))
    if pytest_report_xml:
        stack_file_set.update(extract_stack_files_from_junit_xml(pytest_report_xml))
    for _, report_text in extra_junit_reports:
        stack_file_set.update(extract_stack_files_from_junit_xml(report_text))
    # Also extract stacks from the focused rerun (may surface new file references)
    stack_file_set.update(extract_stack_files(focused_stdout_log, focused_stderr_log))
    stack_files = sorted(stack_file_set)

    failed_tests = extract_failed_tests(analysis_stdout, analysis_stderr)
    failure_trace_excerpt = extract_failure_trace_excerpt(analysis_stdout, analysis_stderr)
    junit_failure_excerpt = extract_junit_failure_excerpt(pytest_report_xml)
    extra_junit_failure_excerpt = extract_junit_failure_excerpt_many(extra_junit_reports)
    if extra_junit_failure_excerpt:
        junit_failure_excerpt = (
            f"{junit_failure_excerpt}\n\n===\n\n{extra_junit_failure_excerpt}"
            if junit_failure_excerpt
            else extra_junit_failure_excerpt
        )
    if junit_failure_excerpt:
        failure_trace_excerpt = (
            f"{failure_trace_excerpt}\n\n===\n\nJUnit failure details:\n{junit_failure_excerpt}"
            if failure_trace_excerpt
            else f"JUnit failure details:\n{junit_failure_excerpt}"
        )

    output_incomplete = is_pytest_output_incomplete(analysis_stdout, analysis_stderr)
    failure_trace_path = sample_dir / "failure_trace.log"
    failure_trace_path.write_text(failure_trace_excerpt)

    traceback_matched = match_patch_files(patch_files, stack_files)
    execution_matched = match_patch_files(patch_files, execution_files)
    focused_matched = match_patch_files(patch_files, focused_execution_files)
    matched = sorted(set(traceback_matched + execution_matched + focused_matched))

    if execution_files:
        (sample_dir / "execution_files.log").write_text("\n".join(execution_files) + "\n")
    if focused_execution_files:
        (sample_dir / "focused_execution_files.log").write_text("\n".join(focused_execution_files) + "\n")
    phase2_cov = workspace / "phase2_coverage.json"
    if phase2_cov.exists():
        import shutil
        shutil.copy2(phase2_cov, sample_dir / "phase2_coverage.json")

    # Persist source snapshot (extracted from Docker container by entryscript)
    source_snapshot = workspace / "source_snapshot"
    if source_snapshot.exists():
        dst_snapshot = sample_dir / "source_snapshot"
        if not dst_snapshot.exists():
            import shutil as _shutil
            _shutil.copytree(str(source_snapshot), str(dst_snapshot), dirs_exist_ok=True)

    # ── Statement-level RCA ──────────────────────────────────────────────────
    rca_candidates: list[dict] = []
    if failed_tests:
        try:
            from statement_tracer import build_statement_traces, load_focused_lines_per_test
            from context_extractor import build_failure_contexts, make_repo_source_reader
            from root_cause_analyzer import analyze_root_cause, save_rca_results

            source_reader = None
            if source_snapshot.exists():
                source_reader = make_repo_source_reader(source_snapshot)

            traces = build_statement_traces(
                workspace=workspace,
                failure_text=failure_trace_excerpt,
                failed_tests=failed_tests,
                junit_xml_path=workspace / "pytest-report.xml" if pytest_report_xml else None,
            )

            contexts = build_failure_contexts(traces, source_reader)
            all_rca = []
            for trace, ctx in zip(traces, contexts):
                candidates = analyze_root_cause(trace, ctx, source_reader)
                all_rca.append(candidates)
                for c in candidates[:10]:
                    rca_candidates.append({
                        "test_nodeid": trace.test_nodeid,
                        "file": c.file,
                        "line": c.line,
                        "score": c.score,
                        "signals": c.signals,
                        "code_snippet": c.code_snippet,
                        "explanation": c.explanation,
                        "func_name": c.func_name,
                    })

            # Persist RCA results
            rca_output = sample_dir / "rca_results.json"
            rca_output.write_text(json.dumps(rca_candidates, indent=2, ensure_ascii=False))
        except Exception:
            pass  # RCA is best-effort, don't fail the run

    return InstanceResult(
        instance_id=instance_id,
        repo=sample.get("repo", ""),
        docker_image=image,
        base_commit=sample.get("base_commit", ""),
        timeout=timeout_hit,
        container_status_code=status_code,
        missing_assets=[],
        patch_files=patch_files,
        stack_files=stack_files,
        # Merge session-level + focused; focused is more precise but may be empty
        execution_files=sorted(set(execution_files) | set(focused_execution_files)),
        traceback_matched_patch_files=traceback_matched,
        execution_matched_patch_files=sorted(set(execution_matched) | set(focused_matched)),
        matched_patch_files=matched,
        stack_contains_patch_file=bool(matched),
        error_excerpt=get_error_excerpt(analysis_stdout, analysis_stderr),
        failure_trace_excerpt=failure_trace_excerpt,
        failure_trace_log=str(failure_trace_path.resolve()),
        failed_tests=failed_tests,
        stdout_log=str((sample_dir / "stdout.log").resolve()),
        stderr_log=str((sample_dir / "stderr.log").resolve()),
        pytest_exit_code=pytest_exit_code,
        pytest_report_xml=str((sample_dir / "pytest-report.xml").resolve()) if pytest_report_xml else "",
        output_incomplete=output_incomplete,
        workspace=str(workspace.resolve()),
        duration_seconds=round(time.time() - start, 3),
    )
