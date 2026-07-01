"""Step 1 & 2: Statement-level execution tracing — data collection and parsing.

Parses existing coverage.py JSON, tracebacks, and focused logs to build
per-failing-test statement-level execution traces.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from common import normalize_path, strip_ansi


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TracebackFrame:
    """One frame in a Python traceback."""
    file: str        # normalized relative path
    lineno: int
    func_name: str
    code_line: str   # the source line shown in the traceback (may be empty)
    is_test_code: bool = False


@dataclass
class CoverageFileData:
    """Line-level coverage for a single file."""
    executed_lines: list[int]
    missing_lines: list[int]
    # branch data (empty if branch_coverage=false)
    executed_branches: list[tuple[int, int]] = field(default_factory=list)
    missing_branches: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class StatementTrace:
    """Per-failing-test statement-level execution trace."""
    test_nodeid: str
    # Traceback frames from crash (innermost last)
    traceback_frames: list[TracebackFrame]
    error_type: str
    error_message: str
    # Files this test executed (from focused_*.log, file-level)
    focused_files: list[str]
    # Session-wide line coverage (from coverage.py JSON): file -> data
    coverage: dict[str, CoverageFileData]
    # Per-test line coverage approximation: intersect coverage with focused_files
    per_test_executed_lines: dict[str, list[int]]
    # Exception locals captured at crash site (from settrace exception events)
    exception_locals: ExceptionLocals | None = None


# ── Regex patterns ───────────────────────────────────────────────────────────

_TRACEBACK_FRAME_RE = re.compile(
    r'^\s*File "([^"]+)", line (\d+)(?:, in (.+))?',
    re.MULTILINE,
)
# pytest short summary line: path/file.py:123: ErrorType
_PYTEST_SHORT_RE = re.compile(
    r"^(\S+\.py):(\d+): (\w+(?:Error|Exception|Warning|Failure))\s*$",
    re.MULTILINE,
)
# pytest section header: ___ TestClass.test_method ___
_PYTEST_SECTION_RE = re.compile(
    r"^_{3,}\s+(.+?)\s+_{3,}\s*$", re.MULTILINE
)
# Exception line at end of traceback: ExceptionType: message
_EXCEPTION_LINE_RE = re.compile(
    r"^E?\s*(\w+(?:Error|Exception|Failure|Warning))\s*:\s*(.+)$",
    re.MULTILINE,
)
# > marker line in pytest output (the failing line)
_PYTEST_GT_LINE_RE = re.compile(r"^>\s+(.+)$", re.MULTILINE)

_TEST_PATH_RE = re.compile(
    r"(?:^|/)(?:tests?|testing|test_[^/]+|[^/]+_test|conftest|__fixtures__)/"
    r"|(?:^|/)test_[^/]+\.py$|[^/]+_test\.py$",
    re.IGNORECASE,
)


# ── Coverage JSON parsing ────────────────────────────────────────────────────

def parse_coverage_json(path: Path) -> dict[str, CoverageFileData]:
    """Parse coverage.py JSON report into per-file line-level data."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}

    has_branches = data.get("meta", {}).get("branch_coverage", False)
    result: dict[str, CoverageFileData] = {}

    for abs_path, file_data in data.get("files", {}).items():
        rel = _normalize_coverage_path(abs_path)
        if not rel or not rel.endswith(".py"):
            continue
        if _is_framework_path(rel):
            continue

        executed = file_data.get("executed_lines", [])
        missing = file_data.get("missing_lines", [])
        exec_branches: list[tuple[int, int]] = []
        miss_branches: list[tuple[int, int]] = []
        if has_branches:
            for br in file_data.get("executed_branches", []):
                if isinstance(br, list) and len(br) == 2:
                    exec_branches.append((br[0], br[1]))
            for br in file_data.get("missing_branches", []):
                if isinstance(br, list) and len(br) == 2:
                    miss_branches.append((br[0], br[1]))

        result[rel] = CoverageFileData(
            executed_lines=sorted(executed),
            missing_lines=sorted(missing),
            executed_branches=exec_branches,
            missing_branches=miss_branches,
        )

    return result


def _normalize_coverage_path(abs_path: str) -> str:
    abs_path = abs_path.replace("\\", "/")
    if "/app/" in abs_path:
        low = abs_path.lower()
        if "/site-packages/" in low or "/dist-packages/" in low:
            return ""
        return normalize_path(abs_path[abs_path.find("/app/") + 5:])
    return normalize_path(abs_path)


_FRAMEWORK_PATH_RE = re.compile(
    r"^(?:_pytest|pluggy|pytest|py)/|"
    r"^(?:jinja2|yaml|coverage|pip|setuptools|pkg_resources|importlib_metadata|"
    r"exceptiongroup|tomli|iniconfig|anyio|filelock|platformdirs|packaging|"
    r"pytest_rerunfailures|pytest_mock|xdist|execnet|apipkg|passlib|cryptography)/"
)


def _is_framework_path(path: str) -> bool:
    if _FRAMEWORK_PATH_RE.search(path):
        return True
    # Filter site-packages / dist-packages / stdlib paths
    low = path.lower()
    if "/site-packages/" in low or "/dist-packages/" in low:
        return True
    if low.startswith("usr/") or low.startswith("/usr/"):
        return True
    return False


# ── Traceback parsing ────────────────────────────────────────────────────────

def parse_traceback_text(text: str) -> list[TracebackFrame]:
    """Extract traceback frames from Python traceback text.

    Handles multiple formats:
    1. Standard Python: File "path.py", line N, in func
    2. Pytest short:    path.py:N: ErrorType
    3. Pytest verbose:  '>' marker lines with context showing the call chain
    """
    text = strip_ansi(text or "")
    frames: list[TracebackFrame] = []
    seen: set[tuple[str, int]] = set()

    # Format 1: Standard Python traceback frames
    for m in _TRACEBACK_FRAME_RE.finditer(text):
        raw_path, lineno_str, func_name = m.group(1), m.group(2), m.group(3) or ""
        rel = normalize_path(raw_path)
        if not rel or not rel.endswith(".py"):
            continue
        if _is_framework_path(rel):
            continue
        lineno = int(lineno_str)
        # Try to find the code line (next non-empty line after File "..." line)
        after = text[m.end():]
        code_line = ""
        for line in after.split("\n", 3)[:3]:
            stripped = line.strip()
            if stripped and not stripped.startswith("File ") and not stripped.startswith("E "):
                code_line = stripped
                break
        key = (rel, lineno)
        if key not in seen:
            seen.add(key)
            frames.append(TracebackFrame(
                file=rel,
                lineno=lineno,
                func_name=func_name,
                code_line=code_line,
                is_test_code=bool(_TEST_PATH_RE.search(rel)),
            ))

    # Format 2: Pytest short summary lines — path.py:N: ErrorType
    for m in _PYTEST_SHORT_RE.finditer(text):
        raw_path, lineno_str, err_type = m.group(1), m.group(2), m.group(3)
        rel = normalize_path(raw_path)
        if not rel or not rel.endswith(".py"):
            continue
        if _is_framework_path(rel):
            continue
        lineno = int(lineno_str)
        key = (rel, lineno)
        if key not in seen:
            seen.add(key)
            frames.append(TracebackFrame(
                file=rel, lineno=lineno, func_name="",
                code_line="", is_test_code=bool(_TEST_PATH_RE.search(rel)),
            ))

    # Format 3: Pytest verbose call chain — look for ">" lines preceded by
    # function definitions and followed by "path.py:N:" location lines.
    # Pattern in pytest output:
    #     def some_function(a, b):
    # >       result = a | b
    # E       TypeError: ...
    #
    # info       = <...>      ← optional variable dumps
    # key        = 16777249
    #
    # path/to/file.py:91: TypeError
    _PYTEST_CHAIN_RE = re.compile(
        r"^>[ \t]+(\S[^\n]*)\n"             # > marker line
        r"(?:E[ \t]+\S[^\n]*\n)*"           # E lines (errors)
        r"(?:\n?"                            # optional blank line
        r"(?:\w[\w ]*=[ \t]+[^\n]*\n)*"     # variable dump lines (name = value)
        r"\n?)?"                             # optional trailing blank
        r"(\S+\.py):(\d+):",                # path.py:N: location
        re.MULTILINE,
    )
    for m in _PYTEST_CHAIN_RE.finditer(text):
        code_line = m.group(1).strip()
        raw_path = m.group(2)
        lineno = int(m.group(3))
        rel = normalize_path(raw_path)
        if not rel or not rel.endswith(".py"):
            continue
        if _is_framework_path(rel):
            continue
        # Try to extract function name from lines before ">"
        func_name = _extract_func_name_before(text, m.start())
        key = (rel, lineno)
        if key not in seen:
            seen.add(key)
            frames.append(TracebackFrame(
                file=rel, lineno=lineno, func_name=func_name,
                code_line=code_line, is_test_code=bool(_TEST_PATH_RE.search(rel)),
            ))
        elif code_line:
            # Update existing frame if this format has a richer code_line
            for f in frames:
                if f.file == rel and f.lineno == lineno and not f.code_line:
                    f.code_line = code_line
                    if func_name and not f.func_name:
                        f.func_name = func_name
                    break

    # Format 4: ImportError / ModuleNotFoundError — extract the source module
    # from the error message itself.  This is *not* test-name hacking: it is
    # information embedded in the execution trace (the exception message).
    # Patterns handled:
    #   ImportError: cannot import name 'X' from 'a.b.c' (/app/a/b/c.py)
    #   ModuleNotFoundError: No module named 'a.b.c'
    _IMPORT_FROM_RE = re.compile(
        r"(?:ImportError|ModuleNotFoundError)\s*:\s*"
        r"(?:cannot import name .+? from '([^']+)'|No module named '([^']+)')",
    )
    for m in _IMPORT_FROM_RE.finditer(text):
        module_dotted = (m.group(1) or m.group(2) or "").strip()
        if not module_dotted:
            continue
        # Convert dotted module path to file path: a.b.c → a/b/c.py
        parts = module_dotted.split(".")
        candidates_paths = [
            "/".join(parts) + ".py",
            "/".join(parts) + "/__init__.py",
        ]
        for rel in candidates_paths:
            rel = normalize_path(rel)
            if not rel or not rel.endswith(".py"):
                continue
            if _is_framework_path(rel):
                continue
            key = (rel, 1)  # line 1 as placeholder — the whole module is suspect
            if key not in seen:
                seen.add(key)
                frames.append(TracebackFrame(
                    file=rel, lineno=1, func_name="<module>",
                    code_line=f"ImportError: {module_dotted}",
                    is_test_code=bool(_TEST_PATH_RE.search(rel)),
                ))

    return frames


_FUNC_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)


def _extract_func_name_before(text: str, pos: int) -> str:
    """Look backwards from pos to find the nearest function definition."""
    # Search in the 500 chars before the position
    start = max(0, pos - 500)
    chunk = text[start:pos]
    matches = list(_FUNC_DEF_RE.finditer(chunk))
    if matches:
        return matches[-1].group(1)
    return ""


def parse_pytest_failure_sections(text: str) -> dict[str, _FailureSection]:
    """Split pytest output into per-test failure sections."""
    text = strip_ansi(text or "")
    sections: dict[str, _FailureSection] = {}
    parts = _PYTEST_SECTION_RE.split(text)
    # parts: [before, header1, body1, header2, body2, ...]
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body = parts[i + 1]
        i += 2
        # Normalize header to a pytest-like node ID
        nodeid = _header_to_nodeid(header)
        if not nodeid:
            continue
        frames = parse_traceback_text(body)
        error_type, error_message = _extract_exception(body)
        sections[nodeid] = _FailureSection(
            frames=frames,
            error_type=error_type,
            error_message=error_message,
            raw_text=body,
        )
    return sections


@dataclass
class _FailureSection:
    frames: list[TracebackFrame]
    error_type: str
    error_message: str
    raw_text: str


def _header_to_nodeid(header: str) -> str:
    """Convert pytest section header like 'TestClass.test_method' to a searchable key."""
    # Remove brackets, parametrize info for matching
    header = header.strip()
    if not header:
        return ""
    return header


def _extract_exception(text: str) -> tuple[str, str]:
    """Extract exception type and message from traceback/pytest output."""
    # Try E-prefixed lines first (pytest format)
    for m in _EXCEPTION_LINE_RE.finditer(text):
        return m.group(1), m.group(2).strip()
    # Try short summary lines
    for m in _PYTEST_SHORT_RE.finditer(text):
        return m.group(3), ""
    return "", ""


def parse_junit_tracebacks(xml_path: Path) -> dict[str, _FailureSection]:
    """Extract per-test tracebacks from JUnit XML."""
    if not xml_path.exists():
        return {}
    try:
        root = ET.fromstring(xml_path.read_text(errors="replace"))
    except Exception:
        return {}

    sections: dict[str, _FailureSection] = {}
    for tc in root.iter("testcase"):
        classname = tc.attrib.get("classname", "").strip()
        name = tc.attrib.get("name", "").strip()
        nodeid = "::".join(p for p in [classname, name] if p)
        if not nodeid:
            continue

        fail_el = tc.find("failure") or tc.find("error")
        if fail_el is None:
            continue

        body = (fail_el.text or "") + "\n" + (fail_el.attrib.get("message", ""))
        frames = parse_traceback_text(body)
        error_type, error_message = _extract_exception(body)
        if not error_type:
            # Try from message attribute
            msg = fail_el.attrib.get("message", "")
            for m in _EXCEPTION_LINE_RE.finditer(msg):
                error_type, error_message = m.group(1), m.group(2).strip()
                break

        sections[nodeid] = _FailureSection(
            frames=frames,
            error_type=error_type,
            error_message=error_message,
            raw_text=body,
        )
    return sections


# ── Focused log parsing ──────────────────────────────────────────────────────

def load_focused_files_per_test(workspace: Path) -> dict[str, list[str]]:
    """Load per-test focused file lists from focused_<pid>_<nodeid>.log files."""
    result: dict[str, list[str]] = {}
    for path in sorted(workspace.glob("focused_*.log")):
        if path.name in ("focused_stdout.log", "focused_stderr.log",
                         "focused_execution_files.log"):
            continue
        # Extract nodeid from filename: focused_<pid>_<safe_nodeid>.log
        stem = path.stem  # e.g. "focused_1508_test_units_..."
        parts = stem.split("_", 2)
        if len(parts) < 3:
            continue
        safe_nodeid = parts[2]  # still mangled, used as key
        try:
            files = [
                normalize_path(line)
                for line in path.read_text(errors="replace").splitlines()
                if line.strip()
            ]
        except Exception:
            continue
        if files:
            result[safe_nodeid] = files
    return result


def _match_nodeid_to_focused(nodeid: str, focused_keys: list[str]) -> str | None:
    """Fuzzy-match a pytest nodeid to a focused log filename key."""
    # Sanitize nodeid the same way the plugin does
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in nodeid)[:120]
    for key in focused_keys:
        if safe == key or safe.endswith(key) or key.endswith(safe):
            return key
        # Substring match (nodeid components)
        if safe in key or key in safe:
            return key
    return None


# ── Build StatementTrace ─────────────────────────────────────────────────────

def build_statement_traces(
    workspace: Path,
    failure_text: str,
    failed_tests: list[str],
    junit_xml_path: Path | None = None,
) -> list[StatementTrace]:
    """Build StatementTrace for each failing test from available data."""
    # Parse coverage JSON (session-wide line data)
    coverage = parse_coverage_json(workspace / "phase2_coverage.json")

    # Parse tracebacks from multiple sources
    sections_from_text = parse_pytest_failure_sections(failure_text)
    sections_from_junit: dict[str, _FailureSection] = {}
    if junit_xml_path and junit_xml_path.exists():
        sections_from_junit = parse_junit_tracebacks(junit_xml_path)

    # Also parse raw traceback from combined stdout/stderr/focused logs
    combined_text = failure_text
    for log_name in ("focused_stdout.log", "focused_stderr.log",
                     "rerun_stdout.log", "rerun_stderr.log"):
        log_path = workspace / log_name
        if log_path.exists():
            combined_text += "\n" + log_path.read_text(errors="replace")

    all_sections = _merge_failure_sections(
        sections_from_text, sections_from_junit, combined_text
    )

    # Per-test focused files (file-level, from setprofile)
    focused_per_test = load_focused_files_per_test(workspace)
    focused_keys = list(focused_per_test.keys())

    # Per-test line-level data (from settrace plugin, Phase 2)
    focused_lines_per_test = load_focused_lines_per_test(workspace)
    focused_lines_keys = list(focused_lines_per_test.keys())

    # Per-test exception locals (from settrace exception events)
    locals_per_test = load_focused_locals_per_test(workspace)
    locals_keys = list(locals_per_test.keys())

    # Build traces
    traces: list[StatementTrace] = []
    for nodeid in failed_tests:
        section = _find_section_for_nodeid(nodeid, all_sections)
        # Fallback: parse full text but cap size to avoid regex catastrophic backtracking
        _MAX_FALLBACK_LEN = 512_000  # 512 KB
        frames = section.frames if section else parse_traceback_text(combined_text[:_MAX_FALLBACK_LEN])
        error_type = section.error_type if section else ""
        error_message = section.error_message if section else ""

        if not error_type:
            error_type, error_message = _extract_exception(combined_text)

        # Find focused files for this test
        focused_key = _match_nodeid_to_focused(nodeid, focused_keys)
        focused_files = focused_per_test.get(focused_key, []) if focused_key else []

        # Approximate per-test line coverage: intersect coverage files with focused files
        per_test_lines: dict[str, list[int]] = {}
        if focused_files:
            focused_set = set(focused_files)
            for f, cov in coverage.items():
                if f in focused_set or any(
                    f.endswith("/" + ff) or ff.endswith("/" + f) for ff in focused_set
                ):
                    per_test_lines[f] = cov.executed_lines
        else:
            # No focused data: use all non-test coverage as approximation
            for f, cov in coverage.items():
                if not _TEST_PATH_RE.search(f):
                    per_test_lines[f] = cov.executed_lines

        # Merge settrace line-level data (more precise, per-test)
        focused_lines_key = _match_nodeid_to_focused(nodeid, focused_lines_keys)
        if focused_lines_key:
            for f, lines in focused_lines_per_test[focused_lines_key].items():
                existing = set(per_test_lines.get(f, []))
                existing.update(lines)
                per_test_lines[f] = sorted(existing)

        # Match exception locals
        locals_key = _match_nodeid_to_focused(nodeid, locals_keys)
        exc_locals = locals_per_test.get(locals_key) if locals_key else None

        traces.append(StatementTrace(
            test_nodeid=nodeid,
            traceback_frames=frames,
            error_type=error_type,
            error_message=error_message,
            focused_files=focused_files,
            coverage=coverage,
            per_test_executed_lines=per_test_lines,
            exception_locals=exc_locals,
        ))

    return traces


def _merge_failure_sections(
    from_text: dict[str, _FailureSection],
    from_junit: dict[str, _FailureSection],
    combined_text: str,
) -> dict[str, _FailureSection]:
    """Merge failure sections from multiple sources, preferring richer data."""
    merged: dict[str, _FailureSection] = {}
    for key, section in from_junit.items():
        merged[key] = section
    for key, section in from_text.items():
        if key not in merged or len(section.frames) > len(merged[key].frames):
            merged[key] = section
    return merged


def _find_section_for_nodeid(
    nodeid: str, sections: dict[str, _FailureSection]
) -> _FailureSection | None:
    """Find the failure section matching a pytest nodeid."""
    if nodeid in sections:
        return sections[nodeid]
    # Fuzzy match: try suffix matching
    nodeid_parts = nodeid.replace("::", ".").replace("/", ".").lower()
    for key, section in sections.items():
        key_parts = key.replace("::", ".").replace("/", ".").lower()
        if nodeid_parts in key_parts or key_parts in nodeid_parts:
            return section
        # Match by test method name
        nodeid_tail = nodeid_parts.rsplit(".", 1)[-1]
        key_tail = key_parts.rsplit(".", 1)[-1]
        if nodeid_tail and nodeid_tail == key_tail:
            return section
    return None


# ── Step 2: Upgraded settrace plugin code generation ─────────────────────────

def generate_settrace_plugin_code() -> str:
    """Generate focused_trace_plugin.py code that uses sys.settrace for line-level tracing.

    This is the Step 2 upgrade: replaces sys.setprofile (call-only) with
    sys.settrace (call + line events) to capture per-test line-level execution.
    Output format: focused_lines_<pid>_<nodeid>.json with {file: [line_numbers]}.

    Also captures exception locals: on 'exception' events, dumps frame.f_locals
    for production code frames. Output: focused_locals_<pid>_<nodeid>.json with
    {file: {line: {var: repr_value}}}.
    """
    return r'''
import atexit
import json
import os
import re
import sys
from pathlib import Path

_order = {}      # pid -> {nodeid -> {file -> list[int]}}
_seen_set = {}   # pid -> {nodeid -> {file -> set[int]}}
_current = {}    # pid -> current nodeid
_exc_locals = {} # pid -> {nodeid -> {file -> {line -> {var: repr_val}}}}

_TEST_RE = re.compile(
    r"(?:^|/)(?:tests?|testing|test_[^/]+|[^/]+_test|conftest|__fixtures__)/"
    r"|(?:^|/)test_[^/]+\.py$|[^/]+_test\.py$",
    re.IGNORECASE,
)
_FRAMEWORK_RE = re.compile(
    r"^(?:_pytest|pluggy|pytest|py)/|"
    r"^(?:jinja2|yaml|coverage|pip|setuptools|pkg_resources|importlib_metadata|"
    r"exceptiongroup|tomli|iniconfig|anyio|filelock|platformdirs|packaging|"
    r"pytest_rerunfailures|pytest_mock|xdist|execnet|apipkg)/",
)

def _norm(filename):
    filename = (filename or "").replace("\\", "/")
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

def _safe_repr(val, max_len=120):
    """Safe repr with length cap — avoid huge objects blowing up JSON."""
    try:
        r = repr(val)
    except Exception:
        return "<repr-error>"
    if len(r) > max_len:
        return r[:max_len - 3] + "..."
    return r

def _capture_locals(frame, rel, lineno, pid):
    """Capture f_locals at exception site, keeping only serializable primitives + short reprs."""
    tid = _current.get(pid)
    if not tid:
        return
    try:
        local_data = {}
        for k, v in frame.f_locals.items():
            if k.startswith("__") and k.endswith("__"):
                continue  # skip dunders
            local_data[k] = {
                "repr": _safe_repr(v),
                "type": type(v).__name__,
                "is_none": v is None,
                "is_empty": False,
            }
            # Detect empty containers
            try:
                if hasattr(v, "__len__") and len(v) == 0:
                    local_data[k]["is_empty"] = True
            except Exception:
                pass
        if local_data:
            store = _exc_locals.setdefault(pid, {}).setdefault(tid, {}).setdefault(rel, {})
            line_key = str(lineno)
            if line_key not in store:
                store[line_key] = local_data
    except Exception:
        pass  # never break tracing

def _make_tracer(pid):
    def _tracer(frame, event, arg):
        if event == "call":
            fn = frame.f_code.co_filename or ""
            if fn.endswith(".py"):
                rel = _norm(fn)
                if rel and not _TEST_RE.search(rel) and not _FRAMEWORK_RE.search(rel):
                    tid = _current.get(pid)
                    if tid:
                        ss = _seen_set.setdefault(pid, {}).setdefault(tid, {}).setdefault(rel, set())
                        if frame.f_lineno not in ss:
                            ss.add(frame.f_lineno)
                            _order.setdefault(pid, {}).setdefault(tid, {}).setdefault(rel, []).append(frame.f_lineno)
            return _tracer  # return tracer to trace lines inside this call
        if event == "line":
            fn = frame.f_code.co_filename or ""
            if fn.endswith(".py"):
                rel = _norm(fn)
                if rel and not _TEST_RE.search(rel) and not _FRAMEWORK_RE.search(rel):
                    tid = _current.get(pid)
                    if tid:
                        ss = _seen_set.setdefault(pid, {}).setdefault(tid, {}).setdefault(rel, set())
                        if frame.f_lineno not in ss:
                            ss.add(frame.f_lineno)
                            _order.setdefault(pid, {}).setdefault(tid, {}).setdefault(rel, []).append(frame.f_lineno)
            return _tracer
        if event == "exception":
            fn = frame.f_code.co_filename or ""
            if fn.endswith(".py"):
                rel = _norm(fn)
                if rel and not _TEST_RE.search(rel) and not _FRAMEWORK_RE.search(rel):
                    _capture_locals(frame, rel, frame.f_lineno, pid)
            return _tracer
        return _tracer
    return _tracer

def _write_focused(pid, tid):
    data = _order.get(pid, {}).get(tid, {})
    filtered = {f: lines for f, lines in data.items()
                if not _TEST_RE.search(f) and not _FRAMEWORK_RE.search(f)}
    if not filtered:
        return
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in tid)[:120]
    # Write line-level JSON
    out = Path("/workspace") / f"focused_lines_{os.getpid()}_{safe}.json"
    if not out.exists():
        out.write_text(json.dumps(filtered))
    # Also write file-level log for backward compatibility
    out_compat = Path("/workspace") / f"focused_{os.getpid()}_{safe}.log"
    if not out_compat.exists():
        out_compat.write_text("\n".join(filtered.keys()) + "\n")
    # Write exception locals
    locals_data = _exc_locals.get(pid, {}).get(tid, {})
    if locals_data:
        out_locals = Path("/workspace") / f"focused_locals_{os.getpid()}_{safe}.json"
        if not out_locals.exists():
            try:
                out_locals.write_text(json.dumps(locals_data))
            except Exception:
                pass

def _dump_on_exit():
    for pid, tests in _order.items():
        for tid in tests:
            _write_focused(pid, tid)

atexit.register(_dump_on_exit)

def pytest_runtest_setup(item):
    pid = os.getpid()
    _current[pid] = item.nodeid
    sys.settrace(_make_tracer(pid))

def pytest_runtest_teardown(item, nextitem):
    pid = os.getpid()
    sys.settrace(None)
    _current.pop(pid, None)

def pytest_runtest_logreport(report):
    if report.when == "call" and report.failed:
        pid = os.getpid()
        _write_focused(pid, report.nodeid)
'''


def load_focused_lines_per_test(workspace: Path) -> dict[str, dict[str, list[int]]]:
    """Load per-test line-level traces from focused_lines_*.json (Step 2 output).

    Returns {safe_nodeid: {file: [line_numbers]}}.
    """
    result: dict[str, dict[str, list[int]]] = {}
    for path in sorted(workspace.glob("focused_lines_*.json")):
        stem = path.stem  # focused_lines_<pid>_<nodeid>
        parts = stem.split("_", 3)
        if len(parts) < 4:
            continue
        safe_nodeid = parts[3]
        try:
            data = json.loads(path.read_text(errors="replace"))
        except Exception:
            continue
        if isinstance(data, dict):
            # Merge with existing (multiple PIDs for same test)
            existing = result.setdefault(safe_nodeid, {})
            for f, lines in data.items():
                if isinstance(lines, list):
                    prev = set(existing.get(f, []))
                    prev.update(lines)
                    existing[f] = sorted(prev)
    return result


# ── Exception locals data ────────────────────────────────────────────────────

@dataclass
class ExceptionLocalVar:
    """One captured local variable at an exception site."""
    name: str
    type_name: str
    repr_value: str
    is_none: bool = False
    is_empty: bool = False


@dataclass
class ExceptionLocals:
    """Captured locals at exception site(s) for one test."""
    # file -> line -> list of captured variables
    frames: dict[str, dict[int, list[ExceptionLocalVar]]] = field(default_factory=dict)


def load_focused_locals_per_test(workspace: Path) -> dict[str, ExceptionLocals]:
    """Load per-test exception locals from focused_locals_*.json.

    Returns {safe_nodeid: ExceptionLocals}.
    """
    result: dict[str, ExceptionLocals] = {}
    for path in sorted(workspace.glob("focused_locals_*.json")):
        stem = path.stem  # focused_locals_<pid>_<nodeid>
        parts = stem.split("_", 3)
        if len(parts) < 4:
            continue
        safe_nodeid = parts[3]
        try:
            data = json.loads(path.read_text(errors="replace"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        exc_locals = result.setdefault(safe_nodeid, ExceptionLocals())
        for file, lines_data in data.items():
            if not isinstance(lines_data, dict):
                continue
            for line_str, vars_data in lines_data.items():
                if not isinstance(vars_data, dict):
                    continue
                lineno = int(line_str)
                var_list = exc_locals.frames.setdefault(file, {}).setdefault(lineno, [])
                for var_name, var_info in vars_data.items():
                    if not isinstance(var_info, dict):
                        continue
                    var_list.append(ExceptionLocalVar(
                        name=var_name,
                        type_name=var_info.get("type", ""),
                        repr_value=var_info.get("repr", ""),
                        is_none=var_info.get("is_none", False),
                        is_empty=var_info.get("is_empty", False),
                    ))
    return result
