"""JavaScript / Node.js test runner: entryscript generation and output analysis."""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path

from common import (
    InstanceResult,
    extract_failure_trace_excerpt,
    extract_patch_files,
    extract_stack_files,
    get_error_excerpt,
    load_execution_files,
    make_error_result,
    match_patch_files,
    prepare_workspace,
    run_container,
    strip_ansi,
)



# ── JS-specific regex patterns ────────────────────────────────────────────────

# Jest: "● test suite › test name" or "✕ test name" / "✗ test name"
JEST_FAIL_RE = re.compile(r"^\s*(?:●|✕|✗|×)\s+(.+)$", re.MULTILINE)
# Mocha: "N failing" and then numbered entries
MOCHA_FAIL_RE = re.compile(r"^\s+\d+\)\s+(.+)$", re.MULTILINE)
# TAP: "not ok N - description"
TAP_FAIL_RE = re.compile(r"^not ok \d+\s*(?:-\s*)?(.*)$", re.MULTILINE)
# Generic "FAIL src/foo.test.js" (Jest suite-level)
JEST_SUITE_FAIL_RE = re.compile(r"^FAIL\s+(\S+)", re.MULTILINE)

JS_OUTPUT_INCOMPLETE_RE = re.compile(r"Test Suites?:", re.MULTILINE)

# Jest module resolution errors:
#   "Cannot find module './foo' from 'src/bar.ts'"
#   "Cannot find module 'src/utils/foo'"
JS_MODULE_NOT_FOUND_RE = re.compile(
    r"Cannot find module ['\"]([^'\"]+)['\"](?:\s+from\s+['\"]([^'\"]+)['\"])?",
)


def extract_js_module_error_files(stdout_text: str, stderr_text: str) -> list[str]:
    """Extract source file paths from Jest module resolution errors.

    When a test suite fails because a module can't be found (e.g. the patch adds
    a new file), extract both the missing module path and the importing file.
    """
    from common import normalize_path
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))
    out: set[str] = set()
    for match in JS_MODULE_NOT_FOUND_RE.finditer(text):
        module_path = match.group(1)
        from_path = match.group(2) or ""
        # Normalize module path: relative imports like './foo' need resolving against from_path
        if module_path.startswith(".") and from_path:
            # Resolve relative to the importing file's directory
            from pathlib import PurePosixPath
            base_dir = str(PurePosixPath(from_path).parent)
            resolved = str(PurePosixPath(base_dir) / module_path)
            resolved = normalize_path(resolved)
            if resolved:
                # Try common extensions
                for ext in ("", ".ts", ".tsx", ".js", ".jsx"):
                    out.add(resolved + ext)
        else:
            norm = normalize_path(module_path)
            if norm:
                for ext in ("", ".ts", ".tsx", ".js", ".jsx"):
                    out.add(norm + ext)
        if from_path:
            norm_from = normalize_path(from_path)
            if norm_from:
                out.add(norm_from)
    return sorted(out)


# ── Focused execution file loading ───────────────────────────────────────────

def load_focused_js_execution_files(workspace: Path) -> list[str]:
    """Load focused V8 coverage result (failing Jest suites only)."""
    from common import normalize_path
    focused_log = workspace / "focused_executed_files.js.log"
    if not focused_log.exists():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in focused_log.read_text(errors="replace").splitlines():
        item = normalize_path(line)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ── JS output analysis ────────────────────────────────────────────────────────

def extract_failed_tests(stdout_text: str, stderr_text: str) -> list[str]:
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))
    out: list[str] = []
    seen: set[str] = set()

    for pattern in (JEST_FAIL_RE, MOCHA_FAIL_RE, TAP_FAIL_RE):
        for match in pattern.finditer(text):
            name = match.group(1).strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)

    # Fall back to Jest suite-level failures when no individual test names found
    if not out:
        for match in JEST_SUITE_FAIL_RE.finditer(text):
            name = match.group(1).strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)

    # Detect "Test suite failed to run" (module resolution errors, syntax errors, etc.)
    if not out and "Test suite failed to run" in text:
        out.append("Test suite failed to run")

    # Mocha JSON reporter: multi-line JSON with {"stats":{"failures":N},...,"failures":[...]}
    if not out:
        import json as _json
        for candidate in _extract_mocha_json_blocks(text):
            try:
                data = _json.loads(candidate)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            stats = data.get("stats", {})
            if isinstance(stats, dict) and stats.get("failures", 0) > 0:
                for fail in data.get("failures", []):
                    title = fail.get("fullTitle") or fail.get("title", "")
                    if title and title not in seen:
                        seen.add(title)
                        out.append(title)
                if out:
                    break

    return out


def _extract_mocha_json_blocks(text: str) -> list[str]:
    """Find Mocha JSON reporter blocks: top-level JSON objects containing "stats"."""
    blocks: list[str] = []
    lines = text.splitlines()
    # Find lines that start a top-level JSON block (line is just "{")
    # and extract using brace counting on lines
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped != "{":
            i += 1
            continue
        # Found a potential JSON block start, collect until balanced
        depth = 0
        start = i
        found_end = False
        for j in range(i, min(i + 100000, len(lines))):
            line = lines[j]
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                # Trim at the first } that balances: handle }{"tests":...} on same line
                raw = "\n".join(lines[start:j + 1])
                # Find the closing } that balances the opening {
                block = _trim_to_first_balanced_brace(raw)
                if block and '"stats"' in block:
                    blocks.append(block)
                found_end = True
                i = j + 1
                break
        if not found_end:
            i += 1
    return blocks


def _trim_to_first_balanced_brace(text: str) -> str:
    """Return text up to the first point where braces are balanced (depth=0)."""
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[:i + 1]
    return text


def is_js_output_incomplete(stdout_text: str, stderr_text: str) -> bool:
    """Heuristic: Jest printed 'Test Suites:' header but process was cut off."""
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))
    has_header = bool(JS_OUTPUT_INCOMPLETE_RE.search(text))
    has_summary = bool(re.search(r"Tests?:\s+\d+", text))
    return has_header and not has_summary


# ── Entryscript ───────────────────────────────────────────────────────────────

def build_entryscript(sample: dict) -> str:
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

    # Node.js script written to /workspace/parse_v8.js inside the container.
    # Parses V8 coverage dirs: filters test files, ranks by hit count descending.
    _v8_parse_js = textwrap.dedent("""\
        const fs = require('fs'), path = require('path');
        function parseV8Dir(dir, outFile) {
          if (!fs.existsSync(dir)) return;
          const jsonFiles = fs.readdirSync(dir).filter(f => f.endsWith('.json'));
          if (!jsonFiles.length) return;
          const testPats = [
            /\\/__tests__\\//, /\\/tests?\\//, /\\/specs?\\//, /\\/__mocks__\\//,
            /\\/fixtures?\\//, /\\/e2e\\//, /\\/cypress\\//, /\\/playwright\\//,
            /\\.test\\.[jt]sx?$/, /\\.spec\\.[jt]sx?$/, /\\.test\\.m?[jt]s$/,
            /\\.spec\\.m?[jt]s$/, /\\.stories\\.[jt]sx?$/, /test-utils\\.[jt]sx?$/,
          ];
          const isTest = r => testPats.some(p => p.test(r));
          const fileHits = new Map();
          for (const f of jsonFiles) {
            try {
              const data = JSON.parse(fs.readFileSync(path.join(dir, f)));
              for (const s of (data.result || [])) {
                const url = s.url || '';
                if (!url.startsWith('file:///app/') || url.includes('node_modules')) continue;
                const rel = url.slice('file:///app/'.length);
                if (!/\\.(js|ts|jsx|tsx|mjs|cjs)$/.test(rel)) continue;
                if (isTest(rel)) continue;
                let hits = 0;
                for (const fn of (s.functions || []))
                  for (const r of (fn.ranges || [])) hits += (r.count || 0);
                fileHits.set(rel, (fileHits.get(rel) || 0) + hits);
              }
            } catch(e) {}
          }
          // Sort by hit count descending: most-executed source files first
          const sorted = [...fileHits.entries()].sort((a,b) => b[1]-a[1]).map(e => e[0]);
          if (sorted.length) fs.writeFileSync(outFile, sorted.join('\\n') + '\\n');
        }
        parseV8Dir('/workspace/v8coverage', '/workspace/executed_files.js.log');
        parseV8Dir('/workspace/v8coverage_focused', '/workspace/focused_executed_files.js.log');
    """)

    script = f"""\
set -e
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
{before_repo_set_cmd}
set +e
# ── Phase 1: full run with V8 coverage ───────────────────────────────────────
mkdir -p /workspace/v8coverage
export NODE_V8_COVERAGE=/workspace/v8coverage
bash /workspace/run_script.sh "{selected_tests_csv}" > /workspace/stdout.log 2> /workspace/stderr.log
exit_code=$?
# On failure: verbose rerun for richer stack traces
if [ "$exit_code" -ne 0 ]; then
  bash /workspace/run_script.sh "{selected_tests_csv}" --verbose > /workspace/rerun_stdout.log 2> /workspace/rerun_stderr.log 2>&1 || true
  printf "%s\\n" "$?" > /workspace/rerun_exit_code.txt
fi
# ── Phase 2: focused rerun of failing Jest suites only ───────────────────────
# Jest prints "FAIL src/foo.test.js" lines; extract those suite paths and rerun
# with isolated V8 coverage to get a much smaller, focused execution trace.
python3 - <<'PY'
import re, os, sys
stdout = open('/workspace/stdout.log').read() if os.path.exists('/workspace/stdout.log') else ''
stderr = open('/workspace/stderr.log').read() if os.path.exists('/workspace/stderr.log') else ''
suites = list(dict.fromkeys(re.findall(r'^FAIL\\s+(\\S+)', stdout + '\\n' + stderr, re.MULTILINE)))
if suites:
    open('/workspace/failed_suites.txt', 'w').write('\\n'.join(suites[:15]) + '\\n')
PY
if [ -s /workspace/failed_suites.txt ]; then
  mkdir -p /workspace/v8coverage_focused
  FAILED_SUITES=$(cat /workspace/failed_suites.txt | tr '\\n' ',' | sed 's/,$//')
  NODE_V8_COVERAGE=/workspace/v8coverage_focused \\
  bash /workspace/run_script.sh "$FAILED_SUITES" > /workspace/focused_stdout.log 2>/workspace/focused_stderr.log || true
fi
# ── Parse both V8 coverage dirs: filter test files, rank by hit count ────────
cat > /workspace/parse_v8.js <<'JSEOF'
{_v8_parse_js}
JSEOF
node /workspace/parse_v8.js 2>/dev/null || true
printf "%s\\n" "$exit_code" > /workspace/pytest_exit_code.txt
exit 0
"""
    return textwrap.dedent(script)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_one_instance(
    sample: dict,
    scripts_dir: Path,
    output_dir: Path,
    timeout_seconds: int,
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

    entryscript = build_entryscript(sample)
    workspace = prepare_workspace(
        output_dir, instance_id, entryscript, run_script_path.read_text(), parser_path.read_text()
    )

    image = f"jefzda/sweap-images:{sample['dockerhub_tag']}"
    (sample_dir / "entryscript.sh").write_text(entryscript)
    (sample_dir / "patch.diff").write_text(sample.get("patch", ""))

    timeout_hit, status_code, startup_error = run_container(image, workspace, timeout_seconds)

    stdout_log = (workspace / "stdout.log").read_text(errors="replace") if (workspace / "stdout.log").exists() else ""
    stderr_log = (workspace / "stderr.log").read_text(errors="replace") if (workspace / "stderr.log").exists() else startup_error
    rerun_stdout_log = (workspace / "rerun_stdout.log").read_text(errors="replace") if (workspace / "rerun_stdout.log").exists() else ""
    rerun_stderr_log = (workspace / "rerun_stderr.log").read_text(errors="replace") if (workspace / "rerun_stderr.log").exists() else ""

    exit_code: int | None = None
    exit_code_path = workspace / "pytest_exit_code.txt"
    if exit_code_path.exists():
        try:
            exit_code = int(exit_code_path.read_text().strip())
        except Exception:
            pass

    (sample_dir / "stdout.log").write_text(stdout_log)
    (sample_dir / "stderr.log").write_text(stderr_log)
    if rerun_stdout_log:
        (sample_dir / "rerun_stdout.log").write_text(rerun_stdout_log)
    if rerun_stderr_log:
        (sample_dir / "rerun_stderr.log").write_text(rerun_stderr_log)
    if exit_code is not None:
        (sample_dir / "pytest_exit_code.txt").write_text(f"{exit_code}\n")

    focused_stdout_log = (workspace / "focused_stdout.log").read_text(errors="replace") if (workspace / "focused_stdout.log").exists() else ""
    focused_stderr_log = (workspace / "focused_stderr.log").read_text(errors="replace") if (workspace / "focused_stderr.log").exists() else ""
    if focused_stdout_log:
        (sample_dir / "focused_stdout.log").write_text(focused_stdout_log)
    if focused_stderr_log:
        (sample_dir / "focused_stderr.log").write_text(focused_stderr_log)

    analysis_stdout = "\n".join([stdout_log, rerun_stdout_log]).strip()
    analysis_stderr = "\n".join([stderr_log, rerun_stderr_log]).strip()
    # Session-level V8 coverage (all executed source files, ranked by hit count)
    execution_files = load_execution_files(workspace)
    # Focused V8 coverage from Phase 2 (only failing Jest suites)
    focused_execution_files = load_focused_js_execution_files(workspace)

    stack_file_set = set(extract_stack_files(analysis_stdout, analysis_stderr))
    # Also extract stacks from focused rerun output
    stack_file_set.update(extract_stack_files(focused_stdout_log, focused_stderr_log))
    # Extract file paths from Jest module resolution errors (e.g. missing new files)
    stack_file_set.update(extract_js_module_error_files(analysis_stdout, analysis_stderr))
    stack_files = sorted(stack_file_set)

    failed_tests = extract_failed_tests(analysis_stdout, analysis_stderr)
    failure_trace_excerpt = extract_failure_trace_excerpt(analysis_stdout, analysis_stderr)
    output_incomplete = is_js_output_incomplete(analysis_stdout, analysis_stderr)

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
        pytest_exit_code=exit_code,
        pytest_report_xml="",
        output_incomplete=output_incomplete,
        workspace=str(workspace.resolve()),
        duration_seconds=round(time.time() - start, 3),
    )
