"""Go test runner: entryscript generation and output analysis."""

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


# ── Go-specific regex patterns ────────────────────────────────────────────────

GO_FAIL_RE = re.compile(r"^--- FAIL:\s+(\S+)", re.MULTILINE)
GO_PACKAGE_FAIL_RE = re.compile(r"^FAIL\s+(\S+)", re.MULTILINE)
GO_TEST_OUTPUT_INCOMPLETE_RE = re.compile(r"^=== RUN\s+\S+", re.MULTILINE)
# Build error: "path/to/file.go:line:col: message"  (relative path, no leading /)
GO_BUILD_ERROR_FILE_RE = re.compile(r"^([^/\s][^\s:]*\.go):\d+:\d+:", re.MULTILINE)
# Goroutine stack trace: "\t/app/foo/bar.go:42 ..." or "foo/bar.go:42 +"
GO_GOROUTINE_FILE_RE = re.compile(r"[\t\s](?:/app/)?([a-zA-Z0-9_./-]+\.go):\d+\s")
# Panic prefix lines in test output
GO_PANIC_RE = re.compile(r"^panic:", re.MULTILINE)


# ── Go output analysis ────────────────────────────────────────────────────────

def extract_failed_tests(stdout_text: str, stderr_text: str) -> list[str]:
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))
    out: list[str] = []
    seen: set[str] = set()

    # Standard text format: "--- FAIL: TestXxx"
    for match in GO_FAIL_RE.finditer(text):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            out.append(name)

    # JSON format: {"Action":"fail","Test":"TestXxx",...} or {"Action":"build-fail",...}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("Action") == "fail":
            test_name = obj.get("Test", "")
            if test_name and test_name not in seen:
                seen.add(test_name)
                out.append(test_name)
            elif not test_name:
                pkg = obj.get("Package", "")
                if pkg and pkg not in seen:
                    seen.add(pkg)
                    out.append(pkg)
        elif obj.get("Action") == "build-fail":
            pkg = obj.get("Package", "")
            label = f"[build-fail] {pkg}" if pkg else "[build-fail]"
            if label not in seen:
                seen.add(label)
                out.append(label)

    # Build errors in text output (e.g. "file.go:line:col: assignment mismatch")
    # Count as a failure even when no test ran
    if not out and GO_BUILD_ERROR_FILE_RE.search(text):
        out.append("[build-error]")

    return out




def load_go_coverage_files(workspace: Path) -> list[str]:
    """Parse executed_files.go.log written from coverage.out: strip Go module prefix.

    Filters out test files (*_test.go) since we want production code only.
    """
    from common import normalize_path
    cov_log = workspace / "executed_files.go.log"
    if not cov_log.exists():
        return []
    # Read go.mod (saved to workspace by entryscript) to strip module prefix precisely
    go_mod_prefix = ""
    for candidate in [workspace / "go_module.txt", workspace / "go.mod"]:
        if candidate.exists():
            for mline in candidate.read_text(errors="replace").splitlines():
                mline = mline.strip()
                if mline.startswith("module "):
                    go_mod_prefix = mline[len("module "):].strip()
                    break
            if go_mod_prefix:
                break

    out: set[str] = set()
    for line in cov_log.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        # If we know the module prefix, strip it directly
        if go_mod_prefix and line.startswith(go_mod_prefix + "/"):
            rel = line[len(go_mod_prefix) + 1:]
        else:
            # Go coverage paths are import paths: github.com/org/repo/pkg/file.go
            # Strip the host + org/repo prefix (first 3 segments for github.com, etc.)
            parts = line.split("/")
            host = parts[0] if parts else ""
            if "." in host and len(parts) > 3:
                # github.com/org/repo/... → strip first 3 segments
                rel = "/".join(parts[3:])
            elif len(parts) > 2:
                rel = "/".join(parts[2:])
            else:
                rel = line
        rel = normalize_path(rel)
        # Exclude test files from the execution trace — they are not buggy production code
        if rel and not rel.endswith("_test.go"):
            out.add(rel)
    return sorted(out)


def load_go_focused_coverage_files(workspace: Path) -> list[str]:
    """Parse focused_executed_files.go.log from Phase 2 focused rerun coverprofiles."""
    from common import normalize_path
    cov_log = workspace / "focused_executed_files.go.log"
    if not cov_log.exists():
        return []
    go_mod_prefix = ""
    for candidate in [workspace / "go_module.txt", workspace / "go.mod"]:
        if candidate.exists():
            for mline in candidate.read_text(errors="replace").splitlines():
                mline = mline.strip()
                if mline.startswith("module "):
                    go_mod_prefix = mline[len("module "):].strip()
                    break
            if go_mod_prefix:
                break
    out: set[str] = set()
    for line in cov_log.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if go_mod_prefix and line.startswith(go_mod_prefix + "/"):
            rel = line[len(go_mod_prefix) + 1:]
        else:
            parts = line.split("/")
            host = parts[0] if parts else ""
            if "." in host and len(parts) > 3:
                rel = "/".join(parts[3:])
            elif len(parts) > 2:
                rel = "/".join(parts[2:])
            else:
                rel = line
        rel = normalize_path(rel)
        if rel and not rel.endswith("_test.go"):
            out.add(rel)
    return sorted(out)


def extract_build_error_files(stdout_text: str, stderr_text: str) -> list[str]:
    """Extract Go source files from build/compile errors AND goroutine stack traces.

    Covers:
    - Build failures: "foo/bar.go:42:10: assignment mismatch"
    - Panic stack traces: goroutine lines like "\t/app/foo/bar.go:42 +"
    - Test output failure context: "foo/bar.go:42: Error ..."
    - JSON test output "Output" fields containing file references
    """
    from common import normalize_path
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))

    # Unwrap JSON "Output" fields to plain text so regex can match
    plain_lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                output = obj.get("Output", "")
                if output:
                    plain_lines.append(output)
            except Exception:
                pass
        plain_lines.append(line)
    expanded = "\n".join(plain_lines)

    out: set[str] = set()
    test_dirs: set[str] = set()
    for match in GO_BUILD_ERROR_FILE_RE.finditer(expanded):
        rel = normalize_path(match.group(1))
        if rel:
            out.add(rel)
            # Track directories of _test.go files with build errors —
            # the production file may be new (not yet on disk)
            if rel.endswith("_test.go"):
                d = str(Path(rel).parent)
                if d and d != ".":
                    test_dirs.add(d)
    for match in GO_GOROUTINE_FILE_RE.finditer(expanded):
        rel = normalize_path(match.group(1))
        if rel and not rel.endswith("_test.go"):
            out.add(rel)
    # For build-error directories where no production .go file was found,
    # add the directory path itself as a candidate (handles new-file patches)
    prod_dirs = {str(Path(f).parent) for f in out if not f.endswith("_test.go")}
    for d in test_dirs - prod_dirs:
        out.add(d)
    return sorted(out)


def load_build_error_pkg_files(workspace: Path) -> list[str]:
    """Load production .go files discovered from build error package analysis.

    The entryscript extracts referenced packages from build errors (type paths,
    undefined symbols, failing package names) and lists their non-test .go files.
    """
    from common import normalize_path
    log = workspace / "build_error_pkg_files.log"
    if not log.exists():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in log.read_text(errors="replace").splitlines():
        rel = normalize_path(line)
        if rel and rel not in seen and not rel.endswith("_test.go"):
            seen.add(rel)
            out.append(rel)
    return out


def is_go_output_incomplete(stdout_text: str, stderr_text: str) -> bool:
    """Heuristic: RUN lines found but no PASS/FAIL summary yet."""
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))
    has_run = bool(GO_TEST_OUTPUT_INCOMPLETE_RE.search(text))
    has_result = bool(re.search(r"^(ok|FAIL)\s+\S+", text, re.MULTILINE))
    return has_run and not has_result


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

    script = f"""\
set -e
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
{before_repo_set_cmd}
set +e
# Save go.mod module path so Python can strip it precisely
if [ -f /app/go.mod ]; then
  grep '^module ' /app/go.mod | head -1 > /workspace/go_module.txt || true
  cp /app/go.mod /workspace/go.mod 2>/dev/null || true
fi
# Enable Go coverage collection: GOCOVERDIR for Go 1.20+
mkdir -p /workspace/coverdir
export GOCOVERDIR=/workspace/coverdir
export GOFLAGS="${{GOFLAGS:-}} -cover"
# GOTRACEBACK=all: print full goroutine stacks on panic (includes production file paths)
export GOTRACEBACK=all
bash /workspace/run_script.sh "{selected_tests_csv}" > /workspace/stdout.log 2> /workspace/stderr.log
exit_code=$?
# Detect failure from output even if exit_code is 0 (run_script.sh may use || true)
if [ "$exit_code" -eq 0 ]; then
  if grep -qE -- '--- FAIL:|^FAIL[[:space:]]|"Action":"fail"|\\[build failed\\]' /workspace/stdout.log /workspace/stderr.log 2>/dev/null; then
    exit_code=1
  fi
fi
# On failure: verbose rerun + Phase 2 focused rerun of only failing tests
if [ "$exit_code" -ne 0 ]; then
  bash /workspace/run_script.sh "{selected_tests_csv}" -v > /workspace/rerun_stdout.log 2> /workspace/rerun_stderr.log 2>&1 || true
  printf "%s\\n" "$?" > /workspace/rerun_exit_code.txt
  # ── Phase 2: extract failing test functions, rerun with -coverprofile ──
  python3 - <<'PY'
import re, json
from pathlib import Path

stdout = Path("/workspace/stdout.log").read_text(errors="replace")
stderr = Path("/workspace/stderr.log").read_text(errors="replace")
rerun_stdout = Path("/workspace/rerun_stdout.log").read_text(errors="replace") if Path("/workspace/rerun_stdout.log").exists() else ""
text = stdout + "\\n" + stderr + "\\n" + rerun_stdout
seen = set()
tests = []
pkgs = set()
# Extract from plain text format
for m in re.finditer(r"^--- FAIL:\\s+(\\S+)", text, re.MULTILINE):
    name = m.group(1)
    if name not in seen:
        seen.add(name)
        tests.append(name)
for m in re.finditer(r"^FAIL\\s+(\\S+)", text, re.MULTILINE):
    pkg = m.group(1)
    if not pkg.startswith("---"):
        pkgs.add(pkg)
# Extract from JSON test output format (go test -json)
for line in text.splitlines():
    line = line.strip()
    if not line.startswith("{{"):
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    action = obj.get("Action", "")
    if action == "fail":
        test_name = obj.get("Test", "")
        pkg_name = obj.get("Package", "")
        if test_name and test_name not in seen:
            seen.add(test_name)
            tests.append(test_name)
        if pkg_name:
            pkgs.add(pkg_name)
    elif action == "build-fail":
        pkg_name = obj.get("Package", "")
        if pkg_name:
            pkgs.add(pkg_name)
Path("/workspace/failed_go_tests.json").write_text(json.dumps({{"tests": tests, "packages": sorted(pkgs)}}))
PY
  # Focused rerun: only failing tests with coverprofile for precise execution trace
  python3 - <<'PY'
import json, subprocess, os
from pathlib import Path

data = json.loads(Path("/workspace/failed_go_tests.json").read_text())
tests = data.get("tests", [])
packages = data.get("packages", [])
if not tests or not packages:
    raise SystemExit(0)
# Build -run regex: TestFoo|TestBar (strip subtest suffixes like /subtest_name)
top_tests = set()
for t in tests:
    top = t.split("/")[0]
    top_tests.add(top)
run_regex = "|".join(sorted(top_tests))

# Read module prefix for coverpkg scoping
go_mod_prefix = ""
mod_file = Path("/workspace/go_module.txt")
if mod_file.exists():
    for mline in mod_file.read_text().splitlines():
        mline = mline.strip()
        if mline.startswith("module "):
            go_mod_prefix = mline[len("module "):].strip()
            break

# Build coverpkg list: only failing packages + their parent (avoids compiling entire repo)
# This prevents C dependency compilation failures from blocking coverage
coverpkg_set = set()
for pkg in packages:
    # Convert full import path to relative ./path for coverpkg
    if go_mod_prefix and pkg.startswith(go_mod_prefix + "/"):
        rel = "./" + pkg[len(go_mod_prefix) + 1:]
    elif pkg.startswith("./"):
        rel = pkg
    else:
        rel = pkg
    coverpkg_set.add(rel)
    # Also add parent package for cross-package calls
    parent = "/".join(rel.rstrip("/").split("/")[:-1])
    if parent and parent != ".":
        coverpkg_set.add(parent + "/...")
coverpkg_arg = ",".join(sorted(coverpkg_set)) if coverpkg_set else "./..."

# Run focused test with coverprofile against each failing package
mkdir = Path("/workspace/focused_coverdir")
mkdir.mkdir(exist_ok=True)
env = {{**os.environ, "GOFLAGS": "", "GOCOVERDIR": str(mkdir)}}
for i, pkg in enumerate(packages[:5]):
    result = subprocess.run(
        ["go", "test", "-v", "-count=1", "-run", run_regex,
         f"-coverprofile=/workspace/focused_coverage_{{i}}.out",
         f"-coverpkg={{coverpkg_arg}}",
         pkg],
        capture_output=True, text=True, cwd="/app", env=env, timeout=120,
    )
    Path(f"/workspace/focused_go_stdout_{{i}}.log").write_text(result.stdout)
    Path(f"/workspace/focused_go_stderr_{{i}}.log").write_text(result.stderr)
PY
fi
# ── Build error + failing package analysis: extract packages → list production .go files ──
python3 - <<'PY'
import os, re, json
from pathlib import Path

stdout = Path("/workspace/stdout.log").read_text(errors="replace")
stderr = Path("/workspace/stderr.log").read_text(errors="replace")
rerun_stdout = Path("/workspace/rerun_stdout.log").read_text(errors="replace") if Path("/workspace/rerun_stdout.log").exists() else ""
text = stdout + "\\n" + stderr + "\\n" + rerun_stdout

# Unwrap JSON "Output" fields and extract package info from JSON test events
plain_lines = []
json_fail_pkgs = set()
for line in text.splitlines():
    line_s = line.strip()
    if line_s.startswith("{{"):
        try:
            obj = json.loads(line_s)
            output = obj.get("Output", "")
            if output:
                plain_lines.append(output)
            # Extract package from fail/build-fail events
            action = obj.get("Action", "")
            pkg = obj.get("Package", "")
            if action in ("fail", "build-fail") and pkg:
                json_fail_pkgs.add(pkg)
        except Exception:
            pass
    plain_lines.append(line)
expanded = "\\n".join(plain_lines)

# Read module prefix from go.mod
go_mod_prefix = ""
mod_file = Path("/workspace/go_module.txt")
if mod_file.exists():
    for mline in mod_file.read_text().splitlines():
        mline = mline.strip()
        if mline.startswith("module "):
            go_mod_prefix = mline[len("module "):].strip()
            break

pkg_paths = set()

# From JSON fail events: strip module prefix
for full_pkg in json_fail_pkgs:
    if go_mod_prefix and full_pkg.startswith(go_mod_prefix + "/"):
        pkg_paths.add(full_pkg[len(go_mod_prefix) + 1:])

# 1. Extract quoted package paths from build error type references
for m in re.finditer(r'"([^"]+)"\\.\\w+', expanded):
    full_pkg = m.group(1)
    if go_mod_prefix and full_pkg.startswith(go_mod_prefix + "/"):
        pkg_paths.add(full_pkg[len(go_mod_prefix) + 1:])

# 2. Extract from "undefined: <pkg>.<Symbol>" patterns
for m in re.finditer(r'undefined:\\s+(\\w[\\w.]*)\\.\\w+', expanded):
    parts = m.group(1).split(".")
    if len(parts) == 1:
        pkg_paths.add(parts[0])

# 3. Extract from build error file paths: _test.go -> same directory production files
build_err_re = re.compile(r'^([^/\\s][^\\s:]*_test\\.go):\\d+:\\d+:', re.MULTILINE)
for m in build_err_re.finditer(expanded):
    test_file = m.group(1)
    d = str(Path(test_file).parent)
    if d and d != ".":
        pkg_paths.add(d)

# 4. Extract from "FAIL <full_package> [build failed]" lines (plain text)
for m in re.finditer(r'^FAIL\\s+(\\S+)\\s+\\[build failed\\]', expanded, re.MULTILINE):
    full_pkg = m.group(1)
    if go_mod_prefix and full_pkg.startswith(go_mod_prefix + "/"):
        pkg_paths.add(full_pkg[len(go_mod_prefix) + 1:])

# 5. For each package path, list production .go files via filesystem
prod_files = []
app_root = Path("/app")
for pkg_dir in sorted(pkg_paths):
    d = app_root / pkg_dir
    if not d.is_dir():
        continue
    for f in sorted(d.glob("*.go")):
        if not f.name.endswith("_test.go"):
            rel = str(f.relative_to(app_root))
            prod_files.append(rel)

if prod_files:
    Path("/workspace/build_error_pkg_files.log").write_text("\\n".join(prod_files) + "\\n")
PY
# Convert GOCOVERDIR data (Go 1.20+) to coverage.out text format
if ls /workspace/coverdir/*.covcounters 2>/dev/null | head -1 | grep -q .; then
  go tool covdata textfmt -i=/workspace/coverdir -o=/workspace/coverage.out 2>/dev/null || true
fi
# Parse coverage files into executed source file import paths
for covfile in /workspace/coverage.out /workspace/focused_coverage_*.out; do
  [ -f "$covfile" ] || continue
  grep -v '^mode:' "$covfile" | awk -F: '{{print $1}}' | sort -u >> /workspace/executed_files.go.log 2>/dev/null || true
done
# Deduplicate
if [ -f /workspace/executed_files.go.log ]; then
  sort -u /workspace/executed_files.go.log -o /workspace/executed_files.go.log
fi
# Parse focused coverprofiles separately (higher signal, fewer files)
for covfile in /workspace/focused_coverage_*.out; do
  [ -f "$covfile" ] || continue
  grep -v '^mode:' "$covfile" | awk -F: '{{print $1}}' | sort -u >> /workspace/focused_executed_files.go.log 2>/dev/null || true
done
if [ -f /workspace/focused_executed_files.go.log ]; then
  sort -u /workspace/focused_executed_files.go.log -o /workspace/focused_executed_files.go.log
fi
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

    analysis_stdout = "\n".join([stdout_log, rerun_stdout_log]).strip()
    analysis_stderr = "\n".join([stderr_log, rerun_stderr_log]).strip()
    # Go coverage files (from coverage.out, excluding test files) + generic execution logs
    execution_files = sorted(set(load_go_coverage_files(workspace)) | set(load_execution_files(workspace)))
    # Phase 2 focused coverage: only failing tests, much fewer files (higher signal)
    focused_execution_files = load_go_focused_coverage_files(workspace)
    # Build error files: for build failures, extract .go files mentioned in compile errors
    build_error_files = extract_build_error_files(analysis_stdout, analysis_stderr)
    # Build error package files: production .go files from packages referenced in build errors
    # (discovered by entryscript using filesystem listing inside the container)
    build_pkg_files = load_build_error_pkg_files(workspace)
    # Also extract stacks from focused rerun stdout
    focused_go_text = ""
    for i in range(5):
        p = workspace / f"focused_go_stdout_{i}.log"
        if p.exists():
            focused_go_text += p.read_text(errors="replace") + "\n"

    stack_file_set = set(extract_stack_files(analysis_stdout, analysis_stderr))
    # Build error files are effectively "stack files" — they point to the affected code
    stack_file_set.update(build_error_files)
    # Package-level files from build error analysis (high signal for build failures)
    stack_file_set.update(build_pkg_files)
    if focused_go_text:
        stack_file_set.update(extract_stack_files(focused_go_text, ""))
    stack_files = sorted(stack_file_set)

    failed_tests = extract_failed_tests(analysis_stdout, analysis_stderr)
    failure_trace_excerpt = extract_failure_trace_excerpt(analysis_stdout, analysis_stderr)
    output_incomplete = is_go_output_incomplete(analysis_stdout, analysis_stderr)

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
    if build_error_files:
        (sample_dir / "build_error_files.log").write_text("\n".join(build_error_files) + "\n")
    if build_pkg_files:
        (sample_dir / "build_error_pkg_files.log").write_text("\n".join(build_pkg_files) + "\n")

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
        execution_files=execution_files,
        traceback_matched_patch_files=traceback_matched,
        execution_matched_patch_files=execution_matched,
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
