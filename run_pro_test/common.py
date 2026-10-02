"""Shared utilities, data structures, and Docker execution for all language runners."""

from __future__ import annotations

import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


# ── Regex patterns ────────────────────────────────────────────────────────────

PYTHON_FILE_RE = re.compile(r'File "([^"]+)", line \d+')
GENERIC_PATH_RE = re.compile(r"(/[^:\s]+\.[A-Za-z0-9]+):\d+(?::\d+)?")
GENERIC_REL_PATH_RE = re.compile(r"\b([A-Za-z0-9_.\-/]+\.[A-Za-z0-9]+):\d+(?::\d+)?")
JS_STACK_PAREN_RE = re.compile(r"\(([^()]+):\d+:\d+\)")
JS_STACK_AT_RE = re.compile(r"\bat\s+([^()\s]+):\d+:\d+")
PATCH_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PYTHON_MODULE_RE = re.compile(r"module ['\"]([A-Za-z_][\w\.]*)['\"]")
# ImportError: cannot import name 'X' from 'some.module'
# ModuleNotFoundError: No module named 'some.module'
IMPORT_FROM_RE = re.compile(r"(?:from|named)\s+['\"]([A-Za-z_][\w\.]*)['\"]")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class InstanceResult:
    instance_id: str
    repo: str
    docker_image: str
    base_commit: str
    timeout: bool
    container_status_code: int | None
    missing_assets: list[str]
    patch_files: list[str]
    stack_files: list[str]
    execution_files: list[str]
    traceback_matched_patch_files: list[str]
    execution_matched_patch_files: list[str]
    matched_patch_files: list[str]
    stack_contains_patch_file: bool
    error_excerpt: str
    failure_trace_excerpt: str
    failure_trace_log: str
    failed_tests: list[str]
    stdout_log: str
    stderr_log: str
    pytest_exit_code: int | None
    pytest_report_xml: str
    output_incomplete: bool
    workspace: str
    duration_seconds: float


def make_error_result(
    instance_id: str,
    error_excerpt: str,
    *,
    repo: str = "",
    docker_image: str = "",
    base_commit: str = "",
    missing_assets: list[str] | None = None,
    patch_files: list[str] | None = None,
    duration_seconds: float = 0.0,
) -> InstanceResult:
    return InstanceResult(
        instance_id=instance_id,
        repo=repo,
        docker_image=docker_image,
        base_commit=base_commit,
        timeout=False,
        container_status_code=None,
        missing_assets=missing_assets or [],
        patch_files=patch_files or [],
        stack_files=[],
        execution_files=[],
        traceback_matched_patch_files=[],
        execution_matched_patch_files=[],
        matched_patch_files=[],
        stack_contains_patch_file=False,
        error_excerpt=error_excerpt,
        failure_trace_excerpt="",
        failure_trace_log="",
        failed_tests=[],
        stdout_log="",
        stderr_log="",
        pytest_exit_code=None,
        pytest_report_xml="",
        output_incomplete=False,
        workspace="",
        duration_seconds=duration_seconds,
    )


# ── Path / text helpers ───────────────────────────────────────────────────────

def normalize_path(path: str) -> str:
    p = (path or "").strip().replace("\\", "/")
    p = re.sub(r"^['\"]|['\"]$", "", p)
    p = p.replace("/workspace/", "")
    p = p.replace("/app/", "")
    p = p.lstrip("./")
    # Strip leading "app/" prefix (from paths like "app/src/foo.js" in Node.js stacks)
    if p.startswith("app/"):
        p = p[4:]
    return p


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text or "")


# ── Patch utilities ───────────────────────────────────────────────────────────

def extract_patch_files(patch: str) -> list[str]:
    files: list[str] = []
    for _, b_path in PATCH_FILE_RE.findall(patch or ""):
        if b_path == "/dev/null":
            continue
        files.append(normalize_path(b_path))
    return sorted(set(files))


def match_patch_files(patch_files: list[str], candidate_files: list[str]) -> list[str]:
    return sorted(
        {
            patch_file
            for patch_file in patch_files
            for candidate_file in candidate_files
            if (
                candidate_file == patch_file
                or candidate_file.endswith("/" + patch_file)
                or patch_file.endswith("/" + candidate_file)
                # Directory-level match: candidate is a directory containing the patch file
                or ("." not in candidate_file.rsplit("/", 1)[-1]
                    and (patch_file.startswith(candidate_file + "/")
                         or ("/" + candidate_file + "/") in ("/" + patch_file)))
            )
        }
    )


# ── Stack-file extraction ─────────────────────────────────────────────────────

def extract_stack_files_from_text(text: str) -> list[str]:
    text = strip_ansi(text or "")
    out: set[str] = set()

    for pattern in (PYTHON_FILE_RE, GENERIC_PATH_RE, GENERIC_REL_PATH_RE, JS_STACK_PAREN_RE, JS_STACK_AT_RE):
        for match in pattern.findall(text):
            item = match[0] if isinstance(match, tuple) else match
            item = normalize_path(item)
            if not item:
                continue
            if "/" not in item and not item.endswith(
                (".py", ".go", ".js", ".ts", ".tsx", ".java", ".rb", ".php")
            ):
                continue
            out.add(item)

    for pattern in (PYTHON_MODULE_RE, IMPORT_FROM_RE):
        for module_name in pattern.findall(text):
            module_name = module_name.strip(".")
            if not module_name:
                continue
            module_path = normalize_path(module_name.replace(".", "/"))
            if module_path:
                out.add(f"{module_path}.py")
                out.add(f"{module_path}/__init__.py")

    return sorted(out)


def extract_stack_files(stdout_text: str, stderr_text: str) -> list[str]:
    return extract_stack_files_from_text("\n".join([stdout_text or "", stderr_text or ""]))


# ── Output analysis helpers ───────────────────────────────────────────────────

def get_error_excerpt(stdout_text: str, stderr_text: str, max_lines: int = 80) -> str:
    stdout_text = strip_ansi(stdout_text or "")
    stderr_text = strip_ansi(stderr_text or "")
    candidates = []
    for text in (stderr_text, stdout_text):
        for line in text.splitlines():
            low = line.lower()
            if any(
                token in low
                for token in ("traceback", "error", "exception", "failed", "assertion", "panic", "fatal")
            ):
                candidates.append(line)
            if len(candidates) >= max_lines:
                break
        if candidates:
            break
    if not candidates:
        candidates = (stderr_text or stdout_text or "").splitlines()[:max_lines]
    return "\n".join(candidates[:max_lines]).strip()


def extract_failure_trace_excerpt(
    stdout_text: str, stderr_text: str, max_blocks: int = 6, block_lines: int = 40
) -> str:
    text = strip_ansi("\n".join([stdout_text or "", stderr_text or ""]))
    lines = text.splitlines()
    if not lines:
        return ""

    marker_re = re.compile(
        r"(Traceback \(most recent call last\):|^=+ (?:FAILURES|ERRORS) =+$|^\s*panic:|Exception|Error:|^\s*(?:FAILED|ERROR)\s)",
        re.IGNORECASE,
    )
    file_frame_re = re.compile(
        r'File "([^"]+)", line \d+|(/[^:\s]+\.[A-Za-z0-9]+):\d+(?::\d+)?'
        r"|\b([A-Za-z0-9_.\-/]+\.[A-Za-z0-9]+):\d+(?::\d+)?"
    )

    blocks: list[str] = []
    used_ranges: list[tuple[int, int]] = []

    for idx, line in enumerate(lines):
        if not marker_re.search(line):
            continue
        start = max(0, idx - 3)
        end = min(len(lines), idx + block_lines)
        if any(not (end <= s or start >= e) for s, e in used_ranges):
            continue
        block = "\n".join(lines[start:end]).strip()
        if not block or not file_frame_re.search(block):
            continue
        blocks.append(block)
        used_ranges.append((start, end))
        if len(blocks) >= max_blocks:
            break

    if not blocks:
        for idx, line in enumerate(lines):
            if not file_frame_re.search(line):
                continue
            start = max(0, idx - 2)
            end = min(len(lines), idx + 10)
            block = "\n".join(lines[start:end]).strip()
            if block and block not in blocks:
                blocks.append(block)
            if len(blocks) >= max_blocks:
                break

    return "\n\n---\n\n".join(blocks).strip()


# ── Workspace helpers ─────────────────────────────────────────────────────────

def prepare_workspace(
    base_dir: Path, instance_id: str, entryscript: str, run_script: str, parser_script: str
) -> Path:
    workspace = base_dir / instance_id / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "entryscript.sh").write_text(entryscript)
    (workspace / "run_script.sh").write_text(run_script)
    (workspace / "parser.py").write_text(parser_script)
    return workspace



def load_execution_files(workspace: Path) -> list[str]:
    out: set[str] = set()
    for path in sorted(workspace.glob("executed_files*.log")):
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            item = normalize_path(line)
            if item:
                out.add(item)
    return sorted(out)


# ── Docker execution ──────────────────────────────────────────────────────────

def _force_remove_container(container_name: str) -> None:
    """Stop and force-remove a container, suppressing all errors."""
    try:
        subprocess.run(
            ["docker", "stop", "-t", "10", container_name],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass


def run_container(
    image: str, workspace: Path, timeout_seconds: int
) -> tuple[bool, int | None, str]:
    """Pull image if needed and run container.  Returns (timeout_hit, status_code, startup_error).

    Uses subprocess docker CLI instead of the docker-py SDK to avoid HTTP read-timeout
    issues that caused SDK's container.wait() to fire at 60 s even when timeout_seconds
    was set to 2400 s (the SDK used the wrong timeout for long-poll /wait requests).
    """
    timeout_hit = False
    status_code: int | None = None
    startup_error = ""
    container_name = f"swebench_{uuid.uuid4().hex[:12]}"

    # ── Step 1: ensure image is available locally ─────────────────────────────
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True, timeout=15,
        )
        if inspect.returncode != 0:
            pull = subprocess.run(
                ["docker", "pull", image],
                capture_output=True, text=True, timeout=1800,  # 30 min for large images
            )
            if pull.returncode != 0:
                msg = (pull.stderr or pull.stdout or "").strip()[:400]
                return False, None, f"Image pull failed ({pull.returncode}): {msg}"
    except subprocess.TimeoutExpired:
        return False, None, f"Docker timed out inspecting/pulling image: {image}"
    except FileNotFoundError:
        return False, None, "docker not found in PATH"
    except Exception as exc:
        return False, None, f"Docker error during image check: {repr(exc)}"

    # ── Step 2: run container ─────────────────────────────────────────────────
    # Do NOT use --rm: it causes `docker run` to block on container removal
    # after the process exits, which can hang indefinitely if volume unmount
    # is delayed.  We clean up explicitly in the finally block instead.
    try:
        proc = subprocess.run(
            [
                "docker", "run",
                "--name", container_name,
                "--entrypoint", "/bin/bash",
                "-v", f"{workspace.resolve()}:/workspace:rw",
                image,
                "-c", "bash /workspace/entryscript.sh",
            ],
            timeout=timeout_seconds,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        status_code = proc.returncode
    except subprocess.TimeoutExpired:
        timeout_hit = True
    except Exception as exc:
        startup_error = f"Container run error: {repr(exc)}"
    finally:
        _force_remove_container(container_name)

    return timeout_hit, status_code, startup_error
