"""Parse the last successful pytest action in a debug trajectory into frames."""
from __future__ import annotations

import re
from dataclasses import dataclass

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_FRAMEWORK_RE = re.compile(
    r"(^|/)(_pytest|pluggy|pytest|py|site-packages|dist-packages)(/|$)"
)
_TEST_PATH_RE = re.compile(r"(^|/)tests?(/|$)|(^|/)test_[^/]+\.py$|(^|/)[^/]+_test\.py$")

_STD_FRAME_RE = re.compile(
    r'File "(?P<path>[^"]+)", line (?P<ln>\d+), in (?P<func>\S+)'
)
_PYTEST_SHORT_RE = re.compile(
    # Pytest short summary lines look like "path:ln: ExcType" or
    # "path:ln: ExcType: message". Exception class names start with an
    # uppercase letter by convention, which distinguishes them from the
    # long-form frame header "path:ln: in <funcname>".
    r'^(?P<path>[^\s:]+\.py):(?P<ln>\d+):\s*(?P<err>[A-Z][\w.]*)(?:\s*$|:)',
    re.MULTILINE,
)
_PYTEST_LONG_HEADER_RE = re.compile(
    r'^(?P<path>[^\s:]+\.py):(?P<ln>\d+):\s*(?:in\s+(?P<func>\S+))?',
    re.MULTILINE,
)
_E_LINE_RE = re.compile(r'^E\s+(?P<body>.+)$', re.MULTILINE)
_E_TYPED_RE = re.compile(r'^(?P<etype>[A-Za-z_][\w.]*):\s*(?P<emsg>.*)$')
_CHAIN_SEP = "During handling of the above exception, another exception occurred:"


@dataclass
class TracebackFrame:
    file: str
    lineno: int
    qualname: str
    code_line: str
    is_test_code: bool


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s or "")


def _is_framework(path: str) -> bool:
    return bool(_FRAMEWORK_RE.search(path))


def _is_test(path: str) -> bool:
    return bool(_TEST_PATH_RE.search(path))


def _last_pytest_turn(trajectory: list[dict]) -> tuple[int, str] | None:
    for i in range(len(trajectory) - 1, -1, -1):
        t = trajectory[i]
        if t.get("kind") != "action" or t.get("action_name") != "pytest":
            continue
        out = t.get("tool_output") or ""
        if not out or out.startswith("ERROR"):
            continue
        return i, out
    return None


def _parse_exc(text: str) -> tuple[str, str]:
    # Prefer the pytest short-summary line for exc_type (qualified exception).
    short_matches = list(_PYTEST_SHORT_RE.finditer(text))
    exc_type = short_matches[-1].group("err") if short_matches else ""

    # E lines carry the message. Pytest expands `assert a == b` into multi-line
    # diff hints (`E    - expected`, `E    ?   ^`, `E    + actual`) — we want
    # the *first* typed E line (e.g. "AssertionError: ..."), not the last.
    # Fall back to the first non-diff E line if no typed form is present.
    exc_msg = ""
    first_typed = None
    first_plain = None
    for m in _E_LINE_RE.finditer(text):
        body = m.group("body").strip()
        # Diff hints from pytest's assertion rewriting start with these chars.
        if body[:1] in {"-", "+", "?"}:
            continue
        if first_plain is None:
            first_plain = body
        if _E_TYPED_RE.match(body):
            first_typed = body
            break
    if first_typed is not None:
        typed = _E_TYPED_RE.match(first_typed)
        exc_msg = typed.group("emsg").strip()
        if not exc_type:
            exc_type = typed.group("etype")
    elif first_plain is not None:
        exc_msg = first_plain
    return exc_type, exc_msg


def _extract_frames(text: str) -> list[TracebackFrame]:
    frames: list[TracebackFrame] = []
    seen: set[tuple[str, int]] = set()

    for m in _STD_FRAME_RE.finditer(text):
        path, ln, func = m.group("path"), int(m.group("ln")), m.group("func")
        if _is_framework(path) or not path.endswith(".py"):
            continue
        key = (path, ln)
        if key in seen:
            continue
        seen.add(key)
        frames.append(TracebackFrame(path, ln, func, "", _is_test(path)))

    for m in _PYTEST_LONG_HEADER_RE.finditer(text):
        path, ln = m.group("path"), int(m.group("ln"))
        if _is_framework(path) or not path.endswith(".py"):
            continue
        key = (path, ln)
        if key in seen:
            continue
        seen.add(key)
        frames.append(TracebackFrame(path, ln, m.group("func") or "", "", _is_test(path)))

    return frames


def parse_text(raw: str, evidence_ref: str) -> dict:
    """Parse a raw pytest failure output into frames + symptom.

    Returns {"status": "ok"|"unavailable", ...}. Used by both the
    repro-test-run path (primary) and the trajectory-scan path (fallback).
    """
    if not raw:
        return {"frames": [], "symptom": None, "exc_type": "", "exc_msg": "",
                "evidence_refs": [], "status": "unavailable",
                "reason": "empty-output"}
    text = _strip_ansi(raw)
    if _CHAIN_SEP in text:
        text = text.rsplit(_CHAIN_SEP, 1)[1]

    frames = _extract_frames(text)
    exc_type, exc_msg = _parse_exc(text)
    if not frames and not exc_type:
        return {"frames": [], "symptom": None, "exc_type": "", "exc_msg": "",
                "evidence_refs": [evidence_ref], "status": "unavailable",
                "reason": "no-frames-parsed"}

    symptom = None
    if frames:
        f = frames[-1]
        symptom = {"file": f.file, "lineno": f.lineno, "qualname": f.qualname,
                   "exc_type": exc_type, "exc_msg": exc_msg,
                   "is_test_code": f.is_test_code}

    return {"frames": frames, "symptom": symptom, "exc_type": exc_type,
            "exc_msg": exc_msg, "evidence_refs": [evidence_ref], "status": "ok"}


def parse_failing_traceback(trajectory: list[dict]) -> dict:
    """Fallback: scan trajectory for a pytest turn with traceback output."""
    found = _last_pytest_turn(trajectory)
    if found is None:
        return {"frames": [], "symptom": None, "exc_type": "", "exc_msg": "",
                "evidence_refs": [], "status": "unavailable",
                "reason": "no-failing-traceback-in-trajectory"}
    idx, raw = found
    return parse_text(raw, f"trajectory[{idx}].tool_output")
