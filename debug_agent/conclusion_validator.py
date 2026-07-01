"""Cross-check LLM-emitted root_cause_files against the actual PDB frame log.

The LLM occasionally hallucinates a path (file→package, missing/extra dir).
PDB frames are ground-truth: the file the program actually executed in.
Rule:
  - If conclusion files share at least one match (after path_norm) with the
    pdb frame set, keep the matched ones and drop the rest.
  - If there's zero overlap and pdb_log has frames, replace the conclusion
    with the pdb frames (deduped, in first-seen order).
  - If pdb_log is empty, pass conclusion through untouched.

Files and functions are kept aligned by index throughout: filtering or
replacing a file always filters/replaces the corresponding function entry,
so downstream consumers (run_debug._build_candidates) zip the two lists
without misalignment. When the PDB-frames replacement path runs, function
names are unknown for the new files and are emitted as empty strings.
"""
from __future__ import annotations

from .path_norm import paths_match, is_noise_file as is_test_file


def _frames_from_log(pdb_log: list[dict]) -> list[str]:
    """Collect every file PDB visited, in first-seen order, EXCLUDING test
    files. PDB legitimately inspects the test to read the failing assertion,
    but the test is never the bug site."""
    out: list[str] = []
    seen = set()
    for turn in pdb_log:
        for key in ("initial_frame", "current_frame"):
            frame = turn.get(key)
            if not frame:
                continue
            f = frame.get("file")
            if not f or f in seen or is_test_file(f):
                continue
            seen.add(f)
            out.append(f)
    return out


def validate_against_pdb_log(
    conclusion_files: list[str],
    conclusion_functions: list[str],
    pdb_log: list[dict],
) -> tuple[list[str], list[str]]:
    """Cross-check LLM conclusion against PDB frames; return (files, funcs)
    aligned by index. Always drop test files from the conclusion (bugs are
    in production code) — and drop the corresponding function entry. If the
    surviving conclusion overlaps with PDB-observed production frames, keep
    the overlap; if not, fall back to the production frames (with empty
    function names, since PDB tells us files but not the bug function); if
    PDB has no production-code evidence either, pass the (test-filtered)
    conclusion through."""
    pairs: list[tuple[str, str]] = []
    for i, f in enumerate(conclusion_files):
        fn = conclusion_functions[i] if i < len(conclusion_functions) else ""
        pairs.append((f, fn))

    cleaned = [(f, fn) for f, fn in pairs if not is_test_file(f)]
    frames = _frames_from_log(pdb_log)
    if not frames:
        return [f for f, _ in cleaned], [fn for _, fn in cleaned]

    kept = [(f, fn) for f, fn in cleaned if any(paths_match(f, x) for x in frames)]
    if kept:
        return [f for f, _ in kept], [fn for _, fn in kept]

    # Replace path: PDB frames as files, unknown functions as empty strings.
    return list(frames), [""] * len(frames)
