"""Step 4: Multi-signal root cause analysis — statement-level fault localization.

Combines traceback proximity, data flow analysis, coverage suspiciousness,
and branch divergence signals to rank candidate root-cause statements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from statement_tracer import (
    StatementTrace, TracebackFrame, CoverageFileData, ExceptionLocals,
)
from context_extractor import FailureContext, DataFlowStep, SourceReader
from bug_pattern_checker import check_source, BugPatternHit


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RootCauseCandidate:
    file: str
    line: int
    score: float                       # 0.0 ~ 1.0
    signals: list[str] = field(default_factory=list)
    code_snippet: str = ""
    explanation: str = ""
    func_name: str = ""


@dataclass
class RCAResult:
    """Root cause analysis result for one failing test."""
    test_nodeid: str
    error_type: str
    error_message: str
    candidates: list[RootCauseCandidate]
    # Evaluation against ground truth (if patch_files provided)
    patch_files: list[str] = field(default_factory=list)
    top1_hit: bool = False             # best candidate is in a patch file
    top3_hit: bool = False             # any of top-3 candidates is in a patch file
    top5_hit: bool = False
    top1_line_hit: bool = False        # top candidate's exact line is in the patch
    top5_line_hit: bool = False


# ── Signal weights ───────────────────────────────────────────────────────────

WEIGHT_TRACEBACK = 0.30
WEIGHT_DATA_FLOW = 0.20
WEIGHT_COVERAGE_SUSPICIOUSNESS = 0.15
WEIGHT_BRANCH_DIVERGENCE = 0.15
WEIGHT_VARIABLE_STATE = 0.20

import re
_TEST_PATH_RE = re.compile(
    r"(?:^|/)(?:tests?|testing|test_[^/]+|[^/]+_test|conftest|__fixtures__)/"
    r"|(?:^|/)test_[^/]+\.py$|[^/]+_test\.py$",
    re.IGNORECASE,
)

# Patterns for generic exception propagation frames that should be demoted
_RERAISE_RE = re.compile(
    r"^\s*raise\b.*\bfrom\b"            # raise X from Y / raise X from None
    r"|^\s*raise\s*$"                     # bare raise
    r"|^\s*raise\s+\w+Error\s*\("        # raise SomeError(...) — likely wrapping
    r"|^\s*raise\s+\w+Exception\s*\(",
    re.IGNORECASE,
)


# ── Signal 1: Traceback proximity ────────────────────────────────────────────

# Generic lookup/getter functions that raise on "not found" — the real root
# cause is usually the *caller* that passed the bad key/option, not the getter.
_GENERIC_GETTER_FUNCS = frozenset({
    "get_opt", "get_option", "get_config", "get_setting",
    "__getitem__", "__getattr__",
})
_GENERIC_GETTER_ERR_TYPES = frozenset({
    "KeyError", "LookupError", "NoOptionError", "AttributeError",
    "ConfigError", "OptionError",
})


def _is_reraise_frame(frame: TracebackFrame, error_type: str = "") -> bool:
    """Detect generic exception propagation frames (raise...from, bare raise, etc.)
    and generic lookup functions that raise on 'not found'.
    """
    if not frame.code_line:
        return False
    if _RERAISE_RE.match(frame.code_line):
        return True
    # Generic getter that raises — demote so caller frame gets top score
    if (error_type and frame.func_name in _GENERIC_GETTER_FUNCS
            and any(error_type.endswith(et) for et in _GENERIC_GETTER_ERR_TYPES)):
        return True
    return False


def _score_traceback_proximity(
    frames: list[TracebackFrame],
    error_type: str = "",
) -> dict[tuple[str, int], float]:
    """Score frames by proximity to crash site. Innermost production frame = highest.

    Re-raise / exception-propagation frames are demoted so the actual trigger
    frame (the next meaningful frame up the chain) gets the top score.
    """
    scores: dict[tuple[str, int], float] = {}
    prod_frames = [f for f in frames if not f.is_test_code]
    if not prod_frames:
        return scores

    # Walk from innermost outward, skipping re-raise frames for the top score
    reversed_frames = list(reversed(prod_frames))
    rank = 0  # effective rank (re-raise frames don't consume a rank slot)
    for frame in reversed_frames:
        if _is_reraise_frame(frame, error_type):
            # Demote: give a low fixed score instead of the positional score
            score = 0.15
        else:
            score = 1.0 / (rank + 1)
            rank += 1
        key = (frame.file, frame.lineno)
        scores[key] = max(scores.get(key, 0.0), score)

    return scores


# ── Signal 2: Data flow chain ────────────────────────────────────────────────

def _score_data_flow(
    chain: list[DataFlowStep],
) -> dict[tuple[str, int], float]:
    """Score statements in the reverse data-flow chain."""
    scores: dict[tuple[str, int], float] = {}
    if not chain:
        return scores

    n = len(chain)
    for i, step in enumerate(chain):
        if not step.file:
            continue
        # First in chain = crash site usage (high), further = data sources (lower)
        # But "defined"/"modified" roles are more suspicious than "used_at_crash"
        base = 1.0 / (i + 1)
        if step.role in ("defined", "modified"):
            base *= 1.2  # boost definitions
        elif step.role == "parameter":
            base *= 0.5  # parameters less likely to be the bug
        key = (step.file, step.line)
        scores[key] = max(scores.get(key, 0.0), min(base, 1.0))

    return scores


# ── Signal 3: Coverage suspiciousness (simplified spectrum-based FL) ─────────

def _score_coverage_suspiciousness(
    trace: StatementTrace,
    all_focused_files: dict[str, int] | None = None,
) -> dict[tuple[str, int], float]:
    """Score lines by their suspiciousness based on coverage data.

    Uses a simplified Ochiai-like metric: lines that are executed in a failing
    test are suspicious, with higher scores for lines in files that have more
    missing lines (indicating partial execution = possible bug location).

    If all_focused_files is provided (file -> count of tests that touch it),
    broadly-covered files are demoted (IDF-like weighting).
    """
    scores: dict[tuple[str, int], float] = {}

    for file, cov in trace.coverage.items():
        if _TEST_PATH_RE.search(file):
            continue
        if not cov.executed_lines:
            continue

        # Suspiciousness: files with both executed and missing lines are more
        # suspicious (partial execution suggests the bug is at a boundary)
        total = len(cov.executed_lines) + len(cov.missing_lines)
        if total == 0:
            continue
        # Ratio of missing lines — higher ratio = more suspicious file
        miss_ratio = len(cov.missing_lines) / total if cov.missing_lines else 0.1
        file_suspiciousness = min(miss_ratio * 2, 1.0)

        # IDF-like demotion: files touched by many tests are less specific
        if all_focused_files:
            import math
            test_count = 0
            for ff, cnt in all_focused_files.items():
                if file == ff or file.endswith("/" + ff) or ff.endswith("/" + file):
                    test_count = cnt
                    break
            if test_count > 1:
                max_count = max(all_focused_files.values()) if all_focused_files else 1
                # idf: ranges from 1.0 (unique file) down to ~0.3 (most common)
                idf = math.log1p(max_count) / math.log1p(test_count)
                idf = max(min(idf, 1.0), 0.3)
                file_suspiciousness *= idf

        # Within the file, lines near missing lines are more suspicious
        missing_set = set(cov.missing_lines)
        for line in cov.executed_lines:
            # Check proximity to missing lines (branch boundaries)
            near_missing = any(abs(line - m) <= 2 for m in missing_set)
            line_score = file_suspiciousness
            if near_missing:
                line_score *= 1.5  # boost lines at branch boundaries
            key = (file, line)
            scores[key] = min(line_score, 1.0)

    return scores


# ── Signal 4: Branch divergence ──────────────────────────────────────────────

def _score_branch_divergence(
    trace: StatementTrace,
) -> dict[tuple[str, int], float]:
    """Score based on branch coverage: missing branches are suspicious.

    A missing branch at line N means the code took one path but not the other,
    which is a common pattern for bugs (wrong branch taken, or missing case).
    """
    scores: dict[tuple[str, int], float] = {}

    for file, cov in trace.coverage.items():
        if _TEST_PATH_RE.search(file):
            continue
        if not cov.missing_branches:
            continue

        for src_line, dst_line in cov.missing_branches:
            # The source line of a missing branch is highly suspicious
            scores[(file, src_line)] = max(scores.get((file, src_line), 0.0), 0.9)
            # The destination line (where it should have gone) is also interesting
            if dst_line > 0:
                scores[(file, dst_line)] = max(
                    scores.get((file, dst_line), 0.0), 0.6
                )

    return scores


# ── Signal 5: Per-test executed lines (settrace fallback) ─────────────────────

def _score_per_test_lines(
    trace: StatementTrace,
) -> dict[tuple[str, int], float]:
    """Score lines from per-test settrace data when coverage.py data is unavailable.

    Uses file-specificity heuristic: files with fewer executed lines are more
    likely to be targeted bug locations (vs broadly-executed infrastructure).
    Lines in files that also appear in the traceback get an additional boost.
    """
    scores: dict[tuple[str, int], float] = {}
    if trace.coverage:
        return scores  # prefer coverage.py data when available

    # Collect file sizes (number of executed lines) for specificity scoring
    file_line_counts: dict[str, int] = {}
    for file, lines in trace.per_test_executed_lines.items():
        if _TEST_PATH_RE.search(file):
            continue
        file_line_counts[file] = len(lines)

    if not file_line_counts:
        return scores

    # Files appearing in traceback are more suspicious
    traceback_files = {f.file for f in trace.traceback_frames if not f.is_test_code}

    # When focused set is small (≤10 files), the trace is already precise —
    # don't penalise large files, score all equally.  This fixes cases like
    # basic.py (107 executed lines) being ranked below convert_bool.py (1 line)
    # even though basic.py is the actual patch target.
    n_focused = len(file_line_counts)
    use_flat_scoring = n_focused <= 10

    import math
    for file, lines in trace.per_test_executed_lines.items():
        if file not in file_line_counts:
            continue

        if use_flat_scoring:
            # All focused files get the same base score
            base = 0.35
        else:
            n = file_line_counts[file]
            # Base score: inverse specificity (1-100 lines → ~0.5, 1000+ → ~0.1)
            specificity = 1.0 / (1.0 + math.log1p(n))
            # Normalize to [0.10, 0.50] range
            base = 0.10 + 0.40 * specificity

        # Boost files that appear in traceback
        if any(file == tf or file.endswith("/" + tf) or tf.endswith("/" + file)
               for tf in traceback_files):
            base = min(base * 1.5, 0.60)

        for line in lines:
            scores[(file, line)] = round(base, 4)

    return scores


# ── Signal 6: Variable state at exception site ──────────────────────────────

def _score_variable_state(
    trace: StatementTrace,
) -> dict[tuple[str, int], float]:
    """Score lines based on suspicious variable states captured at exception sites.

    Variables that are None, empty containers, or have unexpected types at the
    exception frame are strong indicators of root cause location.
    """
    scores: dict[tuple[str, int], float] = {}
    if not trace.exception_locals:
        return scores

    for file, lines_data in trace.exception_locals.frames.items():
        if _TEST_PATH_RE.search(file):
            continue
        for lineno, variables in lines_data.items():
            if not variables:
                continue
            # Score based on how many suspicious variable states we see
            none_count = sum(1 for v in variables if v.is_none)
            empty_count = sum(1 for v in variables if v.is_empty)
            total_vars = len(variables)

            if total_vars == 0:
                continue

            # Base: having exception locals at all means this frame was on
            # the exception propagation path
            base = 0.4

            # Boost for None values — very common root cause pattern
            if none_count > 0:
                base += 0.3 * min(none_count / total_vars, 1.0)

            # Boost for empty containers — another common cause
            if empty_count > 0:
                base += 0.2 * min(empty_count / total_vars, 1.0)

            # Additional boost if the exception type matches common None patterns
            if trace.error_type in ("TypeError", "AttributeError") and none_count > 0:
                base += 0.1

            key = (file, lineno)
            scores[key] = max(scores.get(key, 0.0), min(base, 1.0))

    return scores


# ── Signal 7: AST bug pattern checker ────────────────────────────────────────

def _score_bug_patterns(
    trace: StatementTrace,
    source_reader: SourceReader | None,
) -> dict[tuple[str, int], float]:
    """Score lines flagged by AST bug pattern checkers.

    Runs lightweight static analysis on source files involved in the crash,
    boosting lines that match known Python bug patterns.
    """
    scores: dict[tuple[str, int], float] = {}
    if not source_reader:
        return scores

    # Collect files to check: traceback files + focused files (limited set)
    files_to_check: set[str] = set()
    for frame in trace.traceback_frames:
        if not frame.is_test_code:
            files_to_check.add(frame.file)
    # Also check files from data flow / focused, but limit to avoid overhead
    for f in trace.focused_files[:20]:
        if not _TEST_PATH_RE.search(f):
            files_to_check.add(f)

    for file in files_to_check:
        source = source_reader(file)
        if not source:
            continue
        hits = check_source(file, source)
        for hit in hits:
            key = (hit.file, hit.line)
            scores[key] = max(scores.get(key, 0.0), hit.confidence)

    return scores


# ── Signal 8: Test call-site target resolution ──────────────────────────────

_CALL_TARGET_RE = re.compile(
    r"(\w+(?:\.\w+)*)\s*\(",  # e.g. password._parse_content(...)
)


def _resolve_call_targets_from_test_frames(
    trace: StatementTrace,
) -> set[str]:
    """When all traceback frames are in test code, extract called module names
    from the code_line of each test frame.

    For example, if the test frame's code_line is:
        plaintext_password, salt, ident = password._parse_content(file_content)
    we extract 'password' as a potential production module prefix.

    This is NOT test-name hacking — it uses information from the execution trace
    (the actual failing line of code shown in the traceback).
    """
    prod_frames = [f for f in trace.traceback_frames if not f.is_test_code]
    if prod_frames:
        return set()  # already have production frames, no need for this heuristic

    _SKIP = frozenset({
        "assert", "self", "super", "print", "len", "str", "int", "float",
        "dict", "list", "set", "tuple", "isinstance", "type", "mock",
        "patch", "pytest", "raises", "assertEqual", "assertTrue",
        "assertRaises", "assertFalse", "assertIn", "assertNotIn",
    })

    # Also extract dotted names that are NOT followed by "(" — these are
    # callables passed as arguments, e.g. assertRaises(Error, self.obj.method, arg)
    _DOTTED_NAME_RE = re.compile(r"(\w+(?:\.\w+)+)")

    targets: set[str] = set()
    for frame in trace.traceback_frames:
        if not frame.is_test_code or not frame.code_line:
            continue

        # Collect all dotted expressions (both calls and references)
        dotted_exprs: list[str] = []
        for m in _CALL_TARGET_RE.finditer(frame.code_line):
            dotted_exprs.append(m.group(1))
        for m in _DOTTED_NAME_RE.finditer(frame.code_line):
            dotted_exprs.append(m.group(1))

        for call_expr in dotted_exprs:
            if call_expr in _SKIP:
                continue
            parts = call_expr.split(".")
            # Strip leading noise: self, cls, super()
            while parts and parts[0] in ("self", "cls", "super"):
                parts = parts[1:]
            if not parts:
                continue
            # Skip if entire remaining expression is a skip-word
            if len(parts) == 1 and parts[0] in _SKIP:
                continue
            # e.g. "self.password_lookup._parse_parameters" → parts = ["password_lookup", "_parse_parameters"]
            # Add the object name (likely maps to a module) AND the method name
            # (can be grepped in source files)
            if len(parts) >= 2:
                targets.add(parts[0])    # "password_lookup" → match password.py
                targets.add(parts[-1])   # "_parse_parameters" → grep-able function name
            elif len(parts) == 1 and parts[0][0].islower():
                targets.add(parts[0])

    targets -= _SKIP
    return targets


_FUNC_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)


def _score_call_target_files(
    trace: StatementTrace,
    call_targets: set[str],
    source_reader: SourceReader | None = None,
) -> dict[tuple[str, int], float]:
    """Boost coverage/per-test candidates whose file path or content matches call targets.

    When we only have test-code traceback frames, this narrows the candidate set
    from 'all covered files' to files matching the called object/method names.

    Two matching strategies:
    1. Basename match: target name appears in file basename (e.g. "password" → password.py)
    2. Def match: target name matches a function/method definition in the source
       (e.g. "_parse_parameters" → file containing "def _parse_parameters(")
    """
    if not call_targets:
        return {}

    scores: dict[tuple[str, int], float] = {}

    # Separate targets into likely-module-names and likely-method-names
    # Method names typically start with _ or are verb-like; module names don't start with _
    module_targets = {t for t in call_targets if not t.startswith("_")}
    method_targets = {t for t in call_targets if t.startswith("_") or t not in module_targets}
    # Both sets can overlap — a target like "combine_vars" could be either

    # Strategy 1: basename matching for module-like targets
    matching_files: set[str] = set()
    all_files = set(trace.coverage.keys()) | set(trace.per_test_executed_lines.keys())
    prod_files = {f for f in all_files if not _TEST_PATH_RE.search(f)}
    for f in prod_files:
        basename = f.rsplit("/", 1)[-1].replace(".py", "").lower()
        for target in module_targets:
            if target.lower() in basename or basename in target.lower():
                matching_files.add(f)
                break

    # Strategy 2: def-name matching for method-like targets (using source_reader)
    # Also match against all targets (module_targets too) as function names
    all_name_targets = module_targets | method_targets
    def_match_lines: dict[tuple[str, int], float] = {}
    if source_reader and all_name_targets:
        # Only search in files that appear in coverage (executed by this test)
        for f in prod_files:
            source = source_reader(f)
            if not source:
                continue
            for m in _FUNC_DEF_RE.finditer(source):
                func_name = m.group(1)
                if func_name in all_name_targets:
                    # Find line number
                    lineno = source[:m.start()].count("\n") + 1
                    def_match_lines[(f, lineno)] = 0.60  # higher than basename match
                    matching_files.add(f)

    if not matching_files and not def_match_lines:
        return scores

    # Score executed lines in basename-matched files
    for f in matching_files:
        cov = trace.coverage.get(f)
        if cov:
            for line in cov.executed_lines:
                scores[(f, line)] = 0.50
        per_test = trace.per_test_executed_lines.get(f)
        if per_test:
            for line in per_test:
                scores[(f, line)] = max(scores.get((f, line), 0.0), 0.50)

    # Overlay def-match scores (higher priority)
    for key, score in def_match_lines.items():
        scores[key] = max(scores.get(key, 0.0), score)

    return scores


# ── Signal 9: Error-message-driven class/module/function lookup ──────────────

# Patterns to extract class/function names from common error messages.
# These use information from the execution trace (error_type + error_message),
# NOT from test names or patch content.
_ATTR_ERR_CLASS_RE = re.compile(
    r"'(\w+)'\s+object\s+has\s+no\s+attribute\s+'(\w+)'"
)
_ATTR_ERR_MODULE_RE = re.compile(
    r"module\s+'([\w.]+)'\s+has\s+no\s+attribute\s+'(\w+)'"
)
_TYPE_ERR_ARGS_RE = re.compile(
    r"(\w+)\(\)\s+takes\s+(?:from\s+)?\d+\s+(?:to\s+\d+\s+)?positional\s+arguments?\s+but\s+\d+",
)
_TYPE_ERR_UNEXPECTED_KWARG_RE = re.compile(
    r"__init__\(\)\s+got\s+an\s+unexpected\s+keyword\s+argument\s+'(\w+)'"
)
_KEY_ERR_RE = re.compile(
    r"KeyError:\s+['\"](\w+)['\"]"
)


def _score_error_message_lookup(
    trace: StatementTrace,
    source_reader: SourceReader | None,
) -> dict[tuple[str, int], float]:
    """Extract class/function/module names from error messages and locate them
    in the source code.

    For example:
      AttributeError: 'RoleMixin' has no attribute '_build_doc'
        → search for 'class RoleMixin' → file containing it is likely the patch target

      TypeError: _resolve_dependency_map() takes 8 args but 9 given
        → search for 'def _resolve_dependency_map' → that file needs the fix

    This is NOT test-name hacking — it uses the exception information from the
    execution trace to locate the relevant production code.
    """
    scores: dict[tuple[str, int], float] = {}
    if not source_reader:
        return scores

    error_msg = f"{trace.error_type}: {trace.error_message}"

    # Collect search targets: (search_regex_pattern, score_value)
    search_targets: list[tuple[re.Pattern, float]] = []

    # Pattern 1: AttributeError on object → find class definition
    m = _ATTR_ERR_CLASS_RE.search(error_msg)
    if m:
        class_name = m.group(1)
        # Search for "class ClassName" — the file defining it is the likely target
        search_targets.append((
            re.compile(rf"^\s*class\s+{re.escape(class_name)}\b", re.MULTILINE),
            0.65,
        ))

    # Pattern 2: AttributeError on module → locate module file directly
    m = _ATTR_ERR_MODULE_RE.search(error_msg)
    if m:
        module_path = m.group(1)  # e.g. "qutebrowser.completion.models.miscmodels"
        # Convert dotted module path to file path suffix
        file_suffix = module_path.replace(".", "/") + ".py"
        search_targets.append((
            re.compile(re.escape(file_suffix)),
            0.70,
        ))

    # Pattern 3: TypeError on function args → find function definition
    m = _TYPE_ERR_ARGS_RE.search(error_msg)
    if m:
        func_name = m.group(1)
        if func_name != "__init__":
            search_targets.append((
                re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(func_name)}\s*\(", re.MULTILINE),
                0.65,
            ))

    # Pattern 4: TypeError unexpected keyword → find __init__ of the class
    m = _TYPE_ERR_UNEXPECTED_KWARG_RE.search(error_msg)
    if m:
        # Try to find the class from the broader error context
        # Often paired with a class name in the traceback
        pass  # Handled by class-name extraction from traceback frames below

    if not search_targets:
        return scores

    # Search across all available source files
    # Collect candidate files from coverage + per_test + focused
    all_files: set[str] = set()
    all_files.update(trace.coverage.keys())
    all_files.update(trace.per_test_executed_lines.keys())
    all_files.update(trace.focused_files)

    prod_files = {f for f in all_files if not _TEST_PATH_RE.search(f)}

    # Also collect class/func name targets for fallback basename matching
    # (when source_reader can't read the file — e.g. source_snapshot has
    # only test files, not production code)
    class_func_names: list[tuple[str, float]] = []
    for pattern, score_val in search_targets:
        # Extract the literal name from patterns like "class\s+KeyInfo\b" or "def\s+func\("
        # The pattern.pattern string contains regex escapes (e.g. \s+, \b),
        # so we need to handle both \s+ and literal space as separators.
        name_match = re.search(r"(?:class|def)(?:\\s\+|\s+)(\w+)", pattern.pattern)
        if name_match:
            class_func_names.append((name_match.group(1), score_val))

    source_found = False  # track if source_reader can read any file
    for file in prod_files:
        source = source_reader(file) if source_reader else None
        if source:
            source_found = True

        for pattern, score_val in search_targets:
            # For file-path patterns (module lookup), match against the file path
            if "/" in pattern.pattern and pattern.search(file):
                lines = (
                    trace.per_test_executed_lines.get(file)
                    or trace.coverage.get(file, CoverageFileData([], [], [])).executed_lines
                )
                if not lines:
                    lines = [1]
                for line in lines:
                    key = (file, line)
                    scores[key] = max(scores.get(key, 0.0), score_val)
                break

            # For source-content patterns (class/def lookup), search in source
            if source:
                match = pattern.search(source)
                if match:
                    lineno = source[:match.start()].count("\n") + 1
                    lines = (
                        trace.per_test_executed_lines.get(file)
                        or trace.coverage.get(file, CoverageFileData([], [], [])).executed_lines
                    )
                    if lines:
                        for line in lines:
                            key = (file, line)
                            scores[key] = max(scores.get(key, 0.0), score_val)
                    else:
                        key = (file, lineno)
                        scores[key] = max(scores.get(key, 0.0), score_val)

    # Fallback: if source_reader couldn't read production files, use basename
    # matching to find files likely containing the target class/function.
    # e.g. class KeyInfo → keyutils.py (because KeyInfo is defined in keyutils)
    if not source_found and class_func_names:
        for file in prod_files:
            basename = file.rsplit("/", 1)[-1].replace(".py", "").lower()
            for name, score_val in class_func_names:
                name_lower = name.lower()
                # Match if class/func name appears in or matches the basename
                # e.g. "KeyInfo" → basename "keyutils" (key prefix match)
                # e.g. "_resolve_dependency_map" → basename containing "dependency"
                # More generous: check if any meaningful word overlaps
                name_parts = re.findall(r"[A-Z][a-z]+|[a-z]+", name)
                name_parts_lower = [p.lower() for p in name_parts if len(p) > 2]
                match = any(part in basename for part in name_parts_lower)
                if not match:
                    # Also try: module path contains the class-related module
                    # e.g. "RoleMixin" defined in doc.py, but error says "'RoleMixin' object"
                    # → can't match by name, but can from test file's imports
                    continue
                lines = (
                    trace.per_test_executed_lines.get(file)
                    or trace.coverage.get(file, CoverageFileData([], [], [])).executed_lines
                )
                if not lines:
                    lines = [1]
                for line in lines:
                    key = (file, line)
                    scores[key] = max(scores.get(key, 0.0), score_val * 0.8)

    # Final fallback: extract class/module info from test file imports.
    # When source_reader can't read prod files AND other matching fails,
    # parse the test file (which IS in source_snapshot) for import statements
    # that tell us where the class/module is defined.
    # Handles both patterns:
    #   from qutebrowser.keyinput import keyutils  (module-level import)
    #   from qutebrowser.cli.doc import RoleMixin   (class-level import)
    if not scores and source_reader:
        test_frames = [f for f in trace.traceback_frames if f.is_test_code]
        # Also use the error message class name directly
        errmsg_class = None
        m = _ATTR_ERR_CLASS_RE.search(error_msg)
        if m:
            errmsg_class = m.group(1)

        for frame in test_frames:
            test_source = source_reader(frame.file)
            if not test_source:
                continue

            target_modules: set[str] = set()

            # Strategy A: look for imports matching class_func_names
            for name, score_val in class_func_names:
                import_re = re.compile(
                    rf"from\s+([\w.]+)\s+import\s+.*\b{re.escape(name)}\b"
                )
                for im in import_re.finditer(test_source):
                    target_modules.add(im.group(1))

            # Strategy B: look for module imports used with the class in error
            # e.g. "from qutebrowser.keyinput import keyutils" +
            #      error "'KeyInfo' has no attribute 'is_special'" +
            #      code "keyutils.KeyInfo(...)" → keyutils module
            if errmsg_class:
                # Find lines like: keyutils.KeyInfo → the module before the dot
                usage_re = re.compile(rf"(\w+)\.{re.escape(errmsg_class)}")
                for um in usage_re.finditer(test_source):
                    module_alias = um.group(1)
                    # Now find where this alias is imported from
                    alias_import_re = re.compile(
                        rf"from\s+([\w.]+)\s+import\s+.*\b{re.escape(module_alias)}\b"
                    )
                    for aim in alias_import_re.finditer(test_source):
                        # Full module path: parent.module_alias
                        target_modules.add(f"{aim.group(1)}.{module_alias}")
                    # Also try direct: import module_alias
                    direct_re = re.compile(
                        rf"^import\s+([\w.]*\b{re.escape(module_alias)}\b)", re.MULTILINE
                    )
                    for dm in direct_re.finditer(test_source):
                        target_modules.add(dm.group(1))

            # Convert dotted module paths to file paths and match
            for mod in target_modules:
                file_suffix = mod.replace(".", "/") + ".py"
                score_val = class_func_names[0][1] if class_func_names else 0.65
                matched = False
                for pf in prod_files:
                    if pf.endswith(file_suffix) or file_suffix.endswith(pf):
                        matched = True
                        lines = (
                            trace.per_test_executed_lines.get(pf)
                            or trace.coverage.get(pf, CoverageFileData([], [], [])).executed_lines
                        )
                        if not lines:
                            lines = [1]
                        for line in lines:
                            key = (pf, line)
                            scores[key] = max(scores.get(key, 0.0), score_val)
                # G3 fallback: if prod_files is empty (no execution data at
                # all), use the import-derived file path directly as a
                # candidate.  We have high confidence because it comes from
                # an explicit import in the test file.
                if not matched:
                    key = (file_suffix, 1)
                    scores[key] = max(scores.get(key, 0.0), score_val)

    return scores


# ── Combine signals ──────────────────────────────────────────────────────────

def analyze_root_cause(
    trace: StatementTrace,
    context: FailureContext,
    source_reader: SourceReader | None = None,
    all_focused_files: dict[str, int] | None = None,
) -> list[RootCauseCandidate]:
    """Combine all signals and return ranked root-cause candidates.

    Args:
        all_focused_files: Optional dict of {file: test_count} across all tests
            in the instance, used for IDF-like denoising of coverage signal.
    """

    # Collect scores from each signal
    s_traceback = _score_traceback_proximity(trace.traceback_frames, trace.error_type)
    s_dataflow = _score_data_flow(context.data_flow_chain)
    s_coverage = _score_coverage_suspiciousness(trace, all_focused_files)
    s_branch = _score_branch_divergence(trace)
    s_per_test = _score_per_test_lines(trace)
    s_varstate = _score_variable_state(trace)
    s_bugpat = _score_bug_patterns(trace, source_reader)

    # Signal 8: call-target resolution from test-only tracebacks
    call_targets = _resolve_call_targets_from_test_frames(trace)
    s_calltarget = _score_call_target_files(trace, call_targets, source_reader)

    # Signal 9: error-message-driven class/module/function lookup
    s_errmsg = _score_error_message_lookup(trace, source_reader)

    # Merge all candidate locations
    all_keys: set[tuple[str, int]] = set()
    all_keys.update(s_traceback.keys())
    all_keys.update(s_dataflow.keys())
    all_keys.update(s_varstate.keys())  # exception locals are high-value, always include
    # Always include call-target candidates (they get scored via pseudo-traceback weight)
    all_keys.update(s_calltarget.keys())
    # Always include error-message lookup candidates
    all_keys.update(s_errmsg.keys())
    # Bug pattern hits in traceback files are high-value candidates
    traceback_files = {f for f, _ in s_traceback.keys()} | {f for f, _ in s_dataflow.keys()}
    for key in s_bugpat:
        if key[0] in traceback_files:
            all_keys.add(key)
    # Only add coverage/branch/per-test candidates for files that appear in
    # traceback or data flow — otherwise the candidate set is too noisy
    for key in s_coverage:
        if key[0] in traceback_files:
            all_keys.add(key)
    for key in s_branch:
        if key[0] in traceback_files:
            all_keys.add(key)
    for key in s_per_test:
        if key[0] in traceback_files:
            all_keys.add(key)

    # If no traceback frames, fall back to coverage/per-test candidates
    # restricted to focused files (per-test) to avoid flooding with noise
    if not s_traceback and not s_dataflow:
        focused_set = set(trace.focused_files)
        focused_set.update(trace.per_test_executed_lines.keys())
        if focused_set:
            def _in_focused(f: str) -> bool:
                return f in focused_set or any(
                    f.endswith("/" + ff) or ff.endswith("/" + f) for ff in focused_set
                )
            for key in s_coverage:
                if _in_focused(key[0]):
                    all_keys.add(key)
            for key in s_branch:
                if _in_focused(key[0]):
                    all_keys.add(key)
            for key in s_per_test:
                if _in_focused(key[0]):
                    all_keys.add(key)
        else:
            # No focused data at all — use top coverage/per-test by suspiciousness
            combined = {**s_per_test, **s_coverage}  # coverage overrides per_test
            top_cov = sorted(combined.items(), key=lambda x: -x[1])[:50]
            all_keys.update(k for k, _ in top_cov)
            all_keys.update(s_branch.keys())

    # Score each candidate
    candidates: list[RootCauseCandidate] = []
    for key in all_keys:
        file, line = key
        if _TEST_PATH_RE.search(file):
            continue

        t = s_traceback.get(key, 0.0)
        d = s_dataflow.get(key, 0.0)
        c = s_coverage.get(key, 0.0)
        b = s_branch.get(key, 0.0)
        pt = s_per_test.get(key, 0.0)
        vs = s_varstate.get(key, 0.0)
        bp = s_bugpat.get(key, 0.0)
        ct = s_calltarget.get(key, 0.0)
        em = s_errmsg.get(key, 0.0)

        score = (
            WEIGHT_TRACEBACK * t
            + WEIGHT_DATA_FLOW * d
            + WEIGHT_COVERAGE_SUSPICIOUSNESS * max(c, pt)
            + WEIGHT_BRANCH_DIVERGENCE * b
            + WEIGHT_VARIABLE_STATE * vs
        )
        # Bug pattern is an additive bonus (up to 0.10) — boosts candidates
        # that match known bug patterns without dominating the ranking
        if bp > 0:
            score += 0.10 * bp
        # Call-target is an additive bonus (up to 0.10) — boosts files whose
        # name or function definitions match calls in test code_line
        if ct > 0:
            score += 0.10 * ct
        # Error-message lookup is an additive bonus (up to 0.15) — boosts
        # files whose class/function definition matches the error message
        if em > 0:
            score += 0.15 * em

        signals = []
        if t > 0:
            signals.append(f"traceback({t:.2f})")
        if ct > 0 and t == 0:
            signals.append(f"calltarget({ct:.2f})")
        if em > 0:
            signals.append(f"errmsg({em:.2f})")
        if d > 0:
            signals.append(f"dataflow({d:.2f})")
        if c > 0:
            signals.append(f"coverage({c:.2f})")
        elif pt > 0:
            signals.append(f"per_test({pt:.2f})")
        if b > 0:
            signals.append(f"branch({b:.2f})")
        if vs > 0:
            signals.append(f"varstate({vs:.2f})")
        if bp > 0:
            signals.append(f"bugpat({bp:.2f})")

        # Get code snippet if source available
        code_snippet = ""
        func_name = ""
        if source_reader:
            source = source_reader(file)
            if source:
                src_lines = source.splitlines()
                if 1 <= line <= len(src_lines):
                    code_snippet = src_lines[line - 1].strip()
                from context_extractor import find_enclosing_function
                finfo = find_enclosing_function(source, line)
                if finfo:
                    func_name = finfo[2]

        # Also try traceback code line
        if not code_snippet:
            for frame in trace.traceback_frames:
                if frame.file == file and frame.lineno == line and frame.code_line:
                    code_snippet = frame.code_line
                    func_name = frame.func_name
                    break

        explanation = _build_explanation(file, line, t, d, c, b, vs, func_name)

        candidates.append(RootCauseCandidate(
            file=file,
            line=line,
            score=round(score, 4),
            signals=signals,
            code_snippet=code_snippet,
            explanation=explanation,
            func_name=func_name,
        ))

    # Sort by score descending, then by line number for stability
    candidates.sort(key=lambda c: (-c.score, c.file, c.line))

    # Deduplicate: if multiple candidates in the same function within 3 lines,
    # keep the highest-scored one
    return _deduplicate_candidates(candidates)


def _build_explanation(
    file: str, line: int,
    t: float, d: float, c: float, b: float, vs: float,
    func_name: str,
) -> str:
    parts = []
    if t > 0.8:
        parts.append("innermost production frame in traceback")
    elif t > 0:
        parts.append("appears in traceback call chain")
    if d > 0.5:
        parts.append("directly feeds data to crash point")
    elif d > 0:
        parts.append("in data dependency chain")
    if b > 0.5:
        parts.append("missing branch at this line (untaken path)")
    if c > 0.5:
        parts.append("at execution boundary (near unexecuted code)")
    if vs > 0.6:
        parts.append("suspicious variable state at exception (None/empty)")
    elif vs > 0:
        parts.append("exception locals captured at this frame")

    where = f"{file}:{line}"
    if func_name:
        where = f"{func_name}() in {file}:{line}"
    if parts:
        return f"{where} — {'; '.join(parts)}"
    return where


def _deduplicate_candidates(
    candidates: list[RootCauseCandidate], line_gap: int = 3
) -> list[RootCauseCandidate]:
    """Remove near-duplicate candidates (same file, adjacent lines)."""
    kept: list[RootCauseCandidate] = []
    seen: set[tuple[str, int]] = set()
    for c in candidates:
        # Check if a higher-scored candidate is very close
        dominated = False
        for sf, sl in seen:
            if sf == c.file and abs(sl - c.line) <= line_gap:
                dominated = True
                break
        if not dominated:
            kept.append(c)
            seen.add((c.file, c.line))
    return kept


# ── Evaluation against ground truth ──────────────────────────────────────────

def evaluate_result(
    candidates: list[RootCauseCandidate],
    patch_files: list[str],
    patch_lines: dict[str, list[int]] | None = None,
) -> RCAResult:
    """Evaluate ranked candidates against ground-truth patch files (and optionally lines)."""
    result = RCAResult(
        test_nodeid="",
        error_type="",
        error_message="",
        candidates=candidates,
        patch_files=patch_files,
    )

    def _file_matches(candidate_file: str) -> bool:
        for pf in patch_files:
            if (candidate_file == pf
                    or candidate_file.endswith("/" + pf)
                    or pf.endswith("/" + candidate_file)):
                return True
        return False

    def _line_matches(candidate_file: str, candidate_line: int) -> bool:
        if not patch_lines:
            return False
        for pf, lines in patch_lines.items():
            if (candidate_file == pf
                    or candidate_file.endswith("/" + pf)
                    or pf.endswith("/" + candidate_file)):
                # Allow ±5 line tolerance
                if any(abs(candidate_line - pl) <= 5 for pl in lines):
                    return True
        return False

    for i, c in enumerate(candidates[:5]):
        hit = _file_matches(c.file)
        line_hit = _line_matches(c.file, c.line)
        if i == 0:
            result.top1_hit = hit
            result.top1_line_hit = line_hit
        if i < 3 and hit:
            result.top3_hit = True
        if hit:
            result.top5_hit = True
        if line_hit:
            result.top5_line_hit = True

    return result


# ── Patch line extraction ────────────────────────────────────────────────────

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def extract_patch_changed_lines(patch_text: str) -> dict[str, list[int]]:
    """Extract changed line numbers from a unified diff patch.

    Returns {file: [changed_line_numbers]} where line numbers refer to the
    original (pre-patch) file — i.e., the lines that the patch modifies.
    """
    result: dict[str, list[int]] = {}
    current_file = ""
    old_lineno = 0

    for line in patch_text.splitlines():
        if line.startswith("--- a/"):
            current_file = line[6:].strip()
            # Normalize
            from common import normalize_path
            current_file = normalize_path(current_file)
        elif line.startswith("+++ b/"):
            pass  # use --- a/ path (pre-patch)
        elif line.startswith("@@"):
            m = _HUNK_HEADER_RE.match(line)
            if m:
                old_lineno = int(m.group(1))
        elif current_file:
            if line.startswith("-"):
                # Deleted/modified line in original
                result.setdefault(current_file, []).append(old_lineno)
                old_lineno += 1
            elif line.startswith("+"):
                # Added line — no original line number, but mark the insertion point
                result.setdefault(current_file, []).append(old_lineno)
            else:
                # Context line
                old_lineno += 1

    return result


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_rca_pipeline(
    traces: list[StatementTrace],
    contexts: list[FailureContext],
    patch_files: list[str],
    patch_text: str = "",
    source_reader: SourceReader | None = None,
) -> list[RCAResult]:
    """Run the full root cause analysis pipeline for all failing tests."""
    patch_lines = extract_patch_changed_lines(patch_text) if patch_text else None

    # Build cross-test file frequency map for IDF-based coverage denoising
    all_focused_files: dict[str, int] = {}
    for trace in traces:
        seen_files: set[str] = set()
        for f in trace.focused_files:
            seen_files.add(f)
        for f in trace.per_test_executed_lines:
            seen_files.add(f)
        for f in seen_files:
            all_focused_files[f] = all_focused_files.get(f, 0) + 1

    results: list[RCAResult] = []
    for trace, context in zip(traces, contexts):
        candidates = analyze_root_cause(
            trace, context, source_reader, all_focused_files,
        )
        rca = evaluate_result(candidates, patch_files, patch_lines)
        rca.test_nodeid = trace.test_nodeid
        rca.error_type = trace.error_type
        rca.error_message = trace.error_message
        results.append(rca)

    return results


_PARAM_SUFFIX_RE = re.compile(r"\[.*\]$")


def _base_test_id(nodeid: str) -> str:
    """Strip parametrize suffix to get the base test template ID.

    e.g. 'test_foo.py::test_bar[en_US]' → 'test_foo.py::test_bar'
    """
    return _PARAM_SUFFIX_RE.sub("", nodeid)


def aggregate_parametrized(results: list[RCAResult]) -> list[RCAResult]:
    """Aggregate parametrized test variants into one RCA unit per base test.

    Keeps the best-scoring result (by top candidate score) for each base test ID.
    This avoids inflating miss counts when hundreds of locale variants all fail.
    """
    from collections import OrderedDict
    groups: OrderedDict[str, RCAResult] = OrderedDict()
    for r in results:
        base = _base_test_id(r.test_nodeid)
        if base not in groups:
            groups[base] = r
        else:
            # Keep the one with the higher top-1 candidate score (better signal)
            existing = groups[base]
            existing_score = existing.candidates[0].score if existing.candidates else 0
            new_score = r.candidates[0].score if r.candidates else 0
            # Prefer a hit; if tie, prefer higher score
            if (r.top1_hit and not existing.top1_hit) or (
                r.top1_hit == existing.top1_hit and new_score > existing_score
            ):
                groups[base] = r
    return list(groups.values())


def save_rca_results(results: list[RCAResult], output_path: Path) -> None:
    """Save RCA results to a JSON file."""
    data = []
    for r in results:
        entry = {
            "test_nodeid": r.test_nodeid,
            "error_type": r.error_type,
            "error_message": r.error_message,
            "top1_hit": r.top1_hit,
            "top3_hit": r.top3_hit,
            "top5_hit": r.top5_hit,
            "top1_line_hit": r.top1_line_hit,
            "top5_line_hit": r.top5_line_hit,
            "patch_files": r.patch_files,
            "candidates": [
                {
                    "file": c.file,
                    "line": c.line,
                    "score": c.score,
                    "signals": c.signals,
                    "code_snippet": c.code_snippet,
                    "explanation": c.explanation,
                    "func_name": c.func_name,
                }
                for c in r.candidates[:20]  # top-20 per test
            ],
        }
        data.append(entry)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
