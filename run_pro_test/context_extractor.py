"""Step 3: Failure context extraction — AST slicing and minimal context building.

Given a StatementTrace and access to source code, extracts the minimal code
context needed to understand the root cause of a test failure.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from statement_tracer import StatementTrace, TracebackFrame


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CodeSnippet:
    file: str
    start_line: int
    end_line: int
    code: str
    relevance: str  # "crash_site", "caller", "data_source", "branch_point"


@dataclass
class DataFlowStep:
    file: str
    line: int
    variable: str
    role: str  # "defined", "modified", "used_at_crash"
    code_line: str = ""


@dataclass
class FailureContext:
    test_nodeid: str
    error_type: str
    error_message: str
    # Crash site and caller code snippets
    relevant_snippets: list[CodeSnippet] = field(default_factory=list)
    # Reverse data-flow chain from crash point
    data_flow_chain: list[DataFlowStep] = field(default_factory=list)
    # Production-code frames only (test frames filtered out)
    production_frames: list[TracebackFrame] = field(default_factory=list)


# ── Source reader protocol ───────────────────────────────────────────────────

SourceReader = Callable[[str], str | None]
"""Given a relative file path, return its source code or None."""


def make_repo_source_reader(repo_root: Path) -> SourceReader:
    """Create a SourceReader that reads from a local repo checkout."""
    def _read(rel_path: str) -> str | None:
        full = repo_root / rel_path
        if full.exists() and full.is_file():
            try:
                return full.read_text(errors="replace")
            except Exception:
                return None
        return None
    return _read


# ── AST utilities ────────────────────────────────────────────────────────────

class _NameCollector(ast.NodeVisitor):
    """Collect all Name nodes (variable references) from an AST subtree."""

    def __init__(self):
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name):
        self.names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Collect the attribute chain as "obj.attr"
        parts = []
        n = node
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
            self.names.add(n.id)
            # Also add full dotted name for matching
            self.names.add(".".join(reversed(parts)))
        self.generic_visit(node)


def extract_used_names(node: ast.AST) -> set[str]:
    """Extract all variable/attribute names used (read) in an AST node."""
    collector = _NameCollector()
    collector.visit(node)
    return collector.names


def extract_assigned_names(node: ast.AST) -> set[str]:
    """Extract variable names assigned (written) in a statement."""
    names: set[str] = set()
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if node.target else []
        )
        for target in targets:
            names |= _target_names(target)
    elif isinstance(node, ast.AugAssign):
        names |= _target_names(node.target)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        names |= _target_names(node.target)
    elif isinstance(node, ast.With):
        for item in node.items:
            if item.optional_vars:
                names |= _target_names(item.optional_vars)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        names.add(node.name)
    elif isinstance(node, ast.ClassDef):
        names.add(node.name)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            names.add(alias.asname or alias.name)
    return names


def _target_names(target: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names |= _target_names(elt)
    elif isinstance(target, ast.Starred):
        names |= _target_names(target.value)
    return names


def find_enclosing_function(
    source: str, lineno: int
) -> tuple[int, int, str, str] | None:
    """Find the function/method containing the given line number.

    Returns (start_line, end_line, func_name, func_source) or None.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines(keepends=True)
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno or node.lineno
            if start <= lineno <= end:
                if best is None or node.lineno > best.lineno:
                    best = node

    if best is None:
        return None

    start = best.lineno
    end = best.end_lineno or best.lineno
    func_source = "".join(lines[start - 1:end])
    return start, end, best.name, func_source


def find_statement_at_line(source: str, lineno: int) -> ast.AST | None:
    """Find the AST statement node at or containing the given line number."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    # Walk all statement nodes, find the innermost one containing lineno
    best: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and hasattr(node, "lineno"):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if node.lineno <= lineno <= end:
                if best is None or node.lineno >= getattr(best, "lineno", 0):
                    best = node
    return best


def get_statements_in_function(source: str, func_start: int, func_end: int) -> list[ast.stmt]:
    """Get all top-level statements within a function body, in source order."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno == func_start:
                return list(node.body)
    return []


# ── Reverse data dependency slicing ──────────────────────────────────────────

def reverse_data_slice(source: str, crash_line: int) -> list[DataFlowStep]:
    """From the crash line, trace data dependencies backward within the same function.

    Returns a list of DataFlowStep representing the chain of variable assignments
    that feed into the crash line. Inspired by Arash's _reverse_slice_indices()
    but uses Python's ast module for precision.
    """
    func_info = find_enclosing_function(source, crash_line)
    if func_info is None:
        return []
    func_start, func_end, func_name, _ = func_info

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()

    # Find the function node and its body
    func_stmts = get_statements_in_function(source, func_start, func_end)
    if not func_stmts:
        return []

    # Flatten nested statements (if/for/with bodies) into linear order
    all_stmts = _flatten_stmts(func_stmts)

    # Find crash statement
    crash_stmt = None
    crash_idx = -1
    for i, stmt in enumerate(all_stmts):
        end = getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno
        if stmt.lineno <= crash_line <= end:
            crash_stmt = stmt
            crash_idx = i
            break

    if crash_stmt is None:
        return []

    # Collect variables used at crash site
    work = extract_used_names(crash_stmt)
    chain: list[DataFlowStep] = [
        DataFlowStep(
            file="",  # caller fills this
            line=crash_line,
            variable=", ".join(sorted(work)),
            role="used_at_crash",
            code_line=lines[crash_line - 1].strip() if crash_line <= len(lines) else "",
        )
    ]

    # Walk backward from crash
    visited_lines: set[int] = set()
    for stmt in reversed(all_stmts[:crash_idx]):
        assigned = extract_assigned_names(stmt)
        overlap = assigned & work
        if not overlap:
            continue
        if stmt.lineno in visited_lines:
            continue
        visited_lines.add(stmt.lineno)

        # Add new dependencies from the right-hand side
        rhs_names = extract_used_names(stmt) - assigned
        work = (work - overlap) | rhs_names

        code = lines[stmt.lineno - 1].strip() if stmt.lineno <= len(lines) else ""
        chain.append(DataFlowStep(
            file="",
            line=stmt.lineno,
            variable=", ".join(sorted(overlap)),
            role="defined" if _is_definition(stmt) else "modified",
            code_line=code,
        ))

        if not work:
            break

    # Also check function parameters (they define variables)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno != func_start:
                continue
            param_names = set()
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                param_names.add(arg.arg)
            if node.args.vararg:
                param_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                param_names.add(node.args.kwarg.arg)
            overlap = param_names & work
            if overlap:
                chain.append(DataFlowStep(
                    file="",
                    line=node.lineno,
                    variable=", ".join(sorted(overlap)),
                    role="parameter",
                    code_line=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                ))
            break

    return chain


def _flatten_stmts(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Flatten nested statement bodies into a linear list preserving source order."""
    result: list[ast.stmt] = []
    for stmt in stmts:
        result.append(stmt)
        for attr in ("body", "orelse", "handlers", "finalbody"):
            children = getattr(stmt, attr, None)
            if isinstance(children, list):
                result.extend(_flatten_stmts(children))
    result.sort(key=lambda s: (s.lineno, getattr(s, "col_offset", 0)))
    return result


def _is_definition(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Import, ast.ImportFrom))


# ── Context building ─────────────────────────────────────────────────────────

def build_failure_context(
    trace: StatementTrace,
    source_reader: SourceReader | None = None,
) -> FailureContext:
    """Build a FailureContext from a StatementTrace.

    If source_reader is None, only traceback-based context is produced
    (no AST slicing or code snippets).
    """
    # Separate production frames from test frames
    production_frames = [f for f in trace.traceback_frames if not f.is_test_code]

    ctx = FailureContext(
        test_nodeid=trace.test_nodeid,
        error_type=trace.error_type,
        error_message=trace.error_message,
        production_frames=production_frames,
    )

    if source_reader is None:
        return ctx

    # Extract code snippets for each production frame (crash sites + callers)
    seen_files: set[tuple[str, int]] = set()
    for i, frame in enumerate(reversed(production_frames)):
        # innermost first
        key = (frame.file, frame.lineno)
        if key in seen_files:
            continue
        seen_files.add(key)

        source = source_reader(frame.file)
        if source is None:
            continue

        func_info = find_enclosing_function(source, frame.lineno)
        if func_info:
            start, end, func_name, func_source = func_info
            relevance = "crash_site" if i == 0 else "caller"
            ctx.relevant_snippets.append(CodeSnippet(
                file=frame.file,
                start_line=start,
                end_line=end,
                code=func_source,
                relevance=relevance,
            ))

        # Do reverse data slice for the innermost production frame
        if i == 0:
            chain = reverse_data_slice(source, frame.lineno)
            for step in chain:
                step.file = frame.file
            ctx.data_flow_chain = chain

    return ctx


def build_failure_contexts(
    traces: list[StatementTrace],
    source_reader: SourceReader | None = None,
) -> list[FailureContext]:
    """Build FailureContext for each StatementTrace."""
    return [build_failure_context(t, source_reader) for t in traces]


# ── Utility: extract context without full source ─────────────────────────────

def extract_context_from_coverage(
    trace: StatementTrace,
    patch_files: list[str],
) -> list[CodeSnippet]:
    """Extract approximate context using coverage data alone (no source code).

    Identifies which coverage-reported lines in patch-related files were executed,
    and creates pseudo-snippets marking line ranges.
    """
    snippets: list[CodeSnippet] = []
    for pf in patch_files:
        # Find coverage data for this patch file
        cov = trace.per_test_executed_lines.get(pf)
        if not cov:
            # Try suffix match
            for f, lines in trace.per_test_executed_lines.items():
                if f.endswith("/" + pf) or pf.endswith("/" + f):
                    cov = lines
                    break
        if not cov:
            continue

        # Group consecutive lines into ranges
        ranges = _group_line_ranges(cov)
        for start, end in ranges:
            snippets.append(CodeSnippet(
                file=pf,
                start_line=start,
                end_line=end,
                code=f"[executed lines {start}-{end}]",
                relevance="executed_in_patch_file",
            ))
    return snippets


def _group_line_ranges(lines: list[int], gap: int = 3) -> list[tuple[int, int]]:
    """Group line numbers into contiguous ranges (allowing small gaps)."""
    if not lines:
        return []
    sorted_lines = sorted(set(lines))
    ranges: list[tuple[int, int]] = []
    start = end = sorted_lines[0]
    for line in sorted_lines[1:]:
        if line <= end + gap:
            end = line
        else:
            ranges.append((start, end))
            start = end = line
    ranges.append((start, end))
    return ranges
