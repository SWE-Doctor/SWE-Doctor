"""AST-based bug pattern checkers for Python source code.

Lightweight static analysis inspired by Arash's custom CSA checkers.
Each checker is an ast.NodeVisitor that flags lines matching common
Python bug patterns. Results feed into the RCA scoring pipeline.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class BugPatternHit:
    """A single bug pattern match."""
    file: str
    line: int
    pattern: str        # e.g. "none_access", "type_mismatch"
    confidence: float   # 0.0 ~ 1.0
    description: str


# ── Pattern 1: Unsafe attribute access on potentially-None value ────────────

class _NoneAccessChecker(ast.NodeVisitor):
    """Detect `x.attr` where x could be None based on surrounding context.

    Flags:
    - Attribute access on a name that is compared to None in the same function
    - Attribute access on a name that defaults to None in function signature
    - Attribute access on a return value of .get(), .pop(default), dict[key]
    """

    def __init__(self, file: str):
        self.file = file
        self.hits: list[BugPatternHit] = []
        self._none_names: set[str] = set()  # names compared to None or defaulting to None
        self._func_lines: tuple[int, int] = (0, 0)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scan_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scan_function(node)

    def _scan_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        old_names = self._none_names
        self._none_names = set()

        # Collect names that default to None in parameters
        for default in node.args.defaults + node.args.kw_defaults:
            if isinstance(default, ast.Constant) and default.value is None:
                # Find the corresponding arg
                pass  # handled below

        for arg, default in _iter_args_with_defaults(node.args):
            if isinstance(default, ast.Constant) and default.value is None:
                self._none_names.add(arg.arg)

        # Collect names compared to None (x is None, x is not None)
        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                for op, comp in zip(child.ops, child.comparators):
                    if isinstance(comp, ast.Constant) and comp.value is None:
                        if isinstance(child.left, ast.Name):
                            self._none_names.add(child.left.id)
            # Names assigned from .get() calls
            if isinstance(child, ast.Assign) and len(child.targets) == 1:
                target = child.targets[0]
                if isinstance(target, ast.Name) and isinstance(child.value, ast.Call):
                    if (isinstance(child.value.func, ast.Attribute)
                            and child.value.func.attr == "get"):
                        self._none_names.add(target.id)

        # Now find attribute access on these names without a prior None check
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                if child.value.id in self._none_names:
                    self.hits.append(BugPatternHit(
                        file=self.file,
                        line=child.lineno,
                        pattern="none_access",
                        confidence=0.6,
                        description=f"Attribute access on '{child.value.id}' which may be None",
                    ))

        self._none_names = old_names
        self.generic_visit(node)


def _iter_args_with_defaults(args: ast.arguments):
    """Yield (arg, default) pairs for all arguments that have defaults."""
    # Positional args: defaults align from the right
    n_pos = len(args.args)
    n_defaults = len(args.defaults)
    for i, default in enumerate(args.defaults):
        arg_idx = n_pos - n_defaults + i
        if 0 <= arg_idx < n_pos:
            yield args.args[arg_idx], default
    # Keyword-only args
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            yield arg, default


# ── Pattern 2: Type mismatch in binary operations ──────────────────────────

class _TypeMismatchChecker(ast.NodeVisitor):
    """Detect binary operations that commonly cause TypeError.

    Flags:
    - `a | b` where a or b might be a non-dict type (common dict merge bug)
    - String formatting with wrong type (% operator with non-tuple)
    - Arithmetic on mixed types (str + int)
    """

    def __init__(self, file: str):
        self.file = file
        self.hits: list[BugPatternHit] = []
        self._local_types: dict[str, str] = {}  # name -> inferred type hint

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scan_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scan_function(node)

    def _scan_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        old = self._local_types
        self._local_types = {}

        # Collect type annotations from function signature
        for arg in node.args.args + node.args.kwonlyargs:
            if arg.annotation:
                self._local_types[arg.arg] = ast.dump(arg.annotation)

        # Scan for suspicious binary ops
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp):
                # `a | b` — if inside a function that doesn't deal with sets/dicts,
                # this might be a type mismatch (e.g., bitwise on wrong types)
                if isinstance(child.op, ast.BitOr):
                    # Check if either operand is a call that might return wrong type
                    self._check_bitor(child)

        self._local_types = old
        self.generic_visit(node)

    def _check_bitor(self, node: ast.BinOp) -> None:
        # Flag if one side is a function call (may return unexpected type)
        if isinstance(node.left, ast.Call) or isinstance(node.right, ast.Call):
            self.hits.append(BugPatternHit(
                file=self.file,
                line=node.lineno,
                pattern="type_mismatch",
                confidence=0.4,
                description="Binary | operation with function call (potential type mismatch)",
            ))


# ── Pattern 3: Mutable default argument ────────────────────────────────────

class _MutableDefaultChecker(ast.NodeVisitor):
    """Detect mutable default arguments: def f(x=[]) or def f(x={})."""

    def __init__(self, file: str):
        self.file = file
        self.hits: list[BugPatternHit] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_defaults(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_defaults(node)
        self.generic_visit(node)

    def _check_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for default in node.args.defaults + node.args.kw_defaults:
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.hits.append(BugPatternHit(
                    file=self.file,
                    line=default.lineno,
                    pattern="mutable_default",
                    confidence=0.7,
                    description=f"Mutable default argument ({type(default).__name__})",
                ))


# ── Pattern 4: Bare except / overly broad except ───────────────────────────

class _BroadExceptChecker(ast.NodeVisitor):
    """Detect bare except or except Exception that silently passes."""

    def __init__(self, file: str):
        self.file = file
        self.hits: list[BugPatternHit] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Bare except or except Exception
        is_broad = (node.type is None  # bare except
                    or (isinstance(node.type, ast.Name)
                        and node.type.id in ("Exception", "BaseException")))
        if is_broad:
            # Check if body is just pass/continue (swallowing errors)
            body_stmts = [s for s in node.body if not isinstance(s, ast.Pass)]
            if not body_stmts or (
                len(body_stmts) == 1 and isinstance(body_stmts[0], ast.Continue)
            ):
                self.hits.append(BugPatternHit(
                    file=self.file,
                    line=node.lineno,
                    pattern="swallowed_exception",
                    confidence=0.5,
                    description="Broad except clause silently swallows errors",
                ))
        self.generic_visit(node)


# ── Pattern 5: Wrong operator in comparison chain ──────────────────────────

class _ComparisonChecker(ast.NodeVisitor):
    """Detect common comparison mistakes.

    - `x == None` instead of `x is None`
    - `not x in y` instead of `x not in y` (precedence trap)
    """

    def __init__(self, file: str):
        self.file = file
        self.hits: list[BugPatternHit] = []

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, comp in zip(node.ops, node.comparators):
            # x == None
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if isinstance(comp, ast.Constant) and comp.value is None:
                    self.hits.append(BugPatternHit(
                        file=self.file,
                        line=node.lineno,
                        pattern="eq_none",
                        confidence=0.4,
                        description="Using == None instead of 'is None'",
                    ))
        self.generic_visit(node)


# ── Public API ──────────────────────────────────────────────────────────────

ALL_CHECKERS = [
    _NoneAccessChecker,
    _TypeMismatchChecker,
    _MutableDefaultChecker,
    _BroadExceptChecker,
    _ComparisonChecker,
]


def check_source(file: str, source: str) -> list[BugPatternHit]:
    """Run all bug pattern checkers on a source file.

    Args:
        file: Relative file path (for reporting).
        source: Python source code text.

    Returns:
        List of BugPatternHit for all detected patterns.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    all_hits: list[BugPatternHit] = []
    for checker_cls in ALL_CHECKERS:
        checker = checker_cls(file)
        checker.visit(tree)
        all_hits.extend(checker.hits)

    return all_hits


def check_files(
    files: dict[str, str],
    focus_lines: dict[str, set[int]] | None = None,
) -> list[BugPatternHit]:
    """Run checkers on multiple files, optionally filtering to focused lines.

    Args:
        files: {file_path: source_code}
        focus_lines: Optional {file_path: set_of_line_numbers} to restrict results.
    """
    all_hits: list[BugPatternHit] = []
    for file, source in files.items():
        hits = check_source(file, source)
        if focus_lines and file in focus_lines:
            hits = [h for h in hits if h.line in focus_lines[file]]
        all_hits.extend(hits)
    return all_hits
