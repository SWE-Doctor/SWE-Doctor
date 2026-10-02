"""AST backward dataflow slice — role-label each traceback frame."""
from __future__ import annotations

import ast
from typing import Callable

from .traceback_parser import TracebackFrame

_ROLE_PRIORITY = {"sink": 4, "source": 3, "transform": 2, "pass_through": 1, "unknown": 0}


def _enclosing_func(tree: ast.AST, lineno: int) -> ast.AST | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= lineno <= end:
                if best is None or node.lineno > best.lineno:
                    best = node
    return best


def _names_in(expr: ast.AST | None) -> set[str]:
    if expr is None:
        return set()
    return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}


def _stmt_targets(stmt: ast.stmt) -> set[str]:
    out: set[str] = set()
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            out |= _names_in(t)
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)) and getattr(stmt, "target", None):
        out |= _names_in(stmt.target)
    return out


def _is_pure_alias(stmt: ast.stmt) -> bool:
    return (isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name))


def _is_sink_stmt(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Assert):
        return True
    if isinstance(stmt, ast.Return):
        return True
    if isinstance(stmt, ast.Raise):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Compare):
        return True
    return False


def _better(current: str, new: str) -> str:
    return new if _ROLE_PRIORITY[new] > _ROLE_PRIORITY[current] else current


def _classify(func: ast.AST, frame_line: int, seed_params: set[str],
              has_caller: bool) -> tuple[str, str, set[str]]:
    """Return (role, reason, tainted_params_at_frame_line).

    seed_params: params we know the caller tainted. If has_caller is False
    (first frame from test), we treat the innermost statement as a sink.
    """
    # For the innermost frame (no caller in our chain), the frame's statement
    # is the failure point itself → sink, regardless of flow.
    stmts = [s for s in func.body]

    def _all_stmts(body):
        for s in body:
            yield s
            for child in ast.iter_child_nodes(s):
                if isinstance(child, ast.stmt):
                    yield from _all_stmts([child])

    ordered = sorted(
        (s for s in ast.walk(func) if isinstance(s, ast.stmt)
         and getattr(s, "lineno", 0) and s is not func),
        key=lambda s: s.lineno,
    )

    stmt_at_line = next((s for s in reversed(ordered) if s.lineno == frame_line), None)

    if not has_caller and stmt_at_line is not None and _is_sink_stmt(stmt_at_line):
        return "sink", "", set()

    # Walk statements in order up through frame_line, tracking tainted names.
    # If caller tainted params, seed with those; otherwise treat all params as tainted.
    func_params = {a.arg for a in func.args.args} | {a.arg for a in func.args.kwonlyargs}
    tainted: set[str] = set(seed_params) if seed_params else set(func_params)

    candidate_role = "unknown"
    # Track whether any param was originally tainted so we can distinguish
    # "source" (no tainted input) from downstream roles.
    has_tainted_input = bool(seed_params) or not has_caller and False  # only params count
    # If this function has no params at all → it's a source-of-truth producer.
    if not func_params:
        has_tainted_input = False
    else:
        has_tainted_input = True  # optimistic; we treat all params as tainted by default

    for s in ordered:
        if s.lineno > frame_line:
            break
        tgts = _stmt_targets(s)
        rhs = getattr(s, "value", None)
        rhs_names = _names_in(rhs)
        if not tgts:
            continue

        if rhs_names & tainted:
            if _is_pure_alias(s):
                candidate_role = _better(candidate_role, "pass_through")
            else:
                candidate_role = _better(candidate_role, "transform")
            tainted |= tgts
        else:
            # Assignment with no tainted RHS → producing a fresh value.
            # If this function has no tainted input (no params), it's a source.
            if not has_tainted_input:
                candidate_role = _better(candidate_role, "source")
                tainted |= tgts

    # Determine forward-propagation set: intersect tainted with params.
    return candidate_role, ("no-tainted-flow-identified" if candidate_role == "unknown" else ""), tainted & func_params


def label_frames(frames: list[TracebackFrame],
                 source_resolver: Callable[[str], str | None]) -> list[dict]:
    labeled: list[dict] = []
    # We iterate frames innermost-last (the order given). The "caller" of
    # frame i is frame i-1 (the earlier entry). Taint propagation: first frame
    # (index 0) has no caller, so it's the sink.
    prev_tainted_names: set[str] = set()

    for i, f in enumerate(frames):
        src = source_resolver(f.file)
        if src is None:
            labeled.append({"file": f.file, "lineno": f.lineno, "qualname": f.qualname,
                            "role": "unknown", "reason": "source-unreadable",
                            "observed_values": {}})
            prev_tainted_names = set()
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            labeled.append({"file": f.file, "lineno": f.lineno, "qualname": f.qualname,
                            "role": "unknown", "reason": "ast-parse-error",
                            "observed_values": {}})
            prev_tainted_names = set()
            continue

        func = _enclosing_func(tree, f.lineno)
        if func is None:
            labeled.append({"file": f.file, "lineno": f.lineno, "qualname": f.qualname,
                            "role": "unknown", "reason": "no-enclosing-function",
                            "observed_values": {}})
            prev_tainted_names = set()
            continue

        func_params = {a.arg for a in func.args.args} | {a.arg for a in func.args.kwonlyargs}
        if i == 0:
            # innermost frame (sink candidate)
            role, reason, carry = _classify(func, f.lineno, set(), has_caller=False)
        else:
            seed = prev_tainted_names & func_params if prev_tainted_names else func_params
            role, reason, carry = _classify(func, f.lineno, seed, has_caller=True)

        labeled.append({"file": f.file, "lineno": f.lineno, "qualname": f.qualname,
                        "role": role, "reason": reason, "observed_values": {}})
        prev_tainted_names = carry

    return labeled
