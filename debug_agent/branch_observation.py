"""Enricher Pass 5 — extract which branch arm the debug agent stepped into.

Pure: given a list of pdb_session_log turns and a source resolver, find stops
near If/While/For nodes, look at the next stop's lineno, and classify the
transition as then / else / loop_entered / loop_skipped."""
from __future__ import annotations

import ast
from typing import Callable

_MAX_LOCALS = 8
_BODY_KINDS = (ast.If,)
_LOOP_KINDS = (ast.For, ast.While)


def _find_control_node(tree: ast.AST, lineno: int):
    """Return the innermost If/For/While whose body or header contains `lineno`.

    Strict equality (`node.lineno == lineno`) almost never matches real
    debug-agent trajectories — agents stop at breakpoints inside bodies,
    not on control-flow headers. Walking the body/orelse spans is much
    closer to what the gate intends ("did we cross a branch?")."""
    best = None
    best_span = None
    for node in ast.walk(tree):
        if not isinstance(node, _BODY_KINDS + _LOOP_KINDS):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None) or start
        if start is None or end is None:
            continue
        if start <= lineno <= end:
            span = end - start
            if best is None or span < best_span:
                best, best_span = node, span
    return best


def _arm_for_if(node: ast.If, next_lineno: int, cur_lineno: int = 0) -> str:
    """Classify which arm of `node` was taken.

    With looser matching, `cur_lineno` is inside the body, not on the header.
    Use cur_lineno to decide which arm we're already in; next_lineno just
    confirms we stayed in the same span (sanity)."""
    def _spans(stmts):
        out = []
        for s in stmts:
            lo = getattr(s, "lineno", None)
            hi = getattr(s, "end_lineno", None) or lo
            if lo is not None and hi is not None:
                out.append((lo, hi))
        return out

    def _in_any(lineno, spans):
        return any(lo <= lineno <= hi for lo, hi in spans)

    then_spans = _spans(node.body)
    else_spans = _spans(node.orelse)
    if cur_lineno and _in_any(cur_lineno, then_spans):
        return "then"
    if cur_lineno and _in_any(cur_lineno, else_spans):
        return "else"
    if _in_any(next_lineno, then_spans):
        return "then"
    if _in_any(next_lineno, else_spans):
        return "else"
    return "unknown"


def _arm_for_loop(node, next_lineno: int) -> str:
    body_lines = {n.lineno for n in node.body if hasattr(n, "lineno")}
    if body_lines and next_lineno in body_lines:
        return "loop_entered"
    return "loop_skipped"


def _cond_text(src: str, node) -> str:
    lines = src.splitlines()
    return lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""


def _collect_locals(turns: list[dict], upto_idx: int) -> dict[str, str]:
    locals_seen: dict[str, str] = {}
    for t in turns[: upto_idx + 1]:
        if t.get("kind") != "cmd":
            continue
        cmd = (t.get("cmd") or "").strip()
        if cmd.startswith(("p ", "pp ")):
            expr = cmd.split(None, 1)[1].strip()
            val = (t.get("last_eval") or "").strip()
            if expr and val:
                locals_seen[expr] = val[:80]
        if len(locals_seen) >= _MAX_LOCALS:
            break
    return dict(list(locals_seen.items())[:_MAX_LOCALS])


def collect_branch_observations(
    pdb_log: list[dict],
    source_resolver: Callable[[str], str | None],
) -> list[dict]:
    """Annotate each unique visited frame with its enclosing control-flow node.

    Two provenance flavors:
      - 'arm_inferred': consecutive frames in same file with different lineno
        let us infer which arm was taken. Strongest signal.
      - 'frame_only': single stop inside an If/For/While span. Arm unknown,
        but condition text + observed locals are still useful for repair_agent.
    """
    out: list[dict] = []
    cmd_turns = [(i, t) for i, t in enumerate(pdb_log) if t.get("kind") == "cmd"]
    src_cache: dict[str, str | None] = {}
    tree_cache: dict[str, ast.AST | None] = {}

    def _src_for(path: str) -> str | None:
        if path in src_cache:
            return src_cache[path]
        s = source_resolver(path)
        src_cache[path] = s
        return s

    def _tree_for(path: str):
        if path in tree_cache:
            return tree_cache[path]
        s = _src_for(path)
        if not s:
            tree_cache[path] = None
            return None
        try:
            tree_cache[path] = ast.parse(s)
        except SyntaxError:
            tree_cache[path] = None
        return tree_cache[path]

    seen_frame: set[tuple[str, int]] = set()

    for idx, (orig_i, turn) in enumerate(cmd_turns[:-1]):
        cur = turn.get("current_frame")
        if not cur:
            continue
        nxt = cmd_turns[idx + 1][1].get("current_frame")
        if not nxt or nxt.get("file") != cur.get("file"):
            continue
        if nxt.get("lineno") == cur.get("lineno"):
            continue
        tree = _tree_for(cur["file"])
        if tree is None:
            continue
        node = _find_control_node(tree, cur["lineno"])
        if node is None:
            continue
        arm = (_arm_for_if(node, nxt["lineno"], cur_lineno=cur["lineno"])
               if isinstance(node, ast.If)
               else _arm_for_loop(node, nxt["lineno"]))
        out.append({
            "file": cur["file"],
            "lineno": cur["lineno"],
            "cond_text": _cond_text(_src_for(cur["file"]) or "", node),
            "arm_taken": arm,
            "locals_at_stop": _collect_locals(pdb_log, orig_i),
            "evidence_refs": [orig_i, cmd_turns[idx + 1][0]],
            "provenance": {"kind": "arm_inferred"},
        })
        seen_frame.add((cur["file"], cur["lineno"]))

    for orig_i, turn in cmd_turns:
        cur = turn.get("current_frame")
        if not cur:
            continue
        key = (cur.get("file"), cur.get("lineno"))
        if not key[0] or not key[1] or key in seen_frame:
            continue
        tree = _tree_for(cur["file"])
        if tree is None:
            continue
        node = _find_control_node(tree, cur["lineno"])
        if node is None:
            continue
        out.append({
            "file": cur["file"],
            "lineno": cur["lineno"],
            "cond_text": _cond_text(_src_for(cur["file"]) or "", node),
            "arm_taken": "unknown",
            "locals_at_stop": _collect_locals(pdb_log, orig_i),
            "evidence_refs": [orig_i],
            "provenance": {"kind": "frame_only"},
        })
        seen_frame.add(key)

    return out
