"""Enricher Pass 6 — tag observed pdb values with anomaly classes."""
from __future__ import annotations

import ast as _ast
import re

_EMPTY = {"[]", "{}", "()", "''", '""', "set()"}
_NEG_ONE = "-1"
_FLOAT_NAN = re.compile(r"^nan$|^float\('nan'\)$", re.IGNORECASE)
_FLOAT_INF = re.compile(r"^inf$|^float\('inf'\)$|^-inf$", re.IGNORECASE)
_INT_RE = re.compile(r"^-?\d+$")
_BYTES_RE = re.compile(r"""^b['"].*['"]$""", re.DOTALL)
_STR_RE = re.compile(r"""^['"].*['"]$""", re.DOTALL)


def _tags_for(value: str) -> list[str]:
    v = value.strip()
    tags: list[str] = []
    if v == "None":
        tags.append("none")
    elif v in _EMPTY:
        tags.append("empty")
    elif v == "0":
        tags.append("zero")
    elif v == _NEG_ONE:
        tags.append("neg_one")
    elif _FLOAT_NAN.match(v):
        tags.append("nan")
    elif _FLOAT_INF.match(v):
        tags.append("inf")
    if _INT_RE.match(v):
        try:
            if abs(int(v)) >= (1 << 31):
                tags.append("large_int")
        except ValueError:
            pass
    return tags


def _split_top_level_commas(s: str, expected: int) -> list[str] | None:
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    in_str: str | None = None
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            cur.append(ch)
            if ch == "\\" and i + 1 < len(s):
                cur.append(s[i + 1]); i += 2; continue
            if ch == in_str:
                in_str = None
        elif ch in ("'", '"'):
            in_str = ch; cur.append(ch)
        elif ch in "([{<":
            depth += 1; cur.append(ch)
        elif ch in ")]}>":
            depth = max(0, depth - 1); cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        out.append("".join(cur).strip())
    return out if len(out) == expected else None


def _slice_tuple_source(s: str, n: int) -> list[str] | None:
    if not (s.startswith("(") and s.endswith(")")):
        return None
    inner = s[1:-1]
    return _split_top_level_commas(inner, n)


def _try_atomize(expr: str, value_repr: str) -> list[tuple[str, str]] | None:
    """If expr is a tuple of bare identifiers and value_repr is a tuple
    literal with matching arity, return [(ident, segment_repr), ...].
    Otherwise return None."""
    try:
        tree = _ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None
    body = tree.body
    if not isinstance(body, _ast.Tuple):
        return None
    if not all(isinstance(e, _ast.Name) for e in body.elts):
        return None
    idents = [e.id for e in body.elts]

    s = value_repr.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    segments = _slice_tuple_source(s, len(idents))
    if segments is None or len(segments) != len(idents):
        return None
    return list(zip(idents, segments))


def collect_anomalies(pdb_log: list[dict]) -> list[dict]:
    out: list[dict] = []
    expr_kinds: dict[str, set[str]] = {}
    for i, t in enumerate(pdb_log):
        if t.get("kind") != "cmd":
            continue
        cmd = (t.get("cmd") or "").strip()
        if not cmd.startswith(("p ", "pp ")):
            continue
        expr = cmd.split(None, 1)[1].strip()
        val = (t.get("last_eval") or "").strip()
        if not expr or not val:
            continue
        frame = t.get("current_frame") or {}

        atomic = _try_atomize(expr, val)
        entries = atomic if atomic is not None else [(expr, val)]

        for ent_expr, ent_val in entries:
            tags = _tags_for(ent_val)
            kind = ("bytes" if _BYTES_RE.match(ent_val)
                    else "str" if _STR_RE.match(ent_val) else "other")
            expr_kinds.setdefault(ent_expr, set()).add(kind)
            if {"str", "bytes"}.issubset(expr_kinds[ent_expr]):
                tags.append("type_str_bytes")
            out.append({
                "expr": ent_expr,
                "file": frame.get("file", ""),
                "lineno": frame.get("lineno", 0),
                "value_repr": ent_val[:200],
                "tags": tags,
                "evidence_refs": [i],
            })
    return out
