"""AST signature-diff + repo caller enumeration for contract-impact pass."""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

_FENCED_PY_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_py_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _FENCED_PY_RE.finditer(text or "")]


def _short_and_cls(qualname: str) -> tuple[str, str]:
    parts = (qualname or "").split(".")
    if len(parts) >= 2:
        return parts[-1], parts[-2]
    return parts[-1] if parts else "", ""


def _find_func_by_short(tree: ast.AST, short: str, cls_hint: str) -> ast.FunctionDef | None:
    # Prefer function inside cls_hint if provided.
    if cls_hint:
        for cls in ast.walk(tree):
            if isinstance(cls, ast.ClassDef) and cls.name == cls_hint:
                for n in ast.walk(cls):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == short:
                        return n
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == short:
            return n
    return None


def _sig_fingerprint(fn: ast.FunctionDef) -> tuple:
    args = fn.args
    return (
        tuple(a.arg for a in args.args),
        len(args.defaults),
        tuple(a.arg for a in args.kwonlyargs),
        args.vararg.arg if args.vararg else None,
        args.kwarg.arg if args.kwarg else None,
    )


def _expr_shape(e: ast.AST) -> str:
    # Coarse shape: only distinguish concrete container/literal types.
    # Anything scalar-ish (names, binops, constants, simple exprs) collapses
    # into "scalar" so small expression-level tweaks don't trigger false diffs.
    if isinstance(e, ast.Dict):                     return "dict"
    if isinstance(e, (ast.List, ast.ListComp)):     return "list"
    if isinstance(e, ast.Tuple):                    return "tuple"
    if isinstance(e, (ast.Set, ast.SetComp)):       return "set"
    if isinstance(e, ast.DictComp):                 return "dict"
    if isinstance(e, ast.Call):
        fn = e.func
        name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
        # Calls that construct known containers:
        if name in {"list", "sorted", "tuple", "set", "dict", "frozenset"}:
            return name if name != "sorted" else "list"
        return "scalar"
    return "scalar"


def _return_fingerprint(fn: ast.FunctionDef) -> tuple:
    shapes = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and n.value is not None:
            shapes.append(_expr_shape(n.value))
    return tuple(sorted(shapes))


def _raises_fingerprint(fn: ast.FunctionDef) -> tuple:
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Raise) and n.exc is not None:
            exc = n.exc
            if isinstance(exc, ast.Call):
                exc = exc.func
            if isinstance(exc, ast.Name):
                out.append(exc.id)
            elif isinstance(exc, ast.Attribute):
                out.append(exc.attr)
    return tuple(sorted(out))


def _diff_kind(old: ast.FunctionDef, new: ast.FunctionDef) -> tuple[bool, str]:
    changed: list[str] = []
    if _sig_fingerprint(old) != _sig_fingerprint(new): changed.append("params")
    if _return_fingerprint(old) != _return_fingerprint(new): changed.append("return_shape")
    if _raises_fingerprint(old) != _raises_fingerprint(new): changed.append("raises")
    if not changed: return False, "none"
    if len(changed) == 1: return True, changed[0]
    return True, "multiple"


def _rg_hits(repo_root: Path, short: str, skip_file: str, time_budget_s: float) -> tuple[list[tuple[str, int]], bool]:
    # Note: do NOT pass --glob "!skip_file". Same-file callers (e.g. a sibling
    # method calling self.<short>) are legitimate. The AST confirm rejects the
    # definition line because FunctionDef is not ast.Call.
    if shutil.which("rg"):
        try:
            proc = subprocess.run(
                ["rg", "-n", "--no-heading", "-w", short,
                 "--glob", "!tests/**", "--glob", "!**/test_*.py",
                 str(repo_root)],
                capture_output=True, text=True, timeout=time_budget_s,
            )
        except subprocess.TimeoutExpired:
            return [], True
        hits: list[tuple[str, int]] = []
        for line in proc.stdout.splitlines():
            try:
                path_part, ln_part, _ = line.split(":", 2)
            except ValueError:
                continue
            try:
                ln = int(ln_part)
            except ValueError:
                continue
            try:
                rel = str(Path(path_part).relative_to(repo_root))
            except ValueError:
                rel = path_part
            hits.append((rel, ln))
        return hits, False
    return _py_grep(repo_root, short, skip_file)


def _py_grep(repo_root: Path, short: str, _unused: str) -> tuple[list[tuple[str, int]], bool]:
    # skip_file arg is ignored: same-file callers are legitimate; the AST
    # confirm step rejects the definition line.
    import re as _re
    word_re = _re.compile(rf"\b{_re.escape(short)}\b")
    hits: list[tuple[str, int]] = []
    for p in repo_root.rglob("*.py"):
        try:
            rel = p.relative_to(repo_root)
        except ValueError:
            continue
        parts = rel.parts
        if "tests" in parts:
            continue
        if parts and parts[-1].startswith("test_"):
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if word_re.search(line):
                hits.append((str(rel), i))
    return hits, False


def _confirm_call_in_text(text: str, target_line: int, short: str) -> tuple[bool, str]:
    """AST-confirm that line `target_line` of `text` contains a call to `short`.
    Returns (ok, enclosing_qualname)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False, ""
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node, "lineno", 0) != target_line:
            continue
        fn = node.func
        name = None
        if isinstance(fn, ast.Name): name = fn.id
        elif isinstance(fn, ast.Attribute): name = fn.attr
        if name != short:
            continue
        cur = node
        enclosing: list[str] = []
        while cur in parent:
            cur = parent[cur]
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                enclosing.append(cur.name)
        qual = ".".join(reversed(enclosing))
        return True, qual
    return False, ""


def _confirm_call_at(path: Path, target_line: int, short: str) -> tuple[bool, str]:
    try:
        text = path.read_text()
    except OSError:
        return False, ""
    return _confirm_call_in_text(text, target_line, short)


def _load_old_fn(root_cause_file: str, short: str, cls_hint: str,
                 repo_root: Path | None, container) -> ast.FunctionDef | None:
    """Resolve the existing FunctionDef for `short` from host repo first,
    then live container as fallback."""
    src: str | None = None
    if repo_root is not None:
        try:
            src = (repo_root / root_cause_file).read_text()
        except OSError:
            src = None
    if src is None and container is not None:
        try:
            from .container_search import container_read_text
            src = container_read_text(container, root_cause_file) or None
        except Exception:
            src = None
    if not src:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    return _find_func_by_short(tree, short, cls_hint)


def _hits_for(short: str, skip_file: str, repo_root: Path | None,
              container, time_budget_s: float) -> tuple[list[tuple[str, int]], bool]:
    if repo_root is not None:
        return _rg_hits(repo_root, short, skip_file, time_budget_s)
    if container is not None:
        from .container_search import container_rg
        return container_rg(container, short, time_budget_s=time_budget_s)
    return [], False


def _confirm_call(rel: str, ln: int, short: str,
                  repo_root: Path | None, container) -> tuple[bool, str]:
    if repo_root is not None:
        return _confirm_call_at(repo_root / rel, ln, short)
    if container is not None:
        from .container_search import container_read_text
        text = container_read_text(container, rel)
        if not text:
            return False, ""
        return _confirm_call_in_text(text, ln, short)
    return False, ""


def compute_contract_impact(
    root_cause_file: str,
    root_cause_func_qualname: str,
    suggested_fix_text: str,
    repo_root: Path | None = None,
    time_budget_s: float = 10.0,
    file_budget: int = 500,
    max_callers: int = 50,
    container=None,
) -> dict:
    """Always populates `callers` when `root_cause_func_qualname` resolves to
    a real function. `changed`/`kind` reflect whether the LLM-supplied fix
    actually altered the signature; absent fix code is `signature_unchanged`
    rather than grounds to skip caller search.

    `max_callers` caps the returned list to bound prompt/disk size when the
    short name is generic (e.g. `run`, `process`) and matches widely across
    the repo. The cap is on confirmed callers, not raw grep hits — the AST
    confirm step keeps the work proportional to confirmed matches."""
    base = {"changed": False, "kind": "none", "summary": "",
            "callers": [], "callers_truncated": False, "evidence_refs": []}
    short, cls_hint = _short_and_cls(root_cause_func_qualname)
    if not short:
        return base

    old_fn = _load_old_fn(root_cause_file, short, cls_hint, repo_root, container)
    if old_fn is None:
        return base

    changed, kind = False, "signature_unchanged"
    blocks = _extract_py_blocks(suggested_fix_text)
    new_fn = None
    for block in blocks:
        try:
            bt = ast.parse(block)
        except SyntaxError:
            continue
        found = _find_func_by_short(bt, short, cls_hint)
        if found is not None:
            new_fn = found
            break
    if new_fn is not None:
        diff_changed, diff_kind = _diff_kind(old_fn, new_fn)
        if diff_changed:
            changed, kind = True, diff_kind

    hits, timeout = _hits_for(short, root_cause_file, repo_root, container, time_budget_s)
    callers: list[dict] = []
    seen_files: set[str] = set()
    truncated = timeout
    for rel, ln in hits:
        if len(callers) >= max_callers:
            truncated = True
            break
        if len(seen_files) >= file_budget:
            truncated = True
            break
        seen_files.add(rel)
        ok, qual = _confirm_call(rel, ln, short, repo_root, container)
        if ok:
            callers.append({"file": rel, "lineno": ln, "qualname": qual})

    summary = (f"{short} contract changed ({kind})" if changed
               else f"{short}: signature unchanged")
    return {"changed": changed, "kind": kind, "summary": summary,
            "callers": callers, "callers_truncated": truncated,
            "evidence_refs": ["signature_diff" if changed else "callers_only"]}
