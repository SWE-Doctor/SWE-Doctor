from pathlib import Path
from debug_agent.caller_search import compute_contract_impact


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_no_fenced_block_returns_signature_unchanged(tmp_path):
    """Lever 2: prose-only suggested_fix is not a reason to skip caller search;
    `kind` must reflect that the signature was not (provably) altered."""
    _write(tmp_path, "pkg/mod.py", "def foo(x): return x\n")
    out = compute_contract_impact("pkg/mod.py", "pkg.mod.foo",
                                  "just prose, no code",
                                  tmp_path)
    assert out["changed"] is False
    assert out["kind"] == "signature_unchanged"
    # No callers in this fixture; just confirm callers field is shape-correct.
    assert out["callers"] == []


def test_return_shape_change_detected_and_callers_found(tmp_path):
    _write(tmp_path, "pkg/mod.py",
           "class C:\n"
           "    def add(self, xs):\n"
           "        return xs\n")
    _write(tmp_path, "pkg/caller_a.py",
           "from pkg.mod import C\n"
           "def use():\n"
           "    return C().add([1,2])\n")
    _write(tmp_path, "pkg/caller_b.py",
           "def another(c):\n"
           "    return c.add([3,4]) if c else None\n")
    _write(tmp_path, "tests/test_mod.py",
           "def test_add(): C().add([1])\n")

    fix = "```python\n" \
          "class C:\n" \
          "    def add(self, xs):\n" \
          "        return sorted(set(xs))\n" \
          "```"
    out = compute_contract_impact("pkg/mod.py", "pkg.mod.C.add", fix, tmp_path)
    assert out["changed"] is True
    assert out["kind"] in ("return_shape", "multiple")
    caller_files = {c["file"] for c in out["callers"]}
    assert "pkg/caller_a.py" in caller_files
    assert "pkg/caller_b.py" in caller_files
    assert "tests/test_mod.py" not in caller_files
    assert out["evidence_refs"] == ["signature_diff"]


def test_no_callers_when_func_missing(tmp_path):
    _write(tmp_path, "pkg/mod.py", "def lonely(x): return x\n")
    fix = "```python\ndef lonely(x, y): return x+y\n```"
    out = compute_contract_impact("pkg/mod.py", "pkg.mod.lonely", fix, tmp_path)
    assert out["changed"] is True
    assert out["kind"] in ("params", "multiple")
    assert out["callers"] == []


def test_same_file_caller_is_detected(tmp_path):
    """A sibling method calling self.<short> in the same file as the definition
    must be reported. The AST confirm rejects the FunctionDef line itself."""
    _write(tmp_path, "pkg/mod.py",
           "class C:\n"
           "    def add(self, xs):\n"
           "        return xs\n"
           "    def build(self, xs):\n"
           "        return self.add(xs)\n")
    fix = "```python\nclass C:\n    def add(self, xs):\n        return sorted(set(xs))\n```"
    out = compute_contract_impact("pkg/mod.py", "pkg.mod.C.add", fix, tmp_path)
    files = {(c["file"], c["lineno"]) for c in out["callers"]}
    assert ("pkg/mod.py", 5) in files, f"expected same-file self-caller at line 5, got {files}"


def test_signature_unchanged_returns_not_changed(tmp_path):
    _write(tmp_path, "pkg/mod.py", "def foo(x): return x\n")
    fix = "```python\ndef foo(x): return x + 0  # noop refactor\n```"
    out = compute_contract_impact("pkg/mod.py", "pkg.mod.foo", fix, tmp_path)
    assert out["changed"] is False


def test_callers_populated_when_no_code_blocks(tmp_path):
    """Lever 2: prose-only suggested_fix must still produce callers."""
    _write(tmp_path, "lib.py", "def target():\n    return 1\n")
    _write(tmp_path, "user.py",
           "from lib import target\n\ndef caller():\n    return target()\n")
    out = compute_contract_impact(
        root_cause_file="lib.py",
        root_cause_func_qualname="target",
        suggested_fix_text="The bug is that target returns the wrong int.",
        repo_root=tmp_path,
    )
    assert out["changed"] is False
    assert out["kind"] == "signature_unchanged"
    files = {c["file"] for c in out["callers"]}
    assert "user.py" in files


def test_callers_populated_when_signature_unchanged(tmp_path):
    """Lever 2: code block present but signature identical → still surface callers."""
    _write(tmp_path, "lib.py", "def target(x):\n    return x\n")
    _write(tmp_path, "user.py",
           "from lib import target\n\ndef caller(): return target(1)\n")
    fix = "```python\ndef target(x):\n    return x + 1\n```"
    out = compute_contract_impact("lib.py", "target", fix, tmp_path)
    assert out["changed"] is False
    assert out["kind"] == "signature_unchanged"
    assert any(c["file"] == "user.py" for c in out["callers"])


def test_callers_still_marked_changed_when_signature_differs(tmp_path):
    """Regression guard: existing changed-signature path must keep working."""
    _write(tmp_path, "lib.py", "def target(x):\n    return x\n")
    _write(tmp_path, "user.py",
           "from lib import target\n\ndef caller(): return target(1)\n")
    fix = "```python\ndef target(x, y):\n    return x + y\n```"
    out = compute_contract_impact("lib.py", "target", fix, tmp_path)
    assert out["changed"] is True
    assert out["kind"] == "params"
    assert any(c["file"] == "user.py" for c in out["callers"])


def test_callers_capped_at_max_callers_with_truncated_flag(tmp_path):
    """Generic short names like `run` blow up across large repos. Cap the
    confirmed-caller list so prompt/disk size stays bounded; mark truncated."""
    _write(tmp_path, "lib.py", "def run():\n    return 1\n")
    # Make 8 distinct caller files, all calling `run()`. Cap to 3 → truncated.
    for i in range(8):
        _write(tmp_path, f"u{i}.py",
               f"from lib import run\n\ndef caller_{i}(): return run()\n")
    out = compute_contract_impact(
        root_cause_file="lib.py",
        root_cause_func_qualname="run",
        suggested_fix_text="prose only",
        repo_root=tmp_path,
        max_callers=3,
    )
    assert len(out["callers"]) == 3
    assert out["callers_truncated"] is True
