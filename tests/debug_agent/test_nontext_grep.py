from debug_agent.nontext_grep import find_related_non_code


def _w(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_filters_python_and_tests(tmp_path):
    _w(tmp_path, "docs/fields.yaml", "ebooks: list of dicts\n")
    _w(tmp_path, "README.md", "See `ebooks` in docs.\n")
    _w(tmp_path, "pkg/mod.py", "# ebooks variable here\n")
    _w(tmp_path, "tests/test_x.md", "ebooks in tests/\n")

    out = find_related_non_code("ebooks", tmp_path)
    files = {h["file"] for h in out["hits"]}
    assert "docs/fields.yaml" in files
    assert "README.md" in files
    assert "pkg/mod.py" not in files
    assert "tests/test_x.md" not in files


def test_identifier_too_common_returns_skip(tmp_path):
    for i in range(210):
        _w(tmp_path, f"docs/f{i}.md", "x = common_id\n")
    out = find_related_non_code("common_id", tmp_path, too_common_threshold=200)
    assert out["hits"] == []
    assert "too common" in out["note"]


def test_short_identifier_skipped(tmp_path):
    out = find_related_non_code("x", tmp_path)
    assert out["hits"] == []
    assert "too short" in out["note"]


def test_no_hits_returns_empty(tmp_path):
    _w(tmp_path, "README.md", "nothing here\n")
    out = find_related_non_code("nonexistent_ident_xyz", tmp_path)
    assert out["hits"] == []
    assert "no identifier matches" in out["note"]


def test_find_related_non_code_via_container():
    """Lever 3: when repo_root is None but container is provided, the search
    runs inside the container."""
    class _R:
        def __init__(self, rc, stdout): self.returncode = rc; self.stdout = stdout; self.stderr = ""

    class _C:
        def __init__(self): self.calls = []
        def exec_bash(self, cmd, timeout=120):
            self.calls.append(cmd)
            return _R(0, "/app/docs/readme.md:3:see foo for details\n"
                        "/app/conf/foo.yaml:1:foo: 1\n")

    out = find_related_non_code("foo", repo_root=None, container=_C())
    files = {h["file"] for h in out["hits"]}
    assert "docs/readme.md" in files
    assert "conf/foo.yaml" in files
    assert out["note"] == ""
