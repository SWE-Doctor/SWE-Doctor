from debug_agent.conclusion_validator import validate_against_pdb_log


def test_keeps_files_seen_in_pdb_frames():
    pdb_log = [
        {"kind": "start", "initial_frame": {"file": "/app/lib/foo.py", "lineno": 10}},
        {"kind": "cmd", "current_frame": {"file": "/app/lib/foo.py", "lineno": 12}},
    ]
    files, funcs = validate_against_pdb_log(
        ["/app/lib/foo.py"], ["foo.bar"], pdb_log,
    )
    assert files == ["/app/lib/foo.py"]
    assert funcs == ["foo.bar"]


def test_replaces_hallucinated_with_pdb_frames():
    pdb_log = [
        {"kind": "start", "initial_frame":
            {"file": "/app/openlibrary/catalog/add_book/__init__.py", "lineno": 50}},
    ]
    files, funcs = validate_against_pdb_log(
        ["/app/openlibrary/openlibrary/catalog/add_book.py"],
        ["foo.bar"],
        pdb_log,
    )
    assert files == ["/app/openlibrary/catalog/add_book/__init__.py"]
    # PDB tells us the file but not the bug function — empty string is honest.
    assert funcs == [""]


def test_passthrough_when_no_pdb_evidence():
    files, funcs = validate_against_pdb_log(
        ["/app/lib/x.py"], ["x.do"], pdb_log=[],
    )
    assert files == ["/app/lib/x.py"]
    assert funcs == ["x.do"]


def test_partial_overlap_keeps_validated_drops_hallucinated_with_funcs():
    """The bug this guards: validator must drop the function entry whose
    file was filtered out, otherwise downstream zip-by-index pairs the
    surviving file with the wrong function."""
    pdb_log = [{"kind": "cmd", "current_frame": {"file": "/app/lib/real.py", "lineno": 1}}]
    files, funcs = validate_against_pdb_log(
        ["/app/lib/real.py", "/app/lib/fake.py"],
        ["real.func", "fake.func"],
        pdb_log,
    )
    assert files == ["/app/lib/real.py"]
    assert funcs == ["real.func"]


def test_test_file_drop_preserves_function_alignment():
    """Test files dropped from the file list must also drop the same-index
    function entry — otherwise the surviving production file inherits the
    test file's function name."""
    pdb_log = []
    files, funcs = validate_against_pdb_log(
        ["tests/test_foo.py", "/app/lib/bar.py"],
        ["test_foo.test_case", "bar.work"],
        pdb_log,
    )
    assert files == ["/app/lib/bar.py"]
    assert funcs == ["bar.work"]


def test_function_list_shorter_than_files_pads_with_empty():
    """LLM occasionally emits more files than functions; missing slots
    pair with empty string and survive filtering."""
    pdb_log = []
    files, funcs = validate_against_pdb_log(
        ["/app/a.py", "/app/b.py"],
        ["a.func"],  # only one function for two files
        pdb_log,
    )
    assert files == ["/app/a.py", "/app/b.py"]
    assert funcs == ["a.func", ""]


def test_path_norm_matching():
    """Conclusion uses /app/lib/x.py, frame log uses dist-packages variant —
    after path_norm they match, and the function tags along."""
    pdb_log = [{"kind": "cmd", "current_frame":
                {"file": "/usr/local/lib/python3.9/dist-packages/ansible/utils/vars.py", "lineno": 5}}]
    files, funcs = validate_against_pdb_log(
        ["/app/lib/ansible/utils/vars.py"],
        ["ansible.utils.vars.combine_vars"],
        pdb_log,
    )
    assert files == ["/app/lib/ansible/utils/vars.py"]
    assert funcs == ["ansible.utils.vars.combine_vars"]
