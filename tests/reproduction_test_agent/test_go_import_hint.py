"""Go import-hint: the one genuinely new Stage-1 logic for go (JS proved this
is where languages diverge). Go test files MUST live in the package-under-test's
directory and declare the SAME package to reach unexported symbols, so we derive
the test path beside the localized source and tell the LLM the exact package."""
from reproduction_test_agent.pipeline import (
    _derive_go_test_filename,
    _build_go_gen_hint,
)


def test_derive_go_test_filename_beside_source_package():
    loc = {"relevant_files": ["server/evaluator.go"], "test_files": []}
    assert _derive_go_test_filename(loc) == "server/zzz_repro_test.go"


def test_derive_go_test_filename_skips_test_files_when_picking_source():
    loc = {"relevant_files": ["pkg/foo_test.go", "pkg/foo.go"]}
    assert _derive_go_test_filename(loc) == "pkg/zzz_repro_test.go"


def test_derive_go_test_filename_falls_back_to_existing_test_dir():
    loc = {"relevant_files": [], "test_files": ["internal/store/store_test.go"]}
    assert _derive_go_test_filename(loc) == "internal/store/zzz_repro_test.go"


def test_derive_go_test_filename_root_fallback():
    assert _derive_go_test_filename({"relevant_files": [], "test_files": []}) == "zzz_repro_test.go"


def test_go_gen_hint_states_path_package_and_unexported_access():
    loc = {
        "relevant_files": ["server/evaluator.go"],
        "file_contents": {"server/evaluator.go": "package server\n\nfunc (s *Server) Evaluate() {}\n"},
        "test_contents": {},
    }
    hint = _build_go_gen_hint("server/zzz_repro_test.go", loc)
    assert "server/zzz_repro_test.go" in hint
    assert "package server" in hint
    assert "unexported" in hint.lower()


def test_go_gen_hint_empty_when_no_package_found():
    # No file_contents → still produce a usable hint (path + generic package guidance)
    loc = {"relevant_files": ["server/evaluator.go"], "file_contents": {}, "test_contents": {}}
    hint = _build_go_gen_hint("server/zzz_repro_test.go", loc)
    assert "server/zzz_repro_test.go" in hint
    assert "package" in hint.lower()


def test_derive_prefers_existing_test_dir_over_first_source():
    # Regression: flipt batch-eval bug. relevant_files[0] is errors/errors.go but
    # the behavior under test lives in server/ (where the existing test is). The
    # repro must land in server/, not errors/, so the LLM tests existing server
    # symbols rather than fix-only errors.ErrDisabledf.
    loc = {
        "relevant_files": ["errors/errors.go", "server/evaluator.go"],
        "test_files": ["server/evaluator_test.go"],
    }
    assert _derive_go_test_filename(loc) == "server/zzz_repro_test.go"


def test_gen_hint_package_matches_test_directory_not_first_source():
    loc = {
        "relevant_files": ["errors/errors.go", "server/evaluator.go"],
        "test_files": ["server/evaluator_test.go"],
        "file_contents": {
            "errors/errors.go": "package errors\n\nfunc x(){}\n",
            "server/evaluator.go": "package server\n\nfunc (s *Server) Evaluate() {}\n",
        },
        "test_contents": {},
    }
    test_fn = _derive_go_test_filename(loc)
    hint = _build_go_gen_hint(test_fn, loc)
    assert "package server" in hint
    assert "package errors" not in hint
