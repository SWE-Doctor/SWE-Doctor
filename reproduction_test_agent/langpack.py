"""Per-language registry for Stage-1 reproduction-test generation.

Single source of language truth: every language-specific prompt, filename,
glob, and parser lives here. Existing modules consult get_langpack(config.language).
Default 'python' preserves the original Verified behavior exactly. 'go' mirrors
the JS adaptation's approach for SWE-bench Pro golang instances.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LangPack:
    name: str
    test_filename: str
    code_fence: str
    system_prompt: str
    repair_prompt_lang_hint: str
    source_globs: list[str]
    test_path_substrings: list[str]
    grep_type_args: str
    func_def_grep: str           # a grep -E pattern template; {name} substituted
    error_types: list[str] = field(default_factory=list)


_PY_SYSTEM = """\
You are an expert Python test engineer. Your task is to write a single reproduction \
test for a software bug described in a GitHub issue.

Requirements:
- Write a complete, self-contained Python test file
- The test should FAIL on the current (buggy) codebase
- The test should PASS after the bug is fixed
- Use pytest-style assertions (assert, pytest.raises, etc.)
- Include necessary imports
- The test function name should start with test_
- Keep the test focused on reproducing the specific issue
- Do NOT test anything beyond what the issue describes

CRITICAL — choosing the right code path:
- The issue describes a USER-FACING WORKFLOW. Your test MUST exercise the entry
  point of that workflow, not a lower-level helper that happens to share keywords.
- If a "Relevant Source Files" list is provided below, your test MUST `import`
  and call into at least one of those modules. Prefer the file that is the
  workflow entry-point (e.g. for "importing books from Wikisource" use the
  add_book / import pipeline, NOT the records search/matching API).
- A test that reproduces the symptom in the wrong layer is WORSE than no test —
  it will mislead downstream root-cause analysis. When in doubt, pick the
  highest-level entry point that the issue text actually names.

Output ONLY the Python test code, no explanations. Start with imports.
"""

_GO_SYSTEM = """\
You are an expert Go test engineer. Your task is to write a single reproduction \
test for a software bug described in a GitHub issue, for a repository whose tests \
run under `go test`.

Requirements:
- Write a complete, self-contained Go test file in the SAME package as the code \
under test (declare `package <pkg>` matching the target file, NOT `<pkg>_test`), so \
the test can reach unexported symbols. Use the standard `testing` package.
- The test should FAIL on the current (buggy) codebase and PASS after the bug is fixed.
- Use a single `func TestXxx(t *testing.T)` with focused assertions via \
`t.Fatalf`/`t.Errorf` (or testify `require`/`assert` ONLY if the repo already imports it).
- Import production code by its module path from go.mod; symbols in the same package \
need no import at all.
- Keep it LEAN: exactly ONE test function reproducing only the single behavior in the \
issue; never bundle multiple unrelated checks.
- ISOLATE the unit under test: do NOT spin up the whole server/database/HTTP stack \
unless the bug genuinely requires it — construct the specific struct/function under \
test and stub or fake its heavy dependencies.
- Do NOT add new dependencies; only use what the repo already imports.
- Go is STRICT: an unused import or an unused local variable is a COMPILE ERROR. \
Import only packages you actually reference, and discard an unused return value \
with `_`. A test that does not compile cannot reproduce the bug.
- The test must assert the FIXED (correct) behavior, so it FAILS on the current \
buggy code. Do NOT assert the current buggy result — if the bug is that a call \
wrongly returns an error, assert it should NOT error (and watch it fail now).

CRITICAL — choosing the right code path:
- The issue describes a USER-FACING WORKFLOW. Your test MUST exercise the entry
  point of that workflow (the exported function/method/handler the issue names),
  not a lower-level helper that merely shares a keyword with it.
- If a "Relevant Source Files" list is provided below, your test MUST call into a
  symbol from at least one of those files — same-package for unexported symbols,
  or via its go.mod module path otherwise. Prefer the workflow entry-point file
  over a deep helper.
- A test that reproduces the symptom in the wrong layer is WORSE than no test —
  it will mislead downstream root-cause analysis. When in doubt, pick the
  highest-level entry point that the issue text actually names.

Output ONLY the Go test code, no explanations. Start with the `package` clause.
"""

_REGISTRY: dict[str, LangPack] = {
    "python": LangPack(
        name="python",
        test_filename="repro_test.py",
        code_fence="python",
        system_prompt=_PY_SYSTEM,
        repair_prompt_lang_hint="Python",
        source_globs=["*.py"],
        test_path_substrings=["test_", "_test.py", "/tests/", "conftest.py"],
        grep_type_args="--type py",
        func_def_grep=r"def {name}",
        error_types=["ModuleNotFoundError", "ImportError", "TypeError", "AttributeError"],
    ),
    "go": LangPack(
        name="go",
        test_filename="repro_test.go",
        code_fence="go",
        system_prompt=_GO_SYSTEM,
        repair_prompt_lang_hint="Go",
        source_globs=["*.go"],
        test_path_substrings=["_test.go"],
        grep_type_args="--type go",
        func_def_grep=r"func (\([^)]*\) )?{name}",   # plain func or method receiver
        error_types=["panic:", "undefined:", "cannot use", "build failed", "FAIL"],
    ),
}


def get_langpack(language: str) -> LangPack:
    return _REGISTRY[language]


def parse_test_output(language: str, output: str) -> tuple[str, str]:
    """Return (error_type, error_message) from a test run's combined output."""
    if language == "python":
        from .executor import _parse_error
        return _parse_error(output)
    if language == "go":
        # Reuse the mature go_runner extractor (handles --- FAIL, JSON, build errors).
        _pro = Path(__file__).resolve().parent.parent / "run_pro_test"
        if str(_pro) not in sys.path:
            sys.path.insert(0, str(_pro))
        from go_runner import extract_failed_tests  # type: ignore
        failed = extract_failed_tests(output, "")
        if not failed:
            return "", ""
        if any("build" in f.lower() for f in failed):
            return "build failed", failed[0]
        return "TestFailure", ", ".join(failed[:5])
    raise KeyError(language)
