"""Bug localization: find relevant files, test files, and focal functions.

Uses LLM + repo exploration to identify context for test generation.
Produces a structured dict used by masks to control what context goes into prompts.
"""

import logging
import re

from .llm import llm_call
from .executor import run_command, Environment
from .langpack import get_langpack

logger = logging.getLogger("repro_test.localizer")

LOCALIZE_PROMPT = """\
You are a bug localization expert. Given a software issue description and the repository structure, identify the files that need to change.

## Issue Description
{issue_text}

## Repository Structure (top-level + relevant subdirectories)
{repo_structure}

{keyword_hits_block}
## CRITICAL Instructions

The issue describes a USER-FACING WORKFLOW (e.g. "importing books", "submitting a list", "matching editions"). The bug almost always lives at the **entry point** of that workflow, NOT at a lower-level helper that happens to share keywords.

Before listing files, ask yourself:
1. What is the top-level *workflow* the issue describes? (e.g. "import a book from Wikisource", not "search editions")
2. Which module is the *entry point* for that workflow? (e.g. `catalog/add_book` for book imports, not `records/driver` which is the search API)
3. Are there multiple plausible entry points (CLI script, web handler, library function, batch importer)? List ALL of them — do not commit to one layer prematurely.

Then identify:
1. **Relevant source files** — up to **8** files. INCLUDE the workflow entry-point modules (top of stack), not just symptom-layer helpers. If the issue says "importing X", make sure at least one file is the import entry point.
2. **Relevant test files** — up to 3 existing test files near the changed code.
3. **Focal functions** — up to 6 functions, biased toward the workflow entry-points listed above.

## Output Format (no markdown fences)
RELEVANT_FILES:
path/to/file1.py
path/to/file2.py

TEST_FILES:
tests/test_something.py

FOCAL_FUNCTIONS:
module.ClassName.method_name
module.function_name

Be specific. Prefer files/functions explicitly mentioned in the issue. When in doubt, **broaden the list** to include both the entry-point AND the symptom layer rather than guess one — over-including is cheaper than missing the real fix site.
"""


# Generic English/programming words to drop from keyword extraction.
_KEYWORD_STOPWORDS = {
    "this", "that", "with", "when", "from", "should", "would", "could", "have",
    "been", "into", "must", "will", "than", "then", "they", "them", "their",
    "there", "where", "which", "what", "while", "such", "only", "also", "some",
    "any", "all", "for", "and", "but", "not", "are", "the", "was", "were",
    "has", "had", "use", "used", "uses", "using", "make", "made", "get", "got",
    "set", "see", "new", "old", "via", "out", "off", "yes", "true", "false",
    "none", "null", "self", "function", "method", "class", "module", "import",
    "imports", "issue", "bug", "fix", "fixed", "test", "tests", "case", "cases",
    "value", "values", "field", "fields", "data", "type", "types", "name",
    "names", "code", "file", "files", "line", "lines", "list", "lists", "dict",
    "dicts", "string", "strings", "result", "results", "expected", "actual",
    "behavior", "behaviour", "current", "before", "after", "above", "below",
    "step", "steps", "title", "description", "example", "examples", "interface",
    "interfaces", "introduced", "reproduce", "observe", "ensure", "ensures",
    "wikipedia", "github", "python", "django",
}


def _extract_keywords(issue_text: str, max_keywords: int = 8) -> list[str]:
    """Extract distinctive identifiers/nouns from the issue text.

    Strategy: prefer CamelCase, snake_case, dotted paths, and quoted strings —
    these are almost always domain-specific identifiers. Fall back to long
    lowercase words filtered by a stopword list.
    """
    text = issue_text or ""
    candidates: list[tuple[int, str]] = []  # (rank, token), lower rank = better

    # 1) Quoted/back-ticked tokens — highest priority.
    for m in re.finditer(r"[`'\"]([A-Za-z_][\w./]{2,})[`'\"]", text):
        candidates.append((0, m.group(1)))

    # 2) snake_case / dotted paths
    for m in re.finditer(r"\b([a-z][a-z0-9_]*(?:_[a-z0-9_]+)+)\b", text):
        candidates.append((1, m.group(1)))
    for m in re.finditer(r"\b([a-z_][\w]*(?:\.[a-z_][\w]*){1,})\b", text):
        candidates.append((1, m.group(1)))

    # 3) CamelCase identifiers
    for m in re.finditer(r"\b([A-Z][a-z]+[A-Z][A-Za-z]+)\b", text):
        candidates.append((2, m.group(1)))

    # 4) Lowercase content words ≥5 chars, not in stopwords.
    for m in re.finditer(r"\b([A-Za-z]{5,})\b", text):
        w = m.group(1)
        if w.lower() not in _KEYWORD_STOPWORDS and not w[0].isupper():
            candidates.append((3, w))

    # Dedup, preserve best rank, preserve first-seen order at the same rank.
    seen: dict[str, int] = {}
    order: dict[str, int] = {}
    for i, (rank, tok) in enumerate(candidates):
        key = tok
        if key not in seen or rank < seen[key]:
            seen[key] = rank
            order[key] = i

    sorted_tokens = sorted(seen.keys(), key=lambda k: (seen[k], order[k]))
    return sorted_tokens[:max_keywords]


def _grep_files_for_keywords(
    keywords: list[str], env: Environment, cwd: str, language: str = "python", max_files: int = 25,
) -> list[tuple[str, int]]:
    """Grep the repo for files matching the keywords. Returns [(file, hits)].

    Files matching MORE keywords rank higher. Excludes test/build/vendor dirs.
    """
    if not keywords:
        return []

    lp = get_langpack(language)
    # Build a single ripgrep-style alternation, fall back to grep.
    pat = "|".join(re.escape(k) for k in keywords)
    includes = " ".join(f"--include='{g}'" for g in lp.source_globs)
    if language == "go":
        exclude_dirs = "/(vendor|node_modules|build|dist)/"
    else:
        exclude_dirs = "/(tests?|__pycache__|build|dist|\\.tox|\\.venv|venv|site-packages)/"
    # Try ripgrep first.
    cmd = (
        f"(rg -l {lp.grep_type_args} -e '{pat}' . 2>/dev/null || "
        f"grep -rl {includes} -E '{pat}' . 2>/dev/null) "
        f"| grep -Ev '{exclude_dirs}' "
        "| head -200"
    )
    result = run_command(cmd, env, cwd=cwd, timeout=30)
    if result.get("returncode") not in (0, 1) or not result.get("output"):
        return []

    suffixes = tuple(g.lstrip("*") for g in lp.source_globs)
    files = [
        line.strip().lstrip("./")
        for line in result["output"].splitlines()
        if line.strip().endswith(suffixes)
    ]

    # Score each file by how many distinct keywords it contains.
    scored: list[tuple[str, int]] = []
    for f in files[:60]:  # cap to avoid huge per-file scans
        cmd2 = f"grep -ciE '{pat}' '{f}' 2>/dev/null || true"
        r2 = run_command(cmd2, env, cwd=cwd, timeout=5)
        try:
            count = int((r2.get("output") or "0").strip().splitlines()[0])
        except (ValueError, IndexError):
            count = 1
        scored.append((f, count))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:max_files]


def localize(issue_text: str, env: Environment, cwd: str, model: str, language: str = "python") -> dict:
    """Localize the bug: find relevant files, test files, and focal functions.

    Returns dict with keys:
        relevant_files: list[str]
        test_files: list[str]
        focal_functions: list[str]
        file_contents: dict[str, str]  — contents of relevant files (truncated)
        test_contents: dict[str, str]  — contents of test files (truncated)
    """
    repo_structure = _get_repo_structure(env, cwd, language=language)

    # Deterministic keyword grep — gives the LLM a high-precision shortlist
    # of files that actually mention issue terms. This catches the case where
    # the LLM, looking at a flat 500-file listing, picks the wrong code layer.
    keywords = _extract_keywords(issue_text)
    keyword_hits = _grep_files_for_keywords(keywords, env, cwd, language=language) if keywords else []
    if keyword_hits:
        kw_block_lines = [
            "## Files Mentioning Issue Keywords (deterministic grep — STRONG PRIOR)",
            f"keywords = {', '.join(keywords)}",
            "These files literally contain the issue's distinctive terms. The bug",
            "is highly likely to live in one of these — make sure your RELEVANT_FILES",
            "list contains the strongest workflow-entry-point candidates from here.",
            "",
        ]
        for f, n in keyword_hits[:20]:
            kw_block_lines.append(f"  {f}  (matches={n})")
        kw_block_lines.append("")
        keyword_hits_block = "\n".join(kw_block_lines)
    else:
        keyword_hits_block = ""

    response = llm_call(
        messages=[{"role": "user", "content": LOCALIZE_PROMPT.format(
            issue_text=issue_text,
            repo_structure=repo_structure,
            keyword_hits_block=keyword_hits_block,
        )}],
        model=model,
        caller="localizer",
    )

    result = _parse_localization(response)

    # Defense-in-depth: ensure the top keyword-hit files appear in
    # relevant_files even if the LLM ignored them. We add up to 3 high-signal
    # files at the END of the list, so the LLM's primary picks still dominate
    # but the keyword evidence is never lost.
    if keyword_hits:
        existing = set(result["relevant_files"])
        added = 0
        for f, n in keyword_hits:
            if added >= 3:
                break
            if f not in existing and n >= 2:
                result["relevant_files"].append(f)
                existing.add(f)
                added += 1
        result["keyword_hits"] = [
            {"file": f, "matches": n} for f, n in keyword_hits[:20]
        ]
        result["keywords"] = keywords

    # Read file contents
    result["file_contents"] = {}
    for f in result["relevant_files"]:
        content = _read_file(env, cwd, f)
        if content:
            result["file_contents"][f] = content

    result["test_contents"] = {}
    for f in result["test_files"]:
        content = _read_file(env, cwd, f)
        if content:
            result["test_contents"][f] = content

    logger.info(
        "Localized: %d files, %d tests, %d functions",
        len(result["relevant_files"]),
        len(result["test_files"]),
        len(result["focal_functions"]),
    )
    return result


def _get_repo_structure(env: Environment, cwd: str, language: str = "python", max_depth: int = 3) -> str:
    """Get a compact view of the repo directory tree.

    Strategy: first show directory layout (depth=2), then prioritize test
    directories, finally list remaining source files up to a budget.
    """
    lp = get_langpack(language)
    globs = lp.source_globs
    if len(globs) == 1:
        name_filter = f"-name '{globs[0]}'"
    else:
        name_filter = "\\( " + " -o ".join(f"-name '{g}'" for g in globs) + " \\)"
    exclude_find = " -not -path '*/vendor/*'" if language == "go" else ""

    parts: list[str] = []

    # 1) Top-level directory layout
    dir_tree = run_command(
        "find . -maxdepth 2 -type d | sort | head -100",
        env, cwd=cwd, timeout=10,
    )
    if dir_tree["returncode"] == 0 and dir_tree["output"].strip():
        parts.append("### Directory layout (depth 2)")
        parts.append(dir_tree["output"].strip())

    # 2) Test directories — surface them explicitly
    test_dirs = run_command(
        "find . -maxdepth 3 -type d \\( -name tests -o -name test \\) | sort",
        env, cwd=cwd, timeout=10,
    )
    if test_dirs["returncode"] == 0 and test_dirs["output"].strip():
        parts.append("\n### Test directories")
        parts.append(test_dirs["output"].strip())
        # List source files inside test directories (up to 100)
        if language == "python":
            tf_cmd = "find . -maxdepth 4 -path '*/test*/*.py' -type f | sort | head -100"
        else:
            tf_cmd = f"find . -maxdepth 4 -path '*/test*' -type f {name_filter}{exclude_find} | sort | head -100"
        test_files = run_command(tf_cmd, env, cwd=cwd, timeout=10)
        if test_files["returncode"] == 0 and test_files["output"].strip():
            parts.append(test_files["output"].strip())

    # 3) All source files (increased budget from 200 → 500)
    all_src = run_command(
        f"find . -maxdepth {max_depth} -type f {name_filter}{exclude_find} | sort | head -500",
        env, cwd=cwd, timeout=10,
    )
    if all_src["returncode"] == 0 and all_src["output"].strip():
        parts.append(f"\n### {lp.name.capitalize()} files")
        parts.append(all_src["output"].strip())

    return "\n".join(parts) if parts else "(failed to list files)"


def _read_file(env: Environment, cwd: str, rel_path: str, max_lines: int = 500) -> str:
    """Read a file from the repo, truncated to max_lines."""
    result = run_command(f"head -n {max_lines} '{rel_path}' 2>/dev/null", env, cwd=cwd, timeout=10)
    if result["returncode"] != 0:
        return ""
    content = result["output"]
    # Check if file was truncated
    wc = run_command(f"wc -l < '{rel_path}' 2>/dev/null", env, cwd=cwd, timeout=5)
    if wc["returncode"] == 0:
        total = wc["output"].strip()
        try:
            if int(total) > max_lines:
                content += f"\n... ({int(total) - max_lines} more lines)"
        except ValueError:
            pass
    return content


def _parse_localization(response: str) -> dict:
    """Parse the structured localization response."""
    result = {"relevant_files": [], "test_files": [], "focal_functions": []}
    current_section = None

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("RELEVANT_FILES:"):
            current_section = "relevant_files"
            continue
        elif line.startswith("TEST_FILES:"):
            current_section = "test_files"
            continue
        elif line.startswith("FOCAL_FUNCTIONS:"):
            current_section = "focal_functions"
            continue

        if current_section and line and not line.startswith("#"):
            cleaned = line.lstrip("- ").strip("`").strip()
            if cleaned:
                result[current_section].append(cleaned)

    return result
