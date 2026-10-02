"""e-Otter++ pipeline: heterogeneous prompting + execution-augmented test repair.

Full flow:
  1. Localize relevant code (once)
  2. For each of 5 morphs: morph issue → generate test (with "full" mask)
  3. For each of 5 masks: generate test with original issue
  4. For each of 10 candidates: run execution-augmented repair
  5. Filter: keep only tests where critic says "fails for right reason"
"""

import json
import logging
import re
import time
from pathlib import Path, PurePosixPath

from .config import ReproTestConfig
from .executor import Environment
from .llm import llm_call, set_trace_file
from .localizer import localize
from .morphs import apply_morph
from .generator import generate_test
from .repair import execution_augmented_repair
from .aspect_extractor import split_into_aspects
from .bundle import generate_bundle
from .critic import select_per_aspect

logger = logging.getLogger("repro_test.pipeline")


def _derive_go_test_filename(localization: dict) -> str:
    """Go repro tests must live in the package-under-test's directory (so
    `go test` discovers them and same-package tests reach unexported symbols).
    Prefer the directory of an existing localized test — that's the package
    whose behavior is actually exercised (mirrors the JS adaptation, and keeps
    the LLM on existing symbols instead of a first-listed source file in an
    unrelated package). Fall back to the first non-test source, then root."""
    loc = localization or {}
    for t in loc.get("test_files") or []:
        rel = str(t).strip().lstrip("./")
        if rel.endswith("_test.go"):
            return str(PurePosixPath(rel).parent / "zzz_repro_test.go")
    for f in loc.get("relevant_files") or []:
        rel = str(f).strip().lstrip("./")
        if rel.endswith(".go") and not rel.endswith("_test.go"):
            return str(PurePosixPath(rel).parent / "zzz_repro_test.go")
    return "zzz_repro_test.go"


def _go_package_of(localization: dict, src_file: str) -> str:
    content = (localization.get("file_contents") or {}).get(src_file, "")
    m = re.search(r"^package\s+(\w+)", content, re.MULTILINE)
    return m.group(1) if m else ""


def _pick_go_example_test(test_dir: str, test_contents: dict) -> tuple[str, list[str]] | None:
    """Pick a real existing _test.go (prefer same dir) and return its package
    clause + import header, for the LLM to copy verbatim."""
    def _header(content: str) -> list[str]:
        out: list[str] = []
        in_import = False
        for ln in content.splitlines():
            s = ln.strip()
            if s.startswith("package "):
                out.append(ln)
            elif s.startswith("import ("):
                out.append(ln); in_import = True
            elif in_import:
                out.append(ln)
                if s == ")":
                    break
            elif s.startswith("import "):
                out.append(ln)
            if len(out) > 24:
                break
        return out

    ranked = sorted(
        test_contents.items(),
        key=lambda kv: 0 if str(PurePosixPath(kv[0]).parent) == test_dir else 1,
    )
    for f, content in ranked:
        hdr = _header(content)
        if any(l.strip().startswith("package ") for l in hdr):
            return f, hdr
    return None


def _build_go_gen_hint(test_filename: str, localization: dict) -> str:
    """Go-only generation guidance: the exact save path, the package to declare
    (same package as the code under test, for unexported access), and the
    package/import header of a real nearby test to copy. Empty for Python."""
    test_dir = str(PurePosixPath(test_filename).parent)
    loc_part = (f" (package directory `{test_dir}/`)." if test_dir != "."
                else " (repo-root package).")
    lines = [
        "## Test File Location & Package (CRITICAL)",
        f"Your test file will be saved at: `{test_filename}`" + loc_part,
    ]
    pkg = ""
    # Prefer a source file in the SAME directory as the test so the declared
    # package matches where the test lands; then fall back to any source.
    for prefer_same_dir in (True, False):
        for f in (localization.get("relevant_files") or []):
            rel = str(f).strip().lstrip("./")
            if not (rel.endswith(".go") and not rel.endswith("_test.go")):
                continue
            if prefer_same_dir and str(PurePosixPath(rel).parent) != test_dir:
                continue
            pkg = _go_package_of(localization, f) or _go_package_of(localization, rel)
            if pkg:
                break
        if pkg:
            break
    if pkg:
        lines.append(
            f"Declare `package {pkg}` — the SAME package as the code under test — so your "
            f"test can call its unexported (lowercase) symbols directly with NO import.")
    else:
        lines.append(
            "Declare the SAME `package` as the other .go files in that directory (NOT an "
            "external `_test` package), so you can reach unexported symbols directly.")
    example = _pick_go_example_test(test_dir, localization.get("test_contents") or {})
    if example:
        ef, header = example
        lines.append(
            f"\nAn existing REAL test (`{ef}`) in that area sets up its package & imports "
            "like this — copy this EXACT package clause and import style:")
        lines.append("```go")
        lines.extend(header)
        lines.append("```")
    lines.append(
        "\nOnly call symbols that ALREADY EXIST in the current (buggy) code; do NOT call a "
        "function the issue says the fix will ADD. Same-package symbols need no import; for "
        "other packages import by their module path (from go.mod).")
    lines.append(
        "Go compiles strictly: an unused import or unused variable is a COMPILE ERROR — "
        "import only what you reference (discard unused returns with `_`). Assert the FIXED "
        "behavior so the test FAILS on the current buggy code; do NOT assert the current "
        "buggy result.")
    return "\n".join(lines)


def run_pipeline(
    instance_id: str,
    problem_statement: str,
    env: Environment,
    cwd: str = "/app",
    config: ReproTestConfig | None = None,
) -> dict:
    """Run the full e-Otter++ reproduction test generation pipeline.

    Args:
        instance_id: SWE-bench instance identifier
        problem_statement: The issue description text
        env: Execution environment (local or Docker)
        cwd: Working directory inside the environment
        config: Pipeline configuration

    Returns:
        dict with:
            instance_id: str
            candidates: list[dict] — all generated candidates with repair info
            accepted: list[dict] — candidates where critic says "fails for right reason"
            best_test: str | None — the best test code (first accepted), or None
    """
    config = config or ReproTestConfig()
    start = time.time()
    # Trace file will be set when output_dir is known (by the caller via set_trace_file)
    # but as a fallback, set one based on instance_id in cwd
    logger.info("=== Pipeline start: %s ===", instance_id)

    # Step 1: Localization (shared across all prompts)
    logger.info("Step 1: Localizing relevant code...")
    language = config.language
    localization = localize(problem_statement, env, cwd, config.model_name, language=language)

    # Per-language test filename + generation hint. Go needs a same-package path
    # (so the test reaches unexported symbols) and a package/import hint; python
    # keeps repro_test.py with no hint (gen_hint="" leaves prompts unchanged).
    if language == "go":
        test_filename = _derive_go_test_filename(localization)
        gen_hint = _build_go_gen_hint(test_filename, localization)
    else:
        from .langpack import get_langpack
        test_filename = get_langpack(language).test_filename
        gen_hint = ""

    # Step 2+3: Generate candidates via heterogeneous prompting
    candidates = []

    if config.aspects:
        # Aspect-mode: split issue into aspects, then fan out aspect × mask
        logger.info("Step 2: Aspect-mode — splitting issue into aspects...")

        def _llm_adapter(prompt: str) -> str:
            return llm_call(
                messages=[{"role": "user", "content": prompt}],
                model=config.model_name,
                temperature=config.generation_temperature,
                caller="aspect_extractor",
            )

        aspects = split_into_aspects(
            problem_statement,
            max_aspects=config.max_aspects,
            llm=_llm_adapter,
        )
        logger.info("Got %d aspects", len(aspects))

        bundle = generate_bundle(
            aspects=aspects,
            masks=config.aspect_masks,
            issue_text=problem_statement,
            localization=localization,
            model=config.model_name,
            temperature=config.generation_temperature,
            language=language,
            gen_hint=gen_hint,
        )

        # Normalise bundle candidates to the shape expected by the repair loop
        for b in bundle:
            candidates.append({
                "prompt_type": "aspect",
                "variant": b["candidate_id"],
                "issue_text": problem_statement,
                "initial_test": b["test_code"],
                "aspect_id": b["aspect_id"],
                "aspect_description": b["aspect_description"],
                "mask": b["mask"],
            })
    else:
        # Legacy morph × mask path (unchanged)
        # Morph variants: morph the issue, use "full" mask
        for morph_name in config.morphs:
            logger.info("Generating candidate: morph=%s", morph_name)
            try:
                morphed_issue = apply_morph(
                    morph_name, problem_statement, config.model_name,
                    temperature=config.morph_temperature,
                )
                test_code = generate_test(
                    morphed_issue, mask="full", localization=localization,
                    model=config.model_name, temperature=config.generation_temperature,
                    language=language, gen_hint=gen_hint,
                )
                candidates.append({
                    "prompt_type": "morph",
                    "variant": morph_name,
                    "issue_text": morphed_issue,
                    "initial_test": test_code,
                })
            except Exception as e:
                logger.error("Failed morph=%s: %s", morph_name, e)
                candidates.append({
                    "prompt_type": "morph",
                    "variant": morph_name,
                    "issue_text": problem_statement,
                    "initial_test": "",
                    "error": str(e),
                })

        # Mask variants: original issue, different masks
        for mask_name in config.masks:
            logger.info("Generating candidate: mask=%s", mask_name)
            try:
                test_code = generate_test(
                    problem_statement, mask=mask_name, localization=localization,
                    model=config.model_name, temperature=config.generation_temperature,
                    language=language, gen_hint=gen_hint,
                )
                candidates.append({
                    "prompt_type": "mask",
                    "variant": mask_name,
                    "issue_text": problem_statement,
                    "initial_test": test_code,
                })
            except Exception as e:
                logger.error("Failed mask=%s: %s", mask_name, e)
                candidates.append({
                    "prompt_type": "mask",
                    "variant": mask_name,
                    "issue_text": problem_statement,
                    "initial_test": "",
                    "error": str(e),
                })

    # Step 4: Execution-augmented repair for each candidate
    logger.info("Step 4: Running execution-augmented repair on %d candidates...", len(candidates))

    for i, cand in enumerate(candidates):
        if not cand.get("initial_test") or cand.get("error"):
            cand["repair_result"] = None
            continue

        label = f"{cand['prompt_type']}:{cand['variant']}"
        logger.info("Repairing candidate %d/%d (%s)...", i + 1, len(candidates), label)

        try:
            repair_result = execution_augmented_repair(
                issue_text=cand["issue_text"],
                test_code=cand["initial_test"],
                env=env,
                cwd=cwd,
                model=config.model_name,
                max_attempts=config.max_repair_attempts,
                repair_temperature=config.repair_temperature,
                test_timeout=config.test_timeout,
                language=language,
                test_filename=test_filename,
                gen_hint=gen_hint,
            )
            cand["repair_result"] = repair_result
            cand["final_test"] = repair_result["test_code"]
            cand["fails_for_right_reason"] = repair_result["fails_for_right_reason"]
            cand["repair_attempts"] = repair_result["attempts"]
        except Exception as e:
            logger.error("Repair failed for %s: %s", label, e)
            cand["repair_result"] = None
            cand["final_test"] = cand["initial_test"]
            cand["fails_for_right_reason"] = False

    # Step 5: Filter / select accepted candidates
    _CATEGORY_SCORE = {"assertion_failure": 2, "other_failure": 1}

    if config.aspects:
        # Build evaluated list for select_per_aspect
        evaluated = []
        for c in candidates:
            if not c.get("repair_result"):
                continue
            history = c["repair_result"].get("history", [])
            failure_category = (
                history[-1].get("critique", {}).get("failure_category", "error")
                if history else "error"
            )
            fails = c.get("fails_for_right_reason", False)
            score = _CATEGORY_SCORE.get(failure_category, 0) if fails else 0
            evaluated.append({
                "aspect_id": c["aspect_id"],
                "aspect_description": c["aspect_description"],
                "final_test": c.get("final_test", c.get("initial_test", "")),
                "critic_ok": fails,
                "score": score,
                # carry through for downstream use
                "mask": c.get("mask"),
                "variant": c.get("variant"),
            })

        accepted = select_per_aspect(evaluated)
        aspects_covered = len(accepted)
    else:
        accepted = [c for c in candidates if c.get("fails_for_right_reason")]

        # Sort by failure category priority (assertion_failure > other_failure > error)
        _CATEGORY_PRIORITY = {"assertion_failure": 0, "other_failure": 1, "error": 2}

        def _candidate_sort_key(c):
            history = c.get("repair_result", {}).get("history", [])
            if history:
                cat = history[-1].get("critique", {}).get("failure_category", "error")
            else:
                cat = "error"
            return _CATEGORY_PRIORITY.get(cat, 2)

        accepted.sort(key=_candidate_sort_key)
        aspects_covered = len(accepted)

    elapsed = time.time() - start
    logger.info(
        "=== Pipeline done: %s — %d/%d accepted in %.1fs ===",
        instance_id, len(accepted), len(candidates), elapsed,
    )

    best_test = accepted[0]["final_test"] if accepted else None

    result = {
        "instance_id": instance_id,
        "language": language,
        "test_relpath": test_filename,
        "candidates": candidates,
        "accepted": accepted,
        "best_test": best_test,
        "localization": {k: v for k, v in localization.items()
                         if k not in ("file_contents", "test_contents")},
        "elapsed_seconds": elapsed,
        "aspects_covered": aspects_covered,
    }
    return result


def save_results(results: dict, output_dir: str) -> Path:
    """Save pipeline results to JSON file.

    Produces two files per instance:
      {instance_id}.json          — slim summary (no full repair history messages)
      {instance_id}_full.json     — full detail including every repair iteration's
                                    test code, execution output, and critic feedback
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Full results (with complete repair histories) ──
    full_path = out / f"{results['instance_id']}_full.json"
    full = {**results}
    full_candidates = []
    for c in results["candidates"]:
        fc = {k: v for k, v in c.items()}
        rr = c.get("repair_result")
        if rr:
            fc["repair_attempts"] = rr["attempts"]
            fc["repair_history"] = rr.get("history", [])
        full_candidates.append(fc)
    full["candidates"] = full_candidates
    full_path.write_text(json.dumps(full, indent=2, default=str))
    logger.info("Full results saved to %s", full_path)

    # ── Slim results (backward-compatible, no bulky histories) ──
    path = out / f"{results['instance_id']}.json"
    slim = {**results}
    slim_candidates = []
    for c in results["candidates"]:
        sc = {k: v for k, v in c.items() if k != "repair_result"}
        if c.get("repair_result"):
            sc["repair_attempts"] = c["repair_result"]["attempts"]
        slim_candidates.append(sc)
    slim["candidates"] = slim_candidates
    path.write_text(json.dumps(slim, indent=2, default=str))
    logger.info("Results saved to %s", path)

    # Also save each accepted test as a standalone file. Go needs the .go
    # extension and its package-relative path preserved (sidecar .relpath) so
    # Stage-2 can drop the test back into the package directory it belongs to.
    lang = results.get("language", "python")
    ext = "go" if lang == "go" else "py"
    for i, cand in enumerate(results.get("accepted", [])):
        test_path = out / f"{results['instance_id']}_test_{i}.{ext}"
        test_path.write_text(cand["final_test"])
        if lang == "go" and results.get("test_relpath"):
            (out / f"{results['instance_id']}_test_{i}.relpath").write_text(results["test_relpath"])

    return path
