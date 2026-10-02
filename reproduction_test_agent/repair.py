"""Execution-augmented test repair loop (Algorithm 1 from e-Otter++ paper).

Loop:
  1. Execute test on c_old
  2. Critic: does it fail for the right reason?
  3. If yes → done
  4. If no → gather extra context (read functions critic requested) → repair test
  5. Repeat up to max_attempts
"""

import logging
import re

from .llm import llm_call
from .executor import run_test, run_command, Environment
from .critic import critique_test, critique_passing_test
from .utils import extract_code

logger = logging.getLogger("repro_test.repair")


REPAIR_PROMPT = """\
You are repairing a reproduction test for a software bug. The test was executed on \
the buggy codebase but it is NOT failing for the right reason. Fix the test so that \
it correctly reproduces the issue.

## Issue Description
{issue_text}

## Current Test Code
```{code_fence}
{test_code}
```

## Execution Log
```
{execution_log}
```

## Critic Feedback
- Failure category: {failure_category}
- Reasoning: {reasoning}
- Possible buggy line: {buggy_line}
- Relevant snippet from issue: {relevant_snippet}

## Additional Context from Codebase
{extra_context}

## Instructions
- Fix the test so it fails because of the actual bug described in the issue
- Common problems: wrong imports (ModuleNotFoundError), wrong function signatures (TypeError), \
wrong class/attribute names, missing setup
- Use the additional context to get correct function signatures, argument types, return types
- Keep the test focused on the described issue

Output ONLY the corrected {lang_hint} test code, no explanations. Start with imports.
"""


def execution_augmented_repair(
    issue_text: str,
    test_code: str,
    env: Environment,
    cwd: str,
    model: str,
    max_attempts: int = 10,
    repair_temperature: float = 0.8,
    test_timeout: int = 60,
    language: str = "python",
    test_filename: str | None = None,
    gen_hint: str = "",
) -> dict:
    """Run the execution-augmented test repair loop.

    Returns dict with:
        test_code: str — the final (possibly repaired) test
        fails_for_right_reason: bool
        attempts: int — how many repair iterations were used
        history: list[dict] — per-iteration info
    """
    if test_filename is None:
        from .langpack import get_langpack
        test_filename = get_langpack(language).test_filename
    history = []

    for attempt in range(max_attempts):
        # Step 1: Execute test on c_old
        exec_result = run_test(test_code, env, cwd=cwd, timeout=test_timeout,
                               test_filename=test_filename, language=language)
        logger.info(
            "Repair attempt %d/%d: passed=%s, error_type=%s",
            attempt + 1, max_attempts, exec_result["passed"], exec_result["error_type"],
        )

        if exec_result["passed"]:
            critique = critique_passing_test(
                issue_text=issue_text,
                test_code=test_code,
                execution_log=exec_result["output"],
                model=model,
                language=language,
            )
        else:
            # Step 2: Critic evaluates the failure
            critique = critique_test(
                issue_text=issue_text,
                test_code=test_code,
                execution_log=exec_result["output"],
                model=model,
                language=language,
            )

        history.append({
            "attempt": attempt + 1,
            "test_code": test_code,
            "passed": exec_result["passed"],
            "critique": critique,
            "execution_output": exec_result["output"][:2000],
        })

        # Step 3: If fails for right reason → done
        if critique["fails_for_right_reason"]:
            logger.info("Test fails for the right reason after %d attempt(s)", attempt + 1)
            return {
                "test_code": test_code,
                "fails_for_right_reason": True,
                "attempts": attempt + 1,
                "history": history,
            }

        # Step 4: Gather extra context — the critic's requested functions PLUS
        # symbols auto-extracted from go compile errors. The LLM critic reliably
        # leaves functions_to_read empty for go build failures (undefined / no
        # field or method / type mismatch), so without this the repair never
        # sees the real definition of the symbol it misused and loops on the
        # same error. Feeding the actual func/type definition lets it converge.
        funcs = list(critique.get("functions_to_read") or [])
        if language == "go":
            for sym in _go_error_symbols(exec_result["output"]):
                if sym not in funcs:
                    funcs.append(sym)
        extra_context = _gather_extra_context(funcs, env, cwd, language=language)

        # Step 5: Repair the test
        test_code = _repair_test(
            issue_text=issue_text,
            test_code=test_code,
            execution_log=exec_result["output"],
            critique=critique,
            extra_context=extra_context,
            model=model,
            temperature=repair_temperature,
            language=language,
            gen_hint=gen_hint,
        )

    # Exhausted all attempts
    logger.warning("Repair loop exhausted %d attempts without success", max_attempts)
    return {
        "test_code": test_code,
        "fails_for_right_reason": False,
        "attempts": max_attempts,
        "history": history,
    }


_GO_ERR_PATTERNS = [
    r"undefined:\s*([A-Za-z_][\w.]*)",                         # undefined: X / pkg.X
    r"has no field or method\s+([A-Za-z_]\w*)",                # ...has no field or method Y
    r"\(type\s+([A-Za-z_][\w.]*)\s+has no field or method",    # type Z has no field or method
    r"undeclared name:\s*([A-Za-z_]\w*)",                      # undeclared name: X
    r"\(missing method\s+([A-Za-z_]\w*)\)",                    # does not implement B (missing method M)
    r"does not implement\s+([A-Za-z_][\w.]*)",                 # A does not implement B
    r"as\s+([A-Za-z_][\w.]*)\s+value in",                      # cannot use ... as T value in ...
]


def _go_error_symbols(execution_log: str) -> list[str]:
    """Extract the undefined / mismatched symbol names from a Go build failure
    so their real definitions can be fed back to the repair (the LLM critic does
    not reliably surface them). Package qualifiers are dropped (pkg.Foo -> Foo)
    so the func/type grep can find the definition."""
    out: list[str] = []
    for pat in _GO_ERR_PATTERNS:
        for m in re.findall(pat, execution_log):
            name = m.split(".")[-1]
            if name and name[0].isalpha() and name not in out:
                out.append(name)
    return out[:6]


def _gather_extra_context(functions_to_read: list[str], env: Environment, cwd: str,
                          language: str = "python") -> str:
    """Use grep to find and read the functions/types the critic (or the go error
    extractor) requested."""
    if not functions_to_read:
        return "(no additional context requested)"

    from .langpack import get_langpack
    lp = get_langpack(language)
    glob = lp.source_globs[0]
    parts = []
    not_found = []
    for func_name in functions_to_read[:6]:
        if language == "go":
            # Match a plain func, a method (receiver), OR a type definition —
            # go build errors are often about types/methods, not bare funcs.
            pat = rf"(func (\([^)]*\) )?{func_name}\b|type {func_name}\b)"
        else:
            pat = lp.func_def_grep.format(name=func_name)
        result = run_command(
            f"grep -rnE '{pat}' --include='{glob}' | head -5",
            env, cwd=cwd,
        )
        if result["returncode"] == 0 and result["output"].strip():
            parts.append(f"### Function: {func_name}")
            for match_line in result["output"].strip().splitlines()[:3]:
                if ":" in match_line:
                    filepath, rest = match_line.split(":", 1)
                    if ":" in rest:
                        lineno_str = rest.split(":", 1)[0]
                        try:
                            lineno = int(lineno_str)
                            ctx = run_command(
                                f"sed -n '{max(1,lineno-2)},{lineno+30}p' '{filepath}'",
                                env, cwd=cwd,
                            )
                            if ctx["output"].strip():
                                parts.append(f"```{lp.code_fence}\n# {filepath}:{lineno}\n{ctx['output']}```")
                        except ValueError:
                            pass
        elif language == "go":
            not_found.append(func_name)

    if language == "go" and not_found:
        # A symbol the test referenced has NO definition in the repo — it is
        # almost always something the FIX would add (the test runs on the
        # UNFIXED code). Tell the model to stop inventing it and exercise the
        # behavior through an API that already exists on the current tree.
        parts.append(
            "### These symbols do NOT exist in the current codebase: "
            + ", ".join(not_found)
            + "\nThe test runs on the UNFIXED code, so it must COMPILE using only "
            "symbols that already exist. Do NOT reference the above — instead "
            "exercise the buggy behavior through an existing exported API "
            "(grep the repo for the real entry point) and assert the wrong "
            "current behavior."
        )

    return "\n".join(parts) if parts else "(functions not found in codebase)"


def _repair_test(
    issue_text: str,
    test_code: str,
    execution_log: str,
    critique: dict,
    extra_context: str,
    model: str,
    temperature: float = 0.8,
    language: str = "python",
    gen_hint: str = "",
) -> str:
    """Call the LLM to repair the test based on critic feedback."""
    from .langpack import get_langpack
    lp = get_langpack(language)
    if len(execution_log) > 6000:
        execution_log = execution_log[:3000] + "\n...[truncated]...\n" + execution_log[-3000:]

    prompt = REPAIR_PROMPT.format(
        code_fence=lp.code_fence,
        lang_hint=lp.repair_prompt_lang_hint,
        issue_text=issue_text,
        test_code=test_code,
        execution_log=execution_log,
        failure_category=critique.get("failure_category", "unknown"),
        reasoning=critique.get("reasoning", ""),
        buggy_line=critique.get("buggy_line", "unknown"),
        relevant_snippet=critique.get("relevant_snippet", ""),
        extra_context=extra_context,
    )
    if gen_hint:
        prompt += f"\n\n{gen_hint}"

    response = llm_call(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        max_tokens=4096,
        caller="repair",
    )

    return extract_code(response)
