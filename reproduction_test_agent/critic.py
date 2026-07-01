"""LLM-based critic for execution-augmented test repair.

The critic takes the issue description and the test execution log, and decides:
1. Does the test fail for the RIGHT reason (matching the issue description)?
2. If not, what additional context would help repair it?

The critic identifies:
- Whether the failure matches the described issue
- The possible buggy line in the codebase
- The most relevant code snippet from the issue
- Functions the model should read for context
"""

import logging
from .llm import llm_call
from .langpack import get_langpack

logger = logging.getLogger("repro_test.critic")


CRITIC_PROMPT = """\
You are a test repair critic. A reproduction test was generated for a software bug \
and executed on the buggy codebase. Your job is to analyze whether the test is \
failing for the RIGHT reason — i.e., the failure reproduces the bug described in \
the issue, not some unrelated error.

## Issue Description
{issue_text}

## Test Code
```{code_fence}
{test_code}
```

## Execution Log
```
{execution_log}
```

## Analysis Instructions

Consider these failure categories:
- **Assertion failure**: The test ran but the assertion failed (likely failing for the right reason)
- **Other failure**: The test ran but produced wrong output (may or may not be right reason)
- **Error**: The test did not run properly — e.g., ModuleNotFoundError, TypeError, ImportError, \
SyntaxError (most likely failing for the WRONG reason)

Respond in EXACTLY this format:

FAILS_FOR_RIGHT_REASON: yes/no
FAILURE_CATEGORY: assertion_failure/other_failure/error
REASONING: <1-2 sentences explaining your judgment>
BUGGY_LINE: <the line in the codebase most likely responsible, or "unknown">
RELEVANT_SNIPPET: <the most relevant code/text snippet from the issue description that the test should target>
FUNCTIONS_TO_READ: <comma-separated list of function names the repair should look at, or "none">
"""


def critique_test(
    issue_text: str,
    test_code: str,
    execution_log: str,
    model: str,
    language: str = "python",
) -> dict:
    """Evaluate whether a test fails for the right reason.

    Returns dict with keys:
        fails_for_right_reason: bool
        failure_category: str  ("assertion_failure", "other_failure", "error")
        reasoning: str
        buggy_line: str
        relevant_snippet: str
        functions_to_read: list[str]
    """
    # Truncate long execution logs
    if len(execution_log) > 8000:
        execution_log = execution_log[:4000] + "\n...[truncated]...\n" + execution_log[-4000:]

    prompt = CRITIC_PROMPT.format(
        issue_text=issue_text,
        test_code=test_code,
        execution_log=execution_log,
        code_fence=get_langpack(language).code_fence,
    )

    response = llm_call(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        caller="critic:failing_test",
    )

    result = _parse_critique(response)
    logger.info(
        "Critic: fails_for_right_reason=%s, category=%s",
        result["fails_for_right_reason"],
        result["failure_category"],
    )
    return result


PASSING_TEST_CRITIC_PROMPT = """\
You are a test repair critic. A reproduction test was generated for a software bug \
and executed on the buggy codebase, but the test PASSED when it should have FAILED.

The test should fail on the buggy code to prove the bug exists. Analyze why it passes \
and suggest what needs to change so it correctly exercises the buggy behavior.

## Issue Description
{issue_text}

## Test Code
```{code_fence}
{test_code}
```

## Execution Log (test PASSED)
```
{execution_log}
```

## Analysis Instructions
- The test passed, meaning it does NOT reproduce the bug. Figure out why.
- Common reasons: wrong assertion logic, testing wrong function/method, missing setup \
that would trigger the bug, catching exceptions that should propagate, etc.
- Suggest functions to read from the codebase that would help write a correct test.

Respond in EXACTLY this format:

FAILS_FOR_RIGHT_REASON: no
FAILURE_CATEGORY: pass
REASONING: <1-2 sentences explaining why the test passes when it should fail>
BUGGY_LINE: <the line in the codebase most likely responsible, or "unknown">
RELEVANT_SNIPPET: <the most relevant code/text snippet from the issue that the test should target>
FUNCTIONS_TO_READ: <comma-separated list of function names to inspect for writing a correct test, or "none">
"""


def critique_passing_test(
    issue_text: str,
    test_code: str,
    execution_log: str,
    model: str,
    language: str = "python",
) -> dict:
    """Analyze why a test passes on buggy code when it should fail."""
    if len(execution_log) > 8000:
        execution_log = execution_log[:4000] + "\n...[truncated]...\n" + execution_log[-4000:]

    prompt = PASSING_TEST_CRITIC_PROMPT.format(
        issue_text=issue_text,
        test_code=test_code,
        execution_log=execution_log,
        code_fence=get_langpack(language).code_fence,
    )

    response = llm_call(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        caller="critic:passing_test",
    )

    result = _parse_critique(response)
    result["fails_for_right_reason"] = False
    result["failure_category"] = "pass"
    logger.info("Passing-test critic: reasoning=%s", result["reasoning"])
    return result


def select_per_aspect(evaluated: list[dict]) -> list[dict]:
    """Return the highest-scoring critic_ok=True candidate per aspect_id.

    Aspects where all candidates are rejected (critic_ok=False) are dropped.
    Output order matches first occurrence of each aspect_id in the input.
    """
    order: list[str] = []
    groups: dict[str, list[dict]] = {}
    for entry in evaluated:
        aid = entry["aspect_id"]
        if aid not in groups:
            order.append(aid)
            groups[aid] = []
        groups[aid].append(entry)

    result = []
    for aid in order:
        accepted = [e for e in groups[aid] if e["critic_ok"]]
        if accepted:
            best = max(accepted, key=lambda e: e["score"])
            result.append(best)
    return result


def _parse_critique(response: str) -> dict:
    """Parse the structured critic response."""
    result = {
        "fails_for_right_reason": False,
        "failure_category": "error",
        "reasoning": "",
        "buggy_line": "",
        "relevant_snippet": "",
        "functions_to_read": [],
    }

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("FAILS_FOR_RIGHT_REASON:"):
            val = line.split(":", 1)[1].strip().lower()
            result["fails_for_right_reason"] = val in ("yes", "true", "1")
        elif line.startswith("FAILURE_CATEGORY:"):
            result["failure_category"] = line.split(":", 1)[1].strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
        elif line.startswith("BUGGY_LINE:"):
            result["buggy_line"] = line.split(":", 1)[1].strip()
        elif line.startswith("RELEVANT_SNIPPET:"):
            result["relevant_snippet"] = line.split(":", 1)[1].strip()
        elif line.startswith("FUNCTIONS_TO_READ:"):
            raw = line.split(":", 1)[1].strip()
            if raw.lower() != "none":
                result["functions_to_read"] = [f.strip() for f in raw.split(",") if f.strip()]

    return result
