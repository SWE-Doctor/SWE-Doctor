"""Issue description morphs: rewrite the issue text in 5 different ways.

Each morph gives the LLM a different perspective on the issue,
increasing diversity of generated tests.

Morphs:
  standard  — standardize into title/description/steps/expected/actual
  simple    — simplify, remove jargon
  dropCode  — remove code snippets (which may be misleading)
  initTest  — ask LLM to propose an initial test and embed in description
  initPatch — ask LLM to propose an initial fix and embed in description
"""

import logging
from .llm import llm_call

logger = logging.getLogger("repro_test.morphs")


MORPH_PROMPTS = {
    "standard": """\
Rewrite the following software issue description into a well-structured standard format with these sections:
- **Title**: A concise title
- **Description**: Clear description of the problem
- **Steps to Reproduce**: Numbered steps to trigger the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What happens instead

Keep all technical details intact. Do not add information that is not in the original.

## Original Issue
{issue_text}

## Rewritten Issue (standard format)
""",

    "simple": """\
Simplify the following software issue description. Remove project-specific jargon, \
abbreviations, and overly technical language. Make it understandable to a developer \
who does not work on this project regularly. Preserve the core problem description.

## Original Issue
{issue_text}

## Simplified Issue
""",

    "dropCode": """\
Rewrite the following software issue description, but REMOVE all code snippets, \
code blocks, and inline code references. Keep only the natural language description \
of the problem. Developers sometimes include misleading code snippets that confuse \
automated test generation — the natural language description is often more reliable.

## Original Issue
{issue_text}

## Issue (without code)
""",

    "initTest": """\
Read the following software issue description. Then:
1. Propose a brief initial fail-to-pass test idea (what to test, expected vs actual).
2. Rewrite the issue incorporating this test idea at the end.

The test should fail on the buggy code and pass after the fix. \
Do not write actual test code — just describe the test scenario in plain English.

## Original Issue
{issue_text}

## Issue with Proposed Test Scenario
""",

    "initPatch": """\
Read the following software issue description. Then:
1. Propose a brief initial fix idea (what code change would resolve this).
2. Rewrite the issue incorporating this fix idea at the end.

Do not write actual patch code — just describe the fix approach in plain English.

## Original Issue
{issue_text}

## Issue with Proposed Fix Approach
""",
}


def apply_morph(morph_name: str, issue_text: str, model: str, temperature: float = 0.0) -> str:
    """Apply an issue description morph. Returns the rewritten issue text."""
    if morph_name not in MORPH_PROMPTS:
        raise ValueError(f"Unknown morph: {morph_name}. Available: {list(MORPH_PROMPTS)}")

    prompt = MORPH_PROMPTS[morph_name].format(issue_text=issue_text)
    result = llm_call(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        caller=f"morph:{morph_name}",
    )
    logger.info("Applied morph '%s' (%d → %d chars)", morph_name, len(issue_text), len(result))
    return result
