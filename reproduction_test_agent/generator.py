"""Test generation with context masks.

5 masks control what localization info goes into the test generation prompt:
  planner  — full localization (files + tests + functions + file contents)
  full     — same as planner but no action planning preamble
  testLoc  — only test file locations and contents
  patchLoc — only relevant source file locations and contents
  none     — no localization info, just the issue description

Each mask × issue pair produces one candidate test.
"""

import logging
from .llm import llm_call
from .utils import extract_code
from .langpack import get_langpack

logger = logging.getLogger("repro_test.generator")


def _build_context(mask: str, localization: dict, language: str = "python") -> str:
    """Build the context string based on the mask type."""
    if mask == "none":
        return ""

    fence = get_langpack(language).code_fence
    parts = []

    if mask in ("planner", "full", "patchLoc"):
        if localization.get("relevant_files"):
            parts.append("## Relevant Source Files")
            for f in localization["relevant_files"]:
                parts.append(f"- {f}")
                content = localization.get("file_contents", {}).get(f, "")
                if content:
                    parts.append(f"```{fence}\n# {f}\n{content}\n```")

        if localization.get("focal_functions"):
            parts.append("\n## Focal Functions")
            for fn in localization["focal_functions"]:
                parts.append(f"- {fn}")

    if mask in ("planner", "full", "testLoc"):
        if localization.get("test_files"):
            parts.append("\n## Existing Related Tests")
            for f in localization["test_files"]:
                parts.append(f"- {f}")
                content = localization.get("test_contents", {}).get(f, "")
                if content:
                    parts.append(f"```{fence}\n# {f}\n{content}\n```")

    if mask == "planner":
        parts.insert(0, (
            "## Localization Summary\n"
            "The following files and functions were identified as relevant to this issue "
            "through automated bug localization. Use this information to write a targeted test.\n"
        ))

    return "\n".join(parts)


def build_prompt(
    issue: str,
    mask: str,
    localization: dict,
    aspect_description: str | None = None,
    language: str = "python",
    gen_hint: str = "",
) -> str:
    """Build the user prompt string for test generation (pure, no LLM call).

    Args:
        issue: The issue description text.
        mask: One of "planner", "full", "testLoc", "patchLoc", "none"
        localization: Dict from localizer.localize()
        aspect_description: Optional focus hint; when provided, inserts a
            "FOCUS FOR THIS TEST" section into the prompt.
        language: Programming language ("python" or "go"); selects the code fence.

    Returns:
        The fully assembled user-prompt string.
    """
    context = _build_context(mask, localization, language)

    prompt = f"## Issue Description\n{issue}"
    if aspect_description:
        prompt += f"\n\n## FOCUS FOR THIS TEST\n{aspect_description}"
    if context:
        prompt += f"\n\n{context}"
    if gen_hint:
        prompt += f"\n\n{gen_hint}"
    prompt += "\n\nWrite the reproduction test:"
    return prompt


def generate_test_for_aspect(
    issue_text: str,
    mask: str,
    localization: dict,
    aspect_description: str,
    model: str,
    temperature: float = 0.0,
    language: str = "python",
    gen_hint: str = "",
) -> str:
    """Generate a candidate test focused on a specific aspect.

    Same as generate_test but passes aspect_description to build_prompt.
    """
    user_prompt = build_prompt(issue_text, mask, localization,
                               aspect_description=aspect_description, language=language,
                               gen_hint=gen_hint)

    result = llm_call(
        messages=[
            {"role": "system", "content": get_langpack(language).system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=4096,
        caller=f"generator:mask={mask}:aspect={aspect_description[:20]}",
    )

    test_code = extract_code(result)
    logger.info("Generated test with mask '%s' aspect '%s' (%d chars)", mask, aspect_description[:20], len(test_code))
    return test_code


def generate_test(
    issue_text: str,
    mask: str,
    localization: dict,
    model: str,
    temperature: float = 0.0,
    language: str = "python",
    gen_hint: str = "",
) -> str:
    """Generate a single candidate test using the specified mask.

    Args:
        issue_text: The issue description (possibly morphed)
        mask: One of "planner", "full", "testLoc", "patchLoc", "none"
        localization: Dict from localizer.localize()
        model: LLM model name
        temperature: Generation temperature
        language: Programming language ("python" or "go")

    Returns:
        Generated test code as a string.
    """
    user_prompt = build_prompt(issue_text, mask, localization, language=language,
                               gen_hint=gen_hint)

    result = llm_call(
        messages=[
            {"role": "system", "content": get_langpack(language).system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=4096,
        caller=f"generator:mask={mask}",
    )

    test_code = extract_code(result)
    logger.info("Generated test with mask '%s' (%d chars)", mask, len(test_code))
    return test_code
