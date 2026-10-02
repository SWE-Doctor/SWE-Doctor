"""Fan out (aspect × mask) to produce candidate test bundles."""
from __future__ import annotations

from typing import Callable, Iterable

from .aspect_extractor import IssueAspect
from .generator import generate_test_for_aspect


def generate_bundle(
    aspects: list[IssueAspect],
    masks: Iterable[str],
    issue_text: str,
    localization: dict,
    generate_fn: Callable | None = None,
    model: str = "",
    temperature: float = 0.0,
    language: str = "python",
    gen_hint: str = "",
) -> list[dict]:
    """Fan out aspects × masks into candidate dicts.

    Args:
        aspects: List of IssueAspect objects.
        masks: Iterable of mask strings.
        issue_text: The issue description text.
        localization: Dict from localizer.localize().
        generate_fn: Optional callable(issue_text, mask, localization, aspect_description, model, temperature) -> str.
                     Defaults to generate_test_for_aspect.
        model: LLM model name.
        temperature: Generation temperature.

    Returns:
        List of candidate dicts, one per (aspect, mask) pair.
    """
    if generate_fn is None:
        generate_fn = generate_test_for_aspect

    candidates = []
    masks = list(masks)
    for aspect in aspects:
        for mask in masks:
            test_code = generate_fn(issue_text, mask, localization, aspect.description,
                                    model, temperature, language=language, gen_hint=gen_hint)
            candidates.append({
                "candidate_id": f"{aspect.aspect_id}-{mask}",
                "aspect_id": aspect.aspect_id,
                "aspect_description": aspect.description,
                "mask": mask,
                "test_code": test_code,
            })
    return candidates
