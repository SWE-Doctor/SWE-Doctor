"""Tests for reproduction_test_agent/bundle.py — aspect × mask fan-out."""
import pytest
from reproduction_test_agent.aspect_extractor import IssueAspect
from reproduction_test_agent.bundle import generate_bundle


ASPECTS = [
    IssueAspect("a0", "First aspect description"),
    IssueAspect("a1", "Second aspect description"),
]
MASKS = ["none", "patchLoc"]
ISSUE = "Some issue text"
LOCALIZATION = {}


def fake_generate_fn(issue_text, mask, localization, aspect_description, model,
                     temperature, language="python", gen_hint=""):
    return f"# {mask}-{aspect_description}"


def test_fan_out_count():
    candidates = generate_bundle(
        ASPECTS, MASKS, ISSUE, LOCALIZATION, generate_fn=fake_generate_fn
    )
    assert len(candidates) == 4


def test_fan_out_ids():
    candidates = generate_bundle(
        ASPECTS, MASKS, ISSUE, LOCALIZATION, generate_fn=fake_generate_fn
    )
    ids = {c["candidate_id"] for c in candidates}
    assert ids == {"a0-none", "a0-patchLoc", "a1-none", "a1-patchLoc"}


def test_candidate_keys():
    candidates = generate_bundle(
        ASPECTS, MASKS, ISSUE, LOCALIZATION, generate_fn=fake_generate_fn
    )
    for c in candidates:
        assert "candidate_id" in c
        assert "aspect_id" in c
        assert "aspect_description" in c
        assert "mask" in c
        assert "test_code" in c


def test_candidate_values():
    candidates = generate_bundle(
        ASPECTS, MASKS, ISSUE, LOCALIZATION, generate_fn=fake_generate_fn
    )
    by_id = {c["candidate_id"]: c for c in candidates}

    c = by_id["a0-none"]
    assert c["aspect_id"] == "a0"
    assert c["aspect_description"] == "First aspect description"
    assert c["mask"] == "none"
    assert c["test_code"] == "# none-First aspect description"

    c = by_id["a1-patchLoc"]
    assert c["aspect_id"] == "a1"
    assert c["mask"] == "patchLoc"
    assert c["test_code"] == "# patchLoc-Second aspect description"
