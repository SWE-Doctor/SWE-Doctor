"""Tests for build_prompt pure function in reproduction_test_agent.generator."""

from reproduction_test_agent.generator import build_prompt


def test_build_prompt_with_aspect_contains_focus_header_and_description():
    result = build_prompt("issue text", "none", {}, aspect_description="X")
    assert "FOCUS FOR THIS TEST" in result
    assert "X" in result


def test_build_prompt_without_aspect_omits_focus_header():
    result = build_prompt("issue text", "none", {})
    assert "FOCUS FOR THIS TEST" not in result


def test_build_prompt_sanity_no_aspect():
    result = build_prompt("issue text", "none", {})
    assert "issue text" in result
    assert "Write the reproduction test:" in result
