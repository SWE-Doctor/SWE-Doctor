"""Tests for parse_regex_actions in models/utils/actions_text.py."""

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_text import parse_regex_actions

ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"
FORMAT_ERROR_TEMPLATE = "err {{actions|length}}"


def test_byte_identical_actions_deduped():
    """Byte-identical duplicate actions (transport double-flush artifact) collapse to one."""
    block = "THOUGHT: do thing\n\n```mswea_bash_command\nls -la\n```"
    # Mirrors the smoke artifact: fence-close immediately followed by second THOUGHT (no newline).
    content = block + block

    result = parse_regex_actions(
        content, action_regex=ACTION_REGEX, format_error_template=FORMAT_ERROR_TEMPLATE
    )

    assert result == [{"command": "ls -la"}]


def test_distinct_actions_still_raise():
    """Two genuinely different actions still raise FormatError -- dedup must not mask real failures."""
    content = (
        "THOUGHT: first\n\n```mswea_bash_command\nls -la\n```"
        "THOUGHT: second\n\n```mswea_bash_command\npwd\n```"
    )

    with pytest.raises(FormatError) as exc_info:
        parse_regex_actions(
            content, action_regex=ACTION_REGEX, format_error_template=FORMAT_ERROR_TEMPLATE
        )

    assert exc_info.value.messages[0]["extra"]["n_actions"] == 2


def test_single_action_unchanged():
    """A single action passes through unchanged -- no regression."""
    content = "THOUGHT: do thing\n\n```mswea_bash_command\nls -la\n```"

    result = parse_regex_actions(
        content, action_regex=ACTION_REGEX, format_error_template=FORMAT_ERROR_TEMPLATE
    )

    assert result == [{"command": "ls -la"}]


def test_zero_actions_still_raises():
    """No action block still raises FormatError with n_actions=0."""
    content = "THOUGHT: I forgot to emit a command block."

    with pytest.raises(FormatError) as exc_info:
        parse_regex_actions(
            content, action_regex=ACTION_REGEX, format_error_template=FORMAT_ERROR_TEMPLATE
        )

    assert exc_info.value.messages[0]["extra"]["n_actions"] == 0
