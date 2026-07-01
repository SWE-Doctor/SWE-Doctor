"""debug_agent over-context-window handling.

tu-zi does NOT raise ContextWindowExceededError for over-window input — it hangs
~273s then times out, and _build_llm's 8-retry loop repeats that every round, so
a single over-window instance spins for ~hours (the SIGALRM wall_timeout does not
reliably interrupt the blocked network read). The debug agent uses its OWN
litellm.completion (run_debug._build_llm), NOT LitellmModel, so it needs its own
pre-flight token guard. On overflow the instance must terminate immediately.
"""
from unittest.mock import patch

import litellm
import pytest

from debug_agent.analyzer import DebugAnalyzer
from debug_agent.run_debug import _build_llm


class TestBuildLlmContextWindowGuard:
    @patch("litellm.completion")
    @patch("litellm.token_counter")
    @patch("litellm.get_model_info")
    def test_call_raises_when_input_exceeds_window_without_hitting_api(
        self, mock_info, mock_count, mock_completion
    ):
        mock_info.return_value = {"max_input_tokens": 1000}
        mock_count.return_value = 1500  # over the 1000 limit
        # valid response so that WITHOUT the guard call() would return normally
        # (clean RED: "did not raise" rather than spinning the 8 retries)
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}

        call = _build_llm("openai/gpt-5.4-mini")
        with pytest.raises(litellm.exceptions.ContextWindowExceededError):
            call([{"role": "user", "content": "x"}])
        mock_completion.assert_not_called()  # must NOT send the doomed request

    @patch("litellm.completion")
    @patch("litellm.token_counter")
    @patch("litellm.get_model_info")
    def test_call_proceeds_when_within_window(self, mock_info, mock_count, mock_completion):
        mock_info.return_value = {"max_input_tokens": 100000}
        mock_count.return_value = 50
        mock_completion.return_value = {"choices": [{"message": {"content": "hello"}}]}

        call = _build_llm("openai/gpt-5.4-mini")
        assert call([{"role": "user", "content": "x"}]) == "hello"
        mock_completion.assert_called_once()

    @patch("litellm.completion")
    @patch("litellm.get_model_info")
    def test_call_proceeds_when_window_unknown(self, mock_info, mock_completion):
        mock_info.side_effect = Exception("model not in registry")
        mock_completion.return_value = {"choices": [{"message": {"content": "hello"}}]}

        call = _build_llm("some/unknown-model")
        assert call([{"role": "user", "content": "x"}]) == "hello"
        mock_completion.assert_called_once()


def test_analyzer_run_terminates_on_context_window_exceeded():
    """When the llm callable raises ContextWindowExceededError, the debug loop
    must stop and return a (timed_out) report instead of propagating the error."""
    def overflow_llm(messages):
        raise litellm.exceptions.ContextWindowExceededError(
            message="input exceeds context window", model="openai/gpt-5.4-mini", llm_provider="openai"
        )

    ana = DebugAnalyzer(
        llm=overflow_llm,
        dispatch=lambda action, container, ctx: "",
        container=None,
        ctx={
            "issue": "x",
            "repro_nodeid": "n",
            "workdir": "/app",
            "repro_path": "/app/_repro/repro_0.py",
            "preflight_seed": "(none)",
            "_pdb_session_log": [],
        },
        max_rounds=6,
    )
    report = ana.run()  # must NOT raise
    assert report.timed_out is True
    assert report.root_cause_files == []
