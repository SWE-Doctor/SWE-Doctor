from unittest.mock import MagicMock, patch

import litellm
import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


class TestLitellmModel:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []


class TestContextWindowGuard:
    """Pre-flight guard: some providers (e.g. tu-zi) do NOT raise
    ContextWindowExceededError for over-window input — they hang until the
    request times out, which the retry loop then repeats. We must detect the
    overflow ourselves and raise ContextWindowExceededError (an abort_exception)
    so the instance terminates immediately instead of spinning on retries."""

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    @patch("minisweagent.models.litellm_model.litellm.token_counter")
    @patch("minisweagent.models.litellm_model.litellm.get_model_info")
    def test_query_raises_when_input_exceeds_context_window(
        self, mock_info, mock_count, mock_cost, mock_completion
    ):
        mock_info.return_value = {"max_input_tokens": 1000}
        mock_count.return_value = 1500  # over the 1000-token limit
        # A valid response is set up so that WITHOUT the guard, query() would
        # return normally (clean RED: "did not raise" rather than an unrelated error).
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls"}'
        tool_call.id = "c1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="openai/gpt-5.4-mini")
        with pytest.raises(litellm.exceptions.ContextWindowExceededError):
            model.query([{"role": "user", "content": "x"}])
        mock_completion.assert_not_called()  # must NOT send the doomed request

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    @patch("minisweagent.models.litellm_model.litellm.token_counter")
    @patch("minisweagent.models.litellm_model.litellm.get_model_info")
    def test_query_proceeds_when_within_context_window(
        self, mock_info, mock_count, mock_cost, mock_completion
    ):
        mock_info.return_value = {"max_input_tokens": 100000}
        mock_count.return_value = 50  # well within the limit
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls"}'
        tool_call.id = "c1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="openai/gpt-5.4-mini")
        model.query([{"role": "user", "content": "x"}])
        mock_completion.assert_called_once()

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    @patch("minisweagent.models.litellm_model.litellm.get_model_info")
    def test_query_proceeds_when_context_window_unknown(self, mock_info, mock_cost, mock_completion):
        mock_info.side_effect = Exception("model not in registry")  # unknown window
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls"}'
        tool_call.id = "c1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="some/unknown-model")
        model.query([{"role": "user", "content": "x"}])
        mock_completion.assert_called_once()  # unknown window -> do not block
