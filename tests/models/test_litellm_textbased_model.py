import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import litellm
import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


def _make_response(content: str) -> Mock:
    """Build a Mock litellm response whose .choices[0].message.content == content."""
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = content
    mock_message.model_dump.return_value = {"role": "assistant", "content": content}
    mock_response.choices = [Mock(message=mock_message)]
    mock_response.model_dump.return_value = {"content": content}
    return mock_response


_SINGLE_BLOCK = "```mswea_bash_command\necho one\n```"
_MULTI_BLOCK = (
    "```mswea_bash_command\nsed -n '1,260p' a.py\n```\n"
    "```mswea_bash_command\nsed -n '260,520p' a.py\n```\n"
    "```mswea_bash_command\nsed -n '520,900p' a.py\n```"
)
_DUP_BLOCK = "```mswea_bash_command\necho dup\n```\n```mswea_bash_command\necho dup\n```"
_PROSE_ONLY = "I am thinking about the problem but did not produce any fenced bash block."


def test_authentication_error_enhanced_message():
    """Test that AuthenticationError gets enhanced with config set instruction."""
    model = LitellmTextbasedModel(model_name="gpt-4")

    # Create a mock exception that behaves like AuthenticationError
    original_error = Mock(spec=litellm.exceptions.AuthenticationError)
    original_error.message = "Invalid API key"

    with patch("litellm.completion") as mock_completion:
        # Make completion raise the mock error
        def side_effect(*args, **kwargs):
            raise litellm.exceptions.AuthenticationError("Invalid API key", llm_provider="openai", model="gpt-4")

        mock_completion.side_effect = side_effect

        with pytest.raises(litellm.exceptions.AuthenticationError) as exc_info:
            model._query([{"role": "user", "content": "test"}])

        # Check that the error message was enhanced
        assert "You can permanently set your API key with `mini-extra config set KEY VALUE`." in str(exc_info.value)


def test_model_registry_loading():
    """Test that custom model registry is loaded and registered when provided."""
    model_costs = {
        "my-custom-model": {
            "max_tokens": 4096,
            "input_cost_per_token": 0.0001,
            "output_cost_per_token": 0.0002,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(model_costs, f)
        registry_path = f.name

    try:
        with patch("litellm.utils.register_model") as mock_register:
            _model = LitellmTextbasedModel(model_name="my-custom-model", litellm_model_registry=Path(registry_path))

            # Verify register_model was called with the correct data
            mock_register.assert_called_once_with(model_costs)
    except Exception as e:
        print(e)
        raise e
    finally:
        Path(registry_path).unlink()


def test_model_registry_none():
    """Test that no registry loading occurs when litellm_model_registry is None."""
    with patch("litellm.register_model") as mock_register:
        _model = LitellmTextbasedModel(model_name="gpt-4", litellm_model_registry=None)

        # Verify register_model was not called
        mock_register.assert_not_called()


def test_model_registry_not_provided():
    """Test that no registry loading occurs when litellm_model_registry is not provided."""
    with patch("litellm.register_model") as mock_register:
        _model = LitellmTextbasedModel(model_name="gpt-4o")

        # Verify register_model was not called
        mock_register.assert_not_called()


def test_litellm_model_cost_tracking_ignore_errors():
    """Test that models work with cost_tracking='ignore_errors'."""
    model = LitellmTextbasedModel(model_name="gpt-4o", cost_tracking="ignore_errors")

    initial_cost = GLOBAL_MODEL_STATS.cost

    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "```mswea_bash_command\necho test\n```"
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "```mswea_bash_command\necho test\n```",
        }
        mock_response.choices = [Mock(message=mock_message)]
        mock_response.model_dump.return_value = {"test": "response"}
        mock_completion.return_value = mock_response

        with patch("litellm.cost_calculator.completion_cost", side_effect=ValueError("Model not found")):
            messages = [{"role": "user", "content": "test"}]
            result = model.query(messages)

            assert result["content"] == "```mswea_bash_command\necho test\n```"
            assert result["extra"]["actions"] == [{"command": "echo test"}]
            assert GLOBAL_MODEL_STATS.cost == initial_cost


def test_litellm_model_cost_validation_zero_cost():
    """Test that zero cost raises error when cost tracking is enabled."""
    model = LitellmTextbasedModel(model_name="gpt-4o")

    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.model_dump.return_value = {"test": "response"}
        mock_completion.return_value = mock_response

        with patch("litellm.cost_calculator.completion_cost", return_value=0.0):
            messages = [{"role": "user", "content": "test"}]

            with pytest.raises(RuntimeError) as exc_info:
                model.query(messages)

            assert "Cost must be > 0.0, got 0.0" in str(exc_info.value)
            assert "MSWEA_COST_TRACKING='ignore_errors'" in str(exc_info.value)


def test_format_error_retry_succeeds_on_second_attempt():
    """Multi-action on call 1, single-action on call 2: query() returns single-action with retries==1."""
    model = LitellmTextbasedModel(model_name="gpt-4o", cost_tracking="ignore_errors")
    responses = [_make_response(_MULTI_BLOCK), _make_response(_SINGLE_BLOCK)]

    with patch("litellm.completion", side_effect=responses):
        with patch("litellm.cost_calculator.completion_cost", side_effect=ValueError("ignored")):
            result = model.query([{"role": "user", "content": "test"}])

    assert result["extra"]["actions"] == [{"command": "echo one"}]
    assert result["extra"]["format_error_retries"] == 1
    assert len(result["extra"]["discarded_responses"]) == 1
    assert result["extra"]["discarded_responses"][0]["n_actions"] == 3
    assert result["extra"].get("format_error_fallback") is None


def test_format_error_retry_exhausts_then_fallback_first():
    """Multi-action on every call: fallback fires, take-first, K=3 retries → 4 discarded."""
    model = LitellmTextbasedModel(model_name="gpt-4o", cost_tracking="ignore_errors")
    responses = [_make_response(_MULTI_BLOCK)] * 4

    with patch("litellm.completion", side_effect=responses):
        with patch("litellm.cost_calculator.completion_cost", side_effect=ValueError("ignored")):
            result = model.query([{"role": "user", "content": "test"}])

    assert result["extra"]["format_error_fallback"] is True
    assert result["extra"]["format_error_retries"] == 3
    assert result["extra"]["actions"] == [{"command": "sed -n '1,260p' a.py"}]
    assert len(result["extra"]["discarded_responses"]) == 4
    assert result["extra"]["fallback_n_extracted"] == 3


def test_format_error_zero_action_bubbles_after_retries():
    """Prose-only on every call: FormatError bubbles with format_error_retries==3 augmented."""
    model = LitellmTextbasedModel(model_name="gpt-4o", cost_tracking="ignore_errors")
    responses = [_make_response(_PROSE_ONLY)] * 4

    with patch("litellm.completion", side_effect=responses):
        with patch("litellm.cost_calculator.completion_cost", side_effect=ValueError("ignored")):
            with pytest.raises(FormatError) as exc_info:
                model.query([{"role": "user", "content": "test"}])

    extra = exc_info.value.messages[0]["extra"]
    assert extra["n_actions"] == 0
    assert extra["format_error_retries"] == 3
    assert len(extra["discarded_responses"]) == 3


def test_format_error_dedup_then_no_retry():
    """Byte-identical duplicate blocks are deduped by parse_regex_actions; no retries fire."""
    model = LitellmTextbasedModel(model_name="gpt-4o", cost_tracking="ignore_errors")

    with patch("litellm.completion", return_value=_make_response(_DUP_BLOCK)) as mock_completion:
        with patch("litellm.cost_calculator.completion_cost", side_effect=ValueError("ignored")):
            result = model.query([{"role": "user", "content": "test"}])

    assert mock_completion.call_count == 1
    assert result["extra"]["actions"] == [{"command": "echo dup"}]
    assert "format_error_retries" not in result["extra"]
    assert "discarded_responses" not in result["extra"]


def test_format_error_retries_disabled():
    """With format_error_retries=0, FormatError bubbles immediately on multi-action (pre-C-1 behavior)."""
    model = LitellmTextbasedModel(model_name="gpt-4o", cost_tracking="ignore_errors", format_error_retries=0)

    with patch("litellm.completion", return_value=_make_response(_MULTI_BLOCK)) as mock_completion:
        with patch("litellm.cost_calculator.completion_cost", side_effect=ValueError("ignored")):
            with pytest.raises(FormatError):
                model.query([{"role": "user", "content": "test"}])

    assert mock_completion.call_count == 1


def test_harness_warning_injected_on_fallback():
    """format_observation_messages appends a <harness_warning> user message when fallback fired."""
    model = LitellmTextbasedModel(model_name="gpt-4o", cost_tracking="ignore_errors")
    message = {
        "role": "assistant",
        "content": _MULTI_BLOCK,
        "extra": {"format_error_fallback": True, "fallback_n_extracted": 3},
    }
    outputs = [{"output": "fake stdout", "returncode": 0, "exception_info": None}]

    msgs = model.format_observation_messages(message, outputs)

    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert "fake stdout" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "<harness_warning>" in msgs[1]["content"]
    assert "3 commands" in msgs[1]["content"]
    assert "remaining 2 were dropped" in msgs[1]["content"]
    assert msgs[1]["extra"]["harness_injection"] == "format_error_fallback_warning"


def test_cost_rolls_in_discarded_attempts():
    """2 retries each $0.10, success on attempt 3 costing $0.20 → total cost $0.40, GLOBAL_MODEL_STATS +0.40 once."""
    model = LitellmTextbasedModel(model_name="gpt-4o")
    responses = [_make_response(_MULTI_BLOCK), _make_response(_MULTI_BLOCK), _make_response(_SINGLE_BLOCK)]
    costs = [0.10, 0.10, 0.20]
    initial_cost = GLOBAL_MODEL_STATS.cost
    initial_calls = GLOBAL_MODEL_STATS.n_calls

    with patch("litellm.completion", side_effect=responses):
        with patch("litellm.cost_calculator.completion_cost", side_effect=costs):
            result = model.query([{"role": "user", "content": "test"}])

    assert result["extra"]["cost"] == pytest.approx(0.40)
    assert result["extra"]["format_error_retries"] == 2
    assert GLOBAL_MODEL_STATS.cost == pytest.approx(initial_cost + 0.40)
    assert GLOBAL_MODEL_STATS.n_calls == initial_calls + 1


def test_zero_action_bubble_carries_rolled_cost():
    """On a zero-action storm the bubbled FormatError must carry the total discarded cost,
    so the agent can bill it to self.cost; otherwise cost_limit never trips on format storms."""
    model = LitellmTextbasedModel(model_name="gpt-4o")  # cost tracking ON (default)
    responses = [_make_response(_PROSE_ONLY)] * 4
    costs = [0.10, 0.10, 0.10, 0.10]

    with patch("litellm.completion", side_effect=responses):
        with patch("litellm.cost_calculator.completion_cost", side_effect=costs):
            with pytest.raises(FormatError) as exc_info:
                model.query([{"role": "user", "content": "test"}])

    extra = exc_info.value.messages[0]["extra"]
    assert extra["n_actions"] == 0
    assert extra["cost"] == pytest.approx(0.40)  # all 4 discarded calls billed
