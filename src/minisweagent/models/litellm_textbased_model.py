import logging
import os
import re
import time

import litellm

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("minisweagent.litellm_textbased_model")


class LitellmTextbasedModelConfig(LitellmModelConfig):
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    """Regex to extract the action from the LM's output."""
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    """Template used when the LM's output is not in the expected format."""
    format_error_retries: int = 3
    """Silent retries on FormatError before falling back. 0 disables retry."""


class LitellmTextbasedModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmTextbasedModelConfig, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            model_kwargs = dict(self.config.model_kwargs)
            # DeepSeek thinking-mode toggle (external param, default on). Only
            # DeepSeek models honor `thinking`; gate on model name so other
            # providers are unaffected. DEEPSEEK_THINKING=disabled turns it off.
            if "deepseek" in (self.config.model_name or "").lower():
                _think = os.environ.get("DEEPSEEK_THINKING", "enabled").strip().lower()
                _off = _think in ("disabled", "off", "0", "false", "no")
                model_kwargs["extra_body"] = {"thinking": {"type": "disabled" if _off else "enabled"}}
            # Streaming toggle (default off). Some gateways (tu-zi) serve gpt-5.x
            # ONLY as a stream; a non-stream call returns an empty SSE. When
            # LLM_STREAM is set, collect chunks and rebuild an equivalent
            # ModelResponse so cost calc / model_dump() downstream are unchanged.
            if os.environ.get("LLM_STREAM", "").strip().lower() in ("1", "true", "yes", "on"):
                _chunks = list(litellm.completion(
                    model=self.config.model_name, messages=messages, stream=True,
                    stream_options={"include_usage": True}, **(model_kwargs | kwargs),
                ))
                return litellm.stream_chunk_builder(_chunks, messages=messages)
            return litellm.completion(
                model=self.config.model_name, messages=messages, **(model_kwargs | kwargs)
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse actions from the model response. Raises FormatError if not exactly one action."""
        content = response.choices[0].message.content or ""
        return parse_regex_actions(
            content, action_regex=self.config.action_regex, format_error_template=self.config.format_error_template
        )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        fe_retries: int = self.config.format_error_retries
        discarded: list[dict] = []

        for attempt in range(fe_retries + 1):
            for transport_attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
                with transport_attempt:
                    response = self._query(self._prepare_messages_for_api(messages), **kwargs)
            cost_output = self._calculate_cost(response)
            try:
                actions = self._parse_actions(response)
            except FormatError as e:
                discarded.append(
                    {
                        "content": response.choices[0].message.content or "",
                        "n_actions": e.messages[0]["extra"]["n_actions"],
                        "cost": cost_output["cost"],
                    }
                )
                if attempt < fe_retries:
                    logger.info(
                        f"FormatError on attempt {attempt + 1}/{fe_retries + 1} "
                        f"(n_actions={discarded[-1]['n_actions']}); retrying silently"
                    )
                    continue
                return self._fallback_after_format_error(response, e, discarded, cost_output)

            rolled_cost = cost_output["cost"] + sum(d["cost"] for d in discarded)
            GLOBAL_MODEL_STATS.add(rolled_cost)
            message = response.choices[0].message.model_dump()
            message["extra"] = {
                "actions": actions,
                "response": response.model_dump(),
                "cost": rolled_cost,
                "timestamp": time.time(),
            }
            if discarded:
                message["extra"]["format_error_retries"] = len(discarded)
                message["extra"]["discarded_responses"] = discarded
            return message

        raise RuntimeError("query() loop exited without returning; this should be unreachable")

    def _fallback_after_format_error(
        self, response, fe_exc: FormatError, discarded: list[dict], cost_output: dict
    ) -> dict:
        n_actions = fe_exc.messages[0]["extra"]["n_actions"]
        rolled_cost = cost_output["cost"] + sum(d["cost"] for d in discarded[:-1])
        # Surface the wasted cost on the bubbled exception so the agent can bill it to
        # self.cost; otherwise cost_limit never trips during a persistent format-error storm.
        fe_exc.messages[0]["extra"]["cost"] = rolled_cost

        if self.config.format_error_retries == 0:
            GLOBAL_MODEL_STATS.add(rolled_cost)
            raise fe_exc

        if n_actions == 0:
            GLOBAL_MODEL_STATS.add(rolled_cost)
            fe_exc.messages[0]["extra"]["format_error_retries"] = len(discarded) - 1
            fe_exc.messages[0]["extra"]["discarded_responses"] = discarded[:-1]
            raise fe_exc

        content = response.choices[0].message.content or ""
        extracted = [a.strip() for a in re.findall(self.config.action_regex, content, re.DOTALL)]
        extracted = list(dict.fromkeys(extracted))
        first = extracted[0]
        GLOBAL_MODEL_STATS.add(rolled_cost)
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": [{"command": first}],
            "response": response.model_dump(),
            "cost": rolled_cost,
            "timestamp": time.time(),
            "format_error_fallback": True,
            "format_error_retries": len(discarded) - 1,
            "discarded_responses": discarded,
            "fallback_n_extracted": len(extracted),
        }
        return message

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into observation messages."""
        msgs = format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
        if message.get("extra", {}).get("format_error_fallback"):
            n_extracted = message["extra"].get("fallback_n_extracted", 0)
            warning = {
                "role": "user",
                "content": (
                    f"<harness_warning>\n"
                    f"Your previous response contained {n_extracted} commands but the harness "
                    f"requires single-command responses. Only your FIRST command was executed; "
                    f"the remaining {n_extracted - 1} were dropped. If you intended a sequence, "
                    f"chain commands with && or || in a single block, or issue them across multiple turns.\n"
                    f"</harness_warning>"
                ),
                "extra": {"harness_injection": "format_error_fallback_warning"},
            }
            msgs.append(warning)
        return msgs
