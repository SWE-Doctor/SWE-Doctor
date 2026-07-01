"""Thin wrapper around litellm for direct LLM calls (no tool-calling)."""

import json
import logging
import os
import threading
import time
from pathlib import Path

import litellm

litellm.drop_params = True

logger = logging.getLogger("repro_test.llm")

# ── Call-chain tracer ────────────────────────────────────────────────────
# Each LLM call is appended as a JSON line to a trace file.
# Thread-local: each worker thread can have its own trace file.

_local = threading.local()
_write_lock = threading.Lock()


def set_trace_file(path: str | Path | None) -> None:
    """Set (or clear) the JSONL trace file for LLM calls in this thread."""
    _local.trace_file = Path(path) if path else None
    _local.call_seq = 0


def _write_trace(record: dict) -> None:
    trace_file = getattr(_local, "trace_file", None)
    if trace_file is None:
        return
    with _write_lock:
        with trace_file.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def llm_call(
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    *,
    caller: str = "",
    max_attempts: int = 5,
) -> str:
    """Single LLM completion call. Returns the text content.

    Args:
        caller: optional tag (e.g. "localizer", "critic") for trace logging.

    The provider (gpt-5.4 via tu-zi) intermittently returns
    ``message.content is None`` (empty/filtered/reasoning-only response). We
    retry such empty responses with backoff; retries use a small non-zero
    temperature to break the deterministic decoding that produced the empty
    content. Raises RuntimeError only if every attempt comes back empty.
    """
    _local.call_seq = getattr(_local, "call_seq", 0) + 1
    seq = _local.call_seq
    t0 = time.time()

    # Inject reasoning_effort (e.g. "high") from env when set. gpt-5.4 via tu-zi
    # defaults to a low effort which hurts quality; empty leaves it unset.
    _extra = {}
    _re = os.environ.get("REASONING_EFFORT", "").strip()
    if _re:
        _extra["reasoning_effort"] = _re

    # DeepSeek thinking-mode toggle (external param, default on). Only DeepSeek
    # models honor `thinking`; gate on the model name so non-DeepSeek runs are
    # unaffected. Stage 1 runs with DEEPSEEK_THINKING=disabled to turn reasoning off.
    if "deepseek" in model.lower():
        _think = os.environ.get("DEEPSEEK_THINKING", "enabled").strip().lower()
        _off = _think in ("disabled", "off", "0", "false", "no")
        _extra["extra_body"] = {"thinking": {"type": "disabled" if _off else "enabled"}}

    # Streaming toggle (external param, default off). Some OpenAI-compatible
    # gateways (e.g. tu-zi) serve gpt-5.x ONLY as a stream; a non-stream call
    # returns an empty SSE. When LLM_STREAM is set we collect the chunks and
    # rebuild an equivalent non-stream ModelResponse via stream_chunk_builder,
    # so the rest of this function is unchanged.
    _stream = os.environ.get("LLM_STREAM", "").strip().lower() in ("1", "true", "yes", "on")
    content = None
    for attempt in range(1, max_attempts + 1):
        _temp = temperature if attempt == 1 else max(temperature, 0.5)
        if _stream:
            _chunks = list(litellm.completion(
                model=model, messages=messages, temperature=_temp,
                max_tokens=max_tokens, stream=True,
                stream_options={"include_usage": True}, **_extra,
            ))
            response = litellm.stream_chunk_builder(_chunks, messages=messages)
        else:
            response = litellm.completion(
                model=model, messages=messages, temperature=_temp,
                max_tokens=max_tokens, **_extra,
            )
        raw = response.choices[0].message.content
        if raw is not None:
            content = raw.strip()
            break
        logger.warning(
            "Empty (None) content from model=%s caller=%s attempt=%d/%d; retrying",
            model, caller, attempt, max_attempts,
        )
        time.sleep(min(2 ** attempt, 30))
    if content is None:
        raise RuntimeError(
            f"LLM returned empty content {max_attempts}x (caller={caller!r}, model={model!r})"
        )
    elapsed = time.time() - t0
    usage = getattr(response, "usage", None) or {}
    _reasoning = getattr(response.choices[0].message, "reasoning_content", None)

    _write_trace({
        "seq": seq,
        "caller": caller,
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response": content,
        "has_reasoning": bool(_reasoning),
        "reasoning_chars": len(_reasoning) if _reasoning else 0,
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    return content
