"""Regression guard for the runaway root cause.

The debug loop must hand the model a *role-separated* chat `messages` list
(system / user / assistant / user / ...), NOT a single flat user prompt that
concatenates prior turns as "ASSISTANT:\\n...\\nTOOL (...):\\n...". The flat-text
form let gpt-5.4 (under high reasoning effort) continue the transcript pattern
and fabricate the tool's turns in one giant response — see
reference_tuzi_ignores_length_caps. Proper chat roles end the model's turn at
the assistant boundary so it cannot write the tool's part.
"""
from __future__ import annotations

import debug_agent.analyzer as ana
from debug_agent.analyzer import DebugAnalyzer


class _StubContainer:
    container_id = "stub"
    workdir = "/app"

    def exec_bash(self, cmd, timeout=120):
        class R:
            returncode = 0
            stdout = "ok"
            stderr = ""
            combined = "ok"
        return R()


def _make_analyzer(capture, responses, monkeypatch):
    it = iter(responses)

    def llm(arg):
        capture.append(arg)
        return next(it)

    ctx = {
        "issue": "thing breaks",
        "workdir": "/app",
        "repro_path": "/app/_repro/repro_0.py",
        "repro_nodeid": "repro_0.py::test",
        "preflight_seed": "(none)",
    }
    # The conclusion gate is irrelevant here; allow it so the loop terminates.
    monkeypatch.setattr(ana, "can_conclude", lambda *_a, **_k: True)
    return DebugAnalyzer(
        llm=llm,
        dispatch=lambda action, container, c: "tool ran fine",
        container=_StubContainer(),
        ctx=ctx,
        max_rounds=5,
    )


def test_llm_receives_role_separated_messages(monkeypatch):
    capture: list = []
    responses = [
        '<reason>look</reason><action name="bash">ls /app</action>',
        '<reason>repro</reason><action name="pytest">-x</action>',
        ('<conclusion><root_cause_files>m.py</root_cause_files>'
         '<root_cause_functions>m.f</root_cause_functions>'
         '<reasoning>r</reasoning><suggested_fix>s</suggested_fix></conclusion>'),
    ]
    analyzer = _make_analyzer(capture, responses, monkeypatch)
    analyzer.run()

    assert len(capture) >= 2, "expected at least two model calls"

    # 1) Every call must pass a list of role dicts, never a flat string.
    for arg in capture:
        assert isinstance(arg, list), f"llm got {type(arg).__name__}, expected messages list"
        assert all(isinstance(m, dict) and "role" in m and "content" in m for m in arg)

    first, second = capture[0], capture[1]

    # 2) Exactly one system message, at the front.
    assert first[0]["role"] == "system"
    assert sum(1 for m in second if m["role"] == "system") == 1

    # 3) The model's first response appears as its own assistant turn on the
    #    second call — not embedded inside a user blob.
    assert any(m["role"] == "assistant" and 'name="bash"' in m["content"] for m in second)

    # 4) The tool output is a separate user turn (role boundary is real).
    assert any(m["role"] == "user" and "tool ran fine" in m["content"] for m in second)

    # 5) No assistant turn may carry a fabricated/injected "TOOL (" marker —
    #    tool results live only in user turns.
    for m in second:
        if m["role"] == "assistant":
            assert "TOOL (" not in m["content"]
