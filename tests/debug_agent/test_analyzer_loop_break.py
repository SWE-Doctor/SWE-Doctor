"""R3: when the LLM re-emits identical <conclusion>s after gate rejection,
analyzer must escalate to a STOP-RE-EMITTING directive (after 3 consecutive
identical rejections) instead of repeating the same _diagnose_gate text."""
from debug_agent.analyzer import DebugAnalyzer, _conclusion_fingerprint


def test_fingerprint_collapses_app_prefix_and_leading_slash_variants():
    """The LLM frequently alternates `lib/foo.py` <-> `/app/lib/foo.py` between
    rejected conclusions. Without normalization, each variant fingerprints
    differently and the streak counter never reaches 3."""
    a = (
        "<conclusion>"
        "<root_cause_files>lib/foo.py, lib/bar.py</root_cause_files>"
        "<root_cause_functions>foo.bar, baz.qux</root_cause_functions>"
        "<reasoning>r1</reasoning><suggested_fix>f1</suggested_fix>"
        "</conclusion>"
    )
    b = (
        "<conclusion>"
        "<root_cause_files>/app/lib/bar.py, /app/lib/foo.py</root_cause_files>"
        "<root_cause_functions>baz.qux, foo.bar</root_cause_functions>"
        "<reasoning>r2 — different reasoning</reasoning>"
        "<suggested_fix>different fix</suggested_fix>"
        "</conclusion>"
    )
    assert _conclusion_fingerprint(a) == _conclusion_fingerprint(b)
    # Different files must fingerprint differently
    c = a.replace("lib/foo.py", "lib/other.py")
    assert _conclusion_fingerprint(a) != _conclusion_fingerprint(c)

_CONCL = (
    "<conclusion>"
    "<root_cause_files>foo.py</root_cause_files>"
    "<root_cause_functions>foo.bar</root_cause_functions>"
    "<reasoning>same reasoning every time</reasoning>"
    "<suggested_fix>same fix</suggested_fix>"
    "</conclusion>"
)


def _noop_dispatch(action, _container, _ctx):  # pragma: no cover — never invoked
    return ""


def test_three_identical_rejected_conclusions_inject_escalation():
    """After 3 consecutive identical rejected <conclusion>s, the next prompt
    sent to the LLM must contain an explicit STOP RE-EMITTING directive that
    names the next required action (pdb_start) and the minimum pdb_cmd count."""
    msgs = []

    def fake_llm(prompt):
        msgs.append(prompt)
        return _CONCL

    ana = DebugAnalyzer(
        llm=fake_llm,
        dispatch=_noop_dispatch,
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
    report = ana.run()

    # Each call now receives a role-separated messages list; the escalation
    # directive lives in the latest user turn.
    def _joined(messages):
        return "\n".join(m["content"] for m in messages)

    # Prompts >= 4 (i.e. msgs[3:]) must include the escalation directive
    escalated = [
        p for p in msgs[3:]
        if "STOP RE-EMITTING <conclusion>" in _joined(p)
        and "pdb_start" in _joined(p)
        and "pdb_cmd" in _joined(p)
    ]
    assert escalated, (
        f"no escalation in prompts >= 4. last prompt tail: {_joined(msgs[-1])[-400:]!r}"
    )
    # Loop runs out the budget — gate still rejects
    assert report.timed_out is True
    assert report.root_cause_files == []
