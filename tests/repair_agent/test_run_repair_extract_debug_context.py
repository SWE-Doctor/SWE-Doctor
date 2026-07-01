# tests/repair_agent/test_run_repair_extract_debug_context.py
"""Regression tests for _extract_debug_context's handling of contract_impact.

After reverting c03f2e38, callers alone should NOT trigger emission of
rich_contract_impact — only an actual signature change should.
"""
from repair_agent.run_repair import _extract_debug_context


def _rca(contract):
    return {
        "source": "debug_agent",
        "debug_report": {
            "root_cause_files": ["m.py"], "root_cause_functions": ["m.f"],
            "reasoning": "r", "suggested_fix": "s", "transcript": [],
            "timed_out": False,
        },
        "rich_rca": {"schema_version": 2, "contract_impact": contract},
    }


def test_no_contract_impact_when_unchanged_with_callers():
    """Callers without a signature change must NOT surface in the prompt."""
    rca = _rca({
        "changed": False,
        "kind": "none",
        "summary": "",
        "callers": ["pkg.x.caller_a", "pkg.x.caller_b"],
        "callers_truncated": False,
    })
    ctx = _extract_debug_context(rca)
    assert ctx.get("rich_contract_impact") is None, (
        "callers-only payload must be suppressed — only emit when changed=True"
    )


def test_contract_impact_emitted_when_changed():
    rca = _rca({
        "changed": True,
        "kind": "signature_changed",
        "summary": "added kwarg",
        "callers": ["pkg.x.caller_a"],
        "callers_truncated": False,
    })
    ctx = _extract_debug_context(rca)
    ci = ctx.get("rich_contract_impact")
    assert ci is not None
    assert ci["kind"] == "signature_changed"
    assert ci["callers"] == ["pkg.x.caller_a"]
    # The "changed" key was an internal flag of the c03f2e38 widening; after
    # revert the gate is binary (emit-or-not) so this key should be gone.
    assert "changed" not in ci, "drop the 'changed' key after revert"


def test_contract_impact_kind_default_is_empty_string_not_signature_unchanged():
    """Pre-c03f2e38, kind defaulted to '' when missing. The 'signature_unchanged'
    default was added by c03f2e38 to support the deleted else-branch."""
    rca = _rca({
        "changed": True,
        # kind intentionally absent
        "summary": "",
        "callers": ["caller"],
        "callers_truncated": False,
    })
    ctx = _extract_debug_context(rca)
    assert ctx["rich_contract_impact"]["kind"] == ""


def test_related_non_code_is_not_lifted_into_ctx():
    """`rich_related_non_code` was confirmed net-negative (Δ=+4 when removed)
    in the 2026-04-30 lever ablation. _extract_debug_context must NOT lift
    `rich_rca.related_non_code.hits` into ctx, so the prompt never renders the
    'Related non-code files worth scanning' block.

    Stage2 still stores related_non_code in the rca.json (zero data loss).
    """
    rca = {
        "source": "debug_agent",
        "debug_report": {
            "root_cause_files": ["m.py"], "root_cause_functions": ["m.f"],
            "reasoning": "r", "suggested_fix": "s", "transcript": [],
            "timed_out": False,
        },
        "rich_rca": {
            "schema_version": 2,
            "related_non_code": {
                "hits": [
                    "docs/changes.rst:42 (LookupModule)",
                    "tests/fixtures/sample.yml:7 (LookupModule)",
                ],
                "provenance": {"source": "rg_nontext"},
            },
        },
    }
    ctx = _extract_debug_context(rca)
    assert "rich_related_non_code" not in ctx, (
        "rich_related_non_code must not be propagated to the repair prompt "
        "(net-negative lever per 2026-04-30 ablation)"
    )
