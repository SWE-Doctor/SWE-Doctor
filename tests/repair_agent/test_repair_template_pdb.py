from pathlib import Path
from jinja2 import Environment, FileSystemLoader

REPO = Path(__file__).resolve().parents[2]
TPL = "task_template.j2"


def _render(ctx):
    env = Environment(loader=FileSystemLoader(str(REPO / "repair_agent")),
                      keep_trailing_newline=True)
    return env.get_template(TPL).render(**ctx)


_BASE_CTX = {
    "problem_statement": "issue",
    "requirements": "", "interface": "",
    "rca_top1_files": [], "rca_top5_files": [], "rca_evidence": "",
    "rca_confidence": "missing", "repro_only_files": [],
    "debug_reasoning": "r", "debug_suggested_fix": "s",
    "debug_root_cause_files": ["m.py"], "debug_root_cause_functions": ["m.f"],
    "debug_transcript_tail": "", "debug_timed_out": False,
    "rich_symptom": None, "rich_contract_impact": None,
    "rich_propagation_frames": None, "rich_related_non_code": None,
    "repro_localizer_files": [], "repro_localizer_funcs": [],
    "rich_branch_observations": [], "rich_pdb_anomalies": [],
}


def test_pdb_section_renders_when_observations_present():
    ctx = {**_BASE_CTX,
           "rich_branch_observations": [{
               "file": "m.py", "lineno": 5, "cond_text": "if x is None",
               "arm_taken": "then", "locals_at_stop": {"x": "None"},
               "evidence_refs": [],
           }],
           "rich_pdb_anomalies": [{
               "expr": "x", "file": "m.py", "lineno": 5,
               "value_repr": "None", "tags": ["none"], "evidence_refs": [],
           }]}
    out = _render(ctx)
    assert "── PDB session evidence ──" in out
    assert "if x is None" in out
    assert "arm: then" in out
    assert "x=None" in out
    assert "[none]" in out


def test_pdb_section_absent_when_keys_empty():
    out = _render(_BASE_CTX)
    assert "── PDB session evidence ──" not in out


def test_extract_debug_context_lifts_pdb_evidence_to_top_level():
    """Regression: the template references rich_branch_observations and
    rich_pdb_anomalies at top level, so _extract_debug_context must lift
    them out of rich_rca. Forgetting this silently drops the entire
    'PDB session evidence' section even when the enricher produced data."""
    from repair_agent.run_repair import _extract_debug_context
    rca = {
        "source": "debug_agent",
        "debug_report": {
            "root_cause_files": ["m.py"], "root_cause_functions": ["m.f"],
            "reasoning": "r", "suggested_fix": "s", "transcript": [],
            "timed_out": False,
        },
        "rich_rca": {
            "schema_version": 2,
            "rich_pdb_anomalies": [
                {"expr": "x", "file": "m.py", "lineno": 5,
                 "value_repr": "None", "tags": ["none"], "evidence_refs": []},
            ],
            "rich_branch_observations": [
                {"file": "m.py", "lineno": 7, "cond_text": "if x is None",
                 "arm_taken": "then", "locals_at_stop": {"x": "None"},
                 "evidence_refs": []},
            ],
        },
    }
    ctx = _extract_debug_context(rca)
    assert ctx.get("rich_pdb_anomalies"), "rich_pdb_anomalies must be lifted"
    assert ctx.get("rich_branch_observations"), "rich_branch_observations must be lifted"
    assert ctx["rich_pdb_anomalies"][0]["expr"] == "x"
    assert ctx["rich_branch_observations"][0]["cond_text"] == "if x is None"


def test_render_task_end_to_end_renders_pdb_section():
    """Higher-level: feed a real-shaped _rca.json through _render_task and
    confirm the evidence section makes it into the final prompt."""
    from jinja2 import Template
    from repair_agent.run_repair import _render_task
    rca = {
        "source": "debug_agent",
        "debug_report": {
            "root_cause_files": ["m.py"], "root_cause_functions": ["m.f"],
            "reasoning": "r", "suggested_fix": "s", "transcript": [],
            "timed_out": False,
        },
        "rich_rca": {
            "schema_version": 2,
            "rich_pdb_anomalies": [
                {"expr": "editions", "file": "u.py", "lineno": 759,
                 "value_repr": "[]", "tags": ["empty"], "evidence_refs": [9]},
            ],
        },
    }
    tpl = Template((REPO / "repair_agent" / TPL).read_text())
    out = _render_task(tpl, {"problem_statement": "x"}, rca,
                       repro_files=["t.py"], repro_funcs=["f"])
    assert "── PDB session evidence ──" in out
    assert "editions" in out
    assert "[empty]" in out


def test_signature_unchanged_branch_is_gone():
    """The c03f2e38 'signature unchanged — listed for context' phrasing
    distracted models into over-fixing. After revert, the template must not
    render any callers block when contract is unchanged (i.e. when
    rich_contract_impact is None, which is now the only state for that case).
    """
    ctx = {**_BASE_CTX, "rich_contract_impact": None}
    out = _render(ctx)
    assert "signature unchanged" not in out
    assert "do not modify unless the fix alters the contract" not in out
    assert "Known callers" not in out


def test_callers_heading_simplified_when_changed():
    """When a signature change is detected, the callers heading must be the
    plain 'Known callers:' form, not 'Known callers (update these if the
    contract genuinely changes):' which is hedged language inviting drift."""
    ctx = {**_BASE_CTX,
           "rich_contract_impact": {
               "kind": "signature_changed",
               "summary": "added kwarg foo",
               "callers": ["pkg.a.caller_a", "pkg.b.caller_b"],
               "callers_truncated": False,
           }}
    out = _render(ctx)
    assert "Contract impact — suggested fix changes" in out
    assert "Known callers:" in out
    assert "update these if the contract genuinely changes" not in out
    assert "pkg.a.caller_a" in out


def test_related_non_code_block_never_renders_end_to_end():
    """End-to-end regression: even when `rich_rca.related_non_code.hits` is
    populated by stage2 enrichment, the rendered repair prompt must NOT
    include the 'Related non-code files worth scanning' section. Confirmed
    net-negative lever per 2026-04-30 ablation (drop recovers 24 → 28)."""
    from jinja2 import Template
    from repair_agent.run_repair import _render_task
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
    tpl = Template((REPO / "repair_agent" / TPL).read_text())
    out = _render_task(tpl, {"problem_statement": "x"}, rca,
                       repro_files=[], repro_funcs=[])
    assert "Related non-code files worth scanning" not in out
    assert "docs/changes.rst:42" not in out
    assert "tests/fixtures/sample.yml:7" not in out
