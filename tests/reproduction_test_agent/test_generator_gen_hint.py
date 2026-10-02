"""gen_hint (go import/package guidance) must reach the generation user prompt;
python (gen_hint="") must be unaffected."""
from reproduction_test_agent.generator import build_prompt


def test_build_prompt_appends_gen_hint_when_present():
    p = build_prompt("the issue", "none", {}, language="go", gen_hint="ZZZ_PACKAGE_HINT")
    assert "ZZZ_PACKAGE_HINT" in p
    assert p.rstrip().endswith("Write the reproduction test:")


def test_build_prompt_without_gen_hint_unchanged():
    p = build_prompt("the issue", "none", {})
    assert "ZZZ" not in p
    assert p.rstrip().endswith("Write the reproduction test:")
