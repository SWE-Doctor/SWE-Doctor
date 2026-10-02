"""Smoke test: aspect-mode pipeline (aspects=True)."""
import types
import pytest


def test_aspect_mode_output_schema(monkeypatch):
    """Verify aspect-mode pipeline returns required output schema."""
    from reproduction_test_agent.config import ReproTestConfig
    import reproduction_test_agent.pipeline as pipeline_mod
    import reproduction_test_agent.llm as llm_mod

    # --- Stub localize ---
    dummy_localization = {
        "relevant_files": [],
        "file_contents": {},
        "focal_functions": [],
        "test_files": [],
        "test_contents": {},
    }
    monkeypatch.setattr(pipeline_mod, "localize", lambda *a, **kw: dummy_localization)

    # --- Stub llm_call: return aspect JSON or test code based on prompt content ---
    aspects_json = '[{"aspect_id": "a0", "description": "aspect zero"}, {"aspect_id": "a1", "description": "aspect one"}]'

    def fake_llm_call(messages, model, temperature=0.0, max_tokens=4096, *, caller=""):
        # Detect by message content: aspect extractor sends its own _PROMPT with "Split the issue"
        content = " ".join(m.get("content", "") for m in messages)
        if "Split the issue" in content or "aspect_id" in content:
            return aspects_json
        # Otherwise return trivial test code
        return "def test_stub():\n    assert True\n"

    monkeypatch.setattr(llm_mod, "llm_call", fake_llm_call)
    # Patch in every module that imports llm_call directly
    import reproduction_test_agent.generator as gen_mod
    monkeypatch.setattr(gen_mod, "llm_call", fake_llm_call)
    monkeypatch.setattr(pipeline_mod, "llm_call", fake_llm_call)

    # --- Stub execution_augmented_repair: a0 passes, a1 fails ---
    def fake_repair(issue_text, test_code, env, cwd, model, max_attempts, repair_temperature, test_timeout, **kwargs):
        if "a0" in test_code or "stub" in test_code:
            # Return based on which aspect we're processing; use a counter trick via closure
            fake_repair._calls = getattr(fake_repair, "_calls", 0) + 1
            if fake_repair._calls == 1:
                return {
                    "test_code": "final_test_a0",
                    "fails_for_right_reason": True,
                    "attempts": 1,
                    "history": [{"critique": {"failure_category": "assertion_failure"}}],
                }
        return {
            "test_code": "final_test_a1",
            "fails_for_right_reason": False,
            "attempts": 1,
            "history": [{"critique": {"failure_category": "error"}}],
        }

    monkeypatch.setattr(pipeline_mod, "execution_augmented_repair", fake_repair)

    # Use SimpleNamespace as env (nothing in pipeline uses it after localize/repair are stubbed)
    env = types.SimpleNamespace()

    config = ReproTestConfig(aspects=True, max_aspects=2, aspect_masks=("none",))

    result = pipeline_mod.run_pipeline(
        instance_id="test-instance",
        problem_statement="The thing should work correctly.",
        env=env,
        cwd="/tmp",
        config=config,
    )

    # Top-level shape
    assert "aspects_covered" in result, "result missing 'aspects_covered'"
    assert result["aspects_covered"] == 1, f"Expected 1 aspect covered, got {result['aspects_covered']}"

    # accepted entries must carry required keys
    assert len(result["accepted"]) == 1
    entry = result["accepted"][0]
    for key in ("aspect_id", "aspect_description", "final_test", "critic_ok", "score"):
        assert key in entry, f"accepted entry missing key '{key}'"

    assert entry["critic_ok"] is True
    assert entry["score"] > 0
