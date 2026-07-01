from reproduction_test_agent.aspect_extractor import IssueAspect, split_into_aspects


def test_split_returns_one_aspect_for_trivial_issue():
    llm = lambda _p: '[{"aspect_id": "a0", "description": "TypeError on None"}]'
    out = split_into_aspects("foo() crashes on None", max_aspects=3, llm=llm)
    assert out == [IssueAspect("a0", "TypeError on None")]


def test_split_caps_at_max_aspects():
    llm = lambda _p: '[' + ",".join(f'{{"aspect_id":"a{i}","description":"d{i}"}}' for i in range(5)) + ']'
    assert len(split_into_aspects("x", max_aspects=2, llm=llm)) == 2


def test_split_falls_back_to_single_aspect_on_empty_llm_json():
    llm = lambda _p: '[]'
    out = split_into_aspects("fallback issue text", max_aspects=3, llm=llm)
    assert len(out) == 1
    assert out[0].aspect_id == "a0"
    assert "fallback" in out[0].description
