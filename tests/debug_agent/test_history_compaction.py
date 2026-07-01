from debug_agent.history_compaction import compact_pdb_turns


def _pair(assistant, user):
    return [{"role": "assistant", "content": assistant},
            {"role": "user", "content": user}]


def test_compact_replaces_pdb_cmd_chunks_with_summary():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "kickoff"},
        *_pair("…", "TOOL (pytest):\n…"),
        *_pair('<action name="pdb_cmd">{"cmd":"b m.py:5"}</action>',
               "TOOL (pdb_cmd):\ncurrent_frame: m.py:5 in f\n--- TRANSCRIPT ---\n(Pdb) (Pdb) "),
        *_pair('<action name="pdb_cmd">{"cmd":"p x"}</action>',
               "TOOL (pdb_cmd):\ncurrent_frame: m.py:5 in f\n--- TRANSCRIPT ---\nNone\n(Pdb) "),
        *_pair('<action name="pdb_cmd">{"cmd":"n"}</action>',
               "TOOL (pdb_cmd):\ncurrent_frame: m.py:6 in f\n--- TRANSCRIPT ---\n(Pdb) "),
        *_pair('<action name="bash">ls</action>', "TOOL (bash):\nrc=0\n…"),
    ]
    out = compact_pdb_turns(messages, fold_first_n_pdb_turns=3)
    joined = "\n".join(m["content"] for m in out)
    # system + kickoff preserved at the front
    assert out[0]["role"] == "system" and out[1]["role"] == "user"
    assert "<pdb_session_summary>" in joined
    assert "m.py:5" in joined
    assert "m.py:6" in joined
    assert "p x = None" in joined or "x = None" in joined
    # bash turn (pre-existing) must be preserved
    assert any("TOOL (bash):" in m["content"] for m in out)
    # last pdb turn was inside the fold — folded copy disappears, summary stays.
    assert sum(1 for m in out if "TOOL (pdb_cmd)" in m["content"]) == 0
