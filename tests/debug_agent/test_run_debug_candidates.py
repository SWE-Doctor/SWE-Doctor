from debug_agent.run_debug import _build_candidates
from debug_agent.analyzer import DebugReport


def test_candidates_paths_are_normalized():
    r = DebugReport()
    r.root_cause_files = [
        "/app/lib/ansible/utils/vars.py",
        "/usr/local/lib/python3.9/dist-packages/ansible/plugins/lookup/password.py",
    ]
    r.root_cause_functions = ["combine_vars", "do_lookup"]
    cands = _build_candidates(r)
    assert cands[0]["file"] == "lib/ansible/utils/vars.py"
    assert cands[1]["file"] == "ansible/plugins/lookup/password.py"
    assert cands[0]["raw_file"] == "/app/lib/ansible/utils/vars.py"
    assert cands[0]["score"] == 1.0
    assert cands[1]["score"] == 0.9


def test_already_relative_no_raw_file_field():
    r = DebugReport()
    r.root_cause_files = ["lib/ansible/utils/vars.py"]
    r.root_cause_functions = ["combine_vars"]
    cands = _build_candidates(r)
    assert cands[0]["file"] == "lib/ansible/utils/vars.py"
    assert "raw_file" not in cands[0]


def test_empty_report():
    r = DebugReport()
    assert _build_candidates(r) == []
