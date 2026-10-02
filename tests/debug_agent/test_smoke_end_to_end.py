"""End-to-end smoke: run_debug on a real docker container with a scripted LLM
that exercises all five actions (bash, pytest, probe, read, pdb) before concluding.

No external API calls; proves the full loop works against a live container."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from debug_agent import run_debug
from debug_agent.container import Container


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="docker not available")


_REPRO_TEST = """\
def test_case():
    from mypkg.mod import buggy
    assert buggy(10) == 20  # fails: buggy returns x+1, not x*2
"""

_BUGGY_MODULE = """\
def buggy(x):
    return x + 1
"""


@pytest.mark.docker
def test_run_debug_drives_real_container_with_scripted_llm(tmp_path, monkeypatch):
    # Prepare instance dir with a realistic Phase-A layout.
    inst = tmp_path / "instance_smoke"
    inst.mkdir()
    (inst / "problem_statement.txt").write_text(
        "mypkg.mod.buggy returns x+1 but should return x*2."
    )
    accepted = inst / "stage1_reproduction" / "accepted"
    accepted.mkdir(parents=True)
    (accepted / "test_a0.py").write_text(_REPRO_TEST)

    # Launch a container, seed the buggy package and pytest.
    c = Container.launch(image="python:3.11-slim", workdir="/work")
    setup_ok = False
    try:
        r = c.exec_bash("pip install --quiet pytest")
        assert r.returncode == 0, r.combined
        c.exec_bash("mkdir -p /work/mypkg")
        c.write_file("/work/mypkg/__init__.py", "")
        c.write_file("/work/mypkg/mod.py", _BUGGY_MODULE)
        setup_ok = True

        # Script the LLM: one action per round, exercising each tool, then conclude.
        # Each action turn must carry a <reason> — Path B requires it.
        scripted = iter([
            '<reason>list the workspace</reason><action name="bash">ls /work</action>',
            '<reason>read the suspect module</reason>'
            '<action name="read">{"path": "/work/mypkg/mod.py", "start": 1, "n": 20}</action>',
            '<reason>reproduce the failure</reason><action name="pytest">-x _repro/test_a0.py</action>',
            ('<reason>probe the input value</reason>'
             '<action name="probe">{"file": "/work/mypkg/mod.py", '
             '"before_line": 2, "expr": "x", '
             '"run_cmd": "cd /work && python -c \\"from mypkg.mod import buggy; buggy(10)\\""}</action>'),
            ('<conclusion>'
             '<root_cause_files>mypkg/mod.py</root_cause_files>'
             '<root_cause_functions>mypkg.mod.buggy</root_cause_functions>'
             '<reasoning>buggy returns x+1; the probe confirmed x=10 feeds in and the '
             'return was 11, not 20 as the repro requires.</reasoning>'
             '<suggested_fix>replace x+1 with x*2 in buggy().</suggested_fix>'
             '</conclusion>'),
        ])

        # The new conclusion gate requires a real pdb stop + 3 cmds. This
        # smoke is about the broader analyzer/dispatch wiring, not the gate.
        import debug_agent.analyzer as _ana
        monkeypatch.setattr(_ana, "can_conclude", lambda *_a, **_k: True)

        out_path = run_debug.run_one_instance(
            instance_dir=inst,
            output_dir=tmp_path / "rca_out",
            image=None,          # not used; container_factory wins
            model="stub/model",
            container_factory=lambda: c,
            llm_factory=lambda _m: (lambda _p: next(scripted)),
            max_rounds=10,
            wall_timeout=120,
        )
    finally:
        if setup_ok:
            # container_factory reuses `c`; run_debug will call c.terminate() via attach semantics.
            # It was launched here (owned=True), so terminate in run_debug already ran.
            # Call again is idempotent because _owned stays True; docker stop on a stopped
            # container is a no-op with --rm.
            try:
                c.terminate()
            except Exception:
                pass

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["source"] == "debug_agent"
    assert data["candidates"][0]["file"] == "mypkg/mod.py"
    tr = data["debug_report"]["transcript"]
    tool_names = [row[0] for row in tr]
    assert tool_names == ["bash", "read", "pytest", "probe"], tool_names
    # Verify each tool actually did work against the live container:
    bash_out = tr[0][2]
    assert "mypkg" in bash_out
    read_out = tr[1][2]
    assert "return x + 1" in read_out
    pytest_out = tr[2][2]
    assert "test_case" in pytest_out and "rc=" in pytest_out
    probe_out = tr[3][2]
    assert "PROBE" in probe_out and "10" in probe_out
