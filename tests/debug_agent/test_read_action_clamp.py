"""A `read` action with start<1 or n<1 (LLM 0-indexing mistake) must not crash
the instance — actions.dispatch should clamp to the valid 1-indexed range.

Regression: deepseek-pro RCA runs crashed on
    ValueError("start>=1 and n>=1 required")
when the model emitted <action name="read">{"path": ..., "start": 0}</action>,
taking down the whole instance's RCA (run_rca then exited nonzero and the
pipeline skipped stage3 under `set -e`).
"""
from debug_agent.actions import ActionCall, dispatch


class _Result:
    def __init__(self, rc, stdout, stderr=""):
        self.returncode, self.stdout, self.stderr = rc, stdout, stderr


class _RecordingContainer:
    """Records the bash commands dispatch issues; returns canned file content."""

    def __init__(self):
        self.cmds = []

    def exec_bash(self, cmd, timeout=120):
        self.cmds.append(cmd)
        return _Result(0, "FILE CONTENT\n")


def test_read_action_start_zero_clamped_not_raised():
    c = _RecordingContainer()
    out = dispatch(ActionCall(name="read", kwargs={"path": "/app/x.py", "start": 0}), c, {})
    assert "FILE CONTENT" in out
    # start=0 must be clamped to 1 (sed is 1-indexed); never sed -n 0,...
    assert any("sed -n 1," in cmd for cmd in c.cmds), c.cmds


def test_read_action_n_zero_clamped_not_raised():
    c = _RecordingContainer()
    out = dispatch(ActionCall(name="read", kwargs={"path": "/app/x.py", "start": 5, "n": 0}), c, {})
    assert "FILE CONTENT" in out
    # n>=1 → end >= start; the sed range must be a single line "5,5p", never an
    # inverted range like "5,4p".
    assert any("sed -n 5,5p" in cmd for cmd in c.cmds), c.cmds
