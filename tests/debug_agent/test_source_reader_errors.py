from dataclasses import dataclass

from debug_agent.source_reader import grep_src, read


@dataclass
class _Exec:
    returncode: int
    stdout: str
    stderr: str


class FakeContainer:
    def __init__(self, rc=0, stdout="", stderr=""):
        self._r = _Exec(rc, stdout, stderr)
        self.last_cmd = None

    def exec_bash(self, cmd, timeout=120):
        self.last_cmd = cmd
        return self._r


def test_read_missing_file_returns_explicit_error():
    c = FakeContainer(rc=1, stdout="", stderr="sed: can't read /x: No such file or directory")
    out = read(c, "/x", start=1, n=10)
    assert "ERROR" in out
    assert "/x" in out


def test_read_empty_but_ok_returns_explicit_empty_marker():
    c = FakeContainer(rc=0, stdout="", stderr="")
    out = read(c, "/a.py", start=9999, n=10)
    assert out.strip() != ""
    assert "empty" in out.lower() or "past end" in out.lower()


def test_grep_no_match_reports_zero_hits():
    c = FakeContainer(rc=0, stdout="", stderr="")
    out = grep_src(c, "xyz", "/a.py")
    assert "0 hits" in out or "no match" in out.lower()
