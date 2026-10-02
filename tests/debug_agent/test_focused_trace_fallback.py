from unittest.mock import MagicMock

from debug_agent.focused_trace_fallback import (
    PROFILER_SCRIPT, parse_profile_output, run_focused_trace,
)


def test_parse_profile_output_orders_by_last_seen():
    raw = "\n".join([
        "TRACE_ENTER /app/lib/foo/bar.py",
        "TRACE_ENTER /app/lib/foo/bar.py",
        "TRACE_ENTER /app/lib/foo/baz.py",
        "TRACE_ENTER /app/_repro/repro_0.py",
    ])
    files = parse_profile_output(raw, workdir="/app")
    assert files == ["/app/lib/foo/baz.py", "/app/lib/foo/bar.py"]


def test_parse_profile_output_filters_stdlib():
    raw = "\n".join([
        "TRACE_ENTER /usr/lib/python3.11/json/__init__.py",
        "TRACE_ENTER /app/openlibrary/catalog/add_book/__init__.py",
    ])
    assert parse_profile_output(raw, workdir="/app") == [
        "/app/openlibrary/catalog/add_book/__init__.py"
    ]


def test_parse_profile_filters_conftest():
    raw = "\n".join([
        "TRACE_ENTER /app/lib/x.py",
        "TRACE_ENTER /app/conftest.py",
    ])
    assert parse_profile_output(raw, workdir="/app") == ["/app/lib/x.py"]


def test_profiler_script_is_self_contained():
    assert "sys.setprofile" in PROFILER_SCRIPT
    assert "TRACE_ENTER" in PROFILER_SCRIPT


def test_run_focused_trace_invokes_pytest_with_profiler():
    container = MagicMock()
    out = MagicMock()
    out.returncode = 1
    out.stdout = "TRACE_ENTER /app/lib/x.py\nTRACE_ENTER /app/lib/y.py\n"
    out.stderr = ""
    container.exec_bash.return_value = out

    files = run_focused_trace(
        container, workdir="/app", repro_nodeid="_repro/repro_0.py"
    )
    assert files == ["/app/lib/y.py", "/app/lib/x.py"]
    # the final exec_bash call should be the python invocation
    final_cmd = container.exec_bash.call_args_list[-1][0][0]
    assert "python" in final_cmd and "_profile_runner.py" in final_cmd
    # the heredoc earlier should have referenced pytest and the repro nodeid
    heredoc_cmd = container.exec_bash.call_args_list[-2][0][0]
    assert "_repro/repro_0.py" in heredoc_cmd
    assert "pytest" in heredoc_cmd
    # rerunfailures must NOT be disabled (interacts badly with requiredplugins)
    assert "no:rerunfailures" not in heredoc_cmd
    # xdist must NOT be disabled (breaks pytest-rerunfailures>=10 hook validation)
    assert "no:xdist" not in heredoc_cmd
