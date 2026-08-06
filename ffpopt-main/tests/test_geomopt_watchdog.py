import sys
import time

import pytest

from ffpopt.GeomOpt import _run_geometric_with_watchdog


def _child_writes_then_exits(log_path):
    """Fake geomeTRIC that writes two log lines and exits cleanly."""
    code = (
        "import time;"
        f"open({str(log_path)!r}, 'w').write('step 1\\n');"
        "time.sleep(0.1);"
        f"open({str(log_path)!r}, 'a').write('step 2\\n')"
    )
    return [sys.executable, "-c", code]


def _child_writes_bmatrix_then_hangs(log_path):
    """Fake geomeTRIC that emits the wedge marker and then refuses to exit."""
    code = (
        f"open({str(log_path)!r}, 'w').write("
        "'warning: more than 1000 B-matrices stored, memory leaks likely\\n');"
        "import time; time.sleep(30)"
    )
    return [sys.executable, "-c", code]


def _child_sleeps_silently():
    """Fake geomeTRIC that never touches the log."""
    return [sys.executable, "-c", "import time; time.sleep(30)"]


def test_normal_exit_returns_cleanly(tmp_path):
    log = tmp_path / "opt.log"
    _run_geometric_with_watchdog(
        _child_writes_then_exits(log),
        str(log),
        poll_interval_sec=0.1,
        stall_timeout_sec=10,
    )
    assert log.exists()
    assert "step 2" in log.read_text()


def test_bmatrix_pattern_triggers_wedge(tmp_path):
    log = tmp_path / "opt.log"
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="B-matrices"):
        _run_geometric_with_watchdog(
            _child_writes_bmatrix_then_hangs(log),
            str(log),
            poll_interval_sec=0.1,
            stall_timeout_sec=30,
        )
    # Should trip on the very first log poll, well before stall_timeout.
    assert time.monotonic() - t0 < 5


def test_stalled_log_triggers_timeout(tmp_path):
    log = tmp_path / "opt.log"
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="stalled"):
        _run_geometric_with_watchdog(
            _child_sleeps_silently(),
            str(log),
            poll_interval_sec=0.1,
            stall_timeout_sec=1.5,
        )
    elapsed = time.monotonic() - t0
    assert 1.0 < elapsed < 5
