"""Per-node wall-clock timeout for wavefront opts that hang in C (tblite).

A Python ``signal`` / thread cannot interrupt an xTB SCF. Fork a child, join
with a deadline, then SIGTERM/SIGKILL. Linux only (CHPC); Windows runs the
callable in-process. ``FFPOPT_WF_NODE_WALL_SEC=0`` disables.

Wavefront pool workers are **daemon** spawn processes, so
``multiprocessing.Process.start()`` raises ``AssertionError: daemonic
processes are not allowed to have children``. ``os.fork`` does not check
that flag.
"""

from __future__ import annotations

import os
import pickle
import signal
import tempfile
import time
from typing import Any, Callable

WALL_CHILD_ENV = "FFPOPT_IN_WALL_CHILD"
NODE_WALL_SEC_DEFAULT = 300.0


class NodeWallTimeout(TimeoutError):
    """A wavefront node exceeded ``FFPOPT_WF_NODE_WALL_SEC``."""


def node_wall_sec() -> float:
    from ffpopt.runtime.EnvDefaults import env_float

    return float(env_float("FFPOPT_WF_NODE_WALL_SEC", NODE_WALL_SEC_DEFAULT))


def timed_out_node_result(job=None, *, message: str) -> dict:
    """Slim worker payload for a wall-clock abort."""
    coords = None
    if isinstance(job, dict):
        coords = job.get("coords")
    return {
        "energy": float("inf"),
        "forces": None,
        "coords": coords,
        "complete": True,
        "error": message,
        "active": False,
        "soft_opt": False,
        "opt_recovery": None,
    }


def fork_wall_available() -> bool:
    return hasattr(os, "fork")


def _wait_pid(pid: int, timeout_sec: float) -> int | None:
    """Return wait status, or ``None`` if ``pid`` is still alive."""
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while True:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return status
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def _kill_pid(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def run_with_node_wall(
    fn: Callable[[], Any],
    *,
    wall_sec: float | None = None,
    job=None,
) -> Any:
    """Run ``fn``; on Linux kill it if it exceeds ``wall_sec``.

    Nested calls (the forked child) skip the wrapper. Returns ``fn()``'s
    value, or a failed slim result when the wall is hit.
    """
    from ffpopt.runtime.Console import ascii_for_stdio

    if wall_sec is None:
        wall_sec = node_wall_sec()
    wall_sec = float(wall_sec)
    if wall_sec <= 0.0 or os.environ.get(WALL_CHILD_ENV, "").strip() == "1":
        return fn()
    if not fork_wall_available():
        return fn()

    fd, path = tempfile.mkstemp(prefix="ffpopt_nwall_", suffix=".pkl")
    os.close(fd)
    try:
        pid = os.fork()
    except OSError:
        try:
            os.unlink(path)
        except OSError:
            pass
        return fn()

    if pid == 0:
        try:
            os.environ[WALL_CHILD_ENV] = "1"
            try:
                os.setsid()
            except OSError:
                pass
            try:
                payload = ("ok", fn())
            except Exception as exc:
                payload = ("err", f"{type(exc).__name__}: {exc}")
            with open(path, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass
        os._exit(0)

    loc = "?"
    if isinstance(job, dict):
        loc = job.get("angle", job.get("rcs", "?"))

    status = _wait_pid(pid, wall_sec)
    if status is None:
        print(
            ascii_for_stdio(
                f"[wavefront] node wall timeout ({wall_sec:g}s) at {loc}; "
                "killing hung opt so the wavefront can drain"
            ),
            flush=True,
        )
        _kill_pid(pid, signal.SIGTERM)
        if _wait_pid(pid, 10) is None:
            _kill_pid(pid, signal.SIGKILL)
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass
        return timed_out_node_result(
            job,
            message=f"wall_timeout: node exceeded {wall_sec:g}s (loc={loc})",
        )

    try:
        with open(path, "rb") as fh:
            kind, payload = pickle.load(fh)
    except Exception as exc:
        try:
            os.unlink(path)
        except OSError:
            pass
        return timed_out_node_result(
            job,
            message=f"wall_timeout: child exited with no result ({exc})",
        )
    try:
        os.unlink(path)
    except OSError:
        pass
    if kind == "ok":
        return payload
    print(
        ascii_for_stdio(f"[wavefront] node child error: {payload}"),
        flush=True,
    )
    return timed_out_node_result(job, message=str(payload))
