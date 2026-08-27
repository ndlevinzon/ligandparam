"""Per-node wall-clock timeout for wavefront opts that hang in C (tblite).

A Python ``signal`` / thread cannot interrupt an xTB SCF. Fork a child, join
with a deadline, then SIGTERM/SIGKILL. Linux only (CHPC); Windows runs the
callable in-process. ``FFPOPT_WF_NODE_WALL_SEC=0`` disables.
"""

from __future__ import annotations

import os
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

    import multiprocessing as mp

    ctx = mp.get_context("fork")
    queue = ctx.Queue()

    def _target() -> None:
        os.environ[WALL_CHILD_ENV] = "1"
        try:
            queue.put(("ok", fn()))
        except Exception as exc:
            queue.put(("err", f"{type(exc).__name__}: {exc}"))

    proc = ctx.Process(target=_target)
    proc.start()
    proc.join(wall_sec)
    if proc.is_alive():
        loc = "?"
        if isinstance(job, dict):
            loc = job.get("angle", job.get("rcs", "?"))
        print(
            ascii_for_stdio(
                f"[wavefront] node wall timeout ({wall_sec:g}s) at {loc}; "
                "killing hung opt so the wavefront can drain"
            ),
            flush=True,
        )
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        return timed_out_node_result(
            job,
            message=(
                f"wall_timeout: node exceeded {wall_sec:g}s "
                f"(loc={loc})"
            ),
        )
    if queue.empty():
        return timed_out_node_result(
            job,
            message=(
                f"wall_timeout: child exited with no result "
                f"(exit={proc.exitcode})"
            ),
        )
    kind, payload = queue.get()
    if kind == "ok":
        return payload
    print(
        ascii_for_stdio(f"[wavefront] node child error: {payload}"),
        flush=True,
    )
    return timed_out_node_result(job, message=str(payload))
