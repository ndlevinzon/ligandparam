"""Spawn Pool whose workers are non-daemon (may nest child pools).

Prefer a single parallelism axis (fragment **or** bond **or** wavefront) via
:func:`ffpopt.runtime.fast_wavefront.split_nproc_for_items` with
``flatten_nested=True``. Nested non-daemon pools remain available when a
parent worker must open a child wavefront pool.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

_SpawnProcessBase = __import__("multiprocessing").get_context("spawn").Process

# Mark processes that already sit inside a spawn worker so children can avoid
# opening yet another nested Pool of Pools.
_NESTED_SPAWN_ENV = "FFPOPT_IN_SPAWN_WORKER"


class NonDaemonSpawnProcess(_SpawnProcessBase):
    """Spawn process that ignores daemon=True (may create nested pools)."""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NonDaemonSpawnContext:
    """Context wrapper so ``Pool`` uses :class:`NonDaemonSpawnProcess`."""

    def __init__(self):
        import multiprocessing as mp

        self._ctx = mp.get_context("spawn")
        self.Process = NonDaemonSpawnProcess

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def in_spawn_worker() -> bool:
    """True when this process was started as an ffpopt spawn pool worker."""
    return os.environ.get(_NESTED_SPAWN_ENV, "").strip() == "1"


def _mark_spawn_worker() -> None:
    os.environ[_NESTED_SPAWN_ENV] = "1"


def make_nondaemon_spawn_pool(
    n_workers: int,
    *,
    initializer: Optional[Callable[..., Any]] = None,
    initargs: tuple = (),
):
    """Spawn ``Pool`` whose workers are non-daemon (may nest wavefront pools).

    ``multiprocessing.get_context(...).Pool`` is a factory method, not a class,
    so it cannot be subclassed. Pass ``multiprocessing.pool.Pool`` a context
    whose ``Process`` is :class:`NonDaemonSpawnProcess` instead.

    Workers set ``FFPOPT_IN_SPAWN_WORKER=1`` so nested bond/wavefront code can
    flatten rather than opening a third spawn level.
    """
    from multiprocessing.pool import Pool

    user_init = initializer
    user_args = tuple(initargs)

    def _init(*args):
        _mark_spawn_worker()
        if user_init is not None:
            user_init(*args)

    return Pool(
        processes=max(1, int(n_workers)),
        context=NonDaemonSpawnContext(),
        initializer=_init if user_init is not None else _mark_spawn_worker,
        initargs=user_args if user_init is not None else (),
    )


def make_wavefront_spawn_pool(
    n_workers: int,
    *,
    initializer: Callable[..., Any],
    initargs: tuple,
):
    """Wavefront node pool (daemon spawn OK; nested under fragment/bond)."""
    import multiprocessing as mp

    return mp.get_context("spawn").Pool(
        processes=max(1, int(n_workers)),
        initializer=initializer,
        initargs=initargs,
    )
