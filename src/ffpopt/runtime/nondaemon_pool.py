"""Spawn Pool whose workers are non-daemon (may nest child pools)."""

from __future__ import annotations

_SpawnProcessBase = __import__("multiprocessing").get_context("spawn").Process


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


def make_nondaemon_spawn_pool(n_workers: int):
    """Spawn ``Pool`` whose workers are non-daemon (may nest wavefront pools).

    ``multiprocessing.get_context(...).Pool`` is a factory method, not a class,
    so it cannot be subclassed. Pass ``multiprocessing.pool.Pool`` a context
    whose ``Process`` is :class:`NonDaemonSpawnProcess` instead.
    """
    from multiprocessing.pool import Pool

    return Pool(
        processes=max(1, int(n_workers)),
        context=NonDaemonSpawnContext(),
    )
