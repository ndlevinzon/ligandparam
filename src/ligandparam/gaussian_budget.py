"""Core-budget helpers for concurrent Gaussian jobs.

Implemented once in :func:`ffpopt.fast_wavefront.split_nproc_for_items`
(breadth-first allocation); this module is a thin, domain-named alias.
"""

from __future__ import annotations

from ffpopt.fast_wavefront import split_nproc_for_items


def split_gaussian_job_budget(total_cores: int, n_jobs: int) -> tuple[int, int]:
    """Split a core budget across concurrent Gaussian jobs.

    Prefers as many concurrent jobs as possible up to ``min(total_cores, n_jobs)``,
    with ``n_workers * nproc_per_job <= total_cores``.

    Returns
    -------
    tuple of int
        ``(n_workers, nproc_per_job)`` for the pool and ``%NProc`` header.
    """
    return split_nproc_for_items(total_cores, n_jobs, prefer_depth=False)
