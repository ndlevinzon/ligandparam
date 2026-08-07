"""Core-budget helpers for concurrent Gaussian jobs."""


def split_gaussian_job_budget(total_cores: int, n_jobs: int) -> tuple[int, int]:
    """Split a core budget across concurrent Gaussian jobs.

    Prefers as many concurrent jobs as possible up to ``min(total_cores, n_jobs)``,
    with ``n_workers * nproc_per_job <= total_cores``.

    Parameters
    ----------
    total_cores : int
        Total processor budget (recipe / stage ``nproc``).
    n_jobs : int
        Number of ``.com`` jobs to run.

    Returns
    -------
    tuple of int
        ``(n_workers, nproc_per_job)`` for the pool and ``%NProc`` header.
    """
    total = max(1, int(total_cores))
    n_jobs = max(1, int(n_jobs))
    if n_jobs == 1:
        return 1, total
    n_workers = min(total, n_jobs)
    nproc_per_job = max(1, total // n_workers)
    return n_workers, nproc_per_job
