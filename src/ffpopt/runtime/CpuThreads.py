"""Pin BLAS / OpenMP so nested wavefront workers do not oversubscribe."""

from __future__ import annotations

import os

_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "DP_INTRA_OP_PARALLELISM_THREADS",
    "DP_INTER_OP_PARALLELISM_THREADS",
)


def pin_math_threads(n: int = 1) -> None:
    """Set math-library thread caps if the user has not already exported them.

    tblite / numpy / sander under many spawn workers: 1 thread per process
    (``-n`` is the parallelism). Do not overwrite an explicit ``EXPORT``.
    """
    text = str(max(1, int(n)))
    for key in _THREAD_ENV:
        if key not in os.environ:
            os.environ[key] = text
