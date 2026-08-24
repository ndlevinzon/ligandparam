#!/usr/bin/env python3
"""N-D wavefront public import path (pickle- and CLI-stable facade).

To launch with MPI on amarel::

    mpirun -n $SLURM_NTASKS python3 -m mpi4py `which ffpopt-NDimWavefront.py`

In principle, you should be able to submit this with
``srun --mpi=pmi2`` or ``srun --mpi=pmix``, but on amarel this ends up
launching multiple serial copies of the script without using MPI.
"""

from __future__ import annotations

from ffpopt.scan.WavefrontEngine import (  # noqa: F401
    GetGridNeighbors,
    Wavefront,
    WavefrontLevel,
    WavefrontNode,
    run_ndim_dihed_wavefront as run_dihed_wavefront,
    wavefront_loader,
)

__all__ = [
    "GetGridNeighbors",
    "Wavefront",
    "WavefrontLevel",
    "WavefrontNode",
    "run_dihed_wavefront",
    "wavefront_loader",
]
