"""Pickle-compat alias for pre-``scan/`` wavefront checkpoints.

Canonical implementation: :mod:`ffpopt.scan.WaveFront`.
Older checkpoints pickle classes as ``ffpopt.WaveFront.*``; this module must
exist so ``pickle.load`` can resolve those names.
"""

from ffpopt.scan.WaveFront import (  # noqa: F401
    Wavefront,
    WavefrontLevel,
    WavefrontNode,
    find_adjacent_dihedrals,
    plot_wavefront,
    run_dihed_wavefront,
    wavefront_loader,
)
