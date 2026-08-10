"""Pickle-compat alias for pre-``scan/`` N-D wavefront checkpoints.

Canonical implementation: :mod:`ffpopt.scan.WaveFrontND`.
"""

from ffpopt.scan.WaveFrontND import (  # noqa: F401
    Wavefront,
    WavefrontLevel,
    WavefrontNode,
    run_dihed_wavefront,
    wavefront_loader,
)
