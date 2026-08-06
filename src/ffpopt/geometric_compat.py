"""Compatibility shims for running geomeTRIC under ffpopt constraints.

geomeTRIC's constrained optimizer can fail twice to invert an internal-coordinate
step (``IC.bork``). On the second failure it tries to continue in **Cartesian**
coordinates, but that path explicitly raises when constraints are present:

    ValueError: Cannot continue a constrained optimization; please implement
    constrained optimization in Cartesian coordinates

For dihedral scans that is fatal. This module patches recovery so a second
failure **rebuilds the same IC system (TRIC/DLC) with constraints**, which is
what geomeTRIC already does on the first rebuild — keeping constrained
optimization instead of aborting.
"""

from __future__ import annotations

import sys

# Emit at most one notice per process; constrained wavefront scans hit this
# path on many geometries and would otherwise flood Slurm stderr.
_CARTESIAN_FALLBACK_NOTIFIED = False


def patch_constrained_cartesian_fallback() -> None:
    """Replace unsupported Cartesian recovery under constraints with IC rebuild."""

    from geometric.optimize import Optimizer

    if getattr(Optimizer.checkCoordinateSystem, "_ffpopt_constrained_patch", False):
        return

    _orig = Optimizer.checkCoordinateSystem

    def _check_coordinate_system(self, recover=False, cartesian=False):
        if cartesian and self.IC.haveConstraints():
            global _CARTESIAN_FALLBACK_NOTIFIED
            if not _CARTESIAN_FALLBACK_NOTIFIED:
                sys.stderr.write(
                    "[ffpopt] geomeTRIC requested Cartesian IC recovery under "
                    "constraints; rebuilding the same constrained IC system "
                    "instead (further notices suppressed for this process).\n"
                )
                _CARTESIAN_FALLBACK_NOTIFIED = True
            return _orig(self, recover=True, cartesian=False)
        return _orig(self, recover=recover, cartesian=cartesian)

    _check_coordinate_system._ffpopt_constrained_patch = True  # type: ignore[attr-defined]
    Optimizer.checkCoordinateSystem = _check_coordinate_system


def main(argv: list[str] | None = None) -> None:
    """Entry point used instead of ``geometric-optimize`` from ffpopt."""

    patch_constrained_cartesian_fallback()
    from geometric.optimize import main as geometric_main

    # geometric.optimize.main reads sys.argv; keep CLI parity with geometric-optimize.
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    geometric_main()


if __name__ == "__main__":
    main()
