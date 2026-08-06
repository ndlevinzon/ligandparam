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

A second common abort under frozen dihedrals is Brent's trust-radius root
search raising ``RuntimeError: Not bracketed`` when ``cnorm(0)`` and
``cnorm(step)`` do not straddle the target. That is recovered by rebuilding
the IC and skipping the step (same control-flow as a borked Cartesian
projection).
"""

from __future__ import annotations

import sys

# Emit at most one notice per process; constrained wavefront scans hit these
# paths on many geometries and would otherwise flood Slurm stderr.
_CARTESIAN_FALLBACK_NOTIFIED = False
_BRENT_NOT_BRACKETED_NOTIFIED = False


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


def patch_brent_not_bracketed() -> None:
    """Recover from Brent ``Not bracketed`` by rebuilding IC and skipping the step.

    geomeTRIC's ``optimize_step`` uses Brent's method to match an internal
    step to the Cartesian trust radius. When the endpoints do not bracket a
    root it raises ``RuntimeError('Not bracketed')`` and aborts the whole
    optimize. That is common with frozen dihedrals; treat it like a borked
    IC projection: rebuild coordinates, shrink the trust radius, and let
    ``Optimizer.step`` return early (``dy is None``).
    """

    from geometric.optimize import OPT_STATE, Optimizer

    if getattr(Optimizer.optimize_step, "_ffpopt_brent_patch", False):
        return

    _orig = Optimizer.optimize_step

    def _optimize_step(self):
        try:
            return _orig(self)
        except RuntimeError as exc:
            if "not bracketed" not in str(exc).lower():
                raise
            global _BRENT_NOT_BRACKETED_NOTIFIED
            if not _BRENT_NOT_BRACKETED_NOTIFIED:
                sys.stderr.write(
                    "[ffpopt] geomeTRIC trust-radius Brent search failed "
                    "('Not bracketed'); rebuilding IC and skipping this step "
                    "(further notices suppressed for this process).\n"
                )
                _BRENT_NOT_BRACKETED_NOTIFIED = True
            last_force = bool(getattr(self, "ForceRebuild", False))
            self.ForceRebuild = True
            # Smaller trust → next step less likely to need Brent at all.
            try:
                tmin = float(getattr(self.params, "thre", 1.0e-6))
            except Exception:
                tmin = 1.0e-6
            self.trust = max(tmin, 0.5 * float(self.trust))
            self.checkCoordinateSystem(recover=True, cartesian=last_force)
            self.Iteration -= 1
            self.state = OPT_STATE.SKIP_EVALUATION
            return None

    _optimize_step._ffpopt_brent_patch = True  # type: ignore[attr-defined]
    Optimizer.optimize_step = _optimize_step


def apply_geometric_compat_patches() -> None:
    """Install all ffpopt ↔ geomeTRIC compatibility patches."""
    patch_constrained_cartesian_fallback()
    patch_brent_not_bracketed()


def main(argv: list[str] | None = None) -> None:
    """Entry point used instead of ``geometric-optimize`` from ffpopt."""

    apply_geometric_compat_patches()
    from geometric.optimize import main as geometric_main

    # geometric.optimize.main reads sys.argv; keep CLI parity with geometric-optimize.
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    geometric_main()


if __name__ == "__main__":
    main()
