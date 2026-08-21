"""Specialized geomopt for constrained torsions with near-linear bends.

geomeTRIC raises ``LinearTorsionError`` when a frozen dihedral A-B-C-D has
angle A-B-C or B-C-D near 180 deg (torsion undefined). This module unkinks those
bends slightly, holds them off-linear during the opt, and freezes the target
dihedral via ASE ``FixInternals`` (Cartesian Lagrange constraints), which is
more robust than geomeTRIC ICs for this edge case.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable, Optional, Sequence

import numpy as np

# Match geomeTRIC docs ("175 degrees or higher") and the user's ~178 deg failures.
LINEAR_BEND_THRESHOLD_DEG = 175.0
# Hold bends here so the torsion stays well-defined during the rescue opt.
UNKINK_TARGET_DEG = 170.0
# Soft angle spring reserved for a future restraint-only variant.
ANGLE_HOLD_K = 50.0


def log_linear_torsion(msg: str) -> None:
    """Write a linear-torsion log line to stderr as ASCII.

    Avoids ``UnicodeEncodeError`` / mojibake when the process console is not
    UTF-8 (common on Windows / latin-1 Slurm ``.out`` files).
    """
    from ffpopt.runtime.Console import ascii_for_stdio

    text = ascii_for_stdio(msg if msg.endswith("\n") else msg + "\n")
    sys.stderr.write(text)


def is_linear_torsion_error(exc: BaseException) -> bool:
    """True for geomeTRIC ``LinearTorsionError`` (and close message cousins)."""
    name = type(exc).__name__
    if "LinearTorsion" in name:
        return True
    msg = str(exc).lower()
    return "nearly linear" in msg and ("torsion" in msg or "dihedral" in msg)


def _dihedral_constraints(cons: Optional[Iterable[Any]]) -> list[Any]:
    out = []
    for con in cons or []:
        idxs = getattr(con, "idxs", None)
        if idxs is not None and len(idxs) == 4:
            out.append(con)
    return out


def find_near_linear_bends(
    atoms,
    cons: Optional[Iterable[Any]],
    *,
    threshold_deg: float = LINEAR_BEND_THRESHOLD_DEG,
) -> list[dict[str, Any]]:
    """Return near-linear valence bends inside constrained dihedrals.

    Each hit is ``{dihed_idxs, bend_idxs, angle_deg, which}`` where ``which``
    is ``"abc"`` (atoms 0-1-2) or ``"bcd"`` (atoms 1-2-3). ASE angles are
    in ``[0, 180]``; geomeTRIC treats bends near 180 deg as linear.
    """
    thr = float(threshold_deg)
    hits: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for con in _dihedral_constraints(cons):
        i, j, k, l = (int(x) for x in con.idxs)
        for which, bend in (("abc", (i, j, k)), ("bcd", (j, k, l))):
            key = tuple(bend)
            if key in seen:
                continue
            ang = float(atoms.get_angle(*bend))
            if ang >= thr:
                seen.add(key)
                hits.append(
                    {
                        "dihed_idxs": (i, j, k, l),
                        "bend_idxs": bend,
                        "angle_deg": ang,
                        "which": which,
                    }
                )
    return hits


def has_near_linear_dihedral_bend(
    atoms,
    cons: Optional[Iterable[Any]],
    *,
    threshold_deg: float = LINEAR_BEND_THRESHOLD_DEG,
) -> bool:
    return bool(find_near_linear_bends(atoms, cons, threshold_deg=threshold_deg))


def unkink_near_linear_bends(
    atoms,
    bends: Sequence[dict[str, Any]],
    *,
    target_deg: float = UNKINK_TARGET_DEG,
) -> list[Any]:
    """Nudge near-linear bends to ``target_deg``; return angle Constraint objects.

    Uses ASE ``set_angle`` so the torsion becomes well-defined before a
    constrained dihedral opt. Returned constraints can be applied as hard
    FixInternals angles and/or soft angle restraints during the rescue.
    """
    from ffpopt.geom.Constraints import Constraint

    work = atoms
    angle_cons = []
    tgt = float(target_deg)
    for hit in bends:
        a, b, c = (int(x) for x in hit["bend_idxs"])
        before = float(work.get_angle(a, b, c))
        # Always bend *away* from 180 toward the acute side of the target.
        work.set_angle(a, b, c, tgt)
        after = float(work.get_angle(a, b, c))
        log_linear_torsion(
            f"[ffpopt] linear-torsion unkink {a + 1}-{b + 1}-{c + 1}: "
            f"{before:.2f} deg -> {after:.2f} deg (hold {tgt:.1f} deg)"
        )
        angle_cons.append(Constraint("angle", [a, b, c], value=tgt))
    return angle_cons


def run_linear_torsion_ase_opt(
    atoms,
    calc,
    *,
    dihed_cons: Sequence[Any],
    angle_cons: Sequence[Any],
    fmax: float,
    loose_fmax: float,
    max_steps: int,
    logfile=None,
) -> tuple[Any, str]:
    """Optimize with ASE FixInternals (dihedrals + hold angles).

    Returns ``(atoms, recovery_label)``. Label ends with ``-soft`` when only
    the loose force threshold was met.
    """
    from ase.constraints import FixInternals
    from ase.optimize import BFGS, FIRE, LBFGS

    work = atoms.copy()
    work.calc = calc

    bonds = []
    angles = []
    diheds = []
    for con in angle_cons:
        angles.append([float(con.value), list(con.idxs)])
    for con in dihed_cons:
        diheds.append([float(con.value), list(con.idxs)])

    if not diheds:
        raise ValueError("linear-torsion rescue requires at least one dihedral constraint")

    fi = FixInternals(bonds=bonds, angles_deg=angles, dihedrals_deg=diheds)
    work.set_constraint(fi)

    if logfile is None:
        logfile = sys.stderr

    best = work.copy()
    best.calc = calc
    best_fmax = float("inf")
    accepted_how = None

    for name, Opt in (("BFGS", BFGS), ("LBFGS", LBFGS), ("FIRE", FIRE)):
        try:
            work.calc = calc
            work.set_constraint(fi)
            opt = Opt(work, logfile=logfile)
            opt.run(fmax=float(fmax), steps=int(max_steps))
            accepted_how = name
            best = work.copy()
            best.calc = calc
            break
        except Exception as exc:
            forces = work.get_forces()
            cur = float(np.sqrt((forces**2).sum(axis=1).max()))
            log_linear_torsion(
                f"[ffpopt] linear-torsion ASE {name} failed "
                f"({type(exc).__name__}: {exc}); best fmax={cur:.4f}"
            )
            if cur < best_fmax:
                best_fmax = cur
                best = work.copy()
                best.calc = calc
            if cur <= float(loose_fmax):
                accepted_how = f"{name}-soft"
                break
            continue

    if accepted_how is None:
        forces = best.get_forces()
        cur = float(np.sqrt((forces**2).sum(axis=1).max()))
        if cur <= float(loose_fmax):
            accepted_how = "linear-torsion-soft"
        else:
            raise RuntimeError(
                f"linear-torsion ASE rescue did not reach loose fmax="
                f"{loose_fmax} (got {cur:.4f})"
            )

    # Tag as linear-torsion*; soft variants keep "-soft" for wavefront policy.
    if accepted_how.endswith("-soft") or "soft" in accepted_how:
        label = "linear-torsion-soft"
    else:
        label = "linear-torsion"
    return best, label
