"""Select the smoothest torsional energy profile among candidates.

AFFDO-style composite score: favor low-order Fourier representability and
penalize discontinuous jumps / high-frequency roughness. Used when multiple
centroid-initiated wavefront scans are available for the same torsion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def _wrap_deg(d: np.ndarray) -> np.ndarray:
    return (np.asarray(d, dtype=float) + 180.0) % 360.0 - 180.0


def fourier_residual_score(
    angles_deg: Sequence[float],
    energies: Sequence[float],
    *,
    max_order: int = 3,
) -> float:
    """RMSE of a mean-centered Fourier fit of order ``max_order`` (kcal units)."""
    ang = np.asarray(angles_deg, dtype=float)
    ene = np.asarray(energies, dtype=float)
    if ang.size < max_order * 2 + 2:
        return float("inf")
    y = ene - np.mean(ene)
    rad = np.deg2rad(ang)
    cols = [np.ones_like(rad)]
    for n in range(1, max_order + 1):
        cols.append(np.cos(n * rad))
        cols.append(np.sin(n * rad))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(np.sqrt(np.mean(resid * resid)))


def roughness_score(
    angles_deg: Sequence[float],
    energies: Sequence[float],
    *,
    jump_weight: float = 1.0,
    curv_weight: float = 0.25,
) -> float:
    """Penalize large consecutive jumps and second-difference jaggedness."""
    ang = np.asarray(angles_deg, dtype=float)
    ene = np.asarray(energies, dtype=float)
    if ang.size < 3:
        return float("inf")
    order = np.argsort(ang)
    ang = ang[order]
    ene = ene[order]
    d_ang = _wrap_deg(np.diff(ang))
    # Prefer nearly uniform grids; avoid divide-by-zero.
    d_ang = np.where(np.abs(d_ang) < 1e-6, 1e-6, d_ang)
    dE = np.diff(ene)
    slope = dE / d_ang
    jumps = float(np.sum(np.abs(dE) ** 2))
    curv = float(np.sum(np.diff(slope) ** 2)) if slope.size > 1 else 0.0
    return jump_weight * jumps + curv_weight * curv


def composite_smoothness_score(
    angles_deg: Sequence[float],
    energies: Sequence[float],
    *,
    max_order: int = 3,
    fourier_weight: float = 1.0,
    roughness_weight: float = 0.05,
) -> float:
    """Lower is better. Composite of Fourier residual + roughness."""
    f = fourier_residual_score(angles_deg, energies, max_order=max_order)
    r = roughness_score(angles_deg, energies)
    if not np.isfinite(f) or not np.isfinite(r):
        return float("inf")
    return fourier_weight * f + roughness_weight * r


def load_profile_angles_energies(path) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``(angles, energies)`` from a wavefront ``.dat`` or ``.json``."""
    from ffpopt.scan.ScanAnalysis import load_scan_dat, load_scan_json

    p = Path(path)
    if p.suffix == ".dat":
        return load_scan_dat(p)
    if p.suffix == ".json":
        return load_scan_json(p)
    # Prefer companion .dat when a stem without suffix is given.
    dat = p.with_suffix(".dat")
    if dat.is_file():
        return load_scan_dat(dat)
    return load_scan_json(p.with_suffix(".json"))



def pick_smoothest_profile(
    candidates: Iterable,
    *,
    max_order: int = 3,
) -> Tuple[Optional[object], float, list]:
    """Pick the smoothest profile path among ``candidates``.

    Parameters
    ----------
    candidates
        Paths to ``.dat`` / ``.json`` scan files.

    Returns
    -------
    best_path, best_score, score_rows
        ``score_rows`` is a list of ``(path, score)`` sorted ascending.
    """
    rows = []
    for path in candidates:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            ang, ene = load_profile_angles_energies(p)
            score = composite_smoothness_score(ang, ene, max_order=max_order)
        except Exception:
            score = float("inf")
        rows.append((p, score))
    if not rows:
        return None, float("inf"), []
    rows.sort(key=lambda t: t[1])
    return rows[0][0], float(rows[0][1]), rows


def promote_profile_files(src_stem: Path, dst_stem: Path) -> None:
    """Copy wavefront outputs ``src.*`` onto ``dst.*`` (json/dat/pkl)."""
    import shutil

    src_stem = Path(src_stem)
    dst_stem = Path(dst_stem)
    for suf in (".json", ".dat", ".pkl"):
        sp = src_stem.with_suffix(suf)
        if sp.is_file():
            dp = dst_stem.with_suffix(suf)
            shutil.copy2(sp, dp)
