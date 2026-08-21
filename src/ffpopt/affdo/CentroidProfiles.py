"""Multi-centroid scan starts and smoothest-profile selection.

ConfSearch centroids seed independent HL wavefronts; a Fourier + roughness
score picks the smoothest profile per torsion (AFFDO-style).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def generate_centroid_start_jsons(
    start_json,
    *,
    mol2_path=None,
    nkeep: int = 5,
    nconf: int = 50,
    rmstol: float = 0.5,
    workdir=None,
    logger=None,
) -> list:
    """Write ``start.cent{i}.json`` clones seeded by ConfSearch centroids.

    Uses ``mol2_path`` when given; otherwise attempts to reuse coordinates from
    a ConfSearch on the geometry already in ``start_json`` via an RDKit mol
    built from the Amber parm referenced therein when possible.

    Returns
    -------
    list of pathlib.Path
        One start JSON per retained centroid (including a copy of the original
        as ``start.cent0.json`` when ConfSearch is unavailable).
    """
    from copy import deepcopy

    from ffpopt.affdo.AffdoLog import log_affdo, print_affdo
    from ffpopt.Struct import ListOfStruct
    from ffpopt.confsearch.ConfSearch import ConformerSearch

    def _say(msg: str) -> None:
        if logger is not None:
            log_affdo(logger, "%s", msg)
        else:
            print_affdo(msg)

    start_json = Path(start_json)
    wd = Path(workdir) if workdir is not None else start_json.parent
    wd.mkdir(parents=True, exist_ok=True)

    los0 = ListOfStruct.from_file(str(start_json))
    paths = []

    if mol2_path is not None and Path(mol2_path).is_file():
        nconf_eff = max(int(nconf), int(nkeep))
        out_base = str(wd / "centroids.json")
        _say(
            f"ConfSearch {mol2_path} -> {out_base} "
            f"(nconf={nconf_eff} nkeep={int(nkeep)} rmstol={float(rmstol):g} mmff94)"
        )
        ConformerSearch(
            str(mol2_path),
            out_base,
            nconf=nconf_eff,
            nkeep=int(nkeep),
            mmff94=True,
            maxiter=250,
            rmstol=float(rmstol),
            quiet=True,
        )
        # ConformerSearch writes a multi-struct JSON at out_base
        clus = ListOfStruct.from_file(out_base)
        n_found = len(clus.structs)
        _say(f"ConfSearch retained {n_found} clustered centroid(s)")
        for i, st in enumerate(clus.structs[: int(nkeep)]):
            clone = deepcopy(los0)
            clone.structs = [clone.structs[0]]
            clone.structs[0].Update(
                None, st.data["positions"], st.data.get("forces")
            )
            clone.structs[0].data["name"] = f"centroid_{i}"
            out = wd / f"start.cent{i}.json"
            clone.save(str(out))
            paths.append(out)
        _say(
            "wrote "
            + ", ".join(p.name for p in paths)
            + " (JSON starts; no per-centroid mol2 charge files)"
        )
    else:
        # Fallback: single starting geometry only.
        _say(
            "no mol2 for ConfSearch; using the primary start geometry as the "
            "only centroid"
        )
        out = wd / "start.cent0.json"
        if not out.is_file():
            out.write_bytes(start_json.read_bytes())
        paths.append(out)

    if not paths:
        raise RuntimeError("failed to generate centroid start JSONs")
    return paths


def centroid_energies_from_start(start_cent_json, model_args=None) -> float:
    """Optional single-point energy (eV) for Boltzmann weights; best-effort."""
    try:
        from ffpopt.Struct import ListOfStruct
        from ffpopt.geom.GeomOpt import GeomOpt

        los = ListOfStruct.from_file(str(start_cent_json))
        if model_args is not None:
            los.SetArgs(model_args)
        out = GeomOpt(los, los.structs[0], constraints=None, restraints=None)
        return float(out.data["energy"])
    except Exception:
        return 0.0


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


def score_profile_details(path, *, max_order: int = 3) -> dict:
    """Score one scan file; never raises (``error`` is set on failure)."""
    p = Path(path)
    row = {
        "path": p,
        "score": float("inf"),
        "fourier": float("inf"),
        "roughness": float("inf"),
        "npts": 0,
        "error": None,
    }
    if not p.is_file():
        row["error"] = "missing"
        return row
    try:
        ang, ene = load_profile_angles_energies(p)
        row["npts"] = int(np.asarray(ang).size)
        f = fourier_residual_score(ang, ene, max_order=max_order)
        r = roughness_score(ang, ene)
        row["fourier"] = float(f)
        row["roughness"] = float(r)
        row["score"] = composite_smoothness_score(ang, ene, max_order=max_order)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


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
        ``score_rows`` is a list of detail dicts sorted by ``score``
        (keys: ``path``, ``score``, ``fourier``, ``roughness``, ``npts``,
        ``error``).
    """
    rows = [score_profile_details(path, max_order=max_order) for path in candidates]
    if not rows:
        return None, float("inf"), []
    rows.sort(key=lambda t: t["score"])
    finite = [
        r for r in rows if r.get("error") is None and np.isfinite(r["score"])
    ]
    if not finite:
        return None, float("inf"), rows
    finite.sort(key=lambda t: t["score"])
    return finite[0]["path"], float(finite[0]["score"]), rows


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
