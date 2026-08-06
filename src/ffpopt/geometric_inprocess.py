"""In-process geomeTRIC driver with a reusable ASE calculator.

Avoids the per-opt cost of ``python -m ffpopt.geometric_compat`` (interpreter
bootstrap, JSON reload, model reconstruct). Call
:func:`run_geometric_inprocess` with an already-built ASE calculator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np


PathLike = Union[str, Path]


def use_geometric_subprocess() -> bool:
    """True when ``FFPOPT_GEOMETRIC_SUBPROCESS=1`` forces the legacy CLI path."""
    raw = os.environ.get("FFPOPT_GEOMETRIC_SUBPROCESS", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def calc_cache_key(los, struct) -> tuple:
    """Stable key for reusing the base (unrestrained) calculator across opts.

    Restraint *values* are not part of the key: :class:`RestrainedCalculator`
    reads live targets from the restraint list, so wrapping is cheap and can
    be redone each call without rebuilding the underlying model.
    """
    args = getattr(los, "args", None)
    if isinstance(args, Mapping):
        model = str(args.get("model", "sander")).upper()
    else:
        model = str(getattr(args, "model", "sander")).upper()
    charge = struct.GetCharge()
    parm = struct.data.get("parm")
    return (model, charge, parm)


def _wrap_restrained(base, reslist):
    """Wrap ``base`` with :class:`~ffpopt.ase.calculator.RestrainedCalculator`."""
    from .ase.calculator import RestrainedCalculator

    return RestrainedCalculator(base, reslist)


def get_persistent_calc(los, struct, reslist=None):
    """Return a calculator, caching the expensive base model on ``los``.

    Rebuilds the base calc when model / charge / parm change. When
    ``reslist`` is given, wraps a fresh :class:`RestrainedCalculator` around
    the cached base (same pattern as :meth:`ListOfStruct.BuildRestrainedCalc`).
    """
    key = calc_cache_key(los, struct)
    cache = getattr(los, "_ffpopt_calc_cache", None)
    if cache is not None and cache[0] == key:
        base = cache[1]
    else:
        base = los.BuildCalc(struct)
        los._ffpopt_calc_cache = (key, base)
    if reslist is not None:
        return _wrap_restrained(base, reslist)
    return base


def _normalize_converge(converge) -> Optional[list]:
    if converge is None:
        return None
    if isinstance(converge, str):
        parts = converge.split()
        return parts or None
    if isinstance(converge, Sequence):
        return list(converge)
    return [str(converge)]


def run_geometric_inprocess(
    atoms,
    calc,
    *,
    prefix: PathLike,
    constraints_path: Optional[PathLike] = None,
    coordsys: str = "tric",
    maxiter: int = 500,
    converge="set GAU",
    enforce: Optional[float] = None,
    log_ini: Optional[PathLike] = None,
    **extra_kwargs: Any,
):
    """Run geomeTRIC in-process with an existing ASE calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting geometry (Angstrom).
    calc
        ASE calculator (e.g. restrained xTB). Reused across calls.
    prefix : path-like
        Basename for geometric's log / tmpdir / ``_optim.xyz`` outputs.
    constraints_path : path-like, optional
        Geometric ``$set`` constraint file.
    coordsys, maxiter, converge, enforce
        Forwarded to :func:`geometric.optimize.run_optimizer`.
    log_ini : path-like, optional
        Geometric logging INI (ffpopt packaged default when available).

    Returns
    -------
    dict
        ``coords`` (Å, ndarray), ``energy_ha`` (Hartree or None),
        ``progress`` (geometric Molecule trajectory).
    """
    from .geometric_compat import apply_geometric_compat_patches

    apply_geometric_compat_patches()

    from geometric.ase_engine import EngineASE
    from geometric.molecule import Molecule
    from geometric.optimize import run_optimizer

    prefix = str(prefix)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    xyz_path = prefix + ".xyz"
    atoms.write(xyz_path)

    # Load from the written XYZ so Molecule.Data / topology match CLI path.
    # Index [0] matches geometric.ase_engine.main (frame 0).
    M = Molecule(xyz_path)[0]
    engine = EngineASE(M, calc)

    kwargs: dict[str, Any] = {
        "customengine": engine,
        "input": xyz_path,
        "prefix": prefix,
        "coordsys": coordsys,
        "maxiter": int(maxiter),
    }
    conv = _normalize_converge(converge)
    if conv is not None:
        kwargs["converge"] = conv
    if enforce is not None:
        kwargs["enforce"] = float(enforce)
    if constraints_path is not None:
        kwargs["constraints"] = str(constraints_path)
    if log_ini is not None and str(log_ini):
        kwargs["logIni"] = str(log_ini)
    kwargs.update(extra_kwargs)

    progress = run_optimizer(**kwargs)

    coords = np.asarray(progress.xyzs[-1], dtype=float)
    energy_ha = None
    qm_e = getattr(progress, "qm_energies", None)
    if qm_e is not None and len(qm_e) > 0:
        energy_ha = float(qm_e[-1])

    return {
        "coords": coords,
        "energy_ha": energy_ha,
        "progress": progress,
    }
