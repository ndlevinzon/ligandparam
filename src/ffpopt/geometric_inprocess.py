"""In-process geomeTRIC driver with a reusable ASE calculator.

Avoids the per-opt cost of ``python -m ffpopt.geometric_compat`` (interpreter
bootstrap, JSON reload, model reconstruct). Call
:func:`run_geometric_inprocess` with an already-built ASE calculator.

:func:`run_geometric_robust` adds a recovery ladder for difficult constrained
cases (looser converge, alternate coordsys, soft maxiter accept).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np


PathLike = Union[str, Path]


def use_geometric_subprocess() -> bool:
    """True when ``FFPOPT_GEOMETRIC_SUBPROCESS=1`` forces the legacy CLI path."""
    raw = os.environ.get("FFPOPT_GEOMETRIC_SUBPROCESS", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def use_geometric_robust() -> bool:
    """True unless ``FFPOPT_GEOMOPT_ROBUST=0`` disables the recovery ladder."""
    raw = os.environ.get("FFPOPT_GEOMOPT_ROBUST", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _env_truthy(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def is_geomopt_not_converged(exc: BaseException) -> bool:
    """True for geomeTRIC ``GeomOptNotConvergedError`` (and close cousins)."""
    name = type(exc).__name__
    if "NotConverged" in name or name == "GeomOptNotConvergedError":
        return True
    msg = str(exc).lower()
    return "failed to converge" in msg or "not converged" in msg


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


def write_plain_xyz(path: PathLike, atoms) -> None:
    """Write element + xyz only (no charge columns) for geomeTRIC Molecule.

    ASE's default ``Atoms.write()`` emits extended XYZ with
    ``initial_charges`` as a 4th column, which geomeTRIC rejects.
    """
    import ase.io

    ase.io.write(str(path), atoms, format="xyz", parallel=False)


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
        ``coords`` (Ang, ndarray), ``energy_ha`` (Hartree or None),
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
    write_plain_xyz(xyz_path, atoms)

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


def read_last_optim_xyz(prefix: PathLike) -> Optional[np.ndarray]:
    """Return last-frame coordinates (Ang) from ``{prefix}_optim.xyz``, if any."""
    path = Path(str(prefix) + "_optim.xyz")
    if not path.is_file():
        return None
    try:
        import ase.io

        frames = ase.io.read(str(path), index=":", format="xyz")
        if not frames:
            return None
        last = frames[-1] if isinstance(frames, list) else frames
        return np.asarray(last.get_positions(), dtype=float)
    except Exception:
        return None


def _recovery_attempts(
    *,
    coordsys: str,
    maxiter: int,
    converge,
    enforce: Optional[float],
) -> list[dict[str, Any]]:
    """Ordered recovery attempts for hard constrained optimizations."""
    from ffpopt.runtime.fast_wavefront import fast_recovery_ladder

    primary_conv = _normalize_converge(converge) or ["set", "GAU"]
    loose = ["set", "GAU_LOOSE"]
    soft = ["set", "GAU_LOOSE", "maxiter"]
    maxiter_i = int(maxiter)
    # Full ladder boosts hard; fast mode caps extra work (primary → loose → soft).
    if fast_recovery_ladder():
        boost = max(maxiter_i, min(int(1.5 * maxiter_i), 300))
    else:
        boost = max(maxiter_i, int(1.5 * maxiter_i), 750)

    attempts: list[dict[str, Any]] = [
        {
            "label": "primary",
            "coordsys": coordsys,
            "maxiter": maxiter_i,
            "converge": primary_conv,
            "enforce": enforce,
        },
        {
            "label": "loose",
            "coordsys": coordsys,
            "maxiter": boost,
            "converge": loose,
            "enforce": enforce,
        },
    ]
    if not fast_recovery_ladder():
        alts = []
        for cs in ("dlc", "hdlc", "tric"):
            if cs != coordsys and cs not in alts:
                alts.append(cs)
        # Alternate internal coordinate systems (common rescue under frozen torsions).
        for cs in alts[:2]:
            attempts.append(
                {
                    "label": f"{cs}-loose",
                    "coordsys": cs,
                    "maxiter": boost,
                    "converge": loose,
                    "enforce": enforce,
                }
            )
    # Last geometric resort: treat maxiter as success (keeps best frame).
    if _env_truthy("FFPOPT_GEOMOPT_SOFT_MAXITER", True):
        attempts.append(
            {
                "label": "soft-maxiter",
                "coordsys": coordsys,
                "maxiter": boost,
                "converge": soft,
                "enforce": enforce,
            }
        )
    return attempts


def _geom_retry_note(label: str, exc: BaseException) -> None:
    sys.stderr.write(
        f"[ffpopt] geomeTRIC attempt failed ({type(exc).__name__}: {exc}); "
        f"retrying with recovery '{label}'\n"
    )


def run_geometric_robust(
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
    """Run geomeTRIC with a recovery ladder for difficult cases.

    On :class:`~geometric.errors.GeomOptNotConvergedError` (and similar),
    restarts from the last ``_optim.xyz`` frame with looser converge criteria,
    alternate ``coordsys``, and optionally ``converge maxiter`` so a nearly
    relaxed geometry is accepted rather than aborting the whole scan node.

    Disable with ``FFPOPT_GEOMOPT_ROBUST=0``. Soft maxiter accept can be
    disabled with ``FFPOPT_GEOMOPT_SOFT_MAXITER=0``. With
    ``FFPOPT_FAST_WAVEFRONT=1`` or ``FFPOPT_GEOMOPT_FAST_RECOVERY=1``, the
    ladder is shortened to primary → loose → soft-maxiter (no alt coordsys).
    """
    import copy

    if not use_geometric_robust():
        return run_geometric_inprocess(
            atoms,
            calc,
            prefix=prefix,
            constraints_path=constraints_path,
            coordsys=coordsys,
            maxiter=maxiter,
            converge=converge,
            enforce=enforce,
            log_ini=log_ini,
            **extra_kwargs,
        )

    attempts = _recovery_attempts(
        coordsys=str(coordsys),
        maxiter=int(maxiter),
        converge=converge,
        enforce=enforce,
    )
    work = copy.deepcopy(atoms)
    last_exc: Optional[BaseException] = None

    for i, att in enumerate(attempts):
        # Fresh prefix per attempt so logs / optim.xyz do not collide mid-retry.
        att_prefix = str(prefix) if i == 0 else f"{prefix}.r{i}"
        try:
            result = run_geometric_inprocess(
                work,
                calc,
                prefix=att_prefix,
                constraints_path=constraints_path,
                coordsys=att["coordsys"],
                maxiter=att["maxiter"],
                converge=att["converge"],
                enforce=att["enforce"],
                log_ini=log_ini,
                **extra_kwargs,
            )
            if i > 0:
                sys.stderr.write(
                    f"[ffpopt] geomeTRIC recovered with attempt '{att['label']}'\n"
                )
            result["recovery"] = att["label"]
            return result
        except Exception as exc:
            last_exc = exc
            # Prefer last trajectory frame when available (even for hard errors).
            last = read_last_optim_xyz(att_prefix)
            if last is not None and last.shape == work.get_positions().shape:
                work.set_positions(last)
            if i + 1 >= len(attempts):
                break
            next_label = attempts[i + 1]["label"]
            # Always escalate on non-convergence; also try recovery for IC /
            # Brent / structure issues that patches did not fully absorb.
            if is_geomopt_not_converged(exc) or i == 0:
                _geom_retry_note(next_label, exc)
                continue
            # Unknown hard failure on a later attempt: still try remaining ladder.
            _geom_retry_note(next_label, exc)
            continue

    assert last_exc is not None
    raise last_exc
