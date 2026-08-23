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


# --- in-process driver (was geometric_inprocess.py) ---

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from ffpopt.runtime.EnvDefaults import env_bool


PathLike = Union[str, Path]


def use_geometric_subprocess() -> bool:
    """True when ``FFPOPT_GEOMETRIC_SUBPROCESS=1`` forces the legacy CLI path."""
    return env_bool("FFPOPT_GEOMETRIC_SUBPROCESS")


def use_geometric_robust() -> bool:
    """True unless ``FFPOPT_GEOMOPT_ROBUST=0`` disables the recovery ladder."""
    return env_bool("FFPOPT_GEOMOPT_ROBUST")


def is_geomopt_not_converged(exc: BaseException) -> bool:
    """True for geomeTRIC ``GeomOptNotConvergedError`` (and close cousins)."""
    name = type(exc).__name__
    if "NotConverged" in name or name == "GeomOptNotConvergedError":
        return True
    msg = str(exc).lower()
    return "failed to converge" in msg or "not converged" in msg


def is_linear_torsion_error(exc: BaseException) -> bool:
    """True for geomeTRIC ``LinearTorsionError`` (re-export for callers)."""
    from ffpopt.geom.LinearTorsion import is_linear_torsion_error as _impl

    return _impl(exc)


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
    qdpi_opt = None
    try:
        from ffpopt.runtime.FastWavefront import (
            is_qdpi2_model,
            qdpi2_opt_components,
        )

        if is_qdpi2_model(model):
            qdpi_opt = qdpi2_opt_components()
    except Exception:
        qdpi_opt = None
    return (model, charge, parm, qdpi_opt)


def _wrap_restrained(base, reslist):
    """Wrap ``base`` with :class:`~ffpopt.ase.Calculator.RestrainedCalculator`."""
    from ffpopt.ase.Calculator import RestrainedCalculator

    return RestrainedCalculator(base, reslist)


def get_persistent_calc(los, struct, reslist=None):
    """Return a calculator, caching the expensive base model on ``los``.

    Rebuilds the base calc when model / charge / parm change. When
    ``reslist`` is given, wraps a fresh :class:`RestrainedCalculator` around
    the cached base (same pattern as :meth:`ListOfStruct.BuildRestrainedCalc`).

    For QDpi2 under ``--fast``, the cached base may use XTB-only forces during
    optimization (see ``FFPOPT_QDPI2_OPT``); call
    :func:`refine_qdpi2_energy` afterward for a full HL single-point.
    """
    key = calc_cache_key(los, struct)
    cache = getattr(los, "_ffpopt_calc_cache", None)
    if cache is not None and cache[0] == key:
        base = cache[1]
    else:
        base = los.BuildCalc(struct)
        try:
            from ffpopt.runtime.FastWavefront import (
                is_qdpi2_model,
                qdpi2_opt_components,
            )

            args = getattr(los, "args", None)
            model = (
                args.get("model")
                if isinstance(args, Mapping)
                else getattr(args, "model", None)
            )
            if is_qdpi2_model(model):
                components = qdpi2_opt_components()
                inner = getattr(base, "calc", base)
                if hasattr(inner, "force_components"):
                    inner.force_components = components
        except Exception:
            pass
        los._ffpopt_calc_cache = (key, base)
    if reslist is not None:
        return _wrap_restrained(base, reslist)
    return base


def get_full_qdpi2_calc(los, struct, reslist=None):
    """Calculator forced to full QDpi2 (DeepPot + XTB), bypassing cheap opt mode."""
    key = ("QDPI2_FULL", struct.GetCharge(), struct.data.get("parm"))
    cache = getattr(los, "_ffpopt_qdpi2_full_cache", None)
    if cache is not None and cache[0] == key:
        base = cache[1]
    else:
        base = los.BuildCalc(struct)
        inner = getattr(base, "calc", base)
        if hasattr(inner, "force_components"):
            inner.force_components = "both"
        los._ffpopt_qdpi2_full_cache = (key, base)
    if reslist is not None:
        return _wrap_restrained(base, reslist)
    return base


def refine_qdpi2_energy(los, struct):
    """Re-score an optimized geometry with full QDpi2 (eV bare potential).

    Returns ``None`` when refinement is disabled or the model is not QDpi2.
    """
    from ffpopt.runtime.FastWavefront import (
        is_qdpi2_model,
        qdpi2_refine_energy_after_opt,
    )

    args = getattr(los, "args", None)
    model = (
        args.get("model")
        if isinstance(args, Mapping)
        else getattr(args, "model", None)
    )
    if not is_qdpi2_model(model) or not qdpi2_refine_energy_after_opt():
        return None

    atoms = struct.GetASEAtoms()
    calc = get_full_qdpi2_calc(los, struct)
    atoms.calc = calc
    try:
        calc.reset()
    except Exception:
        pass
    ene = float(atoms.get_potential_energy())
    # Strip restraint penalties if present (matching bare_potential_energy).
    rests = getattr(struct, "restraints", None)
    if rests is not None and len(rests) > 0:
        crds = np.asarray(struct.data["positions"], dtype=float)
        for rst in rests:
            e2, _ = rst.GetValueAndGradients(crds)
            ene -= float(e2)
    return ene


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
    apply_geometric_compat_patches()

    from geometric.ase_engine import EngineASE
    from geometric.molecule import Molecule
    from geometric.optimize import run_optimizer

    geometry_bonds = extra_kwargs.pop("geometry_bonds", None)
    geometry_numbers = extra_kwargs.pop("geometry_numbers", None)
    if geometry_bonds:
        from ffpopt.geom.Constraints import wrap_calculator_geometry_guard

        nums = (
            geometry_numbers
            if geometry_numbers is not None
            else atoms.get_atomic_numbers()
        )
        calc = wrap_calculator_geometry_guard(calc, geometry_bonds, nums)

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


def geometric_prefix_from_node_pkl(node_pkl: PathLike) -> str:
    """Wavefront node pickle ``foo_node.pckl`` -> geomeTRIC prefix ``foo_node_geom``."""
    return str(Path(node_pkl).with_suffix("")) + "_geom"


def _rm_path(path: Path) -> bool:
    """Unlink a file or ``rmtree`` a directory. Returns True if something was removed."""
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
            return True
        if path.is_dir():
            shutil.rmtree(path)
            return True
    except OSError:
        return False
    return False


def cleanup_geometric_scratch(prefix: PathLike, *, keep_optim: bool = False) -> int:
    """Remove geomeTRIC sidecars for one opt prefix.

    Deletes ``{prefix}.tmp/``, ``{prefix}.nsf`` / ``.log`` / ``.xyz`` / ``.json``
    / ``.cons.inp``, recovery-ladder ``{prefix}.r*`` paths, and (unless
    ``keep_optim``) ``{prefix}_optim.xyz``. Does **not** touch a shared
    ``log.nsf`` in the parent directory — that is handled by
    :func:`sweep_geometric_scratch_dir` when no workers are running.
    """
    prefix = os.path.normpath(str(prefix))
    n = 0
    if _rm_path(Path(prefix + ".tmp")):
        n += 1
    for suf in (".log", ".nsf", ".xyz", ".json", ".cons.inp"):
        if _rm_path(Path(prefix + suf)):
            n += 1
    if not keep_optim and _rm_path(Path(prefix + "_optim.xyz")):
        n += 1
    parent = Path(prefix).parent
    stem = Path(prefix).name
    if parent.is_dir():
        try:
            for p in parent.iterdir():
                name = p.name
                if not name.startswith(stem + ".r"):
                    continue
                if keep_optim and p.is_file() and name.endswith("_optim.xyz"):
                    continue
                if _rm_path(p):
                    n += 1
        except OSError:
            pass
    return n


def sweep_geometric_scratch_dir(
    directory: PathLike,
    *,
    keep_optim_prefixes: Optional[Sequence[PathLike]] = None,
    recursive: bool = True,
) -> int:
    """Remove leftover geomeTRIC ``.nsf`` logs, ``*.tmp`` dirs, and ``*_geom*`` sidecars.

    ``keep_optim_prefixes`` retains ``{prefix}_optim.xyz`` (and recovery
    ``{prefix}.r*_optim.xyz``) so an incomplete node can warm-start.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    keep = {os.path.normpath(str(p)) for p in (keep_optim_prefixes or [])}
    n = 0

    def _keep_optim_file(path: Path) -> bool:
        if not path.name.endswith("_optim.xyz"):
            return False
        pref = os.path.normpath(str(path)[: -len("_optim.xyz")])
        if pref in keep:
            return True
        # recovery: {prefix}.r2_optim.xyz
        name = path.name
        if ".r" in name and name.endswith("_optim.xyz"):
            stem = name[: -len("_optim.xyz")]
            for k in keep:
                if stem.startswith(Path(k).name + ".r"):
                    return True
        return False

    for root, dirs, files in os.walk(directory, topdown=True):
        root_p = Path(root)
        next_dirs = []
        for d in dirs:
            dp = root_p / d
            if d.endswith(".tmp"):
                if _rm_path(dp):
                    n += 1
                continue
            if d == "tmpfiles":
                try:
                    for child in dp.iterdir():
                        if child.name.startswith("tmp."):
                            if _rm_path(child):
                                n += 1
                except OSError:
                    pass
                continue
            if recursive:
                next_dirs.append(d)
        dirs[:] = next_dirs

        for f in files:
            fp = root_p / f
            if f.endswith(".nsf"):
                if _rm_path(fp):
                    n += 1
                continue
            if "_geom" not in f:
                continue
            if _keep_optim_file(fp):
                continue
            if f.endswith((".log", ".xyz", ".json", ".cons.inp", ".nsf")):
                if _rm_path(fp):
                    n += 1

        if not recursive:
            break
    return n


def _recovery_attempts(
    *,
    coordsys: str,
    maxiter: int,
    converge,
    enforce: Optional[float],
) -> list[dict[str, Any]]:
    """Ordered recovery attempts for hard constrained optimizations."""
    from ffpopt.runtime.FastWavefront import fast_recovery_ladder

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
    if env_bool("FFPOPT_GEOMOPT_SOFT_MAXITER"):
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
    from ffpopt.runtime.Console import ascii_for_stdio

    sys.stderr.write(
        ascii_for_stdio(
            f"[ffpopt] geomeTRIC attempt failed ({type(exc).__name__}: {exc}); "
            f"retrying with recovery '{label}'\n"
        )
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
            # Linear torsion is not cured by looser converge / coordsys — stop
            # the ladder so GeomOpt can run the dedicated ASE rescue.
            if is_linear_torsion_error(exc):
                raise
            from ffpopt.geom.Constraints import BrokenGeometryError

            if isinstance(exc, BrokenGeometryError):
                raise
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


if __name__ == "__main__":
    main()
