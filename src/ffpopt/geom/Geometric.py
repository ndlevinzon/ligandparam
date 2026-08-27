"""Compatibility shims for running geomeTRIC under ffpopt constraints.

geomeTRIC's constrained optimizer can fail twice to invert an internal-coordinate
step (``IC.bork``). On the second failure it tries to continue in **Cartesian**
coordinates, but that path explicitly raises when constraints are present:

    ValueError: Cannot continue a constrained optimization; please implement
    constrained optimization in Cartesian coordinates

For dihedral scans that is fatal. This module patches recovery so a second
failure **rebuilds the same IC system (TRIC/DLC) with constraints**, which is
what geomeTRIC already does on the first rebuild - keeping constrained
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
            # Smaller trust -> next step less likely to need Brent at all.
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


def patch_geometric_tmp_makedirs() -> None:
    """geomeTRIC ``os.makedirs(prefix+'.tmp')`` has no ``exist_ok`` (FileExistsError)."""
    try:
        import geometric.optimize as go
    except ImportError:
        return
    os_mod = getattr(go, "os", None)
    if os_mod is None:
        return
    current = os_mod.makedirs
    if getattr(current, "_ffpopt_exist_ok", False):
        return

    def _makedirs(name, mode=0o777, exist_ok=False):
        text = str(name).rstrip("/\\")
        if text.endswith(".tmp"):
            exist_ok = True
        return current(name, mode=mode, exist_ok=exist_ok)

    _makedirs._ffpopt_exist_ok = True  # type: ignore[attr-defined]
    os_mod.makedirs = _makedirs


def apply_geometric_compat_patches() -> None:
    """Install all ffpopt <-> geomeTRIC compatibility patches."""
    from ffpopt.runtime.Console import (
        install_ase_futurewarning_filter,
        install_stale_handle_logging_guard,
    )

    install_stale_handle_logging_guard()
    install_ase_futurewarning_filter()
    patch_constrained_cartesian_fallback()
    patch_brent_not_bracketed()
    patch_geometric_tmp_makedirs()


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

    prepare_geometric_tmpdir(prefix)
    try:
        progress = run_optimizer(**kwargs)
    except FileExistsError:
        # Leftover ``{prefix}.tmp`` from a killed worker / k-ramp reuse.
        prepare_geometric_tmpdir(prefix)
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


def prepare_geometric_tmpdir(prefix: PathLike) -> str:
    """Remove leftover ``{prefix}.tmp`` so geomeTRIC ``os.makedirs`` can succeed.

    geomeTRIC calls ``os.makedirs(prefix + '.tmp')`` without ``exist_ok``. A
    killed worker, job restart, or k-ramp reuse leaves that directory and
    raises ``FileExistsError``. Also drops ``{prefix}.r*.tmp`` recovery dirs
    for the same prefix.
    """
    prefix = str(prefix)
    tmpdir = prefix + ".tmp"
    _rm_path(Path(tmpdir))
    parent = Path(prefix).parent
    stem = Path(prefix).name
    if parent.is_dir() and stem:
        try:
            for p in parent.iterdir():
                name = p.name
                if name.startswith(stem + ".r") and name.endswith(".tmp"):
                    _rm_path(p)
        except OSError:
            pass
    return tmpdir


def cleanup_geometric_scratch(prefix: PathLike, *, keep_optim: bool = False) -> int:
    """Remove geomeTRIC sidecars for one opt prefix.

    Deletes ``{prefix}.tmp/``, ``{prefix}.nsf`` / ``.log`` / ``.xyz`` / ``.json``
    / ``.cons.inp``, recovery-ladder ``{prefix}.r*`` paths, and (unless
    ``keep_optim``) ``{prefix}_optim.xyz``. Does **not** touch a shared
    ``log.nsf`` in the parent directory - that is handled by
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
    # Full ladder boosts hard; fast mode caps extra work (primary -> loose -> soft).
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
    ladder is shortened to primary -> loose -> soft-maxiter (no alt coordsys).
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
            # Linear torsion is not cured by looser converge / coordsys - stop
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


# --- subprocess watchdog (moved from GeomOpt) ---

def _linux_process_tree_cputime(pid: int):
    """Sum ``utime+stime`` (jiffies) for ``pid`` and descendants via ``/proc``.

    Returns ``None`` when ``/proc`` is unavailable (non-Linux) or unreadable.
    Used so the geomeTRIC watchdog does not treat a busy energy evaluation as a
    stall merely because the ``.log`` file is quiet.
    """
    import os

    total = 0
    stack = [int(pid)]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            with open(f"/proc/{cur}/stat", "r", encoding="utf-8") as fh:
                data = fh.read()
            # ``comm`` is in parentheses and may contain spaces.
            rparen = data.rfind(")")
            fields = data[rparen + 2 :].split()
            total += int(fields[11]) + int(fields[12])
        except (OSError, IndexError, ValueError):
            continue
        # Prefer the kernel's children list when present.
        try:
            with open(
                f"/proc/{cur}/task/{cur}/children", "r", encoding="utf-8"
            ) as fh:
                stack.extend(int(x) for x in fh.read().split())
            continue
        except OSError:
            pass
        try:
            for name in os.listdir(f"/proc/{cur}/task"):
                with open(
                    f"/proc/{cur}/task/{name}/children", "r", encoding="utf-8"
                ) as fh:
                    stack.extend(int(x) for x in fh.read().split())
        except OSError:
            pass
    return total


def _path_tree_mtime(path: str) -> float:
    """Newest mtime among ``path`` and its immediate children (best-effort)."""
    import os

    try:
        newest = os.path.getmtime(path)
    except OSError:
        return 0.0
    try:
        names = os.listdir(path)
    except OSError:
        return newest
    for name in names:
        try:
            newest = max(newest, os.path.getmtime(os.path.join(path, name)))
        except OSError:
            continue
    return newest


def _geometric_stall_timeout_sec(default: float | None = None) -> float:
    """Stall timeout from ``FFPOPT_GEOMETRIC_STALL_SEC`` (``0`` disables)."""
    from ffpopt.runtime.EnvDefaults import env_float

    if default is not None:
        return float(default)
    return float(env_float("FFPOPT_GEOMETRIC_STALL_SEC"))


def _run_geometric_with_watchdog(
    cmds,
    tmplog,
    activity_dir=None,
    poll_interval_sec=5.0,
    stall_timeout_sec=None,
    bmatrix_wedge_pattern="more than 1000 B-matrices stored",
):
    """Run geometric-optimize as a child process and watch its log for wedged states.

    Raises RuntimeError if the B-matrix accumulation warning appears, or if
    the job shows *no* progress for ``stall_timeout_sec``. Progress is any of:
    log growth, increasing process-tree CPU time, or updates under
    ``activity_dir`` (typically geomeTRIC's ``*.tmp`` folder). Quiet logs alone
    are not enough to declare a stall - ML / xTB gradient evaluations can sit
    for many minutes between log lines.

    Set ``FFPOPT_GEOMETRIC_STALL_SEC=0`` to disable stall kills (B-matrix wedge
    detection remains). On any wedge the child's process group is SIGTERM'd
    (then SIGKILL'd after 10s). The caller is expected to translate the
    exception into a fallback or ``_mark_failed()``.
    """
    import os
    import signal
    import subprocess as subp
    import time

    if stall_timeout_sec is None:
        stall_timeout_sec = _geometric_stall_timeout_sec()

    child_env = os.environ.copy()

    proc = subp.Popen(cmds, text=True, env=child_env,
                      start_new_session=True)

    log_pos = 0
    # carry the tail of the last chunk so a pattern split across two
    # reads is still detected
    log_tail = ""
    last_change = time.monotonic()
    last_cpu = _linux_process_tree_cputime(proc.pid)
    last_dir_mtime = (
        _path_tree_mtime(activity_dir) if activity_dir is not None else 0.0
    )
    wedge_reason = None

    try:
        while proc.poll() is None:
            progressed = False

            try:
                cur_size = os.path.getsize(tmplog)
            except OSError:
                cur_size = 0

            if cur_size > log_pos:
                try:
                    with open(tmplog, "r") as fh:
                        fh.seek(log_pos)
                        chunk = fh.read()
                except OSError:
                    chunk = ""
                if chunk:
                    log_pos += len(chunk)
                    scan_text = log_tail + chunk
                    if bmatrix_wedge_pattern in scan_text:
                        wedge_reason = (
                            f"geomeTRIC wedged: '{bmatrix_wedge_pattern}'"
                            f" detected in {tmplog}"
                        )
                        break
                    log_tail = scan_text[-len(bmatrix_wedge_pattern):]
                    progressed = True
            elif cur_size < log_pos:
                # log was rotated/truncated
                log_pos = 0
                log_tail = ""
                progressed = True

            cpu = _linux_process_tree_cputime(proc.pid)
            if (
                cpu is not None
                and last_cpu is not None
                and cpu > last_cpu
            ):
                last_cpu = cpu
                progressed = True
            elif cpu is not None and last_cpu is None:
                last_cpu = cpu

            if activity_dir is not None:
                dir_mtime = _path_tree_mtime(activity_dir)
                if dir_mtime > last_dir_mtime:
                    last_dir_mtime = dir_mtime
                    progressed = True

            if progressed:
                last_change = time.monotonic()
            elif (
                stall_timeout_sec > 0
                and time.monotonic() - last_change > stall_timeout_sec
            ):
                wedge_reason = (
                    f"geomeTRIC stalled: no log/CPU/tmpdir progress for "
                    f"{stall_timeout_sec}s (log={tmplog})"
                )
                break

            time.sleep(poll_interval_sec)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                proc.wait(timeout=10)
            except subp.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subp.TimeoutExpired:
                    pass

    if wedge_reason is not None:
        raise RuntimeError(wedge_reason)


if __name__ == "__main__":
    main()

