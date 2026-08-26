"""Shared wavefront node helpers (1-D and N-D).

Keep scan engines separate; put duplicated IPC / soft-opt / checkpoint logic
here so it is written once.
"""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np


def clone_struct_geometry(struct, coords, ene=0.0, frcs=None):
    """Prefer ``Struct.clone_geometry``; fall back to deepcopy for test doubles."""
    clone = getattr(struct, "clone_geometry", None)
    if callable(clone):
        return clone(coords=coords, ene=ene, frcs=frcs)
    out = copy.deepcopy(struct)
    out.Update(ene, np.asarray(coords, dtype=float), frcs)
    return out


def clear_los_calc(los) -> None:
    """Drop live calculators so workers rebuild (and cache) in-process."""
    clearer = getattr(los, "clear_runtime_caches", None)
    if callable(clearer):
        clearer()
        return
    calc = getattr(los, "calc", None)
    if calc is not None:
        try:
            calc.reset()
        except Exception:
            pass
        los.calc = None
    if hasattr(los, "_ffpopt_calc_cache"):
        los._ffpopt_calc_cache = None
    if hasattr(los, "_ffpopt_qdpi2_full_cache"):
        los._ffpopt_qdpi2_full_cache = None
    if hasattr(los, "_ffpopt_mm_preopt_los"):
        los._ffpopt_mm_preopt_los = None


def ensure_soft_opt_attrs(node: Any) -> None:
    """Fill soft-opt fields missing from older node pickles / checkpoints."""
    if not hasattr(node, "soft_opt"):
        node.soft_opt = False
    if not hasattr(node, "opt_recovery"):
        node.opt_recovery = None


def tag_opt_recovery_on_geom(opt_geom: Any, opt_recovery: Any) -> None:
    """Stamp ASE vs geomeTRIC recovery tags onto ``opt_geom.data``."""
    if opt_geom is None or not opt_recovery:
        return
    tag = str(opt_recovery)
    if tag in ("BFGS", "LBFGS", "FIRE") or tag.endswith("-soft"):
        opt_geom.data["ase_opt_recovery"] = tag
    else:
        opt_geom.data["geometric_recovery"] = tag


def slim_node_result(node: Any) -> dict:
    """Build the slim multiprocessing result payload for a completed node."""
    coords = None
    if getattr(node, "opt_geom", None) is not None:
        coords = np.asarray(node.opt_geom.data["positions"], dtype=float)
    return {
        "energy": node.energy,
        "forces": node.forces,
        "coords": coords,
        "complete": node.complete,
        "error": node.error,
        "active": node.active,
        "soft_opt": getattr(node, "soft_opt", False),
        "opt_recovery": getattr(node, "opt_recovery", None),
    }


def apply_slim_node_result(node: Any, result: dict, *, clone_fn=None) -> None:
    """Merge a slim worker result into a parent-side node."""
    if clone_fn is None:
        clone_fn = clone_struct_geometry
    node.energy = result.get("energy")
    if result.get("forces") is not None:
        node.forces = result["forces"]
    node.complete = bool(result.get("complete", node.complete))
    node.error = result.get("error", node.error)
    if "active" in result:
        node.active = bool(result["active"])
    node.soft_opt = bool(result.get("soft_opt", False))
    if "opt_recovery" in result:
        node.opt_recovery = result["opt_recovery"]
    else:
        node.opt_recovery = getattr(node, "opt_recovery", None)
    coords = result.get("coords")
    if coords is not None:
        node.opt_geom = clone_fn(
            node.struct, coords, ene=node.energy, frcs=result.get("forces")
        )
        tag_opt_recovery_on_geom(node.opt_geom, node.opt_recovery)


def write_node_pickle(node: Any, *, verbose: bool = False) -> None:
    """Pickle ``node`` without ``los`` (restored after write).

    I/O errors are logged, not raised: a failed opt must still mark the node
    so the wavefront can skip it instead of killing the pool.
    """
    if verbose:
        print_wavefront(
            f"Saving node {node.node_pkl}  (exists? {Path(node.node_pkl).is_file()})"
        )
    los = node.los
    node.los = None
    try:
        atomic_pickle_dump(node, node.node_pkl)
    except OSError as exc:
        print_wavefront(
            f"could not write node pickle {node.node_pkl}: "
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        node.los = los


def atomic_pickle_dump(obj: Any, path) -> None:
    """Write a pickle via unique ``tmp`` + ``os.replace`` (crash-safe).

    Unique tmp names avoid colliding with geomeTRIC ``*.tmp`` directories.
    ``fsync`` plus retries absorb NFS/VAST ``FileNotFoundError`` on replace.
    """
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    replaced = False
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp, path)
        replaced = True
    finally:
        # Do not unlink after a successful replace: NFS can still see the old
        # name and unlinking it would delete the dest file.
        if not replaced:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


def _replace_with_retry(src: Path, dst: Path, *, attempts: int = 8) -> None:
    """``os.replace`` with backoff, then copy, for flaky network filesystems."""
    import time

    last: Optional[OSError] = None
    for i in range(max(1, int(attempts))):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            last = exc
            if not src.is_file():
                time.sleep(0.05 * (2 ** i))
                if not src.is_file():
                    break
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                time.sleep(0.05 * (2 ** i))
    if last is None:
        raise FileNotFoundError(str(src))
    try:
        import shutil

        shutil.copyfile(src, dst)
        src.unlink()
    except OSError:
        raise last from last


def pickle_checkpoint_keep_calc_cache(obj: Any, path, los) -> None:
    """Pickle a wavefront after stripping live calculators, then restore them.

    ``ListOfStruct.clear_runtime_caches`` drops ``_ffpopt_calc_cache`` so the
    checkpoint does not serialize an XTB/DFT/sander handle. The parent process
    still needs that cache: serial ``nproc=1`` would otherwise rebuild the
    model on every checkpoint. Unbind, dump, rebind.
    """
    calc = getattr(los, "calc", None) if los is not None else None
    cache = getattr(los, "_ffpopt_calc_cache", None) if los is not None else None
    qdpi = getattr(los, "_ffpopt_qdpi2_full_cache", None) if los is not None else None
    mm_preopt = getattr(los, "_ffpopt_mm_preopt_los", None) if los is not None else None
    if los is not None:
        clearer = getattr(los, "clear_runtime_caches", None)
        if callable(clearer):
            clearer()
    try:
        atomic_pickle_dump(obj, path)
    finally:
        if los is not None:
            if calc is not None:
                los.calc = calc
            if cache is not None:
                los._ffpopt_calc_cache = cache
            if qdpi is not None:
                los._ffpopt_qdpi2_full_cache = qdpi
            if mm_preopt is not None:
                los._ffpopt_mm_preopt_los = mm_preopt


def uses_soft_dihed_restraint(los) -> bool:
    """True when wavefront should not hard-apply the scanned dihedral.

    Whole-ligand bulky rotors (detergents) clash if the seed angle is snapped
    with a hard IC before opt. ``--soft-dihed-restraint`` exists so the
    optimizer can rotate under a harmonic. ``seed_struct_rigid_dihed_rotates``
    still Cartesian-twists the ``RotateMask`` branch (and reverts on clash);
    the precheck must not additionally hard-snap.
    """
    args = getattr(los, "args", None)
    return bool(getattr(args, "soft_dihed_restraint", False))


SOFT_DIHED_K_DEFAULT = 500.0
SOFT_DIHED_KMAX_DEFAULT = 8000.0
SOFT_DIHED_TOL_DEFAULT = 0.5
# Skip the extra hard-IC opt when the restrained min is already this close
# to phi0. At k=8000 kcal/mol/rad^2, 0.05 deg residual is ~0.003 kcal/mol.
SOFT_DIHED_HARD_IC_SKIP_DEG = 0.05


def soft_dihed_k_schedule(k0, kmax, *, max_steps: int = 8) -> list[float]:
    """``k0, 2 k0, 4 k0, ...`` up to ``kmax`` (inclusive when reachable)."""
    k = float(k0)
    cap = float(kmax)
    if k <= 0.0:
        k = SOFT_DIHED_K_DEFAULT
    if cap < k:
        cap = k
    seq = [k]
    for _ in range(int(max_steps)):
        nxt = seq[-1] * 2.0
        if nxt > cap + 1e-9:
            break
        seq.append(nxt)
    if seq[-1] < cap - 1e-9:
        seq.append(cap)
    return seq


def _struct_from_opt_geom(seed, opt_geom):
    """Topology of ``seed`` with coordinates from a completed ``GeomOpt``."""
    coords = np.asarray(opt_geom.data["positions"], dtype=float)
    ene = opt_geom.data.get("energy", 0.0)
    frcs = opt_geom.data.get("forces")
    return clone_struct_geometry(
        seed, coords, ene=0.0 if ene is None else ene, frcs=frcs
    )


def signed_wrapped_dihed_delta_deg(observed, target) -> float:
    """Signed wrapped delta ``target - observed`` in ``(-180, 180]`` deg."""
    return (float(target) - float(observed) + 180.0) % 360.0 - 180.0


def _wrapped_dihed_delta_deg(observed, target) -> float:
    """Absolute wrapped dihedral difference in degrees."""
    return abs(signed_wrapped_dihed_delta_deg(observed, target))


# Skip a Cartesian branch twist when already this close (matches hard-IC skip).
RIGID_ROTATE_SKIP_DEG = 0.05
# Treat the Rodrigues step as failed if the dihedral is still this far off.
RIGID_ROTATE_HIT_DEG = 1.0


def dihed_seed_targets(node) -> list[tuple[list[int], float]]:
    """Dihedral ``(idxs, target_deg)`` pairs to Cartesian-seed before GeomOpt."""
    items = []
    if getattr(node, "is_nd", False):
        conlist = getattr(node, "conlist", None)
        if conlist is not None:
            items.extend(list(getattr(conlist, "cons", conlist)))
        reslist = getattr(node, "reslist", None)
        if reslist is not None:
            items.extend(list(getattr(reslist, "rests", reslist)))
    else:
        items.extend(list(getattr(node, "constraints", None) or []))
    out: list[tuple[list[int], float]] = []
    for item in items:
        idxs = getattr(item, "idxs", None)
        val = getattr(item, "value", None)
        if idxs is None or val is None or len(idxs) != 4:
            continue
        out.append(([int(x) for x in idxs], float(val)))
    return out


def _format_dihed_idxs(idxs) -> str:
    return "-".join(str(int(x)) for x in idxs)


def seed_struct_rigid_dihed_rotates(
    struct,
    targets,
    *,
    min_dist: float = 0.8,
    node_id=None,
):
    """Rigid-rotate each dihedral's ``RotateMask`` branch, then clash-check.

    Neighbor seeds still copy the parent Cartesian. Applying wrapped ``dphi``
    here (shortest arc about ``b-c``) puts the moving branch near the target
    before geomeTRIC, so TRIC does not slam e.g. 11 deg toward 250 deg.

    If any nonbonded clash or broken covalent geometry appears, return the
    original ``struct`` unchanged (opt then starts from the parent, as before).

    Parameters
    ----------
    struct : ffpopt.Struct.Struct
        Parent / seed geometry.
    targets : sequence of (idxs, target_deg)
        Four-index dihedrals and target angles in degrees.
    min_dist : float
        Nonbonded clash threshold (Ang), same as wavefront precheck.
    node_id : optional
        Included in ``[wavefront]`` log lines.

    Returns
    -------
    struct
        Cloned geometry with rotated coordinates, or the input if skipped.
    """
    if not targets:
        return struct

    from ffpopt.AmberParm import RotateMask, bonds2graph
    from ffpopt.geom.Constraints import covalent_geometry_error, has_nonbonded_clash
    from ffpopt.geom.Geometry import CptDihed, rotate_coords_about_bond

    crd = np.array(struct.data["positions"], dtype=float, copy=True)
    get_graph = getattr(struct, "GetGraph", None)
    graph = get_graph() if callable(get_graph) else bonds2graph(struct.data["bonds"])

    applied = []
    for idxs, target in targets:
        a, b, c, d = (int(x) for x in idxs)
        obs = float(CptDihed(crd[a], crd[b], crd[c], crd[d]))
        if not np.isfinite(obs):
            continue
        dphi = signed_wrapped_dihed_delta_deg(obs, target)
        if abs(dphi) < RIGID_ROTATE_SKIP_DEG:
            continue
        mask = RotateMask(graph, [a, b, c, d])
        trial = rotate_coords_about_bond(crd, b, c, dphi, mask)
        new = float(CptDihed(trial[a], trial[b], trial[c], trial[d]))
        if not np.isfinite(new) or _wrapped_dihed_delta_deg(new, target) > RIGID_ROTATE_HIT_DEG:
            continue
        crd = trial
        applied.append((_format_dihed_idxs(idxs), dphi, obs, new))

    if not applied:
        return struct

    tag = f"Node {node_id}: " if node_id is not None else ""
    bonds = struct.data["bonds"]
    clashed, i, j, dist = has_nonbonded_clash(crd, bonds, min_dist=min_dist)
    if clashed:
        print_wavefront(
            f"{tag}rigid-rotate clash atoms {i}-{j} at {dist:.3f} Ang; "
            "keeping parent coords"
        )
        return struct

    numbers = None
    getter = getattr(struct, "GetAtomicNumbers", None)
    if callable(getter):
        try:
            numbers = getter()
        except Exception:
            numbers = None
    if numbers is not None:
        broken = covalent_geometry_error(crd, bonds, numbers)
        if broken:
            print_wavefront(
                f"{tag}rigid-rotate broke covalent geometry ({broken}); "
                "keeping parent coords"
            )
            return struct

    bits = [
        f"{name} by {dphi:+.1f} deg ({obs:.1f} -> {new:.1f})"
        for name, dphi, obs, new in applied
    ]
    print_wavefront(f"{tag}rigid-rotated {'; '.join(bits)}")
    return clone_struct_geometry(struct, crd)


def _los_model_name(los) -> str:
    args = getattr(los, "args", None)
    if isinstance(args, dict):
        return str(args.get("model") or "sander")
    return str(getattr(args, "model", None) or "sander")


def _copy_los_args_with_model(args, model: str):
    """Shallow-copy CLI args with ``model`` for MM-then-HL preopt.

    Cheap MM uses geomeTRIC (not ASE-first): bulky whole-ligand seeds often
    start above the ASE explode-fmax guard, and TRIC can unclash them before
    the HL refine.
    """
    if args is None:
        from types import SimpleNamespace

        return SimpleNamespace(model=model, geometric_opt=True, no_opt=False)
    if isinstance(args, dict):
        out = dict(args)
        out["model"] = model
        out["geometric_opt"] = True
        return out
    import copy

    out = copy.copy(args)
    out.model = model
    out.geometric_opt = True
    return out


def make_cheap_preopt_los(los, model: str):
    """Shallow-copy ``los`` onto a cheap engine with its own calc cache."""
    import copy

    cheap = copy.copy(los)
    cheap.args = _copy_los_args_with_model(getattr(los, "args", None), model)
    cheap.calc = None
    cheap._ffpopt_calc_cache = None
    cheap._ffpopt_qdpi2_full_cache = None
    cheap._ffpopt_mm_preopt_los = None
    return cheap


def get_cheap_preopt_los(los, struct):
    """Cached cheap ``ListOfStruct`` for MM-then-HL, or ``None`` to skip."""
    from ffpopt.runtime.FastWavefront import (
        cheap_preopt_model_name,
        mm_then_hl_enabled,
    )

    if los is None or not mm_then_hl_enabled(_los_model_name(los)):
        return None
    cheap_name = cheap_preopt_model_name(struct)
    if cheap_name is None:
        return None
    cached = getattr(los, "_ffpopt_mm_preopt_los", None)
    if cached is not None and cached[0] == cheap_name:
        return cached[1]
    cheap = make_cheap_preopt_los(los, cheap_name)
    los._ffpopt_mm_preopt_los = (cheap_name, cheap)
    return cheap


def _geom_prefix_stage(geom_prefix, stage: str):
    if geom_prefix is None:
        return None
    text = str(geom_prefix).strip()
    return f"{text}_{stage}" if text else None


def geomopt_mm_then_hl(
    los,
    struct,
    constraints=None,
    restraints=None,
    geom_prefix=None,
    *,
    opt_fn=None,
    node_id=None,
    cheap_los=None,
):
    """Constrained min on MM (sander / GFN-FF), then one HL opt from those coords.

    On MM failure, HL still runs from the input ``struct`` (current behavior).
    """
    if opt_fn is None:
        from ffpopt.geom.GeomOpt import GeomOpt as opt_fn

    start = struct
    if cheap_los is None:
        cheap_los = get_cheap_preopt_los(los, struct)
    if cheap_los is not None:
        tag = f"Node {node_id}: " if node_id is not None else ""
        cheap_model = _los_model_name(cheap_los)
        try:
            print_wavefront(f"{tag}MM preopt ({cheap_model}), then HL")
            mm_geom = opt_fn(
                cheap_los,
                struct,
                constraints=constraints,
                restraints=restraints,
                geom_prefix=_geom_prefix_stage(geom_prefix, "mm"),
            )
            start = _struct_from_opt_geom(struct, mm_geom)
        except Exception as exc:
            print_wavefront(
                f"{tag}MM preopt failed ({type(exc).__name__}: {exc}); "
                "HL from parent coords"
            )
            start = struct
    return opt_fn(
        los,
        start,
        constraints=constraints,
        restraints=restraints,
        geom_prefix=geom_prefix,
    )


def run_soft_dihed_opt(
    los,
    struct,
    constraints,
    idxs,
    angle,
    geom_prefix,
    *,
    node_id=None,
    opt_fn=None,
    cheap_los=None,
):
    """Soft harmonic dihedral opt with k-doubling, then a hard IC if needed.

    Each failed band check re-opts from the last coordinates at ``2k`` (up to
    ``soft_dihed_kmax`` / ``FFPOPT_SOFT_DIHED_KMAX``, default 8000). A hard-IC
    opt then runs from those coords unless the restrained min is already
    within ``SOFT_DIHED_HARD_IC_SKIP_DEG`` of ``phi0`` (bias is then far below
    DFT noise).

    When MM-then-HL is on, the k-ramp (and optional MM hard IC) run on sander
    / GFN-FF; one HL opt follows at the final k or after that hard IC.
    """
    from ffpopt.geom.Restraints import HarmonicDihedRestraint

    if opt_fn is None:
        from ffpopt.geom.GeomOpt import GeomOpt as opt_fn

    args = getattr(los, "args", None)
    k0 = float(getattr(args, "soft_dihed_k", None) or SOFT_DIHED_K_DEFAULT)
    tol = float(getattr(args, "soft_dihed_tol", None) or SOFT_DIHED_TOL_DEFAULT)
    kmax_arg = getattr(args, "soft_dihed_kmax", None) if args is not None else None
    if kmax_arg is None:
        from ffpopt.runtime.EnvDefaults import env_float

        kmax = float(env_float("FFPOPT_SOFT_DIHED_KMAX", SOFT_DIHED_KMAX_DEFAULT))
    else:
        kmax = float(kmax_arg)

    if cheap_los is None:
        cheap_los = get_cheap_preopt_los(los, struct)
    ramp_los = cheap_los if cheap_los is not None else los
    two_stage = cheap_los is not None
    ramp_prefix = (
        _geom_prefix_stage(geom_prefix, "mm") if two_stage else geom_prefix
    )

    tag = f"Node {node_id}" if node_id is not None else "soft-dihed"
    if two_stage:
        print_wavefront(
            f"{tag}: MM k-ramp ({_los_model_name(cheap_los)}), then one HL"
        )
    ks = soft_dihed_k_schedule(k0, kmax)
    start = struct
    last_geom = None
    last_z = None
    last_rest = None
    skip_hard = False

    for i, k in enumerate(ks):
        rest = HarmonicDihedRestraint(
            k, list(idxs), float(angle), tol_deg=tol
        )
        try:
            k_prefix = (
                _geom_prefix_stage(ramp_prefix, f"k{i:02d}")
                if ramp_prefix
                else None
            )
            last_geom = opt_fn(
                ramp_los,
                start,
                constraints=None,
                restraints=[rest],
                geom_prefix=k_prefix,
            )
        except Exception as exc:
            if last_geom is None:
                raise
            print(
                f"[affdo] {tag}: soft opt at k={k:g} failed "
                f"({type(exc).__name__}: {exc}); warm-starting hard IC",
                flush=True,
            )
            break
        last_rest = rest
        crds = last_geom.data["positions"]
        try:
            ok = rest.within_tolerance(crds)
            last_z = float(rest.GetCrdValue(crds))
        except Exception:
            ok = True
            last_z = None
        if ok:
            if i > 0:
                print(
                    f"[affdo] {tag}: soft dihedral held at k={k:g} "
                    f"kcal/mol/rad^2",
                    flush=True,
                )
            dphi = (
                _wrapped_dihed_delta_deg(last_z, angle)
                if last_z is not None
                else None
            )
            if dphi is not None and dphi <= SOFT_DIHED_HARD_IC_SKIP_DEG:
                skip_hard = True
                print(
                    f"[affdo] {tag}: in-band at k={k:g}; "
                    f"|dphi|={dphi:.3f} deg <= {SOFT_DIHED_HARD_IC_SKIP_DEG:g} "
                    f"deg; skipping hard IC",
                    flush=True,
                )
            else:
                print(
                    f"[affdo] {tag}: in-band at k={k:g}; finishing with hard IC "
                    f"(warm start)",
                    flush=True,
                )
            break
        ztxt = f"{last_z:.2f}" if last_z is not None else "?"
        more = i + 1 < len(ks)
        if more:
            print(
                f"[affdo] {tag}: soft dihedral {ztxt} deg outside "
                f"+/-{tol:g} deg of target {angle} at k={k:g}; "
                f"ramping k -> {ks[i + 1]:g}",
                flush=True,
            )
            start = _struct_from_opt_geom(struct, last_geom)
        else:
            print(
                f"[affdo] {tag}: soft dihedral {ztxt} deg still outside "
                f"+/-{tol:g} deg of target {angle} after kmax={kmax:g}; "
                f"falling back to hard IC (warm start)",
                flush=True,
            )

    if skip_hard and last_geom is not None and not two_stage:
        return last_geom

    hard_start = (
        _struct_from_opt_geom(struct, last_geom)
        if last_geom is not None
        else struct
    )

    if two_stage:
        if skip_hard and last_rest is not None:
            print(
                f"[affdo] {tag}: one HL opt at final k={last_rest.k_kcal:g}",
                flush=True,
            )
            return opt_fn(
                los,
                hard_start,
                constraints=None,
                restraints=[last_rest],
                geom_prefix=geom_prefix,
            )
        try:
            mm_hard = opt_fn(
                cheap_los,
                hard_start,
                constraints=constraints,
                geom_prefix=_geom_prefix_stage(geom_prefix, "mm"),
            )
            hard_start = _struct_from_opt_geom(struct, mm_hard)
        except Exception as exc:
            print_wavefront(
                f"{tag}: MM hard IC failed ({type(exc).__name__}: {exc}); "
                "HL hard IC from last soft coords"
            )
        print(f"[affdo] {tag}: one HL hard IC (warm start)", flush=True)
        return opt_fn(
            los,
            hard_start,
            constraints=constraints,
            geom_prefix=geom_prefix,
        )

    return opt_fn(
        los,
        hard_start,
        constraints=constraints,
        geom_prefix=geom_prefix,
    )


def empty_scan_error_message(wavefront, out_path) -> str:
    """Human-readable reason when a wavefront stored no finite-energy angles."""
    levels = getattr(wavefront, "levels", None) or []
    n_nodes = sum(len(getattr(lv, "nodes", None) or []) for lv in levels)
    failed = []
    for lv in levels:
        for node in getattr(lv, "nodes", None) or []:
            err = getattr(node, "error", None)
            ene = getattr(node, "energy", None)
            if err or ene is None or not np.isfinite(ene):
                failed.append(
                    f"angle={getattr(node, 'angle', '?')} "
                    f"id={getattr(node, 'node_id', '?')} error={err}"
                )
    lines = [
        f"wavefront produced 0 accepted scan angles (nodes={n_nodes}, "
        f"out={out_path}).",
        "No finite energy was stored, so there is no .dat profile to fit.",
    ]
    if n_nodes == 0:
        lines.append(
            "No seed nodes were created: the hard-twist clash check rejected "
            "every starting angle. --soft-dihed-restraint now skips that "
            "snap-before-opt check on bulky whole-ligand torsions."
        )
    elif failed:
        lines.append("Failed / nonfinite nodes:")
        lines.extend(f"  {x}" for x in failed[:12])
        if len(failed) > 12:
            lines.append(f"  ... and {len(failed) - 12} more")
    return "\n".join(lines)


def mark_node_failed(
    node: Any,
    reason: str,
    error: Optional[Exception] = None,
    *,
    where: Any = None,
) -> None:
    """Mark a node failed, write checkpoint, and print a short message."""
    msg = reason if error is None else f"{reason}: {error}"
    node.error = msg
    node.active = False
    node.complete = True
    node.energy = np.inf
    node.forces = np.zeros((len(node.struct.data["elements"]), 3))
    loc = where if where is not None else getattr(node, "angle", getattr(node, "rcs", "?"))
    print_wavefront(f"Node {node.node_id} failed at {loc}: {msg}")
    write_node_pickle(node, verbose=False)


def maybe_write_success_checkpoint(node: Any) -> None:
    """Write a success node pickle when fast-wavefront policy allows."""
    from ffpopt.runtime.FastWavefront import write_success_node_pickle

    if write_success_node_pickle():
        write_node_pickle(node)


def kcal_threshold_to_ev(threshold_kcal: float) -> float:
    """Convert a kcal/mol convergence threshold to eV."""
    from ffpopt.constants import AU_PER_ELECTRON_VOLT, AU_PER_KCAL_PER_MOL

    kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    return float(threshold_kcal) / kcal_per_ev


def evaluate_wavefront_minimum(
    *,
    energy: float,
    soft: bool,
    has_incumbent: bool,
    incumbent_energy: Optional[float],
    incumbent_soft: bool,
    threshold_ev: float,
) -> dict[str, Any]:
    """Decide profile-min update and spawn (``active``) for one completed node.

    Policy (1-D and N-D):

    * Soft, first at bin - store and **spawn once** (coverage seed).
    * Soft, improves soft min - update; no spawn.
    * Soft otherwise - demote; no spawn.
    * Hard vs soft incumbent - replace soft only if ``E_hard <= E_soft``; spawn
      when accepted.
    * Hard, ``E < min - threshold`` - update and spawn.
    * Hard, ``E < min`` within threshold - update quietly; no spawn.
    * Hard, ``E >= min`` - no update; no spawn.
    """
    if energy is None or not np.isfinite(energy):
        return {
            "update_min": False,
            "active": False,
            "reason": "nonfinite",
        }

    if soft:
        if not has_incumbent:
            return {
                "update_min": True,
                "active": True,
                "reason": "soft_first_seed",
            }
        if incumbent_soft and energy < float(incumbent_energy):
            return {
                "update_min": True,
                "active": False,
                "reason": "soft_improve",
            }
        return {
            "update_min": False,
            "active": False,
            "reason": "soft_demoted",
        }

    # Hard-converged node.
    if not has_incumbent:
        return {
            "update_min": True,
            "active": True,
            "reason": "hard_first",
        }

    if incumbent_soft:
        if energy <= float(incumbent_energy):
            return {
                "update_min": True,
                "active": True,
                "reason": "hard_replace_soft",
            }
        return {
            "update_min": False,
            "active": False,
            "reason": "hard_worse_than_soft",
        }

    inc = float(incumbent_energy)
    thr = max(0.0, float(threshold_ev))
    if energy < inc - thr:
        return {
            "update_min": True,
            "active": True,
            "reason": "hard_significant_improve",
        }
    if energy < inc:
        return {
            "update_min": True,
            "active": False,
            "reason": "hard_quiet_improve",
        }
    return {
        "update_min": False,
        "active": False,
        "reason": "hard_not_lower",
    }


def stamp_node_soft_flag(node: Any, *, incumbent_geom=None) -> tuple[bool, bool]:
    """Refresh ``node.soft_opt`` from recovery tags; return (node, incumbent) flags."""
    from ffpopt.geom.GeomOpt import is_soft_opt_recovery

    soft = bool(getattr(node, "soft_opt", False))
    if not soft and getattr(node, "opt_geom", None) is not None:
        soft = is_soft_opt_recovery(node.opt_geom)
        node.soft_opt = soft
    incumbent_soft = False
    if incumbent_geom is not None:
        incumbent_soft = is_soft_opt_recovery(incumbent_geom)
    return soft, incumbent_soft


def finalize_successful_node_opt(node: Any) -> None:
    """Round energy, optional qdpi2 refine, success pickle, mark complete."""
    from ffpopt.geom.GeomOpt import bare_potential_energy

    node.energy = np.round(bare_potential_energy(node.opt_geom), 6)
    try:
        from ffpopt.geom.Geometric import refine_qdpi2_energy

        refined = refine_qdpi2_energy(node.los, node.opt_geom)
        if refined is not None:
            node.energy = np.round(float(refined), 6)
            node.opt_geom.data["energy"] = float(refined)
            node.opt_geom.data["qdpi2_refined"] = True
    except Exception:
        pass
    node.forces = node.opt_geom.data.get("forces", node.forces)
    maybe_write_success_checkpoint(node)
    node.complete = True


def slim_completed_nodes_for_checkpoint(wavefront: Any) -> None:
    """Drop bulky force arrays from completed nodes before pickling."""
    for level in getattr(wavefront, "levels", []) or []:
        for node in getattr(level, "nodes", []) or []:
            if not getattr(node, "complete", False):
                continue
            if getattr(node, "opt_geom", None) is not None:
                if "forces" in node.opt_geom.data:
                    node.opt_geom.data["forces"] = None
            n_atoms = len(node.struct.data["elements"])
            node.forces = np.zeros((n_atoms, 3))


def format_minimum_decision_message(
    reason: str,
    *,
    loc,
    energy,
    old,
    recovery=None,
    noun: str = "coordinate",
) -> str:
    """ASCII log line for a wavefront min-policy decision."""
    rec = recovery
    label = noun.capitalize()
    if reason == "soft_first_seed":
        return (
            f"New {noun} (soft-opt seed): {loc}, Energy: {energy} "
            f"(recovery={rec}; spawn once)"
        )
    if reason == "soft_improve":
        return (
            f"Updating soft-opt {noun}: {loc}, "
            f"Old Energy: {old}, New Energy: {energy} (no spawn)"
        )
    if reason == "hard_first":
        return f"New {noun} detected: {loc}, Energy: {energy}"
    if reason == "hard_replace_soft":
        return (
            f"Replacing soft-opt {noun} {loc} with hard-converged "
            f"Energy: {energy} (was {old})"
        )
    if reason == "hard_significant_improve":
        return f"Updating {noun}: {loc}, Old Energy: {old}, New Energy: {energy}"
    if reason == "hard_quiet_improve":
        return (
            f"Quiet update {noun} {loc}: {old} -> {energy} "
            f"(within threshold; no spawn)"
        )
    if reason == "soft_demoted":
        return (
            f"{label} {loc} soft-opt demoted (recovery={rec}); "
            f"not replacing hard-converged / lower soft minimum."
        )
    if reason == "hard_worse_than_soft":
        return (
            f"{label} {loc} hard-opt higher than soft min "
            f"({energy} > {old}); keeping soft profile."
        )
    if reason == "hard_not_lower":
        return (
            f"{label} {loc} is not active, energy {energy} "
            f"is not lower than minimum {old}."
        )
    if reason == "nonfinite":
        return f"{label} {loc} is inactive due to failed optimization."
    return f"{label} {loc} inactive ({reason}): energy {energy}."


def apply_wavefront_minimum_to_node(
    node: Any,
    *,
    loc,
    threshold_kcal: float,
    has_incumbent: bool,
    incumbent_energy,
    incumbent_soft: bool,
    on_update,
    noun: str = "coordinate",
) -> dict:
    """Shared evaluate-node tail: policy, logs, ``node.active``.

    ``on_update(node, reason, old_energy)`` stores the new min when the
    policy says ``update_min``.
    """
    if not getattr(node, "active", True):
        return {"update_min": False, "active": False, "reason": "already_inactive"}
    if node.energy is None or not np.isfinite(node.energy):
        print_wavefront(
            format_minimum_decision_message(
                "nonfinite", loc=loc, energy=node.energy, old=incumbent_energy, noun=noun
            )
        )
        node.active = False
        return {"update_min": False, "active": False, "reason": "nonfinite"}

    stamp_node_soft_flag(node)
    decision = evaluate_wavefront_minimum(
        energy=node.energy,
        soft=bool(getattr(node, "soft_opt", False)),
        has_incumbent=has_incumbent,
        incumbent_energy=incumbent_energy,
        incumbent_soft=incumbent_soft,
        threshold_ev=kcal_threshold_to_ev(threshold_kcal),
    )
    reason = decision["reason"]
    old = incumbent_energy
    recovery = getattr(node, "opt_recovery", None)
    if decision["update_min"]:
        on_update(node, reason, old)
    print_wavefront(
        format_minimum_decision_message(
            reason,
            loc=loc,
            energy=node.energy,
            old=old,
            recovery=recovery,
            noun=noun,
        )
    )
    node.active = bool(decision["active"])
    return decision


def print_wavefront(msg: str, *, flush: bool = True) -> None:
    """Print one wavefront line with a leading ``[wavefront]`` scope.

    The hierarchical console formatter peels this into ``[ffpopt] [wavefront]``
    (or ``[ligandparam] [wavefront]``), matching ``[affdo]`` / ``[twist]``.
    """
    from ffpopt.runtime.Console import ascii_for_stdio

    text = msg if msg.lstrip().startswith("[") else f"[wavefront] {msg}"
    print(ascii_for_stdio(text), flush=flush)


def format_wavefront_progress(
    wavefront: Any,
    pending: int,
    in_flight: int,
    *,
    extra: str,
) -> str:
    """One-line live progress summary shared by 1-D and N-D engines."""
    completed = sum(
        1
        for level in getattr(wavefront, "levels", []) or []
        for node in getattr(level, "nodes", []) or []
        if getattr(node, "complete", False)
    )
    highest = max(
        (level.level_id for level in getattr(wavefront, "levels", []) or []),
        default=0,
    )
    return (
        f"[wavefront] completed={completed} pending={pending} "
        f"in-flight={in_flight} highest-level={highest} {extra}"
    )


def require_main_guard_for_spawn() -> None:
    """Abort if a spawn worker re-imported the caller's script."""
    import sys

    caller = sys._getframe(2).f_globals
    if caller.get("__name__") == "__mp_main__":
        raise RuntimeError(
            "run_dihed_wavefront was re-invoked by a multiprocessing spawn worker "
            "re-importing the calling script. Wrap the call in "
            "`if __name__ == '__main__':` so the worker re-import doesn't "
            "re-execute it. See "
            "https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming"
        )


def merge_standard_wavefront_kwargs(standard_kwargs: dict, extra_adders=()) -> dict:
    """Merge CLI-default standard options with caller kwargs."""
    import argparse

    from ffpopt.Options import AddStandardOptions

    parser = argparse.ArgumentParser(add_help=False)
    AddStandardOptions(parser)
    for adder in extra_adders:
        adder(parser)
    std_defaults = vars(parser.parse_args([]))
    unknown = set(standard_kwargs) - set(std_defaults)
    if unknown:
        raise TypeError(
            f"run_dihed_wavefront got unexpected keyword argument(s): {sorted(unknown)}"
        )
    return {**std_defaults, **standard_kwargs}


def register_wavefront_pickle_aliases() -> None:
    """Map historical ``ffpopt.WaveFront*`` pickle names onto ``scan.*`` modules."""
    import sys

    from ffpopt.scan import WaveFront as wf
    from ffpopt.scan import WaveFrontND as wfnd
    from ffpopt.scan import WavefrontEngine as engine

    sys.modules.setdefault("ffpopt.WaveFront", wf)
    sys.modules.setdefault("ffpopt.WaveFrontND", wfnd)
    sys.modules.setdefault("ffpopt.scan.WavefrontEngine", engine)


def load_wavefront_pickle(filename: str, *, restore_soft_opt: bool = True):
    """Unpickle a Wavefront object and optionally restore soft-opt attrs.

    Soft-opt restoration matches the 1-D loader behavior and is safe for N-D
    nodes that lack ``_ensure_soft_opt_attrs``.

    Checkpoints written before the ``scan/`` move pickle classes as
    ``ffpopt.WaveFront.*``. :func:`register_wavefront_pickle_aliases` maps
    those names onto :mod:`ffpopt.scan.WaveFront` without extra shim files.
    """
    register_wavefront_pickle_aliases()

    with open(filename, "rb") as f:
        wavefront = pickle.load(f)
    if restore_soft_opt:
        for level in getattr(wavefront, "levels", []) or []:
            for node in getattr(level, "nodes", []) or []:
                ensure = getattr(node, "_ensure_soft_opt_attrs", None)
                if callable(ensure):
                    ensure()
                else:
                    ensure_soft_opt_attrs(node)
        for node in getattr(wavefront, "_resume_queue", None) or []:
            ensure = getattr(node, "_ensure_soft_opt_attrs", None)
            if callable(ensure):
                ensure()
            else:
                ensure_soft_opt_attrs(node)
    print_wavefront(f"wavefront object loaded from {filename}")
    return wavefront


def pickle_load_compat(file_or_path):
    """``pickle.load`` with wavefront module-path aliases registered."""
    register_wavefront_pickle_aliases()

    if hasattr(file_or_path, "read"):
        return pickle.load(file_or_path)
    with open(file_or_path, "rb") as f:
        return pickle.load(f)


def replace_node_with_pickle(node: Any, *, found_msg: Optional[str] = None) -> None:
    """Replace node fields from a sidecar pickle if present (restores ``los``)."""
    filename = Path(f"{node.node_pkl}")
    if not filename.is_file():
        return
    print_wavefront(
        found_msg
        if found_msg is not None
        else f"Found existing pickle file for node: {node.node_id}"
    )
    los = node.los
    loaded_node = pickle_load_compat(filename)
    node.__dict__.update(loaded_node.__dict__)
    if node.los is None:
        node.los = los
    ensure = getattr(node, "_ensure_soft_opt_attrs", None)
    if callable(ensure):
        ensure()
    else:
        ensure_soft_opt_attrs(node)
    print_wavefront("Node data replaced with pickle data.")


def precheck_geometry_clash(
    *,
    get_atoms,
    bonds,
    min_dist: float = 0.8,
) -> Optional[str]:
    """Shared clash / broken-bond precheck after constraints are applied.

    ``get_atoms`` should return an ASE atoms object with constraints applied.
    Skips the optimizer when nonbonded atoms overlap or a covalent bond is
    already crushed / dissociated (hydrogens or carbons flying off).
    """
    try:
        from ffpopt.geom.Constraints import (
            covalent_geometry_error,
            has_nonbonded_clash,
        )

        myatoms = get_atoms()
        pos = myatoms.get_positions()
        clashed, i, j, dist = has_nonbonded_clash(pos, bonds, min_dist=min_dist)
        if clashed:
            print_wavefront(
                f"Precheck clash: atom {i} and atom {j} at {dist:.3f} Ang "
                f"(< {min_dist} Ang)"
            )
            return "clash_precheck"
        broken = covalent_geometry_error(
            pos, bonds, myatoms.get_atomic_numbers()
        )
        if broken:
            print_wavefront(f"Precheck broken geometry: {broken}")
            return "broken_geometry"
    except Exception as e:
        print_wavefront(f"Precheck failed due to error: {e}")
        return f"precheck_error: {e}"
    return None


def run_mp_spawn_drain_loop(
    *,
    pending,
    nproc: int,
    pool,
    run_node_job,
    on_complete,
    set_resume_queue,
    save_checkpoint,
    cleanup_completed,
    print_progress,
    checkpoint_every: int,
    terminate_pool: bool = True,
    on_dispatch=None,
    on_skip=None,
) -> None:
    """Shared multiprocessing drain loop for 1-D and N-D wavefront scans.

    ``pool`` may be ``None`` for serial execution. Callers create the pool and
    own finish bookkeeping after this returns. When ``terminate_pool`` is
    False, the pool is left open for reuse across sequential scans.

    ``on_dispatch`` is called when a node is about to run (pending -> in-flight).
    ``on_skip`` is called when a queued node is dropped as inactive.
    """
    import time

    try:
        in_flight = {}
        since_checkpoint = 0
        while pending or in_flight:
            while pending and len(in_flight) < nproc:
                node = pending.popleft()
                node.replace_with_pickle()
                if node.complete:
                    pending.extend(on_complete(node))
                    continue
                if not node.active:
                    if on_skip is not None:
                        extra = on_skip(node)
                        if extra is not None:
                            pending.append(extra)
                    continue
                if on_dispatch is not None:
                    on_dispatch(node)
                if pool is None:
                    node.calculate()
                    pending.extend(on_complete(node))
                    since_checkpoint += 1
                    break
                in_flight[pool.apply_async(run_node_job, (node.to_job(),))] = node

            if in_flight:
                progressed = False
                for async_result in list(in_flight):
                    if async_result.ready():
                        node = in_flight.pop(async_result)
                        result = async_result.get()
                        node.apply_result(result)
                        pending.extend(on_complete(node))
                        since_checkpoint += 1
                        progressed = True
                if not progressed:
                    time.sleep(0.05)

            if since_checkpoint >= checkpoint_every:
                set_resume_queue(list(pending) + list(in_flight.values()))
                save_checkpoint()
                cleanup_completed()
                print_progress(len(pending), len(in_flight))
                since_checkpoint = 0
    finally:
        if pool is not None and terminate_pool:
            pool.terminate()
            pool.join()


def run_mpi_spawn_drain_loop(
    *,
    pending,
    comm,
    size: int,
    tag_task: int,
    tag_result: int,
    tag_stop: int,
    on_complete,
    set_resume_queue,
    save_checkpoint,
    cleanup_completed,
    print_progress,
    checkpoint_every: int,
    on_dispatch=None,
    on_skip=None,
) -> None:
    """Shared MPI master drain loop (rank 0) for N-D wavefront scans.

    Mirrors :func:`run_mp_spawn_drain_loop`: dispatch slim jobs to idle ranks,
    harvest results, checkpoint periodically, and stop workers in ``finally``.
    Callers own worker broadcast / init and post-loop finish bookkeeping.
    """
    idle_workers = set(range(1, size))
    in_flight = {}
    since_checkpoint = 0

    try:
        while pending or in_flight:
            while pending and idle_workers:
                worker = idle_workers.pop()
                node = pending.popleft()
                node.replace_with_pickle()

                if node.complete:
                    pending.extend(on_complete(node))
                    idle_workers.add(worker)
                    continue
                if not node.active:
                    if on_skip is not None:
                        extra = on_skip(node)
                        if extra is not None:
                            pending.append(extra)
                    idle_workers.add(worker)
                    continue

                if on_dispatch is not None:
                    on_dispatch(node)
                comm.send(node.to_job(), dest=worker, tag=tag_task)
                in_flight[worker] = node

            if in_flight:
                from mpi4py import MPI

                status = MPI.Status()
                result = comm.recv(source=MPI.ANY_SOURCE, tag=tag_result, status=status)
                worker = status.Get_source()
                node = in_flight.pop(worker)
                idle_workers.add(worker)
                node.apply_result(result)
                pending.extend(on_complete(node))
                since_checkpoint += 1

            if since_checkpoint >= checkpoint_every:
                set_resume_queue(list(pending) + list(in_flight.values()))
                save_checkpoint()
                cleanup_completed()
                print_progress(len(pending), len(in_flight))
                since_checkpoint = 0
    finally:
        for worker in range(1, size):
            comm.send(None, dest=worker, tag=tag_stop)


def cleanup_wavefront_geometric_scratch(
    wavefront, *, keep_incomplete_optim: bool = False
) -> None:
    """Remove leftover geomeTRIC ``.nsf`` / ``.tmp`` scratch for a wavefront.

    On checkpoint resume, pass ``keep_incomplete_optim=True`` so unfinished
    nodes keep ``_optim.xyz`` for warm-start. Completed nodes drop everything.
    """
    from ffpopt.geom.Geometric import (
        cleanup_geometric_scratch,
        geometric_prefix_from_node_pkl,
        sweep_geometric_scratch_dir,
    )

    keep = []
    for level in getattr(wavefront, "levels", []) or []:
        for node in getattr(level, "nodes", []) or []:
            pkl = getattr(node, "node_pkl", None)
            if not pkl:
                continue
            prefix = geometric_prefix_from_node_pkl(pkl)
            keep_opt = bool(
                keep_incomplete_optim and not getattr(node, "complete", False)
            )
            cleanup_geometric_scratch(prefix, keep_optim=keep_opt)
            if keep_opt:
                keep.append(prefix)

    workdir = getattr(wavefront, "workdir", None)
    if not workdir:
        ckpt = getattr(wavefront, "checkpoint", None)
        if ckpt:
            workdir = str(Path(ckpt).resolve().parent)
    if not workdir:
        return
    n = sweep_geometric_scratch_dir(
        workdir, keep_optim_prefixes=keep, recursive=False
    )
    if n:
        print(
            f"[ffpopt] removed {n} leftover geomeTRIC scratch path(s) in {workdir}"
        )


def save_wavefront_figure(path, *, close: bool = True) -> None:
    """Save the current matplotlib figure without noisy tight-layout warnings.

    Dense angle/level grids often cannot satisfy ``tight_layout``; the plot is
    still written with ``bbox_inches='tight'``.
    """
    import warnings

    from matplotlib import pyplot as plt

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*[Tt]ight layout not applied.*"
        )
        try:
            plt.tight_layout()
        except Exception:
            pass
    plt.savefig(path, bbox_inches="tight")
    if close:
        plt.close()
