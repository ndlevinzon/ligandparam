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
    """Pickle ``node`` without ``los`` (restored after write)."""
    if verbose:
        print(
            f"Saving node {node.node_pkl}  (exists? {Path(node.node_pkl).is_file()})"
        )
    los = node.los
    node.los = None
    try:
        atomic_pickle_dump(node, node.node_pkl)
    finally:
        node.los = los


def atomic_pickle_dump(obj: Any, path) -> None:
    """Write a pickle via ``tmp`` + ``os.replace`` (crash-safe)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


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
    print(f"Node {node.node_id} failed at {loc}: {msg}")
    write_node_pickle(node, verbose=False)


def maybe_write_success_checkpoint(node: Any) -> None:
    """Write a success node pickle when fast-wavefront policy allows."""
    from ffpopt.runtime.fast_wavefront import write_success_node_pickle

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

    * Soft, first at bin — store and **spawn once** (coverage seed).
    * Soft, improves soft min — update; no spawn.
    * Soft otherwise — demote; no spawn.
    * Hard vs soft incumbent — replace soft only if ``E_hard <= E_soft``; spawn
      when accepted.
    * Hard, ``E < min - threshold`` — update and spawn.
    * Hard, ``E < min`` within threshold — update quietly; no spawn.
    * Hard, ``E >= min`` — no update; no spawn.
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


def load_wavefront_pickle(filename: str, *, restore_soft_opt: bool = True):
    """Unpickle a Wavefront object and optionally restore soft-opt attrs.

    Soft-opt restoration matches the 1-D loader behavior and is safe for N-D
    nodes that lack ``_ensure_soft_opt_attrs``.
    """
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
    print("Wavefront object loaded from", filename)
    return wavefront
