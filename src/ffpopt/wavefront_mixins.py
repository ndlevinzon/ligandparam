"""Shared wavefront node helpers (1-D and N-D)."""
from __future__ import annotations

import copy

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
