#!/usr/bin/env python3
"""ffpopt — force-field parameter optimization toolkit.

Integrated under ``src/ffpopt`` next to ``ligandparam``.

Primary APIs for torsion correction after ligand parameterization:

* :mod:`ffpopt.Workflows` — ``run_dihed_twist_workflow``,
  ``run_fragmented_dihed_twist_workflow``
* :mod:`ffpopt.Dihedrals` — GenDihedFit input types and solvers
* :mod:`ffpopt.WaveFront` — parallel dihedral scan engine

Submodules are imported lazily so lightweight callers (and packaging checks)
do not require every optional calculator stack at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ase",
    "constants",
    "confsearch",
    "cpefit",
    "scosmo",
    "Workflows",
    "Dihedrals",
    "WaveFront",
]

_LAZY = {
    "ase": ".ase",
    "constants": ".constants",
    "confsearch": ".confsearch",
    "cpefit": ".cpefit",
    "scosmo": ".scosmo",
    "Workflows": ".Workflows",
    "Dihedrals": ".Dihedrals",
    "WaveFront": ".WaveFront",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = import_module(_LAZY[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
