#!/usr/bin/env python3
"""ffpopt — force-field parameter optimization toolkit.

Integrated under ``src/ffpopt`` next to ``ligandparam``.

Primary APIs for torsion correction after ligand parameterization:

* :mod:`ffpopt.workflows` — ``run_dihed_twist_workflow``,
  ``run_fragmented_dihed_twist_workflow``,
  ``run_whole_ligand_dihed_twist_workflow``
* :mod:`ffpopt.dihed.Dihedrals` — GenDihedFit input types and solvers
* :mod:`ffpopt.scan.WaveFront` — parallel dihedral scan engine
* :mod:`ffpopt.geom.GeomOpt` — ASE / geomeTRIC optimization
* :mod:`ffpopt.affdo` — optional AFFDO extras (log, centroids, charges)

Root modules ``Workflows``, ``Dihedrals``, ``GeomOpt``, ``WaveFront`` remain
as compatibility re-exports.

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
    "runtime",
    "scan",
    "affdo",
    "dihed",
    "geom",
    "workflows",
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
    "runtime": ".runtime",
    "scan": ".scan",
    "affdo": ".affdo",
    "dihed": ".dihed",
    "geom": ".geom",
    "workflows": ".workflows",
    "Workflows": ".Workflows",
    "Dihedrals": ".Dihedrals",
    # WaveFront / WaveFrontND are real pickle-compat modules at package root.
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = import_module(_LAZY[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
