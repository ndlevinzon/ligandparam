"""High-level torsion-fitting workflows for Amber force fields.

Canonical package for dihedral (twist) corrections.

When to use which entry point
-----------------------------
``run_fragmented_dihed_twist_workflow``
    After ligand parameterization that produces a parent ``mol2`` / ``lib`` /
    ``frcmod`` triplet. Fragments with ``scission``, fits torsions per
    fragment, and merges DIHE terms into a new parent ``frcmod``.

``run_dihed_twist_workflow``
    Single-molecule path when you already have ``parm7`` / ``rst7`` (via
    PrepareInput -> ``start.json``) and known rotatable central bonds
    (0-based atom indices).

``run_whole_ligand_dihed_twist_workflow``
    Full-ligand twist without scission (AFFDO-style extras optional).
"""

from __future__ import annotations

from ffpopt.workflows.helpers import (
    bonds0_from_scission_fit_torsions,
    normalize_bond_pairs0,
)
from ffpopt.workflows.twist import (
    run_batched_dihed_twist_workflow,
    run_dihed_twist_workflow,
)
from ffpopt.workflows.fragmented import (
    clear_fragment_twist_done,
    fragment_twist_done_path,
    is_fragment_twist_done,
    mark_fragment_twist_done,
    run_fragmented_dihed_twist_workflow,
)
from ffpopt.workflows.whole_ligand import run_whole_ligand_dihed_twist_workflow

__all__ = [
    "bonds0_from_scission_fit_torsions",
    "clear_fragment_twist_done",
    "fragment_twist_done_path",
    "is_fragment_twist_done",
    "mark_fragment_twist_done",
    "normalize_bond_pairs0",
    "run_batched_dihed_twist_workflow",
    "run_dihed_twist_workflow",
    "run_fragmented_dihed_twist_workflow",
    "run_whole_ligand_dihed_twist_workflow",
]
