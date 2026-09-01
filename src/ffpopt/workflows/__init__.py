"""High-level torsion-fitting workflows for Amber force fields.

Canonical package for dihedral (twist) corrections on a single molecule.

``run_dihed_twist_workflow``
    When you already have ``parm7`` / ``rst7`` (via PrepareInput ->
    ``start.json``) and known rotatable central bonds (0-based atom
    indices).

Fragmented and whole-ligand orchestration lives outside this package.
"""

from __future__ import annotations

from ffpopt.workflows.TwistHelpers import normalize_bond_pairs0
from ffpopt.workflows.DihedTwist import (
    run_batched_dihed_twist_workflow,
    run_dihed_twist_workflow,
)

_MOVED_TO_ALPS = frozenset(
    {
        "bonds0_from_scission_fit_torsions",
        "clear_fragment_twist_done",
        "fragment_twist_done_path",
        "is_fragment_twist_done",
        "mark_fragment_twist_done",
        "run_fragmented_dihed_twist_workflow",
        "run_whole_ligand_dihed_twist_workflow",
    }
)

__all__ = [
    "normalize_bond_pairs0",
    "run_batched_dihed_twist_workflow",
    "run_dihed_twist_workflow",
]


def __getattr__(name: str):
    if name in _MOVED_TO_ALPS:
        raise ImportError(
            f"{name} lives in alps.workflows (scission + ffpopt "
            "orchestration). Import it from alps.workflows."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
