"""Tests for scission domain hoist + vectorized screen_candidate."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from scission.fragments import build_candidate_fragments, get_fragmentation_topology
from scission.models import (
    Atom,
    Bond,
    CapSite,
    CandidateFragment,
    ClashThresholds,
    Ligand,
    TorsionDefinition,
)
from scission.screen import _build_cap_coordinates, screen_candidate
from scission.torsions import find_rotatable_bonds


def _hexane_like_ligand() -> Ligand:
    """Acyclic C6 chain with terminal H atoms for cuts."""
    atoms = [
        Atom(1, "C1", "C", "c3", -0.1, (0.0, 0.0, 0.0)),
        Atom(2, "C2", "C", "c3", -0.1, (1.5, 0.0, 0.0)),
        Atom(3, "C3", "C", "c3", -0.1, (2.0, 1.4, 0.0)),
        Atom(4, "C4", "C", "c3", -0.1, (3.5, 1.4, 0.0)),
        Atom(5, "C5", "C", "c3", -0.1, (4.0, 0.0, 0.0)),
        Atom(6, "C6", "C", "c3", -0.1, (5.5, 0.0, 0.0)),
        Atom(7, "H1", "H", "hc", 0.1, (-0.5, 0.5, 0.0)),
        Atom(8, "H6", "H", "hc", 0.1, (6.0, 0.5, 0.0)),
    ]
    bonds = [
        Bond(1, 1, 2, "1"),
        Bond(2, 2, 3, "1"),
        Bond(3, 3, 4, "1"),
        Bond(4, 4, 5, "1"),
        Bond(5, 5, 6, "1"),
        Bond(6, 1, 7, "1"),
        Bond(7, 6, 8, "1"),
    ]
    return Ligand(
        name="HEX",
        atoms=atoms,
        bonds=bonds,
        lib_atom_names=[a.name for a in atoms],
        lib_atom_types={a.name: a.atom_type for a in atoms},
        frcmod_text="",
        mol2_path=Path("hex.mol2"),
        lib_path=Path("hex.lib"),
        frcmod_path=Path("hex.frcmod"),
    )


def test_fragmentation_topology_hoisted_across_torsions():
    ligand = _hexane_like_ligand()
    t1 = get_fragmentation_topology(ligand, include_rigid_single_bonds=True)
    t2 = get_fragmentation_topology(ligand, include_rigid_single_bonds=True)
    assert t1 is t2

    bonds1 = find_rotatable_bonds(ligand, include_rigid_single_bonds=True)
    bonds2 = find_rotatable_bonds(ligand, include_rigid_single_bonds=True)
    assert bonds1 == bonds2

    # Different config → new cache entry
    t3 = get_fragmentation_topology(ligand, include_rigid_single_bonds=False)
    assert t3 is not t1

    ligand.clear_geometry_caches()
    t4 = get_fragmentation_topology(ligand, include_rigid_single_bonds=True)
    assert t4 is not t1


def test_build_candidate_fragments_reuses_topology():
    ligand = _hexane_like_ligand()
    torsion_a = TorsionDefinition(
        atom_indices=(1, 2, 3, 4),
        bond=(2, 3),
        label="C1-C2-C3-C4",
    )
    torsion_b = TorsionDefinition(
        atom_indices=(3, 4, 5, 6),
        bond=(4, 5),
        label="C3-C4-C5-C6",
    )
    with patch(
        "scission.fragments._build_domains",
        wraps=__import__("scission.fragments", fromlist=["_build_domains"])._build_domains,
    ) as mock_domains:
        build_candidate_fragments(ligand, torsion_a)
        build_candidate_fragments(ligand, torsion_b)
    # Domains built once via get_fragmentation_topology, not once per torsion.
    assert mock_domains.call_count == 1


def test_screen_candidate_skips_cap_build():
    ligand = _hexane_like_ligand()
    torsion = TorsionDefinition(
        atom_indices=(1, 2, 3, 4),
        bond=(2, 3),
        label="C1-C2-C3-C4",
    )
    candidate = CandidateFragment(
        candidate_id="c1",
        retained_atoms={1, 2, 3, 4, 5},
        cut_bonds={(5, 6)},
        cap_sites=[CapSite(retained_atom=5, removed_atom=6, bond_type="1")],
        shell_level=0,
        torsion_labels={torsion.label},
    )
    with patch(
        "scission.screen._build_cap_coordinates", wraps=_build_cap_coordinates
    ) as mock_caps:
        result = screen_candidate(
            ligand, torsion, candidate, angle_step=60, thresholds=ClashThresholds()
        )
    assert mock_caps.call_count == 0
    assert result.accepted in (True, False)
    assert np.isfinite(result.worst_margin)
