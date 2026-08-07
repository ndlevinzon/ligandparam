"""Tests for scission screening topology caches."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from scission.graph import build_graph
from scission.models import (
    Atom,
    Bond,
    CapSite,
    CandidateFragment,
    ClashThresholds,
    Ligand,
    TorsionDefinition,
)
from scission.screen import screen_candidate


def _butane_like_ligand() -> Ligand:
    """Minimal acyclic C4 chain (indices 1..4) plus a terminal H (5)."""
    atoms = [
        Atom(1, "C1", "C", "c3", -0.1, (0.0, 0.0, 0.0)),
        Atom(2, "C2", "C", "c3", -0.1, (1.5, 0.0, 0.0)),
        Atom(3, "C3", "C", "c3", -0.1, (2.0, 1.4, 0.0)),
        Atom(4, "C4", "C", "c3", -0.1, (3.5, 1.4, 0.0)),
        Atom(5, "H4", "H", "hc", 0.1, (4.1, 1.4, 0.0)),
    ]
    bonds = [
        Bond(1, 1, 2, "1"),
        Bond(2, 2, 3, "1"),
        Bond(3, 3, 4, "1"),
        Bond(4, 4, 5, "1"),
    ]
    return Ligand(
        name="BUT",
        atoms=atoms,
        bonds=bonds,
        lib_atom_names=[a.name for a in atoms],
        lib_atom_types={a.name: a.atom_type for a in atoms},
        frcmod_text="",
        mol2_path=Path("but.mol2"),
        lib_path=Path("but.lib"),
        frcmod_path=Path("but.frcmod"),
    )


def test_ligand_coordinates_cached():
    ligand = _butane_like_ligand()
    first = ligand.coordinates
    second = ligand.coordinates
    assert first is second
    assert set(first) == {1, 2, 3, 4, 5}
    ligand.clear_geometry_caches()
    third = ligand.coordinates
    assert third is not first
    assert np.allclose(third[1], first[1])


def test_build_graph_cached_on_ligand():
    ligand = _butane_like_ligand()
    g1 = build_graph(ligand)
    g2 = build_graph(ligand)
    assert g1 is g2
    assert g1.number_of_nodes() == 5
    ligand.clear_geometry_caches()
    g3 = build_graph(ligand)
    assert g3 is not g1


def test_screen_candidate_builds_graph_once_and_distance_map_once():
    ligand = _butane_like_ligand()
    torsion = TorsionDefinition(
        atom_indices=(1, 2, 3, 4),
        bond=(2, 3),
        label="C1-C2-C3-C4",
    )
    candidate = CandidateFragment(
        candidate_id="c1",
        retained_atoms={1, 2, 3, 4},
        cut_bonds={(4, 5)},
        cap_sites=[CapSite(retained_atom=4, removed_atom=5, bond_type="1")],
        shell_level=0,
        torsion_labels={torsion.label},
    )
    thresholds = ClashThresholds()

    with patch("scission.screen.build_graph", wraps=build_graph) as mock_build, patch(
        "scission.screen.retained_distance_map",
        wraps=__import__(
            "scission.graph", fromlist=["retained_distance_map"]
        ).retained_distance_map,
    ) as mock_dist:
        result = screen_candidate(
            ligand, torsion, candidate, angle_step=60, thresholds=thresholds
        )

    assert mock_build.call_count == 1
    assert mock_dist.call_count == 1
    assert result.accepted in (True, False)
