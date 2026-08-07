"""Tests for distance-map cache, shell dedupe, vectorized cap scan, cycle basis."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from scission.fragments import build_candidate_fragments
from scission.graph import retained_distance_map
from scission.models import (
    Atom,
    Bond,
    CapSite,
    CandidateFragment,
    ClashThresholds,
    Ligand,
    TorsionDefinition,
)
from scission.screen import cap_site_scan_margin


def _hexane_like_ligand() -> Ligand:
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


def test_retained_distance_map_caches_by_atom_set():
    ligand = _hexane_like_ligand()
    retained = {1, 2, 3, 4}
    with patch(
        "scission.graph.graph_distance_map",
        wraps=__import__("scission.graph", fromlist=["graph_distance_map"]).graph_distance_map,
    ) as mock_apsp:
        d1 = retained_distance_map(ligand, retained)
        d2 = retained_distance_map(ligand, set(retained))
        d3 = retained_distance_map(ligand, {1, 2, 3, 4, 5})
    assert d1 is d2
    assert mock_apsp.call_count == 2  # one per distinct frozenset
    assert d3 is not d1
    ligand.clear_geometry_caches()
    assert getattr(ligand, "_distance_maps", None) in (None, {})


def test_shell_enumeration_builds_unique_domain_sets_once():
    ligand = _hexane_like_ligand()
    torsion = TorsionDefinition(
        atom_indices=(1, 2, 3, 4),
        bond=(2, 3),
        label="C1-C2-C3-C4",
    )
    with patch(
        "scission.fragments._candidate_from_domains",
        wraps=__import__(
            "scission.fragments", fromlist=["_candidate_from_domains"]
        )._candidate_from_domains,
    ) as mock_build:
        cands = build_candidate_fragments(ligand, torsion)
    # Each call produces a unique candidate_id; no duplicate domain rebuilds.
    ids = [c.candidate_id for c in cands]
    assert len(ids) == len(set(ids))
    # mock_build called once per unique domain combo (+ parent fallback).
    assert mock_build.call_count == len(cands)


def test_cap_site_scan_margin_vectorized_runs():
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
    margin = cap_site_scan_margin(
        ligand,
        candidate,
        torsion,
        retained_atom=5,
        removed_atom=6,
        cap_element="C",
        angle_step=60,
        thresholds=ClashThresholds(),
    )
    assert isinstance(margin, float)
    assert np.isfinite(margin) or margin == float("inf")


def test_find_min_cycles_uses_cycle_basis_format():
    from ffpopt.GraphSearch import GraphSearch

    # Square: 4-cycle
    g = GraphSearch(["0~1", "1~2", "2~3", "3~0"])
    cycles = g.FindMinCycles()
    assert len(cycles) == 1
    cyc = cycles[0]
    assert len(cyc) == 5  # 4 unique + close
    assert cyc[0] == cyc[-1]
