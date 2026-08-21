"""Compat re-export; canonical: :mod:`ffpopt.workflows.bond_batches`."""

from ffpopt.workflows.bond_batches import (  # noqa: F401
    adjacency_from_parmed,
    adjacency_from_topology_bonds,
    atom_shortest_path_length,
    bond_batching_enabled,
    bond_couple_radius,
    max_bonds_per_twist_batch,
    pack_rotatable_bond_batches,
    rotatable_bond_graph_distance,
    should_batch_bonds,
)
