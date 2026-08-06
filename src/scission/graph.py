from __future__ import annotations

from collections import deque

import networkx as nx

from .models import Bond, Ligand


def build_graph(ligand: Ligand) -> nx.Graph:
    """Construct a NetworkX graph view of the ligand.

    Args:
        ligand: Parent ligand record.

    Returns:
        An undirected graph annotated with atom and bond metadata.
    """

    graph = nx.Graph()
    for atom in ligand.atoms:
        graph.add_node(
            atom.index,
            atom=atom,
            element=atom.element,
            atom_type=atom.atom_type,
        )
    for bond in ligand.bonds:
        graph.add_edge(
            bond.atom1,
            bond.atom2,
            bond=bond,
            bond_type=bond.bond_type,
        )
    return graph


def ring_bond_set(graph: nx.Graph) -> set[tuple[int, int]]:
    """Return all graph edges that participate in at least one cycle.

    Args:
        graph: Molecular graph to analyze.

    Returns:
        Normalized bond pairs that are part of any cycle.
    """

    ring_edges: set[tuple[int, int]] = set()
    for cycle in nx.cycle_basis(graph):
        for idx in range(len(cycle)):
            edge = tuple(sorted((cycle[idx], cycle[(idx + 1) % len(cycle)])))
            ring_edges.add(edge)
    return ring_edges


def heavy_neighbors(graph: nx.Graph, atom_idx: int) -> list[int]:
    """List non-hydrogen neighbors for a given atom index.

    Args:
        graph: Molecular graph.
        atom_idx: Parent atom index whose neighborhood should be inspected.

    Returns:
        A list of heavy-atom neighbor indices.
    """

    return [
        nbr for nbr in graph.neighbors(atom_idx)
        if graph.nodes[nbr]["element"] != "H"
    ]


def graph_distance_map(graph: nx.Graph) -> dict[int, dict[int, int]]:
    """Return shortest-path distances between every atom pair.

    Args:
        graph: Molecular graph.

    Returns:
        Nested mapping of source atom to destination atom to graph distance.
    """

    return {
        source: lengths
        for source, lengths in nx.all_pairs_shortest_path_length(graph)
    }


def component_beyond_bond(graph: nx.Graph, retained: set[int], atom_in: int, atom_out: int) -> set[int]:
    """Collect the connected component reached by crossing a specific bond.

    Args:
        graph: Molecular graph.
        retained: Atom indices that should stop the traversal.
        atom_in: Atom on the retained side of the crossed bond.
        atom_out: Atom on the far side of the crossed bond.

    Returns:
        The connected component reachable beyond the bond.
    """

    visited = {atom_in}
    queue = deque([atom_out])
    component: set[int] = set()
    while queue:
        node = queue.popleft()
        if node in visited or node in retained:
            continue
        visited.add(node)
        component.add(node)
        for nbr in graph.neighbors(node):
            if nbr not in visited and nbr != atom_in:
                queue.append(nbr)
    return component


def sorted_cut_bond(a: int, b: int) -> tuple[int, int]:
    """Normalize a bond key so atom order does not matter.

    Args:
        a: First parent atom index.
        b: Second parent atom index.

    Returns:
        The atom pair sorted into ascending order.
    """

    return tuple(sorted((a, b)))


def bond_lookup(bonds: list[Bond]) -> dict[tuple[int, int], Bond]:
    """Index bond records by their normalized atom-pair key.

    Args:
        bonds: Bond records to index.

    Returns:
        Mapping from normalized bond pair to the corresponding bond record.
    """

    return {sorted_cut_bond(bond.atom1, bond.atom2): bond for bond in bonds}
