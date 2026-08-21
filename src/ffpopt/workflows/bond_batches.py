"""Conservative packing of rotatable bonds into sequential twist batches.

Physical policy (rigor-preserving):

* Do **not** drop bonds for bytype dedupe — each instance still contributes a
  scan profile when jointly fitted.
* Bonds whose central atoms lie within ``couple_radius`` graph bonds are
  treated as coupled: prefer co-batching, and when a coupled cluster must be
  split (size > ``max_batch``), run **contiguous** sub-batches **sequentially**
  with the MM updated between batches (standard iterative 1-D fitting).
* Distant coupling components are separate batches (also sequential).

Defaults favor small batches (``max_batch=2``) once a fragment has more than
two fit bonds. Override with ``FFPOPT_MAX_BONDS_PER_TWIST`` /
``FFPOPT_BOND_COUPLE_RADIUS`` / ``FFPOPT_BOND_BATCH=0``.
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import Iterable, Optional, Sequence


BondPair = tuple[int, int]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_truthy(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def bond_batching_enabled() -> bool:
    """True unless ``FFPOPT_BOND_BATCH=0`` disables automatic packing."""
    return _env_truthy("FFPOPT_BOND_BATCH", True)


def max_bonds_per_twist_batch() -> int:
    """Max bonds per joint twist job (default 2)."""
    return max(1, _env_int("FFPOPT_MAX_BONDS_PER_TWIST", 2))


def bond_couple_radius() -> int:
    """Graph distance at which two rotatable bonds are treated as coupled."""
    return max(0, _env_int("FFPOPT_BOND_COUPLE_RADIUS", 2))


def _normalize_bond(bond: Sequence[int]) -> BondPair:
    a, b = int(bond[0]), int(bond[1])
    return (a, b) if a <= b else (b, a)


def atom_shortest_path_length(adj: dict[int, set[int]], a: int, b: int) -> int:
    """BFS distance between atom indices; ``10**9`` if disconnected."""
    if a == b:
        return 0
    seen = {a}
    q: deque[tuple[int, int]] = deque([(a, 0)])
    while q:
        node, dist = q.popleft()
        for nbr in adj.get(node, ()):
            if nbr in seen:
                continue
            if nbr == b:
                return dist + 1
            seen.add(nbr)
            q.append((nbr, dist + 1))
    return 10**9


def adjacency_from_topology_bonds(
    topo_bonds: Iterable[Sequence[int]],
) -> dict[int, set[int]]:
    """Build an undirected adjacency map from covalent bond index pairs."""
    adj: dict[int, set[int]] = defaultdict(set)
    for bond in topo_bonds:
        i, j = int(bond[0]), int(bond[1])
        adj[i].add(j)
        adj[j].add(i)
    return adj


def adjacency_from_parmed(mol) -> dict[int, set[int]]:
    """Adjacency from a ParmEd structure's ``bonds`` list."""
    pairs = []
    for b in mol.bonds:
        pairs.append((b.atom1.idx, b.atom2.idx))
    return adjacency_from_topology_bonds(pairs)


def rotatable_bond_graph_distance(
    adj: dict[int, set[int]],
    bond_a: Sequence[int],
    bond_b: Sequence[int],
) -> int:
    """Min covalent distance between the two central-atom pairs.

    Sharing an atom → 0; adjacent centrals (A–B and B–C) → 0 or 1.
    """
    a0, a1 = int(bond_a[0]), int(bond_a[1])
    b0, b1 = int(bond_b[0]), int(bond_b[1])
    return min(
        atom_shortest_path_length(adj, a0, b0),
        atom_shortest_path_length(adj, a0, b1),
        atom_shortest_path_length(adj, a1, b0),
        atom_shortest_path_length(adj, a1, b1),
    )


def _order_component_by_proximity(
    indices: list[int],
    dist: list[list[int]],
) -> list[int]:
    """Greedy path through a coupled component (nearest-neighbor)."""
    if not indices:
        return []
    # Stable seed: lowest bond index in the component.
    remaining = set(indices)
    start = min(indices)
    order = [start]
    remaining.remove(start)
    while remaining:
        last = order[-1]
        nxt = min(
            remaining,
            key=lambda j: (dist[last][j], j),
        )
        order.append(nxt)
        remaining.remove(nxt)
    return order


def pack_rotatable_bond_batches(
    bonds: Sequence[Sequence[int]],
    adj: dict[int, set[int]],
    *,
    max_batch: Optional[int] = None,
    couple_radius: Optional[int] = None,
) -> list[list[BondPair]]:
    """Pack rotatable central bonds into sequential twist batches.

    Parameters
    ----------
    bonds
        Central-bond pairs (0-based).
    adj
        Covalent adjacency (atom index → neighbors).
    max_batch
        Soft cap per joint twist (default :func:`max_bonds_per_twist_batch`).
        Coupled clusters larger than this are split into contiguous chunks
        run sequentially with MM updates between chunks.
    couple_radius
        Bonds at graph distance ≤ this value are co-clustered.

    Returns
    -------
    list of batches
        Each batch is a list of ``(i, j)`` bonds for one
        :func:`~ffpopt.workflows.run_dihed_twist_workflow` call.
    """
    pairs = [_normalize_bond(b) for b in bonds]
    # Preserve first-seen order while dropping exact duplicate centrals.
    uniq: list[BondPair] = []
    seen: set[BondPair] = set()
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    n = len(uniq)
    if n == 0:
        return []
    if max_batch is None:
        max_batch = max_bonds_per_twist_batch()
    if couple_radius is None:
        couple_radius = bond_couple_radius()
    max_batch = max(1, int(max_batch))
    couple_radius = max(0, int(couple_radius))
    if n <= max_batch:
        return [list(uniq)]

    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = rotatable_bond_graph_distance(adj, uniq[i], uniq[j])
            dist[i][j] = dist[j][i] = d

    # Coupling graph: edge if distance ≤ couple_radius.
    neighbors: dict[int, set[int]] = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] <= couple_radius:
                neighbors[i].add(j)
                neighbors[j].add(i)

    components: list[list[int]] = []
    unused = set(range(n))
    while unused:
        seed = min(unused)
        unused.remove(seed)
        comp = [seed]
        stack = [seed]
        while stack:
            u = stack.pop()
            for v in neighbors.get(u, ()):
                if v in unused:
                    unused.remove(v)
                    comp.append(v)
                    stack.append(v)
        components.append(sorted(comp))

    batches: list[list[BondPair]] = []
    for comp in components:
        ordered = _order_component_by_proximity(comp, dist)
        for start in range(0, len(ordered), max_batch):
            chunk = ordered[start : start + max_batch]
            batches.append([uniq[i] for i in chunk])
    return batches


def should_batch_bonds(n_bonds: int, *, max_batch: Optional[int] = None) -> bool:
    """True when automatic packing should split a multi-bond twist."""
    if not bond_batching_enabled():
        return False
    if max_batch is None:
        max_batch = max_bonds_per_twist_batch()
    return int(n_bonds) > max(1, int(max_batch))
