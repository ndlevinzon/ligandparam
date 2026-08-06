from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .capping import bond_type_to_order, heavy_cap_bond_length
from .graph import build_graph, graph_distance_map
from .models import CandidateFragment, ClashThresholds, Ligand, TorsionDefinition

VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}

CAP_BOND_LENGTHS = {
    "C": 1.09,
    "N": 1.01,
    "O": 0.98,
    "S": 1.34,
    "P": 1.42,
}


@dataclass(frozen=True)
class ScreenResult:
    """Outcome of clash-screening a candidate over sampled torsion angles.

    Attributes:
        accepted: Whether the candidate passed all sampled angles.
        worst_margin: Smallest nonbonded margin observed during screening.
        reason: Short failure code when the candidate is rejected.
        violation: Optional details describing the worst clash encountered.
    """

    accepted: bool
    worst_margin: float
    reason: str | None = None
    violation: dict[str, object] | None = None


def _rotation_matrix(axis: np.ndarray, theta_rad: float) -> np.ndarray:
    """Return a 3D rotation matrix around ``axis`` by ``theta_rad`` radians.

    Args:
        axis: Rotation axis vector.
        theta_rad: Rotation angle in radians.

    Returns:
        A ``3 x 3`` rotation matrix.
    """

    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    return np.array(
        [
            [cos_t + x * x * (1 - cos_t), x * y * (1 - cos_t) - z * sin_t, x * z * (1 - cos_t) + y * sin_t],
            [y * x * (1 - cos_t) + z * sin_t, cos_t + y * y * (1 - cos_t), y * z * (1 - cos_t) - x * sin_t],
            [z * x * (1 - cos_t) - y * sin_t, z * y * (1 - cos_t) + x * sin_t, cos_t + z * z * (1 - cos_t)],
        ]
    )


def _dihedral_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Compute the signed dihedral angle defined by four points.

    Args:
        p1: First point.
        p2: Second point.
        p3: Third point.
        p4: Fourth point.

    Returns:
        The dihedral angle in degrees.
    """

    b0 = p2 - p1
    b1 = p3 - p2
    b2 = p4 - p3
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))


def _descendants(graph: nx.Graph, start: int, blocked: int, allowed: set[int]) -> set[int]:
    """Traverse one side of a bond while staying inside the allowed atoms.

    Args:
        graph: Molecular graph.
        start: Atom index from which to begin traversal.
        blocked: Atom index on the opposite side of the bond.
        allowed: Atom indices allowed to remain in the traversal.

    Returns:
        Reachable atoms on the chosen side of the bond.
    """

    stack = [start]
    seen = {blocked}
    descendants: set[int] = set()
    while stack:
        node = stack.pop()
        if node in seen or node not in allowed:
            continue
        seen.add(node)
        descendants.add(node)
        for nbr in graph.neighbors(node):
            if nbr not in seen:
                stack.append(nbr)
    return descendants


def cap_direction(
    graph: nx.Graph,
    ligand: Ligand,
    retained_atom: int,
    removed_atom: int,
    coordinates: dict[int, np.ndarray],
) -> np.ndarray:
    """Choose an outward cap direction for a cut bond.

    Args:
        graph: Molecular graph.
        ligand: Parent ligand record.
        retained_atom: Atom that stays in the fragment.
        removed_atom: Bonded atom removed from the fragment.
        coordinates: Coordinate map for the current conformation.

    Returns:
        A unit vector pointing in the preferred cap direction.
    """

    origin = coordinates[retained_atom]
    neighbor_vectors: list[np.ndarray] = []
    for nbr in graph.neighbors(retained_atom):
        if nbr == removed_atom:
            continue
        nbr_coord = coordinates.get(nbr)
        if nbr_coord is None:
            nbr_coord = np.array(ligand.atom(nbr).coords, dtype=float)
        vector = nbr_coord - origin
        norm = np.linalg.norm(vector)
        if norm > 1.0e-8:
            neighbor_vectors.append(vector / norm)

    if len(neighbor_vectors) >= 2:
        direction = -np.sum(neighbor_vectors, axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1.0e-8:
            return direction / norm

    removed = np.array(ligand.atom(removed_atom).coords, dtype=float)
    fallback = origin - removed
    fallback_norm = np.linalg.norm(fallback)
    if fallback_norm > 1.0e-8:
        return fallback / fallback_norm
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _build_cap_coordinates(
    ligand: Ligand,
    candidate: CandidateFragment,
    coordinates: dict[int, np.ndarray],
    graph: nx.Graph,
) -> dict[str, np.ndarray]:
    """Generate trial cap coordinates for a screened fragment conformation.

    Args:
        ligand: Parent ligand record.
        candidate: Candidate fragment being screened.
        coordinates: Current coordinates for retained parent atoms.
        graph: Prebuilt molecular graph (topology is fixed during a scan).

    Returns:
        Mapping from cap identifier to proposed cap coordinate.
    """

    cap_coords: dict[str, np.ndarray] = {}
    for site in sorted(candidate.cap_sites, key=lambda s: (s.retained_atom, s.removed_atom)):
        retained = ligand.atom(site.retained_atom)
        removed = ligand.atom(site.removed_atom)
        origin = coordinates[site.retained_atom]
        direction = cap_direction(graph, ligand, site.retained_atom, site.removed_atom, coordinates)
        # Place the cap heavy atom (the dominant clash contributor) at the
        # element-matched bond length, conservatively assuming the removed atom
        # is recreated even when a bare hydrogen might ultimately be used.
        bond_length = heavy_cap_bond_length(
            retained.element, removed.element, bond_type_to_order(site.bond_type)
        )
        cap_coords[f"cap_{site.retained_atom}_{site.removed_atom}"] = origin + direction * bond_length
    return cap_coords


def _minimum_margin(
    ligand: Ligand,
    retained_atoms: set[int],
    coordinates: dict[int, np.ndarray],
    caps: dict[str, np.ndarray],
    thresholds: ClashThresholds,
    distances: dict[int, dict[int, int]],
) -> tuple[float, dict[str, object] | None]:
    """Measure the worst nonbonded margin among retained heavy atoms.

    Args:
        ligand: Parent ligand record.
        retained_atoms: Parent atom indices retained in the fragment.
        coordinates: Coordinates for retained atoms in the current conformer.
        caps: Generated cap coordinates keyed by cap identifier.
        thresholds: Clash-threshold parameters.
        distances: Precomputed shortest-path lengths on the retained subgraph.

    Returns:
        A tuple of the worst observed margin and optional detail for the most
        restrictive atom pair.
    """

    entities: list[tuple[str | int, str, np.ndarray]] = []
    for atom_idx in sorted(retained_atoms):
        atom = ligand.atom(atom_idx)
        entities.append((atom_idx, atom.element, coordinates[atom_idx]))
    for cap_name, coord in sorted(caps.items()):
        entities.append((cap_name, "H", coord))

    worst_margin = float("inf")
    worst_detail: dict[str, object] | None = None
    for idx, (left_id, left_element, left_coord) in enumerate(entities):
        for right_id, right_element, right_coord in entities[idx + 1:]:
            if not (isinstance(left_id, int) and isinstance(right_id, int)):
                continue
            if left_element == "H" or right_element == "H":
                continue
            path_length = distances.get(left_id, {}).get(right_id, 99)
            if path_length <= 2:
                continue
            scale = thresholds.path3_scale if path_length == 3 else thresholds.far_scale
            allowed = scale * (VDW_RADII.get(left_element, 1.7) + VDW_RADII.get(right_element, 1.2))
            observed = float(np.linalg.norm(left_coord - right_coord))
            margin = observed - allowed
            if margin < worst_margin:
                worst_margin = margin
                worst_detail = {
                    "left_atom_index": left_id,
                    "left_atom_name": ligand.atom(left_id).name,
                    "left_element": left_element,
                    "right_atom_index": right_id,
                    "right_atom_name": ligand.atom(right_id).name,
                    "right_element": right_element,
                    "graph_distance": path_length,
                    "observed_distance": observed,
                    "allowed_distance": allowed,
                    "margin": margin,
                }
    return (worst_margin if worst_margin != float("inf") else 0.0, worst_detail)


def screen_candidate(
    ligand: Ligand,
    torsion: TorsionDefinition,
    candidate: CandidateFragment,
    angle_step: int,
    thresholds: ClashThresholds,
) -> ScreenResult:
    """Reject fragments whose sampled torsions introduce steric clashes.

    Topology (graph / bond-path distances) is fixed for a candidate; only
    Cartesian coordinates change across the torsion scan.

    Args:
        ligand: Parent ligand record.
        torsion: Torsion being preserved by the candidate.
        candidate: Candidate fragment to evaluate.
        angle_step: Sampling interval in degrees for the torsion scan.
        thresholds: Clash-threshold parameters.

    Returns:
        A :class:`ScreenResult` describing whether the candidate passed.
    """

    graph = build_graph(ligand)
    retained = candidate.retained_atoms
    a, b, c, d = torsion.atom_indices
    if not {a, b, c, d}.issubset(retained):
        return ScreenResult(False, -999.0, "candidate_missing_torsion_atoms", None)
    if not candidate.cut_bonds:
        return ScreenResult(True, 0.0, None, None)

    side_from_c = _descendants(graph, d, c, retained)
    side_from_b = _descendants(graph, a, b, retained)
    rotating_side = side_from_c if len(side_from_c) <= len(side_from_b) else side_from_b
    # Distances through the fragment (retained subgraph), not via cut-away atoms.
    distances = graph_distance_map(graph.subgraph(retained))
    original = ligand.coordinates
    pivot1 = original[b]
    pivot2 = original[c]
    axis = pivot2 - pivot1
    if np.linalg.norm(axis) == 0:
        return ScreenResult(False, -999.0, "zero_length_torsion_axis", None)

    current_dihedral = _dihedral_angle(
        original[a],
        original[b],
        original[c],
        original[d],
    )
    worst = float("inf")
    for target_angle in range(-180, 180, angle_step):
        coords = {atom_idx: coord.copy() for atom_idx, coord in original.items() if atom_idx in retained}
        delta_angle = ((target_angle - current_dihedral + 180.0) % 360.0) - 180.0
        rotation = _rotation_matrix(axis, math.radians(delta_angle))
        pivot = pivot1
        for atom_idx in rotating_side:
            shifted = coords[atom_idx] - pivot
            coords[atom_idx] = rotation @ shifted + pivot
        caps = _build_cap_coordinates(ligand, candidate, coords, graph)
        margin, detail = _minimum_margin(
            ligand, retained, coords, caps, thresholds, distances
        )
        worst = min(worst, margin)
        if margin < 0:
            if detail is not None:
                detail = {**detail, "target_angle": target_angle}
            return ScreenResult(False, worst, f"clash_at_{target_angle}", detail)
    return ScreenResult(True, worst if worst != float("inf") else 0.0, None, None)


def cap_site_scan_margin(
    ligand: Ligand,
    candidate: CandidateFragment,
    torsion: TorsionDefinition,
    retained_atom: int,
    removed_atom: int,
    cap_element: str,
    angle_step: int,
    thresholds: ClashThresholds,
) -> float:
    """Worst clash margin for a heavy cap at one site over the torsion scan.

    Used to decide which carbon cut can carry a bulky (methyl) cap: a roomy site
    yields a large margin, a crowded site a small or negative one. The heavy cap
    atom is placed at the cut site and the torsion is rotated through a full turn;
    the smallest margin between that cap atom and any retained heavy atom is
    returned.

    Args:
        ligand: Parent ligand record.
        candidate: Candidate fragment being capped.
        torsion: Torsion that will be scanned during fitting.
        retained_atom: Parent index of the atom the cap attaches to.
        removed_atom: Parent index of the removed atom defining the cap direction.
        cap_element: Element of the trial heavy cap (typically ``"C"``).
        angle_step: Torsion sampling interval in degrees.
        thresholds: Clash-threshold parameters.

    Returns:
        The worst (smallest) nonbonded margin observed for the cap heavy atom, or
        ``inf`` when the torsion cannot be scanned in this candidate.
    """

    graph = build_graph(ligand)
    retained = candidate.retained_atoms
    a, b, c, d = torsion.atom_indices
    if not {a, b, c, d, retained_atom}.issubset(retained):
        return float("inf")

    # Through-bond separation from the cap (one bond past the retained atom),
    # measured within the fragment so distances do not route through removed atoms.
    sub_distances = nx.single_source_shortest_path_length(graph.subgraph(retained), retained_atom)
    retained_element = ligand.atom(retained_atom).element
    bond_length = heavy_cap_bond_length(retained_element, cap_element, 1)

    original = ligand.coordinates
    if candidate.cut_bonds:
        side_from_c = _descendants(graph, d, c, retained)
        side_from_b = _descendants(graph, a, b, retained)
        rotating_side = side_from_c if len(side_from_c) <= len(side_from_b) else side_from_b
        pivot1 = original[b]
        axis = original[c] - pivot1
        current = _dihedral_angle(original[a], original[b], original[c], original[d])
        angles = list(range(-180, 180, angle_step))
    else:
        rotating_side = set()
        pivot1 = original[b]
        axis = np.zeros(3)
        current = 0.0
        angles = [0]

    worst = float("inf")
    for target_angle in angles:
        coords = {idx: original[idx].copy() for idx in retained}
        if np.linalg.norm(axis) > 1.0e-8:
            delta = ((target_angle - current + 180.0) % 360.0) - 180.0
            rotation = _rotation_matrix(axis, math.radians(delta))
            for idx in rotating_side:
                coords[idx] = rotation @ (coords[idx] - pivot1) + pivot1
        direction = cap_direction(graph, ligand, retained_atom, removed_atom, coords)
        cap_pos = coords[retained_atom] + direction * bond_length
        for atom_idx in retained:
            element = ligand.atom(atom_idx).element
            if element == "H":
                continue
            path_length = 1 + sub_distances.get(atom_idx, 99)
            if path_length <= 2:
                continue
            scale = thresholds.path3_scale if path_length == 3 else thresholds.far_scale
            allowed = scale * (VDW_RADII.get(cap_element, 1.7) + VDW_RADII.get(element, 1.7))
            observed = float(np.linalg.norm(cap_pos - coords[atom_idx]))
            worst = min(worst, observed - allowed)
    return worst if worst != float("inf") else 0.0
