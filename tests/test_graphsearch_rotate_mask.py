"""Tests for ffpopt GraphSearch hygiene and RotateMask bipartition."""

from ffpopt.AmberParm import RotateBondMask, RotateMask
from ffpopt.GraphSearch import GraphSearch


def _linear_chain(n: int = 6) -> GraphSearch:
    """Graph 0-1-2-...-(n-1) with string node ids."""
    edges = [f"{i}~{i + 1}" for i in range(n - 1)]
    return GraphSearch(edges)


def test_find_all_paths_no_mutable_default_leak():
    g = _linear_chain(4)
    p1 = g.FindAllPaths("0", "3")
    p2 = g.FindAllPaths("0", "2")
    assert p1 != p2
    assert p1[0] == ["0", "1", "2", "3"]
    assert p2[0] == ["0", "1", "2"]


def test_component_beyond_bond():
    g = _linear_chain(5)
    right = g.ComponentBeyondBond("2", "3")
    assert right == {"3", "4"}
    left = g.ComponentBeyondBond("3", "2")
    assert left == {"0", "1", "2"}


def test_rotate_mask_moves_terminal_side():
    # Dihedral 0-1-2-3 on a hex chain; bond is 1–2.
    g = _linear_chain(6)
    mask = RotateMask(g, [0, 1, 2, 3])
    assert len(mask) == 6
    # Atom 3 must be in the moving set.
    assert mask[3] == 1
    # Bond atoms: one side includes 0,1; other includes 2,3,4,5.
    # Moving side is the one containing 3 → {2,3,4,5}.
    assert mask == [0, 0, 1, 1, 1, 1]


def test_rotate_bond_mask_left_side():
    g = _linear_chain(5)
    mask = RotateBondMask(g, [1, 2])
    # Left of 1–2 is {0,1}
    assert mask == [1, 1, 0, 0, 0]
