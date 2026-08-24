#!/usr/bin/env python3
"""Amber / ParmEd helpers used by ffpopt scans and torsion fits.

``CopyParm`` is the canonical shallow-copy helper. ligandparam's
``multiresp.parmhelper`` re-exports it so both packages share one
implementation.
"""


def parmed2ase(mol):
    from parmed import periodic_table
    import numpy as np
    import ase

    qs = np.array([a.charge for a in mol.atoms])
    qsum = sum(qs)
    charge = int(round(sum([a.charge for a in mol.atoms])))
    qs += (charge - qsum) / len(qs)
    eles = [periodic_table.Element[a.element] for a in mol.atoms]
    crds = np.array([[a.xx, a.xy, a.xz] for a in mol.atoms])
    atlist = "".join(["%s1" % (ele) for ele in eles])

    return ase.Atoms(atlist, positions=crds, charges=qs), charge


def parmed2graph(mol):
    from .GraphSearch import GraphSearch

    edges = []
    for x in mol.bonds:
        edges.append("%i~%i" % (x.atom1.idx, x.atom2.idx))
    return GraphSearch(edges)


def bonds2graph(bonds):
    from .GraphSearch import GraphSearch

    edges = []
    for x in bonds:
        edges.append("%i~%i" % (x[0], x[1]))
    return GraphSearch(edges)


def CopyParm(parm):
    """Shallow-copy a ParmEd AmberParm, including coordinates and box."""
    import copy

    try:
        parm.remake_parm()
    except Exception:
        pass
    p = copy.copy(parm)
    p.coordinates = copy.copy(parm.coordinates)
    p.box = copy.copy(parm.box)
    try:
        p.hasbox = copy.copy(parm.hasbox)
    except Exception:
        p.hasbox = False
    return p


def RotateMask(graph, idxs):
    """Build a 0/1 atom mask for rotating about a central bond.

    For dihedral ``idxs = [a, b, c, d]``, the rotatable bond is ``b-c``.
    Atoms are bipartitioned with a single BFS across that bond (same idea as
    scission ``component_beyond_bond``), then the mask is oriented so atom
    ``d`` moves. Prefer the smaller side before flipping.

    Parameters
    ----------
    graph : GraphSearch
        Covalent graph with string node ids (``"0"``, ``"1"``, ...).
    idxs : sequence of int
        Four 0-based atom indices defining the dihedral.

    Returns
    -------
    list of int
        Length ``N`` mask; ``1`` marks atoms that should move under the twist.
    """
    left = "%i" % (idxs[1],)
    right = "%i" % (idxs[2],)
    nat = len(graph.nodes)
    gright = graph.ComponentBeyondBond(left, right)
    gleft = set(graph.nodes) - gright
    gmove = gleft if len(gleft) < len(gright) else gright
    mask = [0] * nat
    for node in gmove:
        mask[int(node)] = 1
    if mask[idxs[3]] == 0:
        mask = [1 - x for x in mask]
    return mask


def RotateBondMask(graph, bondpair):
    """Mask atoms on the ``bondpair[0]`` side of a central bond.

    Parameters
    ----------
    graph : GraphSearch
        Covalent graph with string node ids.
    bondpair : sequence of int
        Two 0-based atom indices ``(left, right)`` for the cut bond.

    Returns
    -------
    list of int
        Length ``N`` mask with ``1`` on the left-side component (including
        ``left``).
    """
    left = "%i" % (bondpair[0],)
    right = "%i" % (bondpair[1],)
    nat = len(graph.nodes)
    gright = graph.ComponentBeyondBond(left, right)
    gleft = set(graph.nodes) - gright
    mask = [0] * nat
    for node in gleft:
        mask[int(node)] = 1
    return mask


