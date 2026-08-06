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


def RotateMask(graph,idxs):
    left = "%i"%(idxs[1])
    right = "%i"%(idxs[2])
    nat = len(graph.nodes)
    gleft = []
    gright = []
    for n in graph.nodes:
        if n == left:
            gleft.append(int(n))
        elif n == right:
            gright.append(int(n))
        else:
            rleft = len(graph.FindMinPaths(left,n)[0])
            rright = len(graph.FindMinPaths(right,n)[0])
            if rleft < rright:
                gleft.append(int(n))
            else:
                gright.append(int(n))
    mask = [0]*nat
    if len(gleft) < len(gright):
        gmove = gleft
    else:
        gmove = gright
    for i in gmove:
        mask[i] = 1

    if mask[idxs[3]] == 0:
        mask = [ 1-x for x in mask ]
        
    return mask



def RotateBondMask(graph,bondpair):
    left = "%i"%(bondpair[0])
    right = "%i"%(bondpair[1])
    nat = len(graph.nodes)
    gleft = []
    gright = []
    for n in graph.nodes:
        if n == left:
            gleft.append(int(n))
        elif n == right:
            gright.append(int(n))
        else:
            rleft = len(graph.FindMinPaths(left,n)[0])
            rright = len(graph.FindMinPaths(right,n)[0])
            if rleft < rright:
                gleft.append(int(n))
            else:
                gright.append(int(n))
    mask = [0]*nat
    for i in gleft:
        mask[i] = 1
    
    return mask


