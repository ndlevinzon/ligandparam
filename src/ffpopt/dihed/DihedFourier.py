"""Fourier dihedral primitives, period/phase snap+merge, ParmEd type builders."""

from __future__ import annotations

class PrimDihedFcn(object):
    """ A class representing a single dihedral function with a force constant, phase, and period.
    
    Parameters
    ----------
    fc : float
        The force constant for the dihedral function.
    phase : float   
        The phase of the dihedral function in degrees.
    per : int
        The period of the dihedral function.
    
    Attributes
    ----------
    fc : float
        The force constant for the dihedral function.
    phase : float
        The phase of the dihedral function in degrees.
    per : int
        The period of the dihedral function.
    
    """
    def __init__(self,fc,phase,per):
        self.fc = fc
        self.phase = phase
        self.per = per

    def __str__(self):
        """ Return a string representation of the PrimDihedFcn object. """
        return f"({self.fc},{self.phase},{self.per})"
        
    def CptEne(self,ang):
        """ Calculate the energy contribution of the dihedral function for a given angle.
        
        Parameters
        ----------
        ang : float
            The angle in degrees for which the energy contribution is calculated.

        Returns
        -------
        float
            The energy contribution of the dihedral function for the given angle.
        
        """
        return self.fc * self.CptEterm(ang)

    def CptEterm(self,ang):
        """ Calculate the energy term of the dihedral function for a given angle.

        Parameters
        ----------
        ang : float
            The angle in degrees for which the energy term is calculated.
        
        Returns
        -------
        float
            The energy term of the dihedral function for the given angle.
        
        """
        import numpy as np
        a = (self.per * ang + self.phase)*(np.pi/180)
        return 1+np.cos(a)


def snap_amber_dihed_phase(phase):
    """Map a dihedral phase (deg) onto Amber's 0 or 180."""
    p = float(phase) % 360.0
    if p > 180.0:
        p -= 360.0
    if abs(p) <= 90.0:
        return 0.0
    return 180.0


def amber_dihed_period(per):
    """Nearest Amber Fourier periodicity in ``[1, 12]``."""
    per_i = int(round(float(per)))
    return max(1, min(12, per_i))


def merge_duplicate_period_prims(prims, warn=True, label=None):
    """Collapse Fourier terms that share an Amber periodicity.

    ParmEd ``DihedralTypeList`` forbids two terms with the same ``n``.
    ``--fit-full`` can round two optimized periods onto the same integer.
    Same-``n`` terms are linearly dependent (phase 0 vs 180 flips the cosine
    sign); merging keeps the oscillatory part. The constant offset is absorbed
    by the fit.
    """
    from collections import OrderedDict

    groups = OrderedDict()
    for prim in prims:
        groups.setdefault(amber_dihed_period(prim.per), []).append(prim)

    merged = []
    prefix = "[fit-write]" if not label else f"[fit-write] {label}"
    for per, items in groups.items():
        if len(items) == 1:
            p = items[0]
            merged.append(
                PrimDihedFcn(float(p.fc), snap_amber_dihed_phase(p.phase), per)
            )
            continue
        cos_coeff = 0.0
        for p in items:
            fc = float(p.fc)
            if snap_amber_dihed_phase(p.phase) == 0.0:
                cos_coeff += fc
            else:
                cos_coeff -= fc
        if cos_coeff >= 0.0:
            fc, phase = float(cos_coeff), 0.0
        else:
            fc, phase = float(-cos_coeff), 180.0
        if warn:
            parts = ", ".join(
                f"fc={float(p.fc):.6g} phase={float(p.phase):.1f}"
                for p in items
            )
            print(
                f"{prefix} merged {len(items)} Fourier terms with period n={per} "
                f"({parts}) -> fc={fc:.6g} phase={phase:.0f}",
                flush=True,
            )
        merged.append(PrimDihedFcn(fc, phase, per))
    return merged


def parmed_dihedral_types_from_prims(prims, scee=1.2, scnb=2.0, **kwargs):
    """``DihedralType`` list with unique periodicities for ParmEd writes."""
    from parmed import DihedralType

    return [
        DihedralType(p.fc, p.per, p.phase, scee, scnb)
        for p in merge_duplicate_period_prims(prims, **kwargs)
    ]


def parmed_dihedral_type_list_from_prims(prims, scee=1.2, scnb=2.0, **kwargs):
    """``DihedralTypeList`` that will not raise on duplicate periods."""
    from parmed import DihedralTypeList

    typs = DihedralTypeList()
    for typ in parmed_dihedral_types_from_prims(prims, scee, scnb, **kwargs):
        typs.append(typ)
    return typs

    
class MultiDihedFcn(object):
    """ A class representing a multi-term dihedral function.
    
    Parameters
    ----------
    idxs : list of int
        A list of 4 integers representing the indices of the atoms forming the dihedral.
    prims : list of PrimDihedFcn
        A list of PrimDihedFcn objects representing the individual terms of the dihedral function.
    
    Attributes
    ----------
    idxs : list of int
        A list of 4 integers representing the indices of the atoms forming the dihedral.
    prims : list of PrimDihedFcn
        A list of PrimDihedFcn objects representing the individual terms of the dihedral function.


    """
    def __init__(self,idxs,prims):
        import copy
        self.idxs = list(idxs)
        self.prims = [copy.copy(p) for p in prims]

    def CptEne(self,ang):
        """ Calculate the total energy contribution of the multi-term dihedral function for a given angle.

        Parameters
        ----------
        ang : float
            The angle in degrees for which the total energy contribution is calculated.
        
        Returns
        -------
        float
            The total energy contribution of the multi-term dihedral function for the given angle.
        
        """

        import numpy as np
        outs = []
        for p in self.prims:
            outs.append( p.CptEne(ang) )
        return np.sum(outs,axis=0)

    def SetFCs(self,fcs):
        """ Set the force constants for each term in the multi-term dihedral function.
        
        Parameters
        ----------
        fcs : list of float
            A list of force constants to set for each term in the multi-term dihedral function.
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            If the length of fcs does not match the number of terms in the multi-term dihedral function.
            
        """
        if len(fcs) != len(self.prims):
            raise Exception(f"Size mismatch {len(fcs)} {len(self.prims)}")
        for i in range(len(fcs)):
            self.prims[i].fc = fcs[i]


    def SetPhases(self,ps):
        """ Set the force constants for each term in the multi-term dihedral function.
        
        Parameters
        ----------
        ps : list of float
            A list of phases to set for each term in the multi-term dihedral function.
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            If the length of ps does not match the number of terms in the multi-term dihedral function.
            
        """
        if len(ps) != len(self.prims):
            raise Exception(f"Size mismatch {len(ps)} {len(self.prims)}")
        for i in range(len(ps)):
            self.prims[i].phase = ps[i]

            
    def ResetFCs(self):
        """ Reset the force constants to default values. """
        self.SetFCs( [1]*len(self.prims) )

    def ResetPhases(self):
        """ Reset the force constants to default values. """
        self.SetPhases( [0]*len(self.prims) )

    def __str__(self):
        """ Return a string representation of the MultiDihedFcn object. """
        return "[%s]"%(",".join([str(x) for x in self.prims]))


    
def CptDihedralEne(atoms,mdfs):
    """ Calculate the dihedral energy for a list of MultiDihedFcn objects.
    
    Parameters
    ----------
    atoms : parmed.amber.AmberParm
        The AmberParm object containing the molecular structure.
    mdfs : list of MultiDihedFcn
        A list of MultiDihedFcn objects representing the dihedral functions.
    
    Returns
    -------
    numpy.ndarray
        A numpy array containing the energy contributions for each MultiDihedFcn object.
    
    """
    import numpy as np
    enes = []
    for mdf in mdfs:
        idxs = mdf.idxs
        ang = atoms.get_dihedral(idxs[0],idxs[1],idxs[2],idxs[3])
        e = mdf.CptEne(ang)
        enes.append(e)
    return np.array(enes)
        




def GetDihedClasses(idxs=None):
    """ Get a dictionary of MultiDihedFcn classes based on the provided indices.
    
    Parameters
    ----------
    idxs : list of int, optional
        A list of 4 integers representing the indices of the atoms forming the dihedral. If None, defaults to [0, 1, 2, 3].
    
    Returns
    -------
    dict
        A dictionary where keys are integers representing the number of terms in the dihedral function, and values are lists of MultiDihedFcn objects.
        
    """
    if idxs is None:
        idxs = [0, 1, 2, 3]
    class1 = [MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1)] ),
              MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,2)] ),
              MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,3)] ) ]


    class1 = [MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1)] ) ]


    
    # class2 = [MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1),
    #                                 PrimDihedFcn(1,  0,2)] ) ]

    # class3 = [MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1),
    #                                 PrimDihedFcn(1,  0,2),
    #                                 PrimDihedFcn(1,  0,3)] ) ]

    cs = {}
    cs[1] = class1
    for j in range(2,13):
        cs[j] = [MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,n) for n in range(1,j+1)])]
    

    return cs




