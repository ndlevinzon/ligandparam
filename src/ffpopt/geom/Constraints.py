#!/usr/bin/env python3

def is_integer(val_string):
    try:
        int(val_string)
        return True
    except ValueError:
        return False

    
class ConstraintList(object):
    """ A class to represent a list of constraints.
    
    Attributes
    ----------
    cons : a list of Constraint objects
    """

    def __init__(self,cons):
        self.cons = [c for c in cons]

        
    def __iter__(self):
        return iter(self.cons)

    
    def __len__(self):
        return len(self.cons)


    def __getitem__(self, key):
        """Defines reading via bracket syntax: value = obj[key]"""
        return self.cons[key]

    
    def __setitem__(self, key, value):
        """Defines writing/modifying via bracket syntax: obj[key] = value"""
        # You can add custom validation logic here before storing data
        self.cons[key] = value

        
    def __delitem__(self, key):
        """Defines deletion via bracket syntax: del obj[key]"""
        del self.cons[key]
        
    
    @classmethod
    def from_list_of_dict(cls,ddata):
        if ddata is not None:
            return cls([ Constraint.from_dict(d) for d in ddata ])
        else:
            return cls([])

        
    @classmethod
    def from_list_of_str(cls,ddata):
        if ddata is not None:
            return cls([ Constraint.from_str(d) for d in ddata ])
        else:
            return cls([])

        
    def to_list_of_dict(self):
        return [ c.to_dict() for c in self.cons ]

    
    def to_geometric(self):
        return to_geometric(self.cons)


    def find_pucker_pairs(self):
        return find_pucker_pairs(self.cons)

    
    def to_ase(self):
        return to_ase(self.cons)

    
    def SetMask(self,graph):
        return SetMask(graph,self.cons)

    
    def ApplyConstraints(self,atoms,graph=None):
        return ApplyConstraints(atoms,self.cons,graph=graph)


    def FillConstraints(self,atoms,force=False):
        import copy
        cons = FillConstraints(atoms,self.cons,force=force)
        self.cons = copy.deepcopy(cons)
        return cons



class Constraint(object):
    """ A class to represent a constraint on a dihedral angle, bond length, or angle.
    
    Attributes
    ----------
    name : str
        constraint type: "bond", "angle", "dihed", "puckerx", "puckery"
    idxs : list of int
        A list of 2, 3, 4, or 5 integers representing the indices of atoms
        involved in the constraint. For a bond, it should be 2 integers, for an angle 3 integers, and for a dihedral 4 integers.
    value : float, optional
        The value of the constraint. If None, the value will be calculated from the atoms.
    mask : None or object, optional
        A mask object for the constraint, used in dihedral constraints to specify which atoms are involved
    graph : object, optional
        A graph object that can be used to set the mask for the constraint.
    
    """

    def __init__(self,name,idxs,value=None,graph=None):
        self.name = name
        self.idxs = idxs
        self.value = value
        self.mask = None
        if len(self.idxs) == 2:
            if self.name != "bond":
                raise Exception("Constraint of 2 atoms expected "
                                f"name 'bond' but recevied '{self.name}'")
        elif len(self.idxs) == 3:
            if self.name != "angle":
                raise Exception("Constraint of 3 atoms expected "
                                f"name 'angle' but recevied '{self.name}'")
        elif len(self.idxs) == 4:
            if self.name != "dihed":
                raise Exception("Constraint of 4 atoms expected "
                                f"name 'dihed' but recevied '{self.name}'")
        elif len(self.idxs) == 5:
            if self.name != "puckerx" and self.name != "puckery":
                raise Exception("Constraint of 5 atoms expected "
                                "name 'puckerx' or 'puckery' "
                                f"but recevied '{self.name}'")
        if graph is not None:
            self.SetMask(graph)


    def is_same(self,other):
        return (self.idxs == other.idxs or self.idxs == other.idxs[::-1]) and self.name == other.name
    

    @classmethod
    def from_dict(cls,ddata):
        return cls( ddata["name"], ddata["idxs"], ddata["value"] )

    
    def to_dict(self):
        ddata = {}
        ddata["name"] = self.name
        ddata["idxs"] = [x for x in self.idxs]
        if self.value is not None:
            ddata["value"] = float(self.value)
        else:
            ddata["value"] = self.value
        return ddata
            
            
    @classmethod
    def from_str(cls,s,graph=None):
        """ Create a Constraint object from a string representation.
        
        Parameters
        ----------
        s : str
            A string representation of the constraint. It can be in the format "i,j" for
            a bond, "i,j,k" for an angle, or "i,j,k,l" for a dihedral. Optionally, it can include
            a value, e.g., "i,j=1.5" for a bond with a value of 1.5.
        graph : object, optional
            A graph object that can be used to set the mask for the constraint.
        
        Returns
        -------
        Constraint
            An instance of Constraint initialized with the provided indices and value.

        """
        value = None
        if "=" in s:
            s,value = s.split("=")
            value = float(value)
        cs = s.split(",")
        if not is_integer(cs[0]):
            name = cs[0]
            cs = cs[1:]
        else:
            if len(cs) == 2:
                name="bond"
            elif len(cs) == 3:
                name="angle"
            elif len(cs) == 4:
                name="dihed"
            else:
                name="undef"
        idxs = [int(x) for x in cs]
        return cls(name,idxs,value,graph=graph)
    
    def SetMask(self,graph):
        """ Set the mask for the constraint based on the graph.
        
        Parameters
        ----------
        graph : object
            A graph object that can be used to set the mask for the constraint.
        
        """
        from . AmberParm import RotateMask
        self.mask = None
        if len(self.idxs) == 4:
            self.mask = RotateMask(graph,self.idxs)
        elif len(self.idxs) == 5:
            idxs=self.idxs
            if self.name == "puckerx":
                self.mask = RotateMask(graph,[idxs[4],idxs[0],idxs[1],idxs[2]])
            elif self.name == "puckery":
                self.mask = RotateMask(graph,[idxs[1],idxs[2],idxs[3],idxs[4]])
            
    def __str__(self):
        """ String representation of the Constraint object. """
        s = f"{self.name}," + ",".join(["%i"%(x) for x in self.idxs])
        if self.value is not None:
            s += f"={self.value}"
        return s

    def fill(self,atoms,force=False):
        """ Fill the constraint with the current value from the atoms.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object from which to get the current value of the constraint.
        force : bool, optional
            If True, force the calculation of the value even if it is already set. Default is
            False.
        
        Returns
        -------
        Constraint
            A new Constraint object with the value filled in.
            If the value is already set and force is False, it returns a copy of the current
            Constraint object with the same value.
        
        """
        import copy
        import numpy as np
        from . Geometry import Dihedrals2Pucker
        out = copy.deepcopy(self)
        if self.value is None or force:
            if len(self.idxs) == 2:
                out.value = atoms.get_distance(self.idxs[0],self.idxs[1])
            elif len(self.idxs) == 3:
                out.value = atoms.get_angle(self.idxs[0],self.idxs[1],self.idxs[2])
            elif len(self.idxs) == 4:
                out.value = atoms.get_dihedral(self.idxs[0],self.idxs[1],self.idxs[2],self.idxs[3])
            elif len(self.idxs) == 5:
                v1 = atoms.get_dihedral(self.idxs[4],self.idxs[0],self.idxs[1],self.idxs[2])
                v3 = atoms.get_dihedral(self.idxs[1],self.idxs[2],self.idxs[3],self.idxs[4])
                zx,zy = Dihedrals2Pucker(v1,v3)
                #print(v1,v3,zx,zy,self.idxs[4],self.idxs[0],self.idxs[1],self.idxs[2])
                if self.name == "puckerx":
                    out.value = zx
                elif self.name == "puckery":
                    out.value = zy
                else:
                    raise Exception(f"Expected name=puckerx or puckery, but it is {self.name}")
        return out

    def modify(self,atoms):
        """ Modify the atoms object based on the constraint.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to modify based on the constraint.

        Returns
        -------
        ase.Atoms
            A new atoms object with the constraint applied.
            If the value is None, it returns a copy of the original atoms object.
        
        """
        out = atoms.copy()
        if self.value is not None:
            if len(self.idxs) == 2:
                out.set_distance(self.idxs[0],self.idxs[1],self.value)
            elif len(self.idxs) == 3:
                out.set_angle(self.idxs[0],self.idxs[1],self.idxs[2],self.value)
            elif len(self.idxs) == 4:
                #print("mask=",self.mask)
                out.set_dihedral(self.idxs[0],self.idxs[1],self.idxs[2],self.idxs[3],self.value,mask=self.mask)
            elif len(self.idxs) == 5:
                # We can only do puckerx and puckery in pairs
                pass
        return out

    def to_geometric(self):
        """ Convert the constraint to a geometric string representation.
        
        Returns
        -------
        str
            A string representation of the constraint in a format suitable for geometric constraints.
            The format is "distance i,j value" for bonds, "angle i,j,k value
            for angles, and "dihedral i,j,k,l value" for dihedrals.
        
        Raises
        -------
        Exception
            If the value is None, an exception is raised indicating that the value should not be None
        
        """ 
        if self.value is None:
            raise Exception("self.value should not be None. Use the fill method to assign values")
        s = " ".join( [f"{1+x}" for x in self.idxs] )
        if len(self.idxs) == 2:
            s = "distance "+s+f" {self.value}"
        elif len(self.idxs) == 3:
            s = "angle "+s+f" {self.value}"
        elif len(self.idxs) == 4:
            s = "dihedral "+s+f" {self.value}"
        elif len(self.idxs) == 5:
            s = None
        return s

    def isper(self):
        return len(self.idxs) == 4
    
    
    

# def ApplyConstraints(atoms,cons,graph=None):
#     """ Apply a list of constraints to an atoms object.
    
#     Parameters
#     ----------
#     atoms : ase.Atoms
#         The atoms object to which the constraints will be applied.
#     cons : list of Constraint
#         A list of Constraint objects to apply to the atoms.
#     graph : object, optional
#         A graph object that can be used to set the mask for the constraints.
    
#     Returns
#     -------
#     ase.Atoms
#         A new atoms object with the constraints applied.
#         If no constraints are provided, it returns a copy of the original atoms object.
    
#     """
#     out = atoms.copy()
#     for con in cons:
#         if graph is not None:
#             out.SetMask(graph)
#         out = con.modify(out)
#     return out


# def FillConstraints(atoms,cons,force=False):
#     """ Fill the constraints with the current values from the atoms object.
    
#     Parameters
#     ----------
#     atoms : ase.Atoms
#         The atoms object from which to get the current values of the constraints.
#     cons : list of Constraint
#         A list of Constraint objects to fill with values.
#     force : bool, optional
#         If True, force the calculation of the value even if it is already set. Default is
#         False.
    
#     Returns
#     -------
#     list of Constraint
#         A list of Constraint objects with the values filled in.
#         If the value is already set and force is False, it returns a copy of the current
#         Constraint objects with the same values.
    
#     """
#     return [ con.fill(atoms,force=force) for con in cons ]
        

#def constraints2geometric(cons):
#    """ Convert a list of constraints to a geometric string representation. """
#    return [ con.to_geometric() for con in cons ]


# def constraints2ase(cons):
#     """ Convert a list of constraints to an ASE FixInternals object. 
    
#     Parameters
#     ----------
#     cons : list of Constraint
#         A list of Constraint objects to convert to an ASE FixInternals object.
    
#     Returns
#     -------
#     ase.constraints.FixInternals
#         An ASE FixInternals object containing the constraints.
#         The constraints are grouped into bonds, angles, and dihedrals based on the number
#         of indices in each constraint.
    
#     """
#     from ase.constraints import FixInternals
#     bonds=[]
#     angles=[]
#     diheds=[]
#     for con in cons:
#         if len(con.idxs) == 2:
#             bonds.append( [con.value,con.idxs] )
#         elif len(con.idxs) == 3:
#             angles.append( [con.value,con.idxs] )
#         elif len(con.idxs) == 4:
#             diheds.append( [con.value,con.idxs] )
#     return FixInternals(bonds=bonds,angles_deg=angles,dihedrals_deg=diheds)


# def constraints2info(atoms,cons):
#     """ Convert a list of constraints to an info dictionary for the atoms object. 
    
#     Parameters
#     ----------
#     atoms : ase.Atoms
#         The atoms object to which the constraints will be added.
#     cons : list of Constraint
#         A list of Constraint objects to convert to an info dictionary.
    
#     """ 
#     if cons is not None:
#         atoms.info["constraints"] =  "[" + ",".join( ["\"%s\""%(con) for con in cons] ) + "]"


# def info2constraints(atoms,graph=None):
#     """ Convert the constraints stored in the atoms.info dictionary to a list of Constraint objects.
    
#     Parameters
#     ----------
#     atoms : ase.Atoms
#         The atoms object from which to get the constraints.
#     graph : object, optional
#         A graph object that can be used to set the mask for the constraints.
    
#     Returns
#     -------
#     list of Constraint
#         A list of Constraint objects created from the constraints stored in the atoms.info dictionary.
#         If no constraints are found, it returns None.
    
#     """
#     import json
#     cons = None
#     if "constraints" in atoms.info:
#         cs = json.loads( atoms.info["constraints"] )
#         cons = [ Constraint.from_str( c, graph=graph ) for c in cs ]
#     return cons






def to_geometric(cons):
    from . Geometry import Pucker2Dihedrals
    #print("to_geometric",len(cons))
    glist = []
    for con in cons:
        val = con.to_geometric()
        if val is not None:
            glist.append(val)
    pairs = find_pucker_pairs(cons)
    for key in pairs:
        cx = pairs[key][0]
        cy = pairs[key][1]
        zx = cx.value
        zy = cy.value
        v1,v3 = Pucker2Dihedrals(zx,zy)
        idxs = cx.idxs
        conx = Constraint("dihed",[idxs[4],idxs[0],idxs[1],idxs[2]],value=v1)
        cony = Constraint("dihed",[idxs[1],idxs[2],idxs[3],idxs[4]],value=v3)
        #print(str(cx),str(cy),str(conx),str(cony))
        glist.append( conx.to_geometric() )
        glist.append( cony.to_geometric() )
    return glist


def find_pucker_pairs(cons):
    #print("find_pucker_pairs",len(cons))
    pairs = {}
    for ic,c in enumerate(cons):
        #print(ic,str(c))
        if len(c.idxs) == 5:
            a = ",".join([str(x) for x in c.idxs])
            if a not in pairs:
                pairs[a] = [None,None]
            if c.name == "puckerx":
                pairs[a][0] = c
            elif c.name == "puckery":
                pairs[a][1] = c
            else:
                raise Exception(f"len(idxs)==5 but name is {c.name}")
    for key in pairs:
        if pairs[key][0] is None:
            raise Exception("Incomplete pucker pair {str(pair[key][0])}")
        if pairs[key][1] is None:
            raise Exception("Incomplete pucker pair {str(pair[key][1])}")
    return pairs


def to_ase(cons):
    from ase.constraints import FixInternals
    from . Geometry import Pucker2Dihedrals
    bonds=[]
    angles=[]
    diheds=[]
    for con in cons:
        if len(con.idxs) == 2:
            bonds.append( [con.value,con.idxs] )
        elif len(con.idxs) == 3:
            angles.append( [con.value,con.idxs] )
        elif len(con.idxs) == 4:
            diheds.append( [con.value,con.idxs] )
    pairs = find_pucker_pairs(cons)
    for key in pairs:
        cx = pairs[key][0]
        cy = pairs[key][1]
        zx = cx.value
        zy = cy.value
        v1,v3 = Pucker2Dihedrals(zx,zy)
        idxs = cx.idxs
        diheds.append( [v1,[idxs[4],idxs[0],idxs[1],idxs[2]]] )
        diheds.append( [v3,[idxs[1],idxs[2],idxs[3],idxs[4]]] )
    return FixInternals(bonds=bonds,angles_deg=angles,dihedrals_deg=diheds)


def SetMask(graph,cons):
    for c in cons:
        c.SetMask(graph)


# from ase.calculators.calculator import Calculator
# class ZeroedCalculator(Calculator):
#     implemented_properties = ['energy', 'forces']
    
#     def calculate(self, atoms=None, properties=['energy'], system_changes=['positions']):
#         import numpy as np
#         super().calculate(atoms, properties, system_changes)
#         # Set potential energy to zero
#         self.results['energy'] = 0.0
#         # Set forces on all atoms to an array of zeros [0.0, 0.0, 0.0]
#         self.results['forces'] = np.zeros((len(self.atoms), 3))

        
def ApplyConstraints(atoms,cons,graph=None): #,rests=None,k=1.):
    from . Geometry import Pucker2Dihedrals
    
    #import traceback,sys
    #traceback.print_stack(file=sys.stderr)
    #print("ApplyConstraints",len(cons))

    #return EnforceConstraintsFromOpt(atoms,cons,rests,k=k)

    
    out = atoms.copy()
    
    if graph is not None:
        SetMask(graph,cons)
        
    for it in range(1):
        
        for con in cons:
            out = con.modify(out)
        pairs = find_pucker_pairs(cons)
        for key in pairs:
            cx = pairs[key][0]
            cy = pairs[key][1]
            zx = cx.value
            zy = cy.value
            v1,v3 = Pucker2Dihedrals(zx,zy)
            idxs = cx.idxs
            conx = Constraint("dihed",[idxs[4],idxs[0],idxs[1],idxs[2]],value=v1)
            cony = Constraint("dihed",[idxs[1],idxs[2],idxs[3],idxs[4]],value=v3)
            out = conx.modify(out)
            out = cony.modify(out)


        vals = FillConstraints(out,cons,force=True)
        #for v,c in zip(vals,cons):
        #    print(it,v.value,c.value)
    
    # from ase.constraints import FixInternals
    # from ase.optimize import BFGS
    # bonds=[]
    # angles=[]
    # diheds=[]
    # for con in cons:
    #     if len(con.idxs) == 2:
    #         bonds.append( [con.value,con.idxs] )
    #     elif len(con.idxs) == 3:
    #         angles.append( [con.value,con.idxs] )
    #     elif len(con.idxs) == 4:
    #         diheds.append( [con.value,con.idxs] )
    # pairs = find_pucker_pairs(cons)
    # for key in pairs:
    #     cx = pairs[key][0]
    #     cy = pairs[key][1]
    #     zx = cx.value
    #     zy = cy.value
    #     v1,v3 = Pucker2Dihedrals(zx,zy)
    #     idxs = cx.idxs
    #     diheds.append( [v1,[idxs[4],idxs[0],idxs[1],idxs[2]]] )
    #     #diheds.append( [v1,[idxs[2],idxs[1],idxs[0],idxs[4]]] )
    #     diheds.append( [v3,[idxs[1],idxs[2],idxs[3],idxs[4]]] )

    # print(bonds,angles,diheds)
    # constraint = FixInternals(bonds=bonds,angles_deg=angles,dihedrals_deg=diheds)
    # out.set_constraint(constraint)
    # out.calc = ZeroedCalculator()
    
    # opt = BFGS(out)
    # opt.run(fmax=0.01)
    
    # out.set_constraint()
    # out.calc = None

    
     
    # for con in diheds:
    #     val = out.get_dihedral(*con[1])
    #     print(con[0],val)
    # masks = [ [ idxs[3] ],
    #           [ idxs[4], idxs[0] ] ]
    # nat = len(atoms)
    # for ic,con in enumerate(diheds):
    #     print(ic)
    #     print(masks[ic])
    #     mmask = [idx in masks[ic] for idx in range(nat)]
    #     out.set_dihedral(*con[1],con[0],mask=mmask)
    #     val = out.get_dihedral(*con[1])
    #     print(con[0],val)
    #for con in diheds:
    #    val = out.get_dihedral(*con[1])
    #    print(con[0],val)

    return out


def FillConstraints(atoms,cons,force=False):
    return [ con.fill(atoms,force=force) for con in cons ]


def has_nonbonded_clash(positions, bonds, min_dist=0.8):
    """Vectorized clash check: any non-bonded pair closer than ``min_dist``.

    Parameters
    ----------
    positions : array-like, shape (N, 3)
        Cartesian coordinates (Ang).
    bonds : sequence of pair
        Covalent bonds as ``[i, j]`` or ``(i, j)`` (0-based).
    min_dist : float
        Clash threshold in Ang.

    Returns
    -------
    tuple
        ``(clashed, i, j, dist)`` where ``i, j, dist`` are set only if clashed.
    """
    import numpy as np

    pos = np.asarray(positions, dtype=float)
    n = int(pos.shape[0])
    if n < 2:
        return False, None, None, None

    bonded = np.zeros((n, n), dtype=bool)
    for b in bonds:
        i, j = int(b[0]), int(b[1])
        bonded[i, j] = True
        bonded[j, i] = True
    np.fill_diagonal(bonded, True)

    diff = pos[:, None, :] - pos[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    min_d2 = float(min_dist) * float(min_dist)
    clash = (d2 < min_d2) & (~bonded)
    if not np.any(clash):
        return False, None, None, None
    i, j = map(int, np.argwhere(clash)[0])
    return True, i, j, float(np.sqrt(d2[i, j]))




####################################
####################################






def to_primitive_dict(cons):
    from ase.constraints import FixInternals
    from . Geometry import Pucker2Dihedrals
    bonds=[]
    angles=[]
    diheds=[]
    for con in cons:
        if len(con.idxs) == 2:
            bonds.append( [con.value,con.idxs] )
        elif len(con.idxs) == 3:
            angles.append( [con.value,con.idxs] )
        elif len(con.idxs) == 4:
            diheds.append( [con.value,con.idxs] )
    pairs = find_pucker_pairs(cons)
    for key in pairs:
        cx = pairs[key][0]
        cy = pairs[key][1]
        zx = cx.value
        zy = cy.value
        v1,v3 = Pucker2Dihedrals(zx,zy)
        idxs = cx.idxs
        diheds.append( [v1,[idxs[4],idxs[0],idxs[1],idxs[2]]] )
        diheds.append( [v3,[idxs[1],idxs[2],idxs[3],idxs[4]]] )
    #return FixInternals(bonds=bonds,angles_deg=angles,dihedrals_deg=diheds)
    cs = []
    for x in bonds:
        cs.append( { 'type': 'bond',
                     'value': x[0],
                     'indices': x[1] } )
    for x in angles:
        cs.append( { 'type': 'angle',
                     'value': x[0],
                     'indices': x[1] } )
    for x in diheds:
        cs.append( { 'type': 'dihedral',
                     'value': x[0],
                     'indices': x[1] } )
    return cs


def to_primitive_restraints(cons,k=1):
    from . Restraints import BondRestraint, AngleRestraint, DihedRestraint
    from . Restraints import RestraintList
    
    cs = to_primitive_dict(cons)
    rs = []
    for c in cs:
        if c['type'] == 'bond':
           rs.append( BondRestraint(k,c['indices'],c['value'] ) )
        elif c['type'] == 'angle':
            rs.append( AngleRestraint(k,c['indices'],c['value'] ) )
        elif c['type'] == 'dihedral':
            rs.append( DihedRestraint(k,c['indices'],c['value'] ) )
        else:
            raise Exception(f"Unknown constraint type {c['type']}")
    return RestraintList( rs )









def EnforceConstraintsFromOpt(atoms,cons,rests,k=1.):
    import numpy as np
    from ffpopt.ase.Calculator import RestrainedCalculator, CartCalculator
    from ase.optimize import BFGS
    
    oatoms = atoms.copy()
    rlist    = to_primitive_restraints(cons,k=k)
    if rests is not None:
        rlist.rests.extend(rests)
    for r in rlist.rests:
        r.k = k

    nat = len(atoms)
    wts = np.array( [k]*nat )
    for res in rlist.rests:
        wts[ res.idxs[0] ] *= 0.
        wts[ res.idxs[-1] ] *= 0.
    #nullcalc = CartCalculator(oatoms.get_positions(),wts)
    nullcalc = DummyCalculator()
    oatoms.calc = RestrainedCalculator(nullcalc,rlist.rests)
    #oatoms.calc = nullcalc
    
    optimizer = BFGS(oatoms)
    optimizer.run(fmax=1.e-2,steps=500)
    oatoms.calc = None

    #constraints = to_primitive_dict(cons)
    #return enforce_molecular_constraints(oatoms, constraints, max_iter=500, tolerance=1e-6)
    return oatoms
    



