"""Fit-input types (ParamType ... FitInputType) for dihedral parameterization."""

from __future__ import annotations

from ffpopt.dihed.DihedMath import AngularStdDev, align_scan_profiles
from ffpopt.dihed.DihedFourier import (
    GetDihedClasses,
    parmed_dihedral_type_list_from_prims,
    parmed_dihedral_types_from_prims,
)
from ffpopt.dihed.DihedParmEd import ChangeDihedrals, WriteParmedScript
from ffpopt.dihed.DihedFitSolve import (
    IsolatedLinearSolve,
    build_fixed_geometry_ll_cache,
    joint_linear_solve_from_caches,
    use_dihed_fit_reopt,
)


class ParamType(object):
    """ A class representing a type of dihedral parameter.
    
    Parameters
    ----------
    name : str
        The name of the dihedral parameter type.
    nprim : int
        The number of primitive terms in the dihedral function.
    masks : list of list of str
        A list of masks, where each mask is a list of 4 strings representing the atom
        indices forming the dihedral.
    
    Attributes
    ----------
    name : str
        The name of the dihedral parameter type.
    nprim : int
        The number of primitive terms in the dihedral function.
    masks : list of list of str
        A list of masks, where each mask is a list of 4 strings representing the atom
        indices forming the dihedral.
    dfcns : list of MultiDihedFcn, optional
        A list of MultiDihedFcn objects representing the dihedral functions for this parameter type.
    
    """
    
    def __init__(self,name,nprim,masks):
        self.name = name
        self.nprim = nprim
        self.masks = masks
        self.dfcns = None


        
    def get_dict(self):
        """ Get a dictionary representation of the ParamType object. """
        d = {}
        d[self.name] = { "nprim": self.nprim,
                         "masks": self.masks }
        return d

    
        
class ParamInstance(object):
    """ A class representing an instance of a dihedral parameter type.
    
    Parameters
    ----------
    ptype : ParamType
        The ParamType object representing the type of dihedral parameter.
    masks : list of list of str
        A list of masks, where each mask is a list of 4 strings representing the atom
        indices forming the dihedral.
    mol : parmed.AmberParm
        The Parm object containing the molecular structure.
    impropers : bool, optional
        If True, only consider improper dihedrals. Default is False.
    
    Attributes
    ----------
    ptype : ParamType
        The ParamType object representing the type of dihedral parameter.
    masks : list of list of str
        A list of masks, where each mask is a list of 4 strings representing the atom
        indices forming the dihedral.
    dihedidxs : list of tuple of int
        A list of tuples, where each tuple contains 4 integers representing the indices of the atoms
        forming the dihedral.

    """
    
    def __init__(self,ptype,masks,mol,impropers=False):
        from parmed.amber.mask import AmberMask
        
        self.ptype = ptype
        self.masks = masks
        self.dihedidxs = []
        
        possible_diheds = []
        for mask in masks:
            ats = []
            for i in range(4):
                sidxs = [idx for idx in AmberMask(mol,mask[i]).Selected()]
                ats.append(sidxs)
            for aidx in ats[0]:
                #amask = f"@{aidx+1}"
                for bidx in ats[1]:
                    #bmask = f"@{bidx+1}"
                    for cidx in ats[2]:
                        #cmask = f"@{cidx+1}"
                        for didx in ats[3]:
                            #dmask = f"@{didx+1}"
                            fwd = (aidx,bidx,cidx,didx)
                            possible_diheds.append(fwd)
            for tgt in possible_diheds:
                for d in mol.dihedrals:
                    if d.improper and not impropers:
                        continue
                    fwd = (d.atom1.idx,d.atom2.idx,d.atom3.idx,d.atom4.idx)
                    rev = tuple(list(fwd)[::-1])
                    if fwd == tgt or rev == tgt:
                        if tgt not in self.dihedidxs:
                            self.dihedidxs.append(tgt)
                            


    def get_dict(self):
        """ Get a dictionary representation of the ParamInstance object. """
        from collections import defaultdict as ddict
        d = ddict(list)
        name = self.ptype.name
        for didx in self.dihedidxs:
            d[name].append( [ f"@{didx[0]+1}",f"@{didx[1]+1}",
                              f"@{didx[2]+1}",f"@{didx[3]+1}" ] )
        return d


class ProfileType(object):
    """ A class representing a profile for dihedral parameters.
    
    Parameters
    ----------
    hl : str
        The name of the high-energy scan file.
    ll : str
        The name of the low-energy scan file.
    name : str
        The name of the profile.
    plots : list of str
        A list of strings representing the plots to be generated for this profile.
    stdargs
        The standard arguments containing the molecular structure and calculator.
    
    Attributes
    ----------
    hl : str
        The name of the high-energy scan file.
    ll : str
        The name of the low-energy scan file.
    name : str
        The name of the profile.
    plots : list of str
        A list of strings representing the plots to be generated for this profile.
    #hlatoms : list of ase.Atoms
    #    A list of ase.Atoms objects representing the high-energy scan geometries.
    #hlcons : list of Constraint
    #    A list of Constraint objects representing the constraints for the high-energy scan geometries.
    #llatoms : list of ase.Atoms
    #    A list of ase.Atoms objects representing the low-energy scan geometries.
    #llcons : list of Constraint
    #    A list of Constraint objects representing the constraints for the low-energy scan geometries.
    loshl : ffpopt.Struct.ListOfStruct
         The list of high-level calculations
    losll : ffpopt.Struct.ListOfStruct
         The list of low-level calculations
    """
    def __init__(self,hl,ll,name,plots,stride):
        #from ffpopt.Reader import ReadGeomsFromXYZ
        from ffpopt.Struct import ListOfStruct
        self.hl = hl
        self.ll = ll
        self.name = name
        self.plots = plots
        #self.hlatoms,self.hlcons = ReadGeomsFromXYZ(stdargs,self.hl)
        #self.llatoms,self.llcons = ReadGeomsFromXYZ(stdargs,self.ll)
        self.loshl = ListOfStruct.from_file(self.hl)
        self.losll = ListOfStruct.from_file(self.ll)

        n_hl = len(self.loshl.structs)
        n_ll = len(self.losll.structs)
        self.loshl, self.losll, info = align_scan_profiles(
            self.loshl, self.losll, hl_path=self.hl, ll_path=self.ll
        )
        n_common = int(info.get("n_common", 0) or 0)
        if n_common == 0:
            raise ValueError(
                f"Profile '{self.name}': HL/LL scans share no common angles "
                f"(HL={n_hl}, LL={n_ll}); cannot fit."
            )
        if info.get("interpolated"):
            import sys

            sys.stderr.write(
                f"[ffpopt] Profile '{self.name}': interpolated HL energies onto "
                f"{n_common} LL scan angles "
                f"(input HL={n_hl}, LL={n_ll}; kept LL geometries; "
                f"exact matches={info.get('n_exact', 0)}).\n"
            )
        elif (
            n_hl != n_ll
            or info.get("hl_only")
            or info.get("ll_only")
            or n_common != n_hl
            or n_common != n_ll
        ):
            import sys

            sys.stderr.write(
                f"[ffpopt] Profile '{self.name}': HL/LL aligned on "
                f"{n_common} shared scan angles "
                f"(input HL={n_hl}, LL={n_ll}; "
                f"dropped HL-only={len(info['hl_only'])}, "
                f"LL-only={len(info['ll_only'])}).\n"
            )
            if info["hl_only"]:
                sys.stderr.write(
                    f"[ffpopt]   HL-only angles: {info['hl_only']}\n"
                )
            if info["ll_only"]:
                sys.stderr.write(
                    f"[ffpopt]   LL-only angles: {info['ll_only']}\n"
                )
        
        s = stride
        self.loshl.structs = self.loshl.structs[::s]
        self.losll.structs = self.losll.structs[::s]

        #self.hlatoms = self.hlatoms[::s]
        #self.hlcons = self.hlcons[::s]
        #self.llatoms = self.llatoms[::s]
        #self.llcons = self.llcons[::s]

        
    def get_dict(self):
        """ Get a dictionary representation of the ProfileType object. """
        return { "hl": self.hl, "ll": self.ll,
                 "name": self.name, "plots": self.plots }

        
        
class SystemType(object):
    """ A class representing a system for dihedral parameter fitting.
    
    Parameters
    ----------
    parm : Amber parm7 filename
        Original parameter file
    output : str
        The name of the output file for the fitted parameters.
    profilelist : list of dict
        A list of dictionaries representing the profiles for this system.
    pdict : dict
        A dictionary where keys are parameter names and values are lists of masks for those parameters.
    ptypedict : dict
        A dictionary where keys are parameter names and values are ParamType objects representing the types of di
        hedral parameters.
    
    Attributes
    ----------
    parm : str, Amber parm7 filename
        The filename of the original parm7 file
    mol : parmed parm7 structure
        The parmed representation of the parm7 file
    output : str
        The name of the output file for the fitted parameters.
    pinstances : list of ParamInstance
        A list of ParamInstance objects representing the instances of dihedral parameters for this system.
    profiles : list of ProfileType
        A list of ProfileType objects representing the profiles for this system.
    
    """
    
    def __init__(self,parm,output,profilelist,pdict,ptypedict,stride=1):
        import parmed
        #self.stdargs = stdargs
        self.parm = parm
        self.mol = parmed.load_file(self.parm)
        self.output = output
        self.pinstances = []
        self.profiles = []
        
        for pname in pdict:
            if pname not in ptypedict:    
                raise Exception(f"Parameter {pname} in {parm} "
                                +f"not found in parameter list "
                                +f"{[name for name in ptypedict]}")
            
            ptype = ptypedict[pname]
            masks = pdict[pname]
            self.pinstances.append( ParamInstance(ptype,masks,self.mol) )

        for pname in ptypedict:
            if pname not in pdict:
                ptype = ptypedict[pname]
                masks = ptype.masks
                pinst = ParamInstance(ptype,masks,self.mol)
                if len(pinst.dihedidxs) > 0:
                    self.pinstances.append(pinst)
            
        for profile in profilelist:
            for key in ["hl","ll","name","plots"]:
                if key not in profile:
                    raise Exception(f"'profiles' section is missing '{key}' key")
            hl = profile["hl"]
            ll = profile["ll"]
            name = profile["name"]
            plots = profile["plots"]
            self.profiles.append( ProfileType(hl,ll,name,plots,stride) )

            

    def make_new_parm(self):
        """ Create a new Parm object with the dihedral parameters set according to the instances.
        
        Returns
        -------
        parmed.AmberParm
            A new Parm object with the dihedral parameters set according to the instances.
        
        """
        from ffpopt.AmberParm import CopyParm

        p = CopyParm(self.mol)
        
        scee = 1.2
        scnb = 2.0
        owner = getattr(self, "_fit_owner", None)
        if owner is not None:
            scee = float(getattr(owner, "scee", scee))
            scnb = float(getattr(owner, "scnb", scnb))
        for pinst in self.pinstances:
            pname = getattr(pinst.ptype, "name", None)
            xs = parmed_dihedral_types_from_prims(
                pinst.ptype.dfcns.prims,
                scee=scee,
                scnb=scnb,
                label=pname,
            )
            for idxs in pinst.dihedidxs:
                ChangeDihedrals(p,idxs,xs)
        return p

    
            
    def get_dict(self):
        """ Get a dictionary representation of the SystemType object.
        
        Returns
        -------
        dict
            A dictionary containing the standard arguments, output file name, parameter instances, and profiles.
        
        """
        params = {}
        profiles = []

        for p in self.pinstances:
            x = p.get_dict()
            for key in x:
                params[key] = x[key]

        for p in self.profiles:
            profiles.append(p.get_dict())
        
        d = { "parm": self.parm,
              #"crd": self.stdargs.crd,
              "output": self.output,
              "params": params,
              "profiles": profiles }
        return d


    
    def find_pinstance(self,pname):
        """ Find a ParamInstance by its name.
        
        Parameters
        ----------
        pname : str
            The name of the parameter type to find.
        
        Returns
        -------
        ParamInstance or None
            The ParamInstance object if found, otherwise None.
            
        """
        p=None
        for pinst in self.pinstances:
            if pinst.ptype.name == pname:
                p = pinst
        return p


    def write_output(self):
        """ Write the output file with the dihedral parameters.
        
        This method creates a new Parm object with the dihedral parameters set according to the instances,
        and writes the parameters to the specified output file using the WriteParmedScript function.
        
        Returns
        -------
        None
        
        """
        import copy
        dfcns = []
        for pinst in self.pinstances:
            for idxs in pinst.dihedidxs:
                dfcn = copy.deepcopy(pinst.ptype.dfcns)
                dfcn.idxs = idxs
                dfcns.append(dfcn)
        WriteParmedScript(
            self.output,
            self.mol,
            dfcns,
            scee=float(getattr(getattr(self, "_fit_owner", None), "scee", 1.2)),
            scnb=float(getattr(getattr(self, "_fit_owner", None), "scnb", 2.0)),
        )
    
    
class FitInputType(object):
    """ A class representing the input for fitting dihedral parameters.
    
    This class is used to read the input data from a JSON file and create instances of ParamType, SystemType, and ProfileType.
    It also provides methods to create initial guesses for the dihedral parameters and to write the output in a specific format.
    
    Parameters
    ----------
    args
        The standard arguments containing the molecular structure and calculator.
    datadict : dict
        A dictionary containing the input data for fitting dihedral parameters.
    
    Attributes
    ----------
    iteration : int
        The current iteration number for fitting dihedral parameters.
    ptypedict : dict
        A dictionary where keys are parameter names and values are ParamType objects representing the types of di
        hedral parameters.
    output : str
        The name of the output file for the fitted parameters.
    systems : list of SystemType
        A list of SystemType objects representing the systems for which dihedral parameters are to be fitted
    
    """
    
    @classmethod
    def from_file(cls,args,fname):
        import json
        data = {}
        with open(fname,"r") as file:
            data = json.load(file)
        return cls(args,data)
    
        
    def __init__(self,args,datadict):
        #from ffpopt.Options import StandardArgs
        self.iteration = 0

        for key in ["params","output","systems"]:
            if key not in datadict:
                raise Exception(f"Input is missing '{key}' key")
                
        self.ptypedict = {}
        for pname in datadict["params"]:
            nprim = datadict["params"][pname]["nprim"]
            masks = datadict["params"][pname]["masks"]
            if masks is not None:
                for mask in masks:
                    for atmask in mask:
                        if atmask[:2] != "@%":
                            raise Exception(f"Global dihedral {mask} contains mask {atmask}, which should start with @%%")
            self.ptypedict[pname] = ParamType(pname,nprim,masks)
            #print(pname)
        self.output = datadict["output"]
        self.systems = []
        for s in datadict["systems"]:
            for key in ["parm","output","params","profiles"]:
                if key not in s:
                    raise Exception(f"'systems' section is missing '{key}' key")
            parm = s["parm"]
            #crd = s["crd"]
            output = s["output"]
            pdict = s["params"]
            profilelist = s["profiles"]
            #stdargs = StandardArgs.from_manual_parm(parm,crd,args)
            self.systems.append\
                ( SystemType\
                  (parm,output,profilelist,pdict,self.ptypedict) )

        unused = []
        for pname in self.ptypedict:
            found=False
            for s in self.systems:
                for pinst in s.pinstances:
                    if pinst.ptype.name == pname:
                        found=True
            if not found:
                unused.append(pname)
        for pname in unused:
            print(f"Removing parameter {pname} because it was unused")
            del self.ptypedict[pname]
                
            
    def get_dict(self):
        """ Get a dictionary representation of the FitInputType object.
        
        A dictionary containing the parameters, output file name, and systems.

        Returns
        -------
        dict
            A dictionary containing the parameters, output file name, and systems.

        """
        d = {}
        d["params"] = {}
        for pname in self.ptypedict:
            x = self.ptypedict[pname].get_dict()
            d["params"][pname] = x[pname]
        d["output"] = self.output
        d["systems"] = []
        for s in self.systems:
            d["systems"].append( s.get_dict() )

        return d

    def get_json(self):
        """ Get a JSON representation of the FitInputType object."""
        import json
        d = self.get_dict()
        return json.dumps(d,indent=4)
            

    def _ensure_dfcns_templates(self):
        """Create phase=0 MultiDihedFcn shells for every ParamType if missing."""
        for pname, ptype in self.ptypedict.items():
            if ptype.dfcns is not None:
                continue
            idxs = [0, 1, 2, 3]
            for s in self.systems:
                for pinst in s.pinstances:
                    if pinst.ptype is ptype or pinst.ptype.name == pname:
                        if pinst.dihedidxs:
                            idxs = list(pinst.dihedidxs[0])
                            break
                else:
                    continue
                break
            import copy

            classes = GetDihedClasses(idxs=idxs).get(ptype.nprim)
            if not classes:
                raise ValueError(
                    f"No dihedral Fourier template for {pname} nprim={ptype.nprim}"
                )
            ptype.dfcns = copy.deepcopy(classes[0])
            ptype.dfcns.SetFCs([0.0] * ptype.nprim)

    def _count_fit_points(self):
        return sum(len(p.losll.structs) for s in self.systems for p in s.profiles)

    def make_initial_guesses(self, args=None, caches=None):
        """Create initial FC guesses matching the NL (all-torsions-deleted) model.

        Prefers a joint regularized linear solve over all fitted parameter types
        using the fixed-geometry LL cache. Falls back to per-type isolated LS
        when no cache is available or the joint solve fails.

        Parameters
        ----------
        args : optional
            Wavefront/calculator args used to build the LL cache when ``caches``
            is omitted and reopt mode is off.
        caches : list, optional
            Per-system outputs of :func:`build_fixed_geometry_ll_cache`.

        Returns
        -------
        numpy.ndarray
            Force-constant vector in :meth:`get_params` order.
        """
        import numpy as np

        self._ensure_dfcns_templates()
        n = self.get_num_params()
        if n == 0:
            return np.array([], dtype=float)

        npts = self._count_fit_points()
        if npts == 0:
            raise ValueError(
                "make_initial_guesses: no usable HL/LL profile geometries to fit"
            )
        if npts < 2:
            raise ValueError(
                f"make_initial_guesses: need >= 2 scan points; have {npts}"
            )

        if caches is None and args is not None and not use_dihed_fit_reopt():
            caches = [
                build_fixed_geometry_ll_cache(s, args) for s in self.systems
            ]

        if caches is not None:
            try:
                from ffpopt.constants import (
                    AU_PER_ELECTRON_VOLT,
                    AU_PER_KCAL_PER_MOL,
                )
                from ffpopt.dihed.DihedFitRegularize import (
                    apply_nprim_selection_from_caches,
                )

                kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
                apply_nprim_selection_from_caches(self, caches, kcal_per_ev)
                x, info = joint_linear_solve_from_caches(self, caches)
                print(
                    f"[ffpopt] Joint linear initial guess: rank={info['rank']}/"
                    f"{info['nparam']}, cond~={info['cond']:.3e}, "
                    f"npts={info['npts']}"
                )
                self.set_params(x)
                return x
            except Exception as exc:
                print(f"[ffpopt] Joint LS failed ({exc}); falling back to isolated LS")

        for pname in self.ptypedict:
            bests = None
            bestpinst = None
            bestprof = None
            bestidxs = None
            beststd = -1

            for s in self.systems:
                for pinst in s.pinstances:
                    if pinst.ptype.name != pname:
                        continue
                    for idxs in pinst.dihedidxs:
                        for prof in s.profiles:
                            angs = []
                            for struct in prof.losll:
                                atoms = struct.GetASEAtoms()
                                ang = atoms.get_dihedral(*idxs)
                                if abs(ang - 360) < 0.01:
                                    ang = 0.0
                                angs.append(ang)
                            if len(angs) > 2:
                                astd = AngularStdDev(angs)
                                if astd > beststd:
                                    beststd = astd
                                    bestprof = prof
                                    bestpinst = pinst
                                    bestidxs = idxs
                                    bests = s
            if bestprof is None or bestpinst is None:
                raise ValueError(
                    f"make_initial_guesses: no usable profile for parameter {pname}"
                )
            llgeoms = bestprof.losll
            hlenes = [
                hlgeom.data["energy"] for hlgeom in bestprof.loshl
            ]
            nprim = bestpinst.ptype.nprim
            dfcns = IsolatedLinearSolve(
                bests.mol, bestidxs, llgeoms, hlenes, nprim, pname
            )
            # Share one MultiDihedFcn object on the ParamType.
            n_keep = len(dfcns.prims)
            for s in self.systems:
                for pinst in s.pinstances:
                    if pinst.ptype.name == pname:
                        pinst.ptype.dfcns = dfcns
                        pinst.ptype.nprim = n_keep
            self.ptypedict[pname].dfcns = dfcns
            self.ptypedict[pname].nprim = n_keep
        return self.get_params()

            
    def get_num_params(self):
        """ Get the total number of primitive parameters across all parameter types.
        
        Returns
        -------
        int
            The total number of primitive parameters across all parameter types.

        """

        n = 0
        for pname in self.ptypedict:
            n += self.ptypedict[pname].nprim
        return n

    

    def get_params(self):
        """ Get the current values of the dihedral function coefficients.
        
        Returns
        -------
        numpy.ndarray
            A numpy array containing the current values of the dihedral function coefficients for all parameter types.
        
        """
        import numpy as np
        x=[]
        for pname in self.ptypedict:
            ptype = self.ptypedict[pname]
            dfcns = ptype.dfcns
            for prim in dfcns.prims:
                x.append( prim.fc )
        return np.array(x)


    
    def set_params(self,x):
        """ Set the dihedral function coefficients based on the provided array.
        
        Parameters
        ----------
        x : numpy.ndarray
            A numpy array containing the new values for the dihedral function coefficients.
        
        """
        ipar = 0
        for pname in self.ptypedict:
            ptype = self.ptypedict[pname]
            nprim = ptype.nprim
            ptype.dfcns.SetFCs( x[ipar:ipar+nprim] )
            ipar += nprim

    def write_output(self):
        """ Write the output file with the dihedral parameters.
        
        This method creates a new AmberParameterSet object, populates it with the dihedral types from the parameter types,
        and writes the parameters to the specified output file.
        
        Returns
        -------
        None
        
        """
        from parmed.amber import AmberParameterSet
        #from parmed.amber.mask import AmberMask
        
        for s in self.systems:
            s.write_output()
            
        fmod = AmberParameterSet()
        for pname in self.ptypedict:
            ptype = self.ptypedict[pname]
            if ptype.masks is not None and ptype.dfcns is not None:
                prims = list(ptype.dfcns.prims)
                # K=0 means "keep GAFF": omit the key so merge does not
                # overwrite the parent term with a zero barrier.
                if prims and all(abs(float(p.fc)) < 1.0e-12 for p in prims):
                    print(
                        f"[ffpopt] skip writing {pname}: all FCs are 0 "
                        "(keep parent GAFF)"
                    )
                    continue
                scee = float(getattr(self, "scee", 1.2))
                scnb = float(getattr(self, "scnb", 2.0))
                typs = parmed_dihedral_type_list_from_prims(
                    prims,
                    scee=scee,
                    scnb=scnb,
                    label=pname,
                )
                
                for mask in ptype.masks:
                    atypes = [ x[2:] for x in mask ]
                    fwd = tuple(atypes)
                    rev = tuple(list(fwd)[::-1])
                    fmod.dihedral_types[fwd] = typs
                    fmod.dihedral_types[rev] = typs
                    
        fmod.write(self.output)
    
