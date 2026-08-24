"""Isolated linear solve, design-matrix/LL cache, and nonlinear dihedral fit."""

from __future__ import annotations

from ffpopt.dihed.DihedFourier import GetDihedClasses
from ffpopt.dihed.DihedParmEd import DeleteDihedrals, GetMultiDihedFcnFromIdxs

def EnergyScansWithoutDihedrals(mol,list_of_los,cons):
    """ Calculate energies for a list of scans without dihedrals.
    
    Parameters
    ----------
    stdargs
        The standard arguments containing the molecular structure and calculator.
    list_of_scans : list of list of ase.Atoms
        A list of scans, where each scan is a list of ase.Atoms objects representing different geometries.
    cons : list of Constraint
        A list of Constraint objects representing the constraints to be applied to the geometries.
    
    Returns
    -------
    list of list of float
        A list of lists, where each inner list contains the energies for the corresponding scan.
    
    """
    from ffpopt.AmberParm import CopyParm
    from ffpopt.dihed.DihedParmEd import DeleteDihedrals
    from ffpopt.dihed.DihedParmEd import GetMultiDihedFcnFromIdxs
    from ffpopt.Struct import ListOfStruct
    from tempfile import mkstemp
    import os
    
    p = CopyParm(mol)

    DeleteDihedrals(p,[ x.idxs for x in cons ])

    fd,path = mkstemp(dir=".",prefix="tmp.",suffix=".parm7")
    if not os.isatty(fd):  # Check if fd is still valid
        os.close(fd)
    #fh = os.fdopen(fd,"w")
    #p.save("tmp.parm7",overwrite=True)
    p.save(path,overwrite=True,format="amber")
    # Topology-sharing clones with an overridden parm path (avoid full los deepcopy).
    llos = []
    for los in list_of_los:
        structs = [
            s.clone_geometry(coords=s.data["positions"], ene=s.data.get("energy"), frcs=s.data.get("forces"))
            for s in los
        ]
        for s in structs:
            s.data["parm"] = path
        nlos = ListOfStruct.from_structs_shared(structs, args=getattr(los, "args", None))
        nlos.calc = None
        llos.append(nlos)
    
    #calc = stdargs.MakeCalc(parm=path)
    
    list_of_enes = []
    for los in llos:
        enes = []
        
        for geom in los:
            t = geom.copy()
            t.constraints = None
            t.restraints = None
            t.data["constraints"] = []
            t.data["restraints"] = []
            g = t.GetASEAtoms()
            g.calc=los.BuildCalc(t)
            #g.calc = calc
            #g.calc.reset()
            e = g.get_potential_energy()
            enes.append(e)
        list_of_enes.append(enes)
        
    if os.path.exists(path):
        os.remove(path)
        
    for los in llos:
        los.calc = None
        
    return list_of_enes



def IsolatedLinearSolve(mol,idxs,losll,hlenes,nprim,pname):
    """ Solve the isolated linear problem for dihedral parameters.
    
    Parameters
    ----------
    stdargs
        The standard arguments containing the molecular structure and calculator.
    idxs : list of int
        A list of 4 integers representing the indices of the atoms forming the dihedral.
    llgeoms : list of ase.Atoms
        A list of ase.Atoms objects representing the geometries for the low-energy scan.
    hlenes : list of float
        A list of floats representing the energies for the high-energy scan.
    nprim : int
        The number of primitive terms in the dihedral function.
    pname : str
        The name of the dihedral parameter set.
    
    Returns
    -------
    MultiDihedFcn
        A MultiDihedFcn object representing the best-fit dihedral function for the given parameters.
        
    """
    #from ffpopt.AmberParm import GetDihedClasses
    from ffpopt.geom.Constraints import FillConstraints
    from ffpopt.geom.Constraints import Constraint
    import numpy as np
    import copy
    from ffpopt.constants import AU_PER_KCAL_PER_MOL
    from ffpopt.constants import AU_PER_ELECTRON_VOLT

    KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()

    graph = losll.structs[0].GetGraph()
    dfcns = GetDihedClasses(idxs=idxs)[nprim]
    list_of_scans = [ losll ]
    cons = [ Constraint("dihed",idxs,graph=graph) ]
    list_of_enes = EnergyScansWithoutDihedrals(mol,list_of_scans,cons)
    llenes = np.array(list_of_enes[0])
    hlenes = np.array(hlenes,copy=True)

    llenes *= KCAL_PER_EV
    hlenes *= KCAL_PER_EV

    angs = []
    for igeom,llgeom in enumerate(losll):
        g = llgeom.GetASEAtoms()
        o = FillConstraints(g,cons,force=True)
        v = o[0].value
        if abs(360-v) < 0.01:
            v=0
        angs.append(v)

    data = []
    for i in range(len(losll)):
        data.append( [angs[i],llenes[i],hlenes[i]] )
    data = sorted(data,key=lambda x: x[0])
    angs = np.array( [x[0] for x in data] )
    llenes = np.array( [x[1] for x in data] )
    hlenes = np.array( [x[2] for x in data] )
    # Shape match: free vertical offset only (not independent HL/LL min shifts).
    y = hlenes - llenes
    y_c = y - np.mean(y)

    npts = len(y)
    if npts < nprim + 1:
        raise ValueError(
            f"IsolatedLinearSolve({pname}): need >= {nprim + 1} points, have {npts}"
        )

    bestdfcn = None
    bestchisq = 1.e+30
    bestvalues = []
    best_rank = None
    for ifcn,dfcn in enumerate(dfcns):
        A = np.zeros( (npts, nprim) )
        for iprim,prim in enumerate(dfcn.prims):
            A[:,iprim] = prim.CptEterm(angs)
        A_c = A - np.mean(A, axis=0, keepdims=True)
        x, _residues, rank, singular = np.linalg.lstsq(A_c, y_c, rcond=None)
        if rank < nprim:
            print(
                f"[ffpopt] IsolatedLinearSolve({pname}) class {ifcn}: "
                f"rank={rank}/{nprim}, s={singular}"
            )
        dfcn.SetFCs(x)
        v = dfcn.CptEne(angs)
        d = shape_match_delta(hlenes, llenes + v)
        chisq = float(np.dot(d, d))
        if chisq < bestchisq:
            bestchisq = chisq
            bestdfcn = copy.deepcopy(dfcn)
            bestvalues = v
            best_rank = rank

    if bestdfcn is None:
        raise ValueError(f"IsolatedLinearSolve({pname}): no usable Fourier class")
    if best_rank is not None and best_rank < nprim:
        print(
            f"[ffpopt] IsolatedLinearSolve({pname}): best rank={best_rank} "
            f"< nprim={nprim}"
        )

    # Display-only min shifts for iso.*.dat plots.
    hl_plot = hlenes - np.amin(hlenes)
    ll_plot = llenes - np.amin(llenes)
    fit_raw = llenes + bestvalues
    fit_plot = fit_raw - np.amin(fit_raw)

    fh = open(f"iso.{pname}.dat","w")
    fh.write("# %s\n"%(str(bestdfcn)))
    for i in range(npts):
        fh.write("%12.3f %20.10e %20.10e %20.10e\n"%\
                 ( angs[i], hl_plot[i], ll_plot[i], fit_plot[i] ) )
    fh.close()
    return bestdfcn



def _fitted_dihed_idxs(system):
    """All 0-based dihedral index tuples currently being fit for ``system``."""
    idxs = []
    for pinst in system.pinstances:
        for di in pinst.dihedidxs:
            idxs.append(list(di))
    return idxs


def _analytical_fitted_torsion_kcal(system, ang_tables_for_geom):
    """Sum Amber-style torsion energies (kcal/mol) for one geometry.

    ``ang_tables_for_geom[ipinst][idihed]`` are dihedral angles in degrees,
    precomputed at fixed LL coordinates. Force constants come from each
    ``ParamType.dfcns`` (updated by :meth:`FitInputType.set_params`).
    """
    e = 0.0
    for ipinst, pinst in enumerate(system.pinstances):
        dfcns = pinst.ptype.dfcns
        if dfcns is None:
            continue
        for iang, _idxs in enumerate(pinst.dihedidxs):
            e += float(dfcns.CptEne(ang_tables_for_geom[ipinst][iang]))
    return e


def joint_design_matrix_from_caches(finp, caches, kcal_per_ev):
    """Build mean-centered design matrix / target for the fixed-geometry NL model.

    Columns follow :meth:`FitInputType.get_params` order. Each profile is
    mean-centered independently so chi^2 matches :func:`shape_match_delta`.
    """
    import numpy as np

    pname_to_offset = {}
    ipar = 0
    for pname in finp.ptypedict:
        pname_to_offset[pname] = ipar
        ipar += finp.ptypedict[pname].nprim
    nparam = ipar
    if nparam == 0:
        raise ValueError("joint design: no parameters")

    blocks_A = []
    blocks_y = []
    for isys, s in enumerate(finp.systems):
        sys_cache = caches[isys]
        for iprof, prof in enumerate(s.profiles):
            base = np.asarray(
                sys_cache["profiles"][iprof]["base_kcal"], dtype=float
            )
            angs = sys_cache["profiles"][iprof]["angles"]
            hl = np.array(
                [
                    float(struct.data["energy"]) * kcal_per_ev
                    for struct in prof.loshl
                ],
                dtype=float,
            )
            ngeom = len(base)
            if ngeom == 0:
                continue
            if len(hl) != ngeom:
                raise ValueError(
                    f"HL/LL length mismatch in profile {prof.name}: "
                    f"{len(hl)} vs {ngeom}"
                )
            A_prof = np.zeros((ngeom, nparam), dtype=float)
            for igeom in range(ngeom):
                for ipinst, pinst in enumerate(s.pinstances):
                    dfcns = pinst.ptype.dfcns
                    if dfcns is None:
                        continue
                    off = pname_to_offset[pinst.ptype.name]
                    for idihed, _idxs in enumerate(pinst.dihedidxs):
                        ang = angs[igeom][ipinst][idihed]
                        for iprim, prim in enumerate(dfcns.prims):
                            A_prof[igeom, off + iprim] += float(
                                prim.CptEterm(ang)
                            )
            y_prof = hl - base
            y_c = y_prof - np.mean(y_prof)
            A_c = A_prof - np.mean(A_prof, axis=0, keepdims=True)
            blocks_A.append(A_c)
            blocks_y.append(y_c)

    if not blocks_A:
        raise ValueError("joint design: no usable profile geometries")
    return np.vstack(blocks_A), np.concatenate(blocks_y), nparam


def joint_linear_solve_from_caches(finp, caches):
    """Joint ``lstsq`` over all fitted FCs using fixed-geometry caches."""
    import numpy as np
    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

    kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    A, y, nparam = joint_design_matrix_from_caches(finp, caches, kcal_per_ev)
    npts = A.shape[0]
    if npts < nparam + 1:
        raise ValueError(
            f"joint LS: need >= {nparam + 1} points, have {npts}"
        )
    x, residuals, rank, singular = np.linalg.lstsq(A, y, rcond=None)
    cond = (
        float(singular[0] / singular[-1])
        if len(singular) and singular[-1] > 0
        else float("inf")
    )
    info = {
        "npts": npts,
        "nparam": nparam,
        "rank": int(rank),
        "singular": singular,
        "cond": cond,
        "residuals": residuals,
    }
    return np.asarray(x, dtype=float), info


def build_fixed_geometry_ll_cache(system, args):
    """One-time LL base energies (fitted torsions deleted) + dihedral angles.

    When only torsion force constants change, the non-fitted MM energy at a
    fixed geometry is constant. Cache that base once (single-point with the
    fitted dihedrals deleted), then each NL iteration adds analytical torsion
    terms - avoiding parm7 rewrite + ``GeomOpt`` per geometry per step.
    """
    import os
    import numpy as np
    from tempfile import mkstemp
    from ffpopt.AmberParm import CopyParm
    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

    kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()

    fitted = _fitted_dihed_idxs(system)
    p = CopyParm(system.mol)
    if fitted:
        DeleteDihedrals(p, fitted)

    fd, path = mkstemp(dir=".", prefix="tmp.", suffix=".parm7")
    if not os.isatty(fd):
        os.close(fd)
    p.save(path, overwrite=True, format="amber")

    profiles = []
    try:
        for prof in system.profiles:
            prof.losll.SetArgs(args)
            prof.losll.calc = None
            base_kcal = []
            angles = []
            for struct in prof.losll.structs:
                t = struct.copy()
                t.data["parm"] = path
                t.constraints = None
                t.restraints = None
                t.data["constraints"] = []
                t.data["restraints"] = []
                g = t.GetASEAtoms()
                g.calc = prof.losll.BuildCalc(t)
                e_ev = float(g.get_potential_energy())
                base_kcal.append(e_ev * kcal_per_ev)

                atoms = struct.GetASEAtoms()
                geom_angs = []
                for pinst in system.pinstances:
                    geom_angs.append(
                        [float(atoms.get_dihedral(*idxs)) for idxs in pinst.dihedidxs]
                    )
                angles.append(geom_angs)

            profiles.append(
                {
                    "base_kcal": np.asarray(base_kcal, dtype=float),
                    "angles": angles,
                }
            )
    finally:
        if os.path.exists(path):
            os.remove(path)
        for prof in system.profiles:
            prof.losll.calc = None

    return {"profiles": profiles}


def ll_energies_kcal_from_cache(system, sys_cache):
    """Fixed-geometry LL energies (kcal): base + analytical fitted torsions."""
    import numpy as np

    out = []
    for iprof, prof_cache in enumerate(sys_cache["profiles"]):
        base = prof_cache["base_kcal"]
        angs = prof_cache["angles"]
        ll = np.empty(len(base), dtype=float)
        for igeom in range(len(base)):
            ll[igeom] = base[igeom] + _analytical_fitted_torsion_kcal(
                system, angs[igeom]
            )
        out.append(ll)
    return out


def use_dihed_fit_reopt() -> bool:
    """True when ``FFPOPT_DIHED_FIT_REOPT=1`` restores legacy GeomOpt-per-iter."""
    from ffpopt.runtime.EnvDefaults import env_bool

    return env_bool("FFPOPT_DIHED_FIT_REOPT")


def DihedFitObjFcn(x,self):
    """ Objective function for fitting dihedral parameters.
    
    This function computes the chi-squared value based on the dihedral parameters
    and the energies of the low-energy and high-energy scans. It updates the positions
    of the low-energy geometries based on the computed energies and writes the results
    to files for each profile in the system.
    
    Parameters
    ----------
    x : numpy.ndarray
        A numpy array containing the dihedral function coefficients to be optimized.
    self : FitInputType
        An instance of FitInputType containing the systems and profiles for fitting.
    Returns
    ------- 
    float
        The chi-squared value representing the difference between the computed and expected energies.
    
    """
    import numpy as np
    from ffpopt.constants import AU_PER_KCAL_PER_MOL
    from ffpopt.constants import AU_PER_ELECTRON_VOLT
    import os
    
    KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
  
    chisq = 0
    it = self.iteration
    self.set_params(x)

    if use_dihed_fit_reopt():
        return _DihedFitObjFcn_reopt(x, self, KCAL_PER_EV)

    if getattr(self, "_ll_cache", None) is None:
        self._ll_cache = []
        for s in self.systems:
            # args live on each profile's los after NonlinearSolve SetArgs
            args = s.profiles[0].losll.args if s.profiles else None
            self._ll_cache.append(build_fixed_geometry_ll_cache(s, args))

    for isys, s in enumerate(self.systems):
        sys_cache = self._ll_cache[isys]
        ll_by_prof = ll_energies_kcal_from_cache(s, sys_cache)

        for iprof, prof in enumerate(s.profiles):
            llene = np.asarray(ll_by_prof[iprof], dtype=float)
            hlene = np.array(
                [struct.data["energy"] * KCAL_PER_EV for struct in prof.loshl],
                dtype=float,
            )

            d = shape_match_delta(hlene, llene)
            mychisq = float(np.dot(d, d))
            chisq += mychisq

            # Display-only min shifts for mfit.*.dat plots.
            hl_plot = hlene - np.amin(hlene)
            ll_plot = llene - np.amin(llene)

            if prof.name is None or prof.plots is None:
                continue
            elif len(prof.plots) == 0:
                continue

            ang_tables = sys_cache["profiles"][iprof]["angles"]
            for pname in prof.plots:
                pinst = s.find_pinstance(pname)
                if pinst is None:
                    continue
                try:
                    ipinst = s.pinstances.index(pinst)
                except ValueError:
                    ipinst = next(
                        (i for i, c in enumerate(s.pinstances)
                         if c.ptype.name == pinst.ptype.name),
                        None,
                    )
                if ipinst is None:
                    continue

                for idihed, idxs in enumerate(pinst.dihedidxs):
                    angs = [ang_tables[igeom][ipinst][idihed]
                            for igeom in range(len(ang_tables))]
                    data = []
                    for i in range(len(angs)):
                        data.append([angs[i], hl_plot[i], ll_plot[i]])
                    data = sorted(data, key=lambda row: row[0])

                    idxsname = "-".join([f"{i}" for i in idxs])
                    fname = f"mfit.{prof.name}.{idxsname}.{it:04d}.dat"
                    with open(fname, "w") as fh:
                        fh.write("# %25.14f\n" % (mychisq))
                        for row in data:
                            fh.write("%20.10e %20.10e %20.10e\n" % (row[0], row[1], row[2]))

    self.iteration += 1
    return chisq


def _DihedFitObjFcn_reopt(x, self, KCAL_PER_EV):
    """Legacy NL objective: rewrite parm7 and GeomOpt every geometry."""
    import numpy as np
    from ffpopt.geom.GeomOpt import GeomOpt
    from tempfile import mkstemp
    import os

    chisq = 0
    it = self.iteration

    for isys, s in enumerate(self.systems):
        p = s.make_new_parm()

        fd, path = mkstemp(dir=".", prefix="tmp.", suffix=".parm7")
        if not os.isatty(fd):
            os.close(fd)

        p.save(path, overwrite=True, format="amber")

        for iprof, prof in enumerate(s.profiles):
            llene = []
            hlene = []
            prof.losll.calc = None

            for igeom in range(len(prof.losll)):
                inpstruct = prof.losll.structs[igeom].copy()
                inpstruct.data["parm"] = path
                sout = GeomOpt(prof.losll, inpstruct)
                hlene.append(prof.loshl.structs[igeom].data["energy"] * KCAL_PER_EV)
                llene.append(sout.data["energy"] * KCAL_PER_EV)
                prof.losll.structs[igeom].Update(
                    sout.data["energy"],
                    sout.data["positions"],
                    sout.data.get("forces"),
                )

            llene = np.array(llene)
            hlene = np.array(hlene)

            d = shape_match_delta(hlene, llene)
            mychisq = float(np.dot(d, d))

            chisq += mychisq

            hl_plot = hlene - np.amin(hlene)
            ll_plot = llene - np.amin(llene)

            if prof.name is None or prof.plots is None:
                continue
            elif len(prof.plots) == 0:
                continue

            for pname in prof.plots:
                pinst = s.find_pinstance(pname)
                if pinst is None:
                    continue

                for idxs in pinst.dihedidxs:
                    angs = []
                    for struct in prof.losll:
                        atoms = struct.GetASEAtoms()
                        ang = atoms.get_dihedral(*idxs)
                        angs.append(ang)
                    data = []
                    for i in range(len(angs)):
                        data.append([angs[i], hl_plot[i], ll_plot[i]])
                    data = sorted(data, key=lambda row: row[0])

                    idxsname = "-".join([f"{i}" for i in idxs])
                    fname = f"mfit.{prof.name}.{idxsname}.{it:04d}.dat"
                    with open(fname, "w") as fh:
                        fh.write("# %25.14f\n" % (mychisq))
                        for row in data:
                            fh.write("%20.10e %20.10e %20.10e\n" % (row[0], row[1], row[2]))
        if os.path.exists(path):
            os.remove(path)

    self.iteration += 1
    return chisq


def NonlinearSolve(args,finp):
    """ Perform a nonlinear optimization to fit dihedral parameters.
    
    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments containing the optimization parameters.
    finp : FitInputType
        An instance of FitInputType containing the systems and profiles for fitting.

    Returns
    -------
    None
        The function modifies the FitInputType instance in place, setting the optimized dihedral parameters.
    
    """
    from scipy.optimize import minimize, lsq_linear
    import numpy as np
    from ffpopt.dihed.ExtendedFit import (
        configure_fit_input,
        count_extended_params,
        enrich_cache_with_14,
        get_extended_params,
        set_extended_params,
        solve_extended_lbfgsb,
    )

    for s in finp.systems:
        for p in s.profiles:
            p.losll.SetArgs(args)

    configure_fit_input(finp, args)

    finp._ll_cache = None
    reopt = use_dihed_fit_reopt()
    if not reopt:
        finp._ll_cache = [
            build_fixed_geometry_ll_cache(s, args) for s in finp.systems
        ]
        if getattr(finp, "opt_scee_scnb", False):
            from ffpopt.affdo.AffdoLog import print_affdo

            print_affdo(
                "stripping scaled 1-4 elec/vdw from LL cache; "
                f"refitting scee/scnb (start scee={float(getattr(finp, 'scee', 1.2)):g} "
                f"scnb={float(getattr(finp, 'scnb', 2.0)):g})"
            )
            for isys, s in enumerate(finp.systems):
                enrich_cache_with_14(
                    s,
                    finp._ll_cache[isys],
                    scee0=float(getattr(finp, "scee", 1.2)),
                    scnb0=float(getattr(finp, "scnb", 2.0)),
                )

    x = finp.make_initial_guesses(args=args, caches=finp._ll_cache)
    n = finp.get_num_params()
    if n == 0:
        print("[ffpopt] NonlinearSolve: no parameters to fit")
        return

    # Extended AFFDO-style knobs (phase / period / scee*scnb) or explicit backends.
    use_extended = bool(
        getattr(finp, "opt_phase", False)
        or getattr(finp, "opt_periods", False)
        or getattr(finp, "opt_scee_scnb", False)
        or str(getattr(finp, "fit_backend", "lsq")) in ("lbfgsb", "jax")
    )
    if use_extended and not reopt:
        # Seed phases/periods/scee from templates + FC guess.
        set_extended_params(finp, get_extended_params(finp))
        # Ensure FC block matches linear guess when only FCs were solved.
        # get_extended_params already reads current dfcns FCs.
        n_ext = count_extended_params(finp)
        print(
            f"[ffpopt] Extended fit mode={getattr(finp, 'fit_mode', '?')} "
            f"backend={finp.fit_backend} nparam={n_ext} "
            f"opt_phase={finp.opt_phase} opt_periods={finp.opt_periods} "
            f"opt_scee_scnb={finp.opt_scee_scnb}"
        )
        print(
            f"[affdo] GenDihedFit using extended solver "
            f"(mode={getattr(finp, 'fit_mode', '?')} backend={finp.fit_backend} "
            f"nparam={n_ext})"
        )
        solve_extended_lbfgsb(args, finp, finp._ll_cache)
        chisq = DihedFitObjFcn(finp.get_params(), finp)
        print(f"[ffpopt] Final shape-match chi^2 = {chisq:.6e}")
        return

    xlo = x[:] - 2.0
    xhi = x[:] + 5.0

    if not reopt:
        # Fixed-geometry FCs enter linearly - bounded linear least squares.
        from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

        kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
        A, y, nparam = joint_design_matrix_from_caches(
            finp, finp._ll_cache, kcal_per_ev
        )
        print(
            f"[ffpopt] Fixed-geom FC solve via lsq_linear "
            f"(npts={A.shape[0]}, nparam={nparam})"
        )
        res = lsq_linear(A, y, bounds=(xlo, xhi), method="bvls")
        print(
            f"[ffpopt] lsq_linear: success={bool(res.success)} "
            f"status={res.status} cost={float(res.cost):.6e} "
            f"nit={int(res.nit)} msg={res.message}"
        )
        finp.set_params(res.x)
        # One objective evaluation for mfit.*.dat plots / chi^2 report.
        chisq = DihedFitObjFcn(res.x, finp)
        print(f"[ffpopt] Final shape-match chi^2 = {chisq:.6e}")
        return

    bounds = [(lo, hi) for lo, hi in zip(xlo, xhi)]
    res = minimize(
        DihedFitObjFcn,
        x,
        args=(finp,),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": args.nltol,
            "maxiter": args.nlmaxiter,
            "disp": True,
        },
    )

    print(
        f"[ffpopt] L-BFGS-B: success={bool(res.success)} "
        f"nit={getattr(res, 'nit', '?')} "
        f"fun={float(res.fun):.6e} msg={res.message}"
    )
    finp.set_params(res.x)
