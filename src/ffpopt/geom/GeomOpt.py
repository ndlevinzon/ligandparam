#!/usr/bin/env python3


from ffpopt.geom.GeomOptAse import (  # noqa: F401
    GeomOpt_ASE,
    _ase_loose_fmax,
    _struct_bonds_numbers,
    _guard_covalent_geometry,
    _explode_fmax_limit,
    merge_struct_constraints_restraints,
)
from ffpopt.geom.Geometric import (  # noqa: F401
    _linux_process_tree_cputime,
    _path_tree_mtime,
    _geometric_stall_timeout_sec,
    _run_geometric_with_watchdog,
)

def opt_recovery_label(struct) -> str | None:
    """Return the GeomOpt recovery tag on ``struct``, if any."""
    if struct is None:
        return None
    data = getattr(struct, "data", None) or {}
    for key in ("geometric_recovery", "ase_opt_recovery"):
        val = data.get(key)
        if val:
            return str(val)
    return None


def is_soft_opt_recovery(struct_or_label) -> bool:
    """True for soft-accept / loose recoveries that should not drive hard spawn.

    Soft tags: ``soft-maxiter`` (geomeTRIC), ``linear-torsion*`` (near-collinear
    dihedral rescue), and ASE labels ending in ``-soft``. Loose-but-converged
    attempts (``loose``, ``dlc-loose``, ``hdlc-loose``, ...) are also treated as
    soft for wavefront spawn policy.
    """
    if struct_or_label is None:
        return False
    if hasattr(struct_or_label, "data"):
        label = opt_recovery_label(struct_or_label)
    else:
        label = str(struct_or_label)
    if not label:
        return False
    low = label.lower()
    if low == "soft-maxiter" or low.endswith("-soft"):
        return True
    if low == "loose" or low.endswith("-loose"):
        return True
    if low.startswith("linear-torsion"):
        return True
    return False



def GeomOpt_GEOMETRIC(
    los, struct, constraints=None, restraints=None, *, geom_prefix=None
):
    """ Perform a geometry optimization using the GEOMETRIC program.

    By default runs geomeTRIC **in-process** with a persistent ASE calculator
    cached on ``los`` (avoids spawning ``python -m ffpopt.geom.Geometric``
    per call). Set ``FFPOPT_GEOMETRIC_SUBPROCESS=1`` to restore the legacy
    subprocess + watchdog path.

    When ``geom_prefix`` is set (e.g. beside a wavefront node pickle), geomeTRIC
    writes ``{prefix}_optim.xyz`` under that basename. An interrupted opt can
    warm-start from the last trajectory frame on restart (same hook the
    recovery ladder already uses). Random ``./tmpfiles/tmp.*`` prefixes cannot
    be rediscovered after a kill.

    Parameters
    ----------
    los : ffpopt.Struct.ListOfStruct
        The input list of structures. The structures are not directly used or manipulated.
        Instead the CLI arguments are used.
    struct : ffpopt.Struct.Struct
        The initial geometry to optimize.
    constraints : list of Constraint, optional
        A list of constraints to apply during the optimization. Default is None (no constraints).
    restraints : list of Restraint, optional
        A list of restraints to apply during the optimization. Default is None (no constraints)
    geom_prefix : path-like, optional
        Stable basename for geomeTRIC outputs (enables warm restart).

    Returns
    -------
    ffpopt.Struct.Struct
        The optimized geometry with updated positions, forces, and energy
    """
    import os
    from ffpopt.runtime.CpuThreads import pin_math_threads

    pin_math_threads(1)

    import copy
    import ase.io
    from ffpopt.constants import AU_PER_ELECTRON_VOLT
    from ffpopt.Options import argparse2geometric, configure_geometric_logging, GetStandardOptions
    from ffpopt.geom.Constraints import ConstraintList
    from ffpopt.geom.Restraints import RestraintList
    from ffpopt.geom.Constraints import ApplyConstraints, to_geometric
    from ffpopt.Struct import ListOfStruct
    from ffpopt.geom.Geometric import (
        cleanup_geometric_scratch,
        get_persistent_calc,
        read_last_optim_xyz,
        run_geometric_robust,
        use_geometric_subprocess,
    )
    from ffpopt.geom.LinearTorsion import has_near_linear_dihedral_bend
    from tempfile import mkstemp
    from pathlib import Path

    # Build constrained geometry first so we can divert to the linear-torsion
    # rescue *before* allocating geomeTRIC temp files.
    conslist, reslist = merge_struct_constraints_restraints(
        struct, constraints, restraints
    )

    origatoms = struct.GetASEAtoms()
    myatoms = copy.deepcopy(origatoms)

    stable_prefix = geom_prefix is not None and str(geom_prefix).strip()
    if stable_prefix:
        tmpbase_early = str(Path(geom_prefix))
        last = read_last_optim_xyz(tmpbase_early)
        if last is not None and last.shape == myatoms.get_positions().shape:
            myatoms.set_positions(last)
            print(f"[ffpopt] warm-start geomopt from {tmpbase_early}_optim.xyz")

    cons = None
    target_cons = None
    if conslist is not None:
        cons = conslist.FillConstraints(myatoms, force=False)
        target_cons = copy.deepcopy(cons)
        origcons = conslist.FillConstraints(myatoms, force=True)
        myatoms = ApplyConstraints(myatoms, cons, graph=struct.GetGraph())

    if target_cons is not None and has_near_linear_dihedral_bend(myatoms, target_cons):
        from ffpopt.geom.LinearTorsion import log_linear_torsion

        log_linear_torsion(
            "[ffpopt] near-linear bend in constrained torsion; "
            "using linear-torsion geomopt"
        )
        return GeomOpt_LINEAR_TORSION(los, struct, constraints, restraints)

    if stable_prefix:
        tmpbase = str(Path(geom_prefix))
        Path(tmpbase).parent.mkdir(parents=True, exist_ok=True)
        tmpxyz = tmpbase + ".xyz"
        tmpopt = tmpbase + "_optim.xyz"
        tmplog = tmpbase + ".log"
        tmpcons = tmpbase + ".cons.inp"
        tmpdir = tmpbase + ".tmp"
        tmpjson = tmpbase + ".json"
    else:
        tmpfile_loc = "./tmpfiles"
        if not Path(tmpfile_loc).is_dir():
            os.makedirs(tmpfile_loc, exist_ok=True)

        fd, tmpxyz = mkstemp(dir=tmpfile_loc, prefix="tmp.", suffix=".xyz")
        if not os.isatty(fd):  # Check if fd is still valid
            os.close(fd)

        tmpbase = str(Path(tmpxyz).with_suffix(""))
        tmpopt = tmpbase + "_optim.xyz"
        tmplog = tmpbase + ".log"
        tmpcons = tmpbase + ".cons.inp"
        tmpdir = tmpbase + ".tmp"
        tmpjson = tmpbase + ".json"

    if conslist is not None:
        with open(tmpcons, "w") as fh:
            fh.write("$set\n")
            # Write *target* constraints, not the force=True snapshot above.
            for line in to_geometric(target_cons):
                fh.write("%s\n" % (line))

    result = None
    try:
        if use_geometric_subprocess():
            # Legacy path: spawn geom.geometric + watchdog.
            ase.io.write(tmpxyz, myatoms, format="xyz", parallel=False)
            mystruct = copy.deepcopy(struct)
            if conslist is not None:
                mystruct.constraints = None
                mystruct.data["constraints"] = []
            if reslist is not None:
                mystruct.restraints = reslist
                mystruct.data["restraints"] = reslist.to_list_of_dict()
            mylos = ListOfStruct([mystruct])
            mylos.save(tmpjson)
            cmds = argparse2geometric(tmpjson, los.args)
            cmds.append(tmpxyz)
            if conslist is not None:
                cmds.append(tmpcons)
            _run_geometric_with_watchdog(cmds, tmplog, activity_dir=tmpdir)
            if not os.path.exists(tmpopt):
                log_hint = ""
                try:
                    if os.path.exists(tmplog):
                        with open(tmplog, "r", errors="replace") as fh:
                            tail = fh.read()[-2000:]
                        if tail.strip():
                            log_hint = f"\n--- tail of {tmplog} ---\n{tail}"
                except OSError:
                    pass
                raise Exception(
                    f"File not found: {tmpopt} (geomeTRIC did not write an "
                    f"optimized geometry; often a constrained-IC recovery failure)."
                    f"{log_hint}"
                )
            out_atoms = ase.io.read(tmpopt, index="-1", parallel=False)
            out_atoms.set_initial_charges(myatoms.get_initial_charges())
            keys = [key for key in out_atoms.info]
            ene = float(keys[-1]) / AU_PER_ELECTRON_VOLT()
            crd = out_atoms.get_positions()
            frc = None
        else:
            calc = get_persistent_calc(los, struct, reslist=reslist)
            geo = GetStandardOptions(los.args).get("geometric", {})
            log_ini = configure_geometric_logging(
                getattr(los.args, "geometric_ini", None)
            )
            if getattr(los.args, "geometric_ini", None) is not None:
                if len(str(los.args.geometric_ini)) == 0:
                    log_ini = ""
            bonds, numbers, _ = _struct_bonds_numbers(struct)
            result = run_geometric_robust(
                myatoms,
                calc,
                prefix=tmpbase,
                constraints_path=tmpcons if conslist is not None else None,
                coordsys=geo.get("coordsys", "tric"),
                maxiter=int(geo.get("maxiter", getattr(los.args, "geometric_maxiter", 500))),
                converge=geo.get("converge", getattr(los.args, "geometric_converge", "set GAU")),
                enforce=float(geo.get("enforce", getattr(los.args, "geometric_enforce", 0.0))),
                log_ini=log_ini if log_ini else None,
                geometry_bonds=bonds,
                geometry_numbers=numbers,
            )
            crd = result["coords"]
            if result["energy_ha"] is not None:
                ene = result["energy_ha"] / AU_PER_ELECTRON_VOLT()
            else:
                myatoms.set_positions(crd)
                myatoms.calc = calc
                ene = myatoms.get_potential_energy()
            frc = None
    except BaseException:
        cleanup_geometric_scratch(tmpbase, keep_optim=bool(stable_prefix))
        raise

    clone = getattr(struct, "clone_geometry", None)
    if callable(clone):
        out = clone(coords=crd, ene=ene, frcs=frc)
    else:
        out = copy.deepcopy(struct)
        out.Update(ene, crd, frc)
    if isinstance(result, dict) and result.get("recovery"):
        out.data["geometric_recovery"] = result["recovery"]

    from ffpopt.runtime.FastWavefront import geomopt_verbose

    if geomopt_verbose():
        from ffpopt.geom.Constraints import FillConstraints
        if cons is not None:
            cvals = FillConstraints(out,cons,force=True)
            ovals = FillConstraints(origatoms,cons,force=True)
            for i in range(len(cvals)):
                print("Constraint %2i tgt=%9.2f opt=%9.2f orig=%9.2f"%\
                      ( i+1, cons[i].value, cvals[i].value, ovals[i].value ) )

        if reslist is not None:
            ocrd = origatoms.get_positions()
            for i in range(len(reslist)):
                val = reslist[i].GetCrdValue(crd)
                oval = reslist[i].GetCrdValue(ocrd)
                print("Restraint  %2i tgt=%9.2f obs=%9.2f orig=%9.2f"%\
                      ( i+1, reslist[i].value, val, oval ) )

    if cons is not None:
        out.constraints = ConstraintList( cons )
        out.data["constraints"] = out.constraints.to_list_of_dict()

    if reslist is not None:
        out.restraints = reslist
        out.data["restraints"] = out.restraints.to_list_of_dict()

    cleanup_geometric_scratch(tmpbase, keep_optim=False)
    return out



def bare_potential_energy(struct):
    """Return optimized energy excluding restraint penalty terms (eV).

    Wavefront ranking wants the bare potential at the constrained/restrained
    geometry. :func:`GeomOpt` already evaluates energy at the final point; when
    restraints were active that value includes classical penalty terms, which
    can be removed analytically without a second SCF.
    """
    if struct is None or "energy" not in struct.data or struct.data["energy"] is None:
        raise ValueError("Missing energy from optimization output.")
    ene = float(struct.data["energy"])
    rests = getattr(struct, "restraints", None)
    if rests is None or len(rests) == 0:
        return ene
    import numpy as np

    crds = np.asarray(struct.data["positions"], dtype=float)
    for rst in rests:
        e2, _ = rst.GetValueAndGradients(crds)
        ene -= float(e2)
    return ene


def GeomOpt_SinglePoint(los,struct,constraints=None,restraints=None):
    """ Perform a single point energy calculation without geometry optimization.
    
    Parameters
    ----------
    los : ffpopt.Struct.ListOfStruct
        The input list of structures. The structures are not directly used or manipulated.
        Instead the CLI arguments are used.
    struct : ffpopt.Struct.Struct
        The initial geometry to optimize.
    constraints : list of Constraint, optional
        A list of constraints to apply during the optimization. Default is None (no constraints).
    restraints : list of Restraint, optional
        A list of restraints to apply during the optimization. Default is None (no restraints).

    Returns
    -------
    ffpopt.Struct.Struct
        The optimized geometry with updated positions, forces, and energy
    """
    import os
    import copy
    #import subprocess as subp
    #import ase.io
    #from ffpopt.constants import AU_PER_ELECTRON_VOLT
    #from ffpopt.Options import argparse2geometric
    from ffpopt.geom.Constraints import ConstraintList
    from ffpopt.geom.Constraints import to_ase,ApplyConstraints
    from ffpopt.geom.Restraints import RestraintList
    #from ffpopt.geom.Constraints import constraints2info,constraints2ase

    if True:
        # if stdargs.calc is None:
        #     calc = stdargs.MakeCalc()
        # else:
        #     calc = stdargs.calc
        # out = atoms.copy()
        # del out.constraints
        # out.calc = calc
        # out.calc.reset()
        # out.info = {"energy": out.get_potential_energy(), "forces": out.get_forces()}
        # out.calc = None



        conslist, reslist = merge_struct_constraints_restraints(
            struct, constraints, restraints
        )

        myatoms = struct.GetASEAtoms()
        myatoms.calc=los.BuildRestrainedCalc(struct,reslist=reslist)
        calc = myatoms.calc

        asecons = None
        cons = None
        if conslist is not None:
            cons = conslist.FillConstraints(myatoms)
            myatoms = ApplyConstraints(myatoms,cons,graph=struct.GetGraph())
            #asecons = constraints2ase(cons)
            asecons = to_ase(cons)


        # asecons = None
        # cons = None
        # if conslist is not None:
        #     cons = conslist.FillConstraints(myatoms,force=False)
        #     origcons = conslist.FillConstraints(myatoms,force=True)
        # if cons is not None or reslist is not None:
        #     myatoms = ApplyConstraints(myatoms,cons,graph=struct.GetGraph(),rests=reslist.rests)
        #     if cons is not None:
        #         asecons = to_ase(cons)

        #         newcons = conslist.FillConstraints(myatoms,force=True)
        #         for ic in range(len(cons)):
        #             print(ic,str(origcons[ic]),str(newcons[ic]),cons[ic].value)


            
    
        del myatoms.constraints
        myatoms.set_constraint( asecons )
        myatoms.calc = calc
        myatoms.calc.reset()

        ene = myatoms.get_potential_energy()
        crd = myatoms.get_positions()
        frc = myatoms.get_forces()
        out = copy.deepcopy(struct)
        out.Update(ene,crd,frc)
        
        if cons is not None:
            out.constraints = ConstraintList( cons )
            out.data["constraints"] = out.constraints.to_list_of_dict()
        if reslist is not None:
            out.restraints = reslist
            out.data["restraints"] = out.restraints.to_list_of_dict()
            
    return out



def GeomOpt_LINEAR_TORSION(los, struct, constraints=None, restraints=None):
    """Geometry opt for constrained dihedrals with near-linear valence bends.

    geomeTRIC cannot define a torsion when A-B-C or B-C-D is ~180 deg. This path:

    1. Detects near-linear bends in frozen dihedrals (>=175 deg).
    2. Unkinks them to ~170 deg so the torsion is well-defined.
    3. Re-applies the target dihedral(s).
    4. Optimizes with ASE ``FixInternals`` holding both the dihedral(s) and the
       unkinked bend angle(s).

    Tagged ``linear-torsion`` / ``linear-torsion-soft`` (soft for wavefront
    spawn policy - the geometry is slightly biased off-linear by construction).
    """
    import copy

    from ffpopt.geom.Constraints import ApplyConstraints
    from ffpopt.geom.LinearTorsion import (
        find_near_linear_bends,
        log_linear_torsion,
        run_linear_torsion_ase_opt,
        unkink_near_linear_bends,
    )
    from ffpopt.geom.Geometric import get_persistent_calc

    conslist, reslist = merge_struct_constraints_restraints(
        struct, constraints, restraints
    )

    myatoms = struct.GetASEAtoms()
    cons = None
    if conslist is not None:
        cons = conslist.FillConstraints(myatoms, force=False)
        myatoms = ApplyConstraints(myatoms, cons, graph=struct.GetGraph())

    bends = find_near_linear_bends(myatoms, cons)
    if not bends and cons is not None:
        # Caller may have already unkinked, or LinearTorsionError fired mid-opt
        # after drifting linear - force-scan dihedral bends anyway at a lower
        # threshold so we still attempt a rescue.
        bends = find_near_linear_bends(myatoms, cons, threshold_deg=165.0)

    angle_hold = []
    if bends:
        angle_hold = unkink_near_linear_bends(myatoms, bends)
        # Re-apply dihedral targets after the bend nudge.
        if cons is not None:
            myatoms = ApplyConstraints(myatoms, cons, graph=struct.GetGraph())
    else:
        log_linear_torsion(
            "[ffpopt] linear-torsion rescue: no near-linear bends found at "
            "entry; still attempting ASE FixInternals on dihedrals"
        )

    dihed_cons = [c for c in (cons or []) if len(getattr(c, "idxs", ())) == 4]
    if not dihed_cons:
        raise RuntimeError("linear-torsion rescue invoked without dihedral constraints")

    calc = get_persistent_calc(los, struct, reslist=reslist)
    strict_tol = float(getattr(los.args, "ase_opt_tol", 0.01))
    loose_tol = _ase_loose_fmax(strict_tol)
    max_steps = int(getattr(los.args, "geometric_maxiter", 500) or 500)

    opt_atoms, recovery = run_linear_torsion_ase_opt(
        myatoms,
        calc,
        dihed_cons=dihed_cons,
        angle_cons=angle_hold,
        fmax=strict_tol,
        loose_fmax=loose_tol,
        max_steps=max_steps,
    )

    ene = opt_atoms.get_potential_energy()
    crd = opt_atoms.get_positions()
    frc = opt_atoms.get_forces()
    clone = getattr(struct, "clone_geometry", None)
    if callable(clone):
        out = clone(coords=crd, ene=ene, frcs=frc)
    else:
        out = copy.deepcopy(struct)
        out.Update(ene, crd, frc)
    out.data["ase_opt_recovery"] = recovery
    out.data["geometric_recovery"] = recovery
    if cons is not None:
        from ffpopt.geom.Constraints import ConstraintList as _CL

        out.constraints = _CL(cons)
        out.data["constraints"] = out.constraints.to_list_of_dict()
    if reslist is not None:
        out.restraints = reslist
        out.data["restraints"] = out.restraints.to_list_of_dict()
    log_linear_torsion(f"[ffpopt] linear-torsion geomopt recovered ({recovery})")
    return out


def GeomOpt(los, struct, constraints=None, restraints=None, *, geom_prefix=None):
    """ Perform a geometry optimization or single point calculation based on stdargs.
    
    Parameters
    ----------
    los : ffpopt.Struct.ListOfStruct
        The input list of structures. The structures are not directly used or manipulated.
        Instead the CLI arguments are used.
    struct : ffpopt.Struct.Struct
        The initial geometry to optimize.
    constraints : list of Constraint, optional
        A list of constraints to apply during the optimization. Default is None (no constraints).
    restraints : list of Restraint, optional
        A list of restraints to apply during the optimization. Default is None (no constraints)
    geom_prefix : path-like, optional
        Stable geomeTRIC output basename (warm-restart after interrupt).

    Returns
    -------
    ffpopt.Struct.Struct
        The optimized geometry with updated positions, forces, and energy
    """
    bonds, numbers, atoms0 = _struct_bonds_numbers(struct)
    if not los.args.no_opt:
        pos0 = atoms0.get_positions() if atoms0 is not None else None
        if pos0 is not None:
            _guard_covalent_geometry(pos0, bonds, numbers, where="pre-opt")

    if los.args.no_opt:
        out = GeomOpt_SinglePoint(los,struct,constraints,restraints)
    else:
        linear = lambda: GeomOpt_LINEAR_TORSION(
            los, struct, constraints, restraints
        )
        ase = lambda: GeomOpt_ASE(los, struct, constraints, restraints)
        geometric = lambda: GeomOpt_GEOMETRIC(
            los, struct, constraints, restraints, geom_prefix=geom_prefix
        )
        if los.args.geometric_opt:
            out = _geomopt_run_with_fallbacks(
                "geomeTRIC",
                geometric,
                secondary_name="ASE",
                secondary_fn=ase,
                linear_fn=linear,
            )
        else:
            from ffpopt.runtime.FastWavefront import fast_wavefront_enabled

            out = _geomopt_run_with_fallbacks(
                "ASE",
                ase,
                secondary_name="geomeTRIC",
                secondary_fn=geometric,
                linear_fn=linear,
                fast_skip_secondary=fast_wavefront_enabled(None),
            )
    if not los.args.no_opt:
        getter = getattr(out, "GetASEAtoms", None)
        atoms1 = getter() if callable(getter) else None
        if atoms1 is not None:
            _guard_covalent_geometry(
                atoms1.get_positions(), bonds, numbers, where="post-opt"
            )
    return out


def _geomopt_run_with_fallbacks(
    primary_name,
    primary_fn,
    *,
    linear_fn,
    secondary_name=None,
    secondary_fn=None,
    fast_skip_secondary=False,
):
    """Try primary geomopt, then linear-torsion and/or a secondary backend.

    ``BrokenGeometryError`` always propagates. A linear-torsion shaped error
    skips the secondary backend. Under ``fast_skip_secondary``, a generic
    primary failure tries linear-torsion only and re-raises the original
    exception if that rescue also fails.
    """
    from ffpopt.geom.Constraints import BrokenGeometryError
    from ffpopt.geom.LinearTorsion import is_linear_torsion_error

    try:
        return primary_fn()
    except BrokenGeometryError:
        raise
    except Exception as exc:
        if is_linear_torsion_error(exc):
            _geomopt_fallback_note(primary_name, exc, "linear-torsion")
            return linear_fn()
        if fast_skip_secondary:
            try:
                _geomopt_fallback_note(primary_name, exc, "linear-torsion")
                return linear_fn()
            except BrokenGeometryError:
                raise
            except Exception:
                raise exc from exc
        if secondary_fn is None:
            raise
        _geomopt_fallback_note(primary_name, exc, secondary_name)
        try:
            return secondary_fn()
        except BrokenGeometryError:
            raise
        except Exception as exc2:
            if is_linear_torsion_error(exc2):
                _geomopt_fallback_note(secondary_name, exc2, "linear-torsion")
                return linear_fn()
            raise


def _geomopt_fallback_note(failed: str, exc: Exception, fallback: str) -> None:
    """One-line stderr note; full traceback only if FFPOPT_GEOMOPT_TRACEBACK=1."""
    import sys
    import traceback

    from ffpopt.runtime.Console import ascii_for_stdio
    from ffpopt.runtime.EnvDefaults import env_bool

    sys.stderr.write(
        ascii_for_stdio(
            f"[ffpopt] {failed} geomopt failed ({type(exc).__name__}: {exc}); "
            f"falling back to {fallback}\n"
        )
    )
    if env_bool("FFPOPT_GEOMOPT_TRACEBACK"):
        traceback.print_exc()





def CheckForces(los,struct,delta=1.e-2):
    """ Compares analytic and numerical forces and writes a report to stdout
    
    Parameters
    ----------
    los : ffpopt.Struct.ListOfStruct
        The input list of structures. The structures are not directly used or manipulated.
        Instead the CLI arguments are used.
    struct : ffpopt.Struct.Struct
        The initial geometry to optimize.
    delta : float, default=1.e-2
        The displacement (Angstrom). This often needs to be larger than what one
        normally uses. Some models read/write from disk and the output files truncate
        the energies.  Other models use low-precision machine learning networks which
        results in numerical noise.
  
    Returns
    -------
    None
    """
    import copy
    import numpy as np
    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

    DEL = delta
    ana = GeomOpt_SinglePoint(los,struct)
    fana = ana.get_forces()
    fnum = np.zeros( fana.shape )
    ts = copy.deepcopy(struct)
    n = len(struct.data["elements"])
    
    for a in range(n):
        for k in range(3):
            ts.data["positions"][a][k] += DEL
            o = GeomOpt_SinglePoint(los,ts)
            ehi = o.get_potential_energy()
            ts.data["positions"][a][k] -= 2*DEL
            o = GeomOpt_SinglePoint(los,ts)
            elo = o.get_potential_energy()
            ts.data["positions"][a][k] += DEL
            fnum[a,k] = - (ehi-elo)/(2*DEL)

    #conv = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    #fana *= conv
    #fnum *= conv
    
    print("%4s %35s  %35s  %35s"%("Atom","Analytic Frc","Numerical Frc","Ana-Num"))
    for a in range(n):
        print("%5i %11.2e %11.2e %11.2e  %11.2e %11.2e %11.2e  %11.2e %11.2e %11.2e"%\
              ( a+1,
                fana[a,0],fana[a,1],fana[a,2],
                fnum[a,0],fnum[a,1],fnum[a,2],
                fana[a,0]-fnum[a,0],fana[a,1]-fnum[a,1],fana[a,2]-fnum[a,2] ))

            




def ApplyDihedConstraint(atoms,idxs,value,rotmask):
    """ Apply a dihedral constraint to an ASE Atoms object.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The geometry to modify.
    idxs : list of int
        A list of four 0-based atom indices defining the dihedral.
    value : float
        The dihedral angle in degrees to set.
    rotmask : list of bool
        A list of booleans indicating which atoms to rotate.
    
    Returns
    -------
    ase.Atoms
        A new ASE Atoms object with the modified dihedral angle.
    
    """
    import copy
    out = atoms.copy()
    out.set_dihedral(idxs[0],idxs[1],idxs[2],idxs[3],value,
                     mask=rotmask)
    return out


    

def DihedScan(los,struct,con,sched):
    """ Perform a dihedral scan using forward and reverse scans to find the minimum energy conformation.
    
    Parameters
    ----------
    los : ffpopt.Struct.ListOfStruct
        Structure list object used to build calculator
    struct : ffpopt.Struct.Struct
        A Struct object (usually los.structs[0])
    con : Constraint
        The dihedral constraint to scan.
    sched : list of float
        A list of dihedral angles in degrees to scan.
    
    Returns
    -------
    ListOfStruct
        A list of Structs corresponding to the scanned geometries. The Struct names
        are angXXX, where XXX is a zero-padded integer of the dihedral angle in degrees.
    """
    import numpy as np
    import copy
    from ffpopt.Struct import ListOfStruct
    
    idxs = copy.deepcopy( con.idxs )

    mingeom = GeomOpt(los,struct)

    sched = np.array(sched,copy=True)
    minang = mingeom.get_dihedral(idxs[0],idxs[1],idxs[2],idxs[3])

    difangs = [ abs((x-minang)%360) for x in sched ]
    istart = np.argmin(difangs)
    nscan = len(sched)
    scan = []

    for i in range(nscan+1):
        icur = (istart + i) % nscan
        if icur == nscan:
            value = sched[0]
        else:
            value = sched[icur]

        if i == 0:
            igeom = mingeom.copy()
        else:
            igeom = opt.copy()

        fang = igeom.get_dihedral(idxs[0],idxs[1],idxs[2],idxs[3])
            
        cons = [ copy.deepcopy(con) ]
        cons[0].value = value

        opt = GeomOpt(los,igeom,constraints=cons)
        

        print("Finished angle:",value)

        opt.data["name"] = "ang%03i"%(int(round(value)))
        #opt.info["angle"] = value
        if i == nscan:
            if opt.get_potential_energy() < scan[0].get_potential_energy():
                scan[0] = opt
        else:
            scan.append(opt)

    scan = sorted(scan,key=lambda x: x.data["name"])

    #for s in scan:
    #    if s.info["energy"] < mingeom.info["energy"]:
    #        mingeom = s
    
    return ListOfStruct(scan)


def FwdRevDihedScan_worker(args):
    """ Worker function for parallel forward and reverse dihedral scans."""
    return DihedScan(*args)


def FwdRevDihedScan(los,struct,con,sched,parallel=False):
    """ Perform a dihedral scan using both forward and reverse scans to find the minimum energy conformation.
    
    Parameters
    ----------
    los : ffpopt.Struct.ListOfStruct
        Structure list object used to build calculator
    struct : ffpopt.Struct.Struct
        A Struct object (usually los.structs[0])
    con : ffpopt.Constraint.Constraint
        The dihedral constraint to scan.
    sched : list of float
        A list of dihedral angles in degrees to scan.
    parallel : bool, optional
        Whether to perform the forward and reverse scans in parallel. Default is False.
    
    Returns
    -------
    ListOfStruct
        A list of Structs corresponding to the scanned geometries. The Struct names
        are angXXX, where XXX is a zero-padded integer of the dihedral angle in degrees.
    """
    
    import ase.io
    from ffpopt.Struct import ListOfStruct
    
    revsched = sched[::-1]

    if parallel:
        import multiprocessing

        proclist = [ (los,struct,con,sched),
                     (los,struct,con,revsched) ]

        los.calc.reset()
        los.calc = None
        
        with multiprocessing.Pool(processes=2) as pool:
            olists = pool.map(FwdRevDihedScan_worker,proclist)
        fwd = olists[0]
        rev = olists[1]
    else:
        fwd = DihedScan(los,struct,con,sched)
        rev = DihedScan(los,struct,con,revsched)

        
    scan = []
    
    for a,b in zip(fwd,rev):
        if a.data["name"] != b.data["name"]:
            raise Exception("Expected fwd and rev scans at the same set of dihedrals, but found %s and %s\n"%(a.data["name"],b.data["name"]))
        if a.data["energy"] < b.data["energy"]:
            scan.append(a)
        else:
            scan.append(b)

    return ListOfStruct(scan)




###########################################################################################
###########################################################################################
###########################################################################################
        

from ffpopt.geom.GeomOptParallel import (  # noqa: F401
    ParallelGeomOpt,
    CalcNode,
    _run_node,
    is_mpi_worker,
    is_mpi,
    ParallelGeomOpt_threads,
    _worker_init,
    _run_node_mpi,
    ParallelGeomOpt_mpi,
)
