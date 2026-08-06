#!/usr/bin/env python3


def GeomOpt_ASE(los,struct,constraints=None,restraints=None):
    """ Perform a geometry optimization using the ASE BFGS optimizer.
    
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
    Returns
    -------
    ffpopt.Struct.Struct
        The optimized geometry with updated positions, forces, and energy
    
    """
    import os
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"
 
    import copy
    import ase.io
    from . Constraints import ConstraintList
    from . Restraints import RestraintList
    #from . Constraints import constraints2ase
    from . Constraints import ApplyConstraints, to_ase
    import sys
    
    if True:
        
        from ase.optimize import BFGS

        reslist = None
        if struct.restraints is not None:
            if len(struct.restraints.rests) > 0:
                reslist = copy.deepcopy(struct.restraints)
                if restraints is not None:
                    for b in restraints:
                        found=False
                        for a in reslist.rests:
                            if a.is_same(b):
                                a.value=b.value
                                found=True
                        if not found:
                            reslist.rests.append(b)
        
        if reslist is None and restraints is not None:
            reslist = RestraintList( copy.deepcopy(restraints) )

        #print(reslist.to_list_of_dict())

        origatoms = struct.GetASEAtoms()
        myatoms = struct.GetASEAtoms()
        myatoms.calc=los.BuildRestrainedCalc(struct,reslist=reslist)
        calc = myatoms.calc

        conslist = None
        if struct.constraints is not None:
            if len(struct.constraints) > 0:
                conslist = copy.deepcopy(struct.constraints)
                if constraints is not None:
                    for b in constraints:
                        found=False
                        for a in conslist.cons:
                            if a.is_same(b):
                                a.value = b.value
                                found=True
                        if not found:
                            conslist.cons.append(b)
                            
        if conslist is None and constraints is not None:
            conslist = ConstraintList( copy.deepcopy(constraints) )
            
        # asecons = None
        # cons = None
        # if conslist is not None:
        #     cons = conslist.FillConstraints(myatoms)
        #     myatoms = ApplyConstraints(myatoms,cons)
        #     #asecons = constraints2ase(cons)
        #     asecons = to_ase(cons)



        asecons = None
        cons = None
        if conslist is not None:
            cons = conslist.FillConstraints(myatoms,force=False)
            origcons = conslist.FillConstraints(myatoms,force=True)
        #if cons is not None or reslist is not None:
            myatoms = ApplyConstraints(myatoms,cons,graph=struct.GetGraph()) #,rests=reslist.rests,k=1)
            if cons is not None:
                asecons = to_ase(cons)
                #newcons = conslist.FillConstraints(myatoms,force=True)
                #for ic in range(len(cons)):
                #    print(ic,str(origcons[ic]),str(newcons[ic]),cons[ic].value)


            
    
        del myatoms.constraints
        myatoms.set_constraint( asecons )
        myatoms.calc = calc
        myatoms.calc.reset()
        logfile = sys.stderr
        #if los.args.geometric_ini is None:
        #    logfile = os.devnull
        #    if len(los.args.geometric_ini) > 0:
        #        logfile = los.args.geometric_ini
        optimizer = BFGS(myatoms,logfile=logfile)
        optimizer.run(fmax=los.args.ase_opt_tol,steps=los.args.geometric_maxiter)
        
        ene = myatoms.get_potential_energy()
        crd = myatoms.get_positions()
        frc = myatoms.get_forces()
        out = copy.deepcopy(struct)
        out.Update(ene,crd,frc)
                                           

        if True:
            from . Constraints import FillConstraints
            if cons is not None:
                cvals = FillConstraints(out,cons,force=True)
                ovals = FillConstraints(origatoms,cons,force=True)
                for i in range(len(cvals)):
                    print("Constraint %2i tgt=%9.2f opt=%9.2f orig=%9.2f"%\
                          ( i+1, cons[i].value, cvals[i].value, ovals[i].value ) )
            
        if True:
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
    return out




def _linux_process_tree_cputime(pid: int):
    """Sum ``utime+stime`` (jiffies) for ``pid`` and descendants via ``/proc``.

    Returns ``None`` when ``/proc`` is unavailable (non-Linux) or unreadable.
    Used so the geomeTRIC watchdog does not treat a busy energy evaluation as a
    stall merely because the ``.log`` file is quiet.
    """
    import os

    total = 0
    stack = [int(pid)]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            with open(f"/proc/{cur}/stat", "r", encoding="utf-8") as fh:
                data = fh.read()
            # ``comm`` is in parentheses and may contain spaces.
            rparen = data.rfind(")")
            fields = data[rparen + 2 :].split()
            total += int(fields[11]) + int(fields[12])
        except (OSError, IndexError, ValueError):
            continue
        # Prefer the kernel's children list when present.
        try:
            with open(
                f"/proc/{cur}/task/{cur}/children", "r", encoding="utf-8"
            ) as fh:
                stack.extend(int(x) for x in fh.read().split())
            continue
        except OSError:
            pass
        try:
            for name in os.listdir(f"/proc/{cur}/task"):
                with open(
                    f"/proc/{cur}/task/{name}/children", "r", encoding="utf-8"
                ) as fh:
                    stack.extend(int(x) for x in fh.read().split())
        except OSError:
            pass
    return total


def _path_tree_mtime(path: str) -> float:
    """Newest mtime among ``path`` and its immediate children (best-effort)."""
    import os

    try:
        newest = os.path.getmtime(path)
    except OSError:
        return 0.0
    try:
        names = os.listdir(path)
    except OSError:
        return newest
    for name in names:
        try:
            newest = max(newest, os.path.getmtime(os.path.join(path, name)))
        except OSError:
            continue
    return newest


def _geometric_stall_timeout_sec(default: float = 30 * 60) -> float:
    """Stall timeout from ``FFPOPT_GEOMETRIC_STALL_SEC`` (``0`` disables)."""
    import os

    raw = os.environ.get("FFPOPT_GEOMETRIC_STALL_SEC")
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _run_geometric_with_watchdog(
    cmds,
    tmplog,
    activity_dir=None,
    poll_interval_sec=5.0,
    stall_timeout_sec=None,
    bmatrix_wedge_pattern="more than 1000 B-matrices stored",
):
    """Run geometric-optimize as a child process and watch its log for wedged states.

    Raises RuntimeError if the B-matrix accumulation warning appears, or if
    the job shows *no* progress for ``stall_timeout_sec``. Progress is any of:
    log growth, increasing process-tree CPU time, or updates under
    ``activity_dir`` (typically geomeTRIC's ``*.tmp`` folder). Quiet logs alone
    are not enough to declare a stall — ML / xTB gradient evaluations can sit
    for many minutes between log lines.

    Set ``FFPOPT_GEOMETRIC_STALL_SEC=0`` to disable stall kills (B-matrix wedge
    detection remains). On any wedge the child's process group is SIGTERM'd
    (then SIGKILL'd after 10s). The caller is expected to translate the
    exception into a fallback or ``_mark_failed()``.
    """
    import os
    import signal
    import subprocess as subp
    import time

    if stall_timeout_sec is None:
        stall_timeout_sec = _geometric_stall_timeout_sec()

    child_env = os.environ.copy()
    #child_env["PYTHONWARNINGS"] = "ignore:ignore_bad_restart_file:FutureWarning"

    proc = subp.Popen(cmds, text=True, env=child_env,
                      start_new_session=True)

    log_pos = 0
    # carry the tail of the last chunk so a pattern split across two
    # reads is still detected
    log_tail = ""
    last_change = time.monotonic()
    last_cpu = _linux_process_tree_cputime(proc.pid)
    last_dir_mtime = (
        _path_tree_mtime(activity_dir) if activity_dir is not None else 0.0
    )
    wedge_reason = None

    try:
        while proc.poll() is None:
            progressed = False

            try:
                cur_size = os.path.getsize(tmplog)
            except OSError:
                cur_size = 0

            if cur_size > log_pos:
                try:
                    with open(tmplog, "r") as fh:
                        fh.seek(log_pos)
                        chunk = fh.read()
                except OSError:
                    chunk = ""
                if chunk:
                    log_pos += len(chunk)
                    scan_text = log_tail + chunk
                    if bmatrix_wedge_pattern in scan_text:
                        wedge_reason = (
                            f"geomeTRIC wedged: '{bmatrix_wedge_pattern}'"
                            f" detected in {tmplog}"
                        )
                        break
                    log_tail = scan_text[-len(bmatrix_wedge_pattern):]
                    progressed = True
            elif cur_size < log_pos:
                # log was rotated/truncated
                log_pos = 0
                log_tail = ""
                progressed = True

            cpu = _linux_process_tree_cputime(proc.pid)
            if (
                cpu is not None
                and last_cpu is not None
                and cpu > last_cpu
            ):
                last_cpu = cpu
                progressed = True
            elif cpu is not None and last_cpu is None:
                last_cpu = cpu

            if activity_dir is not None:
                dir_mtime = _path_tree_mtime(activity_dir)
                if dir_mtime > last_dir_mtime:
                    last_dir_mtime = dir_mtime
                    progressed = True

            if progressed:
                last_change = time.monotonic()
            elif (
                stall_timeout_sec > 0
                and time.monotonic() - last_change > stall_timeout_sec
            ):
                wedge_reason = (
                    f"geomeTRIC stalled: no log/CPU/tmpdir progress for "
                    f"{stall_timeout_sec}s (log={tmplog})"
                )
                break

            time.sleep(poll_interval_sec)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                proc.wait(timeout=10)
            except subp.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subp.TimeoutExpired:
                    pass

    if wedge_reason is not None:
        raise RuntimeError(wedge_reason)


def GeomOpt_GEOMETRIC(los,struct,constraints=None,restraints=None):
    """ Perform a geometry optimization using the GEOMETRIC program.

    By default runs geomeTRIC **in-process** with a persistent ASE calculator
    cached on ``los`` (avoids spawning ``python -m ffpopt.geometric_compat``
    per call). Set ``FFPOPT_GEOMETRIC_SUBPROCESS=1`` to restore the legacy
    subprocess + watchdog path.

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

    Returns
    -------
    ffpopt.Struct.Struct
        The optimized geometry with updated positions, forces, and energy
    """
    import os
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"

    import copy
    import ase.io
    from . constants import AU_PER_ELECTRON_VOLT
    from . Options import argparse2geometric, configure_geometric_logging, GetStandardOptions
    from . Constraints import ConstraintList
    from . Restraints import RestraintList
    from . Constraints import ApplyConstraints
    from . Struct import ListOfStruct
    from . geometric_inprocess import (
        get_persistent_calc,
        run_geometric_inprocess,
        use_geometric_subprocess,
    )
    from tempfile import mkstemp
    from pathlib import Path

    tmpfile_loc = "./tmpfiles"
    if not Path(tmpfile_loc).is_dir():
        os.makedirs(tmpfile_loc, exist_ok=True)

    fd,tmpxyz = mkstemp(dir=tmpfile_loc,prefix="tmp.",suffix=".xyz")
    if not os.isatty(fd):  # Check if fd is still valid
        os.close(fd)

    tmpbase = str(Path(tmpxyz).with_suffix(""))
    tmpopt  = tmpbase + "_optim.xyz"
    tmplog  = tmpbase + ".log"
    tmpcons = tmpbase + ".cons.inp"
    tmpdir  = tmpbase + ".tmp"
    tmpjson = tmpbase + ".json"

    origatoms = struct.GetASEAtoms()
    myatoms = copy.deepcopy(origatoms)

    reslist = None
    if struct.restraints is not None:
        if len(struct.restraints.rests) > 0:
            reslist = copy.deepcopy(struct.restraints)
            if restraints is not None:
                for b in restraints:
                    found=False
                    for a in reslist.rests:
                        if a.is_same(b):
                            a.value=b.value
                            found=True
                    if not found:
                        reslist.rests.append(b)

    if reslist is None and restraints is not None:
        reslist = RestraintList( copy.deepcopy(restraints) )

    conslist = None
    if struct.constraints is not None:
        if len(struct.constraints) > 0:
            conslist = copy.deepcopy(struct.constraints)

            if constraints is not None:
                for b in constraints:
                    found=False
                    for a in conslist.cons:
                        if a.is_same(b):
                            a.value = b.value
                            found=True
                    if not found:
                        conslist.cons.append(b)

    if conslist is None and constraints is not None:
        conslist = ConstraintList( copy.deepcopy(constraints) )

    cons = None
    if conslist is not None:
        cons = conslist.FillConstraints(myatoms,force=False)
        origcons = conslist.FillConstraints(myatoms,force=True)
        myatoms = ApplyConstraints(myatoms,cons,graph=struct.GetGraph())

    if conslist is not None:
        with open(tmpcons, "w") as fh:
            fh.write("$set\n")
            for line in conslist.to_geometric():
                fh.write("%s\n" % (line))

    ase.io.write(tmpxyz, myatoms, format="xyz", parallel=False)

    if use_geometric_subprocess():
        # Legacy path: spawn geometric_compat + watchdog.
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
        result = run_geometric_inprocess(
            myatoms,
            calc,
            prefix=tmpbase,
            constraints_path=tmpcons if conslist is not None else None,
            coordsys=geo.get("coordsys", "tric"),
            maxiter=int(geo.get("maxiter", getattr(los.args, "geometric_maxiter", 500))),
            converge=geo.get("converge", getattr(los.args, "geometric_converge", "set GAU")),
            enforce=float(geo.get("enforce", getattr(los.args, "geometric_enforce", 0.0))),
            log_ini=log_ini if log_ini else None,
        )
        crd = result["coords"]
        if result["energy_ha"] is not None:
            ene = result["energy_ha"] / AU_PER_ELECTRON_VOLT()
        else:
            myatoms.set_positions(crd)
            myatoms.calc = calc
            ene = myatoms.get_potential_energy()
        frc = None

    out = copy.deepcopy(struct)
    out.Update(ene,crd,frc)

    if True:
        from . Constraints import FillConstraints
        if cons is not None:
            cvals = FillConstraints(out,cons,force=True)
            ovals = FillConstraints(origatoms,cons,force=True)
            for i in range(len(cvals)):
                print("Constraint %2i tgt=%9.2f opt=%9.2f orig=%9.2f"%\
                      ( i+1, cons[i].value, cvals[i].value, ovals[i].value ) )

    if True:
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

    for f in [tmpxyz,tmpopt,tmplog,tmpcons,tmpjson]:
        if os.path.exists(f):
            os.remove(f)

    if os.path.isdir(tmpdir):
        import shutil
        shutil.rmtree(tmpdir)

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
    #from . constants import AU_PER_ELECTRON_VOLT
    #from . Options import argparse2geometric
    from . Constraints import ConstraintList
    from . Constraints import to_ase,ApplyConstraints
    from . Restraints import RestraintList
    #from . Constraints import constraints2info,constraints2ase

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



        reslist = None
        if struct.restraints is not None:
            if len(struct.restraints.rests) > 0:
                reslist = copy.deepcopy(struct.restraints)
                if restraints is not None:
                    for b in restraints:
                        found=False
                        for a in reslist.rests:
                            if a.is_same(b):
                                a.value=b.value
                                found=True
                        if not found:
                            reslist.rests.append(b)
        
        if reslist is None and restraints is not None:
            reslist = RestraintList( copy.deepcopy(restraints) )
        
        
        myatoms = struct.GetASEAtoms()
        myatoms.calc=los.BuildRestrainedCalc(struct,reslist=reslist)
        calc = myatoms.calc

        conslist = None
        if struct.constraints is not None:
            if len(struct.constraints) > 0:
                conslist = copy.deepcopy(struct.constraints)
                if constraints is not None:
                    for b in constraints:
                        found=False
                        for a in conslist.cons:
                            if a.is_same(b):
                                a.value = b.value
                                found=True
                        if not found:
                            conslist.cons.append(b)
        if conslist is None and constraints is not None:
            conslist = ConstraintList( copy.deepcopy(constraints) )

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



def GeomOpt(los,struct,constraints=None,restraints=None):
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

    Returns
    -------
    ffpopt.Struct.Struct
        The optimized geometry with updated positions, forces, and energy
    """

    if los.args.no_opt:
        out = GeomOpt_SinglePoint(los,struct,constraints,restraints)
    elif not los.args.geometric_opt:
        try:
            out = GeomOpt_ASE(los,struct,constraints,restraints)
        except Exception as e:
            import traceback
            print("\n\n\nASE GEOMETRY OPTIMIZATION FAILURE\n")
            print(e)
            traceback.print_exc()
            out = GeomOpt_GEOMETRIC(los,struct,constraints,restraints)
    else:
        try:
            out = GeomOpt_GEOMETRIC(los,struct,constraints,restraints)
        except Exception as e:
            # geomeTRIC sometimes cannot recover its IC system under frozen
            # dihedrals (Cartesian fallback, Brent "Not bracketed", stall
            # watchdog, …). Fall back to ASE BFGS with the same constraints.
            import traceback
            print("\n\n\ngeomeTRIC GEOMETRY OPTIMIZATION FAILURE; falling back to ASE\n")
            print(e)
            traceback.print_exc()
            out = GeomOpt_ASE(los,struct,constraints,restraints)
    return out





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
    from . constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

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
    from . Struct import ListOfStruct
    
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
    from . Struct import ListOfStruct
    
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
        

def ParallelGeomOpt(los,norestene,nproc):
    out = None
    if is_mpi():
        out = ParallelGeomOpt_mpi(los, norestene)
    else:
        out = ParallelGeomOpt_threads(los,norestene,nproc)
    return out
        

###########################################################################################
###########################################################################################
###########################################################################################
        
class CalcNode(object):
    def __init__(self,los,s,norestene):
        self.los = los
        self.s = s
        self.norestene = norestene
        self.out = None

    def calculate(self):
        from ffpopt.GeomOpt import GeomOpt,GeomOpt_SinglePoint
        import copy
        if self.los.args.no_opt:
            self.out = copy.deepcopy(self.s)
        else:
            self.out = GeomOpt(self.los,self.s)
        tmp = copy.deepcopy(self.out)
        if self.norestene:
            tmp.restraints = None
            tmp.constraints = None
        tmp = GeomOpt_SinglePoint(self.los,tmp)
        self.out.Update( tmp.get_potential_energy(), tmp.get_positions(), tmp.get_forces() )
        self.los.calc = None
        #self.los = None

        
def _run_node( node ):
    node.calculate()
    return node


def is_mpi_worker():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return size > 1 and rank > 0


def is_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    size = comm.Get_size()
    return size > 1


def ParallelGeomOpt_threads(los,norestene,nproc):
    import concurrent.futures
    import multiprocessing
    from . Struct import ListOfStruct

    nodes = [ CalcNode(los,s,norestene) for s in los ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
        results = list(executor.map(_run_node, nodes))
    return ListOfStruct( [ node.out for node in results ] )


# -------------------------------------------------------------------------
# WORKER SIDE CAR ENVIRONMENT
# -------------------------------------------------------------------------
# Worker-level global storage variables
_WORKER_LOS = None
_WORKER_NORESTENE = None

def _worker_init(los, norestene):
    """
    Runs ONCE per worker process when it joins the cluster.
    Safely stores context metadata in worker memory space.
    """
    global _WORKER_LOS, _WORKER_NORESTENE
    _WORKER_LOS = los
    _WORKER_NORESTENE = norestene

def _run_node_mpi(s):
    """
    Executes on a single structure using pre-cached environment context.
    """
    global _WORKER_LOS, _WORKER_NORESTENE
    
    # Instantiate the node locally using the cached background variables
    node = CalcNode(_WORKER_LOS, s, _WORKER_NORESTENE)
    node.calculate()
    
    # Return ONLY the structure output payload to minimize MPI data footprint
    return node.out

# -------------------------------------------------------------------------
# TARGET MPI FUNCTION
# -------------------------------------------------------------------------
def ParallelGeomOpt_mpi(los, norestene):
    """
    Asynchronous streaming MPI implementation.
    Safely captures the existing mpirun worker pool.
    """
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    from . Struct import ListOfStruct
    
    # MPICommExecutor partitions COMM_WORLD.
    # Workers enter a passive processing loop inside the 'with' block context.
    # Only Rank 0 exits the block to submit jobs via the executor.
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            # Set up global contextual environments on worker memory pools
            # Note: MPICommExecutor does not support the 'initializer' parameter, 
            # so we map the initialization function across workers manually.
            num_workers = MPI.COMM_WORLD.Get_size() - 1
            los.calc = None
            if num_workers > 0:
                list(executor.map(_worker_init, [los]*num_workers, [norestene]*num_workers))

            # Dynamically stream data chunks to achieve perfect load balancing
            results_iterator = executor.map(_run_node_mpi, list(los), chunksize=1)
            final_outputs = list(results_iterator)
            out = ListOfStruct(final_outputs)
            return out
    return None
