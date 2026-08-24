"""ASE geometry optimization backend and covalent/fmax guards."""

from __future__ import annotations

def _ase_fmax(atoms) -> float:
    """Max atomic force magnitude (eV/Ang)."""
    import numpy as np

    forces = atoms.get_forces()
    return float(np.sqrt((forces ** 2).sum(axis=1).max()))


def _ase_loose_fmax(strict_tol: float) -> float:
    """Soft-accept threshold for ASE when strict fmax is not met.

    Override with ``FFPOPT_ASE_LOOSE_FMAX`` (eV/Ang). Default is
    ``max(3 * ase_opt_tol, 0.05)`` when the JSON value is ``null``.
    """
    from ffpopt.runtime.EnvDefaults import env_value

    raw = env_value("FFPOPT_ASE_LOOSE_FMAX")
    if raw is not None:
        return float(raw)
    return max(3.0 * float(strict_tol), 0.05)


def _struct_bonds_numbers(struct):
    """Covalent bonds and atomic numbers from a Struct, if present."""
    data = getattr(struct, "data", None) or {}
    bonds = data.get("bonds") or []
    getter = getattr(struct, "GetASEAtoms", None)
    atoms = getter() if callable(getter) else None
    numbers = None
    if atoms is not None:
        try:
            numbers = atoms.get_atomic_numbers()
        except Exception:
            numbers = None
    return bonds, numbers, atoms


def _explode_fmax_limit() -> float:
    from ffpopt.runtime.EnvDefaults import env_float

    return float(env_float("FFPOPT_GEOM_EXPLODE_FMAX", 50.0))


def _guard_covalent_geometry(positions, bonds, numbers=None, *, where="geometry"):
    from ffpopt.geom.Constraints import assert_sane_covalent_geometry

    assert_sane_covalent_geometry(positions, bonds, numbers, where=where)


def _attach_ase_geometry_guard(optimizer, atoms, bonds, numbers=None):
    """Abort an ASE optimizer as soon as a covalent bond explodes."""
    from ffpopt.geom.Constraints import BrokenGeometryError, covalent_geometry_error

    def _check():
        err = covalent_geometry_error(atoms.get_positions(), bonds, numbers)
        if err:
            raise BrokenGeometryError(f"ASE step: {err}")

    try:
        optimizer.attach(_check, interval=1)
    except Exception:
        pass
    _check()



def _ase_optimizer_classes():
    """Ordered ASE optimizers for difficult constrained cases.

    Under fast wavefront mode, try LBFGS only (skip BFGS->FIRE ladder).
    """
    from ase.optimize import BFGS, FIRE, LBFGS
    from ffpopt.runtime.FastWavefront import fast_wavefront_enabled

    if fast_wavefront_enabled(None):
        return (("LBFGS", LBFGS),)
    return (("BFGS", BFGS), ("LBFGS", LBFGS), ("FIRE", FIRE))


def GeomOpt_ASE(los,struct,constraints=None,restraints=None):
    """ Perform a geometry optimization using ASE (BFGS -> LBFGS -> FIRE).

    Tries BFGS first, then LBFGS and FIRE from the best geometry so far.
    Accepts a soft-converged result when ``fmax`` is below
    :func:`_ase_loose_fmax` even if the strict ``ase_opt_tol`` was missed -
    important for frozen-dihedral wavefront nodes that otherwise die after
    geomeTRIC already nearly relaxed the structure.
    
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
    import sys
    from ffpopt.geom.Constraints import ConstraintList
    from ffpopt.geom.Restraints import RestraintList
    from ffpopt.geom.Constraints import ApplyConstraints, to_ase

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

    origatoms = struct.GetASEAtoms()
    myatoms = struct.GetASEAtoms()
    try:
        from ffpopt.geom.Geometric import get_persistent_calc

        calc = get_persistent_calc(los, struct, reslist=reslist)
    except Exception:
        myatoms.calc = los.BuildRestrainedCalc(struct, reslist=reslist)
        calc = myatoms.calc
    else:
        myatoms.calc = calc

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
        cons = conslist.FillConstraints(myatoms,force=False)
        origcons = conslist.FillConstraints(myatoms,force=True)
        myatoms = ApplyConstraints(myatoms,cons,graph=struct.GetGraph())
        if cons is not None:
            asecons = to_ase(cons)

    del myatoms.constraints
    myatoms.set_constraint( asecons )
    myatoms.calc = calc
    myatoms.calc.reset()
    logfile = sys.stderr

    bonds, numbers, _ = _struct_bonds_numbers(struct)
    _guard_covalent_geometry(
        myatoms.get_positions(), bonds, numbers, where="ASE pre-opt"
    )
    try:
        f0 = _ase_fmax(myatoms)
    except Exception as exc:
        from ffpopt.geom.Constraints import BrokenGeometryError

        raise BrokenGeometryError(f"ASE pre-opt forces failed: {exc}") from exc
    explode = _explode_fmax_limit()
    if f0 > explode:
        from ffpopt.geom.Constraints import BrokenGeometryError

        raise BrokenGeometryError(
            f"ASE pre-opt fmax={f0:.3g} eV/Ang exceeds {explode:.3g} "
            "(geometry will explode; skipping optimizer)"
        )

    strict_tol = float(los.args.ase_opt_tol)
    loose_tol = _ase_loose_fmax(strict_tol)
    max_steps = int(los.args.geometric_maxiter)
    best_fmax = float("inf")
    accepted = False
    accepted_how = None

    from ffpopt.geom.Constraints import BrokenGeometryError

    for name, OptCls in _ase_optimizer_classes():
        try:
            myatoms.calc.reset()
        except Exception:
            pass
        try:
            optimizer = OptCls(myatoms, logfile=logfile, maxstep=0.2)
        except TypeError:
            optimizer = OptCls(myatoms, logfile=logfile)
        _attach_ase_geometry_guard(optimizer, myatoms, bonds, numbers)
        try:
            converged = bool(optimizer.run(fmax=strict_tol, steps=max_steps))
        except BrokenGeometryError:
            raise
        except Exception as exc:
            from ffpopt.runtime.Console import ascii_for_stdio

            sys.stderr.write(
                ascii_for_stdio(
                    f"[ffpopt] ASE {name} raised ({type(exc).__name__}: {exc}); "
                    f"trying next optimizer\n"
                )
            )
            continue
        try:
            fmax = _ase_fmax(myatoms)
        except Exception:
            fmax = best_fmax
        if fmax < best_fmax:
            best_fmax = fmax
        if converged or fmax <= strict_tol:
            accepted = True
            accepted_how = name
            break
        if fmax <= loose_tol:
            sys.stderr.write(
                f"[ffpopt] ASE {name} soft-accept fmax={fmax:.4g} eV/Ang "
                f"(strict={strict_tol:.4g}, loose={loose_tol:.4g})\n"
            )
            accepted = True
            accepted_how = f"{name}-soft"
            break
        sys.stderr.write(
            f"[ffpopt] ASE {name} not tight (fmax={fmax:.4g} eV/Ang); "
            f"continuing with next optimizer\n"
        )

    if not accepted:
        raise RuntimeError(
            f"ASE geometry optimization failed to reach loose fmax "
            f"{loose_tol:.4g} eV/Ang (best fmax={best_fmax:.4g})"
        )

    ene = myatoms.get_potential_energy()
    crd = myatoms.get_positions()
    frc = myatoms.get_forces()
    clone = getattr(struct, "clone_geometry", None)
    if callable(clone):
        out = clone(coords=crd, ene=ene, frcs=frc)
    else:
        out = copy.deepcopy(struct)
        out.Update(ene, crd, frc)
    if accepted_how is not None:
        out.data["ase_opt_recovery"] = accepted_how

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
    return out



