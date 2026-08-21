"""Extended dihedral fit knobs: phase, period, scee/scnb + L-BFGS-B / JAX.

Default GenDihedFit remains barrier-height (FC) only via ``lsq_linear``.
Enable the AFFDO-style vector with CLI / env flags; backends:

* ``lsq`` — fixed-geometry FC-only (legacy; used when only FCs are free)
* ``lbfgsb`` — SciPy L-BFGS-B on the extended objective
* ``jax`` — same objective with JAX autodiff Jacobian (optional extra)
"""

from __future__ import annotations

from typing import Optional

from ffpopt.runtime.EnvDefaults import env_float, env_str


def fit_mode_from_env(default: str = "barrier") -> str:
    """``barrier`` | ``torsion`` | ``full`` from ``FFPOPT_FIT_MODE``."""
    raw = env_str("FFPOPT_FIT_MODE", default).strip().lower()
    if raw in ("barrier", "barrier-only", "fc", "fcs"):
        return "barrier"
    if raw in ("torsion", "torsions", "phase"):
        return "torsion"
    if raw in ("full", "affdo", "all"):
        return "full"
    return default


def fit_backend_from_env(default: str = "lsq") -> str:
    """``lsq`` | ``lbfgsb`` | ``jax`` from ``FFPOPT_FIT_BACKEND``."""
    raw = env_str("FFPOPT_FIT_BACKEND", default).strip().lower()
    if raw in ("lsq", "lsq_linear", "linear"):
        return "lsq"
    if raw in ("lbfgsb", "l-bfgs-b", "scipy"):
        return "lbfgsb"
    if raw in ("jax", "autodiff"):
        return "jax"
    return default


def apply_fit_flags_to_args(args) -> None:
    """Stamp extended-fit attributes onto an argparse / SimpleNamespace ``args``."""
    mode = getattr(args, "fit_mode", None) or fit_mode_from_env("barrier")
    backend = getattr(args, "fit_backend", None) or fit_backend_from_env("lsq")

    if getattr(args, "barrier_only", False):
        mode = "barrier"
    if getattr(args, "fit_full", False):
        mode = "full"

    args.fit_mode = mode
    args.fit_backend = backend

    opt_phase = bool(getattr(args, "fit_phases", False) or getattr(args, "opt_phase", False))
    opt_periods = bool(getattr(args, "fit_periods", False) or getattr(args, "opt_periods", False))
    opt_scee_scnb = bool(
        getattr(args, "fit_scee_scnb", False) or getattr(args, "opt_scee_scnb", False)
    )

    if mode == "torsion":
        opt_phase = True
        opt_periods = True
        opt_scee_scnb = False
    elif mode == "full":
        opt_phase = True
        opt_periods = True
        opt_scee_scnb = True
    elif mode == "barrier":
        if getattr(args, "barrier_only", False):
            opt_phase = opt_periods = opt_scee_scnb = False

    args.opt_phase = bool(opt_phase)
    args.opt_periods = bool(opt_periods)
    args.opt_scee_scnb = bool(opt_scee_scnb)

    if not hasattr(args, "scee") or getattr(args, "scee", None) is None:
        args.scee = env_float("FFPOPT_SCEE")
    if not hasattr(args, "scnb") or getattr(args, "scnb", None) is None:
        args.scnb = env_float("FFPOPT_SCNB")

    if (args.opt_phase or args.opt_periods or args.opt_scee_scnb) and args.fit_backend == "lsq":
        args.fit_backend = "lbfgsb"


def configure_fit_input(finp, args) -> None:
    """Attach extended-fit state onto a :class:`FitInputType` instance."""
    apply_fit_flags_to_args(args)
    finp.opt_phase = bool(args.opt_phase)
    finp.opt_periods = bool(args.opt_periods)
    finp.opt_scee_scnb = bool(args.opt_scee_scnb)
    finp.fit_backend = str(args.fit_backend)
    finp.scee = float(args.scee)
    finp.scnb = float(args.scnb)
    finp.fit_mode = str(args.fit_mode)
    for s in finp.systems:
        s._fit_owner = finp


def count_extended_params(finp) -> int:
    """Parameter count matching :func:`get_extended_params`."""
    n = 0
    for pname in finp.ptypedict:
        nprim = finp.ptypedict[pname].nprim
        n += nprim  # FCs
        if getattr(finp, "opt_phase", False):
            n += nprim
        if getattr(finp, "opt_periods", False):
            n += nprim
    if getattr(finp, "opt_scee_scnb", False):
        n += 2
    return n


def get_extended_params(finp):
    import numpy as np

    x = []
    for pname in finp.ptypedict:
        dfcns = finp.ptypedict[pname].dfcns
        for prim in dfcns.prims:
            x.append(float(prim.fc))
        if getattr(finp, "opt_phase", False):
            for prim in dfcns.prims:
                x.append(float(prim.phase))
        if getattr(finp, "opt_periods", False):
            for prim in dfcns.prims:
                x.append(float(prim.per))
    if getattr(finp, "opt_scee_scnb", False):
        x.append(float(getattr(finp, "scee", 1.2)))
        x.append(float(getattr(finp, "scnb", 2.0)))
    return np.asarray(x, dtype=float)


def set_extended_params(finp, x) -> None:
    import numpy as np

    x = np.asarray(x, dtype=float)
    ipar = 0
    for pname in finp.ptypedict:
        ptype = finp.ptypedict[pname]
        nprim = ptype.nprim
        ptype.dfcns.SetFCs(x[ipar : ipar + nprim])
        ipar += nprim
        if getattr(finp, "opt_phase", False):
            ptype.dfcns.SetPhases(x[ipar : ipar + nprim])
            ipar += nprim
        if getattr(finp, "opt_periods", False):
            for i, prim in enumerate(ptype.dfcns.prims):
                # Keep periods positive; round toward nearest integer in [1, 12].
                per = float(x[ipar + i])
                per_i = int(round(per))
                prim.per = max(1, min(12, per_i if per_i >= 1 else 1))
            ipar += nprim
    if getattr(finp, "opt_scee_scnb", False):
        finp.scee = float(max(0.5, min(4.0, x[ipar])))
        finp.scnb = float(max(0.5, min(4.0, x[ipar + 1])))


def extended_bounds(finp, x0):
    """L-BFGS-B bounds for the extended parameter vector."""
    import numpy as np

    x0 = np.asarray(x0, dtype=float)
    lo = np.empty_like(x0)
    hi = np.empty_like(x0)
    ipar = 0
    for pname in finp.ptypedict:
        nprim = finp.ptypedict[pname].nprim
        # FCs
        lo[ipar : ipar + nprim] = x0[ipar : ipar + nprim] - 2.0
        hi[ipar : ipar + nprim] = x0[ipar : ipar + nprim] + 5.0
        ipar += nprim
        if getattr(finp, "opt_phase", False):
            lo[ipar : ipar + nprim] = -180.0
            hi[ipar : ipar + nprim] = 180.0
            ipar += nprim
        if getattr(finp, "opt_periods", False):
            lo[ipar : ipar + nprim] = 1.0
            hi[ipar : ipar + nprim] = 6.0
            ipar += nprim
    if getattr(finp, "opt_scee_scnb", False):
        lo[ipar] = 0.5
        hi[ipar] = 4.0
        lo[ipar + 1] = 0.5
        hi[ipar + 1] = 4.0
    return list(zip(lo.tolist(), hi.tolist()))


def _pair14_unscaled_kcal(parm, positions):
    """Unscaled 1–4 Coulomb and LJ (kcal/mol) for all proper dihedral end pairs."""
    import numpy as np

    pos = np.asarray(positions, dtype=float)
    pairs = {}
    for dih in parm.dihedrals:
        if getattr(dih, "improper", False):
            continue
        # End atoms of the torsion define the 1–4 interaction.
        a = dih.atom1
        b = dih.atom4
        i, j = int(a.idx), int(b.idx)
        if i > j:
            i, j = j, i
        key = (i, j)
        if key in pairs:
            continue
        qi = float(a.charge)
        qj = float(b.charge)
        r = float(np.linalg.norm(pos[i] - pos[j]))
        if r < 1e-6:
            continue
        # Amber electrostatic constant (kcal·Å / e^2)
        elec = 332.0522173 * qi * qj / r
        # LJ from combining rules on atom type parameters when present.
        try:
            eps_i = float(a.epsilon)
            eps_j = float(b.epsilon)
            rmin_i = float(a.rmin)
            rmin_j = float(b.rmin)
            eps = np.sqrt(max(eps_i, 0.0) * max(eps_j, 0.0))
            rmin = rmin_i + rmin_j
            rho = rmin / r
            lj = eps * (rho ** 12 - 2.0 * (rho ** 6))
        except Exception:
            lj = 0.0
        pairs[key] = (elec, lj)
    elec_sum = sum(v[0] for v in pairs.values())
    lj_sum = sum(v[1] for v in pairs.values())
    return float(elec_sum), float(lj_sum)


def enrich_cache_with_14(system, sys_cache, scee0: float = 1.2, scnb0: float = 2.0):
    """Add unscaled 1–4 energies and strip scaled 1–4 from base_kcal."""
    import numpy as np

    for iprof, prof in enumerate(system.profiles):
        elec = []
        vdw = []
        base = np.asarray(sys_cache["profiles"][iprof]["base_kcal"], dtype=float).copy()
        for igeom, struct in enumerate(prof.losll.structs):
            atoms = struct.GetASEAtoms()
            e14, v14 = _pair14_unscaled_kcal(system.mol, atoms.get_positions())
            elec.append(e14)
            vdw.append(v14)
            # Remove currently scaled 1–4 contribution from the base.
            base[igeom] -= e14 / float(scee0) + v14 / float(scnb0)
        sys_cache["profiles"][iprof]["base_kcal"] = base
        sys_cache["profiles"][iprof]["elec14"] = np.asarray(elec, dtype=float)
        sys_cache["profiles"][iprof]["vdw14"] = np.asarray(vdw, dtype=float)
        sys_cache["profiles"][iprof]["scee0"] = float(scee0)
        sys_cache["profiles"][iprof]["scnb0"] = float(scnb0)
    return sys_cache


def ll_energies_extended_kcal(system, sys_cache, finp):
    """Fixed-geometry LL energies with optional scee/scnb + analytical torsions."""
    import numpy as np
    from ffpopt.dihed.Dihedrals import _analytical_fitted_torsion_kcal

    scee = float(getattr(finp, "scee", 1.2))
    scnb = float(getattr(finp, "scnb", 2.0))
    out = []
    for iprof, prof_cache in enumerate(sys_cache["profiles"]):
        base = np.asarray(prof_cache["base_kcal"], dtype=float)
        angs = prof_cache["angles"]
        ll = np.empty(len(base), dtype=float)
        elec = prof_cache.get("elec14")
        vdw = prof_cache.get("vdw14")
        for igeom in range(len(base)):
            e = base[igeom] + _analytical_fitted_torsion_kcal(system, angs[igeom])
            if elec is not None and getattr(finp, "opt_scee_scnb", False):
                e += float(elec[igeom]) / scee + float(vdw[igeom]) / scnb
            ll[igeom] = e
        out.append(ll)
    return out


def shape_match_chi2_extended(finp, caches) -> float:
    """Shape-match χ² over all systems/profiles using extended LL energies."""
    import numpy as np
    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT
    from ffpopt.dihed.DihedMath import shape_match_delta

    kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    chisq = 0.0
    for isys, s in enumerate(finp.systems):
        ll_by_prof = ll_energies_extended_kcal(s, caches[isys], finp)
        for iprof, prof in enumerate(s.profiles):
            llene = np.asarray(ll_by_prof[iprof], dtype=float)
            hlene = np.array(
                [struct.data["energy"] * kcal_per_ev for struct in prof.loshl],
                dtype=float,
            )
            d = shape_match_delta(hlene, llene)
            chisq += float(np.dot(d, d))
    return chisq


def objective_extended(x, finp, caches):
    set_extended_params(finp, x)
    return shape_match_chi2_extended(finp, caches)


def _jax_objective_factory(finp, caches):
    """Build a JAX-friendly objective (torsion + optional 1–4; phases continuous)."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_backend=jax requires jax; install with pip install 'ligandparam[jax]'"
        ) from exc

    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

    kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    opt_phase = bool(getattr(finp, "opt_phase", False))
    opt_periods = bool(getattr(finp, "opt_periods", False))
    opt_scee = bool(getattr(finp, "opt_scee_scnb", False))
    pnames = list(finp.ptypedict.keys())
    # Fixed periods when not optimizing them (JAX needs concrete values).
    fixed_per = {
        pname: [float(p.per) for p in finp.ptypedict[pname].dfcns.prims]
        for pname in pnames
    }

    packs = []
    for isys, s in enumerate(finp.systems):
        for iprof, prof in enumerate(s.profiles):
            cache = caches[isys]["profiles"][iprof]
            base = jnp.asarray(cache["base_kcal"], dtype=jnp.float64)
            hl = jnp.asarray(
                [float(st.data["energy"]) * kcal_per_ev for st in prof.loshl],
                dtype=jnp.float64,
            )
            ang = cache["angles"]
            inst_series = []
            for ipinst, pinst in enumerate(s.pinstances):
                dihed_angs = []
                for idihed in range(len(pinst.dihedidxs)):
                    dihed_angs.append(
                        jnp.asarray(
                            [ang[igeom][ipinst][idihed] for igeom in range(len(base))],
                            dtype=jnp.float64,
                        )
                    )
                inst_series.append(
                    {
                        "pname": pinst.ptype.name,
                        "nprim": pinst.ptype.nprim,
                        "angs": dihed_angs,
                    }
                )
            packs.append(
                {
                    "base": base,
                    "hl": hl,
                    "inst": inst_series,
                    "elec": jnp.asarray(cache["elec14"], dtype=jnp.float64)
                    if opt_scee and "elec14" in cache
                    else None,
                    "vdw": jnp.asarray(cache["vdw14"], dtype=jnp.float64)
                    if opt_scee and "vdw14" in cache
                    else None,
                }
            )

    def _unpack(x):
        ip = 0
        out = {}
        for pname in pnames:
            nprim = finp.ptypedict[pname].nprim
            fcs = x[ip : ip + nprim]
            ip += nprim
            phases = None
            periods = None
            if opt_phase:
                phases = x[ip : ip + nprim]
                ip += nprim
            if opt_periods:
                periods = x[ip : ip + nprim]
                ip += nprim
            out[pname] = (fcs, phases, periods)
        scee = x[ip] if opt_scee else 1.2
        scnb = x[ip + 1] if opt_scee else 2.0
        return out, scee, scnb

    def obj(x):
        params, scee, scnb = _unpack(x)
        chisq = 0.0
        for pack in packs:
            ll = pack["base"]
            for inst in pack["inst"]:
                fcs, phases, periods = params[inst["pname"]]
                for angs in inst["angs"]:
                    for iprim in range(inst["nprim"]):
                        per = (
                            periods[iprim]
                            if periods is not None
                            else fixed_per[inst["pname"]][iprim]
                        )
                        phase = phases[iprim] if phases is not None else 0.0
                        a = (per * angs + phase) * (jnp.pi / 180.0)
                        ll = ll + fcs[iprim] * (1.0 + jnp.cos(a))
            if pack["elec"] is not None:
                ll = ll + pack["elec"] / scee + pack["vdw"] / scnb
            d = (pack["hl"] - ll) - jnp.mean(pack["hl"] - ll)
            chisq = chisq + jnp.dot(d, d)
        return chisq

    return obj, jax.grad(obj)


def format_extended_params(finp) -> list[str]:
    """ASCII lines describing the current extended parameter vector."""
    lines: list[str] = []
    for pname in finp.ptypedict:
        dfcns = finp.ptypedict[pname].dfcns
        fcs = [f"{float(p.fc):.4f}" for p in dfcns.prims]
        bits = [f"FC=[{', '.join(fcs)}]"]
        if getattr(finp, "opt_phase", False):
            ph = [f"{float(p.phase):.2f}" for p in dfcns.prims]
            bits.append(f"phase_deg=[{', '.join(ph)}]")
        if getattr(finp, "opt_periods", False):
            pe = [str(int(p.per)) for p in dfcns.prims]
            bits.append(f"period=[{', '.join(pe)}]")
        lines.append(f"  {pname}: " + " ".join(bits))
    if getattr(finp, "opt_scee_scnb", False):
        lines.append(
            f"  scee={float(getattr(finp, 'scee', 1.2)):.4f} "
            f"scnb={float(getattr(finp, 'scnb', 2.0)):.4f}"
        )
    return lines


def solve_extended_lbfgsb(args, finp, caches):
    """Run SciPy (or JAX-jac) L-BFGS-B on the extended objective."""
    from scipy.optimize import minimize
    import numpy as np

    from ffpopt.affdo.AffdoLog import print_affdo

    x0 = get_extended_params(finp)
    bounds = extended_bounds(finp, x0)
    backend = str(getattr(finp, "fit_backend", "lbfgsb"))
    nparam = int(x0.size)
    chi0 = float(objective_extended(x0, finp, caches))
    print_affdo(
        f"extended fit start: mode={getattr(finp, 'fit_mode', '?')} "
        f"backend={backend} nparam={nparam} "
        f"opt_phase={bool(getattr(finp, 'opt_phase', False))} "
        f"opt_periods={bool(getattr(finp, 'opt_periods', False))} "
        f"opt_scee_scnb={bool(getattr(finp, 'opt_scee_scnb', False))} "
        f"chi^2={chi0:.6e} ftol={getattr(args, 'nltol', 0.01)} "
        f"maxiter={getattr(args, 'nlmaxiter', 300)}"
    )
    for line in format_extended_params(finp):
        print_affdo("x0 " + line.lstrip())

    eval_state = {"n": 0}

    def _maybe_log_eval(val: float) -> None:
        eval_state["n"] += 1
        n = eval_state["n"]
        if n == 1 or n % 10 == 0:
            print_affdo(f"L-BFGS-B eval {n}: chi^2={val:.6e}")

    if backend == "jax":
        print_affdo("building JAX objective + autodiff Jacobian (first call may be slow)")
        obj_jax, grad_jax = _jax_objective_factory(finp, caches)
        print_affdo("JAX objective ready")

        def fun(x):
            import jax.numpy as jnp

            val = float(obj_jax(jnp.asarray(x, dtype=jnp.float64)))
            _maybe_log_eval(val)
            return val

        def jac(x):
            import jax.numpy as jnp

            g = grad_jax(jnp.asarray(x, dtype=jnp.float64))
            return np.asarray(g, dtype=float)

        res = minimize(
            fun,
            x0,
            method="L-BFGS-B",
            jac=jac,
            bounds=bounds,
            options={
                "ftol": getattr(args, "nltol", 0.01),
                "maxiter": getattr(args, "nlmaxiter", 300),
                "disp": False,
            },
        )
    else:
        def fun(x):
            val = objective_extended(x, finp, caches)
            _maybe_log_eval(float(val))
            return val

        res = minimize(
            fun,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": getattr(args, "nltol", 0.01),
                "maxiter": getattr(args, "nlmaxiter", 300),
                "disp": False,
            },
        )

    set_extended_params(finp, res.x)
    chi1 = float(res.fun)
    print_affdo(
        f"extended L-BFGS-B ({backend}): success={bool(res.success)} "
        f"nit={getattr(res, 'nit', '?')} nfev={getattr(res, 'nfev', eval_state['n'])} "
        f"chi^2={chi1:.6e} (start {chi0:.6e}) msg={res.message}"
    )
    for line in format_extended_params(finp):
        print_affdo("x* " + line.lstrip())
    print(
        f"[ffpopt] extended L-BFGS-B ({backend}): success={bool(res.success)} "
        f"nit={getattr(res, 'nit', '?')} fun={chi1:.6e} msg={res.message}"
    )
    return res
