"""Ridge / truncated-SVD Fourier FC solve, energy-domain barrier, nprim AIC.

Unbounded least squares on ``K_n (1 + cos(n phi))`` is ill-posed when the
scan is gappy or the residual is not a low-order torsion: huge cancelling
harmonics fit the samples. This module picks the unique small-K series
(Tikhonov / truncated SVD), constrains the reconstructed V(phi) barrier,
and optionally drops unused periodicities by AIC. After AIC, a chemical
group table zeros or caps remaining V(phi) on sensitive rotors (alkane,
sulfate/phosphate, alcohol/ether, amine, generic sp3). Unsaturated
(amide) types keep the 30 kcal ceiling. ``FFPOPT_DIHED_FC_MAX`` is
applied last as an Amber-safety valve, not as the model.
"""

from __future__ import annotations

import math
from typing import Optional


def nprim_select_enabled() -> bool:
    from ffpopt.runtime.EnvDefaults import env_bool

    return env_bool("FFPOPT_DIHED_NPRIM_SELECT")


def svd_rel_cutoff() -> float:
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_SVD_REL", 1.0e-4)))


def ridge_lambda0() -> float:
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_RIDGE_LAMBDA", 0.0)))


def barrier_alpha() -> float:
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_BARRIER_ALPHA", 2.0)))


def barrier_abs_kcal() -> float:
    """Dense-grid V(phi) peak-to-peak ceiling (kcal/mol). 0 disables."""
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_BARRIER_ABS", 30.0)))


def aic_window() -> float:
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_AIC_WINDOW", 2.0)))


def sp3_barrier_max() -> float:
    """sp3-sp3 V(phi) ptp above this (kcal) is rejected (K=0). 0 disables."""
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_SP3_BARRIER_MAX", 20.0)))


def alkane_barrier_max() -> float:
    """Alkane-like V(phi) ptp cap (kcal). 0 disables. Used only below reject."""
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_ALKANE_BARRIER_MAX", 5.0)))


def polar_sp3_barrier_max() -> float:
    """Cap (kcal) for C-OH / C-OS / C-N3 / C-N4 / thioether rotors. 0 disables."""
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_POLAR_SP3_BARRIER_MAX", 8.0)))


def sulfate_barrier_max() -> float:
    """Reject (K=0) sulfate/phosphate Vptp above this (kcal). 0 disables."""
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_SULFATE_BARRIER_MAX", 10.0)))


def sulfate_barrier_cap() -> float:
    """Cap (kcal) for sulfate/phosphate below the reject threshold. 0 disables."""
    from ffpopt.runtime.EnvDefaults import env_float

    return max(0.0, float(env_float("FFPOPT_DIHED_SULFATE_BARRIER_CAP", 4.0)))


# GAFF / GAFF2 tetrahedral heavy atoms (central-bond test).
_SP3_HEAVY = frozenset(
    {
        "c3",
        "cx",
        "cy",
        "n3",
        "n4",
        "oh",
        "os",
        "s",
        "sh",
        "ss",
        "s4",
        "s6",
        "p3",
        "p4",
        "p5",
    }
)
_ALKANE_ATOM = frozenset({"c3", "cx", "cy", "hc", "h1", "h2", "h3", "hx"})
_SULFATE_P = frozenset({"s4", "s6", "p4", "p5"})
_ALCOHOL_ETHER = frozenset({"oh", "os"})
_AMINE_AMMONIUM = frozenset({"n3", "n4"})
_THIO_SP3 = frozenset({"s", "sh", "ss"})


def parse_dihed_type_key(name: str) -> tuple[str, str, str, str] | None:
    """Four Amber types from a DIHE key such as ``c3-c3-s6-o`` or ``c -ns-c3-c3``.

    Fit-input names are ``{res}_{t1}-{t2}-{t3}-{t4}`` (e.g. ``CHA_c3-c3-c3-h1``).
    The residue prefix must be stripped or the extra token makes this look
    like five types and the chemical-group policy classifies the rotor as
    unsaturated (caps never fire).
    """
    import re

    text = str(name or "").strip()
    if "_" in text:
        text = text.rsplit("_", 1)[-1]
    parts = re.findall(r"[A-Za-z][A-Za-z0-9]?", text)
    if len(parts) != 4:
        return None
    return tuple(p.lower() for p in parts)


def classify_dihed_rotor(name: str) -> str:
    """Chemical class from the four Amber types (central bond is types 2-3).

    ``alkane``, ``sulfate_phosphate``, ``alcohol_ether``, ``amine_ammonium``,
    ``polar_sp3`` (thioether), ``sp3_sp3``, or ``unsaturated``.
    """
    types = parse_dihed_type_key(name)
    if types is None:
        return "unsaturated"
    _a, b, c, _d = types
    if not (b in _SP3_HEAVY and c in _SP3_HEAVY):
        return "unsaturated"
    if set(types) <= _ALKANE_ATOM:
        return "alkane"
    if b in _SULFATE_P or c in _SULFATE_P:
        return "sulfate_phosphate"
    if b in _ALCOHOL_ETHER or c in _ALCOHOL_ETHER:
        return "alcohol_ether"
    if b in _AMINE_AMMONIUM or c in _AMINE_AMMONIUM:
        return "amine_ammonium"
    if b in _THIO_SP3 or c in _THIO_SP3:
        return "polar_sp3"
    return "sp3_sp3"


def chemical_barrier_limits(kind: str) -> tuple[float, float]:
    """Return ``(cap_kcal, reject_kcal)``; 0 means that rung is off."""
    sp3_rej = sp3_barrier_max()
    if kind == "unsaturated":
        return 0.0, 0.0
    if kind == "alkane":
        return alkane_barrier_max(), sp3_rej
    if kind == "sulfate_phosphate":
        return sulfate_barrier_cap(), sulfate_barrier_max()
    if kind in {"alcohol_ether", "amine_ammonium", "polar_sp3"}:
        return polar_sp3_barrier_max(), sp3_rej
    return 0.0, sp3_rej


def _zero_dfcn(dfcn):
    import copy
    from ffpopt.dihed.DihedFourier import GetDihedClasses

    idxs = list(getattr(dfcn, "idxs", None) or [0, 1, 2, 3])
    out = copy.deepcopy(GetDihedClasses(idxs=idxs)[1][0])
    out.SetFCs([0.0])
    return out


def apply_sp3_rotor_policy(dfcn, pname: str, *, where: str = "sp3"):
    """Zero or cap a fitted series using the chemical-group table.

    After AIC, a residual that still needs a large V(phi) on a sensitive
    rotor is not a torsion: set K=0 (keep GAFF) or scale down to a
    group-specific cap. Amide / sulfonamide (unsaturated) keep the global
    30 kcal ceiling. ``FFPOPT_DIHED_FC_MAX`` is unchanged.

    Returns ``(dfcn, action, ptp)`` with action ``keep``, ``zero_<kind>``,
    or ``cap_<kind>``.
    """
    import numpy as np

    if dfcn is None or not getattr(dfcn, "prims", None):
        return dfcn, "keep", 0.0
    ptp = dense_torsion_ptp(dfcn)
    kind = classify_dihed_rotor(pname)
    cap, reject = chemical_barrier_limits(kind)
    label = pname or "dihed"

    if kind == "unsaturated":
        return dfcn, "keep", ptp

    if reject > 0.0 and ptp > reject * 1.001:
        dfcn = _zero_dfcn(dfcn)
        print(
            f"[ffpopt] {kind} rotor {label}: Vptp={ptp:.3g} kcal > {reject:g}; "
            f"K=0 (keep GAFF) at {where}"
        )
        return dfcn, f"zero_{kind}", ptp

    if cap > 0.0 and ptp > cap * 1.001:
        fcs = np.array([float(p.fc) for p in dfcn.prims], dtype=float)
        fcs = _scale_to_ptp_limit(fcs, ptp, cap)
        dfcn.SetFCs(fcs)
        print(
            f"[ffpopt] {kind} rotor {label}: Vptp={ptp:.3g} -> {cap:g} kcal "
            f"at {where}"
        )
        return dfcn, f"cap_{kind}", cap

    return dfcn, "keep", ptp


apply_chemical_rotor_policy = apply_sp3_rotor_policy


def peak_to_peak(v) -> float:
    import numpy as np

    arr = np.asarray(v, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.max(arr) - np.min(arr))


def aic_from_rss(rss: float, npts: int, nparam: int, *, eps: float = 1.0e-12) -> float:
    """Gaussian AIC on residual sum of squares (up to a common constant)."""
    npts = max(int(npts), 1)
    rss = max(float(rss), 0.0)
    return npts * math.log(rss / npts + eps) + 2.0 * int(nparam)


def dense_torsion_angles(n: int = 361):
    import numpy as np

    return np.linspace(0.0, 360.0, int(n))


def dense_torsion_ptp(dfcn) -> float:
    """Peak-to-peak of Amber V(phi) = sum K_n (1 + cos(n phi - gamma)) (kcal)."""
    return peak_to_peak(dfcn.CptEne(dense_torsion_angles()))


def sample_barrier_limit_kcal(y) -> float:
    """Allowed reconstructed shape ptp (kcal): alpha * data ptp, then abs cap."""
    rel = barrier_alpha() * peak_to_peak(y)
    abs_cap = barrier_abs_kcal()
    if rel <= 0.0:
        return abs_cap
    if abs_cap <= 0.0:
        return rel
    return min(rel, abs_cap)


def tikhonov_svd_solve(A, y, *, lam: float = 0.0, rel_cutoff: Optional[float] = None):
    """Min ||A x - y||^2 + lam ||x||^2 with SVD modes below ``rel_cutoff`` dropped.

    Returns
    -------
    x, info
        ``info`` has rank, cond, n_kept, n_modes, singular, lambda.
    """
    import numpy as np

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    npts, nparam = A.shape
    if nparam == 0:
        return np.zeros(0, dtype=float), {
            "rank": 0,
            "cond": 0.0,
            "n_kept": 0,
            "n_modes": 0,
            "singular": np.zeros(0),
            "lambda": float(lam),
        }
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    cutoff = svd_rel_cutoff() if rel_cutoff is None else max(0.0, float(rel_cutoff))
    s_max = float(s[0]) if s.size else 1.0
    keep = s >= (cutoff * max(s_max, 1.0e-30))
    if not np.any(keep):
        keep = np.zeros_like(s, dtype=bool)
        if s.size:
            keep[0] = True
    s_safe = np.where(s > 1.0e-30, s, 1.0)
    lam = max(0.0, float(lam))
    filt = (s * s) / (s * s + lam) if lam > 0.0 else np.ones_like(s)
    filt = np.where(keep, filt, 0.0)
    coef = filt * ((U.T @ y) / s_safe)
    coef = np.where(keep, coef, 0.0)
    x = Vt.T @ coef
    n_kept = int(np.sum(keep))
    kept_s = s[keep]
    cond = (
        float(kept_s[0] / kept_s[-1])
        if n_kept and kept_s[-1] > 0.0
        else float("inf")
    )
    info = {
        "rank": n_kept,
        "cond": cond,
        "n_kept": n_kept,
        "n_modes": int(s.size),
        "singular": s,
        "lambda": lam,
        "npts": npts,
        "nparam": nparam,
    }
    return x, info


def _scale_to_ptp_limit(x, ptp: float, limit: float):
    import numpy as np

    if ptp <= 1.0e-12 or limit <= 0.0 or ptp <= limit:
        return np.asarray(x, dtype=float)
    return np.asarray(x, dtype=float) * (limit / ptp)


def solve_regularized_fcs(A, y, *, dfcn=None, where: str = "ridge"):
    """Tikhonov / truncated SVD, then energy-domain barrier, then Amber FC valve.

    Parameters
    ----------
    A, y
        Mean-centered design matrix and target (kcal).
    dfcn
        Optional ``MultiDihedFcn`` used to evaluate V(phi) on a dense grid.
        Mutated via ``SetFCs`` when provided.
    """
    import numpy as np
    from ffpopt.dihed.DihedFitSolve import clip_dihed_fcs

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    sample_limit = sample_barrier_limit_kcal(y)
    dense_limit = sample_limit
    abs_cap = barrier_abs_kcal()
    lam = ridge_lambda0()
    x = np.zeros(A.shape[1], dtype=float)
    info = {}
    scaled = False
    for _ in range(24):
        x, info = tikhonov_svd_solve(A, y, lam=lam)
        sample_ptp = peak_to_peak(A @ x)
        dense_ptp = 0.0
        if dfcn is not None and x.size:
            dfcn.SetFCs(x)
            dense_ptp = dense_torsion_ptp(dfcn)
        over = False
        if sample_limit > 0.0 and sample_ptp > sample_limit * 1.001:
            over = True
        if dfcn is not None and dense_limit > 0.0 and dense_ptp > dense_limit * 1.001:
            over = True
        if dfcn is not None and abs_cap > 0.0 and dense_ptp > abs_cap * 1.001:
            over = True
        if not over:
            break
        lam = 1.0e-8 if lam <= 0.0 else lam * 10.0
        if lam > 1.0e8:
            limit = abs_cap if (abs_cap > 0.0 and dense_ptp > abs_cap) else sample_limit
            ptp = dense_ptp if (dfcn is not None and dense_ptp > 0.0) else sample_ptp
            x = _scale_to_ptp_limit(x, ptp, limit)
            scaled = True
            if dfcn is not None and x.size:
                dfcn.SetFCs(x)
            break
    x = clip_dihed_fcs(x, where=where)
    if dfcn is not None and x.size:
        dfcn.SetFCs(x)
    info = dict(info)
    info["sample_ptp"] = peak_to_peak(A @ x) if x.size else 0.0
    info["dense_ptp"] = dense_torsion_ptp(dfcn) if dfcn is not None else None
    info["sample_limit"] = sample_limit
    info["lambda"] = lam
    info["scaled"] = scaled
    info["residuals"] = np.asarray(y - (A @ x) if x.size else y, dtype=float)
    print(
        f"[ffpopt] Fourier ridge at {where}: kept {info.get('n_kept', 0)}/"
        f"{info.get('n_modes', 0)} SVD modes, lambda={lam:.3g}, "
        f"cond~={float(info.get('cond') or 0):.3e}, "
        f"Vptp(sample)={info['sample_ptp']:.3g} kcal "
        f"(limit {sample_limit:.3g})"
        + (
            f", Vptp(dense)={info['dense_ptp']:.3g}"
            if info["dense_ptp"] is not None
            else ""
        )
    )
    return x, info


def fit_fourier_nprim(angs, y, max_nprim: int, idxs, *, pname: str = ""):
    """Fit nested n=1..max_nprim series; pick smallest k in the AIC window.

    ``y`` must already be mean-centered. Returns ``(dfcn, x, info)``.
    ``k=0`` (no torsion) wins if it is in the AIC window; then a 1-term
    zero series is returned so the ParamType still has a shell.
    """
    import copy
    import numpy as np
    from ffpopt.dihed.DihedFourier import GetDihedClasses

    angs = np.asarray(angs, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    max_nprim = max(1, int(max_nprim))
    npts = int(y.size)
    label = pname or "dihed"
    classes = GetDihedClasses(idxs=idxs)

    candidates = []
    rss0 = float(np.dot(y, y))
    candidates.append((aic_from_rss(rss0, npts, 0), 0, None, np.zeros(0), rss0, None))

    for k in range(1, max_nprim + 1):
        if npts < k:
            break
        pool = classes.get(k) or []
        if not pool:
            continue
        best_k = None
        for dfcn0 in pool:
            dfcn = copy.deepcopy(dfcn0)
            A = np.column_stack([p.CptEterm(angs) for p in dfcn.prims])
            A_c = A - np.mean(A, axis=0, keepdims=True)
            x, info = solve_regularized_fcs(
                A_c, y, dfcn=dfcn, where=f"IsolatedLinearSolve({label}) nprim={k}"
            )
            rss = float(np.dot(y - A_c @ x, y - A_c @ x))
            aic = aic_from_rss(rss, npts, k)
            row = (aic, k, dfcn, x, rss, info)
            if best_k is None or rss < best_k[4]:
                best_k = row
        if best_k is not None:
            candidates.append(best_k)

    if nprim_select_enabled():
        min_aic = min(c[0] for c in candidates)
        window = aic_window()
        in_window = [c for c in candidates if c[0] <= min_aic + window]
        aic, k, dfcn, x, rss, info = min(in_window, key=lambda c: (c[1], c[0]))
    else:
        window = aic_window()
        fitted = [c for c in candidates if c[1] > 0]
        aic, k, dfcn, x, rss, info = fitted[-1] if fitted else candidates[0]
    if k == 0 or dfcn is None:
        dfcn = copy.deepcopy(classes[1][0])
        dfcn.SetFCs([0.0])
        x = np.zeros(1, dtype=float)
        k = 1
        print(
            f"[ffpopt] nprim select {label}: AIC prefers no torsion "
            f"(rss0={rss0:.3g}); keeping nprim=1 with K=0"
        )
    else:
        print(
            f"[ffpopt] nprim select {label}: max={max_nprim} -> {k} "
            f"(AIC={aic:.3g}, rss={rss:.3g}, window={window:g})"
        )
        dfcn, action, ptp = apply_sp3_rotor_policy(
            dfcn, label, where=f"nprim {label}"
        )
        x = np.array([float(p.fc) for p in dfcn.prims], dtype=float)
        k = len(dfcn.prims)
        if action != "keep":
            info = dict(info or {})
            info["sp3_action"] = action
            info["sp3_ptp"] = ptp
    out_info = dict(info or {})
    out_info["nprim"] = k
    out_info["aic"] = aic
    out_info["rss"] = rss
    return dfcn, x, out_info


def _assign_ptype_dfcns(finp, pname: str, dfcn) -> None:
    ptype = finp.ptypedict[pname]
    ptype.dfcns = dfcn
    ptype.nprim = len(dfcn.prims)
    for s in finp.systems:
        for pinst in s.pinstances:
            if pinst.ptype.name == pname:
                pinst.ptype.dfcns = dfcn
                pinst.ptype.nprim = ptype.nprim


def enforce_per_type_dense_barriers(finp, *, where: str = "joint dense"):
    """Scale each ParamType to the global V(phi) ceiling, then chemical policy."""
    import numpy as np
    from ffpopt.dihed.DihedFitSolve import clip_dihed_fcs

    abs_cap = barrier_abs_kcal()
    for pname, ptype in finp.ptypedict.items():
        dfcn = ptype.dfcns
        if dfcn is None or not getattr(dfcn, "prims", None):
            continue
        ptp = dense_torsion_ptp(dfcn)
        if abs_cap > 0.0 and ptp > abs_cap * 1.001:
            scale = abs_cap / ptp
            fcs = np.array([float(p.fc) for p in dfcn.prims], dtype=float) * scale
            dfcn.SetFCs(fcs)
            print(
                f"[ffpopt] energy-domain barrier {where}: {pname} Vptp {ptp:.3g} -> "
                f"{abs_cap:g} kcal (scale={scale:.3g})"
            )
        dfcn2, action, _ptp = apply_sp3_rotor_policy(dfcn, pname, where=where)
        if action != "keep":
            _assign_ptype_dfcns(finp, pname, dfcn2)
    x = clip_dihed_fcs(finp.get_params(), where=where)
    finp.set_params(x)
    return x


def apply_nprim_selection_from_caches(finp, caches, kcal_per_ev: float) -> None:
    """Shrink each ParamType's nprim from a representative cache profile."""
    from ffpopt.dihed.DihedMath import AngularStdDev
    import numpy as np

    if not nprim_select_enabled() or caches is None:
        return
    for pname, ptype in list(finp.ptypedict.items()):
        max_nprim = int(ptype.nprim)
        best = None
        best_std = -1.0
        for isys, s in enumerate(finp.systems):
            sys_cache = caches[isys]
            for ipinst, pinst in enumerate(s.pinstances):
                if pinst.ptype.name != pname:
                    continue
                for idihed, idxs in enumerate(pinst.dihedidxs):
                    for iprof, prof in enumerate(s.profiles):
                        rows = sys_cache["profiles"][iprof]
                        angs = [
                            float(rows["angles"][igeom][ipinst][idihed])
                            for igeom in range(len(rows["base_kcal"]))
                        ]
                        if len(angs) < 2:
                            continue
                        astd = AngularStdDev(angs)
                        if astd > best_std:
                            best_std = astd
                            hl = np.array(
                                [
                                    float(st.data["energy"]) * kcal_per_ev
                                    for st in prof.loshl
                                ],
                                dtype=float,
                            )
                            base = np.asarray(rows["base_kcal"], dtype=float)
                            if hl.size != base.size:
                                continue
                            y = hl - base
                            y_c = y - np.mean(y)
                            best = (angs, y_c, list(idxs))
        if best is None:
            continue
        angs, y_c, idxs = best
        dfcn, _x, info = fit_fourier_nprim(
            angs, y_c, max_nprim, idxs, pname=pname
        )
        ptype.dfcns = dfcn
        ptype.nprim = int(info.get("nprim") or len(dfcn.prims))
        for s in finp.systems:
            for pinst in s.pinstances:
                if pinst.ptype.name == pname:
                    pinst.ptype.dfcns = dfcn
                    pinst.ptype.nprim = ptype.nprim
