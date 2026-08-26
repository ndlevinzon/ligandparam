"""Ridge / truncated-SVD Fourier FC solve, energy-domain barrier, nprim AIC.

Unbounded least squares on ``K_n (1 + cos(n phi))`` is ill-posed when the
scan is gappy or the residual is not a low-order torsion: huge cancelling
harmonics fit the samples. This module picks the unique small-K series
(Tikhonov / truncated SVD), constrains the reconstructed V(phi) barrier,
and optionally drops unused periodicities by AIC. ``FFPOPT_DIHED_FC_MAX``
is applied last as an Amber-safety valve, not as the model.
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
    out_info = dict(info or {})
    out_info["nprim"] = k
    out_info["aic"] = aic
    out_info["rss"] = rss
    return dfcn, x, out_info


def enforce_per_type_dense_barriers(finp, *, where: str = "joint dense"):
    """Scale each ParamType's K so dense-grid V(phi) ptp respects the ceiling."""
    import numpy as np
    from ffpopt.dihed.DihedFitSolve import clip_dihed_fcs

    abs_cap = barrier_abs_kcal()
    if abs_cap <= 0.0:
        x = clip_dihed_fcs(finp.get_params(), where=where)
        finp.set_params(x)
        return x
    for pname, ptype in finp.ptypedict.items():
        dfcn = ptype.dfcns
        if dfcn is None or not getattr(dfcn, "prims", None):
            continue
        ptp = dense_torsion_ptp(dfcn)
        if ptp <= abs_cap * 1.001:
            continue
        scale = abs_cap / ptp
        fcs = np.array([float(p.fc) for p in dfcn.prims], dtype=float) * scale
        dfcn.SetFCs(fcs)
        print(
            f"[ffpopt] energy-domain barrier {where}: {pname} Vptp {ptp:.3g} -> "
            f"{abs_cap:g} kcal (scale={scale:.3g})"
        )
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
