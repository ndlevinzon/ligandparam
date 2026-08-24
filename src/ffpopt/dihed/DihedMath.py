"""Pure dihedral-scan math helpers (shape match, angle keys, profile align).

Kept separate from the large :mod:`ffpopt.dihed.Dihedrals` orchestration module so
tests and callers can import fit-math without pulling solvers. Public names
are re-exported from :mod:`ffpopt.dihed.Dihedrals` for API stability.
"""

from __future__ import annotations

import re

import numpy as np


def shape_match_delta(hlene, llene):
    """Mean-centered HL−LL residual (free vertical offset; shape match only)."""
    d = np.asarray(hlene, dtype=float) - np.asarray(llene, dtype=float)
    return d - np.mean(d)


def AngularStdDev(angs):
    """Circular standard deviation of angles in degrees."""
    from scipy.stats import circstd

    rads = np.deg2rad(np.asarray(angs, dtype=float))
    return float(np.rad2deg(circstd(rads)))


def normalize_scan_angle(ang: float) -> float:
    """Map angle to ``[0, 360)`` with 360 collapsed to 0."""
    a = float(ang) % 360.0
    if abs(a - 360.0) < 1.0e-6 or abs(a) < 1.0e-6:
        return 0.0
    return a


# Back-compat private name used inside Dihedrals historically.
_normalize_scan_angle = normalize_scan_angle


def struct_scan_angle(struct) -> float | None:
    """Best-effort scan-angle key for a wavefront JSON frame.

    Prefers ``data['name']`` like ``d030``, then a dihedral constraint value.
    """
    data = getattr(struct, "data", None) or {}
    name = str(data.get("name") or "").strip()
    m = re.match(r"^d(\d+(?:\.\d+)?)$", name, flags=re.IGNORECASE)
    if m:
        return normalize_scan_angle(float(m.group(1)))

    cons = data.get("constraints") or []
    for c in cons:
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("type") or "").lower()
        if ctype in ("dihedral", "torsion") and "value" in c:
            return normalize_scan_angle(float(c["value"]))

    clist = getattr(struct, "constraints", None)
    if clist is not None:
        try:
            for c in clist:
                ctype = str(getattr(c, "type", getattr(c, "ctype", "")) or "").lower()
                if "dihed" in ctype or "torsion" in ctype:
                    return normalize_scan_angle(float(c.value))
        except Exception:
            pass
    return None


def angle_map_from_los(los):
    """Return ``{angle: struct}``; raise if any frame lacks an angle key."""
    out = {}
    missing = []
    for i, struct in enumerate(los.structs):
        ang = struct_scan_angle(struct)
        if ang is None:
            missing.append(i)
            continue
        out.setdefault(ang, struct)
    if missing:
        raise ValueError(
            f"Could not determine scan angle for structure index(es) {missing}; "
            "expected names like 'd030' or dihedral constraints on each frame."
        )
    return out


_angle_map_from_los = angle_map_from_los


def _looks_like_full_scan(n: int) -> bool:
    """True for a uniform 360/n grid with enough points to interpolate."""
    return int(n) >= 12 and 360 % int(n) == 0


def _circular_delta(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def _frame_energy(struct) -> float:
    data = getattr(struct, "data", None) or {}
    e = data.get("energy")
    if e is None:
        raise ValueError("scan frame missing energy; cannot interpolate profiles")
    return float(e)


def _periodic_interp(src_angles, src_values, tgt_angles):
    """Linear interpolation of a periodic (0–360 deg) 1-D profile."""
    a = np.asarray(src_angles, dtype=float)
    v = np.asarray(src_values, dtype=float)
    order = np.argsort(a)
    a, v = a[order], v[order]
    a_ext = np.concatenate([a[-1:] - 360.0, a, a[:1] + 360.0])
    v_ext = np.concatenate([v[-1:], v, v[:1]])
    tgt = np.asarray(
        [normalize_scan_angle(x) for x in tgt_angles], dtype=float
    )
    return np.interp(tgt, a_ext, v_ext)


def _clone_frame_with_energy(src, energy, angle):
    import copy

    clone = copy.deepcopy(src)
    if hasattr(clone, "Update"):
        clone.Update(
            energy, clone.data.get("positions"), clone.data.get("forces")
        )
    else:
        clone.data["energy"] = energy
    clone.data["name"] = f"d{int(round(normalize_scan_angle(angle))):03d}"
    return clone


def _align_hl_interpolated_onto_ll(loshl, losll, hl_map, ll_map, common, hl_only, ll_only):
    """Keep every LL geometry; fill HL energies by periodic interpolation.

    Isolated / nonlinear fits re-evaluate MM at LL geometries, so those
    frames must stay at their scanned angles. Interpolating the (usually
    coarser) HL QM curve onto the LL grid is the safe mismatch fallback.
    """
    from ffpopt.Struct import ListOfStruct

    tgt = sorted(ll_map)
    hl_angs = sorted(hl_map)
    hl_enes = [_frame_energy(hl_map[a]) for a in hl_angs]
    interp_e = _periodic_interp(hl_angs, hl_enes, tgt)
    hl_structs = []
    for ang, energy in zip(tgt, interp_e):
        if ang in hl_map:
            hl_structs.append(hl_map[ang])
            continue
        nearest = min(hl_angs, key=lambda x, a=ang: _circular_delta(x, a))
        hl_structs.append(
            _clone_frame_with_energy(hl_map[nearest], float(energy), ang)
        )
    ll_structs = [ll_map[a] for a in tgt]
    new_hl = ListOfStruct.from_structs_shared(
        hl_structs, args=getattr(loshl, "args", None)
    )
    new_ll = ListOfStruct.from_structs_shared(
        ll_structs, args=getattr(losll, "args", None)
    )
    info = {
        "n_common": len(tgt),
        "angles": tgt,
        "hl_only": hl_only,
        "ll_only": ll_only,
        "interpolated": True,
        "interp_onto": "ll",
        "n_exact": len(common),
    }
    return new_hl, new_ll, info


def align_scan_profiles(loshl, losll, *, hl_path="", ll_path="", min_points=3):
    """Align HL/LL frames onto the same scan angles (sorted).

    Matching angles are kept as-is. When both sides look like full 360/n
    scans on different grids (e.g. leftover ``--fast`` 15 deg HL vs 10 deg
    orig), HL energies are interpolated onto the LL angles so MM geometries
    stay exact. Otherwise the intersection is used.

    Returns aligned ``ListOfStruct`` objects and a small diagnostic dict.
    """
    from ffpopt.Struct import ListOfStruct

    hl_map = angle_map_from_los(loshl)
    ll_map = angle_map_from_los(losll)
    common = sorted(set(hl_map) & set(ll_map))
    hl_only = sorted(set(hl_map) - set(ll_map))
    ll_only = sorted(set(ll_map) - set(hl_map))

    if (hl_only or ll_only) and _looks_like_full_scan(len(hl_map)) and _looks_like_full_scan(
        len(ll_map)
    ):
        return _align_hl_interpolated_onto_ll(
            loshl, losll, hl_map, ll_map, common, hl_only, ll_only
        )

    if len(common) < int(min_points):
        raise Exception(
            f"Structure count mismatch in {hl_path or 'HL'} and {ll_path or 'LL'} "
            f"({len(loshl)} vs {len(losll)}); after angle alignment only "
            f"{len(common)} shared points remain (need >= {min_points}). "
            f"HL-only angles={hl_only[:12]}{'...' if len(hl_only) > 12 else ''}; "
            f"LL-only angles={ll_only[:12]}{'...' if len(ll_only) > 12 else ''}."
        )

    hl_structs = [hl_map[a] for a in common]
    ll_structs = [ll_map[a] for a in common]
    new_hl = ListOfStruct.from_structs_shared(
        hl_structs, args=getattr(loshl, "args", None)
    )
    new_ll = ListOfStruct.from_structs_shared(
        ll_structs, args=getattr(losll, "args", None)
    )
    info = {
        "n_common": len(common),
        "angles": common,
        "hl_only": hl_only,
        "ll_only": ll_only,
        "interpolated": False,
    }
    return new_hl, new_ll, info
