"""Stdout / logger helpers for AFFDO-style extras.

Messages use a leading ``[affdo]`` scope so the hierarchical console
formatter peels it into ``[ffpopt] [affdo]``.
Keep bodies ASCII-only for latin-1 Slurm ``.out`` files.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence


def print_affdo(msg: str, *, flush: bool = True) -> None:
    """Print one AFFDO line to stdout (wavefront / GenDihedFit subprocesses)."""
    from ffpopt.runtime.Console import ascii_for_stdio

    print(ascii_for_stdio(f"[affdo] {msg}"), flush=flush)


def log_affdo(log: logging.Logger, msg: str, *args) -> None:
    """``log.info`` with a leading ``[affdo]`` scope token."""
    log.info("[affdo] " + msg, *args)


def describe_affdo_extras(
    *,
    whole_ligand: bool = False,
    multi_centroid: int = 0,
    boltzmann_charges: bool = False,
    soft_dihed_restraint: bool = False,
    soft_dihed_k: Optional[float] = None,
    soft_dihed_kmax: Optional[float] = None,
    soft_dihed_tol: Optional[float] = None,
    fit_cli_args: Optional[Sequence[str]] = None,
) -> str:
    """Compact one-line summary of opt-in AFFDO knobs."""
    n_cent = int(multi_centroid or 0)
    parts = [
        f"whole_ligand={bool(whole_ligand)}",
        f"multi_centroid={n_cent}",
        f"boltzmann_charges={bool(boltzmann_charges)}",
        f"soft_dihed_restraint={bool(soft_dihed_restraint)}",
    ]
    if soft_dihed_restraint:
        k = 500.0 if soft_dihed_k is None else float(soft_dihed_k)
        kmax = 8000.0 if soft_dihed_kmax is None else float(soft_dihed_kmax)
        tol = 0.5 if soft_dihed_tol is None else float(soft_dihed_tol)
        parts.append(f"k={k:g} kcal/mol/rad^2")
        parts.append(f"kmax={kmax:g}")
        parts.append(f"tol={tol:g} deg")
    extra = [str(x) for x in (fit_cli_args or []) if str(x).strip()]
    parts.append("fit_flags=" + (" ".join(extra) if extra else "(barrier / default)"))
    return " ".join(parts)


def format_boltzmann_summary(info: dict) -> list[str]:
    """Human-readable lines for a :func:`boltzmann_average_mol2_charges` result."""
    lines: list[str] = []
    n = len(info.get("weights") or [])
    t = info.get("T", 298.15)
    lines.append(
        f"Boltzmann-averaged {info.get('n_atom', '?')} atom charges over "
        f"{n} centroid(s) at T={float(t):g} K -> {info.get('out_mol2', '?')}"
    )
    w = info.get("weights") or []
    if w:
        wtxt = " ".join(f"{float(x):.4f}" for x in w)
        lines.append(f"weights: {wtxt}")
    if info.get("equal_weights"):
        lines.append(
            "equal weights (no per-centroid energies); average is unweighted"
        )
    rms = info.get("rms_vs_first")
    mx = info.get("max_abs_dq")
    if rms is not None:
        lines.append(f"RMS |dq| vs first centroid = {float(rms):.5f} e")
    if mx is not None:
        lines.append(f"max |dq| vs first centroid = {float(mx):.5f} e")
    if info.get("out_lib"):
        lines.append(f"updated lib charges -> {info['out_lib']}")
    return lines
