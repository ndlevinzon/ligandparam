"""Boltzmann-average atomic charges over conformer centroids.

Optional whole-ligand helper: weight per-centroid charge vectors by
``w_k ∝ exp(-(E_k - E_min) / kT)`` at ``T=298 K`` and write averaged charges
into a mol2 (and optionally update an Amber ``.lib`` via parmed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# kcal/mol Boltzmann constant * 298.15 K
_KT_KCAL = 0.001987204258 * 298.15


def boltzmann_weights(energies_kcal: Sequence[float], *, T: float = 298.15) -> np.ndarray:
    """Normalized Boltzmann weights from relative energies (kcal/mol)."""
    e = np.asarray(energies_kcal, dtype=float)
    if e.size == 0:
        return e
    kt = 0.001987204258 * float(T)
    de = e - np.min(e)
    w = np.exp(-de / kt)
    s = float(np.sum(w))
    if s <= 0.0 or not np.isfinite(s):
        return np.full(e.shape, 1.0 / e.size)
    return w / s


def average_charge_vectors(
    charge_rows: Sequence[Sequence[float]],
    energies_kcal: Sequence[float],
    *,
    T: float = 298.15,
) -> np.ndarray:
    """Boltzmann-average charge vectors (shape ``(n_conf, n_atom)``)."""
    q = np.asarray(charge_rows, dtype=float)
    if q.ndim != 2:
        raise ValueError("charge_rows must be 2-D (n_conf, n_atom)")
    w = boltzmann_weights(energies_kcal, T=T)
    if w.shape[0] != q.shape[0]:
        raise ValueError("energies and charge rows length mismatch")
    return np.sum(q * w[:, None], axis=0)


def read_mol2_charges(mol2_path) -> Tuple[list, list]:
    """Return ``(atom_names, charges)`` from a Tripos mol2 (first molecule)."""
    path = Path(mol2_path)
    names: list[str] = []
    charges: list[float] = []
    in_atom = False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atom = False
                continue
            if not in_atom:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            names.append(parts[1])
            charges.append(float(parts[8]))
    if not charges:
        raise ValueError(f"no ATOM charges found in {path}")
    return names, charges


def write_mol2_charges(src_mol2, dst_mol2, charges: Sequence[float]) -> Path:
    """Rewrite mol2 atom charges; preserve all other fields."""
    src = Path(src_mol2)
    dst = Path(dst_mol2)
    q = list(charges)
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    out = []
    in_atom = False
    iq = 0
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            out.append(line)
            continue
        if line.startswith("@<TRIPOS>"):
            in_atom = False
            out.append(line)
            continue
        if in_atom:
            parts = line.split()
            if len(parts) >= 9 and iq < len(q):
                # mol2 ATOM: id name x y z type resn resid charge ...
                parts[8] = f"{float(q[iq]):.6f}"
                iq += 1
                # Reconstruct with spaces; keep a simple fixed layout.
                rebuilt = (
                    f"{int(parts[0]):7d} {parts[1]:<8s} "
                    f"{float(parts[2]):10.4f} {float(parts[3]):10.4f} "
                    f"{float(parts[4]):10.4f} {parts[5]:<6s} "
                    f"{parts[6]:>4s} {parts[7]:<6s} {parts[8]:>10s}"
                )
                if len(parts) > 9:
                    rebuilt += " " + " ".join(parts[9:])
                out.append(rebuilt + ("\n" if line.endswith("\n") else ""))
                continue
        out.append(line)
    if iq != len(q):
        raise ValueError(f"mol2 atom count {iq} != charge vector length {len(q)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(out), encoding="utf-8")
    return dst


def boltzmann_average_mol2_charges(
    mol2_paths: Sequence,
    energies_kcal: Sequence[float],
    out_mol2,
    *,
    T: float = 298.15,
    ref_mol2=None,
) -> dict:
    """Average charges from centroid mol2 files; write ``out_mol2``.

    ``ref_mol2`` (default: first path) supplies the template connectivity.
    """
    paths = [Path(p) for p in mol2_paths]
    if not paths:
        raise ValueError("no mol2 paths")
    rows = []
    for p in paths:
        _names, chg = read_mol2_charges(p)
        rows.append(chg)
    n_atoms = len(rows[0])
    if any(len(r) != n_atoms for r in rows):
        raise ValueError("centroid mol2 files have inconsistent atom counts")
    avg = average_charge_vectors(rows, energies_kcal, T=T)
    template = Path(ref_mol2) if ref_mol2 is not None else paths[0]
    out = write_mol2_charges(template, out_mol2, avg)
    w = boltzmann_weights(energies_kcal, T=T)
    first = np.asarray(rows[0], dtype=float)
    dq = avg - first
    e = np.asarray(energies_kcal, dtype=float)
    equal_weights = e.size <= 1 or float(np.max(e) - np.min(e)) < 1e-12
    return {
        "out_mol2": str(out),
        "weights": w.tolist(),
        "charges": avg.tolist(),
        "T": float(T),
        "n_atom": int(n_atoms),
        "n_conf": int(len(paths)),
        "mol2_paths": [str(p) for p in paths],
        "energies_kcal": [float(x) for x in e.tolist()],
        "equal_weights": bool(equal_weights),
        "rms_vs_first": float(np.sqrt(np.mean(dq * dq))),
        "max_abs_dq": float(np.max(np.abs(dq))),
    }


def update_lib_charges_from_mol2(lib_path, mol2_path, out_lib: Optional[Path] = None) -> Path:
    """Copy atom charges from mol2 into an Amber OFF lib via parmed."""
    import parmed as pmd

    lib_path = Path(lib_path)
    mol2_path = Path(mol2_path)
    out_lib = Path(out_lib) if out_lib is not None else lib_path
    mol = pmd.load_file(str(mol2_path))
    off = pmd.amber.AmberOFFLibrary.parse(str(lib_path))
    # Single-residue libs are the common case.
    if not off:
        raise ValueError(f"empty OFF library: {lib_path}")
    resname = next(iter(off.keys()))
    res = off[resname]
    if len(res.atoms) != len(mol.atoms):
        raise ValueError(
            f"lib atom count {len(res.atoms)} != mol2 {len(mol.atoms)}"
        )
    for a_lib, a_mol in zip(res.atoms, mol.atoms):
        a_lib.charge = float(a_mol.charge)
    pmd.amber.AmberOFFLibrary.write(off, str(out_lib))
    return out_lib
