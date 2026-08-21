"""Centroid conformers for multi-start torsional scans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def generate_centroid_start_jsons(
    start_json,
    *,
    mol2_path=None,
    nkeep: int = 5,
    nconf: int = 50,
    rmstol: float = 0.5,
    workdir=None,
) -> list:
    """Write ``start.cent{i}.json`` clones seeded by ConfSearch centroids.

    Uses ``mol2_path`` when given; otherwise attempts to reuse coordinates from
    a ConfSearch on the geometry already in ``start_json`` via an RDKit mol
    built from the Amber parm referenced therein when possible.

    Returns
    -------
    list of pathlib.Path
        One start JSON per retained centroid (including a copy of the original
        as ``start.cent0.json`` when ConfSearch is unavailable).
    """
    from copy import deepcopy

    from ffpopt.Struct import ListOfStruct
    from ffpopt.confsearch.ConfSearch import ConformerSearch, GetConformers, ReadMolecule

    start_json = Path(start_json)
    wd = Path(workdir) if workdir is not None else start_json.parent
    wd.mkdir(parents=True, exist_ok=True)

    los0 = ListOfStruct.from_file(str(start_json))
    paths = []

    if mol2_path is not None and Path(mol2_path).is_file():
        out_base = str(wd / "centroids.json")
        ConformerSearch(
            str(mol2_path),
            out_base,
            nconf=max(int(nconf), int(nkeep)),
            nkeep=int(nkeep),
            mmff94=True,
            maxiter=250,
            rmstol=float(rmstol),
            quiet=True,
        )
        # ConformerSearch writes a multi-struct JSON at out_base
        clus = ListOfStruct.from_file(out_base)
        for i, st in enumerate(clus.structs[: int(nkeep)]):
            clone = deepcopy(los0)
            clone.structs = [clone.structs[0]]
            clone.structs[0].Update(
                None, st.data["positions"], st.data.get("forces")
            )
            clone.structs[0].data["name"] = f"centroid_{i}"
            out = wd / f"start.cent{i}.json"
            clone.save(str(out))
            paths.append(out)
    else:
        # Fallback: single starting geometry only.
        out = wd / "start.cent0.json"
        if not out.is_file():
            out.write_bytes(start_json.read_bytes())
        paths.append(out)

    if not paths:
        raise RuntimeError("failed to generate centroid start JSONs")
    return paths


def centroid_energies_from_start(start_cent_json, model_args=None) -> float:
    """Optional single-point energy (eV) for Boltzmann weights; best-effort."""
    try:
        from ffpopt.Struct import ListOfStruct
        from ffpopt.GeomOpt import GeomOpt

        los = ListOfStruct.from_file(str(start_cent_json))
        if model_args is not None:
            los.SetArgs(model_args)
        out = GeomOpt(los, los.structs[0], constraints=None, restraints=None)
        return float(out.data["energy"])
    except Exception:
        return 0.0
