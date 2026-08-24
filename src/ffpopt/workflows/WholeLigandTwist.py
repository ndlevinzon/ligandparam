"""Whole-ligand (non-fragmented) dihedral twist with optional AFFDO extras."""

from __future__ import annotations

import logging
from typing import Optional

from ffpopt.workflows.TwistHelpers import (
    PathLike,
    _as_path,
    _list_iteration_frcmods,
    _resolve_logger,
    _run_ffpopt_bin,
)
from ffpopt.workflows.DihedTwist import run_dihed_twist_workflow

def run_whole_ligand_dihed_twist_workflow(
    *,
    mol2: PathLike,
    lib: PathLike,
    frcmod: PathLike,
    out_dir: PathLike = "whole_ligand_twist",
    out_frcmod: PathLike | None = None,
    rotatable_bond_smarts=None,
    delta: int = 10,
    nprim: int = 3,
    maxiter: int = 2,
    nlmaxiter: int = 300,
    nproc: int = 1,
    multi_centroid: int = 0,
    boltzmann_charges: bool = False,
    fit_cli_args: list | None = None,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
    fast_wavefront: bool | None = None,
    **standard_kwargs,
) -> dict:
    """Whole-ligand (non-fragmented) dihedral twist with optional AFFDO extras.

    Builds ``parm7``/``rst7`` via tleap-like PrepareInput from the parent
    Amber triplet, discovers rotatable central bonds with scission, then
    runs :func:`run_dihed_twist_workflow` with ``bytype=True``. Does not
    remove the fragmented path — call this explicitly (e.g. CLI
    ``--whole-ligand``).
    """
    import shutil

    from scission.LigandIo import load_ligand_from_mol2
    from scission.Torsions import find_rotatable_bonds

    log = _resolve_logger(logger)
    out_dir_path = _as_path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    mol2_p = _as_path(mol2).resolve()
    lib_p = _as_path(lib).resolve()
    frcmod_p = _as_path(frcmod).resolve()
    out_frcmod_path = (
        _as_path(out_frcmod).resolve()
        if out_frcmod is not None
        else out_dir_path / f"{mol2_p.stem}.dihed.frcmod"
    )

    from ffpopt.affdo.AffdoLog import describe_affdo_extras, format_boltzmann_summary, log_affdo

    extras_line = describe_affdo_extras(
        whole_ligand=True,
        multi_centroid=multi_centroid,
        boltzmann_charges=boltzmann_charges,
        soft_dihed_restraint=bool(standard_kwargs.get("soft_dihed_restraint")),
        soft_dihed_k=standard_kwargs.get("soft_dihed_k"),
        soft_dihed_kmax=standard_kwargs.get("soft_dihed_kmax"),
        soft_dihed_tol=standard_kwargs.get("soft_dihed_tol"),
        fit_cli_args=fit_cli_args,
    )
    log.info("[whole-twist] whole-ligand twist starting: mol2=%s lib=%s frcmod=%s out_dir=%s", mol2_p, lib_p, frcmod_p, out_dir_path)
    log.info("[whole-twist] extras: %s", extras_line)

    # Materialize parm7/rst7 beside the work dir via leap when needed.
    parm7 = out_dir_path / "ligand.parm7"
    rst7 = out_dir_path / "ligand.rst7"
    if skip_existing and parm7.is_file() and rst7.is_file():
        log.info("[whole-twist] reusing existing %s and %s", parm7, rst7)
    else:
        _write_parent_parm_rst7(mol2_p, lib_p, frcmod_p, parm7, rst7, logger=log)

    start_json = out_dir_path / "start.json"
    if skip_existing and start_json.is_file():
        log.info("[whole-twist] %s exists - skipping PrepareInput", start_json)
    else:
        log.info("[whole-twist] PrepareInput -> %s", start_json)
        _run_ffpopt_bin(
            "ffpopt-PrepareInput.py",
            f"--parm={parm7}",
            f"--crd={rst7}",
            f"--out={start_json.name}",
            cwd=str(out_dir_path),
        )

    ligand = load_ligand_from_mol2(mol2_p)
    smarts = ()
    if rotatable_bond_smarts:
        if isinstance(rotatable_bond_smarts, str):
            smarts = (rotatable_bond_smarts,)
        else:
            smarts = tuple(rotatable_bond_smarts)
    # scission bonds are 1-based; ffpopt expects 0-based.
    bonds = [
        (int(a) - 1, int(b) - 1)
        for a, b in find_rotatable_bonds(ligand, rotatable_bond_smarts=smarts)
    ]
    bonds = sorted({(min(a, b), max(a, b)) for a, b in bonds})
    if not bonds:
        raise RuntimeError("whole-ligand twist: no rotatable bonds found")
    log.info("[whole-twist] %s rotatable bond(s): %s", len(bonds), bonds)

    n_batches = None
    try:
        from ffpopt.workflows.BondBatches import (
            adjacency_from_parmed,
            pack_rotatable_bond_batches,
        )
        from ffpopt.Struct import ListOfStruct

        los = ListOfStruct.from_file(str(start_json))
        mol = los.structs[0].ReadAmberParm()
        n_batches = len(pack_rotatable_bond_batches(bonds, adjacency_from_parmed(mol)))
    except Exception:
        n_batches = None

    from ffpopt.runtime.Console import format_whole_ligand_run_banner, print_run_banner

    print_run_banner(
        format_whole_ligand_run_banner(
            ligand=mol2_p.stem,
            model=str(standard_kwargs.get("model") or "qdpi2"),
            nproc=int(nproc),
            delta=int(delta),
            n_bonds=len(bonds),
            n_batches=n_batches,
            extras=extras_line,
            work_dir=str(out_dir_path),
        )
    )

    boltz_info = None
    if boltzmann_charges and int(multi_centroid or 0) < 2:
        log_affdo(
            log,
            "--boltzmann-charges needs --multi-centroid >= 2; skipping charge rewrite",
        )
    elif boltzmann_charges:
        from ffpopt.affdo.BoltzmannCharges import (
            boltzmann_average_mol2_charges,
            update_lib_charges_from_mol2,
        )
        from ffpopt.affdo.CentroidProfiles import generate_centroid_start_jsons

        log_affdo(
            log,
            "Boltzmann charge average requested (T=298.15 K); generating centroids",
        )
        cents = generate_centroid_start_jsons(
            start_json,
            mol2_path=mol2_p,
            nkeep=int(multi_centroid),
            workdir=out_dir_path,
            logger=log,
        )
        log_affdo(log, "centroid start JSONs: %s", [p.name for p in cents])
        # Without per-centroid charge mol2s, average is a no-op on the parent;
        # write a marker and keep parent charges unless ConfSearch also dumped mol2.
        cent_mol2s = sorted(out_dir_path.glob("centroids_*.mol2"))
        if len(cent_mol2s) >= 2:
            # Energies: equal weights if unknown.
            energies = [0.0] * len(cent_mol2s)
            avg_mol2 = out_dir_path / f"{mol2_p.stem}.boltz.mol2"
            log_affdo(
                log,
                "averaging charges from %s centroid mol2(s) (equal weights; "
                "no per-centroid energies on disk)",
                len(cent_mol2s),
            )
            boltz_info = boltzmann_average_mol2_charges(
                cent_mol2s, energies, avg_mol2, ref_mol2=mol2_p
            )
            out_lib = out_dir_path / f"{mol2_p.stem}.boltz.lib"
            update_lib_charges_from_mol2(lib_p, avg_mol2, out_lib)
            boltz_info["out_lib"] = str(out_lib)
            for line in format_boltzmann_summary(boltz_info):
                log_affdo(log, "%s", line)
        else:
            json_hint = out_dir_path / "centroids.json"
            extra = (
                f"; ConfSearch wrote {json_hint.name}"
                if json_hint.is_file()
                else ""
            )
            log_affdo(
                log,
                "no centroids_*.mol2 charge files under %s%s; "
                "keeping parent mol2/lib charges",
                out_dir_path,
                extra,
            )

    twist = run_dihed_twist_workflow(
        inp=str(start_json),
        bond=bonds,
        delta=delta,
        nprim=nprim,
        maxiter=maxiter,
        bytype=True,
        nlmaxiter=nlmaxiter,
        nproc=nproc,
        skip_existing=skip_existing,
        workdir=out_dir_path,
        logger=log,
        fast_wavefront=fast_wavefront,
        multi_centroid=multi_centroid,
        centroid_mol2=mol2_p,
        fit_cli_args=fit_cli_args,
        job_kind="whole",
        **standard_kwargs,
    )

    # Promote last iteration frcmod to out_frcmod.
    it_frcmods = _list_iteration_frcmods(out_dir_path)
    if it_frcmods:
        shutil.copy2(it_frcmods[-1], out_frcmod_path)
        log.info("[whole-twist] wrote %s from %s", out_frcmod_path, it_frcmods[-1])
    else:
        # No refit needed — copy parent.
        shutil.copy2(frcmod_p, out_frcmod_path)
        log.info("[whole-twist] no itXX.frcmod; copied parent frcmod")

    log_affdo(log, "whole-ligand twist finished: out_frcmod=%s", out_frcmod_path)

    return {
        "out_frcmod": str(out_frcmod_path),
        "out_dir": str(out_dir_path),
        "bonds": bonds,
        "twist": twist,
        "boltzmann_charges": boltz_info,
    }


def _write_parent_parm_rst7(mol2, lib, frcmod, parm7, rst7, logger=None) -> None:
    """Generate parent parm7/rst7 with tleap from mol2/lib/frcmod."""
    import subprocess
    import textwrap

    log = _resolve_logger(logger)
    work = parm7.parent
    leapin = work / "tleap.in"
    leapin.write_text(
        textwrap.dedent(
            f"""
            source leaprc.gaff2
            loadoff {lib}
            loadamberparams {frcmod}
            LIG = loadmol2 {mol2}
            saveamberparm LIG {parm7.name} {rst7.name}
            quit
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    log.info("[whole-twist] tleap -> %s %s", parm7, rst7)
    proc = subprocess.run(
        ["tleap", "-f", str(leapin)],
        cwd=str(work),
        capture_output=True,
        text=True,
    )
    (work / "tleap.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (work / "tleap.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0 or not parm7.is_file() or not rst7.is_file():
        raise RuntimeError(
            f"tleap failed to write {parm7.name}/{rst7.name}; see tleap.*.log"
        )
