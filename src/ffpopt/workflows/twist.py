"""Single-molecule (and batched) dihedral twist workflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from ffpopt.workflows.helpers import (
    PathLike,
    _apply_fit_and_prepare,
    _as_path,
    _compare_per_bond,
    _in_workdir,
    _promote_batch_iteration_outputs,
    _require_files,
    _resolve_logger,
    _resolve_scans_and_params,
    _run_gendihedfit,
    _run_hl_and_orig_scans,
    _run_scans_for_bonds,
    _write_fit_json,
    normalize_bond_pairs0,
)

def run_batched_dihed_twist_workflow(
    *,
    inp: str,
    bond,
    workdir: PathLike | None = None,
    logger: logging.Logger | None = None,
    progress: Callable[[str, str], None] | None = None,
    **twist_kwargs,
) -> dict:
    """Run twist in sequential bond batches (coupled together, then apply).

    Conservative packing (:mod:`ffpopt.workflows.bond_batches`): keep covalently nearby
    rotors in the same joint fit when possible; split oversized clusters into
    contiguous chunks of ``FFPOPT_MAX_BONDS_PER_TWIST`` (default 2) and update
    the MM between chunks so later batches see prior fits.
    """
    import shutil

    from ffpopt.workflows.bond_batches import (
        adjacency_from_parmed,
        pack_rotatable_bond_batches,
    )
    from ffpopt.Struct import ListOfStruct

    log = _resolve_logger(logger)
    bonds0 = normalize_bond_pairs0(bond)
    wd = _as_path(workdir).resolve() if workdir is not None else Path(".").resolve()
    wd.mkdir(parents=True, exist_ok=True)

    inp_path = Path(_in_workdir(wd, inp)).resolve()
    los = ListOfStruct.from_file(str(inp_path))
    mol = los.structs[0].ReadAmberParm()
    adj = adjacency_from_parmed(mol)
    batches = pack_rotatable_bond_batches(bonds0, adj)
    if len(batches) <= 1:
        return run_dihed_twist_workflow(
            inp=str(inp_path),
            bond=bonds0,
            workdir=wd,
            logger=log,
            progress=progress,
            bond_batching=False,
            **twist_kwargs,
        )

    log.info(
        "[twist] bond batching: %s bond(s) -> %s sequential batch(es) %s",
        len(bonds0),
        len(batches),
        batches,
    )

    merged = {
        "scans": [],
        "fit_jsons": [],
        "iterations": [],
        "initial_comparisons": {},
        "iteration_comparisons": [],
        "early_stopped_at": None,
        "bond_batches": [list(b) for b in batches],
    }
    current_inp = str(inp_path)
    batch_dirs: list[Path] = []

    for bi, batch_bonds in enumerate(batches):
        batch_dir = wd / f"torsion_batch_{bi:02d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_dirs.append(batch_dir)
        # Seed each batch dir with the current topology JSON (absolute parm paths).
        batch_inp = batch_dir / "start.json"
        if Path(current_inp).resolve() != batch_inp.resolve():
            shutil.copy2(current_inp, batch_inp)

        def _batch_progress(stage: str, detail: str = "", _bi=bi, _n=len(batches)):
            if progress is not None:
                try:
                    progress(
                        f"batch{_bi + 1}/{_n}/{stage}",
                        detail or f"{len(batch_bonds)} bond(s)",
                    )
                except Exception:
                    pass

        log.info(
            "[twist] batch %s/%s: %s bond(s) %s (workdir=%s)",
            bi + 1,
            len(batches),
            len(batch_bonds),
            batch_bonds,
            batch_dir,
        )
        _batch_progress("start", f"bonds={batch_bonds}")
        batch_result = run_dihed_twist_workflow(
            inp=str(batch_inp),
            bond=batch_bonds,
            workdir=batch_dir,
            logger=log,
            progress=_batch_progress,
            bond_batching=False,
            **twist_kwargs,
        )
        merged["scans"].extend(batch_result.get("scans") or [])
        merged["fit_jsons"].extend(batch_result.get("fit_jsons") or [])
        merged["iterations"].extend(batch_result.get("iterations") or [])
        init_cmp = batch_result.get("initial_comparisons") or {}
        merged["initial_comparisons"].update(init_cmp)
        merged["iteration_comparisons"].extend(
            batch_result.get("iteration_comparisons") or []
        )
        if batch_result.get("early_stopped_at"):
            merged["early_stopped_at"] = (
                f"batch{bi:02d}:{batch_result['early_stopped_at']}"
            )

        # Next batch starts from the latest fitted parm/json when available.
        iters = batch_result.get("iterations") or []
        if iters:
            current_inp = str(iters[-1]["json"])
        else:
            current_inp = str(batch_inp)

    promoted = _promote_batch_iteration_outputs(batch_dirs, wd, logger=log)
    merged["promoted_frcmods"] = [str(p) for p in promoted]
    log.info(
        "[twist] bond batching finished: %s batch(es), %s promoted itXX.frcmod",
        len(batches),
        len(promoted),
    )
    if progress is not None:
        try:
            progress("finished", f"{len(batches)} bond batch(es)")
        except Exception:
            pass
    return merged


def run_dihed_twist_workflow(
    *,
    inp: str,
    bond,
    delta: int = 10,
    nprim: int = 3,
    maxiter: int = 2,
    bytype: bool = False,
    nlmaxiter: int = 300,
    nproc: int = 1,
    wf_starting_nodes: int = 4,
    wf_num_conformers: int = 0,
    wf_max_levels: int = -1,
    wf_convergence_threshold: float = 0.01,
    skip_existing: bool = True,
    compare_config=None,
    skip_converged_initial: bool = True,
    convergence_mode: str = "drop",
    plot_comparisons: bool = False,
    structure_images: dict | None = None,
    workdir: PathLike | None = None,
    logger: logging.Logger | None = None,
    progress: Callable[[str, str], None] | None = None,
    cpu_budget_path: PathLike | None = None,
    budget_owner: str | None = None,
    budget_total: int | None = None,
    fast_wavefront: bool | None = None,
    bond_batching: bool | None = None,
    multi_centroid: int = 0,
    centroid_mol2: PathLike | None = None,
    fit_cli_args: list | None = None,
    **standard_kwargs,
) -> dict:
    """ Wavefront-only twist workflow, run in-process.

    Mirrors the phase structure of ``bin/ffpopt-DihedTwistWorkflow.py`` but
    executes each scan via :func:`ffpopt.scan.WaveFront.run_dihed_wavefront`
    instead of emitting a bash script; fit and prepare steps still shell
    out to the existing bin scripts. The phases are: a high-level scan per
    bond at ``model``; a reference sander scan per bond; an optional
    Phase 2b that drops bonds whose HL and reference already agree; and
    up to ``maxiter`` rounds of fit-then-rescan with an optional
    per-iteration convergence check. See the ``Workflows`` RST page for the
    full phase narrative.

    When more than ``FFPOPT_MAX_BONDS_PER_TWIST`` bonds are requested (default
    2), covalently nearby rotors are packed into sequential batches via
    :func:`run_batched_dihed_twist_workflow` unless ``bond_batching=False``.
    """
    from ffpopt.workflows.bond_batches import should_batch_bonds

    bonds_early = normalize_bond_pairs0(bond)
    do_batch = bond_batching
    if do_batch is None:
        do_batch = should_batch_bonds(len(bonds_early))
    if do_batch and should_batch_bonds(len(bonds_early)):
        return run_batched_dihed_twist_workflow(
            inp=inp,
            bond=bonds_early,
            delta=delta,
            nprim=nprim,
            maxiter=maxiter,
            bytype=bytype,
            nlmaxiter=nlmaxiter,
            nproc=nproc,
            wf_starting_nodes=wf_starting_nodes,
            wf_num_conformers=wf_num_conformers,
            wf_max_levels=wf_max_levels,
            wf_convergence_threshold=wf_convergence_threshold,
            skip_existing=skip_existing,
            compare_config=compare_config,
            skip_converged_initial=skip_converged_initial,
            convergence_mode=convergence_mode,
            plot_comparisons=plot_comparisons,
            structure_images=structure_images,
            workdir=workdir,
            logger=logger,
            progress=progress,
            cpu_budget_path=cpu_budget_path,
            budget_owner=budget_owner,
            budget_total=budget_total,
            fast_wavefront=fast_wavefront,
            multi_centroid=multi_centroid,
            centroid_mol2=centroid_mol2,
            fit_cli_args=fit_cli_args,
            **standard_kwargs,
        )

    # ---- 0. Resolve & validate kwargs ------------------------------------
    valid_modes = {"drop", "all_or_nothing", "off"}
    if convergence_mode not in valid_modes:
        raise ValueError(
            f"convergence_mode must be one of {sorted(valid_modes)}; "
            f"got {convergence_mode!r}"
        )
    log = _resolve_logger(logger)
    wd = _as_path(workdir).resolve() if workdir is not None else None
    nproc = max(1, int(nproc))
    budget_path = (
        _as_path(cpu_budget_path).resolve() if cpu_budget_path is not None else None
    )
    budget_tot = max(1, int(budget_total if budget_total is not None else nproc))

    from ffpopt.runtime.fast_wavefront import (
        apply_fast_wavefront_presets,
        fast_wavefront_enabled,
        prefer_wavefront_depth,
    )

    if fast_wavefront is True:
        os.environ["FFPOPT_FAST_WAVEFRONT"] = "1"
    elif fast_wavefront is False:
        os.environ["FFPOPT_FAST_WAVEFRONT"] = "0"
    fast_on = fast_wavefront_enabled(fast_wavefront)

    def _prog(stage: str, detail: str = "") -> None:
        if progress is not None:
            try:
                progress(stage, detail)
            except Exception:
                pass

    def _lease_nproc(phase: str) -> int:
        """Return cores for this scan phase (re-lease when a shared budget exists)."""
        if budget_path is None or not budget_owner:
            return nproc
        from ffpopt.runtime.cpu_budget import CpuBudget

        budget = CpuBudget(budget_path, budget_tot)
        leased = max(1, int(budget.lease(str(budget_owner))))
        log.info(
            "[twist] %s leased %s/%s cores (owner=%s)",
            phase,
            leased,
            budget_tot,
            budget_owner,
        )
        return leased

    def _release_lease(phase: str) -> None:
        """Drop this fragment's lease during serial non-scan work so siblings grow."""
        if budget_path is None or not budget_owner:
            return
        from ffpopt.runtime.cpu_budget import CpuBudget

        try:
            CpuBudget(budget_path, budget_tot).release(str(budget_owner))
            log.info(
                "[twist] %s released CPU lease (owner=%s)",
                phase,
                budget_owner,
            )
        except Exception:
            log.exception(
                "[twist] %s: failed to release CPU lease (owner=%s)",
                phase,
                budget_owner,
            )

    import argparse
    from types import SimpleNamespace
    from ffpopt.Options import AddStandardOptions
    from ffpopt.Struct import ListOfStruct

    _p = argparse.ArgumentParser(add_help=False)
    AddStandardOptions(_p)
    std_defaults = vars(_p.parse_args([]))
    unknown = set(standard_kwargs) - set(std_defaults)
    if unknown:
        raise TypeError(
            "run_dihed_twist_workflow got unexpected keyword argument(s): "
            f"{sorted(unknown)}"
        )
    std = {**std_defaults, **standard_kwargs}
    model = std["model"]

    # Fast presets: only replace knobs still at library defaults.
    fast_knobs = {
        "delta": delta,
        "wf_convergence_threshold": wf_convergence_threshold,
        "geometric_maxiter": std.get("geometric_maxiter"),
        "geometric_converge": std.get("geometric_converge"),
        "ase_opt_tol": std.get("ase_opt_tol"),
    }
    applied = apply_fast_wavefront_presets(fast_knobs, enabled=fast_on)
    if applied:
        log.info("[twist] fast-wavefront presets applied: %s", applied)
        delta = int(fast_knobs["delta"])
        wf_convergence_threshold = float(fast_knobs["wf_convergence_threshold"])
        for key in ("geometric_maxiter", "geometric_converge", "ase_opt_tol"):
            if key in applied:
                std[key] = fast_knobs[key]
                standard_kwargs[key] = fast_knobs[key]
    prefer_depth = prefer_wavefront_depth(model=model, fast=fast_on)

    from ffpopt.affdo.log import describe_affdo_extras, log_affdo

    affdo_line = describe_affdo_extras(
        multi_centroid=multi_centroid,
        soft_dihed_restraint=bool(std.get("soft_dihed_restraint")),
        soft_dihed_k=std.get("soft_dihed_k"),
        soft_dihed_tol=std.get("soft_dihed_tol"),
        fit_cli_args=fit_cli_args,
    )
    if (
        int(multi_centroid or 0) >= 2
        or bool(std.get("soft_dihed_restraint"))
        or bool(fit_cli_args)
    ):
        log_affdo(log, "twist extras: %s", affdo_line)

    # SimpleNamespace mirroring the CLI's args - needed by los.SetArgs.
    bonds0 = normalize_bond_pairs0(bond)
    inp_path = str(_in_workdir(wd, inp).resolve())
    args = SimpleNamespace(
        inp=inp_path,
        bond=[f"{a},{b}" for a, b in bonds0],  # string form for legacy SetArgs
        delta=delta,
        nprim=nprim,
        maxiter=maxiter,
        bytype=bytype,
        nlmaxiter=nlmaxiter,
        nproc=nproc,
        wf_starting_nodes=wf_starting_nodes,
        wf_num_conformers=wf_num_conformers,
        wf_max_levels=wf_max_levels,
        wf_convergence_threshold=wf_convergence_threshold,
        **std,
    )

    los = ListOfStruct.from_file(args.inp)
    los.structs = [los.structs[0]]
    los.SetArgs(args)
    mol = los.structs[0].ReadAmberParm()
    origparm = str(_in_workdir(wd, los.structs[0].data["parm"]).resolve())

    bonds_parsed = list(bonds0)
    scans, params, s_template = _resolve_scans_and_params(
        mol, bonds_parsed, nprim=nprim, bytype=bytype
    )

    # Common wavefront kwargs, reused on every _run_one_scan call.
    # ``nproc`` is refreshed per scan phase when a shared CPU budget is used.
    wf_kwargs = dict(
        delta=delta,
        nproc=nproc,
        wf_starting_nodes=wf_starting_nodes,
        wf_max_levels=wf_max_levels,
        wf_num_conformers=wf_num_conformers,
        wf_convergence_threshold=wf_convergence_threshold,
        fast_wavefront=fast_on,
        **{k: v for k, v in standard_kwargs.items() if k != "model"},
    )

    hlname = model.replace("/", "_")
    results = {
        "scans": [],
        "fit_jsons": [],
        "iterations": [],
        "initial_comparisons": {},
        "iteration_comparisons": [],
        "early_stopped_at": None,
    }

    try:
        # ---- 1+2. HL and reference sander scans (pipelined) -----------------
        phase_nproc = _lease_nproc("hl_orig_scan")
        wf_kwargs["nproc"] = phase_nproc
        _prog(
            "hl_orig_scan",
            f"HL={model} || orig=sander | {len(scans)} bond(s) | nproc={phase_nproc}",
        )
        results["scans"].extend(
            _run_hl_and_orig_scans(
                scans,
                hl_prefix=hlname,
                hl_model=model,
                inp=args.inp,
                nproc=phase_nproc,
                skip_existing=skip_existing,
                workdir=wd,
                logger=log,
                wf_kwargs=wf_kwargs,
                prefer_wf_depth=prefer_depth,
                multi_centroid=multi_centroid,
                centroid_mol2=centroid_mol2,
            )
        )

        # Serial compare/fit/apply: free cores so sibling fragments can grow.
        _release_lease("post_orig_scan")

        if plot_comparisons:
            plot_dir = wd if wd is not None else Path(".")
        else:
            plot_dir = None

        # ---- 2b. Drop dihedrals that already agree (initial convergence) -----
        if skip_converged_initial:
            _prog("compare", "initial HL vs LL")
            initial = _compare_per_bond(
                scans, hlname, "orig", compare_config,
                plot_dir=plot_dir, structure_images=structure_images, workdir=wd,
            )
            results["initial_comparisons"] = initial
            kept_bonds = []
            for bond, scan in zip(bonds_parsed, scans):
                idx = scan.GetIdxStr()
                r = initial[idx]
                if r.is_close:
                    reason = "flat (HL barrier below threshold)" if r.is_flat \
                        else "agrees with HL within thresholds"
                    log.info("[twist] %s: %s - dropping from iterative fit", idx, reason)
                else:
                    kept_bonds.append(bond)
                    reasons = "; ".join(r.reasons) if r.reasons else "extrema disagree"
                    log.info("[twist] %s: refit needed (%s)", idx, reasons)
            if not kept_bonds:
                log.info("[twist] all dihedrals already agree - skipping Phase 3")
                return results
            if len(kept_bonds) < len(bonds_parsed):
                scans, params, s_template = _resolve_scans_and_params(
                    mol, kept_bonds, nprim=nprim, bytype=bytype
                )
                log.info("[twist] fitting %s of %s dihedrals", len(scans), len(bonds_parsed))

        # ---- 3. Iterative refinement -----------------------------------------
        for it in range(args.maxiter):
            pitname = "it%02i" % (it)
            citname = "it%02i" % (it + 1)
            parm = origparm if it == 0 else f"{pitname}.parm7"
            ll_prefix = "orig" if it == 0 else pitname

            # 3a. Write the fit.json input.
            fit_json = _write_fit_json(
                citname=citname,
                scans=scans,
                params=params,
                s_template=s_template,
                hl_prefix=hlname,
                ll_prefix=ll_prefix,
                parm=parm,
                workdir=wd,
            )
            results["fit_jsons"].append(fit_json)

            # 3b. GenDihedFit -> itNN.py. FUTURE: replace subprocess with API call.
            _prog(f"fit/{citname}", "GenDihedFit")
            _run_gendihedfit(
                citname,
                nlmaxiter=args.nlmaxiter,
                skip_existing=skip_existing,
                logger=log,
                workdir=wd,
                fit_cli_args=fit_cli_args,
            )

            # 3c. Apply fit + PrepareInput. FUTURE: replace subprocess with API calls.
            _prog(f"apply/{citname}", "apply fit + PrepareInput")
            _apply_fit_and_prepare(
                citname=citname,
                origparm=origparm,
                inp=args.inp,
                skip_existing=skip_existing,
                logger=log,
                workdir=wd,
            )
            results["iterations"].append(
                {
                    "parm": str(_in_workdir(wd, f"{citname}.parm7")),
                    "json": str(_in_workdir(wd, f"{citname}.json")),
                }
            )

            # 3d. Sander scans on the updated parm (one per bond, "itNN" prefix).
            phase_nproc = _lease_nproc(f"rescan/{citname}")
            wf_kwargs["nproc"] = phase_nproc
            _prog(
                f"rescan/{citname}",
                f"sander | {len(scans)} bond(s) | nproc={phase_nproc}",
            )
            results["scans"].extend(
                _run_scans_for_bonds(
                    scans,
                    prefix=citname,
                    model="sander",
                    inp=str(_in_workdir(wd, f"{citname}.json")),
                    nproc=phase_nproc,
                    skip_existing=skip_existing,
                    workdir=wd,
                    logger=log,
                    wf_kwargs=wf_kwargs,
                    seed_prefix=ll_prefix,
                )
            )
            _release_lease(f"post_rescan/{citname}")

            # 3e. Per-iteration convergence: compare HL vs itNN per bond.
            if convergence_mode != "off":
                iter_cmp = _compare_per_bond(
                    scans, hlname, citname, compare_config,
                    plot_dir=plot_dir, structure_images=structure_images, workdir=wd,
                )
                results["iteration_comparisons"].append(
                    {"citname": citname, "comparisons": iter_cmp}
                )
                converged_idxs = [idx for idx, r in iter_cmp.items() if r.is_close]
                still_off_idxs = [idx for idx, r in iter_cmp.items() if not r.is_close]

                if not still_off_idxs:
                    # All bonds converged this iteration - break in both modes.
                    log.info("[twist] all dihedrals converged at %s - stopping early", citname)
                    results["early_stopped_at"] = citname
                    break

                if convergence_mode == "drop" and converged_idxs:
                    # Rebuild scans/params/s_template from just the survivors so
                    # the next iteration's fit no longer sees the converged ones.
                    kept_bonds = [
                        (s.idxs[1], s.idxs[2])
                        for s in scans
                        if s.GetIdxStr() in still_off_idxs
                    ]
                    scans, params, s_template = _resolve_scans_and_params(
                        mol, kept_bonds, nprim=nprim, bytype=bytype
                    )
                    log.info(
                        "[twist] %s: dropping converged (%s); continuing with "
                        "%s dihedral(s): %s",
                        citname,
                        ", ".join(converged_idxs),
                        len(scans),
                        ", ".join(still_off_idxs),
                    )
                else:
                    # all_or_nothing mode (or drop mode with nothing converged) -
                    # just log and continue with the same set.
                    log.info(
                        "[twist] %s: %s dihedral(s) still need refit: %s",
                        citname,
                        len(still_off_idxs),
                        ", ".join(still_off_idxs),
                    )

        _prog("finished", "twist workflow complete")
        return results
    finally:
        _release_lease("finished")

