"""Fragmented dihedral-twist workflow (scission + per-fragment twist + merge)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from ffpopt.runtime.NondaemonPool import make_nondaemon_spawn_pool
from ffpopt.workflows.TwistHelpers import (
    PathLike,
    _LOG,
    _as_path,
    _parent_paths_from_args,
    _resolve_logger,
    _run_ffpopt_bin,
    _split_fragment_nproc,
    bonds0_from_scission_fit_torsions,
)
from ffpopt.workflows.DihedTwist import run_dihed_twist_workflow

def _load_existing_fragments(out_dir: Path):
    """ Build lightweight per-fragment records from a prior scission run.

    Reads ``out_dir/fragment_index.json`` and rehydrates each fragment as a
    ``types.SimpleNamespace`` with the attributes the workflow loop reads
    off of ``scission.SelectedFragment``: ``fragment_id``,
    ``manifest_path``, ``parm7_path``, ``rst7_path``, and ``fit_torsions``.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory that holds ``fragment_index.json`` plus the per-fragment
        subdirectories.

    Returns
    -------
    list or None
        List of ``SimpleNamespace`` fragment records, or ``None`` if no
        ``fragment_index.json`` is present in ``out_dir`` (so the caller
        knows to run scission).
    """
    from types import SimpleNamespace

    index_path = out_dir / "fragment_index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    fragments = []
    for entry in index.get("fragments", []):
        directory = Path(entry["directory"])
        if not directory.exists():
            return None
        fit_torsions_path = directory / "fit_torsions.json"
        fit_torsions = (
            json.loads(fit_torsions_path.read_text())
            if fit_torsions_path.exists() else []
        )
        fragments.append(
            SimpleNamespace(
                fragment_id=entry["fragment_id"],
                manifest_path=directory / "manifest.json",
                parm7_path=directory / "fragment.parm7",
                rst7_path=directory / "fragment.rst7",
                mol2_path=directory / "fragment.mol2",
                fit_torsions=fit_torsions,
            )
        )
    return fragments


def _build_structure_image_map(frag_dir: Path, fit_torsions: list) -> dict:
    """ Locate scission's per-torsion 2D drawings for one fragment.

    Reads ``manifest.json`` for the authoritative ``label -> image path``
    mapping; for any label missing from the manifest (older runs, or rdkit
    unavailable when the fragment was written), falls back to the standard
    ``fragment_dir/torsion_{safe_name(label)}.svg`` filename if present.

    Parameters
    ----------
    frag_dir : pathlib.Path
        One fragment's output directory.
    fit_torsions : list of dict
        The fragment's ``fit_torsions.json`` payload (**1-based** atom
        indices). Converted to 0-based pairs for plot keys.

    Returns
    -------
    dict
        Map from ``frozenset({a, b})`` (0-based central-bond atom indices)
        to the matching structure image ``Path``. Empty if no images can
        be located.
    """
    manifest_images: dict[str, str] = {}
    manifest_path = frag_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            manifest_images = manifest.get("torsion_image_paths") or {}
        except (json.JSONDecodeError, OSError):
            manifest_images = {}

    from scission.Writers import safe_name

    out: dict = {}
    for t in fit_torsions:
        label = t.get("label")
        if label is None:
            continue
        img_str = manifest_images.get(label)
        if img_str:
            img = Path(img_str)
        else:
            img = frag_dir / f"torsion_{safe_name(label)}.svg"
        if not img.exists():
            continue
        bond_pair = tuple(i - 1 for i in t["fragment_rotatable_bond"])
        out[frozenset(bond_pair)] = img
    return out


def _prepare_fragment_input(
    fragment,
    skip_existing: bool,
    logger: logging.Logger | None = None,
    workdir: PathLike | None = None,
) -> str:
    """ Run ``ffpopt-PrepareInput.py`` on a scission fragment.

    Produces ``start.json`` next to the fragment's parm7/rst7. FUTURE:
    replace subprocess with an API call once PrepareInput is refactored.

    Parameters
    ----------
    fragment : scission.SelectedFragment or types.SimpleNamespace
        Fragment record exposing ``parm7_path``, ``rst7_path``, and
        ``fragment_id``.
    skip_existing : bool
        If True and ``start.json`` already exists in ``workdir``, PrepareInput
        is skipped.
    workdir : path-like, optional
        Fragment directory. Defaults to ``fragment.manifest_path.parent``.
        PrepareInput runs with ``subprocess(..., cwd=workdir)``; parm/crd/out
        are passed as absolute paths so the parent process never needs
        ``os.chdir``.

    Returns
    -------
    str
        Absolute path to ``start.json``.
    """
    if fragment.parm7_path is None or fragment.rst7_path is None:
        raise RuntimeError(
            f"fragment {fragment.fragment_id} has no parm7/rst7 - scission "
            f"likely failed to run tleap. Check that AmberTools is on PATH."
        )
    log = _resolve_logger(logger)
    frag_dir = (
        _as_path(workdir).resolve()
        if workdir is not None
        else _as_path(fragment.manifest_path).parent.resolve()
    )
    start_json = frag_dir / "start.json"
    parm7 = _as_path(fragment.parm7_path).resolve()
    rst7 = _as_path(fragment.rst7_path).resolve()
    if skip_existing and start_json.exists():
        log.info("[frag-twist] %s exists - skipping PrepareInput", start_json)
        return str(start_json)
    log.info("[frag-twist] PrepareInput -> %s (cwd=%s)", start_json, frag_dir)
    _run_ffpopt_bin(
        "ffpopt-PrepareInput.py",
        f"--parm={parm7}",
        f"--crd={rst7}",
        f"--out={start_json}",
        cwd=str(frag_dir),
    )
    return str(start_json)


def _slim_twist_result(twist_result: Optional[dict]) -> Optional[dict]:
    """Drop heavy ``wf_run`` objects so fragment-pool IPC stays picklable."""
    from ffpopt.runtime.SlimIpc import slim_twist_result

    return slim_twist_result(twist_result)


# Sentinel written when a fragment twist finishes successfully. On parent
# restart with ``skip_existing``, fragments that already have this marker are
# not re-queued (and therefore do not take a CPU lease).
_FRAG_TWIST_DONE = "frag-twist.done"


def fragment_twist_done_path(frag_dir: PathLike) -> Path:
    return Path(frag_dir) / _FRAG_TWIST_DONE


def is_fragment_twist_done(frag_dir: PathLike) -> bool:
    """True when a prior successful twist left a completion sentinel."""
    return fragment_twist_done_path(frag_dir).is_file()


def mark_fragment_twist_done(frag_dir: PathLike) -> Path:
    """Create the completion sentinel for ``frag_dir``."""
    path = fragment_twist_done_path(frag_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")
    return path


def clear_fragment_twist_done(frag_dir: PathLike) -> None:
    """Remove the completion sentinel (forced recompute)."""
    path = fragment_twist_done_path(frag_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _run_fragment_twist_job(job: dict) -> dict:
    """Worker entry: prepare + twist one fragment (picklable job dict)."""
    from types import SimpleNamespace

    from ffpopt.runtime.CpuBudget import CpuBudget
    from ffpopt.runtime.ProgressBoard import (
        FragmentProgressStore,
        fragment_stdio_to_file,
        make_fragment_file_logger,
    )

    frag_dir = Path(job["frag_dir"]).resolve()
    fragment_id = job["fragment_id"]
    fragment = SimpleNamespace(
        fragment_id=fragment_id,
        manifest_path=frag_dir / "manifest.json",
        parm7_path=job["parm7"],
        rst7_path=job["rst7"],
        fit_torsions=job["fit_torsions"],
    )
    bonds = [tuple(b) for b in job["bonds"]]
    frag_log_path = frag_dir / "frag-twist.log"
    store = None
    status_path = job.get("status_path")
    if status_path:
        store = FragmentProgressStore(status_path)

    def _set(**kwargs):
        if store is not None:
            store.update(fragment_id, **kwargs)

    # Already complete from a prior run: report done without leasing CPUs.
    if job.get("skip_existing") and is_fragment_twist_done(frag_dir):
        from ffpopt.geom.Geometric import sweep_geometric_scratch_dir

        sweep_geometric_scratch_dir(frag_dir, recursive=True)
        _set(
            status="done",
            stage="finished",
            detail="skipped (complete)",
            bonds=len(bonds),
            log_path=str(frag_log_path),
        )
        return {
            "fragment_id": fragment_id,
            "dir": str(frag_dir),
            "bonds": bonds,
            "twist_result": None,
            "log_path": str(frag_log_path),
            "skipped_complete": True,
        }

    if not job.get("skip_existing"):
        clear_fragment_twist_done(frag_dir)

    def _progress(stage: str, detail: str = "") -> None:
        _set(status="running", stage=stage, detail=detail)

    frag_log = make_fragment_file_logger(fragment_id, frag_log_path)
    structure_images = _build_structure_image_map(frag_dir, fragment.fit_torsions)

    budget = None
    budget_path = job.get("budget_path")
    budget_total = int(job.get("budget_total") or job.get("wf_nproc") or 1)
    # Do not lease through PrepareInput: that is serial and would park cores
    # while other fragments could be scanning. Twist re-leases per scan phase.
    leased_hint = max(1, int(job.get("wf_nproc") or 1))
    if budget_path:
        budget = CpuBudget(budget_path, budget_total)

    _set(
        status="running",
        stage="prepare",
        detail=f"{len(bonds)} bond(s) | nproc~{leased_hint}",
        bonds=len(bonds),
        log_path=str(frag_log_path),
    )
    frag_log.info(
        "[frag-twist] %s: %s bond(s) %s -> prepare then twist in %s "
        "(CPU leases during scans only; hint %s/%s)",
        fragment_id,
        len(bonds),
        bonds,
        frag_dir,
        leased_hint,
        budget_total,
    )

    try:
        with fragment_stdio_to_file(frag_log_path, fragment_id=fragment_id):
            start_json = _prepare_fragment_input(
                fragment,
                skip_existing=job["skip_existing"],
                logger=frag_log,
                workdir=frag_dir,
            )
            twist_kwargs = dict(job["twist_kwargs"])
            twist_kwargs["nproc"] = int(leased_hint)
            twist_kwargs.pop("logger", None)
            twist_kwargs.pop("progress", None)
            twist_kwargs.pop("cpu_budget_path", None)
            twist_kwargs.pop("budget_owner", None)
            twist_kwargs.pop("budget_total", None)
            frag_mol2 = getattr(fragment, "mol2_path", None)
            if frag_mol2 and Path(frag_mol2).is_file():
                twist_kwargs["centroid_mol2"] = str(Path(frag_mol2).resolve())
            else:
                twist_kwargs["centroid_mol2"] = None
            if budget_path:
                twist_kwargs["cpu_budget_path"] = str(budget_path)
                twist_kwargs["budget_owner"] = fragment_id
                twist_kwargs["budget_total"] = budget_total
            twist_result = run_dihed_twist_workflow(
                inp=start_json,
                bond=bonds,
                structure_images=structure_images or None,
                workdir=frag_dir,
                logger=frag_log,
                progress=_progress,
                **twist_kwargs,
            )
        _set(status="done", stage="finished", detail="ok")
        mark_fragment_twist_done(frag_dir)
        frag_log.info("[frag-twist] %s: twist workflow finished", fragment_id)
        return {
            "fragment_id": fragment_id,
            "dir": str(frag_dir),
            "bonds": bonds,
            "twist_result": _slim_twist_result(twist_result),
            "log_path": str(frag_log_path),
        }
    except Exception as exc:
        _set(
            status="failed",
            stage="failed",
            detail=type(exc).__name__,
            error=str(exc)[:200],
        )
        frag_log.exception("[frag-twist] %s: failed", fragment_id)
        raise
    finally:
        if budget is not None:
            try:
                budget.release(fragment_id)
                frag_log.info(
                    "[frag-twist] %s released CPU lease", fragment_id
                )
            except Exception:
                frag_log.exception(
                    "[frag-twist] %s: failed to release CPU lease", fragment_id
                )


def run_fragmented_dihed_twist_workflow(
    *,
    mol2: PathLike | None = None,
    lib: PathLike | None = None,
    frcmod: PathLike | None = None,
    bundle=None,
    out_dir: PathLike = "fragments",
    merged_frcmod: PathLike = "merged.frcmod",
    fragment_config=None,
    rotatable_bond_smarts=None,
    delta: int = 10,
    nprim: int = 3,
    maxiter: int = 2,
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
    plot_comparisons: bool = True,
    logger: logging.Logger | None = None,
    fast_wavefront: bool | None = None,
    multi_centroid: int = 0,
    centroid_mol2: PathLike | None = None,
    fit_cli_args: list | None = None,
    **standard_kwargs,
) -> dict:
    """ Fragment a ligand with scission, run the twist workflow on each fragment, then recombine.

    Drives ``scission`` (from FragmentMol) to break the parent ligand into
    reduced fragments, runs :func:`run_dihed_twist_workflow` per fragment
    with ``workdir=frag_dir`` (absolute paths + subprocess ``cwd``; no
    process-wide ``os.chdir``), then merges the per-fragment fitted
    DIHE terms back into a unified parent ``frcmod`` via
    ``scission.Merge.merge_fragment_frcmods``. Like
    :func:`run_dihed_twist_workflow`, this must be called from inside an
    ``if __name__ == "__main__":`` guard - the wavefront uses ``spawn``-mode
    multiprocessing. See the ``Workflows`` RST page for the full on-disk
    layout and the re-running semantics.

    Parameters
    ----------
    mol2, lib, frcmod
        Parent Amber triplet paths (``str`` or :class:`~pathlib.Path`).
        Required unless ``bundle`` is provided.
    bundle
        Optional :class:`~ligandparam.io.AmberBundle.AmberLigandBundle` or
        :class:`scission.Models.InputBundle`. When set, overrides ``mol2`` /
        ``lib`` / ``frcmod``.
    out_dir
        Directory where per-fragment subdirs are written, resolved to an
        absolute path before scission runs. Default is ``"fragments"``.
    merged_frcmod
        Path for the final merged parent frcmod. Relative to the call cwd,
        not ``out_dir``. A ``.merge_report.json`` is written alongside.
    logger
        Optional logger for progress (default: ``ffpopt.workflows``).
    fragment_config : scission.FragmentConfig, optional
        Override scission's fragmentation config. Default is None (uses
        ``FragmentConfig()`` - acyclic rotatable torsions, 30 deg step, etc.).
    rotatable_bond_smarts : str or iterable of str, optional
        Extra SMARTS patterns nominating additional central bonds as
        rotatable, forwarded to ``scission.FragmentConfig.rotatable_bond_smarts``.
        Each pattern must map the central bond atoms with atom-map numbers
        ``:1`` and ``:2`` (e.g. ``"[C:1](=[O])[N:2]"`` to make amide-like
        single bonds rotatable, or ``"[C:1]=[N:2]"`` for exocyclic imines).
        Patterns add to - they do not replace - scission's default rotatable
        bond identification. A single string is accepted as shorthand for a
        one-element list. When ``fragment_config`` is also supplied, these
        patterns are appended to ``fragment_config.rotatable_bond_smarts``.
        Requires RDKit. Default is None (no extra patterns).
    delta : int, optional
        Wavefront angle step (degrees). Default is 10.
    nprim : int, optional
        Number of primary cosine terms per parameter family. Default is 3.
    maxiter : int, optional
        Maximum fit-then-rescan iterations per fragment. Default is 2.
    nlmaxiter : int, optional
        Forwarded to ``ffpopt-GenDihedFit.py``. Default is 300.
    nproc : int, optional
        Total CPU budget for the fragmented workflow. Caps concurrent
        fragment workers at ``min(nproc, n_runnable_fragments)``. Per-fragment
        wavefront ``nproc`` is leased dynamically from a shared
        ``.cpu_budget.json`` (fair share re-leased before each scan phase;
        released during prepare / GenDihedFit / compare so siblings can grow)
        so finished fragments free cores for remaining work.
        Default is 1.
    wf_starting_nodes : int, optional
        Wavefront starting nodes. Default is 4.
    wf_num_conformers : int, optional
        Wavefront number of conformers. Default is 0 (auto).
    wf_max_levels : int, optional
        Wavefront max levels. Default is -1.
    wf_convergence_threshold : float, optional
        Wavefront convergence threshold (kcal/mol). Default is 0.01.
    skip_existing : bool, optional
        If True, every short-circuit applies: skip the scission call if
        ``out_dir/fragment_index.json`` exists, skip per-fragment scans /
        fits / prepare steps if their outputs already exist. Set False to
        force a fresh run from scratch. Default is True.
    compare_config : ffpopt.ScanAnalysis.ScanCompareConfig, optional
        Thresholds for the HL-vs-LL comparison heuristic. Default is None.
    skip_converged_initial : bool, optional
        If True, Phase 2b of each fragment's twist drops bonds whose HL
        and reference scans already agree. Default is True.
    convergence_mode : {"drop", "all_or_nothing", "off"}, optional
        Per-iteration convergence behaviour inside each fragment's twist.
        Default is "drop".
    plot_comparisons : bool, optional
        If True, save a PNG plot per bond per comparison alongside the
        ``.dat`` files. Defaults to True here (vs. False for the
        single-molecule workflow) since fragments produce many
        comparisons. Default is True.
    **standard_kwargs
        Forwarded to the wavefront. Accepts anything declared by
        :func:`ffpopt.Options.AddStandardOptions`.

    Returns
    -------
    dict
        A dictionary with keys ``fragmentation``
        (``FragmentationResult.to_dict()`` or ``None`` when scission was
        skipped because a prior run already existed), ``fragments`` (list
        of per-fragment records with ``fragment_id``, ``dir``, ``bonds``,
        ``twist_result``), ``merge_report`` (the report from
        ``scission.Merge.merge_fragment_frcmods``), and ``merged_frcmod``
        (path to the final merged parent frcmod).
    """
    try:
        from dataclasses import replace as _dc_replace

        from scission import FragmentConfig, InputBundle, fragment_ligand
        from scission.Merge import merge_fragment_frcmods
    except ImportError as e:
        raise ImportError(
            "run_fragmented_dihed_twist_workflow requires the integrated "
            "'scission' package (src/scission). Reinstall ligandparam "
            "(pip install -e .) so scission is on PYTHONPATH."
        ) from e

    config = fragment_config if fragment_config is not None else FragmentConfig()
    if rotatable_bond_smarts is not None:
        extra_smarts = (
            (rotatable_bond_smarts,)
            if isinstance(rotatable_bond_smarts, str)
            else tuple(rotatable_bond_smarts)
        )
        if extra_smarts:
            config = _dc_replace(
                config,
                rotatable_bond_smarts=config.rotatable_bond_smarts + extra_smarts,
            )
    # Scission screen + per-fragment Amber writes share the workflow core budget.
    config = _dc_replace(config, nproc=max(1, int(nproc)))
    log = _resolve_logger(logger)
    # Nested spawn pools + ASE/sander: keep BLAS/OMP at 1 unless the user
    # already set it (avoids oversubscribe when many workers share a node).
    from ffpopt.runtime.CpuThreads import pin_math_threads

    pin_math_threads(1)
    mol2_path, lib_path, parent_frcmod = _parent_paths_from_args(
        mol2=mol2, lib=lib, frcmod=frcmod, bundle=bundle
    )
    input_bundle = InputBundle(
        mol2_path=mol2_path,
        lib_path=lib_path,
        frcmod_path=parent_frcmod,
    )
    out_dir_path = _as_path(out_dir).resolve()
    merged_frcmod_path = _as_path(merged_frcmod).resolve()

    existing_fragments = (
        _load_existing_fragments(out_dir_path) if skip_existing else None
    )
    if existing_fragments is not None:
        log.info(
            "[frag-twist] %s/fragment_index.json exists - "
            "skipping scission, reusing %s fragment(s)",
            out_dir_path,
            len(existing_fragments),
        )
        fragmentation_dump = None
        fragments_iter = existing_fragments
    else:
        log.info("[frag-twist] fragmenting parent -> %s", out_dir_path)
        frag_result = fragment_ligand(input_bundle, out_dir_path, config)
        log.info(
            "[frag-twist] selected %s fragment(s)",
            len(frag_result.selected_fragments),
        )
        fragmentation_dump = frag_result.to_dict()
        fragments_iter = frag_result.selected_fragments

    from ffpopt.runtime.Console import format_fragmented_run_banner, print_run_banner

    print_run_banner(
        format_fragmented_run_banner(
            ligand=mol2_path.stem,
            model=str(standard_kwargs.get("model") or "qdpi2"),
            nproc=int(nproc),
            n_fragments=len(fragments_iter),
            work_dir=str(out_dir_path),
        )
    )

    # bytype is forced True here: the per-fragment fits are merged back into
    # the parent frcmod via scission.Merge.merge_fragment_frcmods, which can
    # only map fragment-fit DIHE terms onto parent atoms by atom type - the
    # fragment's atom names don't exist in the parent topology.
    from ffpopt.runtime.FastWavefront import (
        apply_fast_wavefront_presets,
        fast_wavefront_enabled,
    )

    if fast_wavefront is True:
        os.environ["FFPOPT_FAST_WAVEFRONT"] = "1"
    elif fast_wavefront is False:
        os.environ["FFPOPT_FAST_WAVEFRONT"] = "0"
    fast_on = fast_wavefront_enabled(fast_wavefront)
    model = standard_kwargs.get("model", "qdpi2")
    fast_knobs = {
        "delta": delta,
        "wf_convergence_threshold": wf_convergence_threshold,
        "geometric_maxiter": standard_kwargs.get("geometric_maxiter", 500),
        "geometric_converge": standard_kwargs.get(
            "geometric_converge", "set GAU"
        ),
        "ase_opt_tol": standard_kwargs.get("ase_opt_tol", 0.01),
    }
    applied = apply_fast_wavefront_presets(fast_knobs, enabled=fast_on)
    if applied:
        log.info("[frag-twist] fast-wavefront presets applied: %s", applied)
        delta = int(fast_knobs["delta"])
        wf_convergence_threshold = float(fast_knobs["wf_convergence_threshold"])
        for key in ("geometric_maxiter", "geometric_converge", "ase_opt_tol"):
            if key in applied:
                standard_kwargs[key] = fast_knobs[key]

    twist_kwargs = dict(
        delta=delta,
        nprim=nprim,
        maxiter=maxiter,
        bytype=True,
        nlmaxiter=nlmaxiter,
        # nproc filled per job after fragment/wavefront split below.
        nproc=1,
        wf_starting_nodes=wf_starting_nodes,
        wf_num_conformers=wf_num_conformers,
        wf_max_levels=wf_max_levels,
        wf_convergence_threshold=wf_convergence_threshold,
        skip_existing=skip_existing,
        compare_config=compare_config,
        skip_converged_initial=skip_converged_initial,
        convergence_mode=convergence_mode,
        plot_comparisons=plot_comparisons,
        fast_wavefront=fast_on,
        multi_centroid=multi_centroid,
        centroid_mol2=centroid_mol2,
        fit_cli_args=fit_cli_args,
        **standard_kwargs,
    )

    from ffpopt.affdo.AffdoLog import describe_affdo_extras, log_affdo

    if (
        int(multi_centroid or 0) >= 2
        or bool(standard_kwargs.get("soft_dihed_restraint"))
        or bool(fit_cli_args)
    ):
        log_affdo(
            log,
            "fragment extras: %s",
            describe_affdo_extras(
                multi_centroid=multi_centroid,
                soft_dihed_restraint=bool(standard_kwargs.get("soft_dihed_restraint")),
                soft_dihed_k=standard_kwargs.get("soft_dihed_k"),
                soft_dihed_kmax=standard_kwargs.get("soft_dihed_kmax"),
                soft_dihed_tol=standard_kwargs.get("soft_dihed_tol"),
                fit_cli_args=fit_cli_args,
            ),
        )

    runnable = []
    for fragment in fragments_iter:
        if not fragment.fit_torsions:
            log.info("[frag-twist] %s: no fit_torsions - skipping", fragment.fragment_id)
            continue
        bonds = bonds0_from_scission_fit_torsions(fragment.fit_torsions)
        frag_dir = _as_path(fragment.manifest_path).parent.resolve()
        if not skip_existing:
            clear_fragment_twist_done(frag_dir)
        runnable.append(
            {
                "fragment_id": fragment.fragment_id,
                "frag_dir": str(frag_dir),
                "parm7": str(_as_path(fragment.parm7_path).resolve()),
                "rst7": str(_as_path(fragment.rst7_path).resolve()),
                "fit_torsions": fragment.fit_torsions,
                "bonds": [list(b) for b in bonds],
                "skip_existing": skip_existing,
                "twist_kwargs": twist_kwargs,
            }
        )

    if not runnable:
        raise RuntimeError(
            "no fragments had fittable torsions - nothing to merge"
        )

    already_done = []
    if skip_existing:
        # Migrate prior runs that finished before frag-twist.done existed:
        # trust the last progress-board "done" status and write the sentinel.
        prior_status_path = out_dir_path / ".frag_progress.json"
        prior_done_ids: set[str] = set()
        if prior_status_path.is_file():
            try:
                from ffpopt.runtime.ProgressBoard import FragmentProgressStore

                prior = FragmentProgressStore(prior_status_path).snapshot()
                prior_done_ids = {
                    fid
                    for fid, entry in prior.items()
                    if str((entry or {}).get("status") or "").lower() == "done"
                }
            except Exception:
                prior_done_ids = set()
        for job in runnable:
            if (
                job["fragment_id"] in prior_done_ids
                and not is_fragment_twist_done(job["frag_dir"])
            ):
                mark_fragment_twist_done(job["frag_dir"])
                log.info(
                    "[frag-twist] %s: prior progress was done - wrote %s",
                    job["fragment_id"],
                    _FRAG_TWIST_DONE,
                )

        pending = []
        for job in runnable:
            if is_fragment_twist_done(job["frag_dir"]):
                already_done.append(job)
            else:
                pending.append(job)
        if already_done:
            log.info(
                "[frag-twist] %s fragment(s) already complete - not leasing CPUs: %s",
                len(already_done),
                ", ".join(j["fragment_id"] for j in already_done),
            )
        runnable = pending

    budget_path = out_dir_path / ".cpu_budget.json"
    from ffpopt.runtime.CpuBudget import CpuBudget

    # Drop stale leases from a prior killed / timed-out parent so finished
    # owners cannot starve unfinished fragments on restart.
    CpuBudget(budget_path, nproc, clear_leases=True)

    from ffpopt.runtime.ProgressBoard import FragmentBoardWatcher, FragmentProgressStore

    status_path = out_dir_path / ".frag_progress.json"
    board_path = out_dir_path / "FRAG_STATUS.txt"
    store = FragmentProgressStore(status_path)
    for job in already_done:
        store.register(
            job["fragment_id"],
            bonds=len(job["bonds"]),
            frag_dir=job["frag_dir"],
            log_path=str(Path(job["frag_dir"]) / "frag-twist.log"),
        )
        store.update(
            job["fragment_id"],
            status="done",
            stage="finished",
            detail="skipped (complete)",
        )

    per_fragment_results = [
        {
            "fragment_id": j["fragment_id"],
            "dir": j["frag_dir"],
            "bonds": j["bonds"],
            "twist_result": None,
            "skipped_complete": True,
        }
        for j in already_done
    ]

    if not runnable:
        log.info(
            "[frag-twist] all %s fragment(s) already complete - skipping twist pool",
            len(already_done),
        )
    else:
        from ffpopt.runtime.FastWavefront import prefer_fragment_pool_depth

        prefer_depth = prefer_fragment_pool_depth(
            model=str(model),
            nproc=nproc,
            n_fragments=len(runnable),
            fast=fast_on,
        )
        n_frag_workers, _n_wf_hint = _split_fragment_nproc(
            nproc, len(runnable), prefer_depth=prefer_depth
        )
        for job in runnable:
            job["budget_path"] = str(budget_path)
            job["budget_total"] = int(nproc)
            # Hint only; workers lease dynamically from the shared budget.
            job["wf_nproc"] = max(1, int(nproc // max(1, n_frag_workers)))
            job["status_path"] = str(status_path)
            store.register(
                job["fragment_id"],
                bonds=len(job["bonds"]),
                frag_dir=job["frag_dir"],
                log_path=str(Path(job["frag_dir"]) / "frag-twist.log"),
            )

        log.info(
            "[frag-twist] parallel plan: %s unfinished fragment(s)%s, nproc=%s -> "
            "%s concurrent fragment worker(s)%s; dynamic CPU leases via %s",
            len(runnable),
            f" (+{len(already_done)} complete)" if already_done else "",
            nproc,
            n_frag_workers,
            " (prefer wf depth)" if prefer_depth else "",
            budget_path,
        )
        log.info(
            "[frag-twist] live status board: %s "
            "(per-fragment detail: <frag>/frag-twist.log)",
            board_path,
        )
        watcher = FragmentBoardWatcher(
            store,
            board_path=board_path,
            logger=log,
            interval_sec=5.0,
            log_root_hint="<out_dir>/<fragment>/frag-twist.log",
        )
        watcher.start()
        try:
            if n_frag_workers == 1:
                for i, job in enumerate(runnable, start=1):
                    result = _run_fragment_twist_job(job)
                    per_fragment_results.append(result)
                    log.info(
                        "[frag-twist] fragment job finished (%s/%s): %s",
                        i,
                        len(runnable),
                        result["fragment_id"],
                    )
            else:
                pool = make_nondaemon_spawn_pool(n_frag_workers)
                try:
                    by_id: dict[str, dict] = {}
                    finished = 0
                    for result in pool.imap_unordered(
                        _run_fragment_twist_job, runnable
                    ):
                        finished += 1
                        by_id[result["fragment_id"]] = result
                        log.info(
                            "[frag-twist] fragment job finished (%s/%s): %s",
                            finished,
                            len(runnable),
                            result["fragment_id"],
                        )
                    missing = [
                        j["fragment_id"]
                        for j in runnable
                        if j["fragment_id"] not in by_id
                    ]
                    if missing:
                        raise RuntimeError(
                            "fragment pool returned incomplete results; missing: "
                            + ", ".join(missing)
                        )
                    per_fragment_results.extend(
                        by_id[j["fragment_id"]] for j in runnable
                    )
                finally:
                    pool.close()
                    pool.join()
        finally:
            watcher.stop()

    log.info(
        "[frag-twist] all %s fragment twist job(s) finished",
        len(per_fragment_results),
    )

    fragment_dirs_for_merge = [Path(r["dir"]) for r in per_fragment_results]

    log.info(
        "[frag-twist] merging %s fragment frcmod(s) -> %s",
        len(fragment_dirs_for_merge),
        merged_frcmod_path,
    )
    report_path = merged_frcmod_path.with_name(
        merged_frcmod_path.name + ".merge_report.json"
    )
    merge_report = merge_fragment_frcmods(
        parent_frcmod_path=parent_frcmod,
        output_frcmod_path=merged_frcmod_path,
        fragment_dirs=fragment_dirs_for_merge,
        report_path=report_path,
    )

    return {
        "fragmentation": fragmentation_dump,
        "fragments": per_fragment_results,
        "merge_report": merge_report,
        "merged_frcmod": str(merged_frcmod_path),
    }
