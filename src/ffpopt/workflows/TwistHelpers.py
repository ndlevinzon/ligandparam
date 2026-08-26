"""Shared helpers for ffpopt twist / fragment / whole-ligand workflows."""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ffpopt.runtime.NondaemonPool import make_nondaemon_spawn_pool


_LOG = logging.getLogger("ffpopt.workflows")

PathLike = Union[str, Path]

# Central bond of a proper dihedral as a pair of **0-based** atom indices
# (ParmEd / ffpopt convention). Scission fit_torsions use 1-based indices;
# convert at the boundary with :func:`bonds0_from_scission_fit_torsions`.
BondPair0 = tuple[int, int]


def _as_path(value: PathLike) -> Path:
    """Normalize path-like inputs to :class:`pathlib.Path`."""
    return value if isinstance(value, Path) else Path(value)


def _in_workdir(workdir: Optional[Path], name: PathLike) -> Path:
    """Resolve ``name`` under ``workdir`` when relative; leave absolutes alone."""
    path = _as_path(name)
    if workdir is None or path.is_absolute():
        return path
    return workdir / path


def _subprocess_cwd(workdir: Optional[Path]) -> Optional[str]:
    """Return a ``cwd`` string for ``subprocess.run``, or ``None``."""
    return None if workdir is None else str(workdir)


def _resolve_logger(logger: logging.Logger | None) -> logging.Logger:
    """Return ``logger`` or the module logger for workflow progress messages.

    Ensures ``ffpopt.*`` loggers mirror INFO to stdout and WARNING+ to stderr
    with a single timestamp and hierarchical ``[tag]`` brackets. Does not
    replace handlers already attached (e.g. per-fragment loggers).
    """
    log = logger if logger is not None else _LOG
    name = getattr(log, "name", "") or ""
    if name == "ffpopt" or name.startswith("ffpopt."):
        from ffpopt.runtime.Console import attach_console_handlers

        attach_console_handlers(log, tag="ffpopt")
    return log


def _ffpopt_bin_script(script_name: str) -> str:
    """Absolute path to a script under ``ffpopt.bin``."""
    from importlib import resources

    import ffpopt.bin as bin_pkg

    return str(resources.files(bin_pkg).joinpath(script_name))


def _run_current_python(
    script: PathLike,
    *args: str,
    cwd: Optional[str] = None,
) -> None:
    """Run ``script`` with ``sys.executable`` (same env as the parent process).

    Bare ``python3`` on HPC often resolves to a system interpreter without
    ParmEd / ffpopt, which breaks generated fit scripts (``it01.py``, ...).
    """
    import os

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(
        [sys.executable, "-u", str(script), *args],
        check=True,
        cwd=cwd,
        env=env,
    )


def _run_fit_script_inprocess(
    script: PathLike,
    iparm: PathLike,
    oparm: PathLike,
) -> None:
    """Apply a GenDihedFit ``itNN.py`` script in-process (no nested python).

    Avoids silent hangs / wrong-interpreter failures from shelling out to
    ``python3``. Progress prints from the generated script go to this process.
    """
    import runpy

    script = str(Path(script).resolve())
    iparm = str(Path(iparm).resolve())
    oparm = str(Path(oparm).resolve())
    old_argv = sys.argv[:]
    sys.argv = [script, iparm, oparm]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = old_argv
    if not Path(oparm).is_file():
        raise FileNotFoundError(
            f"fit script {script} finished without writing {oparm}. "
            "A truncated itNN.py (GenDihedFit crash mid-write) has no p.save; "
            "delete itNN.py / itNN.frcmod and rerun so GenDihedFit rewrites them."
        )


def _run_ffpopt_bin(
    script_name: str,
    *args: str,
    cwd: Optional[str] = None,
) -> None:
    """Run an ``ffpopt.bin`` console script with the current interpreter."""
    _run_current_python(_ffpopt_bin_script(script_name), *args, cwd=cwd)


def normalize_bond_pairs0(bond) -> list[BondPair0]:
    """Normalize central bonds to 0-based ``(i, j)`` tuples.

    Parameters
    ----------
    bond
        Iterable of:

        * ``(i, j)`` or ``[i, j]`` - **0-based** atom indices (preferred API)
        * ``"i,j"`` - CLI string form (also **0-based**)

    Returns
    -------
    list of tuple[int, int]
        Central bond pairs for ffpopt scans / fits.

    Raises
    ------
    TypeError, ValueError
        If an entry is not a pair of integers or a ``"i,j"`` string.
    """
    if bond is None:
        raise TypeError("bond must be an iterable of pairs or 'i,j' strings")
    out: list[BondPair0] = []
    for entry in bond:
        if isinstance(entry, str):
            parts = entry.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"bond string must be 'i,j' (0-based); got {entry!r}"
                )
            out.append((int(parts[0]), int(parts[1])))
            continue
        try:
            a, b = entry
            out.append((int(a), int(b)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "each bond must be a 0-based (i, j) pair or an 'i,j' string; "
                f"got {entry!r}"
            ) from exc
    return out


def bonds0_from_scission_fit_torsions(fit_torsions) -> list[BondPair0]:
    """Map scission ``fit_torsions`` (1-based) to ffpopt central bonds (0-based).

    Each record's ``fragment_rotatable_bond`` is a two-element list of
    **1-based** fragment-local atom indices. Returns **0-based** pairs for
    :func:`run_dihed_twist_workflow`.
    """
    bonds: list[BondPair0] = []
    for record in fit_torsions:
        pair = record["fragment_rotatable_bond"]
        if len(pair) != 2:
            raise ValueError(
                "fragment_rotatable_bond must have two 1-based indices; "
                f"got {pair!r}"
            )
        bonds.append((int(pair[0]) - 1, int(pair[1]) - 1))
    return bonds


def _parent_paths_from_args(
    *,
    mol2: PathLike | None,
    lib: PathLike | None,
    frcmod: PathLike | None,
    bundle: Any | None,
) -> tuple[Path, Path, Path]:
    """Resolve parent mol2/lib/frcmod from paths or a duck-typed bundle.

    Accepts :class:`~ligandparam.io.AmberBundle.AmberLigandBundle`
    (``mol2`` / ``lib`` / ``frcmod``) or :class:`scission.Models.InputBundle`
    (``mol2_path`` / ``lib_path`` / ``frcmod_path``).
    """
    if bundle is not None:
        if all(hasattr(bundle, attr) for attr in ("mol2", "lib", "frcmod")):
            return (
                _as_path(bundle.mol2).resolve(),
                _as_path(bundle.lib).resolve(),
                _as_path(bundle.frcmod).resolve(),
            )
        if all(
            hasattr(bundle, attr)
            for attr in ("mol2_path", "lib_path", "frcmod_path")
        ):
            return (
                _as_path(bundle.mol2_path).resolve(),
                _as_path(bundle.lib_path).resolve(),
                _as_path(bundle.frcmod_path).resolve(),
            )
        raise TypeError(
            "bundle must provide mol2/lib/frcmod or mol2_path/lib_path/frcmod_path"
        )
    if mol2 is None or lib is None or frcmod is None:
        raise TypeError(
            "run_fragmented_dihed_twist_workflow requires mol2, lib, and frcmod "
            "(or a bundle= AmberLigandBundle / InputBundle)"
        )
    return (
        _as_path(mol2).resolve(),
        _as_path(lib).resolve(),
        _as_path(frcmod).resolve(),
    )


class _TwistParam(object):
    """ Per-bond torsion parameter record used by the twist workflow.

    Copied from ``bin/ffpopt-DihedTwistWorkflow.py``'s ``Parameter`` class
    so this module has no dependency on the bin script. If the bin-script
    class diverges meaningfully, deduplicate.

    Attributes
    ----------
    idxs : list of int
        The four 0-based atom indices of the proper dihedral.
    res : str
        Residue name of the first atom.
    names : list of str
        Atom names corresponding to ``idxs``.
    types : list of str
        Atom force-field types corresponding to ``idxs``.
    instances : list
        Per-dihedral-instance mask lists accumulated across bonds that
        share this parameter family.
    """

    def __init__(self, mol, idxs):
        self.idxs = idxs
        self.res = mol.atoms[idxs[0]].residue.name
        self.names = [mol.atoms[i].name for i in idxs]
        self.types = [mol.atoms[i].type for i in idxs]
        self.instances = []

    def GetIdxStr(self):
        return "-".join("%i" % x for x in self.idxs)

    def GetTypeStr(self):
        return "-".join(self.types)

    def GetParamByType(self):
        return "%s_%s" % (self.res, self.GetTypeStr())

    def GetNameMasks(self):
        return [f"@{n}" for n in self.names]


def _resolve_scans_and_params(mol, bonds, nprim: int, bytype: bool):
    """ Walk ``mol.dihedrals`` once per bond, build the scan list and fit params.

    Parameters
    ----------
    mol : parmed.Structure
        Parsed amber topology. Used only for its ``dihedrals`` list and
        atom records.
    bonds : list of tuple[int, int] or list of list of int
        Central-bond atom pairs as ``[(a, b), ...]`` (**0-based** indices).
    nprim : int
        Number of primary cosine terms to fit per parameter family.
    bytype : bool
        If True, fit-input masks are by atom *type* (one entry per unique
        type string). If False, masks list every atom-name instance
        explicitly.

    Returns
    -------
    scans : list of _TwistParam
        One entry per bond (its first matching proper dihedral).
    params : dict
        Maps parameter name -> ``{'nprim': nprim, 'masks': ...}`` for the
        ``ffpopt-GenDihedFit.py`` input.
    s_template : dict
        Per-system fit-input template with ``params`` filled in and
        ``profiles`` empty.
    """
    scans = []
    allparams = []
    ps = {}
    for bond in bonds:
        made_scan = False
        for d in mol.dihedrals:
            if d.improper:
                continue
            idxs = [d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx]
            if idxs[1] == bond[0] and idxs[2] == bond[1]:
                myidxs = idxs
            elif idxs[2] == bond[0] and idxs[1] == bond[1]:
                myidxs = idxs[::-1]
            else:
                continue
            p = _TwistParam(mol, myidxs)
            name = p.GetParamByType()
            if name not in ps:
                ps[name] = p
            ps[name].instances.append(p.GetNameMasks())
            allparams.append(name)
            if not made_scan:
                scans.append(p)
                made_scan = True
        if not made_scan:
            raise ValueError(
                f"--bond {bond[0]},{bond[1]} has no proper dihedral with that "
                f"pair as the central bond. This usually means at least one of "
                f"the two atoms is terminal (no bonded neighbors beyond the "
                f"other). Check your bond indices (0-based) against the parm "
                f"topology."
            )

    uparams = list(set(allparams))
    s_template = {"output": None, "params": {}, "profiles": []}
    params = {}
    if not bytype:
        for name in ps:
            s_template["params"][name] = ps[name].instances
        for name in uparams:
            params[name] = {"nprim": nprim, "masks": None}
    else:
        for name in uparams:
            typestr = name.split("_")[1]
            ts = [f"@%{t}" for t in typestr.split("-")]
            params[name] = {"nprim": nprim, "masks": [ts]}

    return scans, params, s_template


def _scan_json_nframes(path: Path) -> int | None:
    """Frame count from a wavefront scan JSON, or None if unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, list):
        return len(data)
    if not isinstance(data, dict):
        return None
    structs = data.get("structs") or data.get("structures")
    if isinstance(structs, list):
        return len(structs)
    return None


def existing_scan_grid_mismatch(path: Path, delta) -> bool:
    """True when ``path`` looks like a uniform 360/n grid that is not ``delta``."""
    if delta is None:
        return False
    n = _scan_json_nframes(path)
    if not n:
        return False
    expected = max(1, 360 // int(delta))
    if n == expected:
        return False
    return bool(n > 0 and 360 % n == 0)


def scan_outputs_complete(path: Path, delta=None) -> bool:
    """True when ``skip_existing`` may reuse this scan (JSON + ``.dat`` + grid).

    A leftover JSON without its companion ``.dat``, an unreadable/empty JSON, or
    a frame count that is not ``360/delta`` must be rescanned. Incomplete
    wavefronts (killed after JSON, before ``.dat``) used to be skipped, then
    compare crashed in ``np.loadtxt``.
    """
    json_path = Path(path)
    if json_path.suffix.lower() != ".json":
        json_path = json_path.with_suffix(".json")
    if not json_path.is_file():
        return False
    dat_path = json_path.with_suffix(".dat")
    if not dat_path.is_file() or dat_path.stat().st_size < 1:
        return False
    n = _scan_json_nframes(json_path)
    if not n:
        return False
    if delta is None:
        return True
    expected = max(1, 360 // int(delta))
    return n == expected


def gendihedfit_outputs_complete(py_path: PathLike, frcmod_path: PathLike | None = None) -> bool:
    """True when ``skip_existing`` may reuse a GenDihedFit ``itNN.py``.

    A crash inside ``WriteParmedScript`` leaves a truncated ``itNN.py`` with no
    ``p.save`` (and usually no ``itNN.frcmod``). Reusing that script "succeeds"
    without writing ``itNN.parm7``, then PrepareInput dies on a missing parm.
    """
    py_path = Path(py_path)
    if frcmod_path is None:
        frcmod_path = py_path.with_suffix(".frcmod")
    else:
        frcmod_path = Path(frcmod_path)
    if not py_path.is_file() or py_path.stat().st_size < 1:
        return False
    if not frcmod_path.is_file() or frcmod_path.stat().st_size < 1:
        return False
    try:
        text = py_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "p.save(" in text


def _is_sander_ll_model(model: str | None) -> bool:
    m = (model or "").strip().lower()
    return m in {"sander", "amber", "mm"} or m.startswith("sander")


def _wf_kwargs_for_scan_model(model: str, wf_kwargs: dict) -> dict:
    """Specialize wavefront kwargs per energy model.

    Sander / Amber MM wavefronts default to ASE-first constrained opts (skip the
    geomeTRIC recovery ladder) - the dominant wall-time win for ``orig`` /
    ``rescan/itNN`` stages. Under ``--fast``, XTB-like and QDpi2 HL scans also
    default to ASE-first. Explicit ``geometric_opt=True`` in ``wf_kwargs``
    still wins.
    """
    from ffpopt.runtime.FastWavefront import (
        fast_wavefront_enabled,
        prefer_ase_first_model,
    )

    out = dict(wf_kwargs)
    fast = out.get("fast_wavefront")
    if fast is None:
        fast = fast_wavefront_enabled(None)
    if "geometric_opt" not in out and prefer_ase_first_model(model, fast=bool(fast)):
        out["geometric_opt"] = False
    return out


def _prior_ll_checkpoint_path(
    *,
    workdir: Optional[Path],
    seed_prefix: str | None,
    idx_str: str,
) -> Optional[str]:
    """Path to ``checkpoint_{seed_prefix}_{idx}.pkl`` when present (warm-start)."""
    if not seed_prefix:
        return None
    prior_out = f"{seed_prefix}_{idx_str}.json"
    prior_ckpt = f"checkpoint_{Path(prior_out).with_suffix('.pkl').name}"
    path = Path(_in_workdir(workdir, prior_ckpt))
    if path.is_file():
        return str(path.resolve())
    return None


def _run_one_scan(
    *,
    inp: str,
    model: str,
    dihed_idxs,
    out: str,
    skip_existing: bool,
    logger: logging.Logger | None = None,
    workdir: Optional[Path] = None,
    **wf_kwargs,
):
    """ Run one wavefront scan via :func:`ffpopt.scan.WaveFront.run_dihed_wavefront`.

    Parameters
    ----------
    inp : str
        Path to the input ``ListOfStruct`` JSON.
    model : str
        ``--model`` value (e.g. ``"qdpi2"``, ``"sander"``).
    dihed_idxs : sequence of int
        Four 0-based atom indices defining the dihedral to scan.
    out : str
        Output JSON path. If it already exists and ``skip_existing`` is
        True, the scan is skipped.
    skip_existing : bool
        Whether to skip the scan when ``out`` is already on disk.
    logger
        Optional logger for progress messages.
    workdir : pathlib.Path, optional
        When set, relative ``inp`` / ``out`` paths are resolved under this
        directory (absolute paths are left unchanged).
    **wf_kwargs
        Forwarded unchanged to :func:`ffpopt.scan.WaveFront.run_dihed_wavefront`.

    Returns
    -------
    dict or None
        The dict returned by ``run_dihed_wavefront``, or ``None`` if the
        scan was skipped because ``out`` already exists.
    """
    from ffpopt.scan.WaveFront import run_dihed_wavefront

    log = _resolve_logger(logger)
    inp_path = str(_in_workdir(workdir, inp))
    out_path = str(_in_workdir(workdir, out))
    if skip_existing and Path(out_path).exists():
        if scan_outputs_complete(Path(out_path), wf_kwargs.get("delta")):
            log.info("[twist] %s exists - skipping.", out_path)
            from ffpopt.geom.Geometric import sweep_geometric_scratch_dir

            n = sweep_geometric_scratch_dir(Path(out_path).parent, recursive=True)
            if n:
                log.info("[twist] removed %s leftover geomeTRIC scratch path(s)", n)
            return None
        log.warning(
            "[twist] %s exists but is incomplete or its angle grid does not "
            "match delta=%s (need JSON + .dat with 360/delta frames); rescanning",
            out_path,
            wf_kwargs.get("delta"),
        )

    dihed_str = ",".join(str(i) for i in dihed_idxs)
    log.info(
        "[twist] scan: inp=%s model=%s dihed=%s out=%s",
        inp_path,
        model,
        dihed_str,
        out_path,
    )
    scan_kwargs = _wf_kwargs_for_scan_model(model, wf_kwargs)
    scan_kwargs.pop("fast_wavefront", None)
    seed_ckpt = scan_kwargs.pop("seed_checkpoint", None)
    out_ckpt = Path(out_path).parent / f"checkpoint_{Path(out_path).with_suffix('.pkl').name}"
    if seed_ckpt and not out_ckpt.is_file():
        # Fresh rescan: warm-start geometries from the prior LL wavefront.
        scan_kwargs["wf_alt_starting_checkpoint"] = str(seed_ckpt)
        scan_kwargs["wf_change_theory"] = True
        log.info("[twist] warm-start from prior LL checkpoint %s", seed_ckpt)
    return run_dihed_wavefront(
        inp=inp_path,
        out=out_path,
        dihed=dihed_str,
        model=model,
        **scan_kwargs,
    )


def _slim_scan_result(scan_result: Optional[dict]) -> Optional[dict]:
    """Drop heavy ``wf_run`` objects so bond-pool IPC stays picklable."""
    from ffpopt.runtime.SlimIpc import slim_scan_result

    return slim_scan_result(scan_result)


def _run_bond_scan_job(job: dict) -> dict:
    """Worker entry: one wavefront scan for one central bond (picklable)."""
    workdir = job.get("workdir")
    r = _run_one_scan(
        inp=job["inp"],
        model=job["model"],
        dihed_idxs=job["dihed_idxs"],
        out=job["out"],
        skip_existing=job["skip_existing"],
        logger=_LOG,
        workdir=Path(workdir) if workdir else None,
        **job["wf_kwargs"],
    )
    return {
        "prefix": job["prefix"],
        "dihed_idxs": list(job["dihed_idxs"]),
        "result": _slim_scan_result(r),
    }


def _build_bond_scan_jobs(
    scans,
    *,
    prefix: str,
    model: str,
    inp: str,
    skip_existing: bool,
    workdir: Optional[Path],
    wf_kwargs: dict,
    seed_prefix: str | None = None,
) -> list[dict]:
    """Build picklable bond-scan job dicts for one (prefix, model) phase."""
    base_kwargs = _wf_kwargs_for_scan_model(model, wf_kwargs)
    jobs = []
    for scan in scans:
        idx_str = scan.GetIdxStr()
        job_kwargs = dict(base_kwargs)
        seed = _prior_ll_checkpoint_path(
            workdir=workdir, seed_prefix=seed_prefix, idx_str=idx_str
        )
        if seed:
            job_kwargs["seed_checkpoint"] = seed
        jobs.append(
            {
                "prefix": prefix,
                "inp": inp,
                "model": model,
                "dihed_idxs": list(scan.idxs),
                "out": f"{prefix}_{idx_str}.json",
                "skip_existing": skip_existing,
                "workdir": str(workdir) if workdir is not None else None,
                "wf_kwargs": job_kwargs,
            }
        )
    return jobs


def _interleave_job_groups(*groups: list[dict]) -> list[dict]:
    """Round-robin independent job lists so cheap orig work can fill cores."""
    groups = [g for g in groups if g]
    if not groups:
        return []
    if len(groups) == 1:
        return list(groups[0])
    out: list[dict] = []
    n = max(len(g) for g in groups)
    for i in range(n):
        for g in groups:
            if i < len(g):
                out.append(g[i])
    return out


def _split_fragment_nproc(
    nproc: int,
    n_fragments: int,
    *,
    prefer_depth: bool = False,
    flatten_nested: bool = True,
) -> tuple[int, int]:
    """Split ``nproc`` across outer workers and nested wavefront size.

    Used for both fragment-level pooling and per-bond scan pooling inside
    :func:`run_dihed_twist_workflow`.

    Returns
    -------
    tuple of int
        ``(n_workers, n_wavefront_per_worker)`` such that
        ``n_workers * n_wavefront_per_worker <= nproc`` (when
        ``n_items > 1``). By default prefers as many outer workers as
        possible; with ``prefer_depth=True`` keeps a minimum inner width
        (see :func:`ffpopt.runtime.FastWavefront.split_nproc_for_items`).
        Pass ``flatten_nested=False`` to keep a 2-D bondxwavefront split
        (whole-ligand / top-level twist). Fragment spawn workers keep the
        default flatten so they do not open a third pool.
    """
    from ffpopt.runtime.FastWavefront import split_nproc_for_items

    return split_nproc_for_items(
        nproc, n_fragments, prefer_depth=prefer_depth, flatten_nested=flatten_nested
    )


def _execute_bond_scan_jobs(
    jobs: list[dict],
    *,
    nproc: int,
    prefer_wf_depth: bool | None,
    logger: logging.Logger | None,
    label: str = "scans",
) -> list[tuple[str, tuple, Optional[dict]]]:
    """Run bond-scan jobs; nest bondxwavefront only at the top-level twist."""
    log = _resolve_logger(logger)
    if not jobs:
        return []

    from ffpopt.runtime.FastWavefront import prefer_bond_pool_depth
    from ffpopt.runtime.NondaemonPool import in_spawn_worker
    from ffpopt.scan.WaveFront import close_reused_wavefront_pool

    models = {str(j.get("model") or "") for j in jobs}
    model_hint = next(iter(models)) if len(models) else None
    prefer = prefer_bond_pool_depth(
        model=model_hint,
        nproc=nproc,
        n_bonds=len(jobs),
        prefer=prefer_wf_depth,
    )
    # Already inside a fragment spawn worker: one axis only (wavefront depth).
    already_nested = in_spawn_worker()
    if already_nested and prefer is not True:
        prefer = True

    flatten = already_nested or not prefer
    n_bond_workers, n_wf = _split_fragment_nproc(
        nproc, len(jobs), prefer_depth=prefer, flatten_nested=flatten
    )
    for job in jobs:
        job["wf_kwargs"] = dict(job["wf_kwargs"])
        job["wf_kwargs"]["nproc"] = int(n_wf)

    ase_first = any(j["wf_kwargs"].get("geometric_opt") is False for j in jobs)
    if prefer and n_bond_workers > 1 and n_wf > 1:
        split_note = " (prefer wf depth; nested spawn)"
    elif prefer:
        split_note = " (prefer wf depth)"
    else:
        split_note = " (flat; no nested spawn)"
    used = int(n_bond_workers) * int(n_wf)
    leftover = max(0, int(nproc) - used)
    leftover_note = f", {used}/{nproc} cores" + (
        f" ({leftover} leftover)" if leftover else ""
    )
    log.info(
        "[twist] parallel bond scans: %s, %s job(s), nproc=%s -> "
        "%s bond worker(s) x wf_nproc=%s%s%s%s",
        label,
        len(jobs),
        nproc,
        n_bond_workers,
        n_wf,
        split_note,
        leftover_note,
        " (ASE-first)" if ase_first else "",
    )

    try:
        if n_bond_workers == 1:
            raw = [_run_bond_scan_job(job) for job in jobs]
        else:
            pool = make_nondaemon_spawn_pool(n_bond_workers)
            try:
                raw = pool.map(_run_bond_scan_job, jobs)
            finally:
                pool.close()
                pool.join()
    finally:
        close_reused_wavefront_pool()

    return [
        (item["prefix"], tuple(item["dihed_idxs"]), item["result"])
        for item in raw
    ]


def _run_scans_for_bonds(
    scans,
    *,
    prefix: str,
    model: str,
    inp: str,
    nproc: int,
    skip_existing: bool,
    workdir: Optional[Path],
    logger: logging.Logger | None,
    wf_kwargs: dict,
    prefer_wf_depth: bool | None = None,
    seed_prefix: str | None = None,
) -> list[tuple[str, tuple, Optional[dict]]]:
    """Run one wavefront scan per bond, pooling when the core budget allows.

    Splits ``nproc`` as ``n_bond_workers x wf_nproc`` (flattened so both are
    never >1 - nested spawn pools are too expensive). When ``seed_prefix`` is
    set (e.g. ``\"orig\"`` before ``it01``), each bond warm-starts from that
    prefix's wavefront checkpoint if present.
    """
    jobs = _build_bond_scan_jobs(
        scans,
        prefix=prefix,
        model=model,
        inp=inp,
        skip_existing=skip_existing,
        workdir=workdir,
        wf_kwargs=wf_kwargs,
        seed_prefix=seed_prefix,
    )
    return _execute_bond_scan_jobs(
        jobs,
        nproc=nproc,
        prefer_wf_depth=prefer_wf_depth,
        logger=logger,
        label=f"prefix={prefix}",
    )


def _log_centroid_profile_rows(log, idx: str, rows: list, *, best=None) -> None:
    from ffpopt.affdo.AffdoLog import log_affdo

    best_path = Path(best) if best is not None else None
    for row in rows:
        err = row.get("error")
        if err:
            log_affdo(log, "%s skip %s (%s)", idx, Path(row["path"]).name, err)
            continue
        marker = ""
        if best_path is not None and Path(row["path"]) == best_path:
            marker = " <- selected"
        log_affdo(
            log,
            "%s %s: score=%.4g fourier=%.4g roughness=%.4g npts=%s%s",
            idx,
            Path(row["path"]).name,
            row["score"],
            row["fourier"],
            row["roughness"],
            row["npts"],
            marker,
        )


def _promote_centroid_pick(
    log,
    *,
    idx: str,
    hl_prefix: str,
    workdir: Optional[Path],
    candidates: list,
) -> None:
    from ffpopt.affdo.AffdoLog import log_affdo
    from ffpopt.affdo.CentroidProfiles import pick_smoothest_profile, promote_profile_files

    best, score, rows = pick_smoothest_profile(candidates)
    if best is None:
        detail = []
        for row in rows or []:
            p = Path(row.get("path") or "?")
            err = row.get("error") or f"score={row.get('score')}"
            detail.append(f"  {p.name}: {err}")
        if not detail:
            detail = [f"  {Path(c).name}: missing" for c in candidates]
        raise FileNotFoundError(
            f"[affdo] {idx}: no usable centroid HL profiles to promote to "
            f"{hl_prefix}_{idx}.dat. Compare needs that file. Candidates:\n"
            + "\n".join(detail)
            + "\nIf this is a restart, delete the matching "
            f"{hl_prefix}.c*_{idx}.json files (and checkpoints) so the HL "
            "scan is not skip_existing'd without a .dat."
        )
    _log_centroid_profile_rows(log, idx, rows, best=best)
    src_stem = Path(best)
    dst_stem = _in_workdir(workdir, f"{hl_prefix}_{idx}")
    promote_profile_files(src_stem, dst_stem)
    log_affdo(
        log,
        "%s: promoted %s -> %s.* (score=%.4g, %s centroid(s))",
        idx,
        Path(best).name,
        dst_stem.name,
        score,
        len(rows),
    )


def _run_hl_and_orig_scans(
    scans,
    *,
    hl_prefix: str,
    hl_model: str,
    inp: str,
    nproc: int,
    skip_existing: bool,
    workdir: Optional[Path],
    logger: logging.Logger | None,
    wf_kwargs: dict,
    prefer_wf_depth: bool | None = None,
    multi_centroid: int = 0,
    centroid_mol2: Optional[PathLike] = None,
) -> list[tuple[str, tuple, Optional[dict]]]:
    """Pipeline HL (optionally multi-centroid) and reference-sander scans.

    Independent HL and ``orig`` jobs share one pool. With
    ``multi_centroid >= 2``, centroid-0 HL is scored first; extra ConfSearch
    starts run only for jagged torsions (Fourier RMSE), and those extra
    centroidxbond jobs share one pool. The smoothest HL profile is promoted
    to ``{hl_prefix}_{idxs}.*``. ``orig`` always starts from primary ``inp``.
    """
    log = _resolve_logger(logger)
    results: list[tuple[str, tuple, Optional[dict]]] = []
    n_cent = max(0, int(multi_centroid or 0))
    orig_jobs = _build_bond_scan_jobs(
        scans,
        prefix="orig",
        model="sander",
        inp=inp,
        skip_existing=skip_existing,
        workdir=workdir,
        wf_kwargs=wf_kwargs,
    )
    exec_kw = dict(
        nproc=nproc,
        prefer_wf_depth=prefer_wf_depth,
        logger=logger,
    )

    if n_cent >= 2:
        from ffpopt.affdo.AffdoLog import log_affdo
        from ffpopt.affdo.CentroidProfiles import (
            generate_centroid_start_jsons,
            profile_is_smooth_enough,
            score_profile_details,
        )

        cent_starts = generate_centroid_start_jsons(
            inp,
            mol2_path=centroid_mol2,
            nkeep=n_cent,
            workdir=workdir,
            logger=log,
        )
        n_starts = len(cent_starts)
        log_affdo(
            log,
            "multi-centroid HL: %s start(s); centroid-0 + orig share one pool; "
            "extra starts only if Fourier RMSE > FFPOPT_CENTROID_FOURIER_MAX",
            n_starts,
        )
        c0_jobs = _build_bond_scan_jobs(
            scans,
            prefix=f"{hl_prefix}.c0",
            model=hl_model,
            inp=str(cent_starts[0]),
            skip_existing=skip_existing,
            workdir=workdir,
            wf_kwargs=wf_kwargs,
        )
        results.extend(
            _execute_bond_scan_jobs(
                _interleave_job_groups(c0_jobs, orig_jobs),
                label=f"HL({hl_model}).c0+orig(sander)",
                **exec_kw,
            )
        )
        extra_scans = []
        for scan in scans:
            idx = scan.GetIdxStr()
            c0_dat = _in_workdir(workdir, f"{hl_prefix}.c0_{idx}.dat")
            row = score_profile_details(c0_dat)
            if n_starts > 1 and profile_is_smooth_enough(row):
                log_affdo(
                    log,
                    "%s centroid-0 kept (fourier=%.4g npts=%s); skip extra starts",
                    idx,
                    row["fourier"],
                    row["npts"],
                )
                _promote_centroid_pick(
                    log,
                    idx=idx,
                    hl_prefix=hl_prefix,
                    workdir=workdir,
                    candidates=[c0_dat],
                )
            elif n_starts > 1:
                extra_scans.append(scan)
                why = row.get("error") or f"fourier={row.get('fourier')}"
                log_affdo(log, "%s needs extra centroids (%s)", idx, why)
            else:
                _promote_centroid_pick(
                    log,
                    idx=idx,
                    hl_prefix=hl_prefix,
                    workdir=workdir,
                    candidates=[c0_dat],
                )
        if extra_scans and n_starts > 1:
            extra_groups = [
                _build_bond_scan_jobs(
                    extra_scans,
                    prefix=f"{hl_prefix}.c{ci}",
                    model=hl_model,
                    inp=str(cent_inp),
                    skip_existing=skip_existing,
                    workdir=workdir,
                    wf_kwargs=wf_kwargs,
                )
                for ci, cent_inp in enumerate(cent_starts[1:], start=1)
            ]
            results.extend(
                _execute_bond_scan_jobs(
                    _interleave_job_groups(*extra_groups),
                    label=f"HL({hl_model}).c1-{n_starts - 1} extra centroids",
                    **exec_kw,
                )
            )
            for scan in extra_scans:
                idx = scan.GetIdxStr()
                cands = [
                    _in_workdir(workdir, f"{hl_prefix}.c{ci}_{idx}.dat")
                    for ci in range(n_starts)
                ]
                _promote_centroid_pick(
                    log,
                    idx=idx,
                    hl_prefix=hl_prefix,
                    workdir=workdir,
                    candidates=cands,
                )
    else:
        hl_jobs = _build_bond_scan_jobs(
            scans,
            prefix=hl_prefix,
            model=hl_model,
            inp=inp,
            skip_existing=skip_existing,
            workdir=workdir,
            wf_kwargs=wf_kwargs,
        )
        results.extend(
            _execute_bond_scan_jobs(
                _interleave_job_groups(hl_jobs, orig_jobs),
                label=f"HL({hl_model})+orig(sander)",
                **exec_kw,
            )
        )
    return results


def _write_fit_json(
    *,
    citname: str,
    scans,
    params: dict,
    s_template: dict,
    hl_prefix: str,
    ll_prefix: str,
    parm: str,
    workdir: Optional[Path] = None,
) -> str:
    """ Write ``<citname>.fit.json`` for ``ffpopt-GenDihedFit.py``.

    Parameters
    ----------
    citname : str
        Current-iteration name (e.g. ``"it01"``).
    scans : list of _TwistParam
        Per-bond scan records from :func:`_resolve_scans_and_params`.
    params : dict
        Parameter family -> fit-input metadata.
    s_template : dict
        Per-system fit-input template with ``params`` filled in.
    hl_prefix : str
        Filename prefix of the HL scan JSONs (e.g. ``"qdpi2"``).
    ll_prefix : str
        Filename prefix of the LL scan JSONs (e.g. ``"orig"`` or
        ``"it00"``).
    parm : str
        Path to the current parm7 (origparm on the first iteration,
        ``<prev>.parm7`` thereafter).
    workdir : pathlib.Path, optional
        When set, the fit JSON and all referenced outputs/profiles are
        written as absolute paths under this directory so GenDihedFit does
        not depend on process cwd.

    Returns
    -------
    str
        Path to the written ``<citname>.fit.json`` file.
    """
    ss = copy.deepcopy(s_template)
    # Absolute outputs / profiles: GenDihedFit opens these relative to its
    # own process cwd; hardcoding absolutes avoids silent writes to the
    # launch directory if cwd is wrong.
    py_out = str(_in_workdir(workdir, f"{citname}.py").resolve())
    frcmod_out = str(_in_workdir(workdir, f"{citname}.frcmod").resolve())
    ss["output"] = py_out
    ss["parm"] = str(_in_workdir(workdir, parm).resolve())
    ss["profiles"] = [
        {
            "hl": str(
                _in_workdir(workdir, f"{hl_prefix}_{scan.GetIdxStr()}.json").resolve()
            ),
            "ll": str(
                _in_workdir(workdir, f"{ll_prefix}_{scan.GetIdxStr()}.json").resolve()
            ),
            "name": citname,
            "plots": [scan.GetParamByType()],
        }
        for scan in scans
    ]
    datadict = {
        "params": params,
        "output": frcmod_out,
        "systems": [ss],
    }
    out_path = _in_workdir(workdir, f"{citname}.fit.json").resolve()
    with open(out_path, "w") as fh:
        json.dump(datadict, fh, indent=4)
    return str(out_path)


def _require_files(paths: list[PathLike], *, step: str) -> None:
    """Raise if any expected output path is missing after a workflow step."""
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{step}: expected output file(s) missing:\n  "
            + "\n  ".join(missing)
        )


def _run_gendihedfit(
    citname: str,
    nlmaxiter: int,
    skip_existing: bool,
    logger: logging.Logger | None = None,
    workdir: Optional[Path] = None,
    fit_cli_args: Optional[list] = None,
) -> None:
    """ Subprocess into ``ffpopt-GenDihedFit.py`` to produce ``<citname>.py``.

    FUTURE: replace subprocess with API call once GenDihedFit is refactored.

    Parameters
    ----------
    citname : str
        Current-iteration name (e.g. ``"it01"``). The fit consumes
        ``<citname>.fit.json`` and emits ``<citname>.py``.
    nlmaxiter : int
        Forwarded as ``--nlmaxiter`` to ``ffpopt-GenDihedFit.py``.
    skip_existing : bool
        If True and ``<citname>.py`` plus ``<citname>.frcmod`` already exist
        (and the script contains ``p.save``), the call is skipped.
    logger
        Optional logger for progress messages.
    workdir : pathlib.Path, optional
        Directory containing the fit JSON; passed as ``subprocess`` ``cwd``.
    fit_cli_args : list, optional
        Extra CLI flags (e.g. ``--fit-full``, ``--fit-backend jax``).
    """
    log = _resolve_logger(logger)
    py_out = _in_workdir(workdir, f"{citname}.py").resolve()
    frcmod_out = _in_workdir(workdir, f"{citname}.frcmod").resolve()
    fit_json = str(_in_workdir(workdir, f"{citname}.fit.json").resolve())
    if skip_existing and gendihedfit_outputs_complete(py_out, frcmod_out):
        log.info("[twist] %s + %s exist - skipping GenDihedFit.", py_out, frcmod_out)
        return
    log.info("[twist] GenDihedFit -> %s (cwd=%s)", py_out, _subprocess_cwd(workdir))
    extra = list(fit_cli_args or [])
    if extra:
        from ffpopt.affdo.AffdoLog import log_affdo

        log_affdo(log, "GenDihedFit extra flags: %s", " ".join(str(x) for x in extra))
    _run_ffpopt_bin(
        "ffpopt-GenDihedFit.py",
        f"--nlmaxiter={nlmaxiter}",
        *extra,
        fit_json,
        cwd=_subprocess_cwd(workdir),
    )
    _require_files([py_out, frcmod_out], step="GenDihedFit")
    log.info("[twist] wrote %s and %s", py_out, frcmod_out)


def _compare_per_bond(
    scans,
    hl_prefix: str,
    ll_prefix: str,
    config,
    plot_dir=None,
    structure_images=None,
    workdir: Optional[Path] = None,
):
    """ Run :func:`ffpopt.ScanAnalysis.compare_scan_files` for each scan.

    Compares ``{hl_prefix}_{idxs}.dat`` against ``{ll_prefix}_{idxs}.dat``
    bond-by-bond. When ``plot_dir`` is set, also writes a
    ``compare_{hl_prefix}_vs_{ll_prefix}_{idxs}.png`` plot per bond into
    that directory (extrema, unmatched-extrema highlights, failed
    criteria). When ``structure_images`` is set, the matching 2D drawing
    is rendered as a top panel on each plot.

    Parameters
    ----------
    scans : list of _TwistParam
        Per-bond scan records from :func:`_resolve_scans_and_params`.
    hl_prefix : str
        High-level filename prefix (e.g. ``"qdpi2"``).
    ll_prefix : str
        Low-level filename prefix (e.g. ``"orig"`` or ``"it01"``).
    config : ffpopt.ScanAnalysis.ScanCompareConfig or None
        Tunable thresholds; forwarded to ``compare_scan_files``. ``None``
        uses the default thresholds.
    plot_dir : str or pathlib.Path, optional
        Directory to save comparison PNGs into. Default is None (no plots).
    structure_images : dict, optional
        Map from ``frozenset({a, b})`` (0-based central-bond atom indices)
        to a 2D structure image path. Default is None (no structure panel).
    workdir : pathlib.Path, optional
        When set, relative ``.dat`` paths are resolved under this directory.

    Returns
    -------
    dict
        Map of ``scan.GetIdxStr()`` to
        :class:`ffpopt.ScanAnalysis.ScanComparison`.
    """
    from ffpopt.scan.ScanAnalysis import compare_scan_files

    out = {}
    for scan in scans:
        idx = scan.GetIdxStr()
        def _profile(prefix: str) -> Path | None:
            dat = Path(_in_workdir(workdir, f"{prefix}_{idx}.dat"))
            if dat.is_file():
                return dat
            js = Path(_in_workdir(workdir, f"{prefix}_{idx}.json"))
            if js.is_file():
                return js
            return None

        hl_file = _profile(hl_prefix)
        ll_file = _profile(ll_prefix)
        if hl_file is None or ll_file is None:
            missing = []
            if hl_file is None:
                missing.append(str(_in_workdir(workdir, f"{hl_prefix}_{idx}.dat")))
            if ll_file is None:
                missing.append(str(_in_workdir(workdir, f"{ll_prefix}_{idx}.dat")))
            hints = []
            parent = Path(_in_workdir(workdir, "."))
            related = sorted(parent.glob(f"*{idx}.dat")) + sorted(
                parent.glob(f"*{idx}.json")
            )
            if related:
                hints.append(
                    "related files: " + ", ".join(r.name for r in related[:8])
                )
            extra = ("; " + "; ".join(hints)) if hints else ""
            raise FileNotFoundError(
                f"HL/LL compare for {idx}: missing {', '.join(missing)}.{extra} "
                "If multi-centroid HL never promoted a profile, the scan likely "
                "failed or skip_existing reused a JSON without a .dat."
            )
        hl_path = str(hl_file)
        ll_path = str(ll_file)
        plot_path = None
        if plot_dir is not None:
            plot_path = Path(plot_dir) / f"compare_{hl_prefix}_vs_{ll_prefix}_{idx}.png"
        structure_image_path = None
        if structure_images is not None:
            structure_image_path = structure_images.get(
                frozenset((scan.idxs[1], scan.idxs[2]))
            )
        out[idx] = compare_scan_files(
            hl_path, ll_path, config,
            plot_path=plot_path,
            structure_image_path=structure_image_path,
        )
    return out


def _apply_fit_and_prepare(
    *,
    citname: str,
    origparm: str,
    inp: str,
    skip_existing: bool,
    logger: logging.Logger | None = None,
    workdir: Optional[Path] = None,
) -> None:
    """ Apply the fit script to ``origparm`` and rebuild the JSON input.

    Runs ``python <citname>.py origparm <citname>.parm7`` (current
    interpreter) to bake the new torsion terms into a fresh parm7, then
    PrepareInput ``--update`` to produce ``<citname>.json``. FUTURE: replace
    subprocess calls with API calls once PrepareInput is refactored.

    Parameters
    ----------
    citname : str
        Current-iteration name (e.g. ``"it01"``).
    origparm : str
        Path to the original parm7 that the fit script transforms.
    inp : str
        Input JSON (passed as ``--crd`` to PrepareInput so coordinates and
        metadata survive the update).
    skip_existing : bool
        If True and both ``<citname>.parm7`` and ``<citname>.json`` already
        exist, both subprocess calls are skipped.
    workdir : pathlib.Path, optional
        Working directory for relative paths and ``subprocess`` ``cwd``.
    """
    log = _resolve_logger(logger)
    parm_out = _in_workdir(workdir, f"{citname}.parm7")
    json_out = _in_workdir(workdir, f"{citname}.json")
    py_script = _in_workdir(workdir, f"{citname}.py")
    origparm_path = _in_workdir(workdir, origparm).resolve()
    inp_path = _in_workdir(workdir, inp).resolve()
    if skip_existing and parm_out.exists() and json_out.exists():
        log.info(
            "[twist] %s & %s exist - skipping apply+prepare.",
            parm_out,
            json_out,
        )
        return
    log.info("[twist] applying fit -> %s (in-process: %s)", parm_out, py_script)
    sys.stdout.flush()
    _run_fit_script_inprocess(py_script, origparm_path, parm_out.resolve())
    _require_files([parm_out], step="apply fit script")
    log.info("[twist] fit script finished -> %s", parm_out.resolve())
    sys.stdout.flush()
    log.info("[twist] PrepareInput -> %s", json_out)
    _run_ffpopt_bin(
        "ffpopt-PrepareInput.py",
        "--update",
        f"--parm={parm_out.resolve()}",
        f"--crd={inp_path}",
        f"--out={json_out.resolve()}",
        cwd=_subprocess_cwd(workdir),
    )
    _require_files([parm_out, json_out], step="apply+prepare")
    log.info("[twist] wrote %s and %s", parm_out.resolve(), json_out.resolve())


def _list_iteration_frcmods(directory: Path) -> list[Path]:
    """Return ``itXX.frcmod`` paths in ascending iteration order."""
    import re

    rx = re.compile(r"^it(\d+)\.frcmod$", re.IGNORECASE)
    found: list[tuple[int, Path]] = []
    if not directory.is_dir():
        return []
    for path in directory.iterdir():
        m = rx.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]


def _promote_batch_iteration_outputs(
    batch_dirs: list[Path],
    dest_dir: Path,
    *,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Copy batch ``itXX.frcmod`` (+ fit.json) into ``dest_dir`` with global numbering."""
    import shutil

    log = _resolve_logger(logger)
    dest_dir.mkdir(parents=True, exist_ok=True)
    promoted: list[Path] = []
    global_it = 1
    for batch_dir in batch_dirs:
        for frcmod in _list_iteration_frcmods(batch_dir):
            dest_frcmod = dest_dir / f"it{global_it:02d}.frcmod"
            shutil.copy2(frcmod, dest_frcmod)
            fit_src = batch_dir / f"{frcmod.stem}.fit.json"
            if fit_src.is_file():
                shutil.copy2(fit_src, dest_dir / f"it{global_it:02d}.fit.json")
            # Also keep parm/json aliases when present (restart / inspect).
            for suffix in (".parm7", ".json", ".py"):
                src = batch_dir / f"{frcmod.stem}{suffix}"
                if src.is_file():
                    shutil.copy2(src, dest_dir / f"it{global_it:02d}{suffix}")
            promoted.append(dest_frcmod)
            log.info(
                "[twist] promoted %s -> %s",
                frcmod,
                dest_frcmod.name,
            )
            global_it += 1
    return promoted

