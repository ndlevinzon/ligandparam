"""High-level torsion-fitting workflows for Amber force fields.

This module is the recommended Python API for dihedral (twist) corrections.
Prefer it over shelling out to ``ffpopt-DihedTwistWorkflow.py``, which only
emits a bash script.

When to use which entry point
-----------------------------
``run_fragmented_dihed_twist_workflow``
    After ligand parameterization that produces a parent ``mol2`` / ``lib`` /
    ``frcmod`` triplet (e.g. ligandparam FreeLigand). Fragments the ligand
    with ``scission``, fits torsions per fragment against a high-level model,
    and merges DIHE terms into a new parent ``frcmod``. The ``lib`` is left
    unchanged - reuse the original library with the merged frcmod in LEaP.

``run_dihed_twist_workflow``
    Single-molecule path when you already have ``parm7`` / ``rst7`` (via
    ``ffpopt-PrepareInput.py`` -> ``start.json``) and known rotatable central
    bonds (``bond=[(i, j), ...]``, **0-based** atom indices). CLI-style
    ``"i,j"`` strings are still accepted for compatibility. Use
    ``bytype=True`` if you need a frcmod-writable global parameter set.

Requirements and caveats
------------------------
* Call either workflow from an ``if __name__ == "__main__":`` guard. The
  wavefront scan uses ``spawn``-mode multiprocessing. ``run_dihed_twist_workflow``
  may also pool over bonds (``n_bond_workers × wf_nproc``); the fragmented
  workflow pools over fragments the same way. ``nproc`` is always a total
  core budget so worker counts times nested wavefront size stay within it.
* Fragmented mode requires the integrated ``scission`` package
  (``src/scission``) and AmberTools (``tleap``) on ``PATH``.
* High-level ``model`` values (e.g. ``qdpi2``, ``mace``) need the matching
  ffpopt install group (tensorflow / pytorch).
"""

from __future__ import annotations

import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union


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
    """Return ``logger`` or the module logger for workflow progress messages."""
    return logger if logger is not None else _LOG


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

    Accepts :class:`~ligandparam.io.amber_bundle.AmberLigandBundle`
    (``mol2`` / ``lib`` / ``frcmod``) or :class:`scission.models.InputBundle`
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
    """ Run one wavefront scan via :func:`ffpopt.WaveFront.run_dihed_wavefront`.

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
        Forwarded unchanged to :func:`ffpopt.WaveFront.run_dihed_wavefront`.

    Returns
    -------
    dict or None
        The dict returned by ``run_dihed_wavefront``, or ``None`` if the
        scan was skipped because ``out`` already exists.
    """
    from .WaveFront import run_dihed_wavefront

    log = _resolve_logger(logger)
    inp_path = str(_in_workdir(workdir, inp))
    out_path = str(_in_workdir(workdir, out))
    if skip_existing and Path(out_path).exists():
        log.info("[twist] %s exists - skipping.", out_path)
        return None

    dihed_str = ",".join(str(i) for i in dihed_idxs)
    log.info(
        "[twist] scan: inp=%s model=%s dihed=%s out=%s",
        inp_path,
        model,
        dihed_str,
        out_path,
    )
    return run_dihed_wavefront(
        inp=inp_path,
        out=out_path,
        dihed=dihed_str,
        model=model,
        **wf_kwargs,
    )


def _slim_scan_result(scan_result: Optional[dict]) -> Optional[dict]:
    """Drop heavy ``wf_run`` objects so bond-pool IPC stays picklable."""
    if not isinstance(scan_result, dict):
        return scan_result
    return {k: v for k, v in scan_result.items() if k != "wf_run"}


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
) -> list[tuple[str, tuple, Optional[dict]]]:
    """Run one wavefront scan per bond, pooling when the core budget allows.

    Splits ``nproc`` as ``n_bond_workers × wf_nproc`` (same rule as fragment
    pooling) so concurrent bond scans do not oversubscribe cores.
    """
    log = _resolve_logger(logger)
    jobs = [
        {
            "prefix": prefix,
            "inp": inp,
            "model": model,
            "dihed_idxs": list(scan.idxs),
            "out": f"{prefix}_{scan.GetIdxStr()}.json",
            "skip_existing": skip_existing,
            "workdir": str(workdir) if workdir is not None else None,
            "wf_kwargs": dict(wf_kwargs),
        }
        for scan in scans
    ]
    if not jobs:
        return []

    n_bond_workers, n_wf = _split_fragment_nproc(nproc, len(jobs))
    for job in jobs:
        job["wf_kwargs"]["nproc"] = int(n_wf)

    log.info(
        "[twist] parallel bond scans: prefix=%s, %s bond(s), nproc=%s -> "
        "%s bond worker(s) x wf_nproc=%s",
        prefix,
        len(jobs),
        nproc,
        n_bond_workers,
        n_wf,
    )

    if n_bond_workers == 1:
        raw = [_run_bond_scan_job(job) for job in jobs]
    else:
        pool = _make_nondaemon_spawn_pool(n_bond_workers)
        try:
            raw = pool.map(_run_bond_scan_job, jobs)
        finally:
            pool.close()
            pool.join()

    return [
        (item["prefix"], tuple(item["dihed_idxs"]), item["result"])
        for item in raw
    ]


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
        If True and ``<citname>.py`` already exists, the call is skipped.
    logger
        Optional logger for progress messages.
    workdir : pathlib.Path, optional
        Directory containing the fit JSON; passed as ``subprocess`` ``cwd``.
    """
    log = _resolve_logger(logger)
    py_out = _in_workdir(workdir, f"{citname}.py").resolve()
    frcmod_out = _in_workdir(workdir, f"{citname}.frcmod").resolve()
    fit_json = str(_in_workdir(workdir, f"{citname}.fit.json").resolve())
    if skip_existing and py_out.exists():
        log.info("[twist] %s exists - skipping GenDihedFit.", py_out)
        return
    log.info("[twist] GenDihedFit -> %s (cwd=%s)", py_out, _subprocess_cwd(workdir))
    _run_ffpopt_bin(
        "ffpopt-GenDihedFit.py",
        f"--nlmaxiter={nlmaxiter}",
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
    from .ScanAnalysis import compare_scan_files

    out = {}
    for scan in scans:
        idx = scan.GetIdxStr()
        hl_path = str(_in_workdir(workdir, f"{hl_prefix}_{idx}.dat"))
        ll_path = str(_in_workdir(workdir, f"{ll_prefix}_{idx}.dat"))
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
    **standard_kwargs,
) -> dict:
    """ Wavefront-only twist workflow, run in-process.

    Mirrors the phase structure of ``bin/ffpopt-DihedTwistWorkflow.py`` but
    executes each scan via :func:`ffpopt.WaveFront.run_dihed_wavefront`
    instead of emitting a bash script; fit and prepare steps still shell
    out to the existing bin scripts. The phases are: a high-level scan per
    bond at ``model``; a reference sander scan per bond; an optional
    Phase 2b that drops bonds whose HL and reference already agree; and
    up to ``maxiter`` rounds of fit-then-rescan with an optional
    per-iteration convergence check. See the ``Workflows`` RST page for the
    full phase narrative.

    Parameters
    ----------
    inp : str
        Input JSON file (``ListOfStruct``). Only the first structure is used.
    bond
        Central bonds to scan. Preferred form: sequence of **0-based**
        ``(i, j)`` pairs. CLI-style ``"i,j"`` strings are also accepted
        (still 0-based). Each pair is the central bond of a proper dihedral.
    delta : int, optional
        Wavefront angle step (degrees). Default is 10.
    nprim : int, optional
        Number of primary cosine terms per parameter family in the fit.
        Default is 3.
    maxiter : int, optional
        Maximum number of fit-then-rescan iterations. Default is 2.
    bytype : bool, optional
        If True, fit-input masks are by atom *type* rather than by
        explicit atom-name instances. Default is False.
    nlmaxiter : int, optional
        Forwarded as ``--nlmaxiter`` to ``ffpopt-GenDihedFit.py``.
        Default is 300.
    nproc : int, optional
        Total core budget for bond scans and nested wavefront workers.
        Split as ``n_bond_workers × wf_nproc`` across concurrent bonds.
        Default is 1.
    wf_starting_nodes : int, optional
        Wavefront starting nodes. Default is 4.
    wf_num_conformers : int, optional
        Wavefront number of conformers. Default is 0 (auto).
    wf_max_levels : int, optional
        Wavefront max levels. Default is -1 (unlimited).
    wf_convergence_threshold : float, optional
        Wavefront convergence threshold (kcal/mol). Default is 0.01.
    skip_existing : bool, optional
        If True, skip any output (.json, .parm7, .py) that already exists.
        Mimics the ``if [ ! -e ... ]`` guards in the bash workflow, making
        the function re-runnable. Default is True.
    compare_config : ffpopt.ScanAnalysis.ScanCompareConfig, optional
        Thresholds for the HL-vs-LL comparison heuristic. Used by both the
        initial convergence check (Phase 2b) and the per-iteration check
        (Phase 3e). Default is None (uses the
        :class:`~ffpopt.ScanAnalysis.ScanCompareConfig` defaults).
    skip_converged_initial : bool, optional
        If True, Phase 2b drops bonds whose HL and reference sander scans
        already agree (no torsion correction needed). Default is True.
    convergence_mode : {"drop", "all_or_nothing", "off"}, optional
        How Phase 3e behaves. ``"drop"``: when some bonds converge but
        others don't, drop the converged ones from later iterations and
        break the loop when none are left to refit. ``"all_or_nothing"``:
        never drop mid-loop; break only when every surviving bond agrees
        with HL in the same iteration. ``"off"``: skip the per-iteration
        comparison entirely. Default is "drop".
    plot_comparisons : bool, optional
        If True, save a PNG plot per bond per comparison alongside the
        ``.dat`` files (filenames like
        ``compare_{hl}_vs_{ll}_{idxs}.png``). Useful for eyeballing why a
        dihedral was kept or dropped. Default is False.
    structure_images : dict, optional
        Map of ``frozenset({a, b})`` (0-based central-bond atom indices)
        to a 2D structure image path (PNG or SVG). When provided alongside
        ``plot_comparisons=True``, the matching image is rendered as a top
        panel on each comparison plot. Used by
        :func:`run_fragmented_dihed_twist_workflow` to surface scission's
        per-torsion drawings. Default is None.
    workdir : path-like, optional
        Directory for relative inputs/outputs and subprocess ``cwd``. When
        set, this workflow never calls ``os.chdir`` - paths are resolved
        under ``workdir`` and shell-outs use ``subprocess(..., cwd=workdir)``.
        Default is None (use the process cwd / relative paths as given).
    logger : logging.Logger, optional
        Logger for workflow progress messages. Default is the module logger.
    progress : callable, optional
        ``progress(stage, detail)`` hook used by the fragmented parent to
        update a live status board. Stages include ``hl_scan``, ``orig_scan``,
        ``compare``, ``fit/...``, ``apply/...``, ``rescan/...``, ``finished``.
        Default is None.
    **standard_kwargs
        Forwarded to the wavefront. Accepts anything declared by
        :func:`ffpopt.Options.AddStandardOptions` (``model``, ``mfile``,
        ``geometric_opt``, ``ase_opt_tol``, ``cpu``, ...). Unknown standard
        kwargs raise ``TypeError``.

    Returns
    -------
    dict
        A dictionary with keys ``scans`` (list of
        ``(prefix, bond_idxs, scan_result)``), ``fit_jsons`` (list of fit
        JSON paths), ``iterations`` (list of
        ``{'parm': ..., 'json': ...}``), ``initial_comparisons`` (map of
        bond-idx string to :class:`~ffpopt.ScanAnalysis.ScanComparison` from
        Phase 2b), ``iteration_comparisons`` (per-iteration map), and
        ``early_stopped_at`` (the ``citname`` at which the loop broke, or
        ``None`` if it ran to ``maxiter``).
    """
    # ---- 0. Resolve & validate kwargs ------------------------------------
    valid_modes = {"drop", "all_or_nothing", "off"}
    if convergence_mode not in valid_modes:
        raise ValueError(
            f"convergence_mode must be one of {sorted(valid_modes)}; "
            f"got {convergence_mode!r}"
        )
    log = _resolve_logger(logger)
    wd = _as_path(workdir).resolve() if workdir is not None else None

    def _prog(stage: str, detail: str = "") -> None:
        if progress is not None:
            try:
                progress(stage, detail)
            except Exception:
                pass

    import argparse
    from types import SimpleNamespace
    from .Options import AddStandardOptions
    from .Struct import ListOfStruct

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
    wf_kwargs = dict(
        delta=delta,
        nproc=nproc,
        wf_starting_nodes=wf_starting_nodes,
        wf_max_levels=wf_max_levels,
        wf_num_conformers=wf_num_conformers,
        wf_convergence_threshold=wf_convergence_threshold,
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

    # ---- 1. High-level scans (one per bond; pooled when nproc allows) ----
    _prog("hl_scan", f"model={model} · {len(scans)} bond(s)")
    results["scans"].extend(
        _run_scans_for_bonds(
            scans,
            prefix=hlname,
            model=model,
            inp=args.inp,
            nproc=nproc,
            skip_existing=skip_existing,
            workdir=wd,
            logger=log,
            wf_kwargs=wf_kwargs,
        )
    )

    # ---- 2. Reference sander scans (one per bond, "orig" prefix) ---------
    _prog("orig_scan", f"sander reference · {len(scans)} bond(s)")
    results["scans"].extend(
        _run_scans_for_bonds(
            scans,
            prefix="orig",
            model="sander",
            inp=args.inp,
            nproc=nproc,
            skip_existing=skip_existing,
            workdir=wd,
            logger=log,
            wf_kwargs=wf_kwargs,
        )
    )

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
        _prog(f"rescan/{citname}", f"sander · {len(scans)} bond(s)")
        results["scans"].extend(
            _run_scans_for_bonds(
                scans,
                prefix=citname,
                model="sander",
                inp=str(_in_workdir(wd, f"{citname}.json")),
                nproc=nproc,
                skip_existing=skip_existing,
                workdir=wd,
                logger=log,
                wf_kwargs=wf_kwargs,
            )
        )

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

    import re
    def _safe(label: str) -> str:
        # Mirror scission.writers._safe_name (re-emitted here to avoid a
        # scission dep just for one regex).
        return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")

    out: dict = {}
    for t in fit_torsions:
        label = t.get("label")
        if label is None:
            continue
        img_str = manifest_images.get(label)
        if img_str:
            img = Path(img_str)
        else:
            img = frag_dir / f"torsion_{_safe(label)}.svg"
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


def _split_fragment_nproc(nproc: int, n_fragments: int) -> tuple[int, int]:
    """Split ``nproc`` across outer workers and nested wavefront size.

    Used for both fragment-level pooling and per-bond scan pooling inside
    :func:`run_dihed_twist_workflow`.

    Returns
    -------
    tuple of int
        ``(n_workers, n_wavefront_per_worker)`` such that
        ``n_workers * n_wavefront_per_worker <= nproc`` (when
        ``n_items > 1``), preferring as many outer workers as possible up
        to ``min(nproc, n_items)``.
    """
    nproc = max(1, int(nproc))
    n_fragments = max(1, int(n_fragments))
    if n_fragments == 1:
        return 1, nproc
    n_frag_workers = min(nproc, n_fragments)
    n_wf = max(1, nproc // n_frag_workers)
    return n_frag_workers, n_wf


# Spawn Process subclass must live at module scope so pool workers pickle.
_SpawnProcessBase = __import__("multiprocessing").get_context("spawn").Process


class _NonDaemonSpawnProcess(_SpawnProcessBase):
    """Spawn process that ignores daemon=True (may create nested pools)."""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class _NonDaemonSpawnContext:
    """Context wrapper so ``Pool`` uses :class:`_NonDaemonSpawnProcess`."""

    def __init__(self):
        import multiprocessing as mp

        self._ctx = mp.get_context("spawn")
        self.Process = _NonDaemonSpawnProcess

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _make_nondaemon_spawn_pool(n_workers: int):
    """Spawn ``Pool`` whose workers are non-daemon (may nest wavefront pools).

    ``multiprocessing.get_context(...).Pool`` is a factory method, not a class,
    so it cannot be subclassed. Pass ``multiprocessing.pool.Pool`` a context
    whose ``Process`` is :class:`_NonDaemonSpawnProcess` instead.
    """
    from multiprocessing.pool import Pool

    return Pool(
        processes=max(1, int(n_workers)),
        context=_NonDaemonSpawnContext(),
    )


def _slim_twist_result(twist_result: Optional[dict]) -> Optional[dict]:
    """Drop heavy ``wf_run`` objects so fragment-pool IPC stays picklable."""
    if twist_result is None:
        return None
    slim = {k: v for k, v in twist_result.items() if k != "scans"}
    scans_out = []
    for item in twist_result.get("scans", []) or []:
        if isinstance(item, tuple) and len(item) == 3:
            prefix, idxs, payload = item
            if isinstance(payload, dict):
                payload = {k: v for k, v in payload.items() if k != "wf_run"}
            scans_out.append((prefix, idxs, payload))
        else:
            scans_out.append(item)
    slim["scans"] = scans_out
    return slim


def _run_fragment_twist_job(job: dict) -> dict:
    """Worker entry: prepare + twist one fragment (picklable job dict)."""
    from types import SimpleNamespace

    from .fragment_progress import (
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

    def _progress(stage: str, detail: str = "") -> None:
        _set(status="running", stage=stage, detail=detail)

    frag_log = make_fragment_file_logger(fragment_id, frag_log_path)
    structure_images = _build_structure_image_map(frag_dir, fragment.fit_torsions)

    _set(
        status="running",
        stage="prepare",
        detail=f"{len(bonds)} bond(s) · wf_nproc={job['wf_nproc']}",
        bonds=len(bonds),
        log_path=str(frag_log_path),
    )
    frag_log.info(
        "[frag-twist] %s: %s bond(s) %s -> running twist in %s (wf_nproc=%s)",
        fragment_id,
        len(bonds),
        bonds,
        frag_dir,
        job["wf_nproc"],
    )

    try:
        with fragment_stdio_to_file(frag_log_path):
            start_json = _prepare_fragment_input(
                fragment,
                skip_existing=job["skip_existing"],
                logger=frag_log,
                workdir=frag_dir,
            )
            twist_kwargs = dict(job["twist_kwargs"])
            twist_kwargs["nproc"] = int(job["wf_nproc"])
            twist_kwargs.pop("logger", None)
            twist_kwargs.pop("progress", None)
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
    **standard_kwargs,
) -> dict:
    """ Fragment a ligand with scission, run the twist workflow on each fragment, then recombine.

    Drives ``scission`` (from FragmentMol) to break the parent ligand into
    reduced fragments, runs :func:`run_dihed_twist_workflow` per fragment
    with ``workdir=frag_dir`` (absolute paths + subprocess ``cwd``; no
    process-wide ``os.chdir``), then merges the per-fragment fitted
    DIHE terms back into a unified parent ``frcmod`` via
    ``scission.merge.merge_fragment_frcmods``. Like
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
        Optional :class:`~ligandparam.io.amber_bundle.AmberLigandBundle` or
        :class:`scission.models.InputBundle`. When set, overrides ``mol2`` /
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
        Total CPU workers for the fragmented workflow. Split across
        fragment-level jobs and per-fragment wavefront pools:
        ``n_fragment_workers = min(nproc, n_runnable_fragments)`` and
        ``wf_nproc = max(1, nproc // n_fragment_workers)`` (all ``nproc``
        go to wavefront when only one fragment runs). Default is 1.
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
        ``scission.merge.merge_fragment_frcmods``), and ``merged_frcmod``
        (path to the final merged parent frcmod).
    """
    try:
        from dataclasses import replace as _dc_replace

        from scission import FragmentConfig, InputBundle, fragment_ligand
        from scission.merge import merge_fragment_frcmods
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

    # bytype is forced True here: the per-fragment fits are merged back into
    # the parent frcmod via scission.merge.merge_fragment_frcmods, which can
    # only map fragment-fit DIHE terms onto parent atoms by atom type - the
    # fragment's atom names don't exist in the parent topology.
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
        **standard_kwargs,
    )

    runnable = []
    for fragment in fragments_iter:
        if not fragment.fit_torsions:
            log.info("[frag-twist] %s: no fit_torsions - skipping", fragment.fragment_id)
            continue
        bonds = bonds0_from_scission_fit_torsions(fragment.fit_torsions)
        frag_dir = _as_path(fragment.manifest_path).parent.resolve()
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

    n_frag_workers, n_wf = _split_fragment_nproc(nproc, len(runnable))
    for job in runnable:
        job["wf_nproc"] = n_wf

    from .fragment_progress import FragmentBoardWatcher, FragmentProgressStore

    status_path = out_dir_path / ".frag_progress.json"
    board_path = out_dir_path / "FRAG_STATUS.txt"
    store = FragmentProgressStore(status_path)
    for job in runnable:
        job["status_path"] = str(status_path)
        store.register(
            job["fragment_id"],
            bonds=len(job["bonds"]),
            frag_dir=job["frag_dir"],
            log_path=str(Path(job["frag_dir"]) / "frag-twist.log"),
        )

    log.info(
        "[frag-twist] parallel plan: %s fragment(s), nproc=%s -> "
        "%s fragment worker(s) x wf_nproc=%s",
        len(runnable),
        nproc,
        n_frag_workers,
        n_wf,
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
            per_fragment_results = []
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
            pool = _make_nondaemon_spawn_pool(n_frag_workers)
            try:
                # Unordered so progress logs appear as each fragment finishes;
                # restore runnable order afterward for stable merge input.
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
                per_fragment_results = [
                    by_id[j["fragment_id"]] for j in runnable
                ]
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
