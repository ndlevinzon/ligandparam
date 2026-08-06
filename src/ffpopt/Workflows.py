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
    unchanged — reuse the original library with the merged frcmod in LEaP.

``run_dihed_twist_workflow``
    Single-molecule path when you already have ``parm7`` / ``rst7`` (via
    ``ffpopt-PrepareInput.py`` → ``start.json``) and known rotatable central
    bonds (``bond=["i,j", ...]``, 0-based). Use ``bytype=True`` if you need a
    frcmod-writable global parameter set.

Requirements and caveats
------------------------
* Call either workflow from an ``if __name__ == "__main__":`` guard. The
  wavefront scan uses ``spawn``-mode multiprocessing.
* Fragmented mode requires the integrated ``scission`` package
  (``src/scission``) and AmberTools (``tleap``) on ``PATH``.
* High-level ``model`` values (e.g. ``qdpi2``, ``mace``) need the matching
  ffpopt install group (tensorflow / pytorch).
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path


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
    bonds : list of list of int
        Central-bond atom pairs as ``[[a, b], ...]`` (0-based indices).
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
        Maps parameter name → ``{'nprim': nprim, 'masks': ...}`` for the
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
    **wf_kwargs
        Forwarded unchanged to :func:`ffpopt.WaveFront.run_dihed_wavefront`.

    Returns
    -------
    dict or None
        The dict returned by ``run_dihed_wavefront``, or ``None`` if the
        scan was skipped because ``out`` already exists.
    """
    from .WaveFront import run_dihed_wavefront

    if skip_existing and Path(out).exists():
        print(f"[twist] {out} exists — skipping.")
        return None

    dihed_str = ",".join(str(i) for i in dihed_idxs)
    print(f"[twist] scan: inp={inp} model={model} dihed={dihed_str} out={out}")
    return run_dihed_wavefront(
        inp=inp,
        out=out,
        dihed=dihed_str,
        model=model,
        **wf_kwargs,
    )


def _write_fit_json(
    *,
    citname: str,
    scans,
    params: dict,
    s_template: dict,
    hl_prefix: str,
    ll_prefix: str,
    parm: str,
) -> str:
    """ Write ``<citname>.fit.json`` for ``ffpopt-GenDihedFit.py``.

    Parameters
    ----------
    citname : str
        Current-iteration name (e.g. ``"it01"``).
    scans : list of _TwistParam
        Per-bond scan records from :func:`_resolve_scans_and_params`.
    params : dict
        Parameter family → fit-input metadata.
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

    Returns
    -------
    str
        Path to the written ``<citname>.fit.json`` file.
    """
    ss = copy.deepcopy(s_template)
    ss["output"] = citname + ".py"
    ss["parm"] = parm
    ss["profiles"] = [
        {
            "hl": f"{hl_prefix}_{scan.GetIdxStr()}.json",
            "ll": f"{ll_prefix}_{scan.GetIdxStr()}.json",
            "name": citname,
            "plots": [scan.GetParamByType()],
        }
        for scan in scans
    ]
    datadict = {
        "params": params,
        "output": f"{citname}.frcmod",
        "systems": [ss],
    }
    out_path = f"{citname}.fit.json"
    with open(out_path, "w") as fh:
        json.dump(datadict, fh, indent=4)
    return out_path


def _run_gendihedfit(citname: str, nlmaxiter: int, skip_existing: bool) -> None:
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
    """
    if skip_existing and Path(f"{citname}.py").exists():
        print(f"[twist] {citname}.py exists — skipping GenDihedFit.")
        return
    print(f"[twist] GenDihedFit → {citname}.py")
    subprocess.run(
        ["ffpopt-GenDihedFit.py", f"--nlmaxiter={nlmaxiter}", f"{citname}.fit.json"],
        check=True,
    )


def _compare_per_bond(
    scans,
    hl_prefix: str,
    ll_prefix: str,
    config,
    plot_dir=None,
    structure_images=None,
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
        hl_path = f"{hl_prefix}_{idx}.dat"
        ll_path = f"{ll_prefix}_{idx}.dat"
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
    *, citname: str, origparm: str, inp: str, skip_existing: bool
) -> None:
    """ Apply the fit script to ``origparm`` and rebuild the JSON input.

    Runs ``python3 <citname>.py origparm <citname>.parm7`` to bake the new
    torsion terms into a fresh parm7, then ``ffpopt-PrepareInput.py
    --update`` to produce ``<citname>.json``. FUTURE: replace subprocess
    calls with API calls once PrepareInput is refactored.

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
    """
    parm_out = f"{citname}.parm7"
    json_out = f"{citname}.json"
    if skip_existing and Path(parm_out).exists() and Path(json_out).exists():
        print(f"[twist] {parm_out} & {json_out} exist — skipping apply+prepare.")
        return
    print(f"[twist] applying fit → {parm_out}")
    subprocess.run(["python3", f"{citname}.py", origparm, parm_out], check=True)
    print(f"[twist] PrepareInput → {json_out}")
    subprocess.run(
        [
            "ffpopt-PrepareInput.py",
            "--update",
            f"--parm={parm_out}",
            f"--crd={inp}",
            f"--out={json_out}",
        ],
        check=True,
    )


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
    bond : list of str
        Bonds to scan, each as ``"a,b"`` (0-based atom indices, central pair
        of a proper dihedral). Same shape as repeated ``--bond`` on the CLI.
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
        Wavefront parallelism. Default is 1.
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

    # SimpleNamespace mirroring the CLI's args — needed by los.SetArgs.
    args = SimpleNamespace(
        inp=inp,
        bond=list(bond),
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
    origparm = los.structs[0].data["parm"]

    bonds_parsed = [[int(x) for x in b.split(",")] for b in args.bond]
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

    # ---- 1. High-level scans (one per bond) ------------------------------
    for scan in scans:
        out = f"{hlname}_{scan.GetIdxStr()}.json"
        r = _run_one_scan(
            inp=args.inp,
            model=model,
            dihed_idxs=scan.idxs,
            out=out,
            skip_existing=skip_existing,
            **wf_kwargs,
        )
        results["scans"].append((hlname, scan.idxs, r))

    # ---- 2. Reference sander scans (one per bond, "orig" prefix) ---------
    for scan in scans:
        out = f"orig_{scan.GetIdxStr()}.json"
        r = _run_one_scan(
            inp=args.inp,
            model="sander",
            dihed_idxs=scan.idxs,
            out=out,
            skip_existing=skip_existing,
            **wf_kwargs,
        )
        results["scans"].append(("orig", scan.idxs, r))

    plot_dir = Path(".") if plot_comparisons else None

    # ---- 2b. Drop dihedrals that already agree (initial convergence) -----
    if skip_converged_initial:
        initial = _compare_per_bond(
            scans, hlname, "orig", compare_config,
            plot_dir=plot_dir, structure_images=structure_images,
        )
        results["initial_comparisons"] = initial
        kept_bonds = []
        for bond, scan in zip(bonds_parsed, scans):
            idx = scan.GetIdxStr()
            r = initial[idx]
            if r.is_close:
                reason = "flat (HL barrier below threshold)" if r.is_flat \
                    else "agrees with HL within thresholds"
                print(f"[twist] {idx}: {reason} — dropping from iterative fit")
            else:
                kept_bonds.append(bond)
                reasons = "; ".join(r.reasons) if r.reasons else "extrema disagree"
                print(f"[twist] {idx}: refit needed ({reasons})")
        if not kept_bonds:
            print("[twist] all dihedrals already agree — skipping Phase 3")
            return results
        if len(kept_bonds) < len(bonds_parsed):
            scans, params, s_template = _resolve_scans_and_params(
                mol, kept_bonds, nprim=nprim, bytype=bytype
            )
            print(f"[twist] fitting {len(scans)} of {len(bonds_parsed)} dihedrals")

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
        )
        results["fit_jsons"].append(fit_json)

        # 3b. GenDihedFit → itNN.py. FUTURE: replace subprocess with API call.
        _run_gendihedfit(citname, nlmaxiter=args.nlmaxiter, skip_existing=skip_existing)

        # 3c. Apply fit + PrepareInput. FUTURE: replace subprocess with API calls.
        _apply_fit_and_prepare(
            citname=citname,
            origparm=origparm,
            inp=args.inp,
            skip_existing=skip_existing,
        )
        results["iterations"].append(
            {"parm": f"{citname}.parm7", "json": f"{citname}.json"}
        )

        # 3d. Sander scans on the updated parm (one per bond, "itNN" prefix).
        for scan in scans:
            out = f"{citname}_{scan.GetIdxStr()}.json"
            r = _run_one_scan(
                inp=f"{citname}.json",
                model="sander",
                dihed_idxs=scan.idxs,
                out=out,
                skip_existing=skip_existing,
                **wf_kwargs,
            )
            results["scans"].append((citname, scan.idxs, r))

        # 3e. Per-iteration convergence: compare HL vs itNN per bond.
        if convergence_mode != "off":
            iter_cmp = _compare_per_bond(
                scans, hlname, citname, compare_config,
                plot_dir=plot_dir, structure_images=structure_images,
            )
            results["iteration_comparisons"].append(
                {"citname": citname, "comparisons": iter_cmp}
            )
            converged_idxs = [idx for idx, r in iter_cmp.items() if r.is_close]
            still_off_idxs = [idx for idx, r in iter_cmp.items() if not r.is_close]

            if not still_off_idxs:
                # All bonds converged this iteration — break in both modes.
                print(f"[twist] all dihedrals converged at {citname} — stopping early")
                results["early_stopped_at"] = citname
                break

            if convergence_mode == "drop" and converged_idxs:
                # Rebuild scans/params/s_template from just the survivors so
                # the next iteration's fit no longer sees the converged ones.
                kept_bonds = [
                    [s.idxs[1], s.idxs[2]]
                    for s in scans
                    if s.GetIdxStr() in still_off_idxs
                ]
                scans, params, s_template = _resolve_scans_and_params(
                    mol, kept_bonds, nprim=nprim, bytype=bytype
                )
                print(
                    f"[twist] {citname}: dropping converged "
                    f"({', '.join(converged_idxs)}); continuing with "
                    f"{len(scans)} dihedral(s): {', '.join(still_off_idxs)}"
                )
            else:
                # all_or_nothing mode (or drop mode with nothing converged) —
                # just log and continue with the same set.
                print(
                    f"[twist] {citname}: {len(still_off_idxs)} dihedral(s) "
                    f"still need refit: {', '.join(still_off_idxs)}"
                )

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

    Reads ``manifest.json`` for the authoritative ``label → image path``
    mapping; for any label missing from the manifest (older runs, or rdkit
    unavailable when the fragment was written), falls back to the standard
    ``fragment_dir/torsion_{safe_name(label)}.svg`` filename if present.

    Parameters
    ----------
    frag_dir : pathlib.Path
        One fragment's output directory.
    fit_torsions : list of dict
        The fragment's ``fit_torsions.json`` payload (1-indexed atom
        indices, per scission's convention).

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


def _prepare_fragment_input(fragment, skip_existing: bool) -> str:
    """ Run ``ffpopt-PrepareInput.py`` on a scission fragment.

    Produces ``start.json`` next to the fragment's parm7/rst7. FUTURE:
    replace subprocess with an API call once PrepareInput is refactored.

    Parameters
    ----------
    fragment : scission.SelectedFragment or types.SimpleNamespace
        Fragment record exposing ``parm7_path``, ``rst7_path``, and
        ``fragment_id``.
    skip_existing : bool
        If True and ``start.json`` already exists in the cwd, PrepareInput
        is skipped.

    Returns
    -------
    str
        Always ``"start.json"`` (relative to the fragment directory).
    """
    if fragment.parm7_path is None or fragment.rst7_path is None:
        raise RuntimeError(
            f"fragment {fragment.fragment_id} has no parm7/rst7 — scission "
            f"likely failed to run tleap. Check that AmberTools is on PATH."
        )
    if skip_existing and Path("start.json").exists():
        print("[frag-twist] start.json exists — skipping PrepareInput")
        return "start.json"
    print(f"[frag-twist] PrepareInput → start.json (in {Path.cwd()})")
    subprocess.run(
        [
            "ffpopt-PrepareInput.py",
            f"--parm={fragment.parm7_path.name}",
            f"--crd={fragment.rst7_path.name}",
            "--out=start.json",
        ],
        check=True,
    )
    return "start.json"


def run_fragmented_dihed_twist_workflow(
    *,
    mol2: str,
    lib: str,
    frcmod: str,
    out_dir: str = "fragments",
    merged_frcmod: str = "merged.frcmod",
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
    **standard_kwargs,
) -> dict:
    """ Fragment a ligand with scission, run the twist workflow on each fragment, then recombine.

    Drives ``scission`` (from FragmentMol) to break the parent ligand into
    reduced fragments, runs :func:`run_dihed_twist_workflow` independently
    inside each fragment directory, then merges the per-fragment fitted
    DIHE terms back into a unified parent ``frcmod`` via
    ``scission.merge.merge_fragment_frcmods``. Like
    :func:`run_dihed_twist_workflow`, this must be called from inside an
    ``if __name__ == "__main__":`` guard — the wavefront uses ``spawn``-mode
    multiprocessing. See the ``Workflows`` RST page for the full on-disk
    layout and the re-running semantics.

    Parameters
    ----------
    mol2 : str
        Path to the parent ligand MOL2.
    lib : str
        Path to the parent Amber LIB.
    frcmod : str
        Path to the parent FRCMOD that the merge step splices fitted DIHE
        terms into.
    out_dir : str, optional
        Directory where per-fragment subdirs are written, resolved to an
        absolute path before scission runs. Default is "fragments".
    merged_frcmod : str, optional
        Path for the final merged parent frcmod. Relative to the call cwd,
        not ``out_dir``. A ``.merge_report.json`` is written alongside.
        Default is "merged.frcmod".
    fragment_config : scission.FragmentConfig, optional
        Override scission's fragmentation config. Default is None (uses
        ``FragmentConfig()`` — acyclic rotatable torsions, 30° step, etc.).
    rotatable_bond_smarts : str or iterable of str, optional
        Extra SMARTS patterns nominating additional central bonds as
        rotatable, forwarded to ``scission.FragmentConfig.rotatable_bond_smarts``.
        Each pattern must map the central bond atoms with atom-map numbers
        ``:1`` and ``:2`` (e.g. ``"[C:1](=[O])[N:2]"`` to make amide-like
        single bonds rotatable, or ``"[C:1]=[N:2]"`` for exocyclic imines).
        Patterns add to — they do not replace — scission's default rotatable
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
        Wavefront parallelism per fragment. Default is 1.
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
    parent_frcmod = Path(frcmod).resolve()
    bundle = InputBundle(
        mol2_path=Path(mol2).resolve(),
        lib_path=Path(lib).resolve(),
        frcmod_path=parent_frcmod,
    )
    out_dir_path = Path(out_dir).resolve()
    merged_frcmod_path = Path(merged_frcmod).resolve()

    existing_fragments = (
        _load_existing_fragments(out_dir_path) if skip_existing else None
    )
    if existing_fragments is not None:
        print(
            f"[frag-twist] {out_dir_path}/fragment_index.json exists — "
            f"skipping scission, reusing {len(existing_fragments)} fragment(s)"
        )
        fragmentation_dump = None
        fragments_iter = existing_fragments
    else:
        print(f"[frag-twist] fragmenting parent → {out_dir_path}")
        frag_result = fragment_ligand(bundle, out_dir_path, config)
        print(f"[frag-twist] selected {len(frag_result.selected_fragments)} fragment(s)")
        fragmentation_dump = frag_result.to_dict()
        fragments_iter = frag_result.selected_fragments

    # bytype is forced True here: the per-fragment fits are merged back into
    # the parent frcmod via scission.merge.merge_fragment_frcmods, which can
    # only map fragment-fit DIHE terms onto parent atoms by atom type — the
    # fragment's atom names don't exist in the parent topology.
    twist_kwargs = dict(
        delta=delta,
        nprim=nprim,
        maxiter=maxiter,
        bytype=True,
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
        **standard_kwargs,
    )

    per_fragment_results = []
    fragment_dirs_for_merge = []
    cwd = Path.cwd()

    for fragment in fragments_iter:
        if not fragment.fit_torsions:
            print(f"[frag-twist] {fragment.fragment_id}: no fit_torsions — skipping")
            continue
        # fit_torsions are 1-indexed; ffpopt's bond= takes 0-indexed pairs.
        bonds = [
            "{0},{1}".format(*(i - 1 for i in t["fragment_rotatable_bond"]))
            for t in fragment.fit_torsions
        ]
        frag_dir = fragment.manifest_path.parent
        structure_images = _build_structure_image_map(frag_dir, fragment.fit_torsions)
        print(
            f"[frag-twist] {fragment.fragment_id}: "
            f"{len(bonds)} bond(s) {bonds} → running twist in {frag_dir}"
        )

        os.chdir(frag_dir)
        try:
            _prepare_fragment_input(fragment, skip_existing=skip_existing)
            twist_result = run_dihed_twist_workflow(
                inp="start.json",
                bond=bonds,
                structure_images=structure_images or None,
                **twist_kwargs,
            )
        finally:
            os.chdir(cwd)

        per_fragment_results.append(
            {
                "fragment_id": fragment.fragment_id,
                "dir": str(frag_dir),
                "bonds": bonds,
                "twist_result": twist_result,
            }
        )
        fragment_dirs_for_merge.append(frag_dir)

    if not fragment_dirs_for_merge:
        raise RuntimeError(
            "no fragments had fittable torsions — nothing to merge"
        )

    print(
        f"[frag-twist] merging {len(fragment_dirs_for_merge)} fragment "
        f"frcmod(s) → {merged_frcmod_path}"
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
