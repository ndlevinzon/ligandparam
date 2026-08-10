"""Standalone CLI for ffpopt dihedral corrections after ligandparam.

Typical same-session workflow::

    lig-getparam -i chaps.mol2 -r CHA -d CHA3 -rn freeligand ...
    lig-dihed-correct -d CHA3 -r CHA --label chaps

Or pass explicit ``--mol2`` / ``--lib`` / ``--frcmod`` paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ligandparam.io.amber_bundle import AmberLigandBundle, resolve_getparam_bundle
from ligandparam.log import get_logger, set_file_logger, set_stream_logger
from ligandparam.stages.ffpopt_dihed import StageDihedTwistCorrection


def run_dihed_correct(
    *,
    bundle: AmberLigandBundle | None = None,
    mol2: Path | None = None,
    lib: Path | None = None,
    frcmod: Path | None = None,
    work_dir: Path | None = None,
    out_frcmod: Path | None = None,
    out_dir: Path | None = None,
    model: str = "qdpi2",
    maxiter: int = 2,
    nprim: int = 3,
    delta: int = 10,
    nproc: int = 1,
    geometric_opt: bool = True,
    skip_existing: bool = True,
    dry_run: bool = False,
    fast_wavefront: bool | None = None,
    logger=None,
):
    """Execute :class:`StageDihedTwistCorrection` on an Amber ligand bundle."""
    if logger is None:
        logger = get_logger()
    if bundle is None:
        if mol2 is None or lib is None or frcmod is None or work_dir is None:
            raise ValueError(
                "Provide bundle= or all of mol2, lib, frcmod, and work_dir"
            )
        bundle = AmberLigandBundle(
            mol2=Path(mol2),
            lib=Path(lib),
            frcmod=Path(frcmod),
            work_dir=Path(work_dir),
        )
    out_frcmod = out_frcmod or bundle.work_dir / f"{bundle.stem}.dihed.frcmod"
    out_dir = out_dir or bundle.work_dir / f"{bundle.stem}.dihed_fragments"

    stage = StageDihedTwistCorrection(
        "DihedTwist",
        main_input=bundle.mol2,
        cwd=bundle.work_dir,
        in_lib=bundle.lib,
        in_frcmod=bundle.frcmod,
        out_frcmod=out_frcmod,
        out_dir=out_dir,
        model=model,
        maxiter=maxiter,
        nprim=nprim,
        delta=delta,
        nproc=nproc,
        geometric_opt=geometric_opt,
        skip_existing=skip_existing,
        fast_wavefront=fast_wavefront,
        logger=logger,
    )
    return stage.execute(dry_run=dry_run, nproc=nproc)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``lig-dihed-correct``."""
    from ffpopt.runtime.console import print_startup_banner

    parser = argparse.ArgumentParser(
        description=(
            "Fit dihedral corrections with ffpopt after ligandparam "
            "(fragmented twist -> merged frcmod; lib unchanged)."
        )
    )
    parser.add_argument(
        "-d",
        "--data_cwd",
        type=Path,
        default=None,
        help="Same --data_cwd used with lig-getparam (directory under CWD)",
    )
    parser.add_argument(
        "-r",
        "--resname",
        type=str,
        default=None,
        help="Same --resname used with lig-getparam (subdir under data_cwd)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Recipe file stem (default: resname, or auto-detect unique *.mol2)",
    )
    parser.add_argument("--mol2", type=Path, default=None, help="Parent ligand mol2")
    parser.add_argument("--lib", type=Path, default=None, help="Parent Amber lib")
    parser.add_argument("--frcmod", type=Path, default=None, help="Parent frcmod")
    parser.add_argument(
        "-o",
        "--out-frcmod",
        type=Path,
        default=None,
        help="Merged output frcmod (default: {stem}.dihed.frcmod)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Fragment / scan working directory (default: {stem}.dihed_fragments)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qdpi2",
        help="High-level ffpopt model (default: qdpi2)",
    )
    parser.add_argument("--maxiter", type=int, default=2, help="Fit iterations (default: 2)")
    parser.add_argument("--nprim", type=int, default=3, help="Cosine primitives (default: 3)")
    parser.add_argument(
        "--delta",
        type=int,
        default=10,
        help="Wavefront dihedral step in degrees (default: 10; try 5 if geomeTRIC is unstable)",
    )
    parser.add_argument("-n", "--nproc", type=int, default=1, help="Wavefront parallelism")
    parser.add_argument(
        "--no-geometric-opt",
        action="store_true",
        help=(
            "Use ASE BFGS instead of geomeTRIC for constrained scans "
            "(only if you intentionally want to skip geomeTRIC)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Do not reuse existing fragment/scan artifacts",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Faster wavefront defaults: looser geomeTRIC converge, lower "
            "maxiter, delta=15, shorter recovery ladder, less checkpoint I/O; "
            "for xtb prefer wavefront depth over fragment breadth. "
            "Same as FFPOPT_FAST_WAVEFRONT=1. Explicit --delta / "
            "--geometric-* overrides still win when not at library defaults."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Log planned work only")
    parser.add_argument(
        "--logger",
        choices=("stream", "file"),
        default="stream",
        help="Logging destination (default: stream)",
    )

    args = parser.parse_args(argv)

    print_startup_banner()

    try:
        bundle = resolve_getparam_bundle(
            cwd=Path.cwd(),
            data_cwd=args.data_cwd,
            resname=args.resname,
            label=args.label,
            mol2=args.mol2,
            lib=args.lib,
            frcmod=args.frcmod,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if args.logger == "file":
        logger = set_file_logger(bundle.work_dir / f"{bundle.stem}.dihed.log")
    else:
        logger = set_stream_logger()

    logger.info(
        "lig-dihed-correct: mol2=%s lib=%s frcmod=%s",
        bundle.mol2,
        bundle.lib,
        bundle.frcmod,
    )
    result = run_dihed_correct(
        bundle=bundle,
        out_frcmod=args.out_frcmod,
        out_dir=args.out_dir,
        model=args.model,
        maxiter=args.maxiter,
        nprim=args.nprim,
        delta=args.delta,
        nproc=args.nproc,
        geometric_opt=not args.no_geometric_opt,
        skip_existing=not args.force,
        dry_run=args.dry_run,
        fast_wavefront=True if args.fast else None,
        logger=logger,
    )
    if result is not None:
        logger.info(
            "Done (all fragment scans + merge finished). merged_frcmod=%s",
            result.get("merged_frcmod"),
        )
    return 0


if __name__ == "__main__":
    # Required for ffpopt wavefront spawn-mode multiprocessing.
    raise SystemExit(main())
