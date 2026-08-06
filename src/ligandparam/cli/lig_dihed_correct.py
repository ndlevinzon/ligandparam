"""Standalone CLI for ffpopt dihedral corrections after ligandparam.

Typical same-session workflow::

    lig-getparam -i chaps.mol2 -r CHA -d CHA3 -rn freeligand ...
    lig-dihed-correct -d CHA3 -r CHA --label chaps

Or pass explicit ``--mol2`` / ``--lib`` / ``--frcmod`` paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ligandparam.log import get_logger, set_file_logger, set_stream_logger
from ligandparam.stages.ffpopt_dihed import StageDihedTwistCorrection


def _resolve_bundle(
    *,
    cwd: Path,
    data_cwd: Path | None,
    resname: str | None,
    label: str | None,
    mol2: Path | None,
    lib: Path | None,
    frcmod: Path | None,
) -> tuple[Path, Path, Path, Path]:
    """Return ``(work_dir, mol2, lib, frcmod)`` from CLI arguments."""
    if mol2 is not None and lib is not None and frcmod is not None:
        work_dir = mol2.resolve().parent
        return work_dir, mol2.resolve(), lib.resolve(), frcmod.resolve()

    if data_cwd is None or resname is None:
        raise ValueError(
            "Provide either (--mol2, --lib, --frcmod) or (--data_cwd and --resname)."
        )

    work_dir = (cwd / data_cwd / resname).resolve()
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Working directory does not exist: {work_dir}")

    stem = label or resname
    cand_mol2 = work_dir / f"{stem}.mol2"
    cand_lib = work_dir / f"{stem}.lib"
    cand_frcmod = work_dir / f"{stem}.frcmod"

    if not cand_mol2.is_file() and label is None:
        # Fall back to a unique non-ancillary mol2 in the directory.
        mol2s = sorted(
            p
            for p in work_dir.glob("*.mol2")
            if ".initial." not in p.name
            and ".centered." not in p.name
            and ".resp." not in p.name
            and not p.name.startswith("final_")
        )
        if len(mol2s) == 1:
            cand_mol2 = mol2s[0]
            stem = cand_mol2.stem
            cand_lib = work_dir / f"{stem}.lib"
            cand_frcmod = work_dir / f"{stem}.frcmod"

    missing = [p for p in (cand_mol2, cand_lib, cand_frcmod) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Could not find ligandparam outputs in "
            f"{work_dir}. Missing: {', '.join(str(p.name) for p in missing)}. "
            "Pass --label (input stem used by the recipe) or explicit "
            "--mol2/--lib/--frcmod."
        )
    return work_dir, cand_mol2, cand_lib, cand_frcmod


def run_dihed_correct(
    *,
    mol2: Path,
    lib: Path,
    frcmod: Path,
    work_dir: Path,
    out_frcmod: Path | None = None,
    out_dir: Path | None = None,
    model: str = "qdpi2",
    maxiter: int = 2,
    nprim: int = 3,
    nproc: int = 1,
    geometric_opt: bool = True,
    skip_existing: bool = True,
    dry_run: bool = False,
    logger=None,
):
    """Execute :class:`StageDihedTwistCorrection` on an Amber ligand bundle."""
    if logger is None:
        logger = get_logger()
    out_frcmod = out_frcmod or work_dir / f"{mol2.stem}.dihed.frcmod"
    out_dir = out_dir or work_dir / f"{mol2.stem}.dihed_fragments"

    stage = StageDihedTwistCorrection(
        "DihedTwist",
        main_input=mol2,
        cwd=work_dir,
        in_lib=lib,
        in_frcmod=frcmod,
        out_frcmod=out_frcmod,
        out_dir=out_dir,
        model=model,
        maxiter=maxiter,
        nprim=nprim,
        nproc=nproc,
        geometric_opt=geometric_opt,
        skip_existing=skip_existing,
        logger=logger,
    )
    return stage.execute(dry_run=dry_run, nproc=nproc)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``lig-dihed-correct``."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit dihedral corrections with ffpopt after ligandparam "
            "(fragmented twist → merged frcmod; lib unchanged)."
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
    parser.add_argument("-n", "--nproc", type=int, default=1, help="Wavefront parallelism")
    parser.add_argument(
        "--no-geometric-opt",
        action="store_true",
        help="Disable geomeTRIC constrained optimization",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Do not reuse existing fragment/scan artifacts",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log planned work only")
    parser.add_argument(
        "--logger",
        choices=("stream", "file"),
        default="stream",
        help="Logging destination (default: stream)",
    )

    args = parser.parse_args(argv)
    cwd = Path.cwd()

    try:
        work_dir, mol2, lib, frcmod = _resolve_bundle(
            cwd=cwd,
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
        logger = set_file_logger(work_dir / f"{mol2.stem}.dihed.log")
    else:
        logger = set_stream_logger()

    logger.info("lig-dihed-correct: mol2=%s lib=%s frcmod=%s", mol2, lib, frcmod)
    result = run_dihed_correct(
        mol2=mol2,
        lib=lib,
        frcmod=frcmod,
        work_dir=work_dir,
        out_frcmod=args.out_frcmod,
        out_dir=args.out_dir,
        model=args.model,
        maxiter=args.maxiter,
        nprim=args.nprim,
        nproc=args.nproc,
        geometric_opt=not args.no_geometric_opt,
        skip_existing=not args.force,
        dry_run=args.dry_run,
        logger=logger,
    )
    if result is not None:
        logger.info("Done. merged_frcmod=%s", result.get("merged_frcmod"))
    return 0


if __name__ == "__main__":
    # Required for ffpopt wavefront spawn-mode multiprocessing.
    raise SystemExit(main())
