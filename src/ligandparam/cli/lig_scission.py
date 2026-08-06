"""Standalone CLI for scission ligand fragmentation.

Supports the full upstream ``scission`` subcommands (``fragment``, ``merge``,
``pick-bond``), plus a ligandparam-friendly shortcut that resolves mol2/lib/frcmod
from a ``lig-getparam`` output directory::

    lig-getparam -i chaps.mol2 -r CHA -d CHA3 -rn freeligand ...
    lig-scission fragment -d CHA3 -r CHA --label chaps

``lig-scission`` and ``scission`` are both installed entry points for the same
integrated package under ``src/scission``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scission.cli import main as scission_main


def _resolve_getparam_bundle(
    *,
    cwd: Path,
    data_cwd: Path,
    resname: str,
    label: str | None,
) -> tuple[Path, Path, Path, Path]:
    """Return ``(work_dir, mol2, lib, frcmod)`` under a lig-getparam layout."""
    work_dir = (cwd / data_cwd / resname).resolve()
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Working directory does not exist: {work_dir}")

    stem = label or resname
    cand_mol2 = work_dir / f"{stem}.mol2"
    cand_lib = work_dir / f"{stem}.lib"
    cand_frcmod = work_dir / f"{stem}.frcmod"

    if not cand_mol2.is_file() and label is None:
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
            f"{work_dir}. Missing: {', '.join(p.name for p in missing)}. "
            "Pass --label or explicit --mol2/--lib/--frcmod."
        )
    return work_dir, cand_mol2, cand_lib, cand_frcmod


def _expand_fragment_shortcuts(argv: list[str]) -> list[str]:
    """Rewrite ``fragment -d/-r/--label`` into explicit ``--mol2/--lib/--frcmod``."""
    if not argv or argv[0] != "fragment":
        return argv

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_cwd", type=Path, default=None)
    parser.add_argument("-r", "--resname", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--mol2", type=Path, default=None)
    parser.add_argument("--lib", type=Path, default=None)
    parser.add_argument("--frcmod", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    known, remaining = parser.parse_known_args(argv[1:])

    if known.data_cwd is None or known.resname is None:
        return argv
    if known.mol2 is not None and known.lib is not None and known.frcmod is not None:
        return argv

    work_dir, mol2, lib, frcmod = _resolve_getparam_bundle(
        cwd=Path.cwd(),
        data_cwd=known.data_cwd,
        resname=known.resname,
        label=known.label,
    )
    outdir = known.outdir or (work_dir / f"{mol2.stem}.scission_fragments")

    expanded = [
        "fragment",
        "--mol2",
        str(mol2),
        "--lib",
        str(lib),
        "--frcmod",
        str(frcmod),
        "--outdir",
        str(outdir),
        *remaining,
    ]
    return expanded


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``lig-scission`` / convenience wrapper."""
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        expanded = _expand_fragment_shortcuts(raw)
    except FileNotFoundError as exc:
        print(f"lig-scission: {exc}", file=sys.stderr)
        return 2
    return scission_main(expanded)


if __name__ == "__main__":
    raise SystemExit(main())
