"""Shared opt-in helpers for ffpopt dihedral corrections on recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping


def pop_dihed_options(kwargs: MutableMapping[str, Any]) -> dict[str, Any]:
    """Pull dihedral-correction options out of recipe kwargs.

    Returns a dict with keys used by recipes:
    ``dihed_correct``, ``dihed_model``, ``dihed_maxiter``, ``dihed_nprim``,
    ``dihed_geometric_opt``, ``dihed_skip_existing``, ``dihed_out_frcmod``,
    ``dihed_out_dir``, ``dihed_rotatable_bond_smarts``.
    """
    return {
        "dihed_correct": bool(kwargs.pop("dihed_correct", False)),
        "dihed_model": kwargs.pop("dihed_model", "qdpi2"),
        "dihed_maxiter": int(kwargs.pop("dihed_maxiter", 2)),
        "dihed_nprim": int(kwargs.pop("dihed_nprim", 3)),
        "dihed_geometric_opt": bool(kwargs.pop("dihed_geometric_opt", True)),
        "dihed_skip_existing": bool(kwargs.pop("dihed_skip_existing", True)),
        "dihed_out_frcmod": kwargs.pop("dihed_out_frcmod", None),
        "dihed_out_dir": kwargs.pop("dihed_out_dir", None),
        "dihed_rotatable_bond_smarts": kwargs.pop("dihed_rotatable_bond_smarts", None),
    }


def apply_dihed_options(obj: Any, kwargs: MutableMapping[str, Any]) -> None:
    """Set dihedral-correction attributes on a recipe from kwargs."""
    opts = pop_dihed_options(kwargs)
    for key, value in opts.items():
        setattr(obj, key, value)


def append_dihed_twist_stage(
    stages: list,
    *,
    recipe: Any,
    mol2: Path,
    lib: Path,
    frcmod: Path,
) -> None:
    """Append :class:`StageDihedTwistCorrection` when ``recipe.dihed_correct``."""
    if not getattr(recipe, "dihed_correct", False):
        return
    # Lazy import: stages package pulls optional heavy deps (rdkit, etc.).
    from ligandparam.stages.ffpopt_dihed import StageDihedTwistCorrection

    out_frcmod = (
        Path(recipe.dihed_out_frcmod)
        if recipe.dihed_out_frcmod is not None
        else recipe.cwd / f"{recipe.label}.dihed.frcmod"
    )
    out_dir = (
        Path(recipe.dihed_out_dir)
        if recipe.dihed_out_dir is not None
        else recipe.cwd / f"{recipe.label}.dihed_fragments"
    )
    stages.append(
        StageDihedTwistCorrection(
            "DihedTwist",
            main_input=mol2,
            cwd=recipe.cwd,
            in_lib=lib,
            in_frcmod=frcmod,
            out_frcmod=out_frcmod,
            out_dir=out_dir,
            model=recipe.dihed_model,
            maxiter=recipe.dihed_maxiter,
            nprim=recipe.dihed_nprim,
            nproc=recipe.nproc,
            geometric_opt=recipe.dihed_geometric_opt,
            skip_existing=recipe.dihed_skip_existing,
            rotatable_bond_smarts=recipe.dihed_rotatable_bond_smarts,
            logger=recipe.logger,
        )
    )
