"""Shared opt-in helpers for ffpopt dihedral corrections on recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping


def coerce_fragment_config(value: Any):
    """Normalize a recipe ``dihed_fragment_config`` value.

    Parameters
    ----------
    value
        ``None``, a :class:`scission.models.FragmentConfig`, or a plain
        ``dict`` accepted by ``FragmentConfig.from_dict``.

    Returns
    -------
    scission.models.FragmentConfig or None
    """
    if value is None:
        return None
    try:
        from scission.models import FragmentConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dihed_fragment_config requires the integrated scission package"
        ) from exc
    if isinstance(value, FragmentConfig):
        return value
    if isinstance(value, dict):
        return FragmentConfig.from_dict(value)
    raise TypeError(
        "dihed_fragment_config must be None, a FragmentConfig, or a dict; "
        f"got {type(value).__name__}"
    )


def pop_dihed_options(kwargs: MutableMapping[str, Any]) -> dict[str, Any]:
    """Pull dihedral-correction options out of recipe kwargs.

    Returns a dict with keys used by recipes:

    * ``dihed_correct``, ``dihed_model``, ``dihed_maxiter``, ``dihed_nprim``
    * ``dihed_delta`` — wavefront angle step in degrees (CLI ``--delta``)
    * ``dihed_geometric_opt``, ``dihed_skip_existing``
    * ``dihed_out_frcmod``, ``dihed_out_dir``
    * ``dihed_rotatable_bond_smarts``
    * ``dihed_fragment_config`` — :class:`~scission.models.FragmentConfig`,
      dict for ``FragmentConfig.from_dict``, or ``None``
    """
    return {
        "dihed_correct": bool(kwargs.pop("dihed_correct", False)),
        "dihed_model": kwargs.pop("dihed_model", "qdpi2"),
        "dihed_maxiter": int(kwargs.pop("dihed_maxiter", 2)),
        "dihed_nprim": int(kwargs.pop("dihed_nprim", 3)),
        "dihed_delta": int(kwargs.pop("dihed_delta", 10)),
        "dihed_geometric_opt": bool(kwargs.pop("dihed_geometric_opt", True)),
        "dihed_skip_existing": bool(kwargs.pop("dihed_skip_existing", True)),
        "dihed_out_frcmod": kwargs.pop("dihed_out_frcmod", None),
        "dihed_out_dir": kwargs.pop("dihed_out_dir", None),
        "dihed_rotatable_bond_smarts": kwargs.pop("dihed_rotatable_bond_smarts", None),
        "dihed_fragment_config": kwargs.pop("dihed_fragment_config", None),
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
            delta=getattr(recipe, "dihed_delta", 10),
            nproc=recipe.nproc,
            geometric_opt=recipe.dihed_geometric_opt,
            skip_existing=recipe.dihed_skip_existing,
            rotatable_bond_smarts=recipe.dihed_rotatable_bond_smarts,
            fragment_config=coerce_fragment_config(
                getattr(recipe, "dihed_fragment_config", None)
            ),
            logger=recipe.logger,
        )
    )
