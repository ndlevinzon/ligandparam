"""Shared recipe stage builders (write once)."""

from __future__ import annotations

from typing import Any, List

from ligandparam.stages import StageLeap, StageParmChk, StageUpdate


def charge_update_parmchk_leap_stages(
    *,
    recipe: Any,
    initial_mol2,
    final_mol2,
    nonminimized_mol2,
    frcmod,
    lib,
) -> List:
    """Tail used by most ligand recipes: copy charges onto initial coords, parmchk, leap."""
    return [
        StageUpdate(
            "UpdateCharges",
            main_input=initial_mol2,
            cwd=recipe.cwd,
            source_mol2=final_mol2,
            out_mol2=nonminimized_mol2,
            update_charges=True,
            net_charge=recipe.net_charge,
            logger=recipe.logger,
            **recipe.kwargs,
        ),
        StageParmChk(
            "ParmChk",
            main_input=nonminimized_mol2,
            cwd=recipe.cwd,
            out_frcmod=frcmod,
            logger=recipe.logger,
            **recipe.kwargs,
        ),
        StageLeap(
            "Leap",
            main_input=nonminimized_mol2,
            cwd=recipe.cwd,
            in_frcmod=frcmod,
            out_lib=lib,
            logger=recipe.logger,
            **recipe.kwargs,
        ),
    ]
