"""Dihedral fitting: types, solvers, extended AFFDO knobs, sugar pucker."""

from __future__ import annotations

from typing import Any

__all__ = [
    "Dihedrals",
    "DihedMath",
    "DihedFourier",
    "DihedParmEd",
    "DihedFitSolve",
    "ExtendedFit",
    "DeltaPuckerFit",
]

_LAZY = {name: f".{name}" for name in __all__}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        module = import_module(_LAZY[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
