"""Dihedral / reaction-coordinate scan engines (1-D and N-D wavefront)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "WaveFront",
    "WaveFrontND",
    "wavefront_mixins",
    "ScanAnalysis",
]

_LAZY = {
    "WaveFront": ".WaveFront",
    "WaveFrontND": ".WaveFrontND",
    "wavefront_mixins": ".wavefront_mixins",
    "ScanAnalysis": ".ScanAnalysis",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        module = import_module(_LAZY[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
