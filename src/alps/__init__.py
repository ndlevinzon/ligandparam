"""ALPS: installable orchestrator for ligandparam, ffpopt, and scission.

``pip install alps`` (this repository) installs all four import packages
and the shared Python dependencies. ligandparam owns parameterization,
scission owns fragmentation, ffpopt owns single-molecule torsion fitting.
This package binds those trees and drives ``lig-dihed-correct`` /
``lig-scission``.
"""

from __future__ import annotations

__version__ = "1.6.1"

from .companions import install_import_hook as _install_companion_hook

_install_companion_hook()
