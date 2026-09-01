"""ALPS orchestrator: run ligandparam, scission, and ffpopt as independent tools.

ligandparam owns parameterization only. This package binds companion trees
and will drive fragmentation / torsion correction. ffpopt and scission stay
importable on their own.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .companions import install_import_hook as _install_companion_hook

_install_companion_hook()
