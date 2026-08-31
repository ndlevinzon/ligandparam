"""Amber ligand parameterization toolkit.

Public entry points typically come from :mod:`ligandparam.recipes`
(e.g. :class:`~ligandparam.recipes.FreeLigand`,
:class:`~ligandparam.recipes.LazyLigand`) and the stage pipeline under
:mod:`ligandparam.stages`.
"""

__version__ = "1.6.1"
__logging_name__ = "ligandparam"

from .companions import install_import_hook as _install_companion_hook

_install_companion_hook()
