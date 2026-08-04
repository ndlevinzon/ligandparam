"""Recipes for building solvated or target-bound ligand systems.

This module is a placeholder; :class:`BuildLigand` is not implemented yet.
"""

from ligandparam.parametrization import Recipe
from ligandparam.stages import *


class BuildLigand(Recipe):
    """Build gas/aqueous/target systems around a parameterized ligand.

    Not implemented. Use :class:`~ligandparam.recipes.LazyLigand` or
    :class:`~ligandparam.recipes.FreeLigand` for ligand parameterization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nproc = kwargs.get("nproc", 12)
        self.mem = kwargs.get("mem", "60GB")
        self.net_charge = kwargs.get("net_charge", 0)
        self.atom_type = kwargs.get("atom_type", "gaff2")
        self.leaprc = kwargs.get("leaprc", None)
        self.target_pdb = kwargs.get("target_pdb")
        self.force_gaussian_rerun = kwargs.get("force_gaussian_rerun", False)
        raise NotImplementedError(
            "The BuildLigand recipe is not yet implemented. "
            "Please use LazyLigand or another recipe for now."
        )
