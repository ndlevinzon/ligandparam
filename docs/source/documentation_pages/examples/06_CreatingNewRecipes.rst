Creating New Recipes
====================

Recipes subclass :class:`~ligandparam.parametrization.Recipe`, populate
``self.stages`` in ``setup()``, and run them with ``execute()``. Built-in
examples include :class:`~ligandparam.recipes.LazyLigand` and
:class:`~ligandparam.recipes.FreeLigand`.

One-off pipeline
----------------

For a single script, construct a ``Recipe`` and assign stages directly:

.. code-block:: python

    from pathlib import Path
    from ligandparam.parametrization import Recipe
    from ligandparam.stages import StageInitialize, StageNormalizeCharge

    cwd = Path("output")
    ligand = cwd / "my_ligand.pdb"
    initial_mol2 = cwd / "my_ligand.initial.mol2"

    recipe = Recipe(ligand, cwd, logger="stream")
    recipe.stages = [
        StageInitialize(
            "Initialize",
            main_input=ligand,
            cwd=cwd,
            out_mol2=initial_mol2,
            net_charge=0,
            logger=recipe.logger,
        ),
        StageNormalizeCharge(
            "Normalize",
            main_input=initial_mol2,
            cwd=cwd,
            net_charge=0,
            out_mol2=initial_mol2,
            logger=recipe.logger,
        ),
    ]
    recipe.execute(dry_run=False)

Reusable recipe
---------------

For a workflow you will reuse, subclass ``Recipe`` and implement ``setup``:

.. code-block:: python

    from pathlib import Path
    from typing import Union

    from ligandparam.parametrization import Recipe
    from ligandparam.stages import StageInitialize, StageNormalizeCharge


    class MyNewRecipe(Recipe):
        def __init__(self, in_filename, cwd, *, net_charge: int, **kwargs):
            super().__init__(in_filename, cwd, **kwargs)
            self.net_charge = net_charge
            self.kwargs = kwargs

        def setup(self):
            initial_mol2 = self.cwd / f"{self.label}.initial.mol2"
            self.stages = [
                StageInitialize(
                    "Initialize",
                    main_input=self.in_filename,
                    cwd=self.cwd,
                    out_mol2=initial_mol2,
                    net_charge=self.net_charge,
                    logger=self.logger,
                    **self.kwargs,
                ),
                StageNormalizeCharge(
                    "Normalize1",
                    main_input=initial_mol2,
                    cwd=self.cwd,
                    net_charge=self.net_charge,
                    out_mol2=initial_mol2,
                    logger=self.logger,
                    **self.kwargs,
                ),
            ]


    recipe = MyNewRecipe("my_ligand.pdb", "output", net_charge=0, logger="stream")
    recipe.setup()
    recipe.execute(dry_run=False, nproc=12, mem=8)

See ``src/ligandparam/recipes/lazyligand.py`` and ``freeligand.py`` for fuller
pipelines to copy from.
