Parametrization Recipes
=======================

Pre-built pipelines for ligand parameterization. Each recipe populates
``self.stages`` in ``setup()`` and runs them via ``execute()``.

Multi-orientation recipes (:class:`~ligandparam.recipes.FreeLigand`,
:class:`~ligandparam.recipes.DPFreeLigand`) default to the ``so3_n28``
quaternion orientation pack. Pass ``orientation_protocol="legacy_euler"`` for
the historical Euler grid.

Additional recipes such as :class:`~ligandparam.recipes.DPLigand`,
:class:`~ligandparam.recipes.DPFreeLigand`, and
:class:`~ligandparam.recipes.SQMLigand` are exported from
:mod:`ligandparam.recipes` even when they are not listed below.

.. toctree::
   :maxdepth: 2
   :caption: Available Recipes:

   ./recipes/lazyligand.rst
   ./recipes/freeligand.rst
   ./recipes/buildligand.rst
   ./recipes/rnaligand.rst
