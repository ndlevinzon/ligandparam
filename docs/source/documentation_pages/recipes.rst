Parametrization Recipes
=======================

Pre-built pipelines for ligand parameterization. Each recipe populates
``self.stages`` in ``setup()`` and runs them via ``execute()``.

Multi-orientation recipes (:class:`~ligandparam.recipes.FreeLigand`,
:class:`~ligandparam.recipes.DPFreeLigand`) default to the ``so3_n28``
quaternion orientation pack. Pass ``orientation_protocol="legacy_euler"`` for
the historical Euler grid.

Optional torsion correction
---------------------------

:class:`~ligandparam.recipes.FreeLigand`,
:class:`~ligandparam.recipes.LazyLigand`, and
:class:`~ligandparam.recipes.DPFreeLigand` accept ``dihed_correct=True`` to
append :class:`~ligandparam.stages.ffpopt_dihed.StageDihedTwistCorrection`
after Leap. Recipe kwargs mirror the CLI where applicable, including
``dihed_delta`` (wavefront step) and ``dihed_fragment_config`` (a
:class:`~scission.models.FragmentConfig` or dict). For interactive sessions,
prefer the separate ``lig-dihed-correct`` CLI after ``lig-getparam``
(see :doc:`dihedrals` and :doc:`cli`).

Additional recipes such as :class:`~ligandparam.recipes.DPLigand` and
:class:`~ligandparam.recipes.SQMLigand` are exported from
:mod:`ligandparam.recipes` even when they are not listed below.

.. toctree::
   :maxdepth: 2
   :caption: Available Recipes:

   ./recipes/lazyligand.rst
   ./recipes/freeligand.rst
