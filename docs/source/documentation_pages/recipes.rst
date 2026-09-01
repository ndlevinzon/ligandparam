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
**record** twist options for ALPS. ligandparam does not import ffpopt or
scission and does not append a twist stage. After ``lig-getparam``, run
``lig-dihed-correct`` (see :doc:`dihedrals`, :doc:`cli`, and
:doc:`companions`).

Additional recipes such as :class:`~ligandparam.recipes.DPLigand` and
:class:`~ligandparam.recipes.SQMLigand` are exported from
:mod:`ligandparam.recipes` even when they are not listed below.

**Support tiers:** ``freeligand`` / ``lazyligand`` / ``lazierligand`` (and their
DeepMD variants) are the primary product path. ``sqmligand`` is
**secondary-supported** (registry + tests, not the default CLI story).
Sage conversion is via ``lig-to-sage`` with the optional ``[sage]`` extra.

.. toctree::
   :maxdepth: 2
   :caption: Available Recipes:

   ./recipes/lazyligand.rst
   ./recipes/freeligand.rst
