Overview
========

``ligandparam`` provides a stage-based interface for parameterizing nonstandard
ligands and residues for Amber MD. Recipes such as
:class:`~ligandparam.recipes.FreeLigand` and
:class:`~ligandparam.recipes.LazyLigand` assemble a pipeline; each stage wraps
a concrete step (Gaussian ESP, RESP fitting, Leap, …).

Repository layout
-----------------

As of version **1.4**, the installable tree under ``src/`` is:

.. code-block:: text

   src/
   ├── ligandparam/   # recipes, stages, CLI (lig-getparam, …)
   ├── ffpopt/        # torsion / dihedral fitting (lig-dihed-correct)
   └── scission/      # ligand fragmentation (lig-scission / scission)

``ligandparam`` owns parameterization (charges, typing, baseline
``frcmod`` / ``lib``). ``ffpopt`` + ``scission`` own optional **post-hoc**
torsion correction on that Amber triplet. After ``pip install``, only the
packages under ``src/`` are used. Optional ``ffpopt-main/`` / ``scission-main/``
checkouts (often gitignored) are upstream reference trees only — not a runtime
dependency.

Multi-orientation RESP
----------------------

:class:`~ligandparam.recipes.FreeLigand` (and
:class:`~ligandparam.recipes.DPFreeLigand`) sample multiple ligand orientations
before averaging charges. The default ``so3_n28`` protocol uses a fixed
28-point quaternion pack that covers SO(3) more uniformly than the historical
Euler alpha/beta grid (``legacy_euler``). Both protocols keep the same job
count and feed the same multi-RESP → ``parmchk2`` → LEaP path.

See :mod:`ligandparam.io.orientations` and the :doc:`recipes` / :doc:`examples`
sections for details.

Optional dihedral corrections
-----------------------------

After a recipe finishes, you typically have ``{label}.mol2``,
``{label}.lib``, and ``{label}.frcmod``. Run :doc:`dihedrals` (CLI
``lig-dihed-correct``) to fragment with scission, fit torsions against a
high-level model (for example ``xtb`` or ``qdpi2``), and write a merged
``{label}.dihed.frcmod``. The ``lib`` is left unchanged.

See also :doc:`cli`, :doc:`ffpopt`, and :doc:`scission`.
