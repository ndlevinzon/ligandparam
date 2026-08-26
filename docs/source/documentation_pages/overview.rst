Overview
========

``ligandparam`` provides a stage-based interface for parameterizing nonstandard
ligands and residues for Amber MD. Recipes such as
:class:`~ligandparam.recipes.FreeLigand` and
:class:`~ligandparam.recipes.LazyLigand` assemble a pipeline; each stage wraps
a concrete step (Gaussian ESP, RESP fitting, Leap, ...).

Repository layout
-----------------

As of version **1.6**, the installable tree under ``src/`` is:

.. code-block:: text

   src/
   +-- ligandparam/          # recipes, stages, CLI (lig-getparam, ...)
   |   +-- recipes/
   |   +-- stages/           # includes StageDihedTwistCorrection
   |   +-- cli/
   |   +-- io/               # gaussian_io, leap_io, smiles, orientations, ...
   |   +-- ...
   +-- ffpopt/               # torsion / dihedral fitting (lig-dihed-correct)
   |   +-- runtime/          # console, progress boards, CPU budget, --fast
   |   +-- scan/             # WavefrontEngine + WaveFront / WaveFrontND facades
   |   +-- workflows/        # twist, fragmented, whole-ligand
   |   +-- dihed/            # thin Dihedrals facade; FitTypes / Fourier / ParmEd / solvers
   |   +-- geom/             # GeomOpt, constraints, geomeTRIC
   |   +-- affdo/            # optional AFFDO extras
   |   +-- ...
   +-- scission/             # ligand fragmentation (lig-scission / scission)

``ligandparam`` owns parameterization (charges, typing, baseline
``frcmod`` / ``lib``). ``ffpopt`` + ``scission`` own optional **post-hoc**
torsion correction on that Amber triplet. After ``pip install``, only the
packages under ``src/`` are used. Optional ``ffpopt-main/`` / ``scission-main/``
checkouts (often gitignored) are upstream reference trees only - not a runtime
dependency.

Canonical imports use ``ffpopt.workflows``, ``ffpopt.scan``, ``ffpopt.geom``,
and ``ffpopt.runtime``. New wavefront checkpoints pickle as
``ffpopt.scan.WavefrontEngine.Wavefront``. Loaders still map historical
``ffpopt.WaveFront`` / ``ffpopt.WaveFrontND`` names onto the ``scan`` facades.

Multi-orientation RESP
----------------------

:class:`~ligandparam.recipes.FreeLigand` (and
:class:`~ligandparam.recipes.DPFreeLigand`) sample multiple ligand orientations
before averaging charges. The default ``so3_n28`` protocol uses a fixed
28-point quaternion pack that covers SO(3) more uniformly than the historical
Euler alpha/beta grid (``legacy_euler``). Both protocols keep the same job
count and feed the same multi-RESP -> ``parmchk2`` -> LEaP path.

See :mod:`ligandparam.io.Orientations` and the :doc:`recipes` / :doc:`examples`
sections for details.

Optional dihedral corrections (ffpopt)
--------------------------------------

After a recipe finishes you have ``{label}.mol2``, ``{label}.lib``, and
``{label}.frcmod``. :doc:`dihedrals` (``lig-dihed-correct``) fits torsions
against a high-level model (``xtb``, ``qdpi2``, ...) and writes
``{label}.dihed.frcmod``. The ``lib`` is left unchanged.

Two ffpopt modes:

* **Fragment (default)** - scission caps, per-fragment wavefront, merge DIHE
  by atom type. Cheaper HL opts; good for typical drug-like ligands.
  Fragments with 3+ fit bonds use whole-ligand joint packing; those
  jobs wait until cheap 1-D fragments finish, then each takes all cores.
* **Whole-ligand** (``--whole-ligand``) - twist rotatable bonds on the intact
  parent. Use when fragments would distort coupled rotors. Optional extras:
  ``--soft-dihed-restraint``, ``--multi-centroid``, ``--fit-full``.

See also :doc:`cli`, :doc:`ffpopt`, :doc:`wavefront`, :doc:`fourier_fit`,
and :doc:`scission`.

For how we intend the code to stay maintainable (SOLID, DRY, KISS, YAGNI,
separation of concerns) and a maintainability score, see
:doc:`design_philosophy`.
