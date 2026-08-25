ffpopt
======

``ffpopt`` is the integrated force-field torsion optimizer under ``src/ffpopt``.
ligandparam uses it for post-parameterization dihedral correction
(:doc:`dihedrals`).

Package layout
--------------

.. code-block:: text

   ffpopt/
   +-- runtime/     # console logging, progress boards, CPU budget, --fast presets
   +-- scan/        # WavefrontEngine; WaveFront / WaveFrontND facades; mixins
   +-- workflows/   # twist, fragmented, whole-ligand, bond batches
   +-- dihed/       # Dihedrals facade; FitTypes, Fourier, ParmEd, solvers, pucker
   +-- geom/        # GeomOpt, Constraints, Restraints, Geometric, linear-torsion
   +-- affdo/       # log, charges, multi-centroid profiles
   +-- ase/, cpefit/, confsearch/, ...

Canonical imports:

.. code-block:: python

   from ffpopt.workflows import run_fragmented_dihed_twist_workflow
   from ffpopt.scan.WaveFront import run_dihed_wavefront
   from ffpopt.runtime.Console import attach_console_handlers

Primary API
-----------

Fragment path (default; scission + per-fragment twist + merge)::

   from ffpopt.workflows import run_fragmented_dihed_twist_workflow

   result = run_fragmented_dihed_twist_workflow(
       mol2="LIG.mol2",
       lib="LIG.lib",
       frcmod="LIG.frcmod",
       out_dir="fragments",
       merged_frcmod="LIG.dihed.frcmod",
       model="xtb",
       geometric_opt=True,
       nproc=8,
       maxiter=2,
   )

Whole-ligand path (no scission; AFFDO extras optional)::

   from ffpopt.workflows import run_whole_ligand_dihed_twist_workflow

   result = run_whole_ligand_dihed_twist_workflow(
       mol2="LIG.mol2",
       lib="LIG.lib",
       frcmod="LIG.frcmod",
       out_dir="whole_ligand_twist",
       model="xtb",
       nproc=44,
       fast_wavefront=True,
   )

Call from an ``if __name__ == "__main__":`` guard (wavefront uses spawn-mode
multiprocessing).

CLI (same two modes)::

   lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 10 --fast
   lig-dihed-correct ... --whole-ligand --soft-dihed-restraint --fit-full

The ligandparam wrapper is ``lig-dihed-correct`` /
:class:`~ligandparam.stages.FfpoptDihed.StageDihedTwistCorrection`. Prefer
that after ``lig-getparam``. See :doc:`dihedrals`.

Single-molecule twist (when you already have ``parm7`` / ``rst7`` and explicit
bonds) is :func:`ffpopt.workflows.run_dihed_twist_workflow`. Pass
``bond=[(i, j), ...]`` with **0-based** atom indices (CLI ``"i,j"`` strings
still work). Scission's ``fit_torsions`` use 1-based indices and are converted
at the fragmented-workflow boundary via
:func:`ffpopt.workflows.bonds0_from_scission_fit_torsions`.

Wavefront evaluate policy
-------------------------

How the scan expands, which neighbors it visits, and which opts it skips
are documented in :doc:`wavefront` (seed coalescing, von Neumann stencil,
calculator cache, ``--fast`` presets, soft-dihed k-ramp).

Profile minima and neighbor spawn share one policy in
``ffpopt.scan.WavefrontMixins.evaluate_wavefront_minimum`` (1-D and N-D):

* Soft first-at-bin: store and **spawn once** (coverage seed).
* Soft improves soft min: update; no spawn.
* Hard replaces soft only if ``E_hard <= E_soft``.
* Hard within energy threshold: quiet min update; no spawn.
* Hard below ``min - threshold``: update and spawn.

``loose`` / ``*-loose`` recoveries are treated like soft for spawn.
``--fast`` / ``FFPOPT_FAST_WAVEFRONT=1`` is a wall-time trade (looser
converge, shorter maxiter); it does not coarsen ``delta`` and does not
change this policy beyond soft/loose handling.
Packaged defaults for every ``export FFPOPT_*`` knob are in
``ffpopt/pkgdata/files/env_defaults.json``.

Fit objective (chi^2)
---------------------

GenDihedFit uses a **shape match**: mean-centered HL-LL residual
(``d = (hl - ll) - mean(hl - ll)``). Under fixed geometry, force constants
enter linearly and are solved with bounded ``lsq_linear`` (phase fixed at 0).
Status lines in logs use ASCII (``cond~=``, ``chi^2``) for Slurm / Windows
compatibility.

ligandparam wrapper
-------------------

:class:`~ligandparam.stages.FfpoptDihed.StageDihedTwistCorrection` and the
``lig-dihed-correct`` CLI wrap both workflows (fragment default;
``--whole-ligand`` for the parent). Prefer those for everyday use after
``lig-getparam``.

Module reference
----------------

.. automodule:: ffpopt
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ffpopt.workflows
   :members: run_fragmented_dihed_twist_workflow, run_dihed_twist_workflow, run_whole_ligand_dihed_twist_workflow
   :undoc-members:
   :show-inheritance:

Runtime package is ``src/ffpopt``. An optional ``ffpopt-main/`` checkout is
upstream reference only. See also :doc:`wavefront`, ``src/ffpopt/GLOSSARY.md``,
and ``src/ffpopt/README.md``.
