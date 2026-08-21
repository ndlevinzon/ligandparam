ffpopt
======

``ffpopt`` is the integrated force-field torsion optimizer under ``src/ffpopt``.
ligandparam uses it for post-parameterization dihedral correction
(:doc:`dihedrals`).

Package layout
--------------

.. code-block:: text

   ffpopt/
   ├── runtime/     # console logging, progress boards, CPU budget, --fast presets
   ├── scan/        # WaveFront, WaveFrontND, wavefront_mixins, ScanAnalysis
   ├── workflows/   # twist, fragmented, whole-ligand, bond_batches
   ├── dihed/       # Dihedrals, math, fit_ext, pucker
   ├── geom/        # GeomOpt, Constraints, Restraints, geometric, linear_torsion
   ├── affdo/       # log, charges, multi-centroid profiles
   └── ase/, cpefit/, confsearch/, …

Canonical imports:

.. code-block:: python

   from ffpopt.workflows import run_fragmented_dihed_twist_workflow
   from ffpopt.scan.WaveFront import run_dihed_wavefront
   from ffpopt.runtime.console import attach_console_handlers

Primary API
-----------

.. code-block:: python

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

Call from an ``if __name__ == "__main__":`` guard (wavefront uses spawn-mode
multiprocessing).

Single-molecule twist (when you already have ``parm7`` / ``rst7`` and explicit
bonds) is :func:`ffpopt.workflows.run_dihed_twist_workflow`. Pass
``bond=[(i, j), ...]`` with **0-based** atom indices (CLI ``"i,j"`` strings
still work). Scission's ``fit_torsions`` use 1-based indices and are converted
at the fragmented-workflow boundary via
:func:`ffpopt.workflows.bonds0_from_scission_fit_torsions`.

Wavefront evaluate policy
-------------------------

Profile minima and neighbor spawn share one policy in
``ffpopt.scan.wavefront_mixins.evaluate_wavefront_minimum`` (1-D and N-D):

* Soft first-at-bin: store and **spawn once** (coverage seed).
* Soft improves soft min: update; no spawn.
* Hard replaces soft only if ``E_hard <= E_soft``.
* Hard within energy threshold: quiet min update; no spawn.
* Hard below ``min - threshold``: update and spawn.

``loose`` / ``*-loose`` recoveries are treated like soft for spawn.
``--fast`` / ``FFPOPT_FAST_WAVEFRONT=1`` is a wall-time trade (coarser Δ,
looser converge); it does not change this policy beyond soft/loose handling.

Fit objective (chi^2)
---------------------

GenDihedFit uses a **shape match**: mean-centered HL−LL residual
(``d = (hl - ll) - mean(hl - ll)``). Under fixed geometry, force constants
enter linearly and are solved with bounded ``lsq_linear`` (phase fixed at 0).
Status lines in logs use ASCII (``cond~=``, ``chi^2``) for Slurm / Windows
compatibility.

ligandparam wrapper
-------------------

:class:`~ligandparam.stages.ffpopt_dihed.StageDihedTwistCorrection` and the
``lig-dihed-correct`` CLI wrap the fragmented workflow. Prefer those for
everyday use after ``lig-getparam``.

Module reference
----------------

.. automodule:: ffpopt
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ffpopt.workflows
   :members: run_fragmented_dihed_twist_workflow, run_dihed_twist_workflow
   :undoc-members:
   :show-inheritance:

Runtime package is ``src/ffpopt``. An optional ``ffpopt-main/`` checkout is
upstream reference only. See also ``src/ffpopt/GLOSSARY.md`` and
``src/ffpopt/README.md``.
