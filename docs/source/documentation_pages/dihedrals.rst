Dihedral corrections (ffpopt + scission)
=======================================

After a ligandparam recipe finishes, you typically have an Amber **triplet**:

* ``{label}.mol2`` — charged structure
* ``{label}.lib`` — Leap library
* ``{label}.frcmod`` — parmchk / GAFF parameters

Optional **dihedral correction** improves torsion parameters by fitting
against a high-level energy model along rotatable bonds, then merging DIHE
terms into a new parent frcmod.

Pipeline
--------

.. code-block:: text

   lig-getparam  →  mol2 + lib + frcmod
         │
         ▼
   scission fragment  →  per-fragment parm7/rst7 + fit_torsions
         │
         ▼
   ffpopt twist (HL scan vs sander LL, GenDihedFit, rescan)
         │
         ▼
   merge DIHE by atom type  →  {label}.dihed.frcmod
                               (lib unchanged)

CLI
---

.. code-block:: bash

   lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 10

Python stage (also used when ``dihed_correct=True`` on FreeLigand / LazyLigand /
DPFreeLigand). Recipe kwargs ``dihed_delta`` and ``dihed_fragment_config``
mirror CLI ``--delta`` and scission ``FragmentConfig`` settings:

.. code-block:: python

   from ligandparam.stages import StageDihedTwistCorrection

   StageDihedTwistCorrection(
       "DihedTwist",
       main_input="chaps.mol2",
       cwd="CHA3/CHA",
       in_lib="chaps.lib",
       in_frcmod="chaps.frcmod",
       out_frcmod="chaps.dihed.frcmod",
       model="xtb",
       delta=10,
       nproc=10,
   ).execute()

High-level models
-----------------

Pass ``--model`` to ``lig-dihed-correct``. Useful options without qdpi:

* ``xtb`` — GFN2-xTB via tblite (recommended light default)
* ``ani2x`` / ``ani1x`` / ``ani1ccx`` — TorchANI (element limits apply)
* ``mace`` / ``mace-off23_*`` — MACE-OFF (pytorch + model files)
* ``aimnet2`` (and variants)
* Psi4 as ``theory/basis`` (separate psi4 environment)
* ``dftb2`` / ``dftb3`` — via Amber/sander SQM

Avoid ``sander`` as the HL target: that compares the force field to itself.

``qdpi2`` remains available if you install the DeepMD / qdpi stack.

geomeTRIC notes
---------------

Constrained dihedral scans keep the frozen torsion via geomeTRIC. If the
optimizer fails twice to invert an IC step, upstream geomeTRIC tries a
Cartesian recovery that **cannot** keep constraints and raises
``Cannot continue a constrained optimization``. ffpopt runs geomeTRIC through
``python -m ffpopt.geometric_compat``, which rebuilds the same constrained IC
system instead of aborting.

If opts are still unstable:

* Prefer Python **3.11/3.12** over very new interpreters (e.g. 3.14)
* Use a smaller wavefront angle step (``--delta 5``)
* Keep ``--coordsys tric`` (default); do **not** switch to ``cart`` for
  constrained scans
* Last resort: ``--no-geometric-opt`` (ASE BFGS)

Requirements
------------

* Integrated ``ffpopt`` and ``scission`` (installed with ligandparam from ``src/``)
* AmberTools ``tleap`` on ``PATH`` (scission writes fragment ``parm7`` / ``rst7``)
* Calculator stack for the chosen ``--model``

See :doc:`ffpopt`, :doc:`scission`, and :doc:`cli`.
