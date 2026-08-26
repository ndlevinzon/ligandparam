Dihedral corrections (ffpopt + scission)
=======================================

After a ligandparam recipe finishes, you typically have an Amber **triplet**:

* ``{label}.mol2`` - charged structure
* ``{label}.lib`` - Leap library
* ``{label}.frcmod`` - parmchk / GAFF parameters

Optional **dihedral correction** improves torsion parameters by fitting
against a high-level energy model along rotatable bonds, then merging DIHE
terms into a new parent frcmod.

Two ffpopt modes
----------------

``lig-dihed-correct`` always starts from the Amber triplet. Choose how the
molecule is scanned:

**Fragment (default).** Scission cuts the parent at rotatable bonds, ffpopt
twists each cap in its own directory, then DIHE terms are merged by atom
type into ``{label}.dihed.frcmod``. Cheaper per-opt; good for typical
drug-like ligands. Fragments with one or two fit bonds keep independent
1-D wavefronts; a fragment with more rotors switches to whole-ligand
joint packing so those dihedrals are one correlated system.

.. code-block:: bash

   lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 10 --fast

**Whole-ligand** (``--whole-ligand``). Skip scission and twist rotatable
bonds on the intact parent. Use when fragments would distort coupled rotors
(detergents, fused systems). Optional extras (all default off):
``--soft-dihed-restraint``, ``--multi-centroid N``, ``--fit-full``,
``--fit-backend jax``, ``--boltzmann-charges``.

.. code-block:: bash

   lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 44 --fast \
       --whole-ligand --soft-dihed-restraint --fit-full --fit-backend jax

``--fast`` loosens optimizer / I/O presets; scan ``delta`` stays 10 deg so
HL and LL share one grid. Console lines are ASCII (``+/-``, ``deg``,
``chi^2``).

Pipeline
--------

Fragment path::

   lig-getparam              ->  mol2 + lib + frcmod
   scission fragment         ->  per-fragment parm7/rst7 + fit_torsions
   ffpopt twist (per fragment)
   merge DIHE by atom type   ->  {label}.dihed.frcmod  (lib unchanged)

Whole-ligand path::

   lig-getparam              ->  mol2 + lib + frcmod
   ffpopt twist (parent)     ->  {label}.dihed.frcmod  (lib unchanged)

Drop-mode fragment iterations accumulate DIHE from **all** ``itXX.frcmod``
files in order so earlier survivors are kept unless a later iteration
explicitly refits the same key.

CLI
---

.. code-block:: bash

   lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 10 --fast

``--fast`` / ``FFPOPT_FAST_WAVEFRONT=1`` loosens geomeTRIC / ASE converge
and shortens maxiter (wall-time trade). Scan ``delta`` stays 10 deg.
Explicit ``--delta`` and related knobs still win when not left at library
defaults. See :doc:`wavefront`.

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

* ``xtb`` - GFN2-xTB via tblite (recommended light default)
* ``aimnet2`` - AIMNet2 neural net (wB97M-D3); faster than DFT, often
  similar wall time to ``xtb`` on CPU. Variants: ``aimnet2-2025``,
  ``aimnet2-b973c``, ``aimnet2-nse``, ``aimnet2-pd``, ``aimnet2-rxn``
* ``ani2x`` / ``ani1x`` / ``ani1ccx`` - TorchANI (element limits apply)
* ``mace`` / ``mace-off23_*`` - MACE-OFF (pytorch + model files)
* Psi4 as ``theory/basis`` (separate psi4 environment)
* ``dftb2`` / ``dftb3`` - via Amber/sander SQM

Avoid ``sander`` as the HL target: that compares the force field to itself.

``qdpi2`` remains available if you install the DeepMD / qdpi stack.

Fit numerics (shape-match chi^2)
--------------------------------

GenDihedFit matches **profile shape**, not absolute energy zero:

* Objective: ``d = (hl - ll) - mean(hl - ll)`` (free vertical offset)
* Fixed-geometry path: ridge / truncated SVD + energy-domain :math:`V(\phi)`
  barrier; ``|PK|<=25`` is an Amber-safety valve only
* Nested ``nprim`` AIC (fewest harmonics in the AIC window)
* HL/LL scan JSONs are always angle-aligned before fitting

See :doc:`fourier_fit` for why unbounded least squares explodes, and
``src/ffpopt/README.md`` / :doc:`wavefront` for the wavefront expansion.

geomeTRIC notes
---------------

Constrained dihedral scans keep the frozen torsion via geomeTRIC. Constraint
files use the **target** dihedral (scan step), not the pre-twist snapshot.
If the optimizer fails twice to invert an IC step, upstream geomeTRIC tries a
Cartesian recovery that **cannot** keep constraints and raises
``Cannot continue a constrained optimization``. ffpopt runs geomeTRIC through
``python -m ffpopt.geom.Geometric``, which rebuilds the same constrained IC
system instead of aborting.

Soft / loose recoveries (``soft-maxiter``, ``*-soft``, ``loose``, ``*-loose``)
may fill the profile but follow the soft spawn policy (seed once; do not
displace a lower soft min with a worse hard point). ``--soft-dihed-restraint``
uses a harmonic k-ramp then an optional hard IC; see :doc:`wavefront`.

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
* For ``xtb``: ``pip install ".[tblite,dihed]"``
* For ``aimnet2``: Python **3.11-3.13** (not 3.14), then
  ``pip install torch --index-url https://download.pytorch.org/whl/cpu``
  and ``pip install ".[aimnet]"``. First run downloads weights (do this
  on a login node). CPU-only: ``export FFPOPT_AIMNET_DEVICE=cpu``.
  GPU Slurm: CUDA torch wheel, ``FFPOPT_AIMNET_DEVICE=cuda``, GPU
  partition + ``--gres=gpu``. Do not use ``-n`` as CPU-core count; the
  wavefront caps AIMNet2 workers to ``n_gpu * FFPOPT_AIMNET_PER_GPU``
  (default 4) and pins each spawn worker to one visible GPU.

See :doc:`ffpopt`, :doc:`wavefront`, :doc:`scission`, and :doc:`cli`.
