Wavefront scans
===============

A **wavefront** is a parallel relaxed dihedral scan. Instead of walking
the torsion sequentially (0, 10, 20, ...), it seeds a few angles,
optimizes them, and expands to neighboring bins until neighboring
energies agree within a threshold. High-level (HL) and low-level (LL /
sander) scans share the same ``delta`` grid so GenDihedFit can shape-match
them.

Implementation: one class in ``ffpopt.scan.WavefrontEngine``. Public
imports stay on thin facades:

* ``ffpopt.scan.WaveFront`` - 1-D (``run_dihed_wavefront``, CLI
  ``ffpopt-DihedWavefront.py``)
* ``ffpopt.scan.WaveFrontND`` - N-D (same class; ``GetGridNeighbors``,
  CLI ``ffpopt-NDimWavefront.py``)

Shared IPC, soft-opt, evaluate policy, and drain loops live in
``ffpopt.scan.WavefrontMixins``. New checkpoints pickle as
``ffpopt.scan.WavefrontEngine.Wavefront``; loaders still map historical
``ffpopt.WaveFront`` / ``ffpopt.WaveFrontND`` names.

How the scan expands
--------------------

1. **Seed.** Optimize the unconstrained minimum (1-D) or each starting
   conformer (N-D). Snap the scanned dihedral(s) onto the ``delta`` grid
   and enqueue those bins.
2. **Optimize a node.** Rigid-rotate the ``RotateMask`` branch by wrapped
   ``dphi`` (clash-check; revert to the parent Cartesian on overlap), then
   constrained (or softly restrained) geometry optimization at that bin.
   geomeTRIC is the default; ASE BFGS / L-BFGS / FIRE is the recovery
   ladder. Sander LL scans default to ASE-first.
3. **Evaluate.** ``evaluate_wavefront_minimum``
   (in ``WavefrontMixins``)
   decides whether this energy becomes the bin minimum and whether the
   node may spawn neighbors (see below).
4. **Spawn.** Active nodes enqueue unvisited neighbor bins. 1-D neighbors
   are ``angle +/- delta``. N-D neighbors are the von Neumann stencil
   (axis-aligned only) by default.
5. **Stop.** No pending or in-flight nodes, or ``wf_max_levels`` is hit.

Several levels can be in flight at once; progress lines report completed /
pending / in-flight rather than a single "current level".

Evaluate policy (spawn vs quiet min)
------------------------------------

Profile minima and neighbor spawn share one policy
(``evaluate_wavefront_minimum``), 1-D and N-D:

.. list-table::
   :header-rows: 1
   :widths: 36 32 32

   * - Case
     - Profile min
     - Spawn?
   * - Soft, first at bin
     - Store soft energy/geom
     - Yes once (coverage seed)
   * - Soft, improves soft min
     - Update if lower
     - No
   * - Hard vs soft incumbent
     - Replace soft only if ``E_hard <= E_soft``
     - Only if hard accepted
   * - Hard, ``E < min`` within threshold
     - Update quietly
     - No
   * - Hard, ``E < min - threshold``
     - Update
     - Yes
   * - Hard, ``E >= min``
     - No change
     - No

``loose`` / ``*-loose`` recoveries are treated like soft for spawn.
``linear-torsion`` ASE rescue (near-linear constrained bends) is also
soft for spawn. ``--fast`` is a wall-time trade (looser converge, shorter
maxiter); it does **not** change this table, and it does **not** coarsen
``delta`` (HL and LL must share one grid).

Algorithms that keep the wavefront moving
-----------------------------------------

These are the pieces that decide *which* opts run and how they recover,
not the CPU packing (that is the next section).

Seed coalescing
~~~~~~~~~~~~~~~

Each grid location (snapped 1-D angle, or N-D global bin index) has at
most **one pending job**. If two finished nodes both want to seed the
same empty bin, the cheaper parent energy wins and replaces the queued
seed in place. If that bin is already **in flight**, the better seed is
deferred and enqueued when the in-flight opt finishes.

Without coalescing, bulky whole-ligand rotors enqueue many redundant
visits to the same angle. Occupancy is rebuilt from the resume queue on
checkpoint restart.

Re-expansion guards
~~~~~~~~~~~~~~~~~~~

The 1-D wavefront is BFS on the cycle ``C_{360/delta}``. After every
bin has a hard minimum, a tiny ``hard_significant_improve`` can ping-pong
two neighbors for dozens of levels (177 nodes / 31 levels for a 36-bin
scan). After the usual energy test, spawn is demoted when:

* that bin has already re-expanded ``FFPOPT_WF_MAX_EXPAND`` times
  (default 3);
* the profile is filled and ``dE`` is below
  ``FFPOPT_WF_COVERAGE_SPAWN_FACTOR`` times the usual threshold
  (default 4);
* the last ``FFPOPT_WF_PINGPONG_WINDOW`` spawns used at most two bins
  and this loc is one of them (default window 8).

The better energy is still stored. ``wf_max_levels`` now stops spawning
instead of raising.

Outlier rescue
~~~~~~~~~~~~~~

Inactive BFS nodes never spawn, so a wrong-basin bin can sit forever
(DDM 240 deg was ~6 kcal above 230/250). After each completion the engine
inspects that angle and its two cycle neighbors. If the stored min is a
discrete Laplacian spike
(``E - 0.5(E_- + E_+) >= FFPOPT_WF_RESCUE_KCAL``, default 2 kcal) or the
bin failed, it reseeds from the lower-energy neighbor. When both
neighbors agree, their coordinates are Kabsch-aligned and lerped.
At most ``FFPOPT_WF_RESCUE_MAX`` retries per bin (default 2). Logs:
``[wavefront] rescue angle 240 from 230+250 lerp (spike ... kcal)``.

Rigid-rotate seed
~~~~~~~~~~~~~~~~~

Neighbor (and first) nodes still copy the parent Cartesian. geomeTRIC
would otherwise slam a large IC step (e.g. current 11 deg, target 250
deg). ``RotateMask`` bipartitions about bond ``b-c`` so atom ``d``
moves; wrapped ``dphi`` is applied with a Rodrigues rotation, then the
same 0.8 Ang nonbonded / covalent precheck. Clash or a broken bond
reverts to the parent coords and the existing constraint / restraint
still holds. Same frozen (or soft) target; far fewer TRIC steps.

N-D neighbor stencil
~~~~~~~~~~~~~~~~~~~~

``GetGridNeighbors`` defaults to
**von Neumann** (axis-aligned, ``2 * ndim`` neighbors: 4 in 2-D, 6 in
3-D). That is enough to fill the grid. ``stencil="moore"`` restores the
old ``3**ndim - 1`` cube (diagonals included) for extra multi-starts,
not required for coverage.

Soft dihedral k-ramp
~~~~~~~~~~~~~~~~~~~~

``--soft-dihed-restraint`` (typical for ``--whole-ligand`` detergents)
does **not** hard-snap the scanned dihedral before opt. Before GeomOpt,
``seed_struct_rigid_dihed_rotates`` applies wrapped ``dphi`` to the
``RotateMask`` branch (shortest arc about ``b-c``) and clash-checks; a
clash keeps the parent Cartesian. The optimizer then sees a harmonic
spring (default ``k=500`` kcal/mol/rad^2, ``+/-0.5`` deg). If the angle
is still out of band, ``k`` doubles from the last coordinates up to
``FFPOPT_SOFT_DIHED_KMAX`` (default 8000), unless ``|dphi|`` exceeds
``FFPOPT_SOFT_DIHED_LOST_WELL_DEG`` (default 30 deg): that is a 180 deg
miss, not a 10 deg neighbor step, so the node fails and waits for
outlier rescue instead of yanking ``k`` to 1000. Once the k-ramp is
**in-band**, unconstrained hard IC is skipped (residual at 0.5 deg and
``k=500`` is ~0.02 kcal/mol). MM-only scans keep the restrained min.
Two-stage (MM then HL) does one restrained HL opt at the final ``k``.
While in-flight nodes produce no completions, the drain loop prints
``waiting: pending=N in-flight=M for Xs angles=...`` every 60s
(``FFPOPT_WF_HEARTBEAT_SEC``). Precheck still does not hard-snap. Logs:
``[wavefront]`` for the rigid rotate, ``[affdo]`` for the k-ramp
(``HL at final k (skip unconstrained hard IC)`` / ``lost well``).

geomeTRIC recovery
~~~~~~~~~~~~~~~~~~

Constrained scans keep the frozen torsion via geomeTRIC. If an IC step
fails, ffpopt rebuilds the constrained IC instead of aborting into
unconstrained Cartesian recovery. Further failures walk an ASE ladder
(BFGS, L-BFGS, FIRE) and optional ``*-soft`` / ``loose`` attempts. Broken
covalent geometry (flying atoms) aborts the node instead of burning
minutes on a detonating opt.

Shape-match fit (after the scan)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GenDihedFit is not the wavefront, but it is why the grid must stay
uniform. Objective: ``d = (hl - ll) - mean(hl - ll)`` (free vertical
offset). Under fixed geometry, force constants enter linearly
(ridge / truncated SVD, phase 0). Isolated linear guesses and joint LS share
the same residual. After AIC, a chemical-group table zeros or caps
remaining Vptp on alkane (including parmchk2 analog ``c6``),
sulfate/phosphate, polar sp3, and generic sp3 rotors.
``--fit-full`` optionally frees phase /
period / scee/scnb (SciPy L-BFGS-B or JAX). See :doc:`fourier_fit`.

Algorithms that keep wall-time down
-----------------------------------

MM then HL
~~~~~~~~~~

Under ``--fast`` (or ``FFPOPT_MM_THEN_HL=1``), each HL node is a cheap
constrained min then one dear refine. Sander is used when a ``parm7``
is on the structure; otherwise tblite GFN-FF. Soft-dihed k-ramps run
entirely on MM; once in-band, one restrained XTB / AIMNet2 / QDpi2 opt
follows at the final ``k`` (no unconstrained HL hard IC). Sander /
GFN-FF scans skip this (already cheap). Force off with
``FFPOPT_MM_THEN_HL=0``. MM failure falls back to HL from the parent
Cartesian. Logs: ``[wavefront]`` for the staging, ``[affdo]`` for
k-ramp details.

Node wall-clock
~~~~~~~~~~~~~~~

A single in-flight HL opt (``pending=0 in-flight=1``) can stall the
whole node: extra CPUs do not parallelize that SCF. After
``FFPOPT_WF_NODE_WALL_SEC`` (default 300 s) the worker ``os.fork`` child
is SIGTERM'd (wavefront pool workers are daemons, so
``multiprocessing.Process`` is not allowed). The node is marked failed
and a deferred neighbor seed can run. ``0`` disables. Linux only;
Windows runs the opt in-process. Logs:
``[wavefront] node wall timeout (300s) at 280``.

Persistent calculator cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``get_persistent_calc`` caches the expensive
base model (XTB / AIMNet2 / QDpi2 / sander) on ``ListOfStruct._ffpopt_calc_cache``.
Serial ``nproc=1`` checkpoints unbind that cache for the pickle, then
**restore it** so the next node does not rebuild XTB. Spawn workers never
receive the live handle: ``ListOfStruct.__getstate__`` drops ``calc`` and
the cache (sander ``InputOptions`` is not pickleable). Workers rebuild
once per process.

Reused spawn pool
~~~~~~~~~~~~~~~~~

``close_reused_wavefront_pool`` /
``_acquire_wavefront_pool`` keep one spawn pool per process when the
model, charge, parm, and ``nproc`` match. Sequential bonds in the same
worker skip Pool bootstrap.

Flattened vs nested ``nproc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``split_nproc_for_items`` (in ``ffpopt.runtime.FastWavefront``) splits
``nproc`` into ``(n_outer, n_inner)``.

* **Parent fragment pool** is two-phase. Cheap 1-D fragments (1-2 fit
  bonds) share the node first: as many workers as
  ``min(nproc, n_cheap)`` (e.g. 11 x 5 on 60 cores). Correlated
  fragments (3+ bonds) stay queued during that pool, then each runs
  alone with all ``-n`` cores so nested packing can be ``4 x 11`` instead
  of stalling at ``nproc=4``. Flattening ``1 x nproc`` on the cheap
  pool parked every sibling behind one fat worker.
* **Inside a 1–2-bond fragment worker**, bond scans flatten (never both
  axes ``> 1``) so a third spawn pool is not opened.
* **Whole-ligand**, and fragments with more than two fit bonds, keep a
  2-D bond x wavefront split when each wavefront is fat (e.g. 4 x 11 on
  44 cores; 4 x 15 on 60 cores / 8 bonds, not 6 x 10). Jobs are
  dispatched in waves of ``n_bond_workers``; leftover bonds re-split so
  the last scan is not parked at a skinny ``wf_nproc`` with idle
  siblings. A tiny lease (8 bonds at ``nproc=4``) stays sequential
  ``1 x nproc`` so leftover cores can widen remaining jobs instead of
  locking ``4 x wf=1`` for the whole HL+orig phase.
* Override with ``FFPOPT_PREF_WF_DEPTH=1`` (one cheap fragment at a time) or
  ``FFPOPT_PREF_WF_BREADTH=1``.

CPU leases are held only during wavefront scan phases. PrepareInput /
GenDihedFit / compare release cores so siblings can grow. A scan never
starts on one core when the budget can spare ``FFPOPT_MIN_WF_NPROC``
(at least 2); extra owners wait. Cheap 1-D fragments share the budget;
correlated fragments do not join that pool. Sequential leftover
bonds re-lease so free cores are picked up before the next phase.

``--fast`` presets
~~~~~~~~~~~~~~~~~~

``--fast`` / ``FFPOPT_FAST_WAVEFRONT=1`` (explicit CLI knobs still win):

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Knob
     - Library default
     - Fast preset
   * - ``delta``
     - 10 deg
     - **unchanged** (shared HL/LL grid)
   * - ``geometric_maxiter``
     - 500
     - 200
   * - ``geometric_converge``
     - GAU
     - GAU_LOOSE
   * - ``wf_convergence_threshold``
     - 0.01 eV
     - 0.05 eV
   * - ``ase_opt_tol``
     - 0.01
     - 0.03

Also under ``--fast``: QDpi2 optimizes with XTB-only forces then a full
QDpi2 single-point (``FFPOPT_QDPI2_OPT``); HL nodes MM-relax (sander or
GFN-FF) then one XTB/QDpi2 opt (``FFPOPT_MM_THEN_HL``); XTB/QDpi2 use
ASE-first; shorter ASE ladder. Packaged defaults for every ``FFPOPT_*``
knob: ``ffpopt/pkgdata/files/env_defaults.json``.

Pipelined HL + orig, skip_existing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Independent HL and reference-sander (``orig``) scans share one job
queue. ``skip_existing`` reuses on-disk JSON **only** when the companion
``.dat`` exists and the frame count is exactly ``360/delta``. ``itNN``
LL rescans warm-start from the prior LL checkpoint when present.
Fitted ``itNN.frcmod`` is also skipped: after a chemical-group table
change (for example adding ``c6``), delete the fit files and keep the
scan ``.dat`` / JSON so the wavefront is not rerun.

Logging
-------

Stdout uses a leading ``[scope]`` token. The console formatter peels it
into ``TIMESTAMP [ffpopt:...] [scope] ...``:

* ``[wavefront]`` - checkpoint found/missing, starting scan, rigid-rotate
  seed, MM-then-HL staging, min-update / coalesce / rescue, node fail /
  wall timeout, drain heartbeat, progress, summary, finished
* ``[affdo]`` - soft-dihed k-ramp (in-band / lost well), extras, extended-fit chi^2
* ``[twist]`` - bond batches, skip_existing, GenDihedFit orchestration
* ``[ffpopt]`` - geomeTRIC / ASE recovery, Fourier ridge / nprim AIC

All of these are ASCII (``+/-``, ``deg``, ``chi^2``) for latin-1 Slurm
``.out`` files.

See also
--------

* :doc:`ffpopt` - workflows and Python API
* :doc:`dihedrals` - ``lig-dihed-correct`` fragment vs whole-ligand
* :doc:`fourier_fit` - ridge / SVD, energy-domain barrier, nprim AIC
* :doc:`design_philosophy` - why 1-D and N-D share one engine
* ``src/ffpopt/GLOSSARY.md`` - short definitions
* ``src/ffpopt/README.md`` - package README (same policy table)
