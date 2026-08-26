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
``FFPOPT_SOFT_DIHED_KMAX`` (default 8000). A hard-IC opt then runs from
those coords unless ``|dphi| <= 0.05`` deg (bias then ~0.003 kcal/mol,
skipped). Precheck still does not hard-snap. Logs: ``[wavefront]`` for
the rigid rotate, ``[affdo]`` for the k-ramp.

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
(``lsq_linear``, phase 0). Isolated linear guesses and joint LS share
the same residual. ``--fit-full`` optionally frees phase / period /
scee/scnb (SciPy L-BFGS-B or JAX).

Algorithms that keep wall-time down
-----------------------------------

MM then HL
~~~~~~~~~~

Under ``--fast`` (or ``FFPOPT_MM_THEN_HL=1``), each HL node is a cheap
constrained min then one dear refine. Sander is used when a ``parm7``
is on the structure; otherwise tblite GFN-FF. Soft-dihed k-ramps run
entirely on MM; one XTB / AIMNet2 / QDpi2 opt follows at the final k (in-band) or
after the MM hard IC. Sander / GFN-FF scans skip this (already cheap).
Force off with ``FFPOPT_MM_THEN_HL=0``. MM failure falls back to HL from
the parent Cartesian. Logs: ``[wavefront]`` for the staging, ``[affdo]``
for k-ramp details.

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

* **Parent fragment pool** is breadth-first: as many fragments as
  ``min(nproc, n_fragments)`` (e.g. 11 x 5 on 60 cores). Flattening
  ``1 x nproc`` here parked every sibling behind one fat worker.
* **Inside a fragment worker**, bond scans flatten (never both axes
  ``> 1``) so a third spawn pool is not opened.
* **Whole-ligand** (not nested under a fragment pool) may keep a 2-D
  bond x wavefront split (e.g. 4 x 11 on 44 cores for 8 bonds) when
  ``prefer_wf_depth`` is on.
* Override with ``FFPOPT_PREF_WF_DEPTH=1`` (one fragment at a time) or
  ``FFPOPT_PREF_WF_BREADTH=1``.

CPU leases are held only during wavefront scan phases. PrepareInput /
GenDihedFit / compare release cores so siblings can grow.

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

Logging
-------

Stdout uses a leading ``[scope]`` token. The console formatter peels it
into ``TIMESTAMP [ffpopt:...] [scope] ...``:

* ``[wavefront]`` - checkpoint found/missing, starting scan, rigid-rotate
  seed, MM-then-HL staging, min-update / coalesce, node fail, progress,
  summary, finished
* ``[affdo]`` - soft-dihed k-ramp, extras, extended-fit chi^2
* ``[twist]`` - bond batches, skip_existing, GenDihedFit orchestration
* ``[ffpopt]`` - geomeTRIC / ASE recovery, isolated LS rank notes

All of these are ASCII (``+/-``, ``deg``, ``chi^2``) for latin-1 Slurm
``.out`` files.

See also
--------

* :doc:`ffpopt` - workflows and Python API
* :doc:`dihedrals` - ``lig-dihed-correct`` fragment vs whole-ligand
* :doc:`design_philosophy` - why 1-D and N-D share one engine
* ``src/ffpopt/GLOSSARY.md`` - short definitions
* ``src/ffpopt/README.md`` - package README (same policy table)
