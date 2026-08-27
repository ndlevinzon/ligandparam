# Changelog

All notable changes to **ligandparam** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Fixed

- **Gaussian orientation ESP OOM** - Rotate pooled 28 ``so3_n28`` jobs as
  ``n_jobs x 1`` core and wrote the full ``--mem`` into every ``%MEM``
  header (Triton: 28 x 32 GB on a 32 GB node; ``slurmstepd`` reported
  6 ``oom_kill`` events). The empty ``RuntimeError`` from ``gau.call``
  hid the kill. Cores and GB are now split
  (``n_workers * %NProc <= --nproc``, ``n_workers * %MEM <= --mem``,
  at least 4 GB per job unless the allocation is smaller). Failed
  Gaussian / Amber subprocesses include returncode, SIGKILL / OOM hint,
  and stderr. ``ParmHelper`` slurm ``sed`` no longer uses an invalid
  ``\!`` escape (SyntaxWarning on Python 3.12+).

## [1.6.1] - 2026-08-27

### Added

- **Whole-ligand wavefront rescue** - a bin whose stored min is a discrete
  Laplacian spike vs its cycle neighbors (DDM 240 deg was ~6 kcal above
  230/250) is reseeded from the lower neighbor, Kabsch-lerping both when
  they agree. Failed / lost-well bins retry the same way. Caps:
  ``FFPOPT_WF_RESCUE_KCAL`` (default 2), ``FFPOPT_WF_RESCUE_MAX`` (2).
  Soft k-ramps abort when ``|dphi|`` exceeds
  ``FFPOPT_SOFT_DIHED_LOST_WELL_DEG`` (30 deg) instead of yanking k
  across a 180 deg miss.

- **Wavefront node wall-clock** - a single in-flight HL hard IC (DDM
  280 deg, ``pending=0 in-flight=1``) can stall the whole 60-core job
  for hours; extra CPUs cannot parallelize that opt. In-band MM k-ramps
  now skip the unconstrained HL hard IC and do one restrained HL at the
  final k (bias ~0.02 kcal). Any node still running after
  ``FFPOPT_WF_NODE_WALL_SEC`` (default 300s) is SIGTERM'd so a deferred
  neighbor seed can run. The timeout child uses ``os.fork`` (not
  ``multiprocessing.Process``) because wavefront pool workers are
  daemons.

### Fixed

- **Sugar ``c6`` rotor caps** - parmchk2 analogizes maltoside carbons to
  ``c3``, but the chemical-group table only listed ``c3``/``cx``/``cy``.
  Every ``c3-c6`` / ``c6-os`` torsion was classified unsaturated, so DDM
  fragment-2 terms such as ``h1-c3-c6-c6`` PK=-16 and ``oh-c3-c6-os``
  PK=15 sat on the 30 kcal ceiling. ``c6`` is now tetrahedral carbon:
  H-C-C-H/C alkane cap, C-O ether cap, generic sp3 reject.

- **ASE ``ignore_bad_restart_file`` FutureWarning** - ``ase.calculators.amber.SANDER``
  still uses the deprecated Calculator constructor, so every spawn worker
  reprinted the warning to stderr. The warning is filtered at process
  start (and via ``PYTHONWARNINGS`` for child interpreters); our MOPAC
  wrapper no longer forwards that keyword.

- **VAST stale log handles** - geomeTRIC's ``RawFileHandler`` flush on
  scratch raises ``OSError: [Errno 116] Stale file handle``; Python then
  dumps a ``Logging error`` traceback per optimizer step in spawn
  workers. ``Handler.handleError`` now swallows ESTALE and reopens
  file-backed handlers. Fragment tees reopen ``frag-twist.log`` the same
  way.

- **Wavefront ping-pong** - after a 36-bin profile is filled, BFS can
  walk two neighboring angles for dozens of levels (DDM
  ``orig_10-11-19-16``: 177 nodes / 31 levels, last 12 levels were a
  2-node cycle). Re-expansion now stops via an expand cap per bin (3),
  a coverage Cauchy test (spawn only if ``dE`` is 4x the usual
  threshold), and a ping-pong detector on the last 8 spawns.
  ``max_levels`` demotes spawn instead of raising. Tunable:
  ``FFPOPT_WF_MAX_EXPAND``, ``FFPOPT_WF_COVERAGE_SPAWN_FACTOR``,
  ``FFPOPT_WF_PINGPONG_WINDOW``.

- **MM-only hard-IC stall** - sander rescans with ``--soft-dihed-restraint``
  logged ``in-band at k=500; finishing with hard IC`` then went silent for
  an hour. The drain loop only prints on checkpoint, so a long ASE/geomeTRIC
  hard IC looks hung. In-band MM k-ramps skip that second opt; two-stage
  HL does one restrained opt at the final k (no unconstrained hard IC).
  The drain loop heartbeats every 60s (``FFPOPT_WF_HEARTBEAT_SEC``) with
  pending/in-flight angles.

- **Nested bond leftover waves** - a phase-2 correlated fragment (no
  shared CPU budget) dumped all 8 bonds into a 6-worker pool at
  ``wf_nproc=10``. After 6 finished, the last two stayed at 10 workers
  with idle siblings, so the Slurm log froze at ``finished 7/8``. Bond
  scans now dispatch one wave of ``n_bond_workers``, then re-split
  remaining jobs (60 cores / 8 bonds is ``4 x 15`` twice). Nested
  ``min_inner`` keeps those wavefronts fat. Each bond logs
  ``starting scan`` as it begins.

- **Chemical-group dihedral caps** - fit keys are ``{res}_{types}``
  (``CHA_c3-c3-c3-h1``). The rotor classifier treated the residue as a
  fifth Amber type and marked every torsion unsaturated, so alkane /
  sulfate / n4 caps never ran. Parser now strips the residue prefix.
  All-zero FCs are omitted from the fragment frcmod so merge keeps
  parent GAFF instead of writing PK=0.
- **Wavefront node failure I/O** - leftover geomeTRIC ``{prefix}.tmp`` dirs
  no longer abort with ``FileExistsError``; node pickle writes use unique
  tmp names, fsync, and retries so NFS ``os.replace`` cannot kill the pool.
  A broken-geometry node is marked failed instead of crashing
  ``lig-dihed-correct``. Completed-node pickle cleanup uses
  ``unlink(missing_ok=True)`` so a vanished ``*_node.pckl`` on VAST
  cannot kill the pool. Dihedral ``arccos`` clips collinear atoms.
- **Fragment pool breadth** - ``--fast`` xTB no longer dumps ``-n`` onto
  one fragment (``flatten_nested`` was collapsing 11 jobs on 60 cores
  to ``1 x 60``). The parent pool runs ``min(nproc, n_fragments)``
  workers; each leases a fair share. ``FFPOPT_PREF_WF_DEPTH=1`` still
  serializes fragments.
- **Fragment multi-centroid** - ConfSearch used the parent mol2 (e.g. 101
  atoms) and wrote those coords onto fragment ``start.json`` (e.g. 17
  atoms). Fragment twists now ConfSearch ``fragment.mol2``; mismatched
  atom counts fall back to the fragment geometry instead of crashing
  ASE.
- **Whole-ligand freeze after orig scan** - after ``wavefront plot saved``,
  the bond worker still held the 113-node ``wf_run`` plus all scan
  frames and pickled them back to the parent; the parent ``pool.map``
  stayed silent until every interleaved HL+orig job finished. Scan
  returns keep only energies; ``.pkl`` writes are atomic; matplotlib
  uses Agg; reused spawn pools log close/join; the parent logs each
  finished bond as it completes.
- **Whole-ligand empty-scan abort** - a bulky DDM seed can start at
  ASE fmax 50-86 eV/Ang; the explode guard then stored 0 scan angles
  and ``pool.imap_unordered`` killed the whole job. MM-then-HL preopt
  now uses geomeTRIC (not ASE-first); sander/GFN-FF may continue past
  the explode fmax with small steps. A failed bond is logged and
  dropped from the fit so sibling torsions keep running.
- **Dihedral FC cap** - barrier-only still wrote PK of thousands because
  unbounded ``lstsq`` seeded the fit, then ``x0 +/- 2/5`` kept that
  explosion. ``FFPOPT_DIHED_FC_MAX`` (default 25 kcal/mol) remains as an
  Amber-safety valve; the model is now ridge / SVD + energy-domain
  barrier (see Added).

### Added

- **Success quote** - after ``lig-dihed-correct`` writes a dihedral
  frcmod with no failed fragments, stdout logs
  ``YYYY-mm-dd HH:MM:SS [ligandparam] INFO: LIGANDPARAM reminds you: ...``
  from a random line in ``src/ligandparam/pkgdata/quotes.txt`` (one quote
  per line; spoken quotes in the file are kept, not wrapped again).

- **Two-phase fragment CPU schedule** - cheap 1-D fragments (1-2 fit
  bonds) share ``-n`` in parallel first. Correlated / AFFDO-style
  fragments (3+ bonds) stay queued and do not reserve cores. After the
  1-D pool finishes, each correlated fragment runs alone with all cores
  so an 8-bond pack is not stuck at ``nproc=4`` beside cheap siblings.

- **Weighted CPU juggling** - correlated / AFFDO-style fragments lease
  cores proportional to bond count (capped at 8) instead of an equal
  split with 1-D siblings. A scan never starts on one core when the
  budget can spare ``FFPOPT_MIN_WF_NPROC`` (at least 2): extra owners
  wait. Unfinished fragments reserve a share so the last leftover job
  can take the node instead of staying at ``nproc=1``. Correlated /
  AFFDO batches do not lock a shallow ``N x wf=1`` pool: they scan
  sequential fat wavefronts until the lease is wide enough to nest.
  Sequential bond scans re-lease remaining jobs; in-flight fat 2-D pools
  stay fixed.

- **XTB core split** - ``prefer_depth`` nproc packing now uses leftover cores
  (44 cores / 8 bonds -> 4x11, not 8x5 with 4 idle; pipelined HL+orig 16
  jobs -> 11x4, not 16x2 with 12 idle). BLAS/OpenMP stay 1 thread per
  worker unless already exported.

- **Correlated fragment twist** - fragmentation stays the default path.
  Fragments with at most two fit bonds keep independent 1-D wavefronts.
  A fragment with more rotors switches to whole-ligand packing
  (``FFPOPT_WHOLE_MAX_BONDS_PER_TWIST``, default 8) and nested
  bond x wavefront so those dihedrals are one correlated joint system.
  9+ rotors on one fragment still chunk at 8 with MM updates between
  batches, matching ``--whole-ligand``.

- **Fourier FC regularization** - unbounded ``lstsq`` + post-hoc PK clip is
  replaced by truncated SVD / Tikhonov ridge, an energy-domain cap on
  reconstructed ``V(phi)`` (default 2x data barrier, abs 30 kcal/mol on a
  dense grid), and nested ``nprim`` AIC. After AIC, a chemical-group table
  zeros or caps remaining ``V(phi)``: alkane cap 5 / reject 20; sulfate
  or phosphate cap 4 / reject 10; alcohol, ether, amine, thioether cap 8
  / reject 20; other sp3-sp3 reject 20; unsaturated (amide) keep the 30
  kcal ceiling. ``FFPOPT_DIHED_FC_MAX`` (25) remains an Amber-safety valve
  only. See Sphinx ``fourier_fit``.

- **AIMNet2 HL model** - ``--model aimnet2`` (PyPI ``aimnet``, wB97M-D3).
  Family aliases: ``aimnet2-2025``, ``aimnet2-b973c``, ``aimnet2-nse``,
  ``aimnet2-pd``, ``aimnet2-rxn`` (older ``aimnet2_wb97m`` / ``aimnet2_qr``
  still resolve). Extra ``pip install -e ".[aimnet]"`` (Python 3.11-3.13,
  PyTorch 2.8+). Under ``--fast``: ASE-first, MM-then-HL, wavefront depth
  like ``xtb``. ``FFPOPT_AIMNET_DEVICE=cpu|cuda``. On GPU, wavefront spawn
  workers are capped to ``n_gpu * FFPOPT_AIMNET_PER_GPU`` (default 4) and
  round-robin pinned to ``CUDA_VISIBLE_DEVICES`` so ``-n 44`` does not OOM.
- **Whole-ligand orig-vs-HL plots** - ``--whole-ligand`` now writes the
  same ``compare_{xtb}_vs_orig_{idxs}.png`` overlays as fragments, as soon
  as both profiles for a bond exist (and again vs ``itNN`` after each
  fit). Logs ``[twist] wrote ... (close|disagree; barrier HL=... LL=...)``.

---

## [1.6.0] - 2026-08-25

### Added

- **MM then HL** - under ``--fast``, HL wavefront nodes sander-relax (or
  GFN-FF) at the target angle, then one XTB/QDpi2 opt from those coords.
  Soft-dihed k-ramps on MM, one HL at final k or after the MM hard IC.
  ``FFPOPT_MM_THEN_HL=0`` restores full-HL opts.
- **Wavefront algorithms page** - Sphinx ``wavefront.rst`` documents expansion,
  evaluate/spawn policy, seed coalescing, N-D von Neumann neighbors, calculator
  cache, reused spawn pools, ``nproc`` flatten vs nest, ``--fast`` presets,
  soft-dihed k-ramp, rigid-rotate seed, MM-then-HL, and ``[wavefront]`` /
  ``[affdo]`` log scopes.
- **Rigid-rotate seed** - wavefront nodes Rodrigues-rotate the ``RotateMask``
  branch by wrapped ``dphi`` before GeomOpt (clash-check; keep parent coords
  on overlap) so TRIC does not slam large dihedral gaps.
- **Seed coalescing** - at most one pending wavefront job per grid location;
  cheaper parent energy replaces the queued seed (or is deferred if in-flight).
- **N-D von Neumann neighbors** - default stencil is axis-aligned only
  (``2 * ndim``); ``moore`` keeps the old cube including diagonals.
- **Checkpoint calc-cache restore** - serial checkpoints unbind then restore
  ``ListOfStruct._ffpopt_calc_cache`` so XTB/sander is not rebuilt every node.
- **``[wavefront]`` log scopes** - checkpoint/startup/progress/summary prints
  use the same ``[routine]`` convention as ``[affdo]`` / ``[twist]``.
- **AFFDO-style whole-ligand extras** (default off; fragmented path unchanged):
  - ``--whole-ligand`` / ``run_whole_ligand_dihed_twist_workflow`` - full-ligand twist without scission
  - ``--multi-centroid N`` - ConfSearch starts + smoothest HL profile (Fourier + roughness)
  - ``--soft-dihed-restraint`` - harmonic dihedral spring (500 kcal/mol/rad^2, +/-0.5 deg) with geomeTRIC
  - ``--fit-full`` / ``--fit-mode`` / ``--fit-backend {lsq,lbfgsb,jax}`` - phase, period, scee/scnb (or barrier-only)
  - ``--boltzmann-charges`` - Boltzmann-average charges over centroid mol2s
  - Optional extra: from the clone, ``pip install -e '.[jax]'`` (not PyPI ``ligandparam[jax]``, which is 1.0.0)
  - Tagged stdout ``[affdo]`` lines for extras, centroid ConfSearch, profile scores, Boltzmann weights, soft-restraint fallbacks, and extended-fit chi^2 / parameters

### Changed

- **Unified wavefront engine** - 1-D and N-D share ``WavefrontEngine.Wavefront``;
  ``WaveFront.py`` / ``WaveFrontND.py`` are pickle-stable re-export facades.
  ``Dihedrals.py`` is a thin facade over FitTypes / Fourier / ParmEd / solvers.
- **ffpopt layout** - domain code grouped into ``workflows/``, ``dihed/``, ``geom/``, and ``affdo/``. Small siblings merged (centroid+profile select; geomeTRIC compat+in-process driver). ``Workflows.py`` split by entry point. Root import-redirect shims removed; callers use the canonical packages (``python -m ffpopt.geom.Geometric``).
- **PascalCase library modules** - snake_case ``src/`` modules renamed to descriptive PascalCase (``TwistHelpers``, ``AffdoLog``, ``LigandIo``, ...) to match existing ffpopt domain files. Package directories stay lowercase. Hyphenated ``ffpopt.bin`` CLIs unchanged.
- **Git case-only names** - Windows ``core.ignorecase`` left ``log.py`` / ``scripts.py`` / ``cli.py`` in the index while imports expect ``Log`` / ``Scripts`` / ``Cli``. Re-recorded with two-step ``git mv`` so Linux checkouts match.
- **Docs (ASCII)** - README, Sphinx pages, comments, and changelogs use ASCII stand-ins (``+/-``, ``deg``, ``chi^2``, ``->``). README now leads with ffpopt **fragment** vs **whole-ligand** modes.
- **Whole-ligand logs** - ``--whole-ligand`` / AFFDO runs now use the same identity pattern as fragment jobs: ``TIMESTAMP [ffpopt:torsion_batch_XX] [whole-twist]``. Per-batch ``whole-twist.log``, live ``WHOLE_STATUS.txt``, and an ASCII run card under the startup logo.
- **Env defaults JSON** - all user ``FFPOPT_*`` knobs live in ``ffpopt/pkgdata/files/env_defaults.json`` (commented JSONC; this is the store the code reads). Overlay with ``FFPOPT_DEFAULTS=/path.json``; per-key ``export FFPOPT_*=`` still wins.
- **Multi-centroid HL pooling** - centroid-0 HL and ``orig`` share one job queue; extra ConfSearch starts run only for jagged torsions (Fourier RMSE vs ``FFPOPT_CENTROID_FOURIER_MAX``, default 0.5 kcal) and those centroidxbond jobs share one pool.
- **Whole-ligand core use** - ``--whole-ligand --fast`` no longer serializes 2-bond batches behind a 1xnproc wavefront. Top-level twist keeps a nested bondxwavefront split (e.g. 8x5 on 44 cores). Fragment spawn workers still flatten to one axis. Pack size is ``FFPOPT_WHOLE_MAX_BONDS_PER_TWIST`` (default 8).
- **Wavefront node logs** - min-update (``New angle detected``), seed coalesce/defer, opt error/fail, spawn, pickle reuse, and precheck lines now use ``[wavefront]``.

### Fixed

- **``_split_fragment_nproc``** - bond-scan pooling in ``TwistHelpers`` called a helper that stayed behind in ``FragmentedTwist`` after the workflow split (``NameError`` on ``lig-dihed-correct --whole-ligand``).
- **Layout-relative imports** - ``ffpopt.geom.Constraints`` imported ``AmberParm`` as a sibling (``ffpopt.geom.AmberParm``); ``FindFuncGrps`` still imported ``ffpopt.Dihedrals``. Both now use the canonical modules. A developer test walks ``src/`` import graphs so this class of miss does not ship again.
- **Broken-geometry abort** - whole-ligand wavefront nodes that already have crushed/dissociated covalent bonds (or huge starting forces) skip the optimizer instead of spending minutes watching hydrogens/carbons fly off. Mid-opt ASE/geomeTRIC steps abort on the same check.
- **geomeTRIC scratch cleanup** - ``.nsf`` logs, ``{prefix}.tmp`` dirs, and other geomeTRIC sidecars are removed after each opt, when a completed node is folded into the checkpoint, and when a scan/fragment is resumed or skipped. Incomplete nodes keep ``_optim.xyz`` so a killed opt can still warm-start.
- **``--fit-backend jax`` without jax** - fall back to SciPy L-BFGS-B with an ``[affdo]`` note instead of aborting GenDihedFit. From the clone, ``pip install -e '.[jax]'`` (PyPI ``ligandparam[jax]`` is 1.0.0).
- **JAX CPU/x64 for GenDihedFit** - default ``JAX_PLATFORMS=cpu`` and ``jax_enable_x64`` so CHPC CPU nodes do not load the CUDA plugin (``CUDA_ERROR_NO_DEVICE``) and kcal/mol residuals stay float64. Dropped SciPy ``disp`` (unknown to current L-BFGS-B).
- **HL/LL scan-grid skip** - ``skip_existing`` no longer reuses a scan JSON whose frame count is a different uniform 360/n grid than the current ``delta``.
- **``--fast`` keeps ``delta=10``** - coarser 15 deg HL vs leftover 10 deg orig was collapsing fits to 12 shared angles. Fast mode still loosens geomeTRIC / I/O, not the scan grid.
- **Mismatched leftover scans** - if both files still look like full 360/n grids, GenDihedFit interpolates HL energies onto the LL angles (keeps MM geometries) instead of dropping to the intersection.
- **JAX CPU before first import** - ``JAX_PLATFORMS=cpu`` / ``CUDA_VISIBLE_DEVICES=-1`` are set at GenDihedFit process start and in ``jax_is_available()``, not only inside the JAX objective factory.
- **Incomplete skip_existing scans** - reuse requires the companion ``.dat`` and exactly ``360/delta`` frames. JSON-only leftovers (killed after JSON, before ``.dat``) are rescanned instead of later crashing ``np.loadtxt``.
- **Missing centroid HL profiles** - if no centroid scan is scoreable, twist raises with the candidate paths instead of warning and then dying in HL/LL compare on ``xtb_<idxs>.dat``.
- **Wavefront plot margins** - ``tight_layout`` UserWarning on dense angle/level grids is suppressed; plots still save with ``bbox_inches='tight'``.
- **Empty wavefront ``np.amin`` crash** - a scan with no accepted angles (every seed clash-rejected, or every opt failed) raised ``ValueError: zero-size array``. It now prints the node summary and raises ``RuntimeError`` with the failed-node list.
- **``--soft-dihed-restraint`` seed clash** - hard-twist clash checks no longer snap the scanned dihedral before opt (that rejected every bulky whole-ligand seed). If every seed still fails, a native-angle node is forced so the wavefront can start.
- **Soft-dihedral k-doubling + selective hard IC** - if the harmonic misses the ``+/-tol`` band, ``k`` doubles up to ``--soft-dihed-kmax`` / ``FFPOPT_SOFT_DIHED_KMAX`` (default 8000 kcal/mol/rad^2) from the last coordinates. A hard-IC opt then runs from those coords unless the restrained min is already within 0.05 deg of ``phi0`` (bias then ~0.003 kcal/mol).
- **``--fit-full`` duplicate periodicity** - rounding optimized Fourier periods onto the same integer made ParmEd reject ``DihedralTypeList.append`` (``Cannot add two DihedralType instances with the same periodicity``). Same-``n`` terms are now merged (phase 0 vs 180 subtracts; same phase sums) before frcmod / parm7 write.
- **GenDihedFit missing imports after Dihedrals split** - ``shape_match_delta`` in ``IsolatedLinearSolve``, ``WriteParmedScript`` in ``SystemType.write_output``, and ``merge_duplicate_period_prims`` in ``WriteParmedScript`` (``--fit-full`` apply).
- **Spawn pickle of live sander calculators** - ``ListOfStruct.__getstate__`` drops ``_ffpopt_calc_cache`` so nested wavefront pools do not serialize ``sander.pysander.InputOptions``.
- **Truncated GenDihedFit ``itNN.py`` treated as complete** - ``skip_existing`` reused a script that existed after a crash mid-``WriteParmedScript`` (no ``p.save``). Apply then "succeeded" without ``itNN.parm7``, and PrepareInput died. Skip now requires ``.py`` + ``.frcmod`` and ``p.save(``; apply raises if the parm7 is missing; script write is atomic (``.tmp`` then replace).

---

## [1.5.1] - 2026-08-21

### Changed

- **Fragment CPU saturation** - CPU leases are held only during wavefront scan phases (released for PrepareInput / GenDihedFit / compare); small fair-share leases prefer bond/fragment breadth over a single narrow wavefront; ``OMP_NUM_THREADS=1`` when unset on fragmented entry. Env overrides: ``FFPOPT_PREF_WF_DEPTH``, ``FFPOPT_PREF_WF_BREADTH``.
- **Flattened spawn parallelism** - bondxwavefront splits never nest both axes (``split_nproc_for_items(..., flatten_nested=True)``); fragment workers skip bond pools; wavefront worker pools are reused across sequential bonds in-process.
- **Pipelined HL ? orig scans** - independent high-level and reference-sander scans share one job queue.
- **Cheaper HL opts under ``--fast``** - QDpi2 optimizes with XTB-only forces then full QDpi2 single-point (``FFPOPT_QDPI2_OPT``, ``FFPOPT_QDPI2_REFINE``); ASE-first for XTB/QDpi2; shorter ASE optimizer ladder; skip geomeTRIC fallback after ASE failure.
- **Tighter MM E/F** - ``SanderCalculator`` / restrained MM path calls ``sander.set_positions`` + ``energy_forces`` directly when possible.
- **Conservative multi-bond batching** - fragments with more than ``FFPOPT_MAX_BONDS_PER_TWIST`` (default 2) fit torsions pack into sequential batches by covalent proximity (``FFPOPT_BOND_COUPLE_RADIUS``, default 2); MM is updated between batches; no bytype scan dedupe. Disable with ``FFPOPT_BOND_BATCH=0``.
- **Faster fragment frcmod merge** - locate ``itXX.frcmod`` via targeted probes (avoid listing wavefront-heavy fragment dirs on VAST/Lustre); load fragment updates concurrently.
- **bytype scan collisions** - two fragments scanning the same DIHE atom-type family no longer abort merge; first scanned contributor wins (identical terms keep first), recorded under ``conflicts`` with a warning.

---

## [1.5.0] - 2026-08-10

### Added

- **Startup banner** - CLI / console attach prints the ligandparam ASCII logo, authors, and version at the top of stdout (once per process).
- **Wavefront pickle path aliases** - ``ffpopt.WaveFront`` / ``ffpopt.WaveFrontND`` re-export ``scan.*`` so checkpoints written before the ``scan/`` move still load.
- **Install validation suite** - ``python -m unittest tests.test_install_validation -v`` checks packages, CLIs, and core helpers after ``pip install``.
- **Developer regression suite** - ``python -m unittest tests.test_developer_regression -v`` covers recipe ``setup()`` graphs, logging contracts, Amber bundle I/O, dihed/bond helpers, scission torsions/merge, and ffpopt pure helpers. Specialized / duplicate unit tests were removed; these two modules are the supported test entry points.
- **ASCII fit logs** - dihedral-fit status lines use ``cond~=`` / ``chi^2`` (and concise ``lsq_linear`` summaries) so Windows / latin-1 Slurm logs do not mojibake Unicode.

### Fixed

- **geomeTRIC constraint targets** - constraint files / enforce paths use the scan **target** dihedral (``force=False`` fill), not the post-twist ``force=True`` snapshot.
- **Wavefront evaluate policy** - soft first-at-bin seeds spawn once; quiet min updates within threshold; hard replaces soft only if ``E_hard <= E_soft``; ``loose`` / ``*-loose`` recoveries treated as soft for spawn. Shared helper in ``ffpopt.scan.WavefrontMixins`` (1-D and N-D).
- **HL/LL angle align** - GenDihedFit always angle-aligns profiles (not only when lengths differ); empty common-angle sets raise.
- **Drop-mode frcmod merge** - fragment DIHE merge accumulates **all** ``itXX.frcmod`` in order so earlier survivors are kept unless a later file refits the same key.
- **Dihedral fit chi^2 / solver** - objective is mean-centered shape match only (invariant to a constant LL offset); joint linear initial guess over all fitted torsions; ``lstsq`` / ``lsq_linear`` for fixed-geometry FC fits (replaces COBYLA); reopt mode uses L-BFGS-B.
- **Wavefront checkpoints** - atomic pickle writes (``tmp`` + ``os.replace``); N-D ``restart_options`` restores soft-opt attrs like 1-D.

### Changed

- **Package layout** - ``ffpopt.runtime/`` (console, progress boards, CPU budget, fast presets) and ``ffpopt.scan/`` (WaveFront engines + mixins); import the canonical packages (root ``WaveFront`` / ``WaveFrontND`` exist only as pickle-compat aliases). ligandparam: ``gaussian_io`` / ``leap_io`` / ``smiles_to_pdb``; recipe charge->parmchk->leap tail in ``recipes.common``.
- **Docs** - Sphinx overview / installation / ffpopt / dihedrals / scission / CLI updated for 1.5 layout, wavefront policy, shape-match chi^2, cumulative merge, and the two supported test modules.
- **Further DRY** - MCS/PDB atom-name helpers in ``ligandparam.io.Smiles``; StageSmilesToPDB uses ``PDBFromSMILES``; ``MakeUniqueParams`` / ``Disang``; shared ``load_wavefront_pickle``; cpefit/Gaussian budgets via ``split_core_budget``; Reader owns ``FixParmedAtomicNumbers`` / ``ReadMol2``; UFF radius from ``constants``.
- ligandparam: lazy ``stages`` / ``recipes`` exports (incl. Sage/Build); ``recipes.registry.get_recipe``; ``deprecated/`` removed; deleted redundant ``gaussian_budget.py``.

---

## [1.4.1] - 2026-08-07

Parallelism and ConfSearch follow-ons after the v1.4.0 ffpopt/scission merge.

### Performance

- **Fast wavefront mode** (`lig-dihed-correct --fast` / `FFPOPT_FAST_WAVEFRONT=1`) - looser geomeTRIC converge (`GAU_LOOSE`), `geometric_maxiter=200`, `delta=15`, milder wavefront energy threshold; shorter recovery ladder (skip alt coordsys); less frequent checkpoints and skip success `*_node.pckl`; for XTB prefer wavefront depth over fragment breadth when splitting `-n`. Explicit non-default knobs still win. Related: `FFPOPT_GEOMOPT_FAST_RECOVERY`, `FFPOPT_WF_CHECKPOINT_EVERY`, `FFPOPT_WF_NODE_PICKLE`, `FFPOPT_PREF_WF_DEPTH`, `FFPOPT_MIN_WF_NPROC`, `FFPOPT_GEOMOPT_VERBOSE`.
- **GeomOpt I/O cuts** - in-process path skips duplicate pre-write XYZ (geomeTRIC writes it); `clone_geometry` instead of full Struct deepcopy on success.
- **ConfSearch RMS matrix** - for ensembles at/above `FFPOPT_CONFSEARCH_RMS_FAST_N` (default **50**), align once to the first conformer and use vectorized heavy-atom RMS for Butina clustering instead of per-pair `GetBestRMS`. Set the env var to `0` to force the legacy path.
- **Parallel multi-ESP rotations** (`StageGaussianRotation`) - process pool over orientation `.com` jobs; `nproc` is a total core budget (`n_workers x %NProc <= nproc`). Per-job bash scripts and `GAUSS_SCRDIR`.
- **Parallel per-bond scans** - `run_dihed_twist_workflow` pools HL / reference / iteration wavefront scans across bonds (`n_bond_workers x wf_nproc`).
- **Parallel RespFit / CpeFit conformer ESP** - independent per-conformer ab initio ESP (and cosmo/harmonic sets) in a process pool; charge / CPE fit stays serial (O(natoms) assembly).
- **Parallel scission screen / writes** - pool over torsions for candidate screening and over fragments for `parmchk2`/`tleap`; `FragmentConfig.nproc` / CLI `--nproc` / fragmented-workflow `nproc`.
- **Parallel fragments** - `run_fragmented_dihed_twist_workflow` splits `-n` / `nproc` across a non-daemon fragment pool and nested wavefront workers.
- **Vectorized `cap_site_scan_margin`** - NumPy rotate + margin vs retained heavies (same pattern as `screen_candidate`).
- **Cached retained APSP** - `retained_distance_map` keyed by `frozenset(retained)` for shell-sibling reuse.
- **Unique domain-shell enum** - skip left/right depths that do not change the domain set; build each fragment once.
- **`FindMinCycles` via cycle basis** - NetworkX `cycle_basis` instead of DFS min-path search (sugar puckers).
- **Cheaper wavefront / dihedral clones** - `Struct.clone_geometry` + `ListOfStruct.from_structs_shared` replace full deepcopies for coord updates and parm-path overrides; `MultiDihedFcn` uses shallow prim copies.

### Reliability

- Gaussian `call` uses unique submit scripts and env copies so concurrent jobs do not collide.
- Windows-safe `ligandparam.Utils` libc loading (no `ctypes.CDLL(None)` crash on import).
- **Non-daemon fragment/bond pools** - `_make_nondaemon_spawn_pool` no longer subclasses `ctx.Pool` (a factory method on Python 3.8+ / 3.14); uses `multiprocessing.pool.Pool` with a module-level non-daemon spawn `Process` so nested wavefront pools work on CHPC.
- **Robust GeomOpt recovery** - on geomeTRIC `GeomOptNotConvergedError`, restart from the last `_optim.xyz` frame with a ladder: `GAU_LOOSE` + more iterations, alternate `dlc`/`hdlc` coordsys, then soft `converge maxiter`. ASE fallback tries BFGS -> LBFGS -> FIRE and soft-accepts near-converged `fmax`. Disable ladder with `FFPOPT_GEOMOPT_ROBUST=0`; tune soft ASE with `FFPOPT_ASE_LOOSE_FMAX`.
- **`--geometric-opt` help** - corrected to match code (flag prefers geomeTRIC; default ASE-first).
- **WaveFront soft-opt gates** - soft-maxiter / ASE `*-soft` geometries may fill the profile but cannot displace hard-converged minima and do not spawn neighbors; tags survive slim IPC.
- **WaveFront failure reporting** - precheck exceptions are `precheck_error` (not all `clash_precheck`); summary lists failed / soft-accepted counts instead of always saying "successfully."
- **Constraints f-string** - nested quotes fixed so `ffpopt.Constraints` imports on Python 3.10-3.11.
- **Noisier fallbacks** - calculator device / SANDER load / ConfSearch mol parse / Gaussian I/O report the underlying exception instead of bare `except:`.
- **Charge normalization** - safe for nonzero `net_charge`, tiny diffs, and `|diff| > natoms*precision` (residual on largest-|q| atom); asserts final sum.
- **Docs / packaging** - READMEs clarify `src/` is runtime SoT vs optional `*-main/` trees; `numpy<2` documented; optional `ligandparam[ml-potentials]` for MACE / TorchANI.
- **GenDihedFit HL/LL mismatch** - when wavefront HL and LL JSON scans differ in length (failed / soft-opt holes), align on shared `dXXX` scan angles instead of aborting; raise only if fewer than 3 common points remain.
- **Fragment-scan completion logs** - per-scan / per-fragment messages no longer say bare "done/completed"; parent logs `[frag-twist] all N fragment twist job(s) finished` only after the fragment pool joins, and CLI `Done` only after merge.
- **Fragment live status board** - parallel fragment twist runs write an ASCII board (`FRAG_STATUS.txt` + parent log) with per-fragment name / status / stage / detail; wavefront spam goes to `<frag>/frag-twist.log` instead of interleaving on the console.
- **Gaussian orientation status board** - FreeLigand / multi-ESP `StageGaussianRotation` writes the same style of board (`gaussianCalcs/ROT_STATUS.txt`) tracking each orientation/angle (`q012` or Euler triple), status, stage (`gaussian` / finished), and log detail while jobs run in parallel.
- **Slurm-friendly console logging** - per-fragment / recipe `.log` content is teed to stdout (INFO) and stderr (WARNING+) with timestamps and tags (`[ligandparam]`, `[ffpopt]`, `[ffpopt:<fragment_id>]`) so job `.out` / `.err` capture the full trail.
- **Gaussian resume / `-O`** - Gaussian stages skip logs that already show `Normal termination` (including partial multi-orientation ESP resumes). `lig-getparam -O` / `--force-gaussian-rerun` overrides the skip. Incomplete logs are re-run; a complete `gaussianCalcs/*.log` is promoted to the final path when needed.
- **Dynamic fragment CPU leases** - fragmented dihedral twist no longer freezes `nproc // n_frags` for the whole run. A shared `.cpu_budget.json` fair-shares cores at fragment start and again before each scan phase (`hl_scan` / `orig_scan` / `rescan`), so finished fragments free cores for remaining work.

---

## [1.4.0] - 2026-08-06

### 1. Package merge - ffpopt + scission (highest priority)

Vendored and installed the York-lab **ffpopt** and **scission** stacks inside ligandparam so dihedral fragmentation and twist fitting no longer depend on a separate out-of-tree install.

- **ffpopt** under `src/ffpopt/` - wavefront dihedral scans, GenDihedFit, GeomOpt (ASE / geomeTRIC), calculators (SANDER, XTB/tblite, ML models), and CLI entry points (`ffpopt-*.py`).
- **scission** under `src/scission/` - ligand fragmentation, rotatable-bond / torsion discovery, fragment parm/lib/frcmod prep, and merge of fragment DIHE terms back into a parent frcmod.
- **Packaging** - `pyproject.toml` includes `ligandparam*`, `ffpopt*`, and `scission*`; optional extras `[dihed]` (ndfes, geometric) and `[tblite]` (GFN2-xTB); console scripts `lig-dihed-correct`, `lig-scission`, `scission`, and the ffpopt bin wrappers.
- **Vendor trees** - `ffpopt-main/` / `scission-main/` kept for upstream reference; `.gitignore` clarified so the installed `src/` trees remain the source of truth.
- **Version bump** to **1.4.0** with docs pages for ffpopt, scission, dihedral correction stages, and CLI.

### 2. Performance enhancements

Targeted wall-time and I/O reductions for HL wavefront scans (especially `--model xtb`), GenDihedFit, scission screening, and related RESP / Gaussian paths.

#### Geometry optimization & XTB

- **In-process geomeTRIC** with a persistent calculator cache on `ListOfStruct` (avoids per-opt subprocess bootstrap and model reload). Legacy CLI path via `FFPOPT_GEOMETRIC_SUBPROCESS=1`.
- **Plain XYZ for geomeTRIC** (`write_plain_xyz`) - ASE's default `Atoms.write()` emitted extended XYZ with charge columns that broke `Molecule` parsing; fixed so in-process opts do not fall back to ASE spuriously.
- **XTB / tblite** - robust SCF defaults, env knobs (`FFPOPT_XTB_MAX_ITER`, `FFPOPT_XTB_ETEMP`, `FFPOPT_XTB_MIXER_DAMPING`, `FFPOPT_XTB_GUESS`), SCF retry ladder, and calculator reuse.
- **geomeTRIC robustness** - Cartesian IC recovery notice handling, stall watchdog / progress detection, Brent "Not bracketed" recovery; quieter GeomOpt ASE<->geomeTRIC fallbacks (one-line stderr; full traceback only with `FFPOPT_GEOMOPT_TRACEBACK=1`).

#### Wavefront & IPC

- **Slim wavefront IPC** - pool/MPI workers share `los` once via initializer / bcast; jobs carry angles/RCs + coords; results merge without shipping full calculators.
- **Lighter checkpoints** - clear live calc / `_ffpopt_calc_cache`, drop redundant forces on completed nodes, `pickle.HIGHEST_PROTOCOL`.
- **Vectorized clash precheck** (`has_nonbonded_clash`) replaces O(n^2) ASE `get_distance` loops; bonded pairs handled correctly via a bond mask.
- **Absolute paths beside `out`** for 1D/ND checkpoints, plots, and dat/pkl so absolute `out=` does not write into the launch cwd.

#### GenDihedFit & torsions

- **Fixed-geometry NL cache** - base LL energy once with fitted torsions deleted; each solver step adds analytical torsions (no GeomOpt per iteration). As of 1.5 the fixed-geom path uses ``lsq_linear`` (not COBYLA). Legacy GeomOpt-per-iter path: `FFPOPT_DIHED_FIT_REOPT=1`.
- **`bare_potential_energy`** - reuse opt energy by subtracting restraint penalties analytically (drops post-opt single-point SCF on wavefront nodes).

#### Scission / RDKit / graphs

- **Vectorized `screen_candidate`** (numpy rotate + heavy-heavy clash); skip unused cap builds on the hot path.
- **Hoisted fragmentation topology** + caches for rotatable bonds / ring edges on `Ligand`.
- **Cached RDKit mol** and process-wide compiled SMARTS for central bonds.
- **Faster `RotateMask` / graph bipartition** (`ComponentBeyondBond`) without defensive `deepcopy(GetGraph())` on every call.
- **SANDER wrappers** reuse a scratch ASE `Atoms` instead of rebuilding every `calculate()`.

#### RESP / Gaussian I/O

- Faster **ESP / EOF** parsing (`ReadGauEsp` substring gates + token parse; compiled regex; `GaussianReader.check_complete` reads a short tail for `Normal termination`).

#### Misc I/O

- **Compact temp JSON** (`ListOfStruct.save(..., indent=None)` by default).

### 3. Integration (workflows, CLI, reliability)

Wiring the merged packages into ligandparam recipes/CLI and making HPC runs finish reliably.

#### User-facing integration

- **`lig-dihed-correct`** / **`StageDihedTwistCorrection`** - fragmented twist workflow: scission -> per-fragment ffpopt twist -> merge DIHE into parent frcmod (lib unchanged).
- **Recipe opt-in** via `dihed_options` (`dihed_correct`, `dihed_model`, `dihed_delta`, `dihed_nprim`, fragment config / SMARTS, etc.).
- **`lig-scission`** CLI for fragmentation-only runs.
- **Bond indexing** clarified (0-based in ffpopt workflows; 1-based scission torsions converted at the boundary).
- **Workdir-safe workflows** - absolute paths + `subprocess(..., cwd=fragment_dir)`; no `os.chdir` in fragmented twist.
- Docs / README / extras for tblite, DeepMD, and dihedral install paths.

#### Reliability & HPC fixes (post-merge hardening)

- Run fit scripts and ffpopt bin tools with **`sys.executable`** (bare `python3` on CHPC often lacks ParmEd).
- **In-process apply** of GenDihedFit `itNN.py` (`runpy`) so apply uses the same env and logs progress to the parent `.out`.
- **Absolute paths in `*.fit.json`** (script, frcmod, HL/LL profiles) so GenDihedFit cannot write to the wrong cwd.
- **Post-step file checks** (`_require_files`) after GenDihedFit and apply+prepare.
- **WriteParmedScript progress prints** that do not embed mask f-strings (avoids `SyntaxError` in generated `it01.py`); regenerate-safe logging.
- Graph / mask helpers and RDKit mol construction consolidated for maintainability across scission writers/torsions.

---

## [1.0.1] and earlier

Prior releases focused on core RESP / FreeLigand recipes, orientation protocols (`so3_n28` vs legacy Euler), documentation, and packaging. See git history before the ffpopt/scission merge commits (`23e217b`, `f0f6c97`) for details.

---

[Unreleased]: https://github.com/piskulichz/ligandparam/compare/v1.6.1...HEAD
[1.6.1]: https://github.com/piskulichz/ligandparam/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/piskulichz/ligandparam/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/piskulichz/ligandparam/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/piskulichz/ligandparam/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/piskulichz/ligandparam/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/piskulichz/ligandparam/compare/v1.0.1...v1.4.0
