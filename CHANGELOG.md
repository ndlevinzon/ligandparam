# Changelog

All notable changes to **ligandparam** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

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
- **GenDihedFit missing imports after Dihedrals split** - ``shape_match_delta`` in ``IsolatedLinearSolve`` and ``WriteParmedScript`` in ``SystemType.write_output``.
- **Spawn pickle of live sander calculators** - ``ListOfStruct.__getstate__`` drops ``_ffpopt_calc_cache`` so nested wavefront pools do not serialize ``sander.pysander.InputOptions``.

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

[Unreleased]: https://github.com/piskulichz/ligandparam/compare/v1.5.1...HEAD
[1.5.1]: https://github.com/piskulichz/ligandparam/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/piskulichz/ligandparam/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/piskulichz/ligandparam/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/piskulichz/ligandparam/compare/v1.0.1...v1.4.0
