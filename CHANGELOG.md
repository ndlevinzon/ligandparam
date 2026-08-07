# Changelog

All notable changes to **ligandparam** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.4.1] — 2026-08-07

Parallelism and ConfSearch follow-ons after the v1.4.0 ffpopt/scission merge.

### Performance

- **ConfSearch RMS matrix** — for ensembles at/above `FFPOPT_CONFSEARCH_RMS_FAST_N` (default **50**), align once to the first conformer and use vectorized heavy-atom RMS for Butina clustering instead of per-pair `GetBestRMS`. Set the env var to `0` to force the legacy path.
- **Parallel multi-ESP rotations** (`StageGaussianRotation`) — process pool over orientation `.com` jobs; `nproc` is a total core budget (`n_workers × %NProc ≤ nproc`). Per-job bash scripts and `GAUSS_SCRDIR`.
- **Parallel per-bond scans** — `run_dihed_twist_workflow` pools HL / reference / iteration wavefront scans across bonds (`n_bond_workers × wf_nproc`).
- **Parallel RespFit / CpeFit conformer ESP** — independent per-conformer ab initio ESP (and cosmo/harmonic sets) in a process pool; charge / CPE fit stays serial (O(natoms) assembly).
- **Parallel scission screen / writes** — pool over torsions for candidate screening and over fragments for `parmchk2`/`tleap`; `FragmentConfig.nproc` / CLI `--nproc` / fragmented-workflow `nproc`.
- **Parallel fragments** — `run_fragmented_dihed_twist_workflow` splits `-n` / `nproc` across a non-daemon fragment pool and nested wavefront workers.
- **Vectorized `cap_site_scan_margin`** — NumPy rotate + margin vs retained heavies (same pattern as `screen_candidate`).
- **Cached retained APSP** — `retained_distance_map` keyed by `frozenset(retained)` for shell-sibling reuse.
- **Unique domain-shell enum** — skip left/right depths that do not change the domain set; build each fragment once.
- **`FindMinCycles` via cycle basis** — NetworkX `cycle_basis` instead of DFS min-path search (sugar puckers).
- **Cheaper wavefront / dihedral clones** — `Struct.clone_geometry` + `ListOfStruct.from_structs_shared` replace full deepcopies for coord updates and parm-path overrides; `MultiDihedFcn` uses shallow prim copies.

### Reliability

- Gaussian `call` uses unique submit scripts and env copies so concurrent jobs do not collide.
- Windows-safe `ligandparam.utils` libc loading (no `ctypes.CDLL(None)` crash on import).
- **Non-daemon fragment/bond pools** — `_make_nondaemon_spawn_pool` no longer subclasses `ctx.Pool` (a factory method on Python 3.8+ / 3.14); uses `multiprocessing.pool.Pool` with a module-level non-daemon spawn `Process` so nested wavefront pools work on CHPC.
- **Robust GeomOpt recovery** — on geomeTRIC `GeomOptNotConvergedError`, restart from the last `_optim.xyz` frame with a ladder: `GAU_LOOSE` + more iterations, alternate `dlc`/`hdlc` coordsys, then soft `converge maxiter`. ASE fallback tries BFGS → LBFGS → FIRE and soft-accepts near-converged `fmax`. Disable ladder with `FFPOPT_GEOMOPT_ROBUST=0`; tune soft ASE with `FFPOPT_ASE_LOOSE_FMAX`.
- **`--geometric-opt` help** — corrected to match code (flag prefers geomeTRIC; default ASE-first).
- **WaveFront soft-opt gates** — soft-maxiter / ASE `*-soft` geometries may fill the profile but cannot displace hard-converged minima and do not spawn neighbors; tags survive slim IPC.
- **WaveFront failure reporting** — precheck exceptions are `precheck_error` (not all `clash_precheck`); summary lists failed / soft-accepted counts instead of always saying “successfully.”
- **Constraints f-string** — nested quotes fixed so `ffpopt.Constraints` imports on Python 3.10–3.11.
- **Noisier fallbacks** — calculator device / SANDER load / ConfSearch mol parse / Gaussian I/O report the underlying exception instead of bare `except:`.
- **Charge normalization** — safe for nonzero `net_charge`, tiny diffs, and `|diff| > natoms*precision` (residual on largest-|q| atom); asserts final sum.
- **Docs / packaging** — READMEs clarify `src/` is runtime SoT vs optional `*-main/` trees; `numpy<2` documented; optional `ligandparam[ml-potentials]` for MACE / TorchANI.
- **GenDihedFit HL/LL mismatch** — when wavefront HL and LL JSON scans differ in length (failed / soft-opt holes), align on shared `dXXX` scan angles instead of aborting; raise only if fewer than 3 common points remain.
- **Fragment-scan completion logs** — per-scan / per-fragment messages no longer say bare “done/completed”; parent logs `[frag-twist] all N fragment twist job(s) finished` only after the fragment pool joins, and CLI `Done` only after merge.
- **Fragment live status board** — parallel fragment twist runs write an ASCII board (`FRAG_STATUS.txt` + parent log) with per-fragment name / status / stage / detail; wavefront spam goes to `<frag>/frag-twist.log` instead of interleaving on the console.

---

## [1.4.0] — 2026-08-06

### 1. Package merge — ffpopt + scission (highest priority)

Vendored and installed the York-lab **ffpopt** and **scission** stacks inside ligandparam so dihedral fragmentation and twist fitting no longer depend on a separate out-of-tree install.

- **ffpopt** under `src/ffpopt/` — wavefront dihedral scans, GenDihedFit, GeomOpt (ASE / geomeTRIC), calculators (SANDER, XTB/tblite, ML models), and CLI entry points (`ffpopt-*.py`).
- **scission** under `src/scission/` — ligand fragmentation, rotatable-bond / torsion discovery, fragment parm/lib/frcmod prep, and merge of fragment DIHE terms back into a parent frcmod.
- **Packaging** — `pyproject.toml` includes `ligandparam*`, `ffpopt*`, and `scission*`; optional extras `[dihed]` (ndfes, geometric) and `[tblite]` (GFN2-xTB); console scripts `lig-dihed-correct`, `lig-scission`, `scission`, and the ffpopt bin wrappers.
- **Vendor trees** — `ffpopt-main/` / `scission-main/` kept for upstream reference; `.gitignore` clarified so the installed `src/` trees remain the source of truth.
- **Version bump** to **1.4.0** with docs pages for ffpopt, scission, dihedral correction stages, and CLI.

### 2. Performance enhancements

Targeted wall-time and I/O reductions for HL wavefront scans (especially `--model xtb`), GenDihedFit, scission screening, and related RESP / Gaussian paths.

#### Geometry optimization & XTB

- **In-process geomeTRIC** with a persistent calculator cache on `ListOfStruct` (avoids per-opt subprocess bootstrap and model reload). Legacy CLI path via `FFPOPT_GEOMETRIC_SUBPROCESS=1`.
- **Plain XYZ for geomeTRIC** (`write_plain_xyz`) — ASE’s default `Atoms.write()` emitted extended XYZ with charge columns that broke `Molecule` parsing; fixed so in-process opts do not fall back to ASE spuriously.
- **XTB / tblite** — robust SCF defaults, env knobs (`FFPOPT_XTB_MAX_ITER`, `FFPOPT_XTB_ETEMP`, `FFPOPT_XTB_MIXER_DAMPING`, `FFPOPT_XTB_GUESS`), SCF retry ladder, and calculator reuse.
- **geomeTRIC robustness** — Cartesian IC recovery notice handling, stall watchdog / progress detection, Brent “Not bracketed” recovery; quieter GeomOpt ASE↔geomeTRIC fallbacks (one-line stderr; full traceback only with `FFPOPT_GEOMOPT_TRACEBACK=1`).

#### Wavefront & IPC

- **Slim wavefront IPC** — pool/MPI workers share `los` once via initializer / bcast; jobs carry angles/RCs + coords; results merge without shipping full calculators.
- **Lighter checkpoints** — clear live calc / `_ffpopt_calc_cache`, drop redundant forces on completed nodes, `pickle.HIGHEST_PROTOCOL`.
- **Vectorized clash precheck** (`has_nonbonded_clash`) replaces O(n²) ASE `get_distance` loops; bonded pairs handled correctly via a bond mask.
- **Absolute paths beside `out`** for 1D/ND checkpoints, plots, and dat/pkl so absolute `out=` does not write into the launch cwd.

#### GenDihedFit & torsions

- **Fixed-geometry NL cache** — base LL energy once with fitted torsions deleted; each COBYLA step adds analytical torsions (no GeomOpt per iteration). Legacy path: `FFPOPT_DIHED_FIT_REOPT=1`.
- **`bare_potential_energy`** — reuse opt energy by subtracting restraint penalties analytically (drops post-opt single-point SCF on wavefront nodes).

#### Scission / RDKit / graphs

- **Vectorized `screen_candidate`** (numpy rotate + heavy–heavy clash); skip unused cap builds on the hot path.
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

- **`lig-dihed-correct`** / **`StageDihedTwistCorrection`** — fragmented twist workflow: scission → per-fragment ffpopt twist → merge DIHE into parent frcmod (lib unchanged).
- **Recipe opt-in** via `dihed_options` (`dihed_correct`, `dihed_model`, `dihed_delta`, `dihed_nprim`, fragment config / SMARTS, etc.).
- **`lig-scission`** CLI for fragmentation-only runs.
- **Bond indexing** clarified (0-based in ffpopt workflows; 1-based scission torsions converted at the boundary).
- **Workdir-safe workflows** — absolute paths + `subprocess(..., cwd=fragment_dir)`; no `os.chdir` in fragmented twist.
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

[1.4.1]: https://github.com/piskulichz/ligandparam/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/piskulichz/ligandparam/compare/v1.0.1...v1.4.0
