---
name: ffpopt-user
description: Use when running ffpopt CLI scripts or invoking the ffpopt Python API as a consumer. Covers entry-point scripts, common --model / --inp / --out options, JSON I/O contract, example workflows under examples/.
---

# User — ffpopt

## Scope
What you need to *use* ffpopt without modifying it: installable CLI scripts under `src/python/bin/`, the public Python API exposed by `ffpopt` (mainly `ffpopt.Struct`, `ffpopt.GeomOpt`, `ffpopt.Constraints`, `ffpopt.Restraints`, `ffpopt.Dihedrals`, `ffpopt.WaveFront`, `ffpopt.confsearch`, `ffpopt.cpefit`, `ffpopt.scosmo`, `ffpopt.ase.calculator`), and the JSON-based input/output contract that ties scripts together. Does *not* cover code conventions (see TechLead), build/install (see Operator), or external model SDK quirks (see Integrator).

## Canonical facts

- **Package name:** `ffpopt`. Version `1.1.0` (see `pyproject.toml:10`). Python `>=3.8`, but CMake requires Python `3.12` (`CMakeLists.txt:62`); conda env pins `python=3.12` (`environment.yml:8`).
- **Installed CLI scripts** (defined `pyproject.toml:101-118`, dispatched by `src/python/lib/ffpopt/scripts.py`):
  - Workflow drivers: `ffpopt-PrepareInput.py`, `ffpopt-ConfSearch.py`, `ffpopt-Optimize.py`, `ffpopt-DihedScan.py`, `ffpopt-DihedWavefront.py`, `ffpopt-DihedTwistWorkflow.py`, `ffpopt-GenDihedFit.py`, `ffpopt-RespFit.py`, `ffpopt-CpeFit.py`.
  - Conversion: `ffpopt-Json2Crds.py`, `ffpopt-JsonJoin.py`, `ffpopt-JsonSplit.py`, `ffpopt-xyz2mol2.py`, `ffpopt-WavefrontToDP.py`.
  - Visualization: `ffpopt-DihedTwistAnimate.py`, `ffpopt-WavefrontAnimate.py`.
  - Native helper: `ffpopt-respf` — Fortran 90 binary built from `src/resp/resp.f` (`src/resp/CMakeLists.txt`), used by `ffpopt-RespFit.py --respf`.
- **Standard CLI options** are injected by `ffpopt.Options.AddStandardOptions(parser)` (`src/python/lib/ffpopt/Options.py:110-124`), which adds model options (`--model`, `--mfile`, `--cpu`, `--psi4-memory`, `--psi4-num-threads`) and geometry-optimization options (`--no-opt`, `--geometric-opt`, `--ase-opt-tol`, `--geometric-maxiter`, `--geometric-coordsys`, `--geometric-converge`, `--geometric-enforce`, `--geometric-ini`).
- **`--model` default is `sander`**; full enumerated list (sander, dftb2, dftb3, qdpi2, xtb, mace*, aimnet2*, ani1x/2x/1ccx/1xbb, pm6ml, fennix-bio1m/s, orb-v3-*, AM1/MNDO/MNDOD/PM3/PM6*/PM7*/RM1, OMOL25-ESEN-*) is dispatched in `src/python/lib/ffpopt/ase/calculator.py:86-291`. A `theory/basis` string (containing `/`) is dispatched to Psi4 (`Options.ModelIsPsi4` at `Options.py:321-357`).
- **Standard I/O is JSON.** Scripts read `--inp <name>.json` via `ListOfStruct.from_file(...)` (e.g., `bin/ffpopt-Optimize.py:54`) and write `--out <name>.json` via `outs.save(...)`. The legacy XYZ flow is mostly commented out; current pipelines convert at the end with `ffpopt-Json2Crds.py`.
- **In-process Python API (no subprocess)** — for callers driving ffpopt from another Python package (e.g. FragmentMol → ffpopt in a shared conda env):
  - `from ffpopt.WaveFront import run_dihed_wavefront` — same options as the CLI `ffpopt-DihedWavefront.py` but as kwargs. Required: `inp`, `out`, `dihed`. Optional standard-options like `model="qdpi2"`, `geometric_opt=True`, `ase_opt_tol=0.01` pass through unchanged. Returns `{'wf_run','angles','energies','energies_noshift','structures'}`.
  - `from ffpopt.Workflows import run_dihed_twist_workflow` — wavefront-only re-implementation of the twist workflow. Required: `inp`, `bond=["a,b","c,d",...]`. Phases: high-level scan → reference sander scan → iterative fit-and-rescan. `skip_existing=True` (default) makes it re-runnable.
  - `from ffpopt.Workflows import run_fragmented_dihed_twist_workflow` — fragment → twist-per-fragment → recombine workflow. Required: `mol2`, `lib`, `frcmod` (parent ligand triplet, same shape `scission`/FragmentMol expects). Drives `scission.fragment_ligand` to produce reduced fragments under `out_dir/fragment_N/`, runs `run_dihed_twist_workflow` inside each fragment dir (`PrepareInput` → twist), then calls `scission.merge.merge_fragment_frcmods` to splice the highest-numbered `itXX.frcmod` from each fragment into a unified parent frcmod at `merged_frcmod`. Twist kwargs pass through unchanged. Requires the `scission` package importable in the env.
  - **All three require an `if __name__ == "__main__":` guard in the caller's entry script** — the wavefront uses `multiprocessing.set_start_method('spawn', force=True)`, and an unguarded top-level call gets caught by an explicit `__mp_main__` check that raises with a clear fix-it message instead of the cryptic multiprocessing traceback.
  - Unknown kwargs (typos, extra options) raise `TypeError("... got unexpected keyword argument(s): [...]")`.
  - The existing `bin/ffpopt-DihedTwistWorkflow.py` (bash-script generator) is unchanged — use it when you want a checkpointed bash pipeline; use `run_dihed_twist_workflow` when you want in-process execution.
- **Input file types accepted by `ffpopt-PrepareInput.py`**: amber `parm7+rst7` *or* `mol2`. `mol2` is required for QM/ML model paths that don't go through sander; only `parm7+rst7` supports `--model=sander` and torsion-parameter optimization (README §"COMMON OPTIONS").
- **mol2 partial-charge constraint:** the sum of partial charges must equal the net molecular charge. Use `ffpopt-xyz2mol2.py --charge=N -i in.xyz -o out.mol2` to seed a mol2 from xyz (README:222-231).
- **Examples** (each with a runnable `run.sh`): `examples/confsearch/`, `examples/dihedscan/`, `examples/dihedtwistfit/`, `examples/geometric/`, `examples/optimize/`, `examples/resp/`.
- **GitLab Pages user docs:** https://ffpopt-b083ab.gitlab.io/ (README:1).
- **Documentation page for JSON input file format** of `ffpopt-GenDihedFit.py`: README §"JSON input file format for ffpopt-GenDihedFit.py" (README:674-815). Three top-level keys: `params`, `output` (frcmod path), `systems` (list with `parm`/`crd`/`output`/`params`/`profiles`).

## Conventions

- Drive multi-step workflows from a `run.sh` in a working directory. Start with `ffpopt-PrepareInput.py` to produce `start.json`, then chain scripts that consume and emit `.json`.
- Pass coordinates and structure metadata as `.json` between scripts. Only convert to xyz/mol2/rst7 at the end (`ffpopt-Json2Crds.py`).
- For QM/ML model runs, prepare input from `mol2` rather than `parm7+rst7`; reserve `parm7+rst7` for sander runs and for any workflow that touches amber torsion parameters.
- Set the `QUICK_BASIS` env var when using QUICK from AmberTools (README:103-106; `modulefiles/ffpopt:20`).
- When using mopac-backed methods, ensure `mopac` is available; mopac files are written with randomized prefixes and cleaned up after the call (commit `e55517c`, `f44fb2a`).
- To force CPU evaluation of ML models, pass `--cpu` (sets `JAX_PLATFORMS=cpu` and `CUDA_VISIBLE_DEVICES=-1`; `Options.py:103-106`).
- The `--mfile` option overrides the packaged model checkpoint for a given `--model` (see `pkgdata/` mapping in `ase/calculator.py`).

## Anti-patterns

- Do not pass mol2 input to `--model=sander` or to amber-torsion fitting workflows — sander and `parmed.tools` require `parm7+rst7`.
- Do not write a mol2 whose partial charges do not sum to the net charge; QM dispatch trusts `sum(charges)` for the wave function charge.
- Do not run pytorch and tensorflow models in the same conda env — `pyproject.toml` exposes them as mutually incompatible dependency groups (`[dependency-groups] pytorch` vs `tensorflow`, README §Installation step 4).
- Do not install `--group fairchem` into the same env as `--group pytorch` or `--group tensorflow`; the three groups install incompatible stacks (README:36-46).
- Do not pin `numpy>=2` in any env used with parmed/ambertools — parmed requires `numpy<2` (README:80-87, `pyproject.toml:31`).
- Do not assume `--model` is case-sensitive; `GenCalculator` normalizes to uppercase before dispatch (`ase/calculator.py:66`). But the public README and scripts spell models lowercased — match the README spelling in CLI invocations.
- Do not edit `__pycache__/` artifacts that appear under `src/python/lib/ffpopt/`; they are not authoritative.

## Pointers

- Top-level user docs: `README.md` (long-form; CLI flags, models, JSON format).
- Sphinx user docs: `docs/UserDocs/GettingStarted.rst`, `docs/UserDocs/Examples.rst`, `docs/UserDocs/Examples/WavefrontExample/tutorial.rst`.
- Public Python entrypoints: `src/python/lib/ffpopt/__init__.py:3-10` (re-exports `ase`, `constants`, `confsearch`, `cpefit`, `scosmo`).
- Kwargs API entry points: `src/python/lib/ffpopt/WaveFront.py` (`run_dihed_wavefront`), `src/python/lib/ffpopt/Workflows.py` (`run_dihed_twist_workflow`).
- CLI scripts: `src/python/bin/ffpopt-*.py`.
- Script entry-point dispatcher: `src/python/lib/ffpopt/scripts.py`.
- Standard option definitions: `src/python/lib/ffpopt/Options.py`.
- Examples directory: `examples/` (one subdir per workflow, each with `run.sh`).
- Module example for HPC clusters: `modulefiles/ffpopt`.

## Gaps

- No `LICENSE` file is committed (the `pyproject.toml:17` `license` field is commented out). Author/distribution license is unclear.
- No tagged releases or `CHANGELOG.md`. `pyproject.toml` is the only version source.
- `ffpopt-DihedWavefront.py`, `ffpopt-CpeFit.py`, `ffpopt-WavefrontToDP.py`, and the `*Animate.py` tools are not yet documented in `README.md`; verify behaviour from their argparse before recommending flags.
- `--model=dpmlp` is mentioned in `--model` help text (`Options.py:81`) and `Options.ModelIsPsi4`, but the dispatch in `GenCalculator` does not have an explicit `"DPMLP"` branch; treat as un-verified.

---
Last reviewed: 2026-05-20 (added run_dihed_wavefront + run_dihed_twist_workflow Python APIs)
Owner: piskuliche
