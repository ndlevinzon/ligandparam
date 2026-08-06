---
name: ffpopt-architect
description: Use when you need to understand ffpopt's structure before changing it — covers the Python package layout, the JSON-centric data flow between scripts, the ASE calculator dispatch, and the CMake/scikit-build build pipeline that installs packaged ML model weights.
---

# Architect — ffpopt

## Scope
ffpopt is a Python library + CLI suite for *force-field parameter optimization*. The package wraps a constellation of energy/force back-ends behind an ASE `Calculator` interface, drives geometry optimizations (BFGS via ASE *or* geomeTRIC), runs relaxed dihedral scans + a "wavefront" algorithm, and fits amber torsion / electrostatic-potential / RESP parameters. This skill covers component boundaries, the JSON data contract that flows between scripts, and the build pipeline. Does not cover internal coding style (TechLead) or external SDK quirks (Integrator).

## Canonical facts

- **Stack:** Python 3.12-targeted package built with `scikit-build-core` over CMake (`pyproject.toml:1-3`); one Fortran 90 executable (`src/resp/resp.f`, installed as `ffpopt-respf`); CMake fetches/installs ML model checkpoints into `pkgdata/`. No JavaScript, no service layer.
- **Package layout** (`src/python/lib/ffpopt/`):
  - Top-level modules: `Struct.py`, `GeomOpt.py`, `Geometry.py`, `Constraints.py`, `Restraints.py`, `Reader.py`, `AmberParm.py`, `Dihedrals.py`, `Options.py`, `WaveFront.py`, `Workflows.py`, `ASECalc.py`, `scripts.py`. Module file names are **PascalCase** (`Workflows.py`, not `workflows.py`); match the existing case when adding new modules.
  - Subpackages: `ase/` (the calculator dispatch — `calculator.py`, `fennolase.py`, `mopac.py`), `cpefit/` (charge fitting: `Molecule.py`, `Conformer.py`, `MoleculeCollection.py`, `GaussianEsp.py`, `GaussianOutput.py`, `Psi4Esp.py`, `FixCharges.py`, `QuickEsp.py`, `AbInitioOptions.py`), `confsearch/` (`ConfSearch.py`), `scosmo/` (Smooth COSMO solvation: `CosmoAtom.py`, `CosmoElement.py`, `CosmoSurface.py`, `Lebedev.py`, `SwitchFcn.py`, `UFFRadii.py`), `constants/` (`PeriodicTable.py`, `Conversions.py`), `pkgdata/` (bundled model files: `qdpi/qdpi-2.0.pb`, `files/geometric_log.ini`, plus installed `fennix/`, `mace-off/`, `pm6ml/`).
- **CLI surface** lives under `src/python/bin/ffpopt-*.py` (`pyproject.toml:81` registers it as the `ffpopt/bin` wheel subpackage). Each script is `if __name__ == "__main__":` argparse, imports from `ffpopt.*`, calls into the library. `pyproject.toml:101-118` registers shim entry points that re-launch the matching `bin/ffpopt-*.py` via `subprocess.run` (`scripts.py:_run_bin_script`).
- **Python API surface (kwargs entry points)** — for callers who want to drive ffpopt in-process without spawning subprocesses:
  - `ffpopt.WaveFront.run_dihed_wavefront(**kwargs)` — same options as the CLI `ffpopt-DihedWavefront.py`, plus an `**standard_kwargs` catch-all for `AddStandardOptions` fields (`model`, `geometric_opt`, ...). Returns `{'wf_run','angles','energies','energies_noshift','structures'}`.
  - `ffpopt.Workflows.run_dihed_twist_workflow(**kwargs)` — wavefront-only twist workflow. Mirrors the CLI `ffpopt-DihedTwistWorkflow.py` phase structure (high-level scan → reference sander scan → iterative fit+rescan) but runs each scan in-process via `run_dihed_wavefront`. Fit (`ffpopt-GenDihedFit.py`) and PrepareInput steps still shell out — marked `# FUTURE: replace subprocess with API` for follow-up.
  - Both functions require the caller's entry to be inside `if __name__ == "__main__":` (the wavefront drives a per-`calculate()` `multiprocessing.get_context('spawn').Pool` **calculation queue** — nodes are scheduled with `apply_async` and harvested as they finish, so a slow node no longer blocks the rest of its "level"; levels are now just a post-processing label). A `__mp_main__` caller-frame check in `run_dihed_wavefront` raises a clear `RuntimeError` if a script forgot the guard.
  - The bin scripts are thin argparse wrappers that call these functions via `func(**vars(args))`. Standard options have one source of truth (`Options.AddStandardOptions`); the library uses a throwaway parser internally to derive defaults for unspecified standard kwargs.
- **Data contract:** the primary in-memory object is `ffpopt.Struct.Struct` (per-conformer: positions, names/types/elements, charges, bonds, restraints, constraints, parm path) and `ListOfStruct`. Persistence is JSON: `ListOfStruct.from_file(path)` / `los.save(path)` — see `Struct.py:28-…`. The chain `ffpopt-PrepareInput.py → ffpopt-ConfSearch.py → ffpopt-Optimize.py → ffpopt-DihedScan.py → ffpopt-GenDihedFit.py / ffpopt-RespFit.py` all read/write `.json` files. The previous interchange format was xyz/extxyz; commit `71f2f6a` ("changed underlying data structures. major change to all library components and scripts") flipped the contract to JSON and is the boundary between the old and new pipelines.
- **Calculator dispatch:** `ffpopt.ase.calculator.GenCalculator(mode, charge, spin, parm, crd, **kwargs)` (`ase/calculator.py:58-291`) returns an ASE `Calculator` subclass. Modes (uppercased): SANDER → `SanderCalculator`; DFTB2/DFTB3/AM1D → `SanderSQMCalculator`; QDPI2 → `QDpi2Calculator` over `DPModel`; XTB → `tblite.ase.TBLite`; MACE-* → `mace.calculators.MACECalculator`; PYSCFNEO → either `PySCF_DFT_Calculator` or `pyscf.neo.Pyscf_NEO`; AIMNET* → `aimnet.calculators.AIMNet2ASE`; OMOL25-* → `fairchem.core.FAIRChemCalculator`; ANI1CCX/ANI1X/ANI2X → `torchani.models.*`; FENNIX* → `FENNIXCalculator`; mopac family (`AM1, MNDO, MNDOD, PM3, PM6*, PM7*, RM1`) → `.ase.mopac.MOPAC`; PM6ML → `PM6MLCalculator`; `theory/basis` (contains `/`) → `ase.calculators.psi4.Psi4`; ORB-* → `WrappedORBCalculator`.
- **Geometry optimization** is in `ffpopt.GeomOpt`. Two paths: ASE BFGS (`GeomOpt_ASE`, default ASE tol `--ase-opt-tol=0.01`) and geomeTRIC (`--geometric-opt`, configured via `--geometric-coordsys` (default `tric`), `--geometric-converge` (default `'set GAU'`), `--geometric-maxiter` (default `500`), `--geometric-enforce` (default `0.1`); `Options.py:39-55`). The geomeTRIC path runs as a subprocess (`Options.argparse2geometric` builds `["geometric-optimize", "--engine", "ase", "--ase-class", "ffpopt.Struct.RestCalculator", ...]` at `Options.py:459-468`).
- **Constraints / restraints model:** `ConstraintList`/`Constraint` (`Constraints.py`) for hard internal-coordinate constraints (bonds/angles/dihedrals, exported to both ASE `FixInternals` and geomeTRIC); `RestraintList`/`Restraint` (`Restraints.py`) for soft restraints (bond/angle/dihed/r12/rms/twist). `RestrainedCalculator` (`Struct.RestCalculator` / `ase/calculator.py`) wraps a base calculator and adds restraint energies.
- **Build pipeline** (`CMakeLists.txt`):
  1. CMake variables `USE_FENNOL`, `USE_MACE`, `USE_PM6ML`, `USE_DFTB`, `ACADEMIC` gate optional model downloads (`CMakeLists.txt:22-57`).
  2. `if(NOT ACADEMIC)` → `USE_FENNOL/MACE/PM6ML` are forced FALSE (industry users can't use these without permission; `CMakeLists.txt:53-57`).
  3. `FetchContent_Declare` / `ExternalProject_Add` pull model weights from GitHub (mace-off, mopac-ml, dftbparams/mio, dftbparams/3ob) at configure time.
  4. When invoked under scikit-build (`SKBUILD_SCRIPTS_DIR` defined), CMake compiles `ffpopt-respf` and installs downloaded models under `ffpopt/pkgdata/{fennix,mace-off,pm6ml}`; DFTB params land under `$AMBERHOME/dat/slko/{mio-1-1,3ob-3-1}`.
  5. Wheel packaging: `pyproject.toml:81` maps `ffpopt` → `src/python/lib/ffpopt` and `ffpopt/bin` → `src/python/bin`.
- **Dependency groups** (`pyproject.toml:37-61`): `pytorch`, `tensorflow`, `fairchem` — mutually incompatible, install into separate conda envs (README §Installation step 4).
- **CI:** `.gitlab-ci.yml` (single `pages` job) builds Sphinx docs and publishes to GitLab Pages on the default branch. There is no test or lint stage.
- **Hosting:** GitLab (`pyproject.toml:72`, `gitlab.com/RutgersLBSR/ffpopt`); GitLab Pages site at `https://ffpopt-b083ab.gitlab.io/`.

## Conventions

- One `argparse.ArgumentParser` per script in `src/python/bin/ffpopt-*.py`. The script always calls `AddStandardOptions(parser)` to inherit the model/geomeTRIC options. Imports are scoped inside `if __name__ == "__main__":` to make CLI startup snappy and to avoid importing heavy ML stacks at module-import time.
- The library follows a *"load lazily"* discipline: heavy imports (`parmed`, `ase`, `torch`, `torchani`, framework-specific) are deferred into the function body that needs them. See `GeomOpt.GeomOpt_ASE` (`GeomOpt.py:23-…`), `Struct.from_parmed`, `ase/calculator.py:GenCalculator.__init__`. Maintain this; do not hoist heavy imports to module scope.
- File I/O for structures goes through `Struct.from_file` / `ListOfStruct.from_file` / `los.save`. New flows should not invent ad-hoc xyz/json formats.
- Energies in JSON are stored in eV (ASE convention). Use `ffpopt.constants` (`AU_PER_KCAL_PER_MOL`, `AU_PER_ELECTRON_VOLT`, `AU_PER_ANGSTROM`) for conversions; e.g., `bin/ffpopt-DihedScan.py:95-101`.
- Model dispatch is by *uppercased substring match* on `mode` (`ase/calculator.py:66 + 86-291`). New models are added as additional `elif` branches in `GenCalculator.__init__` and a corresponding entry in `README.md §MODELS`.
- Bundled model files go under `src/python/lib/ffpopt/pkgdata/<family>/` and are resolved at runtime via `importlib.resources.files("ffpopt") / "pkgdata/..."`.

## Anti-patterns

- Do not import a model SDK (`torch`, `torchani`, `mace`, `aimnet`, `fairchem`, `fennol`, `psi4`, `tblite`, `pyscf`) at module scope — the dispatcher must remain importable without any specific group installed.
- Do not pin or special-case `numpy>=2` anywhere the parmed/ambertools stack is loaded; the install matrix assumes `numpy<2` in the main runtime (`pyproject.toml:31`).
- Do not add new top-level interchange formats. Extend `Struct` keys; convert at the boundary via `ffpopt-Json2Crds.py`.
- Do not write a new CLI in `src/python/bin/` without:
  1. calling `AddStandardOptions(parser)`,
  2. reading via `ListOfStruct.from_file`,
  3. writing via `out.save(args.out)`,
  4. registering the entry-point shim in `pyproject.toml:[project.scripts]` and `scripts.py`.
- Do not couple library modules to a specific calculator family — go through `GenCalculator` so users can swap `--model`.
- Do not split the build into multiple CMake projects; the single root `CMakeLists.txt` + scikit-build flow is the supported install path.

## Pointers

- Build / packaging: `pyproject.toml`, `CMakeLists.txt`, `MANIFEST.in`, `src/python/CMakeLists.txt`, `src/resp/CMakeLists.txt`.
- Calculator dispatch (the single most load-bearing file): `src/python/lib/ffpopt/ase/calculator.py`.
- Data model: `src/python/lib/ffpopt/Struct.py`.
- Geometry optimization: `src/python/lib/ffpopt/GeomOpt.py` and the geomeTRIC adapter in `src/python/lib/ffpopt/Options.py:394-468`.
- Standard CLI options: `src/python/lib/ffpopt/Options.py`.
- Wavefront algorithm + `run_dihed_wavefront` Python API: `src/python/lib/ffpopt/WaveFront.py`.
- Twist-workflow Python API (`run_dihed_twist_workflow`): `src/python/lib/ffpopt/Workflows.py`.
- Amber parameter manipulation: `src/python/lib/ffpopt/AmberParm.py`, `src/python/lib/ffpopt/Dihedrals.py`.
- Conformer search: `src/python/lib/ffpopt/confsearch/ConfSearch.py`.
- Charge fitting (RESP, CPE): `src/python/lib/ffpopt/cpefit/`.
- Smooth COSMO solvation: `src/python/lib/ffpopt/scosmo/`.
- Docs build: `docs/Makefile`, `docs/conf.py`.

## Gaps

- The `WrappedORBCalculator`, `QDpi2Calculator`, `SanderCalculator`, `SanderSQMCalculator`, `PM6MLCalculator`, `DPModel`, `PySCF_DFT_Calculator` classes are referenced from `ase/calculator.py` but their definitions (or imports) are not visible in the first 300 lines of that file; check the remainder of `ase/calculator.py` and `ase/fennolase.py` / `ase/mopac.py` before relying on their exact signatures.
- There is no architectural diagram or design doc committed; the layering above is reconstructed from the source. Decisions under `decisions/` is empty (only `README.md` + `_template.md` exist).
- No tests directory and no test runner config (see Inspector); claims about expected behavior cannot be auto-verified.
- `ASECalc.py` is a thin module — verify whether it is dead code or used by external integrators.

---
Last reviewed: 2026-05-28 (wavefront level-barrier → spawn-context calculation queue)
Owner: piskuliche
