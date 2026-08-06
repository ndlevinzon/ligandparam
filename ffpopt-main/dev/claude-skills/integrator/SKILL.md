---
name: ffpopt-integrator
description: Use when touching an external SDK or model in ffpopt — covers the ASE Calculator dispatch in ase/calculator.py, the parmed/ambertools binding, geomeTRIC subprocess invocation, the per-model SDK quirks (MACE, AIMNet2, ANI, FairChem/OMol25, fennol, mopac, psi4, tblite, orb-models), and the build-time model downloads.
---

# Integrator — ffpopt

## Scope
Every external dependency that ffpopt brokers behind a Python API and the quirks that survive in the dispatch code today. The center of gravity is `src/python/lib/ffpopt/ase/calculator.py:GenCalculator.__init__` (`ase/calculator.py:58-291`), which selects a back-end based on `--model`. This skill captures per-integration version pins, env-var requirements, known limitations, and the build-time model-download paths. Does not cover internal data model (Architect) or install recipes (Operator).

## Canonical facts

- **Dispatcher entry:** `ffpopt.ase.calculator.GenCalculator(mode, charge, spin, parm, crd, **kwargs)`. `mode` is uppercased once (`ase/calculator.py:66`); dispatch is a long `elif` chain (`ase/calculator.py:86-291`).
- **Optional dependency groups** (`pyproject.toml:37-61`): `pytorch`, `tensorflow`, `fairchem`. Each group lives in its own conda env (README §Installation step 4); the SDKs below are not all importable in any one env.
- **AmberTools / parmed (`from parmed import load_file`, `parmed.tools.actions.deleteDihedral`, `parmed.tools.actions.addDihedral`):**
  - Channel pin: `dacase::ambertools-dac=25` (`environment.yml:9`). The unofficial `ambertools` package has a charmm-module memory-leak/segfault if reloaded > 1 time (README:91-100; see Historian).
  - `numpy<2` is required for parmed (`pyproject.toml:31`).
  - The repo defines a GAFF/GAFF2 atom-type → atomic-number override in `Reader.FixParmedAtomicNumbers` (`Reader.py:45-74`) because parmed's auto-detection mis-types these atoms.
  - `Struct.ReadAmberParm` does *not* use a temp rst7 round-trip — see `Struct.py:170-200`; coordinates are merged in-place onto the loaded `parmed.AmberParm`.
- **sander / pysander (`SanderCalculator`):** invoked for `--model=sander`, `dftb2`, `dftb3`, `am1d` (the last three through `SanderSQMCalculator`, `ase/calculator.py:86-93`). Requires `parm7+rst7`. The README:108-117 documents the `execstack -c .../libsander.so` workaround on modern Linux when the executable stack flag breaks the loader.
- **geomeTRIC (`geometric-optimize` CLI):** invoked as a subprocess from `Options.argparse2geometric` (`Options.py:394-468`). The Python-level ASE engine is configured with `--engine ase --ase-class ffpopt.Struct.RestCalculator --ase-kwargs "{...}"`. Default coords `tric`, default converge `'set GAU'`, default maxiter `500`, default enforce `0.1`. The packaged `pkgdata/files/geometric_log.ini` is auto-resolved by `Options.configure_geometric_logging` (`Options.py:359-390`).
- **xtb / tblite:** `--model=xtb` instantiates `tblite.ase.TBLite(method="GFN2-xTB", charge=self.charge, verbosity=-1)` (`ase/calculator.py:103-105`). The legacy `xtb-python` path is documented (commented out, `ase/calculator.py:6-31`) but no longer used — the xtb-python package is unmaintained.
- **QDPI-2 (`QDpi2Calculator`):** the *only* ML model committed to the repo (`src/python/lib/ffpopt/pkgdata/qdpi/qdpi-2.0.pb`). Wraps `DPModel` (deepmd-kit). Requires `--group tensorflow` (deepmd-kit[tf]) or `--group pytorch` (deepmd-kit[torch]) depending on env — they are not interchangeable. See `ase/calculator.py:94-102`.
- **MACE (`mace.calculators.MACECalculator`):** models bundled under `pkgdata/mace-off/mace_off23/{small,medium,large}.model` and `mace_off24/MACE-OFF24_medium.model`. Mapping in `ase/calculator.py:112-128`. Auto-selects GPU if `torch.cuda.is_available()`, else CPU. Bare `--model=mace` aliases to `mace-off23_medium`.
- **AIMNet2 (`aimnet.calculators.AIMNet2ASE`):** `--model=aimnet2*` (aimnet2, aimnet2_b973c, aimnet2_2025, aimnet2nse, aimnet2pd); `base_calc` is set to `self.mode.lower()`. Installed from `git+https://github.com/isayevlab/aimnetcentral.git` (`pyproject.toml:47`). Dispatch at `ase/calculator.py:150-152`.
- **TorchANI (`torchani.models.{ANI1ccx,ANI1x,ANI2x}`):** dispatch at `ase/calculator.py:170-198`. There is a `try/except` because old torchani versions don't expose `.to(device)` — keep the fallback.
- **PySCF / pyscf.neo:** model string `pyscfneo/<xc>/<basis>/<quantum_nuc>/<nuc_basis>` (5 fields, slash-separated). `quantum_nuc=['']` falls back to plain DFT (`PySCF_DFT_Calculator`). Dispatch at `ase/calculator.py:137-149`. To install pyscf_neo, follow the README:124-136 git-clone-and-build path.
- **fennol (`FENNIXCalculator`):** `--model=fennix-bio1m` or `fennix-bio1s`. Models installed under `pkgdata/fennix/`. Implementation is `ffpopt.ase.fennolase.FENNIXCalculator`. JAX-driven; `JAX_PLATFORMS=cpu` removes `CUDA_VISIBLE_DEVICES` so JAX picks the CPU device (`ase/calculator.py:200-222`).
- **mopac (`ffpopt.ase.mopac.MOPAC`):** `--model in {AM1, MNDO, MNDOD, PM3, PM6, PM6-D3, PM6-DH+, PM6-DH2, PM6-DH2X, PM6-D3H4, PM6-D3H4X, PMEP, PM7, PM7-TS, RM1}` (`ase/calculator.py:224-229`). Writes temp files with randomized prefix and cleans them up (commits `e55517c`, `f44fb2a`).
- **PM6ML (`PM6MLCalculator`):** uses the bundled `pkgdata/pm6ml/PM6-ML_correction_seed8_best.ckpt` (downloaded from `Honza-R/mopac-ml`). Auto-selects CUDA if available (`ase/calculator.py:231-247`).
- **FairChem / OMOL25 (`fairchem.core.pretrained_mlip`, `FAIRChemCalculator`):** model strings `OMOL25-ESEN-SM-DIRECT`, `OMOL25-ESEN-SM-CONSERVING`, `OMOL25-ESEN-MD-DIRECT`. `OMOL25-ESEN-LG-DIRECT` raises `NotImplementedError` (`ase/calculator.py:165-168`). Loaded on CPU (`device="cpu"` is hard-coded; if you want GPU, change here). Requires `--group fairchem` and a HuggingFace login with permission to access `huggingface.co/facebook/OMol25` (README:471-501).
- **Psi4 (`ase.calculators.psi4.Psi4`):** any `--model` containing `/` is treated as `theory/basis` (`Options.ModelIsPsi4` at `Options.py:321-357`, dispatch at `ase/calculator.py:251-278`). `PSI_SCRATCH` env var is auto-defaulted to `cwd` if missing/invalid. `--psi4-memory` (default `'1gb'`) and `--psi4-num-threads` (default `4`) are passed through.
- **Orb-models (`WrappedORBCalculator`):** `--model in {orb-v3-direct-inf-omat, orb-v3-conservative-inf-omat}`. Hard-coded precision `'float32-high'`, device `"cuda"`, spin `1` (`ase/calculator.py:280-288`). Note the **GPU is hard-coded** — `--cpu` does *not* fall back the orb path automatically.
- **deepmd-kit:** installed as `deepmd-kit[torch]` (pytorch group) or `deepmd-kit[tf]` (tensorflow group). The two flavors are mutually exclusive.
- **rdkit:** installed via conda (`environment.yml:14`). Used by `Struct.from_rdkit` and `confsearch.ConvertMol2toRDKIT` for SMILES/Inchi/mol2 round-trips.
- **openbabel, dpdata, dftd3-python, simple-dftd3, libint:** declared in `environment.yml`; not directly dispatched from `GenCalculator` but pulled in by other tools / models.
- **Build-time model downloads (`CMakeLists.txt`):**
  - `https://github.com/FeNNol-tools/FeNNol-PMC` direct-URL downloads of `FENNIX-BIO1/v1.0/fennix-bio1{M,S}.fnx` (the `if(FALSE)` block at lines 71-92 is a deliberately retired LFS-clone path, see Historian).
  - `https://github.com/ACEsuit/mace-off` (FetchContent, `main` branch).
  - `https://github.com/Honza-R/mopac-ml` (FetchContent, `main` branch).
  - `https://github.com/dftbparams/{mio,3ob}` (FetchContent, `main` branch).
  - All `GIT_TAG main` — there are no upstream pins.

## Conventions

- New back-ends are added as additional `elif` branches in `GenCalculator.__init__`. Match the existing pattern: lazy-import the SDK, derive any model-file path via `importlib.resources.files("ffpopt") / "pkgdata/..."`, accept `mfile` kwarg for explicit override (`kwargs.get("mfile")`).
- When an SDK requires GPU-vs-CPU branching, follow the `torch.cuda.is_available()` try/except pattern (`ase/calculator.py:128-133`) so missing-torch envs don't crash.
- Use the `--cpu` env-mutation pattern (`Options.py:103-106`) when a model defaults to GPU but users sometimes need CPU.
- Document the new `--model=<name>` in `README.md §MODELS` with: description, link, element coverage.
- For models gated by upstream license (fennol, mace, pm6ml, fairchem/OMol25), the CMake `USE_<NAME>` flag must respect `ACADEMIC` (`CMakeLists.txt:53-57`).

## Anti-patterns

- Do not import any ML SDK (`torch`, `mace`, `aimnet`, `torchani`, `fairchem`, `fennol`, `tblite`, `psi4`, `pyscf`, `deepmd-kit`) at module top in `ase/calculator.py` or anywhere else. The dispatcher must remain importable in environments missing any specific SDK.
- Do not pin upstream model repos by tag without confirming the model file naming hasn't drifted — most are tracked at `GIT_TAG main`. If you pin, update both `CMakeLists.txt` and the `data_file_name` strings.
- Do not assume `aimnet2` and `aimnet2_wb97m` are different models — they are aliases (`base_calc=self.mode.lower()`, see README `--model` help at `Options.py:81`).
- Do not assume `mace` (without suffix) maps to the latest MACE-OFF — it aliases to `mace-off23_medium` (`ase/calculator.py:123-124`), not `mace-off24_medium`.
- Do not assume `--cpu` propagates to every model. Orb (`WrappedORBCalculator`) is hard-coded to `"cuda"`; OMOL25 is hard-coded to `"cpu"`. Verify the dispatch branch before promising CPU/GPU behavior.
- Do not assume mol2 input works for `--model=sander` — pysander requires `parm7+rst7` (README:155-165, 207-215). Conversely, torsion-fitting workflows (`ffpopt-GenDihedFit.py`) hard-code `args.model="sander"` (`bin/ffpopt-GenDihedFit.py:55`), so they require `parm7+rst7` even if the upstream scan used an ML model.
- Do not assume `psi4` cross-references resolve in Sphinx — `docs/conf.py:62` silences `parmed.*`; psi4 follows the same pattern.
- Do not change the `RestCalculator` ASE class name. It is referenced by string from the geomeTRIC subprocess (`ffpopt.Struct.RestCalculator`, `Options.py:461`). Renaming requires updating both call sites.
- Do not switch the geomeTRIC integration to in-process. The subprocess pattern is intentional — geomeTRIC spawns its own logging/state machine, and the subprocess boundary lets ML SDKs reload cleanly between scans.

## Pointers

- **The single most important file:** `src/python/lib/ffpopt/ase/calculator.py`.
- Mopac wrapper: `src/python/lib/ffpopt/ase/mopac.py`.
- Fennol wrapper: `src/python/lib/ffpopt/ase/fennolase.py`.
- Sub-package init exposing the ase namespace: `src/python/lib/ffpopt/ase/__init__.py`.
- geomeTRIC adapter: `src/python/lib/ffpopt/Options.py:394-468` (CLI builder) and `pkgdata/files/geometric_log.ini` (subprocess logging config).
- Build-time model downloads: `CMakeLists.txt` (one block per upstream).
- README §MODELS (the user-facing model catalogue): `README.md:236-514`.
- Per-model element coverage: `README.md:236-514` (each model lists supported elements).
- HuggingFace auth flow for OMOL25: `README.md:471-490`.
- AmberTools workarounds (execstack, QUICK_BASIS, glibc): `README.md:66-137`.

## Gaps

- `WrappedORBCalculator`, `QDpi2Calculator`, `PM6MLCalculator`, `SanderCalculator`, `SanderSQMCalculator`, `PySCF_DFT_Calculator`, `DPModel` are referenced from `ase/calculator.py` but their definitions are beyond the first 300 lines of that file — confirm import sites before depending on exact signatures.
- Upstream model URLs use `GIT_TAG main`; if upstream renames a file or moves a directory, the CMake fetch silently breaks. There is no pinning policy.
- Element-coverage tables in README §MODELS are hand-maintained and may drift from upstream model cards.
- OMOL25 device selection is hard-coded to CPU (`ase/calculator.py:157`), and orb-models is hard-coded to CUDA (`ase/calculator.py:288`). These asymmetries are intentional today, but undocumented.
- AmberTools workarounds (`execstack`, `QUICK_BASIS`, glibc) are README prose only — no automation, no helper script.

---
Last reviewed: 2026-05-19
Owner: piskuliche
