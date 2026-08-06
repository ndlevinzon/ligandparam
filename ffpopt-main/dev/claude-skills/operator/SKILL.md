---
name: ffpopt-operator
description: Use when changing ffpopt's build, install, conda environment, CMake configuration, CI pipeline, or env-var contract. Covers scikit-build-core + CMake flow, conda env recipe, ML-model download gating, GitLab Pages CI, modulefile setup.
---

# Operator — ffpopt

## Scope
Build, install, environment configuration, and CI. ffpopt is a *library + CLI suite distributed as a Python wheel*, with a CMake-driven scikit-build flow that also fetches/installs ML model checkpoints and a Fortran 90 binary. There is no service to deploy; "operations" here means making install/build/CI work and not break for downstream users.

## Canonical facts

- **Build backend:** `scikit-build-core>=0.5` (`pyproject.toml:1-3`). CMake >= 3.15. Python >= 3.8 declared, 3.12 in practice (`CMakeLists.txt:62`).
- **Top-level build entry:** `CMakeLists.txt` (in repo root).
  - Detects `SKBUILD_SCRIPTS_DIR` to switch between pip/scikit-build install and a classic CMake build.
  - Fortran 90 executable target `ffpopt-respf` built from `src/resp/resp.f` (`CMakeLists.txt:223-235`); installed under `${SKBUILD_SCRIPTS_DIR}` (the wheel `Scripts/` dir).
  - Python package install for the non-scikit path is delegated to `src/python/CMakeLists.txt`, which calls `python -m pip install --prefix=<>...`.
- **Wheel layout** (`pyproject.toml:81`):
  - `ffpopt` → `src/python/lib/ffpopt`
  - `ffpopt/bin` → `src/python/bin`
  - `wheel.install-dir = "."`, `wheel.py-api = "py3"`
  - `sdist.include`: `src/python/lib/ffpopt/**/*`, `src/python/bin/**/*`, `src/python/scripts.py`, `CMakeLists.txt`
- **CMake user-tunable variables** (`CMakeLists.txt:22-57`):
  - `ACADEMIC` (env var). Defaults FALSE (industry user). When FALSE, `USE_FENNOL/USE_MACE/USE_PM6ML` are forced FALSE because the upstream models are not redistributable to industry users without explicit permission.
  - `FFPOPTVERSION` — must match `pyproject.toml:10`. scikit-build passes `FFPOPTVERSION=1.1.0` via `[tool.scikit-build.cmake.define]`.
  - `USE_FENNOL`, `USE_MACE`, `USE_PM6ML`, `USE_DFTB` — each gates a CMake `FetchContent_Declare` / `ExternalProject_Add` for model artifacts.
  - `USE_DFTB` is auto-enabled when `AMBERHOME` is set; auto-disabled when `AMBERHOME` is missing (with a warning).
- **Model-checkpoint install destinations:**
  - Fennix: `ffpopt/pkgdata/fennix/{fennix-bio1M.fnx,fennix-bio1S.fnx}`.
  - MACE-OFF: `ffpopt/pkgdata/mace-off/mace_off23/{MACE-OFF23_small,medium,large}.model` and `mace_off24/MACE-OFF24_medium.model`.
  - PM6ML: `ffpopt/pkgdata/pm6ml/PM6-ML_correction_seed8_best.ckpt`.
  - DFTB params: `$AMBERHOME/dat/slko/{mio-1-1,3ob-3-1}/` (note: outside the wheel — they go into the user's AmberTools install).
  - Bundled in-tree: `src/python/lib/ffpopt/pkgdata/qdpi/qdpi-2.0.pb` (the QDPI-2 model is the only model checkpoint committed to the repo).
- **Conda env recipe:** `environment.yml`. Pins `python=3.12`, `dacase::ambertools-dac=25`, `parmed`, `ase`, `openbabel`, `geometric`, `dpdata`, `dftd3-python` (twice), `rdkit`, `tblite`, `tblite-python`, `simple-dftd3`, `psi4`, `conda-forge/label/libint_dev::libint`, `mopac`. Channel order: `conda-forge`, `pytorch`, `nvidia`, `nodefaults`.
- **Optional pip dependency groups** (`pyproject.toml:37-61`):
  - `pytorch`: torch/torchvision/torchaudio, cuequivariance + Torch ops (CUDA 12), mace-torch, torchani, aimnet (via git), fennol[cuda], torchmd-net, deepmd-kit[torch], orb-models.
  - `tensorflow`: tensorflow[and-cuda], deepmd-kit[tf].
  - `fairchem`: fairchem-core (git), huggingface_hub.
  - **Each group is meant for its own conda env** — install with `python3 -m pip install --group <name> --extra-index-url https://download.pytorch.org/whl/cu121 .` (README:55-63).
- **Install recipe (README:8-63):**
  ```
  mamba env create --yes -n ffpopt-pytorch    -f environment.yml
  mamba env create --yes -n ffpopt-tensorflow -f environment.yml
  mamba activate ffpopt-tensorflow
  ACADEMIC=TRUE python3 -m pip install --group tensorflow --extra-index-url https://download.pytorch.org/whl/cu121 .
  mamba activate ffpopt-pytorch
  ACADEMIC=TRUE python3 -m pip install --group pytorch    --extra-index-url https://download.pytorch.org/whl/cu121 .
  ```
- **HPC modulefile:** `modulefiles/ffpopt` — TCL modulefile that activates the `ffpopt` mamba env, sets `QUICK_BASIS` to `${base_path}/envs/${env_name}/AmberTools/src/quick/basis`, and prepends mamba `bin/` to `PATH`. The placeholder `PATH_TO_CONDA_BASE` must be replaced per-cluster.
- **CI pipeline (`.gitlab-ci.yml`):** image `python:3.12`. One `pages` job (stage `deploy`):
  ```
  pip install --no-cache-dir "numpy<2" scipy matplotlib ase parmed geometric sphinx sphinx_rtd_theme
  export PYTHONPATH="$CI_PROJECT_DIR/src/python/lib:$PYTHONPATH"
  sphinx-build -b html -n docs public
  ```
  - Only runs on `$CI_DEFAULT_BRANCH` (main). Output: `public/` artifact, published as GitLab Pages.
  - Does not install ffpopt itself or any ML SDK — autodoc imports succeed because `parmed`, `ase`, `geometric` are sufficient for the modules that are autodoc-listed.
- **Hosting:** GitLab (`pyproject.toml:72`, `gitlab.com/RutgersLBSR/ffpopt`); user-facing site at `https://ffpopt-b083ab.gitlab.io/`. SKILLS_CONTEXT.md is explicit: the repo is hosted on GitLab, not GitHub.

## Conventions

- Run installs via the documented recipe (README §INSTALLATION); maintain `ACADEMIC=TRUE` as the conventional install env var for academic users — it gates downloadable ML models.
- Keep `pyproject.toml:10` `version` and `CMakeLists.txt:[tool.scikit-build.cmake.define] FFPOPTVERSION` in sync. Updating one requires updating the other.
- New optional ML back-ends:
  1. Add a CMake `FetchContent`/`ExternalProject` block under a `USE_<NAME>` guard.
  2. Install model files under `ffpopt/pkgdata/<family>/`.
  3. Force-disable for `ACADEMIC=FALSE` if the upstream license requires it.
  4. Add an `elif "<NAME>" in self.mode:` branch in `src/python/lib/ffpopt/ase/calculator.py:GenCalculator.__init__`.
  5. Add a `--model=<name>` entry in README §MODELS.
- Env vars consumed at runtime: `AMBERHOME`, `QUICK_BASIS`, `PSI_SCRATCH`, `JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `DP_INTRA_OP_PARALLELISM_THREADS`, `DP_INTER_OP_PARALLELISM_THREADS`. The last three are auto-set to `"1"` if unset by `ase/calculator.py:35-37`. Document any new env vars in README and the modulefile.
- For shared-cluster modulefile installs, replace `PATH_TO_CONDA_BASE` in `modulefiles/ffpopt:13` per cluster; do *not* commit a hard-coded path.

## Anti-patterns

- Do not let `numpy>=2` enter any conda env that also has `parmed`/`ambertools` — `pyproject.toml:31` pins `numpy<2`, and the psi4 conda package's `numpy==2.x` requires the env-isolation workaround documented in README:80-87.
- Do not combine `--group pytorch`, `--group tensorflow`, `--group fairchem` into one env. They install mutually incompatible CUDA/cuDNN/cuequivariance stacks.
- Do not switch the build backend away from `scikit-build-core`. The CMake step is load-bearing: it builds the Fortran `ffpopt-respf` and fetches ML model files.
- Do not vendor ML model checkpoints into the repo. The only intentionally bundled checkpoint is `pkgdata/qdpi/qdpi-2.0.pb`; everything else is fetched at build time so industry/academic gating remains enforceable.
- Do not expand the CI pipeline to install ML SDKs without ensuring the GitLab runner has the GPU/CUDA resources — the current `python:3.12` image will fail any pytorch/cuequivariance install.
- Do not enable `--no-cache-dir` removal in the CI Sphinx install to "speed it up"; the `pip install --no-cache-dir "numpy<2" ...` is intentional (CI ephemeral runners benefit from no cache).
- Do not use `git clone` of fennol-pmc with Git LFS — the original approach failed because LFS rate-limits downloads. The current CMake flow downloads model files via direct URLs (`CMakeLists.txt:98-114`; commit `b6fa841`).
- Do not assume GitHub mirrors exist. The canonical remote is GitLab; any CI/CD or CODEOWNERS changes that assume GitHub will break.

## Pointers

- Build manifest: `pyproject.toml`.
- Root CMake: `CMakeLists.txt`.
- Sub-CMakes: `src/resp/CMakeLists.txt`, `src/python/CMakeLists.txt`.
- Conda env: `environment.yml`.
- Pip requirements (legacy/extras): `requirements.txt`.
- Install instructions: `README.md` §INSTALLATION (lines 1-137).
- CI: `.gitlab-ci.yml`.
- Modulefile: `modulefiles/ffpopt`.
- Manifest for sdist extras: `MANIFEST.in`.
- ML model URL constants: `CMakeLists.txt` (one `FetchContent_Declare`/`ExternalProject_Add` per family).

## Gaps

- There is no Dockerfile and no Helm/k8s manifests — the repo is intentionally not container-deployable; do not assume the existence of an OCI image.
- No release process is documented (no tags, no `CHANGELOG.md`). Bumping version means editing both `pyproject.toml:10` and the `FFPOPTVERSION` line in `pyproject.toml [tool.scikit-build.cmake.define]`, but there is no CI release stage to publish artifacts.
- `requirements.txt` partially duplicates `[dependency-groups] pytorch` and is not used by the install recipe in README; treat it as legacy unless a maintainer confirms otherwise.
- The `pages` CI job runs `sphinx-build -n` (nitpicky) but pre-silences many cross-references; verify whether new warnings are real before silencing them.
- No `.dockerignore`, no `Makefile` at repo root, no `justfile`/`Taskfile` — local build orchestration is README-driven only.

---
Last reviewed: 2026-05-19
Owner: piskuliche
