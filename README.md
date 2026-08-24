[![Documentation Status](https://app.readthedocs.org/projects/ligandparam/badge/?version=latest)](https://ligandparam.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ligandparam

**Amber ligand parameterization made modular**

Code originally written by York Lab (Rutgers), then in collaboration with Cheatham Lab (Utah)

`ligandparam` is a Python toolkit for generating force field parameters for nonstandard ligands and residues. It wraps familiar Amber / Gaussian tools behind a stage-based pipeline so you can run a full RESP workflow (or swap individual steps) without rewriting shell scripts.

**Docs:** [ligandparam.readthedocs.io](https://ligandparam.readthedocs.io/en/latest/)  
**Repo:** [github.com/piskulichz/ligandparam](https://github.com/piskulichz/ligandparam)

---

## Features

- **Recipe-based workflows** for common parameterization paths (BCC, single-orientation RESP, multi-orientation RESP, DeepMD-assisted minimization)
- **Composable stages** to add, remove, or reorder steps without forking the package
- **Gaussian integration** for geometry optimization and ESP / RESP charge fitting
- **Amber tooling** via Antechamber, `parmchk2`, and LEaP (`mol2` / `frcmod` / `lib`)
- **Integrated ffpopt + scission** for optional post-hoc dihedral corrections (`lig-dihed-correct`, `lig-scission`)
- **CLI utilities** for batch parameterization, fragmentation, SMILES -> PDB, and related prep tasks
- **Optional extras** for DeepMD / SQM / tblite, dihedral tooling, and OpenFF Sage conversion

---

## Requirements

### Python

- Python **≥ 3.10**
- Core dependencies are installed with the package (`numpy<2` for ParmEd compatibility, `MDAnalysis`, `ParmEd`, `RDKit`, …)
- Optional ML potentials (`mace`, `ani*`, `aimnet2*`) need `pip install ligandparam[ml-potentials]` (or a manual install of those stacks)

### External tools

Depending on the recipe you run, you will also need these on your `PATH` (or configured via recipe kwargs / environment variables):

| Tool | Used for |
|------|----------|
| [AmberTools](https://ambermd.org/AmberTools.php) (`antechamber`, `parmchk2`, `tleap`) | Atom typing, frcmod / lib generation |
| Gaussian (`g16` or compatible) | Optimization and RESP ESP calculations |

---

## Installation

Clone the repository and install into an environment (Miniforge / conda recommended):

```bash
git clone https://github.com/piskulichz/ligandparam.git
cd ligandparam
mamba env create -f env.yaml   # or: conda env create -f env.yaml
conda activate ligandparam
pip install .
```

### Validate your install

After installing, run the install-validation suite (no AmberTools / Gaussian required):

```bash
python -m unittest tests.test_install_validation -v
```

Optional extras (`tblite`, `geometric`, AmberTools on `PATH`) are checked when
present and skipped with an explicit reason when absent. A clean core install
should report `OK` (with possible `skipped` optional tests).

### Developer regression tests

After changing code under `src/`, run the developer regression suite:

```bash
python -m unittest tests.test_developer_regression -v
```

Both suites together:

```bash
python -m unittest tests.test_install_validation tests.test_developer_regression -v
```

### Optional extras

Run these **from the clone** (the extra is on this tree, not the PyPI ``1.0.0`` wheel):

```bash
pip install -e ".[tblite]" # GFN2-xTB for lig-dihed-correct --model xtb
pip install -e ".[dihed]"  # ndfes + geometric (geomeTRIC) for lig-dihed-correct
pip install -e ".[ml]"     # DeepMD (install TensorFlow via conda on HPC)
pip install -e ".[jax]"    # JAX autodiff for --fit-backend jax
pip install -e ".[sage]"   # OpenFF Sage conversion
pip install -e ".[docs]"   # Sphinx documentation build
pip install -e ".[all]"    # everything above (still needs TF from conda on many HPCs)
```

``pip install 'ligandparam[jax]'`` (no ``-e``, no ``.``) pulls **PyPI 1.0.0** and
can uninstall a local 1.5.x install. On HPC, prefer conda-forge
for JAX itself, then keep the editable tree:

```bash
conda install -c conda-forge jax jaxlib
pip install -e .
```

Dihedral corrections use the integrated [`src/ffpopt`](src/ffpopt/) and
[`src/scission`](src/scission/) packages (installed with the package). Use
`pip install -e ".[dihed,tblite]"`, keep AmberTools on `PATH`, then run
`lig-dihed-correct` (or `lig-scission` alone) after `lig-getparam`.

For DeepMD recipes, prefer conda on HPC (pip TensorFlow often has no wheel):

```bash
conda install -c conda-forge tensorflow deepmd-kit
pip install -e ".[ml]"
```

Editable install for development:

```bash
pip install -e .
```

---

## Quick start

Parameterize a ligand with the multi-orientation RESP recipe:

```python
from pathlib import Path
from ligandparam.recipes import FreeLigand

recipe = FreeLigand(
    in_filename=Path("ligand.pdb"),
    cwd=Path("output"),
    net_charge=0,
    atom_type="gaff2",
    molname="LIG",
    logger="file",
    nproc=12,
    mem=8,
)

recipe.setup()
recipe.list_stages()
recipe.execute(dry_run=False, nproc=12, mem=8)
```

A faster single-orientation RESP path:

```python
from ligandparam.recipes import LazyLigand

recipe = LazyLigand(
    in_filename="ligand.pdb",
    cwd="output",
    net_charge=0,
    molname="LIG",
    logger="stream",
)
recipe.setup()
recipe.execute(nproc=8, mem=8)
```

Typical outputs (depending on recipe) include minimized / RESP `mol2` files, `frcmod`, and Leap `lib` libraries under your working directory.

---

## Recipes

| Recipe | What it does | Best when |
|--------|----------------|-----------|
| **`LazierLigand`** | Antechamber charges (e.g. BCC) + Leap | You want a fast, no-Gaussian path |
| **`LazyLigand`** | Gaussian minimize + single-orientation RESP | Standard RESP without multi-orientation sampling |
| **`FreeLigand`** | 28-point quaternion SO(3) sampling + multi-RESP fit | Higher-quality charge averaging over well-separated orientations |
| **`DPLigand`** | DeepMD minimization + Gaussian RESP | You have a DeepMD model for geometry |
| **`DPFreeLigand`** | DeepMD + ``so3_n28`` multi-RESP | DeepMD geometry + FreeLigand-quality charges |
| **`SQMLigand`** | SQM / DeepMD-assisted minimize + RESP | Semiempirical-assisted workflows |

Each recipe builds an ordered list of **stages**. You can inspect and modify that list after `setup()`:

```python
recipe.setup()
recipe.list_stages()
recipe.remove_stage("Normalize1")
# recipe.insert_stage(new_stage, "SomeExistingStage")
recipe.execute()
```

`FreeLigand` and `DPFreeLigand` use the deterministic `so3_n28` quaternion pack
by default. To reproduce the previous alpha/beta Euler grid, pass
`orientation_protocol="legacy_euler"`. Both protocols use 28 Gaussian ESP jobs
and feed the same multi-RESP → `parmchk2` → LEaP path (`.frcmod` / `.lib`).

### Optional dihedral corrections (ffpopt + scission)

Runtime packages live at [`src/ffpopt`](src/ffpopt/) and
[`src/scission`](src/scission/) (next to `ligandparam`). After
`lig-getparam` finishes, run torsion correction in the same session
(fragmented dihed-twist → merged frcmod; the `.lib` is unchanged). You need
AmberTools on `PATH` plus an HL model stack (e.g. `xtb` via tblite, or `qdpi2`).
Fragment workers fair-share `-n` / `nproc` cores via a live lease file and
reclaim cores from finished fragments at the next scan phase.

```bash
lig-getparam -i chaps.mol2 -r CHA -d CHA3 -rn freeligand --net_charge 0 -n 10 -mem 32
lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 10 --fast
```

Use ``--model qdpi2`` (or ``mace``, …) if that stack is installed
(``ligandparam[ml-potentials]`` for MACE / TorchANI / AIMNet); ``xtb``
only needs ``tblite``.

Fragment alone with scission (without fitting):

```bash
lig-scission fragment -d CHA3 -r CHA --label chaps
# or the upstream-style CLI:
scission fragment --mol2 ... --lib ... --frcmod ... --outdir frags
```

Python recipes can still append the stage with ``dihed_correct=True``
(``FreeLigand`` / ``LazyLigand`` / ``DPFreeLigand``). Use ``dihed_delta`` for
the wavefront step (CLI ``--delta``) and ``dihed_fragment_config`` for
scission settings. Prefer the standalone CLIs when running interactively
after ``lig-getparam``.

---

## Command-line tools

Installed entry points (see `pyproject.toml`):

| Command | Purpose |
|---------|---------|
| `lig-getparam` | Batch-run parameterization recipes |
| `lig-dihed-correct` | ffpopt dihedral corrections on recipe mol2/lib/frcmod |
| `lig-scission` / `scission` | Fragment a ligand (or merge fragment frcmods) |
| `smiles-to-pdb` | Convert a SMILES string to a 3D PDB |
| `lighfix` | Fix ligand hydrogenation / bonding from PDB input |
| `lig-to-sage` | Convert mol2 parameters toward OpenFF Sage (optional `[sage]`) |
| `ffpopt-specialty` | Quarantined tools: sugar/pucker, JSON, animate |

Supported torsion prep scripts (`ffpopt-PrepareInput.py`,
`ffpopt-DihedWavefront.py`, …) and secondary RespFit/CPE CLIs remain as
console scripts; see `docs/.../cli.rst`.

```bash
lig-getparam --help
lig-dihed-correct --help
lig-scission --help
smiles-to-pdb --help
```

Re-run after walltime without repeating finished Gaussian work (default):
complete logs are skipped, and only incomplete orientation ESP jobs are
re-submitted. Force all Gaussian jobs again with ``-O``:

```bash
lig-getparam -i ligand.pdb -r LIG -d param -rn freeligand -O
```

---

## Examples

Runnable examples live under [`examples/`](examples/):

| Directory | Topic |
|-----------|--------|
| `01_LazyLigand` | Single-orientation RESP |
| `02_FreeLigand` | Multi-orientation RESP |
| `03_ModifySteps` | Editing the stage pipeline |
| `04_FromSmiles` | SMILES -> PDB prep |
| `05_FromSDF` | Working from SDF libraries |

Sphinx also documents example 07 (dihedral correction after `lig-getparam`);
see the [documentation examples](https://ligandparam.readthedocs.io/en/latest/).

---

## Documentation

- **User guide & API:** https://ligandparam.readthedocs.io/en/latest/
- Build locally:

```bash
pip install ".[docs]"
cd docs
make html
```

---

## Project layout

```text
src/
├── ligandparam/         # Parameterization recipes, stages, CLI
│   ├── recipes/
│   ├── stages/          # Includes StageDihedTwistCorrection
│   ├── cli/             # lig-getparam, lig-dihed-correct, lig-scission, …
│   ├── io/              # gaussian_io, leap_io, smiles, orientations, …
│   └── …
├── ffpopt/              # Torsion optimization
│   ├── runtime/         # console, progress boards, CPU budget, --fast
│   ├── scan/            # WaveFront, WaveFrontND, mixins, ScanAnalysis
│   └── workflows/, dihed/, geom/, affdo/
└── scission/            # Integrated ligand fragmentation + frcmod merge

tests/
├── test_install_validation.py    # user install gate
└── test_developer_regression.py  # developer regression after code changes
```

---

## Contributing

Issues and pull requests are welcome.

1. Fork and clone the repository
2. Create a feature branch
3. Install in editable mode: `pip install -e ".[docs]"`
4. Run `python -m unittest tests.test_developer_regression -v`
5. Make focused changes with clear commit messages
6. Open a PR describing *why* the change is needed

### Releasing a new version

1. Bump `version` in `pyproject.toml` **and** `__version__` in `src/ligandparam/__init__.py` (keep them in sync)
2. Commit the bump, then tag it:

```bash
git tag 1.5.1
git push origin --tags
```

---

## Authors

- [Zeke Piskulich (York Lab)](https://theory.rutgers.edu/profile.php?people_id=399)
- [German P. Barletta (York Lab)](https://theory.rutgers.edu/profile.php?people_id=407)
- [Timothy J. Giese (York Lab)](https://theory.rutgers.edu/profile.php?people_id=3)
- [Nate Levinzon (Cheatham Lab)](https://people.utah.edu/basic.hml?eid=273961099)

---

## License

This project is licensed under the **MIT License**.
