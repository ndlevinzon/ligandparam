[![Documentation Status](https://app.readthedocs.org/projects/ligandparam/badge/?version=latest)](https://ligandparam.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ligandparam

**Amber ligand parameterization made modular.**

`ligandparam` is a Python toolkit for generating force-field parameters for nonstandard ligands and residues. It wraps familiar Amber / Gaussian tools behind a stage-based pipeline so you can run a full RESP workflow (or swap individual steps) without rewriting shell scripts.

**Docs:** [ligandparam.readthedocs.io](https://ligandparam.readthedocs.io/en/latest/)  
**Repo:** [github.com/piskulichz/ligandparam](https://github.com/piskulichz/ligandparam)

---

## Features

- **Recipe-based workflows** for common parameterization paths (BCC, single-orientation RESP, multi-orientation RESP, DeepMD-assisted minimization)
- **Composable stages** — add, remove, or reorder steps without forking the package
- **Gaussian integration** for geometry optimization and ESP / RESP charge fitting
- **Amber tooling** via Antechamber, `parmchk2`, and LEaP (`mol2` / `frcmod` / `lib`)
- **CLI utilities** for batch parameterization, SMILES -> PDB, and related prep tasks
- **Optional extras** for DeepMD / SQM minimization and OpenFF Sage conversion

---

## Requirements

### Python

- Python **≥ 3.10**
- Core dependencies are installed with the package (`numpy`, `MDAnalysis`, `ParmEd`, `RDKit`, …)

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

### Optional extras

```bash
pip install ".[ml]"     # DeepMD / SQM-related workflows
pip install ".[sage]"   # OpenFF Sage conversion
pip install ".[docs]"   # Sphinx documentation build
pip install ".[all]"    # everything above
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
| **`FreeLigand`** | Multi-orientation ESP + multi-RESP fit | Higher-quality charge averaging over rotations |
| **`DPLigand`** | DeepMD minimization + Gaussian RESP | You have a DeepMD model for geometry |
| **`DPFreeLigand`** | DeepMD + multi-orientation RESP | DeepMD geometry + FreeLigand-quality charges |
| **`SQMLigand`** | SQM / DeepMD-assisted minimize + RESP | Semiempirical-assisted workflows |

Each recipe builds an ordered list of **stages**. You can inspect and modify that list after `setup()`:

```python
recipe.setup()
recipe.list_stages()
recipe.remove_stage("Normalize1")
# recipe.insert_stage(new_stage, "SomeExistingStage")
recipe.execute()
```

---

## Command-line tools

Installed entry points (see `pyproject.toml`):

| Command | Purpose |
|---------|---------|
| `lig-getparam` | Batch-run parameterization recipes |
| `smiles-to-pdb` | Convert a SMILES string to a 3D PDB |
| `lighfix` | Fix ligand hydrogenation / bonding from PDB input |
| `lig-to-sage` | Convert mol2 parameters toward OpenFF Sage |

```bash
lig-getparam --help
smiles-to-pdb --help
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

More walkthroughs are in the [documentation examples](https://ligandparam.readthedocs.io/en/latest/).

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
src/ligandparam/
├── recipes/        # End-to-end parameterization workflows
├── stages/         # Pipeline steps (Gaussian, RESP, Leap, …)
├── io/             # Coordinate / Gaussian / Leap I/O helpers
├── multiresp/      # Multi-orientation RESP utilities
├── cli/            # Command-line entry points
├── interfaces.py   # Wrappers for external binaries
└── driver.py       # Stage orchestration
```

---

## Contributing

Issues and pull requests are welcome.

1. Fork and clone the repository
2. Create a feature branch
3. Install in editable mode: `pip install -e ".[docs]"`
4. Make focused changes with clear commit messages
5. Open a PR describing *why* the change is needed

### Releasing a new version

1. Bump `version` in `pyproject.toml` **and** `__version__` in `src/ligandparam/__init__.py` (keep them in sync)
2. Commit the bump, then tag it:

```bash
git tag 1.0.2
git push origin --tags
```

---

## Authors

- [Zeke Piskulich](mailto:piskulichz@gmail.com)
- [Nate Levinzon](mailto:ndlevinzon@gmail.com)

---

## License

This project is licensed under the **MIT License**.
