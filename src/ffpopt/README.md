# ffpopt (integrated)

Python package for force-field torsion optimization, vendored alongside
``ligandparam`` under ``src/ffpopt``.

## Package layout

Shared monorepo convention: public ``__init__.py``, ``README.md``, CLI/bin
entrypoints, domain modules. Runtime UX helpers and scan engines live in
subpackages; PascalCase mega-modules stay intact at the package root (or are
re-exported from ``scan/``).

| Path | Concern |
|------|---------|
| ``runtime/`` | ``console``, ``progress_board`` (+ fragment aliases), ``cpu_budget``, ``fast_wavefront`` |
| ``scan/`` | ``WaveFront``, ``WaveFrontND``, ``wavefront_mixins``, ``ScanAnalysis`` |
| ``GeomOpt`` | ASE / geomeTRIC optimization |
| ``Workflows`` | Twist + fragmented twist orchestration |
| ``Dihedrals`` | Fit types, solvers, Parmed script, puckers |
| ``ase/``, ``cpefit/``, ``confsearch/``, ``constants/``, ``scosmo/``, ``bin/`` | Specialty stacks + CLIs |

Root modules such as ``ffpopt.console`` / ``ffpopt.WaveFront`` remain as **thin
compatibility re-exports**. Prefer ``ffpopt.runtime.*`` and ``ffpopt.scan.*``
in new code.

Primary API for ligandparam integration:

```python
from ffpopt.Workflows import run_fragmented_dihed_twist_workflow
```

CLI (installed with ligandparam):

```bash
lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb --fast
```

Fragmentation is provided by the integrated ``src/scission`` package (also
exposed as ``lig-scission`` / ``scission``).

## What is (and is not) overlapping with ligandparam

ligandparam owns **parameterization** (antechamber / Gaussian RESP / parmchk /
LEaP → mol2+lib+frcmod). ffpopt owns **post-hoc torsion fitting**. Those are
complementary, not duplicates.

Shared helper (deduplicated): ``CopyParm`` lives in ``ffpopt.AmberParm`` and is
re-exported from ``ligandparam.multiresp.parmhelper``. Core-budget splitting
lives in ``ffpopt.runtime.fast_wavefront.split_core_budget``.

Not used by ``lig-dihed-correct`` (kept for standalone ffpopt CLIs): RespFit,
cpefit, confsearch, DeltaPuckerFit, WaveFrontND, Json* utilities.

See ``GLOSSARY.md`` for models and terminology. Runtime code lives in this
``src/ffpopt`` tree. The optional ``ffpopt-main/`` checkout (gitignored) is an
upstream reference only — not required after ``pip install``.
