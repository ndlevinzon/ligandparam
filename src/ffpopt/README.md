# ffpopt (integrated)

Python package for force-field torsion optimization, vendored alongside
``ligandparam`` under ``src/ffpopt``.

Primary API for ligandparam integration:

```python
from ffpopt.Workflows import run_fragmented_dihed_twist_workflow
```

Layout (write once):

| Module | Concern |
|--------|---------|
| ``GeomOpt`` | ASE / geomeTRIC optimization |
| ``WaveFront`` / ``WaveFrontND`` | Constrained dihedral wavefront scans |
| ``wavefront_mixins`` | Shared node IPC / soft-opt / checkpoints |
| ``Workflows`` | Twist + fragmented twist orchestration |
| ``Dihedrals`` | Fit types, solvers, Parmed script, puckers |
| ``cpu_budget``, ``progress_board``, ``console``, ``fast_wavefront``, ``fragment_progress`` | Parallelism / UX helpers (single home each) |

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
re-exported from ``ligandparam.multiresp.parmhelper``.

Not used by ``lig-dihed-correct`` (kept for standalone ffpopt CLIs): RespFit,
cpefit, confsearch, DeltaPuckerFit, WaveFrontND, Json* utilities.

See ``GLOSSARY.md`` for models and terminology. Runtime code lives in this
``src/ffpopt`` tree. The optional ``ffpopt-main/`` checkout (gitignored) is an
upstream reference only — not required after ``pip install``.
