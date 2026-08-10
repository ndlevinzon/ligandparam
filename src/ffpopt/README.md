# ffpopt (integrated)

Python package for force-field torsion optimization, vendored alongside
``ligandparam`` under ``src/ffpopt``.

## Package layout

Shared monorepo convention: public ``__init__.py``, ``README.md``, CLI/bin
entrypoints, domain modules. Runtime UX helpers and scan engines live in
subpackages; PascalCase mega-modules stay intact where they are meaningful
(``Workflows``, ``Dihedrals``, ``GeomOpt`` at package root; wavefront engines
under ``scan/``).

| Path | Concern |
|------|---------|
| ``runtime/`` | ``console``, ``progress_board`` (+ fragment aliases), ``cpu_budget``, ``fast_wavefront`` |
| ``scan/`` | ``WaveFront``, ``WaveFrontND``, ``wavefront_mixins``, ``ScanAnalysis`` |
| ``GeomOpt`` | ASE / geomeTRIC optimization |
| ``Workflows`` | Twist + fragmented twist orchestration |
| ``Dihedrals`` | Fit types, solvers, Parmed script, puckers |
| ``ase/``, ``cpefit/``, ``confsearch/``, ``constants/``, ``scosmo/``, ``bin/`` | Specialty stacks + CLIs |

Root entrypoints use the canonical packages ``ffpopt.runtime.*`` and
``ffpopt.scan.*`` (no compatibility shims).

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

## Wavefront evaluate policy

After each node finishes, profile minima and neighbor spawn follow a shared
policy (``ffpopt.scan.wavefront_mixins.evaluate_wavefront_minimum``):

| Case | Profile min | Spawn? |
|------|-------------|--------|
| Soft, first at bin | Store soft energy/geom | Yes once (seed coverage) |
| Soft, improves soft min | Update if lower | No |
| Hard vs soft incumbent | Replace soft only if ``E_hard <= E_soft`` | Only if hard accepted |
| Hard, ``E < min`` within threshold | Update quietly | No |
| Hard, ``E < min - threshold`` | Update | Yes |
| Hard, ``E >= min`` | No change | No |

``loose`` / ``*-loose`` recoveries are treated like soft for spawn. Soft-maxiter
stays soft. ``--fast`` remains a wall-time trade (coarser Δ, looser converge).

## Dihedral fit chi^2

GenDihedFit's objective is a **shape match**: mean-centered HL-LL residual
(``d = (hl - ll) - mean(hl - ll)``). Independent min-shifts of HL and LL are
not used in chi^2 (plot files may still min-shift for display). Under fixed
geometry, force constants enter linearly and are solved with bounded linear
least squares (phase fixed at 0).
