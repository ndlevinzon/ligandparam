# ffpopt (integrated)

Python package for force-field torsion optimization, vendored alongside
``ligandparam`` under ``src/ffpopt``.

ligandparam drives it through ``lig-dihed-correct``. Two scan modes:

* **Fragment (default)** - scission caps, twist each piece, merge DIHE by atom type.
* **Whole-ligand** (``--whole-ligand``) - twist rotatable bonds on the intact parent.

```bash
# fragment
lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb --fast

# whole-ligand
lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb --fast \
  --whole-ligand --soft-dihed-restraint --fit-full --fit-backend jax
```

## Package layout

Shared monorepo convention: public ``__init__.py``, ``README.md``, CLI/bin
entrypoints, domain packages.

| Path | Concern |
|------|---------|
| ``runtime/`` | ``Console``, ``ProgressBoard``, ``CpuBudget``, ``FastWavefront`` |
| ``scan/`` | ``WavefrontEngine`` (1-D + N-D); ``WaveFront`` / ``WaveFrontND`` facades; ``WavefrontMixins``; ``ScanAnalysis`` |
| ``workflows/`` | Twist, fragmented, whole-ligand orchestration + bond batches |
| ``dihed/`` | Thin ``Dihedrals`` facade; ``DihedFitTypes``, Fourier, ParmEd, solvers, pucker |
| ``geom/`` | ``GeomOpt``, constraints/restraints, geomeTRIC driver, linear-torsion |
| ``affdo/`` | Opt-in extras: log, centroid profiles, Boltzmann charges |
| ``ase/``, ``cpefit/``, ``confsearch/``, ``constants/``, ``scosmo/``, ``bin/`` | Specialty stacks + CLIs |

Canonical imports: ``ffpopt.workflows``, ``ffpopt.dihed``, ``ffpopt.geom``,
``ffpopt.scan``, ``ffpopt.runtime``.

Primary API for ligandparam integration:

```python
from ffpopt.workflows import run_fragmented_dihed_twist_workflow
```

CLI (installed with ligandparam):

```bash
lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb --fast
```

Tunable ``export FFPOPT_*`` defaults live in
``ffpopt/pkgdata/files/env_defaults.json`` (commented JSON; this is the store
the code reads). Overlay a copy with ``export FFPOPT_DEFAULTS=/path/to.json``;
per-key ``EXPORT`` still wins.

Supported torsion / prep scripts stay as console entry points
(``PrepareInput``, ``DihedWavefront``, twist workflow, ...). Specialty
tools (sugar/pucker, JSON, animate) are quarantined behind one dispatcher:

```bash
ffpopt-specialty Json2Img --help
```

Secondary-supported (not on the ``lig-*`` path): RespFit, DeltaRespFit,
CpeFit. Fragmentation is provided by the integrated ``src/scission``
package (also exposed as ``lig-scission`` / ``scission``).

## What is (and is not) overlapping with ligandparam

ligandparam owns **parameterization** (antechamber / Gaussian RESP / parmchk /
LEaP -> mol2+lib+frcmod). ffpopt owns **post-hoc torsion fitting**. Those are
complementary, not duplicates.

Shared helper (deduplicated): ``CopyParm`` lives in ``ffpopt.AmberParm`` and is
re-exported from ``ligandparam.multiresp.ParmHelper``. Core-budget splitting
lives in ``ffpopt.runtime.FastWavefront.split_core_budget``.

See ``GLOSSARY.md`` for models and terminology. Runtime code lives in this
``src/ffpopt`` tree. The optional ``ffpopt-main/`` checkout (gitignored) is an
upstream reference only - not required after ``pip install``.

## Wavefront evaluate policy

After each node finishes, profile minima and neighbor spawn follow a shared
policy (``ffpopt.scan.WavefrontMixins.evaluate_wavefront_minimum``):

| Case | Profile min | Spawn? |
|------|-------------|--------|
| Soft, first at bin | Store soft energy/geom | Yes once (seed coverage) |
| Soft, improves soft min | Update if lower | No |
| Hard vs soft incumbent | Replace soft only if ``E_hard <= E_soft`` | Only if hard accepted |
| Hard, ``E < min`` within threshold | Update quietly | No |
| Hard, ``E < min - threshold`` | Update | Yes |
| Hard, ``E >= min`` | No change | No |

``loose`` / ``*-loose`` recoveries are treated like soft for spawn. Soft-maxiter
stays soft. Near-linear constrained torsions use a dedicated
``linear-torsion`` ASE rescue (also soft for spawn). Sander / Amber LL
``orig`` and ``rescan/itNN`` stages default to **ASE-first** (no geomeTRIC
ladder). With small per-fragment CPU leases, bond pools prefer **breadth**
(concurrent bonds) over a single narrow wavefront; override with
``FFPOPT_PREF_WF_DEPTH=1`` or ``FFPOPT_PREF_WF_BREADTH=1``. Fragmented runs
start ``min(nproc, n_fragments)`` concurrent workers (not one fragment
with all cores) and lease cores only during scan phases (not PrepareInput /
GenDihedFit / compare), set ``OMP_NUM_THREADS=1`` when unset, and warm-start
``itNN`` from the prior LL checkpoint when available. Inside a 1–2-bond
fragment worker, spawn splits are flattened (never bondxwavefront nested).
Fragments with more rotors nest like whole-ligand. HL and
``orig`` scans pipeline in one queue. Under
``--fast``, QDpi2 opts with XTB then full QDpi2 single-point
(``FFPOPT_QDPI2_OPT``), HL nodes MM-relax then one XTB / AIMNet2 / QDpi2
(``FFPOPT_MM_THEN_HL``), and XTB / AIMNet2 / QDpi2 use ASE-first. ``--fast`` remains a
wall-time trade (looser converge, shorter maxiter). Scan ``delta`` stays 10
deg.

**Scan algorithms** (full write-up: Sphinx ``wavefront`` page):

* **Seed coalescing** - one pending job per loc; cheaper parent energy wins.
* **N-D von Neumann neighbors** - axis-aligned only (``moore`` optional).
* **Calculator cache** - keep XTB/sander across serial checkpoints; never
  pickle live sander handles into spawn workers.
* **Reused spawn pool** - sequential bonds in one process share workers.
* **Rigid-rotate seed** - ``RotateMask`` branch twisted by wrapped ``dphi``
  before GeomOpt; clash reverts to the parent Cartesian.
* **MM then HL** - under ``--fast``, sander (or GFN-FF) min at the target,
  then one XTB/QDpi2 refine. Soft-dihed: k-ramp on MM, one HL at final k.
* **Soft-dihed k-ramp** - harmonic ``k`` doubles to 8000, then optional hard IC.

## Dihedral fit chi^2

GenDihedFit's objective is a **shape match**: mean-centered HL-LL residual
(``d = (hl - ll) - mean(hl - ll)``). Independent min-shifts of HL and LL are
not used in chi^2 (plot files may still min-shift for display). Under fixed
geometry, force constants enter linearly and are solved with ridge /
truncated SVD plus an energy-domain ``V(phi)`` barrier. ``|PK|<=25`` is
an Amber-safety valve. Nested ``nprim`` AIC keeps the fewest harmonics
that fit (see Sphinx ``fourier_fit``). After AIC, a chemical-group table
zeros or caps remaining Vptp (alkane 5/20, sulfate 4/10, polar sp3 8/20,
generic sp3 reject 20; amide keeps 30 kcal).

## AFFDO-style extras (opt-in)

Fragmented twist remains the default. For whole-ligand / AFFDO-like runs::

    lig-dihed-correct ... --whole-ligand --multi-centroid 5 \\
        --soft-dihed-restraint --fit-full --fit-backend jax \\
        --boltzmann-charges

| Flag | Behavior |
|------|----------|
| ``--whole-ligand`` | No scission; twist parent rotatable bonds |
| ``--multi-centroid N`` | ConfSearch starts; pick smoothest HL profile (Fourier + roughness). Centroid-0 + ``orig`` share one CPU pool; extra starts only if Fourier RMSE exceeds ``FFPOPT_CENTROID_FOURIER_MAX`` (default 0.5 kcal). |
| ``--soft-dihed-restraint`` | Harmonic dihedral spring (500 kcal/mol/rad^2, +/-0.5 deg) via geomeTRIC + ASE |
| ``--fit-full`` / ``--fit-mode`` | FC + phase + period + scee/scnb (default remains barrier-only) |
| ``--fit-backend jax`` | L-BFGS-B with JAX autodiff (``pip install -e '.[jax]'`` from the clone; conda-forge ``jax`` on HPC) |
| ``--boltzmann-charges`` | Average centroid mol2 charges (when available) |

Wavefront sampling is unchanged when these are off.
