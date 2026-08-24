Design philosophy
=================

This page describes how ``ligandparam`` (and its companion packages
``ffpopt`` and ``scission``) is meant to evolve. The goal is a research
codebase that stays **usable as a product** without becoming an
unmaintainable pile of one-off scripts.

Maintainability score
---------------------

**Overall: 9 / 10** (research monorepo after specialty CLI quarantine,
thin public facades, merged 1-D/N-D wavefront, and shared helpers).

This is a judgment call, not a CI metric. It reflects how hard it is for a
new developer (or future-you) to change behavior safely.

Rubric (what the score means)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======= ================================================================
Score   Meaning
======= ================================================================
9-10    Small surface, consistent patterns, god-modules rare, tests
        cover contracts thoroughly
7-8     Clear package boundaries and extension points; some large
        modules remain but helpers are extracted
5-6     Works, but many files mix concerns; changes require tribal
        knowledge
3-4     Fragile; fear of touching core paths
1-2     Effectively unmaintainable without a rewrite
======= ================================================================

What pulls the score **up**
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Package separation of concerns:** ``ligandparam`` owns
  parameterization; ``ffpopt`` owns torsion fitting / wavefront scans;
  ``scission`` owns fragmentation and frcmod merge. Public CLIs map cleanly
  (``lig-getparam``, ``lig-dihed-correct``, ``lig-scission``).
* **Recipe + stage pipeline:** workflows are ordered lists of stages.
  Recipes compose; stages do one job. Shared builders live in
  :mod:`ligandparam.recipes.Common`.
* **Template method on stages:** :class:`~ligandparam.stages.AbstractStage.AbstractStage`
  owns logging / setup / ``new_files`` tracking; thin stages implement
  ``_run``.
* **Write-once helpers:** Gaussian recipe configure, wavefront mixins
  (MP + MPI drain, evaluate policy), ``dihed.math`` / ``ipc_slim``,
  ``runtime/`` (console, CPU budget, non-daemon pools), scission
  ``safe_name`` / DIHE key helpers.
* **Thin public facades:** ``Dihedrals``, ``WaveFront``, and
  ``WaveFrontND`` re-export a single implementation (fit types / Fourier /
  ParmEd / solvers; one ``Wavefront`` class in ``WavefrontEngine``).
  Pickle and CLI import paths stay stable.
* **Two deliberate test entry points:** install validation for users;
  developer regression for wiring and pure helpers (no AmberTools /
  Gaussian required for most of that suite).
* **Quarantined specialty CLIs:** sugar/pucker, JSON, and animate tools
  go through ``ffpopt-specialty`` so the default install map matches the
  product path.
* **Documented periphery:** ``sqmligand``, Sage (``lig-to-sage``), and
  CPE/RespFit stay secondary-supported rather than ambiguous.

What pulls the score **down**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **God modules** still dominate a few files by line count:
  :mod:`ffpopt.geom.GeomOpt` and :mod:`ligandparam.multiresp.ParmHelper`
  keep large public surfaces even after ASE / parallel / job-script
  helpers were extracted. Wavefront and dihedral public filenames are
  now thin re-export facades.
* **Control-flow-heavy stages** still override ``execute`` end-to-end
  (Gaussian rotation, DeepMD, dihed twist). Thin stages use ``_run``.
* **Scientific coupling:** many stages need RDKit / ParmEd / Gaussian /
  AmberTools at import or runtime, so "unit test everything" is
  unrealistic without heavy mocking. Tests therefore target contracts
  and pure helpers first.
* **Checkpoint / pickle compatibility:** wavefront loaders register
  historical ``ffpopt.WaveFront`` / ``ffpopt.WaveFrontND`` names onto the
  ``scan`` facades, and new dumps live on ``WavefrontEngine``.

Toward 10/10
~~~~~~~~~~~

* Continue extracting **shared loops and policies** into mixins/runtime
  instead of splitting mega-files into dozens of tiny modules.
* Keep growing developer tests around recipe builders, wavefront policy,
  and merge helpers whenever those areas change.
* Shrink remaining GeomOpt / ParmHelper hot paths only when a helper has
  a clear name and tests.

Principles we follow
--------------------

These are **working rules**, not slogans. When two principles conflict,
prefer the one that keeps the **public CLI and recipe APIs stable**.

SOLID
~~~~~

**S - Single responsibility**

* A **stage** does one pipeline step (initialize, RESP, Leap, ...).
* A **recipe** only assembles and runs stages.
* **scission** fragments and merges; it does not fit torsions.
* **ffpopt** fits / scans torsions; it does not own RESP recipes.

Large modules (``GeomOpt``, ``Workflows``) are tolerated when they
represent one scientific domain and extracting would create many tiny
files with worse navigation. Prefer **internal helpers** over file
explosion.

**O - Open/closed**

* New parameterization workflows should **compose** stage builders in
  :mod:`ligandparam.recipes.Common` (or add a small builder) rather than
  copy-paste a 150-line ``setup()``.
* New wavefront policy belongs in :mod:`ffpopt.scan.WavefrontMixins`
  (or ``runtime/``), not duplicated across 1-D and N-D call paths.

**L - Liskov substitution**

* Any :class:`~ligandparam.stages.AbstractStage.AbstractStage` subclass
  must be runnable via ``Driver.execute`` / ``Recipe.execute``.
* Stages that cannot use the base template must still honor the
  ``execute(dry_run=..., nproc=..., mem=...)`` contract.

**I - Interface segregation**

* Recipes do not require DeepMD, Gaussian paths, or dihedral options
  unless they opt in (``configure_gaussian_recipe(..., with_dihed=...)``,
  etc.).
* CLIs expose only what that entry point needs.

**D - Dependency inversion**

* High-level workflows depend on stage/recipe abstractions and small
  helpers, not on copying low-level Gaussian argv construction.
* External tools (Gaussian, AmberTools) are wrapped behind interfaces /
  stage boundaries so the recipe graph stays readable.

DRY (Don't Repeat Yourself)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Shared recipe chunks: :mod:`ligandparam.recipes.Common`.
* Shared recipe ``__init__``: ``configure_gaussian_recipe``.
* Shared wavefront IPC / soft-opt / drain loop:
  :mod:`ffpopt.scan.WavefrontMixins`.
* Shared console / CPU / pool helpers: ``ffpopt.runtime.*``.

Write once. Prefer a slightly larger helper over three near-copies.

KISS (Keep It Simple)
~~~~~~~~~~~~~~~~~~~~~

* Prefer **fewer meaningful files**. Do not split a coherent 2k-line
  domain file into twenty 100-line files "for SOLID" if navigation gets
  worse.
* Prefer obvious functions over deep inheritance hierarchies.
* Keep pickle-compat aliases as thin re-exports - do not invent a plugin
  system for checkpoint loading.

YAGNI (You Aren't Gonna Need It)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Do not add unused recipe stubs, empty stages, or "future" APIs that
  raise ``NotImplementedError`` in ``__init__``.
* Do not introduce heavy DIP interfaces / DI containers unless a second
  real backend appears.
* Do not document or export specialty tools as core product unless they
  are on a supported CLI path.

Separation of concerns
~~~~~~~~~~~~~~~~~~~~~~

==================== ==================================================
Layer                Responsibility
==================== ==================================================
CLI (``lig-*``)      argv, logging, banner, call into recipes/workflows
Recipes              assemble stage lists; option defaults
Stages               one external or data step
``ligandparam.io``   file formats, orientations, leap/gaussian I/O
``ffpopt.runtime``   process/console/CPU/pool cross-cuts
``ffpopt.scan``      wavefront engines + shared mixins
``ffpopt.workflows`` high-level twist / fragmented orchestration
``scission``         fragment selection, writers, frcmod merge
Tests                install gate vs developer regression (two suites)
==================== ==================================================

Crossing these boundaries (e.g. embedding a full wavefront loop inside a
RESP stage) is a design smell. Crossing via a **thin stage adapter**
(``StageDihedTwistCorrection`` calling ``Workflows``) is fine.

How to extend safely
--------------------

1. **New recipe:** register it; build ``setup()`` from
   :mod:`ligandparam.recipes.Common` builders; rely on
   :class:`~ligandparam.Parametrization.Recipe` ``execute`` logging.
2. **New stage:** subclass ``AbstractStage``, implement ``_run`` unless
   you need custom control flow; keep constructor kwargs explicit.
3. **Wavefront behavior change:** change mixins / policy helpers first;
   keep 1-D and N-D engines as thin orchestration.
4. **Tests:** add a developer regression for wiring/contracts; keep
   install validation focused on imports and CLI entry points.

Related pages
-------------

* :doc:`overview` - repository layout and product flow
* :doc:`recipes` - recipe catalog
* :doc:`stages` - stage catalog
* :doc:`ffpopt` / :doc:`scission` - companion packages
* :doc:`cli` - supported entry points
