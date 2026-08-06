---
name: ffpopt-techlead
description: Use when writing or reviewing Python code in ffpopt — covers naming, file layout under src/python/lib/ffpopt, lazy-import discipline, argparse-driven CLI shape, numpy-style docstrings, error handling, and JSON-first persistence.
---

# TechLead — ffpopt

## Scope
How to make new code blend in with the existing ffpopt codebase. Covers naming, file/module layout, the argparse-only CLI convention, lazy-import discipline, numpy-style docstrings, and the JSON-first persistence model. Does not cover what the system does (Architect), tests/security (Inspector), or external-API quirks (Integrator).

## Canonical facts

- **Language:** Python 3 only on the application side (target 3.12 in practice; `requires-python = ">=3.8"` in `pyproject.toml:11`). One Fortran 90 program at `src/resp/resp.f` (no other Fortran or C code). No type annotations are used in most modules — duck typing is the norm.
- **Module naming** is **PascalCase** (`Struct.py`, `GeomOpt.py`, `WaveFront.py`, `Workflows.py`) for top-level modules under `src/python/lib/ffpopt/` — including new files; lowercase only for subpackages (`ase/`, `cpefit/`, `confsearch/`, `scosmo/`, `constants/`, `pkgdata/`). Subpackage modules are typically PascalCase again (e.g. `cpefit/Molecule.py`, `scosmo/CosmoAtom.py`) — match the surrounding directory. (Exceptions: `scripts.py` is a legacy dispatcher; the `pkgdata/` and `constants/` subpackage files like `Conversions.py` follow the same PascalCase rule.)
- **Function and class naming** is `PascalCase` for classes (`ConstraintList`, `RestraintList`, `Wavefront`) and a mix for functions — historical ffpopt code uses `PascalCase` (`GeomOpt`, `FwdRevDihedScan`, `AddStandardOptions`, `parmed2ase`), but the newer Python kwargs-API entry points use `snake_case` (`run_dihed_wavefront`, `run_dihed_twist_workflow`) to signal "library-grade, kwargs-driven, importable". Keep that distinction: argparse-flavored helpers can stay PascalCase; new public Python APIs should be snake_case. Module-private helpers begin with `_` (e.g. `Geometry._cross`, `Workflows._run_one_scan`).
- **CLI script names** are `ffpopt-<Action>.py` (kebab between the prefix and the action; the action itself is `CamelCase`). Files live under `src/python/bin/`. Each script:
  1. starts with `#!/usr/bin/env python3`,
  2. opens with `if __name__ == "__main__":`,
  3. imports inside that block,
  4. builds an `argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="...")`,
  5. calls `AddStandardOptions(parser)` for any geometry/model surface,
  6. reads input via `ListOfStruct.from_file(args.inp)` and writes output via `outs.save(args.out)`.
- **Lazy imports.** Heavy/optional dependencies (`torch`, `torchani`, `mace`, `aimnet`, `fairchem`, `fennol`, `tblite`, `psi4`, `pyscf`, `parmed`, even `ase.io`) are imported *inside* the function or class method that uses them. See `src/python/lib/ffpopt/GeomOpt.py:23-…`, `src/python/lib/ffpopt/ase/calculator.py:GenCalculator.__init__`, `src/python/lib/ffpopt/Struct.py:from_parmed`.
- **Docstrings** are NumPy-style (`napoleon` is configured in `docs/conf.py:21`). The `Options.py` / `Geometry.py` / `Dihedrals.py` / `Constraints.py` / `GeomOpt.py` modules are the canonical style examples: `Parameters / Returns / Attributes` sections with `name : type` headers, `, optional` after the type for optional kwargs, descriptions ending with `Default is X.`, and a leading space after the opening `"""`. New public functions are expected to follow this format; verbose discussion (phases, output layout, re-running semantics) belongs in the `.rst` page under `docs/API/documentation/`, not in the docstring.
- **No module-level docstrings.** ffpopt modules under `src/python/lib/ffpopt/` (and its subpackages) start directly with imports — 16/19 modules have no module docstring (`GeomOpt.py`, `Struct.py`, `WaveFront.py`, `Options.py`, …). Module overviews live in the `.rst` page that points at the module via `automodule`, not at the top of the `.py` file. The same rule applies to new modules.
- **Argparse conventions.** Use the splat continuation style:
  ```
  parser.add_argument \
      ("-i", "--inp",
       help="...",
       required=True,
       type=str)
  ```
  Default values appear in the `help=` text (e.g. `"... Default: 10"`); the `default=` and the help string must agree. Boolean options use `action='store_true'`.
- **Energy units.** Internal ASE-calculator energies are eV; bin scripts that output kcal/mol convert via `KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()` (see `bin/ffpopt-DihedScan.py:95`). Constants are *callables* in `ffpopt.constants.Conversions` — note the trailing `()` when using them.
- **Logging.** No logging framework is used in library code other than to instrument geomeTRIC subprocesses. `WaveFront.py:32-46` attaches a `ShowOriginFilter` to every existing `StreamHandler` so that geometric/ASE log records announce their origin. `Options.configure_geometric_logging` resolves a packaged `geometric_log.ini` (`pkgdata/files/geometric_log.ini`). Regular code uses `print` to stdout/stderr.
- **No concurrency abstractions** beyond `multiprocessing.Pool` with the `spawn` start method (commit `af67ce3` — tensorflow won't initialize correctly otherwise). Prefer a *local* spawn context — `multiprocessing.get_context('spawn').Pool(...)` — over the global `set_start_method('spawn', force=True)` for functions that may run repeatedly in one process (e.g. `WaveFront.Wavefront.calculate`, which `Workflows.run_dihed_twist_workflow` calls many times); both keep the required `spawn` semantics. For dynamic, non-barrier scheduling use `pool.apply_async` + polling `ready()`/`get()` in the main process (the wavefront calculation queue), not callbacks/threads. Run `--parallel` paths through `Pool`; do not introduce threading or asyncio.
- **Python kwargs-API pattern** (used by `run_dihed_wavefront` in `WaveFront.py` and `run_dihed_twist_workflow` in `Workflows.py`): explicit `*, kw1, kw2=…, **standard_kwargs` signature where `**standard_kwargs` accepts anything declared by `Options.AddStandardOptions`. The library function:
  1. instantiates a throwaway `argparse.ArgumentParser(add_help=False)` + `AddStandardOptions` and calls `parser.parse_args([])` to derive defaults for missing standard kwargs (single source of truth — defaults stay in `Options.py`);
  2. rejects unknown standard kwargs with `TypeError("function got unexpected keyword argument(s): [...]")`;
  3. synthesizes a `SimpleNamespace` so existing internal code (`los.SetArgs(args)`, `args.model`, etc.) keeps working unchanged.
  The matching bin script keeps its full argparse and ends with `func(**vars(args))` — same parser, same defaults, no duplication.
- **Spawn-safety guard for API functions** that internally use `multiprocessing.Pool` (`run_dihed_wavefront`): inspect the caller's frame globals (`sys._getframe(1).f_globals['__name__']`) and raise `RuntimeError` if it equals `'__mp_main__'` (a spawn worker is re-importing an unguarded entry script). Cheaper than letting `_check_not_importing_main` throw its cryptic traceback. Apply the same pattern to any new API that internally spawns workers.
- **Style/lint configs**: none. There is no `ruff.toml`, `pyproject.toml [tool.ruff]`, `.flake8`, `pre-commit-config.yaml`, or `mypy.ini` in the repo. Existing code is hand-formatted with 4-space indents and 80–100 column lines, mixed.

## Conventions

- Match the *case and spacing* of the file you're editing rather than rewriting it. The codebase is internally consistent per-file but not globally consistent.
- When you add a new CLI: register entry-point shims in both `pyproject.toml [project.scripts]` and `src/python/lib/ffpopt/scripts.py` (`def ffpopt_<Name>(): _run_bin_script("ffpopt-<Name>.py")`) so `pip install` produces a launcher; ensure the file is `chmod +x`.
- When you add a kwargs-API counterpart to a CLI (so callers can drive in-process), put it next to the relevant class — e.g. `run_dihed_wavefront` lives in `WaveFront.py` alongside the `Wavefront` class; multi-step orchestration goes in `Workflows.py`. Don't duplicate the argparse — the bin script's argparse stays and ends with `func(**vars(args))`.
- When adding new options for an existing CLI, prefer extending `AddStandardOptions`/`AddGeomOptOptions`/`AddModelOptions` in `Options.py` if the option is shared, so all scripts pick it up uniformly.
- `Struct`/`ListOfStruct` is the only shared in-memory data type for atomic configurations. Functions that take ASE `Atoms` should accept them via `Struct.GetASEAtoms()`; do not pass `ase.Atoms` instances across CLI boundaries.
- Use `pathlib.Path` for path manipulations (`Path(args.out).suffix`, `Path(args.out).with_suffix(".dat")` — see `bin/ffpopt-DihedScan.py:105`).
- For Amber atom-mask formatting use the `@<index+1>` (1-based) convention — see `Dihedrals.DeleteDihedrals`/`ChangeDihedrals` and `parmed.tools.actions.deleteDihedral`. Internal indices remain 0-based; convert at the parmed boundary.
- Add `from collections import defaultdict as ddict` if you need a `defaultdict` — the codebase uses that alias consistently.
- Reuse `parmed2graph` / `bonds2graph` in `AmberParm.py` to obtain a `GraphSearch` for connectivity queries; do not roll a new graph type.

## Anti-patterns

- Do not hoist optional-dependency imports to module top. Importing `torch` (or any ML SDK, `psi4`, `tblite`, `pyscf`, `mace`, `aimnet`, `torchani`, `fairchem`, `fennol`) at module scope breaks installs where that group is intentionally absent.
- Do not introduce a *second* CLI option-injection helper alongside `AddStandardOptions`. Extend `Options.py` instead.
- Do not invent a parallel persistence format. Add fields to `Struct.data` and document them; convert at the edge via `ffpopt-Json2Crds.py` or `Struct.SaveCrds`.
- Do not require asyncio or threading. The only parallelism in the codebase is `multiprocessing.Pool` with `spawn` start method.
- Do not silently catch and discard exceptions in new code without leaving the error reachable; `except: pass` patterns appear in `GeomOpt`/`Struct` for *backwards-compat shims* (e.g. `parm.remake_parm()` may not exist on every parmed version) — do not add new ones for new logic.
- Do not write ad-hoc lint-style code reorganizations in PRs that don't otherwise need them — the repo has no formatter, so cosmetic churn obscures the real change.
- Do not use the `RuntimeError`/`ValueError`/`TypeError` distinction strictly — the codebase raises bare `Exception(...)` (e.g. `GenCalculator` at `ase/calculator.py:291`, `Struct.ReadAmberParm:178-184`). Match that style unless you have a specific reason to refine.

## Pointers

- Style exemplar (numpy docstrings, argparse continuation, lazy import): `src/python/lib/ffpopt/Options.py`, `src/python/lib/ffpopt/Geometry.py`.
- CLI exemplar: `src/python/bin/ffpopt-DihedScan.py` (full read → compute → write JSON + companion `.dat`).
- Lazy import exemplar: `src/python/lib/ffpopt/GeomOpt.py:GeomOpt_ASE`.
- Subpackage-style exemplar: `src/python/lib/ffpopt/scosmo/__init__.py` (re-exports flat names; long top-of-file module docstring).
- Entry-point dispatch: `src/python/lib/ffpopt/scripts.py`.
- Sphinx Napoleon config: `docs/conf.py:50-81` (note: GAFF/parmed/numpy-shape strings are silenced via `nitpick_ignore_regex` — keep doc strings consistent with existing patterns).

## Gaps

- No formal style guide, formatter, or linter is configured. Conventions above are inferred from majority-vote across existing files.
- Type annotations: a few modules type-hint (`WaveFront.py` uses `typing.Generator, Optional`, `contextmanager`), but most do not. There is no policy.
- Test pattern is undefined (no `tests/` directory or `pytest.ini` — see Inspector).
- Some modules (`Reader.py`, `Restraints.py`, parts of `Dihedrals.py`) carry large blocks of commented-out legacy code (e.g. `StandardArgs` class in `Options.py:172-313`). Leave them in place unless the PR is specifically a cleanup; commit history shows they were intentionally retained during the JSON refactor (`71f2f6a`).

---
Last reviewed: 2026-05-28 (spawn-context Pool + apply_async queue concurrency note)
Owner: piskuliche
