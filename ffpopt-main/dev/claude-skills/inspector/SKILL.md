---
name: ffpopt-inspector
description: Use when touching tests, validation, security, or code-quality gates in ffpopt — covers the absence of automated tests, the example-based smoke-test pattern, secret/credential surface (HuggingFace, AmberTools), input-validation expectations on parmed/mol2 inputs.
---

# Inspector — ffpopt

## Scope
Tests, validation, security posture, and code-quality gates in ffpopt. The headline finding is that there is *no automated test suite, no linter, and no CI quality gate*; validation is driven by the runnable `examples/*/run.sh` scripts and Sphinx doctest. This skill captures what currently exists, the security-sensitive surfaces (credential handling for HuggingFace, AmberTools), and the conservative posture demanded by the fact that the codebase has been used in a published paper. Does not cover code style (TechLead) or release/deploy mechanics (Operator).

## Canonical facts

- **A small opt-in `tests/` directory exists** (`tests/conftest.py` puts `src/python/lib` on `sys.path`; `tests/test_geomopt_watchdog.py`, `tests/test_wavefront_queue.py`). Run with `pytest` from a configured env (e.g. the `ffpopt-tensorflow` conda env, which has `ase`/`parmed`/`geometric`/`pytest`). It is **not** wired into CI (`.gitlab-ci.yml` still builds only docs) and there is no `pytest.ini`/`tox.ini`/`noxfile.py`. These are targeted unit tests for orchestration logic the QM `examples/` can't reach (subprocess watchdog, the wavefront calculation queue), not a comprehensive suite.
- **No linter, formatter, type checker.** `pyproject.toml` has no `[tool.ruff]`, `[tool.black]`, `[tool.mypy]`, `[tool.pyright]` section.
- **CI gate:** `.gitlab-ci.yml` has a single `pages` job that only runs Sphinx HTML build on the default branch. It does not run tests, lint, or builds of any non-doc artifact. The script:
  ```
  pip install --no-cache-dir "numpy<2" scipy matplotlib ase parmed geometric sphinx sphinx_rtd_theme
  export PYTHONPATH="$CI_PROJECT_DIR/src/python/lib:$PYTHONPATH"
  sphinx-build -b html -n docs public
  ```
  Sphinx is invoked with `-n` (nitpicky) — broken cross-references and missing references are warnings; a number of them are pre-silenced via `nitpick_ignore_regex` in `docs/conf.py:61-81`.
- **Smoke tests are the `examples/`.** Each directory under `examples/` contains a `run.sh` plus pinned inputs (parm7/rst7/mol2). The reference outputs are not committed, so re-running requires manual inspection of outputs.
- **No `SECURITY.md`, no `CODEOWNERS`, no `LICENSE`.** Authors are Tim Giese and Zeke Piskulich (`pyproject.toml:13-16`).
- **Secret/credential surface:**
  - **HuggingFace token** for `--group fairchem` models (OMOL25). Acquired via `hf auth login` (interactive). README §"The following methods are available only if you" (README:471-501) is the authoritative install path. The repo itself does not store the token.
  - **`AMBERHOME` env var.** Gates DFTB parameter download (`CMakeLists.txt:36-47`); also implicitly drives sander/pysander discovery. Not a secret, but a *required env* whose absence silently disables features.
  - **`QUICK_BASIS` env var.** Required for QUICK ab initio (`modulefiles/ffpopt:20`, README:103-106).
  - **`PSI_SCRATCH` env var.** Auto-filled to `cwd` if missing/invalid by `GenCalculator` (`ase/calculator.py:255-267`). Writing into cwd may be undesirable on shared filesystems.
- **Input-validation pattern:** library code validates with `raise Exception("...")` and explicit `Path(p).exists` checks (e.g. `Struct.ReadAmberParm:170-185`, `ConfSearch` requires `.json` suffix at `bin/ffpopt-ConfSearch.py:72`). There is no schema for the JSON input format; validation is positional/duck-typed.
- **Multiprocessing posture:** wavefront pools must use the `spawn` start method (commit `af67ce3`) because tensorflow won't initialize with the default `fork`. The wavefront now opens a *local* `multiprocessing.get_context('spawn').Pool` per `calculate()` (rather than the global `set_start_method('spawn', force=True)`) so the repeatedly-invoked twist workflow doesn't churn global start-method state; both satisfy the `spawn` requirement. Never add a `fork`-based path.

## Conventions

- Treat `examples/*/run.sh` as the canonical regression set. Before merging a behavioral change, run the relevant `examples/<name>/run.sh` in a configured env and diff `*.json`, `*.xyz`, and `*.dat` outputs against the prior commit.
- Add an example, not a unit test, when validating a new workflow: a new `examples/<name>/` with a `run.sh` and a small set of `*.parm7`/`*.rst7`/`*.mol2` inputs.
- For doc changes, run `cd docs && make html` (or `sphinx-build -b html -n . _build/html`) before merging. Address any new nitpick warnings either by fixing the reference or extending `nitpick_ignore_regex` in `docs/conf.py`.
- Validate inputs in CLI scripts with bare `raise Exception(...)` and a one-line message, consistent with `bin/ffpopt-DihedScan.py:79` and `Struct.ReadAmberParm`.
- For new env-var dependencies, mirror the `PSI_SCRATCH` pattern (`ase/calculator.py:255-267`): if missing/invalid, write a warning to stderr and fall back to a safe default rather than failing.

## Anti-patterns

- Do not commit credentials, HuggingFace tokens, ORCID keys, ambertools licenses, or model files outside `pkgdata/qdpi/qdpi-2.0.pb` (which is intentionally bundled). All other model weights are pulled at build time by CMake from upstream and must remain external.
- Do not add a *required* test gate or a heavy test-framework dependency for a quick sanity check — the repo's posture is still "no required tests" and CI runs none. Opt-in `pytest` unit tests are appropriate for orchestration logic the QM `examples/` cannot reach (see `tests/test_geomopt_watchdog.py`, `tests/test_wavefront_queue.py`); for a new *workflow*, still prefer adding an `examples/<name>/` over a unit test.
- Do not relax `parm7/rst7` requirements for `--model=sander` or torsion-fitting paths without a regression-grade plan. The sander interface is via pysander, and the charmm calculator has a documented memory-leak bug if reloaded more than once (README:91-100; see Historian).
- Do not introduce a `fork`-based multiprocessing path. Tensorflow's initializer requires `spawn` (commit `af67ce3`).
- Do not add `-Werror`-style CI gates to Sphinx without removing the documented stale references first (see `docs/conf.py:72-81`).
- Do not silently broaden numpy version pins past `<2` in any code path that touches parmed/ambertools (`pyproject.toml:31`, README:80-87).
- Do not write breaking changes carelessly: the README in `SKILLS_CONTEXT.md` notes "This repo has been used for a paper, so major breaking changes should be considered carefully." Maintain backward-compatibility for published example workflows where feasible, or document the migration.

## Pointers

- CI config: `.gitlab-ci.yml`.
- Smoke-test set: `examples/{confsearch,dihedscan,dihedtwistfit,geometric,optimize,resp}/run.sh`.
- Sphinx config (sole quality gate): `docs/conf.py`.
- Input-validation exemplars: `bin/ffpopt-ConfSearch.py:72-73`, `bin/ffpopt-DihedScan.py:78-80`, `Struct.py:ReadAmberParm`.
- Env-var fallback exemplar: `ase/calculator.py:255-267` (`PSI_SCRATCH`).
- Multiprocessing constraint: commit `af67ce3` ("add `multiprocessing.set_start_method('spawn', force=True)`").

## Gaps

- There is no documented coverage target, no reference output for examples, and no pre-merge checklist. Verification is by-eye against prior runs.
- Authentication flow for HuggingFace (`hf auth login`) is not exercised in CI; OMOL25 model paths in `GenCalculator` (`ase/calculator.py:153-169`) would silently fail if the user's token is missing.
- The `industry vs academic` flag in `CMakeLists.txt:6-14, 53-57` is the only gating mechanism for model-redistribution permissions; there is no automated check that an industry user did not bypass it.
- No `SECURITY.md` and no documented vulnerability-report process for what is, in effect, a scientific computing toolkit downloading multiple third-party model artifacts at build time. Track this as a known gap rather than fabricating policy.

---
Last reviewed: 2026-05-28 (tests/ now exists; spawn-context multiprocessing note)
Owner: piskuliche
