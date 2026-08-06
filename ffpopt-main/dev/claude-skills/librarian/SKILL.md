---
name: ffpopt-librarian
description: Use when writing, building, or publishing ffpopt documentation — covers the Sphinx + Napoleon + RTD-theme setup under docs/, autodoc imports, the GitLab Pages CI publish step, and numpy-style docstring conventions.
---

# Librarian — ffpopt

## Scope
How documentation is structured, built, and published for ffpopt. Sphinx is the toolchain; GitLab Pages is the publish target; docstrings follow NumPy style processed by `sphinx.ext.napoleon`. Does not cover what to *write about* — content lives in the relevant code module's docstring + the `docs/UserDocs/` narrative pages.

## Canonical facts

- **Build toolchain:** Sphinx with the following extensions (`docs/conf.py:17-25`):
  - `sphinx.ext.autodoc` — pulls docstrings from the installed `ffpopt` package (needs `ffpopt` importable; `import ase` and `import ffpopt` at the top of `conf.py`).
  - `sphinx.ext.viewcode`, `sphinx.ext.githubpages`, `sphinx.ext.napoleon`, `sphinx.ext.intersphinx`, `sphinx.ext.todo`, `sphinx.ext.doctest`.
- **Theme:** `sphinx_rtd_theme` with a custom `responsive.css` override (`docs/_static/responsive.css`).
- **Intersphinx targets:** `python` (https://docs.python.org/3), `numpy` (https://numpy.org/doc/stable), `ase` (https://wiki.fysik.dtu.dk/ase) — see `docs/conf.py:44-48`.
- **Page tree:**
  - `docs/index.rst` is the root toctree.
  - `docs/UserDocs/GettingStarted.rst` and `docs/UserDocs/Examples.rst` are the narrative user docs.
  - `docs/UserDocs/Examples/WavefrontExample/tutorial.rst` is the in-depth wavefront tutorial.
  - `docs/API/api_docs.rst` is the API toctree; it lists one `docs/API/documentation/<Module>.rst` per public module (currently: `Options`, `Struct`, `Geometry`, `GeomOpt`, `Reader`, `Dihedrals`, `Constraints`, `Restraints`, `AmberParm`, `Calculator`, `Wavefront`, `ConfSearch`).
- **Docstring style:** NumPy-style (Napoleon). The `napoleon_type_aliases` and `napoleon_preprocess_types` are configured (`docs/conf.py:50-55`) to recognize `numpy.array`, `Constraint`, and `ListOfStruct`.
- **Build commands:**
  - Local: `cd docs && make html` → output under `docs/_build/html/`. The `Makefile` also exposes `make clean` and `make doctest`.
  - CI: see `.gitlab-ci.yml` — `sphinx-build -b html -n docs public` (nitpicky `-n`, output dir is `public/`, which GitLab Pages serves).
- **Publish path:** GitLab Pages. The `pages` job in `.gitlab-ci.yml` runs only on the default branch (`if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH`) and uploads the `public/` artifact. Live URL: https://ffpopt-b083ab.gitlab.io/ (referenced from `README.md:1`).
- **`docs/_build/` is gitignored** (`.gitignore:18`). Do not commit built HTML.
- **Nitpick exceptions** are catalogued in `docs/conf.py:61-81` — including stale references to `GeomOpt`, `ffpopt.GeomOpt.GeomOpt`, `Constraint`, `Dihedral`, `DihedralType`, `parmed.*` (parmed has no public objects.inv), and numpy-shape pseudo-types (`shape=(nat,3)`). When you add a new docstring, do not introduce *new* nitpick warnings; if a new reference is genuinely unresolvable, extend this regex list.

## Conventions

- New public functions/classes get NumPy-style docstrings with `Parameters`, `Returns`, and (for classes) `Attributes` sections. See `src/python/lib/ffpopt/Options.py`, `src/python/lib/ffpopt/Geometry.py`, `src/python/lib/ffpopt/Dihedrals.py` for canonical examples.
- New public modules under `src/python/lib/ffpopt/` get a corresponding `docs/API/documentation/<Module>.rst` file (a single `.. automodule:: ffpopt.<Module>` directive is the common pattern — check existing files before mirroring) and an entry under `docs/API/api_docs.rst` toctree.
- Narrative user docs that describe a workflow live under `docs/UserDocs/Examples/<Workflow>/tutorial.rst` and mirror an `examples/<workflow>/run.sh`.
- `docstring is the docs.` Do not duplicate long argument descriptions in narrative `.rst` files; reference autodoc instead. The current `README.md` is the exception — it is the human-facing readme and intentionally restates many CLI flags.
- For type aliases that Napoleon can't resolve (e.g. `numpy.array` with a `shape=(...)` annotation), prefer fixing the docstring; if not feasible, extend `napoleon_type_aliases` or `nitpick_ignore_regex` in `docs/conf.py`.
- Build doc PRs locally with `make html` to confirm no new warnings — CI runs with `-n` so warnings are loud.

## Anti-patterns

- Do not write Google-style or Sphinx-style (`:param x:`) docstrings. The project is uniformly NumPy-style.
- Do not commit anything under `docs/_build/` or `docs/public/` — gitignored, CI builds fresh.
- Do not introduce a competing static site builder (MkDocs, Docusaurus). The whole flow assumes Sphinx + RTD theme.
- Do not add `sphinx.ext.autosummary` or rebuild the API tree generation. The current `docs/API/api_docs.rst` is a hand-maintained toctree; switching introduces churn without payoff.
- Do not assume `parmed` cross-references resolve — `docs/conf.py:62` already silences them. New `parmed.*` references should follow the same silencing pattern or use the term in plain prose.
- Do not change the published URL or strip the GitLab Pages link from `README.md:1` without coordinating with the maintainer — it's the one canonical source listed in the README.

## Pointers

- Sphinx config: `docs/conf.py`.
- Build entrypoint: `docs/Makefile`.
- Toctree roots: `docs/index.rst`, `docs/API/api_docs.rst`.
- Custom CSS: `docs/_static/responsive.css`.
- CI publish job: `.gitlab-ci.yml`.
- Live site: https://ffpopt-b083ab.gitlab.io/.
- Docstring style exemplars: `src/python/lib/ffpopt/Options.py`, `src/python/lib/ffpopt/Geometry.py`, `src/python/lib/ffpopt/Dihedrals.py:102-160`.
- Subpackage module-docstring exemplar: `src/python/lib/ffpopt/scosmo/__init__.py:1-69`.

## Gaps

- `release` in `docs/conf.py:13` is hard-coded to `'0.1'` while `pyproject.toml:10` says `1.1.0`. Verify which is correct before publishing — they have drifted.
- Several API modules referenced from `docs/API/api_docs.rst` are not in the API directory listing (the toctree shows files like `WaveFront`/`Calculator`/`ConfSearch`); spot-check that each `documentation/<Module>.rst` actually exists before adding new entries.
- README.md (the long-form human doc) is partially out-of-sync with the JSON-first script flow (it still describes some xyz-based `--oscan`/`--iscan` arguments). Recent commits `6980f00` / `f60e8c5` / `ac877a1` were partial fixes; expect ongoing drift between README and the JSON-only CLIs.
- No doctest examples are committed under `docs/`; the `make doctest` target exists but is presumably a no-op today.

---
Last reviewed: 2026-05-19
Owner: piskuliche
