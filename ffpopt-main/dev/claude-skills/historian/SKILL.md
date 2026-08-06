---
name: ffpopt-historian
description: Use when you're about to reinvent something or question a weird choice in ffpopt. Covers the JSON refactor (commit 71f2f6a), the AmberTools/parmed/numpy<2 constraints, the multiprocessing spawn requirement, and notable reversed decisions captured in commit messages.
---

# Historian — ffpopt

## Scope
Why ffpopt looks the way it does. There is one committed ADR so far
(`decisions/0001-wavefront-calculation-queue.md`) and no `CHANGELOG.md`; most
history is uncaptured. This skill collects the *load-bearing* historical decisions reconstructed from the git log and surviving comments in the code so a future change doesn't accidentally undo them.

## Canonical facts

- **JSON-first data contract (commit `71f2f6a`, "changed underlying data structures. major change to all library components and scripts"):** the entire bin/lib was refactored to thread `Struct`/`ListOfStruct` JSON objects between scripts instead of xyz/extxyz. Follow-ups `f60e8c5`, `ac877a1`, `6980f00` updated the docs to reflect the new format. Treat any pre-`71f2f6a` xyz-based pattern in code comments (`bin/ffpopt-Optimize.py:73-96` and the commented-out `StandardArgs` class in `Options.py:172-313`) as superseded; do not revive them.
  - **Refactor was incomplete in places — expect dangling renames inside function bodies.** `71f2f6a` renamed `StandardArgs stdargs` → `ListOfStruct los` and `ase.Atoms atoms` → `Struct struct` across signatures, but several *internal* references kept the old names and only fail when the code path is executed. Known landings fixed post-hoc in `WaveFront.py` (2026-05-19): `restart_options` `node.stdargs = stdargs` → `node.los = los`; `init_conformer` `self.stdargs.graph` → `self.struct.GetGraph()`, `self.init_check(..., self.stdargs, ...)` → drop the bogus positional, `GeomOpt(new_atoms, self.stdargs, ...)` → `GeomOpt(self.los, new_atoms, ...)`, and the two recursive-retry calls `self.init_conformer(atoms, ...)` → `self.init_conformer(struct, ...)`. If a Wavefront / conformer path raises `NameError`/`AttributeError` on `stdargs` or `atoms`, the fix is almost always a missed rename of this kind — grep the file for `self.stdargs` and bare `atoms` references before assuming a deeper bug.
- **`numpy<2` is a hard constraint** (`pyproject.toml:31`, README:80-87). Reason: `parmed` (and therefore `ambertools`, `pysander`, charmm action) requires `numpy<2`. The psi4 conda package installs `numpy==2.x`, so the workaround is to keep psi4 in a *separate* conda env and rely on `PYTHONPATH` ordering — see README:80-87 and `pyproject.toml` comments at 24-27.
- **Industry vs academic gating (`CMakeLists.txt:6-14, 53-57`):** the default install path assumes the user is *industrial* and therefore disables `USE_FENNOL`, `USE_MACE`, `USE_PM6ML`. Academic users must opt in with `ACADEMIC=TRUE`. This is because the upstream models are not redistributable without permission. Do not flip the default; do not bypass the gate to "make CI green."
- **`dacase::ambertools-dac=25` is required, not optional (README:91-100):** the unofficial conda `ambertools` package has a charmm-module memory-leak/segfault when the calculator is reloaded more than once. Build/test infrastructure must pin to `dacase::ambertools-dac`. There is no in-repo fork of charmm; downstream consumers must use this exact channel.
- **`multiprocessing.set_start_method('spawn', force=True)`** before launching the wavefront pool (commit `af67ce3`). Reason: tensorflow won't initialize correctly under `fork` (the default on Linux), because the wavefront optimizer launches a new process with model state. Do not change to `fork`.
- **Fennol model download switched from `git clone` (LFS) to direct URL (`CMakeLists.txt:71-123`, commit `b6fa841`):** fennol-pmc uses Git LFS, which rate-limits downloads to a few per month. The old `FetchContent_Declare` path is preserved under `if(FALSE)` for reference but not used. Do not re-enable the LFS path even if it looks more idiomatic.
- **DFTB2/DFTB3 parameters are downloaded *into AmberTools*, not the wheel (`CMakeLists.txt:271-275, 311-315`, commit `d751698`):** the mio-1-1 and 3ob-3-1 skfiles are missing from AmberTools but are required to run dftb2/dftb3 via pysander's sqm. `AMBERHOME` must point at a writable AmberTools install for this to succeed. This is unusual for a Python wheel; it is intentional.
- **mopac files are written with randomized prefixes and cleaned up after use** (commits `e55517c`, `f44fb2a`). Reason: mopac writes many temp files in the cwd; without isolation, parallel runs collided.
- **`--cpu` flag and JAX/CUDA env vars (`Options.py:103-106`, commit `5980dc8`):** added because some ML models initialize a GPU even when one is unavailable or undesirable; `--cpu` sets `JAX_PLATFORMS='cpu'` and `CUDA_VISIBLE_DEVICES='-1'`.
- **Fennix-bio1m/s and OMOL25 were added in `5980dc8` / `702cf05`** as new `--model` values; `OMOL25-*` requires the fairchem dependency group + HuggingFace auth (README:471-501).
- **`--geometric-ini=''` is a valid option** that disables file-based logging and prints geometric output to screen (`Options.argparse2geometric` at `Options.py:419-421`, commit `5980dc8`).
- **`PSI_SCRATCH` auto-default (`ase/calculator.py:255-267`):** if missing/invalid, the Psi4 calculator sets it to `cwd` with a stderr warning rather than failing — keeps single-user runs working without per-machine setup.
- **GitLab Pages publish step (`f60e8c5`, `31a2e28`):** docs publish was added relatively late; the link at `README.md:1` was finalized in `9c7a416`. The site URL is canonical.
- **No tagged releases / no semver discipline.** `pyproject.toml:10` says `1.1.0`, `docs/conf.py:13` says `0.1`; this mismatch is a *known drift* (see Librarian Gaps).

## Conventions

- When you read a heavily commented-out block (`Options.py:172-313`, parts of `Reader.py`, `Restraints.py`, `Dihedrals.py`, `Struct.py`), assume it was retained intentionally during the JSON refactor (`71f2f6a`). Do not delete in unrelated PRs; touch only if the PR is explicitly a cleanup.
- When in doubt about why a constraint exists, search the git log with `git log --oneline -- <path>` and read the commit message — they are unusually descriptive in this repo (e.g. `af67ce3`, `5980dc8`, `b6fa841`).
- New decisions belong in `decisions/NNNN-<slug>.md` (see `decisions/README.md` for the ADR template). If the question is "why is X this way", and X is load-bearing across modules, write the decision down rather than annotating the code.

## Anti-patterns

- Do not revive the xyz/extxyz interchange format. The JSON refactor (`71f2f6a`) is the supersedes-all decision.
- Do not pin `numpy>=2` to "modernize" — every parmed/ambertools/sander/dftb path will break.
- Do not switch the default `ACADEMIC` to TRUE. The current gating is the licensing safety mechanism.
- Do not replace `multiprocessing.set_start_method('spawn', ...)` with `fork`. Tensorflow's optimizer will silently misbehave.
- Do not switch the fennol model download back to `git clone` / LFS. Rate-limited, intentionally rewritten in `b6fa841`.
- Do not assume `--model` defaults to anything other than `sander`. The pre-JSON xyz flow allowed mol2-only inputs for most ML models; that behavior is preserved but the sander default is intentional for amber-torsion workflows.
- Do not drop the `dacase::ambertools-dac=25` pin in `environment.yml` — the unofficial `ambertools` channel has a known charmm-module memory-leak bug.

## Pointers

- ADR directory: `decisions/` (`README.md`, `_template.md`, and
  `0001-wavefront-calculation-queue.md`).
- Most informative commits (read full messages):
  - `71f2f6a` — JSON data-structure refactor (the largest behavioral inflection point).
  - `af67ce3` — multiprocessing spawn + mol2 wavefront support.
  - `b6fa841` — fennol model download rewritten to avoid LFS rate-limit.
  - `5980dc8` — `--cpu`, fennix-bio1m/s, mopac net-charge fix, `--geometric-ini=''` behavior.
  - `d751698` — DFTB2/DFTB3 parameter download into AmberTools.
  - `e55517c`, `f44fb2a` — mopac file isolation/cleanup.
  - `702cf05` — orb-v3 ML models added.
  - `eeab3ab` — installation procedure overhaul.
- README notes block with operational rationale: `README.md:66-137` (the "NOTES" section).
- Surviving legacy code (do not delete in unrelated PRs):
  - `src/python/lib/ffpopt/Options.py:172-313` (commented-out `StandardArgs` class).
  - `src/python/bin/ffpopt-Optimize.py:73-96` (commented-out xyz output path).
  - `src/python/lib/ffpopt/Reader.py`, `Restraints.py`, `Dihedrals.py` — interleaved comment blocks from pre-refactor logic.

## Gaps

- `decisions/` has one entry (`0001-wavefront-calculation-queue.md`); every other fact above was reconstructed from commit messages and inline comments. Promote the most load-bearing ones (numpy<2, multiprocessing spawn, ACADEMIC gating) into actual ADRs.
- No CHANGELOG.md; semantic version `1.1.0` vs docs `0.1` mismatch suggests version tagging is informal.
- Earlier history may be in a private commit prior to `28d23ea` ("initial import"); that initial import is opaque and may itself contain rationale that has been lost.
- The note at `README.md:817-826` (HTML-commented "A simple installation..." block) implies a non-conda dev install that may no longer match `pyproject.toml`/`CMakeLists.txt`; treat as undocumented.

---
Last reviewed: 2026-05-28 (first ADR committed: 0001 wavefront calculation queue)
Owner: piskuliche
