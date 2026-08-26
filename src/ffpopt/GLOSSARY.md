# GLOSSARY - ffpopt

> Canonical definitions for domain terms used in this repo. The aim is
> *consistency*: one definition per term, one place to update it when the
> meaning shifts.

## What goes in

- Domain terms with non-obvious meaning (a reader who has not worked on
  ffpopt for six months should reach for this file).
- Terms used in three or more docs or modules with a specific repo meaning
  that diverges from the generic English meaning.
- Cross-cutting concepts that appear in `dev/claude-skills/` skill files,
  README, and code identifiers - anchor them once here, point everywhere else.

## What does NOT go in

- Private internal names (variables, classes) - those live in the code.
- Generic programming concepts (database, queue, retry) unless this repo uses
  them in a non-standard way.
- Per-team jargon that belongs in a chat channel description, not here.

## How to add an entry

Use the shape below. Lead with the name. One-sentence definition. Add detail
only when the one-sentence form is genuinely ambiguous or invites misuse.
Always cite an authoritative source - a file path plus section.

```markdown
### TermName

**Definition.** One-sentence definition.

**Detail.** Optional longer explanation, including when the term applies and
common confusions with similar terms.

**Authoritative source.** `<path>:<section or line range>`
```

## Platform primitives

### Wavefront scan

**Definition.** Parallel relaxed dihedral scan that expands angular nodes
until neighboring energies agree within a threshold.

**Detail.** Replaces the older sequential forward/reverse `DihedScan` path in
modern twist workflows. One implementation (`ffpopt.scan.WavefrontEngine`)
with 1-D / N-D facades (`WaveFront`, `WaveFrontND`). Neighbors, spawn, and
recovery follow `evaluate_wavefront_minimum`. Speedups: seed coalescing
(one pending job per loc), N-D von Neumann stencil, persistent calculator
cache (restored after checkpoint; never pickled into spawn workers), reused
spawn pools, flattened fragment `nproc`. Soft-dihed k-ramp is `[affdo]`;
scan lifecycle lines are `[wavefront]`. New pickles are
`ffpopt.scan.WavefrontEngine.Wavefront`; loaders still map historical
`ffpopt.WaveFront` / `ffpopt.WaveFrontND` names.

**Authoritative source.** `src/ffpopt/scan/WavefrontEngine.py`; Sphinx
`docs/source/documentation_pages/wavefront.rst`

### Seed coalescing

**Definition.** At most one pending wavefront job per grid location; a
cheaper parent energy replaces the queued seed, or is deferred if that loc
is already in flight.

**Authoritative source.** `src/ffpopt/scan/WavefrontEngine.py` (`_enqueue_visit`)

### Rigid-rotate seed

**Definition.** Cartesian twist of the `RotateMask` branch by wrapped `dphi`
before GeomOpt; a clash or broken bond keeps the parent coordinates.

**Detail.** Neighbor nodes still copy the parent Cartesian. Without this,
geomeTRIC slams a large TRIC step (e.g. 11 deg vs 250 deg). The frozen
constraint or soft restraint is unchanged.

**Authoritative source.** `src/ffpopt/scan/WavefrontMixins.py`
(`seed_struct_rigid_dihed_rotates`)

### MM then HL

**Definition.** Cheap constrained min (sander or GFN-FF) at the scanned
angle, then one high-level (XTB / QDpi2) refine from those coordinates.

**Detail.** Default on under `--fast` for non-MM models (`FFPOPT_MM_THEN_HL`).
Soft-dihed runs the k-ramp on MM and does one HL opt at the final k or after
the MM hard IC. Stored node energies are always the HL values.

**Authoritative source.** `src/ffpopt/scan/WavefrontMixins.py`
(`geomopt_mm_then_hl`, `run_soft_dihed_opt`)

### Von Neumann stencil

**Definition.** N-D neighbor set of axis-aligned bins only (`2 * ndim`),
enough to fill the grid. Contrast Moore (`3**ndim - 1`, includes diagonals).

**Authoritative source.** `src/ffpopt/scan/WavefrontEngine.py` (`GetGridNeighbors`)


### Twist workflow

**Definition.** Iterative loop that scans a torsion at high level (HL) and
with the current Amber force field (LL), fits cosine force constants, applies
them, and rescans until HL~=LL or `maxiter` is reached.

**Detail.** Exposed as `run_dihed_twist_workflow` (single molecule) and
`run_fragmented_dihed_twist_workflow` (scission fragments + merge).

**Authoritative source.** `src/ffpopt/workflows/`

## Domain concepts

### Dihed twist / torsion correction

**Definition.** Refitting Amber proper-dihedral force constants so MM energy
profiles match a chosen high-level model along rotatable bonds.

**Detail.** Does not rewrite atomic charges or the Amber `.lib`; only DIHE
(and related) terms are updated, typically into a new `.frcmod`.

**Authoritative source.** `src/ffpopt/workflows/` (package docstring)

### HL scan / LL scan

**Definition.** High-level (HL) is the target chemistry (`--model`, e.g.
`qdpi2`); low-level (LL) is the current Amber force field evaluated with
`sander`. Light HL options without DeepMD: `xtb` (tblite) and `aimnet2`
(neural net, extra `[aimnet]`).

**Detail.** Fit residuals are mean-centered HL-LL energy profiles along the
scanned dihedral (shape match). Convergence heuristics can drop bonds that
already match.

**Authoritative source.** `src/ffpopt/workflows/DihedTwist.py:run_dihed_twist_workflow`

### Shape-match chi^2

**Definition.** Dihedral-fit objective `d = (hl - ll) - mean(hl - ll)` with a
free vertical offset (not independent HL/LL min-shifts).

**Detail.** Under fixed geometry, FCs enter linearly and are solved with
bounded `lsq_linear`. Phase is kept at 0 for this pass.

**Authoritative source.** `src/ffpopt/dihed/DihedMath.py` (`shape_match_delta`);
`src/ffpopt/dihed/DihedFitSolve.py` (`NonlinearSolve`)

### bytype (global) vs bespoke parameters

**Definition.** `bytype=True` fits one potential per atom-*type* quartet and
writes it to a frcmod; bespoke (`masks: null`) maps potentials to atom-*name*
quartets via a Parmed Python patch script.

**Detail.** Fragmented / scission merges **require** `bytype=True` because
fragment atom names do not exist in the parent topology.

**Authoritative source.** `src/ffpopt/workflows/FragmentedTwist.py:run_fragmented_dihed_twist_workflow`

### nprim

**Definition.** Number of cosine primitives (periodicities 1...nprim) fitted
per torsion parameter family.

**Detail.** Default is 3 (periods 1, 2, and 3). Nested AIC keeps the
smallest ``k`` in the AIC window unless ``FFPOPT_DIHED_NPRIM_SELECT=0``.

**Authoritative source.** `src/ffpopt/dihed/DihedFitTypes.py` (`nprim` on `ParamType`); `docs/source/documentation_pages/fourier_fit.rst`

### Fourier ridge (dihedral FC)

**Definition.** Tikhonov / truncated-SVD solve for Fourier force constants,
then an energy-domain cap on reconstructed ``V(phi)``. ``FFPOPT_DIHED_FC_MAX``
is an Amber-safety valve only.

**Detail.** Unbounded least squares invents huge cancelling harmonics on
gappy or non-torsional residuals. Ridge picks the unique small-K series;
dense-grid ``V(phi)`` peak-to-peak is capped (default 30 kcal/mol).
After AIC, a chemical-group table zeros or caps remaining ``V(phi)``
(alkane 5/20, sulfate/phosphate 4/10, polar sp3 8/20, generic sp3 reject
20). Unsaturated (amide) types keep the 30 kcal ceiling.

**Authoritative source.** `src/ffpopt/dihed/DihedFitRegularize.py`

## Operational terms

### Fragmented dihed twist

**Definition.** Scission breaks the parent ligand into smaller fragments,
runs the twist workflow in each fragment directory, then merges fitted DIHE
terms back into a parent frcmod.

**Detail.** Inputs are parent `mol2` + `lib` + `frcmod`. Output is
`merged.frcmod` (+ `.merge_report.json`); the parent `lib` is unchanged.
Per-fragment merge accumulates DIHE from all `itXX.frcmod` files in order
(drop-mode survivors retained unless a later iteration refits the same key).

**Authoritative source.** `src/ffpopt/workflows/FragmentedTwist.py:run_fragmented_dihed_twist_workflow`

### Whole-ligand dihed twist

**Definition.** Twist rotatable bonds on the intact parent ligand (no scission
caps) and write a parent ``.frcmod``.

**Detail.** CLI ``lig-dihed-correct --whole-ligand``. Optional AFFDO extras
(``--soft-dihed-restraint``, ``--multi-centroid``, ``--fit-full``,
``--boltzmann-charges``) default off. Bond batches use
``FFPOPT_WHOLE_MAX_BONDS_PER_TWIST`` (default 8). Top-level twist may nest
bond x wavefront workers. Fragments with at most two fit bonds flatten to
one axis; larger fragments use the same nested joint packing.

**Authoritative source.** `src/ffpopt/workflows/WholeLigandTwist.py:run_whole_ligand_dihed_twist_workflow`

### skip_existing

**Definition.** When True, reuse on-disk scan/fit/fragment artifacts instead
of recomputing them.

**Detail.** Enables restart-friendly runs; set False for a fully fresh
calculation. Fragmented twist also writes ``frag-twist.done`` per fragment so
completed fragments are not re-queued (and do not take a CPU lease) on parent
restart. Parent start clears stale entries in ``.cpu_budget.json``. Leases
are held only during wavefront scan phases; prepare / fit / compare release
cores so siblings can grow. A scan never starts on one core when the
budget can spare ``FFPOPT_MIN_WF_NPROC`` (at least 2). Correlated
fragments (weight = n_bonds, cap 8)
get a larger share than 1-D jobs; sequential leftover bonds re-lease to
pick up free cores. With many fragments and a modest ``nproc``,
pools prefer fragment/bond breadth over depth (see
``prefer_fragment_pool_depth`` / ``prefer_bond_pool_depth``;
``FFPOPT_PREF_WF_DEPTH`` / ``FFPOPT_PREF_WF_BREADTH`` override). Fragments
with more than two fit bonds use whole-ligand joint packing
(``ffpopt.workflows.BondBatches``; ``FFPOPT_WHOLE_MAX_BONDS_PER_TWIST``,
``FFPOPT_BOND_COUPLE_RADIUS``) so those rotors are one correlated system.
1–2-bond fragments keep independent 1-D wavefronts.

**Authoritative source.** `src/ffpopt/workflows/`

## External integrations

### scission (FragmentMol)

**Definition.** Integrated package under ``src/scission`` that fragments a
ligand Amber bundle and merges per-fragment frcmods by atom type.

**Detail.** Required for ``run_fragmented_dihed_twist_workflow`` and for the
``lig-scission`` / ``scission`` CLIs. Needs AmberTools (``tleap``) on ``PATH``
to build fragment ``parm7``/``rst7``.

**Authoritative source.** ``src/scission/`` ; ``examples/scission-interface/run.py``
