# GLOSSARY — ffpopt

> Canonical definitions for domain terms used in this repo. The aim is
> *consistency*: one definition per term, one place to update it when the
> meaning shifts.

## What goes in

- Domain terms with non-obvious meaning (a reader who has not worked on
  ffpopt for six months should reach for this file).
- Terms used in three or more docs or modules with a specific repo meaning
  that diverges from the generic English meaning.
- Cross-cutting concepts that appear in `dev/claude-skills/` skill files,
  README, and code identifiers — anchor them once here, point everywhere else.

## What does NOT go in

- Private internal names (variables, classes) — those live in the code.
- Generic programming concepts (database, queue, retry) unless this repo uses
  them in a non-standard way.
- Per-team jargon that belongs in a chat channel description, not here.

## How to add an entry

Use the shape below. Lead with the name. One-sentence definition. Add detail
only when the one-sentence form is genuinely ambiguous or invites misuse.
Always cite an authoritative source — a file path plus section.

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
modern twist workflows. Driven by `ffpopt.WaveFront.run_dihed_wavefront`.

**Authoritative source.** `src/python/lib/ffpopt/WaveFront.py`

### Twist workflow

**Definition.** Iterative loop that scans a torsion at high level (HL) and
with the current Amber force field (LL), fits cosine force constants, applies
them, and rescans until HL≈LL or `maxiter` is reached.

**Detail.** Exposed as `run_dihed_twist_workflow` (single molecule) and
`run_fragmented_dihed_twist_workflow` (scission fragments + merge).

**Authoritative source.** `src/python/lib/ffpopt/Workflows.py`

## Domain concepts

### Dihed twist / torsion correction

**Definition.** Refitting Amber proper-dihedral force constants so MM energy
profiles match a chosen high-level model along rotatable bonds.

**Detail.** Does not rewrite atomic charges or the Amber `.lib`; only DIHE
(and related) terms are updated, typically into a new `.frcmod`.

**Authoritative source.** `src/python/lib/ffpopt/Workflows.py` (module docstring)

### HL scan / LL scan

**Definition.** High-level (HL) is the target chemistry (`--model`, e.g.
`qdpi2`); low-level (LL) is the current Amber force field evaluated with
`sander`.

**Detail.** Fit residuals are HL−LL energy profiles along the scanned
dihedral. Convergence heuristics can drop bonds that already match.

**Authoritative source.** `src/python/lib/ffpopt/Workflows.py:run_dihed_twist_workflow`

### bytype (global) vs bespoke parameters

**Definition.** `bytype=True` fits one potential per atom-*type* quartet and
writes it to a frcmod; bespoke (`masks: null`) maps potentials to atom-*name*
quartets via a Parmed Python patch script.

**Detail.** Fragmented / scission merges **require** `bytype=True` because
fragment atom names do not exist in the parent topology.

**Authoritative source.** `src/python/lib/ffpopt/Workflows.py:run_fragmented_dihed_twist_workflow`

### nprim

**Definition.** Number of cosine primitives (periodicities 1…nprim) fitted
per torsion parameter family.

**Detail.** Default is 3 (periods 1, 2, and 3).

**Authoritative source.** `README.md` (GenDihedFit / DihedTwistWorkflow)

## Operational terms

### Fragmented dihed twist

**Definition.** Scission breaks the parent ligand into smaller fragments,
runs the twist workflow in each fragment directory, then merges fitted DIHE
terms back into a parent frcmod.

**Detail.** Inputs are parent `mol2` + `lib` + `frcmod`. Output is
`merged.frcmod` (+ `.merge_report.json`); the parent `lib` is unchanged.

**Authoritative source.** `src/python/lib/ffpopt/Workflows.py:run_fragmented_dihed_twist_workflow`

### skip_existing

**Definition.** When True, reuse on-disk scan/fit/fragment artifacts instead
of recomputing them.

**Detail.** Enables restart-friendly runs; set False for a fully fresh
calculation.

**Authoritative source.** `src/python/lib/ffpopt/Workflows.py`

## External integrations

### scission (FragmentMol)

**Definition.** External package that fragments a ligand Amber bundle and
merges per-fragment frcmods by atom type.

**Detail.** Required only for `run_fragmented_dihed_twist_workflow`. Needs
AmberTools (`tleap`) on `PATH` to build fragment `parm7`/`rst7`.

**Authoritative source.** `examples/scission-interface/run.py`
