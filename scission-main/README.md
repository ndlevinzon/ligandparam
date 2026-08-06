# scission

Gitlab pages link: https://scission-da161d.gitlab.io/

`scission` is a Python package for generating torsion-scan fragments from an AMBER-style ligand input triplet:

- charged `.mol2`
- matching `.lib`
- matching `.frcmod`

The current workflow is aimed at small-molecule torsion fitting. It identifies acyclic single-bond torsions, including amide-like cases by default, can optionally use SMARTS patterns to nominate additional non-ring valid dihedrals, generates reduced fragments, rejects fragments that are too clash-prone for rigid scans, writes scan-ready fragment directories with AMBER files and torsion metadata, and can merge fitted fragment torsions back into a final parent `frcmod`.

## What It Does

Given one ligand triplet, `scission` currently:

- parses the parent ligand from `.mol2/.lib/.frcmod`
- enumerates acyclic single-bond torsions
- builds reduced fragment candidates
- keeps whole ring systems intact
- prefers fragments that reuse rigid ring-containing domains
- rejects torsions that cannot be represented by a reduced fragment
- writes per-fragment output directories with:
  - `fragment.mol2`
  - `fragment.xyz`
  - `fragment.lib`
  - `fragment.frcmod`
  - `fragment.auto.frcmod`
  - `fragment.parm7`
  - `fragment.rst7`
  - `manifest.json`
  - `fit_torsions.json`
- writes a top-level `fragment_index.json`
- can merge fitted `itXX.frcmod` outputs from fragment scan directories into a final parent `frcmod`

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,chem]'
```

In this repo, `parm7/rst7` writing also depends on AMBER tools being on `PATH`, typically via:

```bash
source ~/.bashrc
load_flow
```

## Run

```bash
scission fragment \
  --mol2 examples/jmc2025-1/binder_jmc2025-1.mol2 \
  --lib examples/jmc2025-1/binder_jmc2025-1.lib \
  --frcmod examples/jmc2025-1/binder_jmc2025-1.frcmod \
  --outdir examples/jmc2025-1/fragmentmol_output_latest
```

The command prints a JSON summary to stdout and writes a persistent `summary.json` in the output directory.

To restore the stricter legacy behavior that excludes amide-like acyclic single bonds:

```bash
scission fragment \
  --acyclic-rotatable-only \
  --mol2 examples/jmc2025-1/binder_jmc2025-1.mol2 \
  --lib examples/jmc2025-1/binder_jmc2025-1.lib \
  --frcmod examples/jmc2025-1/binder_jmc2025-1.frcmod \
  --outdir examples/jmc2025-1/fragmentmol_output_latest
```

To nominate otherwise excluded bonds with SMARTS, use `--include-bond-smarts`.
The pattern must mark the central bond atoms as `:1` and `:2`. This only
widens bond matching; the final torsion must still be a valid non-ring
heavy-atom dihedral.

```bash
scission fragment \
  --acyclic-rotatable-only \
  --include-bond-smarts "[C:1](=[O])[N:2]" \
  --mol2 examples/jmc2025-1/binder_jmc2025-1.mol2 \
  --lib examples/jmc2025-1/binder_jmc2025-1.lib \
  --frcmod examples/jmc2025-1/binder_jmc2025-1.frcmod \
  --outdir examples/jmc2025-1/fragmentmol_output_latest
```

The same override can be provided in YAML config:

```yaml
rotatable_bond_smarts:
  - "[C:1](=[O])[N:2]"
  - "[C:1]=[N:2]"
```

To instead **restrict** fragmentation to only the torsions whose central
bond matches one or more SMARTS patterns (an allow-list), use
`--restrict-bond-smarts` (or `restrict_to_bond_smarts:` in YAML). The
pattern marks the central bond atoms as `:1` and `:2`, just like
`--include-bond-smarts`. Torsions whose central bond matches no pattern are
dropped before any fragment is built and reported under `rejected_torsions`.
The two knobs compose: nomination decides what counts as rotatable,
restriction decides which of those to actually fit.

```bash
scission fragment \
  --restrict-bond-smarts "[c:1]-[c:2]" \
  --mol2 examples/jmc2025-1/binder_jmc2025-1.mol2 \
  --lib examples/jmc2025-1/binder_jmc2025-1.lib \
  --frcmod examples/jmc2025-1/binder_jmc2025-1.frcmod \
  --outdir examples/jmc2025-1/fragmentmol_output_latest
```

```yaml
restrict_to_bond_smarts:
  - "[c:1]-[c:2]"
```

### Pick a bond interactively

Writing `:1`/`:2`-mapped SMARTS by hand is fiddly. `scission pick-bond` opens a
small page in your browser so you can build one by clicking. It needs the
`chem` extra (RDKit).

```bash
scission pick-bond --mol2 examples/jmc2025-1/binder_jmc2025-1.mol2
```

This serves a localhost page showing the 2D structure. Click two bonded atoms to
select the central bond, then drag the **radius** slider: radius `0` is just the
two atoms and their bond order (broad), and larger radii fold in more
neighboring context (more specific). The page reports how many bonds in this
molecule the pattern matches — computed with the same matcher the pipeline uses
— so dial the radius until it reads `unique` (one match), then **Copy SMARTS**
and pass it to `--restrict-bond-smarts`. Use `--no-browser` to just print the
URL, or `--port`/`--host` to control the bind address.

To merge fitted fragment torsions back into a parent frcmod:

```bash
scission merge \
  --parent-frcmod examples/tyk2_dihedral/ejm_45/ejm_45_0.frcmod \
  --fragments-root examples/tyk2_dihedral/ejm_45/molcleaver_output_latest \
  --out examples/tyk2_dihedral/ejm_45/molcleaver_output_latest/final.frcmod \
  --report examples/tyk2_dihedral/ejm_45/molcleaver_output_latest/final.merge_report.json
```

## Current Example Result

For `examples/jmc2025-1`, the current output lives in:

- [examples/jmc2025-1/fragmentmol_output_latest](/home/piskulic/Project/SoftwareDevelopment/FragmentMol/examples/jmc2025-1/fragmentmol_output_latest)

At the moment it selects:

- `3` reduced fragments
- `6` covered torsions
- `1` rejected torsion: `C1-C2-O1-C3`

For `examples/tyk2_dihedral/ejm_45`, the current merged parent handoff includes:

- `final.mol2`
- `final.lib`
- `final.frcmod`
- `final.merge_report.json`

## File Meanings

### `manifest.json`

This is the main provenance record for a fragment. It contains:

- retained parent atom indices
- cut bonds
- cap atoms and cap charges
- `cap_decisions`: per-cut record of the cap chosen and why (`reason`, `cap`,
  `bare_h_charge`, ...)
- parent-to-fragment atom mapping
- assigned torsion labels
- integer-adjusted fragment net charge

### `fit_torsions.json`

This file is intended for downstream torsion-scan code. For each torsion assigned to a fragment it records:

- `label`
- `fragment_rotatable_bond`
- `fragment_dihedral_atoms`
- `parent_rotatable_bond`
- `parent_dihedral_atoms`
- matching atom-name fields for readability

All atom indices in these files are `1`-indexed.

Example:

```json
{
  "label": "C9-N1-C10-S2",
  "fragment_rotatable_bond": [5, 6],
  "fragment_dihedral_atoms": [3, 5, 6, 10]
}
```

If your downstream code expects `0`-indexed atoms, subtract `1` from every value.

### `fragment.auto.frcmod`

This is generated with `parmchk2` from the fragment MOL2 and supplements the copied parent `fragment.frcmod`. It exists to fill in any cap-related or fragment-specific missing terms before `tleap` writes the topology.

### `fragment_index.json`

This top-level file summarizes all selected fragments in one place. It includes:

- `fragment_id`
- `source_candidate_id`
- output directory and manifest path
- retained atoms
- cut bonds
- assigned torsions
- parent-fallback status

### `fragment.parm7` / `fragment.rst7`

These are written with `tleap`. The topology is usually reliable; the starting geometry is only a constructed scan seed, not a minimized QM-quality structure.

## Cap Behavior

Capping is chemistry-aware by default (`cap_strategy: chemistry_aware`). When a
bond `R-X` is cut:

- a bare hydrogen (`R-H`) is preferred when `R` is carbon and `X` is
  carbon/hydrogen — this adds no spurious hydrogen-bond donor
- otherwise the removed atom is recreated to match what was severed and the cut
  bond order: `C-C -> C-CH3` (`C=CH2`, `C#CH`), `C-O -> C-OH` (`C=O`),
  `C-N -> C-NH2` (`C=NH`, `C#N`), `C-S -> C-SH`
- a carbon cut is escalated to a methyl charge sink only when no other sink
  exists and a bare hydrogen would fall below `cap_h_min_charge` (default `0.0`),
  so a cap hydrogen is never negative; when several carbon cuts could be the
  sink, the sterically roomiest one (best rotated-scan clash margin) is chosen so
  bulk lands where there is room and H stays where space is tight
- a carbon cut is always kept as a matched cap (never a bare H) when its retained
  atom is in, or within `torsion_neighborhood_radius` bonds of, the fitted
  dihedral or is an sp2/conjugated centre — stripping a substituent there would
  change the steric/conjugation environment that sets the torsional barrier
  (toggles: `preserve_torsion_neighborhood`, `torsion_neighborhood_radius`,
  `preserve_conjugated_caps`)

Two legacy strategies remain available via `cap_strategy`: `hydroxyl` (the old
`-OH`-everywhere behavior) and `hydrogen` (always a bare hydrogen).

Important details:

- caps are given unique element-first names like `CX01`, `OX01`, `NX01` with
  hydrogens named `HX01`, `HX02`, ...; these behave cleanly in viewers like VMD
- retained atoms keep their parent charges; cap atoms get representative base
  charges (typical GAFF2/AM1-BCC values, so polar caps keep their polarity rather
  than being forced neutral) and the small integer-charge residual is routed to
  the cap that can best absorb it — heteroatom caps first, then carbon caps, and
  hydrogens only as a last resort (no charge model is invoked)
- cap placement is geometry-aware:
  - for multiply connected atoms, the cap is placed opposite the local bonded-neighbor directions
  - this avoids caps being projected through ring systems
  - saturating hydrogens use idealized (non-minimized) geometry

## Important Limitations

- only acyclic single-bond torsions are targeted by default, unless a SMARTS override nominates another valid non-ring dihedral
- no full parent fallback is counted as success by default
- clash screening is still heuristic
- output `rst7` coordinates are not minimized automatically
- the current merge step is driven by fitted `DIHE` atom-type families and assumes `ffpopt`-style `itXX.frcmod` plus `itXX.fit.json` outputs
- when two fragments fit the same torsion family, the one that directly scanned it wins and the choice is only recorded in the merge report; the families are not refit jointly

## Tests

```bash
./.venv/bin/pytest -q
```

Current status in this repo:

- `48 passed`

## Next Step After Scans

Once fragment scans are complete, the downstream step is:

- read the latest `itXX.frcmod` from each fragment directory
- use the `params` block of the companion `itXX.fit.json` to determine which torsion families were intentionally fit — this is the full coupled family the optimizer relaxed, not just the scanned/plotted torsions
- merge those updated `DIHE` terms into a new parent `frcmod`, replacing each fitted family as a whole block

There is a fuller workflow note in [docs/workflow.md](/home/piskulic/Project/SoftwareDevelopment/FragmentMol/docs/workflow.md).

The Python module path is `scission`.
