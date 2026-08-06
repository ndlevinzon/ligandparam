# Fragment Workflow

This document describes the current `scission` workflow as it exists in this repository.

## 1. Inputs

One ligand is provided as:

- `.mol2`
- `.lib`
- `.frcmod`

These are assumed to describe the same ligand in the same atom order.

## 2. Fragment Generation

`scission`:

1. loads the parent ligand
2. finds acyclic single-bond torsions
3. builds reduced fragment candidates from rigid heavy-atom domains
4. keeps whole rings intact
5. screens candidates for rigid scan clashes
6. selects a small reusable set of reduced fragments

If a torsion cannot be represented by a reduced fragment, it is explicitly marked as rejected.

By default, this includes amide-like acyclic single bonds. The stricter legacy
behavior can be restored with the `scission fragment --acyclic-rotatable-only`
CLI flag or by setting `include_rigid_single_bonds: false` in the config.

If you need to nominate otherwise excluded bonds, you can provide
`rotatable_bond_smarts` patterns in the config or `--include-bond-smarts` on
the CLI. Each SMARTS pattern must mark the central bond atoms as `:1` and `:2`.
These overrides do not bypass the core safety checks: the final torsion must
still be non-ring and must still admit a valid heavy-atom dihedral.

## 3. Fragment Directory Contents

Each selected fragment directory currently contains:

- `fragment.mol2`
- `fragment.xyz`
- `fragment.lib`
- `fragment.frcmod`
- `fragment.auto.frcmod`
- `fragment.parm7`
- `fragment.rst7`
- `manifest.json`
- `fit_torsions.json`
- `tleap.in`
- `tleap.stdout.log`
- `tleap.stderr.log`
- `parmchk2.stdout.log`
- `parmchk2.stderr.log`

The output root also includes:

- `summary.json`
- `fragment_index.json`

Your own scan code may also add files such as:

- `fragment_for_scan.json`
- `itNN.fit.json`
- `scan.sh`
- checkpoint files

## 4. Indexing Convention

All fragment and parent atom indices written by `scission` are `1`-indexed.

This includes:

- `parent_to_fragment_atom_map`
- `fit_torsions.json`
- `manifest.json`

If a downstream tool wants `0`-indexed atoms, subtract `1`.

## 5. `fit_torsions.json`

This file is the main handoff to torsion-scan code.

For each torsion assigned to a fragment it provides:

- `label`
- `fragment_rotatable_bond`
- `fragment_dihedral_atoms`
- `parent_rotatable_bond`
- `parent_dihedral_atoms`
- matching atom names

Interpretation:

- `fragment_rotatable_bond` gives the central bond to scan in that fragment
- `fragment_dihedral_atoms` gives the local `a-b-c-d` dihedral definition for the scan
- the `parent_*` fields preserve the mapping back to the original ligand

## 6. Caps

Capping is controlled by `cap_strategy` (default `chemistry_aware`):

- **`chemistry_aware`** — when a bond `R-X` is cut, prefer a bare hydrogen
  (`R-H`). A bare hydrogen is only used when `R` is carbon and `X` is
  carbon/hydrogen (so no spurious hydrogen-bond donor is introduced and no
  heteroatom electronics are stripped). Otherwise the removed atom `X` is
  recreated as a hydrogen-saturated group matching its element and the cut bond
  order:
  - `C-C -> C-CH3` (or `C=CH2`, `C#CH`)
  - `C-O -> C-OH` (or `C=O`)
  - `C-N -> C-NH2` (or `C=NH`, `C#N`)
  - `C-S -> C-SH`
  A carbon cut is also escalated to a methyl charge sink when no other sink
  exists and a bare hydrogen would fall below `cap_h_min_charge` (default `0.0`)
  — so a cap hydrogen is never negative. When several carbon cuts could become
  the sink, the **sterically roomiest** one is chosen: a methyl-sized cap is
  rotated through the torsion scan at each candidate site, and the bulky cap
  lands where there is room while hydrogens stay where space is tight.
  Finally, a carbon cut is **always** kept as a matched cap (never a bare H) when
  its retained atom is part of, or within `torsion_neighborhood_radius` bonds of,
  the fitted dihedral, or is an sp2/conjugated centre — stripping a substituent
  there would change the steric/conjugation environment that sets the torsional
  barrier. This is controlled by `preserve_torsion_neighborhood` (default
  `true`), `torsion_neighborhood_radius` (default `1`; `0` = dihedral atoms only,
  `2` = the next shell), and `preserve_conjugated_caps` (default `true`).
- **`hydroxyl`** — legacy behavior: every cut is capped with `-OH`, named
  `OX01`/`HX01`, with the oxygen carrying most of the integer-charge correction.
- **`hydrogen`** — every cut is capped with a single bare hydrogen.

Charges (chemistry-aware / hydrogen strategies): retained atoms keep their parent
charges; cap atoms get **representative** base charges (typical GAFF2/AM1-BCC
values, so an `OH`/`NH2` keeps its real polarity rather than being forced
neutral). No charge model is invoked. The small residual needed for an integer
fragment charge is routed to the cap that can best absorb it — heteroatom caps
first, then carbon caps, and hydrogens only as a last resort — so polar groups
soak up the correction and hydrogens stay at reasonable values. The truly
self-consistent alternative (recomputing fragment charges with antechamber
AM1-BCC) is intentionally out of scope.

Cap atoms are named element-first (`CX01`, `OX01`, `NX01`, ...) with their
hydrogens named `HX01`, `HX02`, ...

Each fragment's `manifest.json` records why every cut was capped the way it was
under `cap_decisions`, one entry per cut:

```json
{
  "parent_atom": 2, "removed_atom": 11, "bond_order": 1,
  "cap": "bare_hydrogen", "reason": "bare_hydrogen",
  "bare_h_charge": 0.098, "h_min_charge": 0.0
}
```

`reason` is one of `bare_hydrogen` (carbon-like cut, bare H non-negative),
`charge_escalation` (bare-H eligible but the balanced charge would be too low, so
a matched cap is used — `bare_h_charge` shows the rejected value),
`heteroatom_severed`, `retained_not_carbon`, `near_fitted_torsion` /
`conjugated_center` (forced matched to preserve the torsion's environment),
`forced_hydrogen_strategy`, or `legacy_hydroxyl_strategy`. The run-level
`summary.json` tallies these across all selected fragments under
`cap_decision_counts`.

Placement rule:

- the cap is not simply placed along the severed bond vector
- instead, the cap direction is chosen from the local valence environment at the retained atom
- this helps avoid caps cutting through ring systems
- saturating hydrogens on a matched cap use idealized geometry (tetrahedral,
  trigonal, or linear by bond order); coordinates are not energy-minimized

## 7. AMBER File Generation

The current AMBER topology path is:

1. write `fragment.mol2`
2. copy parent `fragment.frcmod`
3. run `parmchk2` to generate `fragment.auto.frcmod`
4. run `tleap`
5. write `fragment.parm7` and `fragment.rst7`

This path depends on `parmchk2` and `tleap` being available on `PATH`.

In this development environment, that is typically done with:

```bash
source ~/.bashrc
load_flow
```

## 8. Running External Scan Code

A typical scan workflow is:

1. choose a fragment directory
2. read `fit_torsions.json`
3. convert to `0`-indexed atoms if required by the scan tool
4. run the torsion scan on the specified central bond / dihedral
5. write an iteration-specific fit result such as `it01.fit.json`
6. write the matching fitted torsion parameters to `it01.frcmod`

The current merge implementation is built around `ffpopt`-style outputs:

- `itXX.frcmod` contains fitted `DIHE` terms
- `itXX.fit.json` identifies the torsion families intentionally fit in that iteration: its
  top-level `params` block lists every family the optimizer touched, while
  `systems[].profiles[].plots` names only the families whose profiles were scanned and plotted
- the highest-numbered `itXX.frcmod` in each fragment directory is used by default

## 9. Stitching Back to the Parent

The merge step is not literal coordinate stitching. It is parameter stitching:

1. keep the original full parent ligand
2. read the latest fitted torsion terms from each fragment directory
3. use the `params` block of `itXX.fit.json` to limit the merge to torsion families
   intentionally fit for that fragment
4. keep `fit_torsions.json` as the parent/fragment provenance record
5. write a new parent `frcmod` with updated `DIHE` terms

A fragment fit almost never touches only the torsions that were scanned. The optimizer
relaxes the whole coupled family of dihedral types around the scanned bonds — for an amidate
motif, scanning `ca-ce-nf-cd` and `ce-nf-cd-nc` also refits `o -ce-nf-cd` and `ce-nf-cd-ss`.
Each fitted family is therefore promoted as a whole replacement block:

- every fitted `DIHE` key in `itXX.frcmod` replaces the matching parent block, not just the
  plotted ones — promoting a subset leaves a torsional surface that is part new fit and part
  old generic parameters
- forward and reverse keys are canonicalized, so `ce-nf-cd-ss` and `ss-cd-nf-ce` are one key
- all periodicity lines for a key move together, so profiles are replaced rather than doubled
- a fitted key absent from the parent `frcmod` is appended, overriding the `gaff2` default

When two fragments fit the same family, the fragment that directly scanned it wins; if neither
did, the first fragment in merge order wins. Either way the choice is recorded under
`conflicts` in the merge report. Two fragments scanning the same family is ambiguous and is a
hard error.

A typical merge command is:

```bash
scission merge \
  --parent-frcmod examples/tyk2_dihedral/ejm_45/ejm_45_0.frcmod \
  --fragments-root examples/tyk2_dihedral/ejm_45/molcleaver_output_latest \
  --out examples/tyk2_dihedral/ejm_45/molcleaver_output_latest/final.frcmod \
  --report examples/tyk2_dihedral/ejm_45/molcleaver_output_latest/final.merge_report.json
```

That means:

- parent coordinates remain the parent’s problem
- fragment caps are only local scan artifacts
- the real deliverable is a corrected parent torsion parameter set

## 10. Known Gaps

- no automatic post-`tleap` minimization of fragment coordinates
- no automatic symmetry handling for fitted torsion reuse yet
- no parent-side validation against a full parameterized topology yet
- the merge currently assumes `ffpopt`-style iteration files rather than an abstract backend interface
