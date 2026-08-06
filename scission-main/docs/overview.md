# Overview

`scission` accepts:

- a charged `.mol2`
- a matching `.lib`
- a matching `.frcmod`

It then:

1. loads the parent ligand
2. identifies acyclic single-bond torsions
3. builds reduced fragment candidates
4. screens them for rigid scan clashes
5. selects a reusable fragment set
6. writes fragment directories with:
   - `fragment.mol2`
   - `fragment.xyz`
   - `fragment.lib`
   - `fragment.frcmod`
   - `fragment.auto.frcmod`
   - `fragment.parm7`
   - `fragment.rst7`
   - `manifest.json`
   - `fit_torsions.json`
7. writes a top-level `fragment_index.json`
8. can merge fitted fragment torsion outputs back into a final parent `frcmod`

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,chem,docs]'
source ~/.bashrc
load_flow
scission fragment \
  --mol2 examples/jmc2025-1/binder_jmc2025-1.mol2 \
  --lib examples/jmc2025-1/binder_jmc2025-1.lib \
  --frcmod examples/jmc2025-1/binder_jmc2025-1.frcmod \
  --outdir examples/jmc2025-1/fragmentmol_output_latest
```

By default this includes amide-like acyclic single bonds as torsion targets.
To restore the stricter legacy behavior, add `--acyclic-rotatable-only`.
To nominate otherwise excluded bonds, add `--include-bond-smarts` with a
SMARTS pattern whose central bond atoms are mapped as `:1` and `:2`.
Ring bonds and other invalid dihedrals are still rejected.

To build such a `:1`/`:2` SMARTS by clicking instead of by hand, run
`scission pick-bond --mol2 <file>` (requires the `chem` extra). It serves a
localhost page with the 2D structure; click two bonded atoms and adjust the
environment radius until the live match count is unique, then copy the pattern
into `--restrict-bond-smarts` or `--include-bond-smarts`.

To merge fitted torsions back into a parent frcmod:

```bash
scission merge \
  --parent-frcmod examples/tyk2_dihedral/ejm_45/ejm_45_0.frcmod \
  --fragments-root examples/tyk2_dihedral/ejm_45/molcleaver_output_latest \
  --out examples/tyk2_dihedral/ejm_45/molcleaver_output_latest/final.frcmod \
  --report examples/tyk2_dihedral/ejm_45/molcleaver_output_latest/final.merge_report.json
```

## Indexing

All atom indices written by `scission` are `1`-indexed.

The import/module path is `scission`.

The CLI also supports `python -m scission ...` through `scission.__main__`.

If a downstream scan tool expects `0`-indexed atoms, subtract `1`.
