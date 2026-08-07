# scission (integrated)

AMBER-aware torsion fragment generation, vendored under ``src/scission`` next to
``ligandparam`` and ``ffpopt``.

Used by ``ffpopt.Workflows.run_fragmented_dihed_twist_workflow`` and by the
standalone CLIs:

```bash
scission fragment --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod --outdir frags
lig-scission fragment --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod --outdir frags

# After lig-getparam (same layout as lig-dihed-correct):
lig-scission fragment -d CHA3 -r CHA --label chaps
```

Requires AmberTools (``tleap``) on ``PATH`` for ``parm7``/``rst7`` writing.
RDKit is already a ligandparam dependency and is used for SMARTS / drawings.

Runtime package is this ``src/scission`` tree. An optional ``scission-main/``
checkout (gitignored) may exist as an upstream reference for historical docs /
examples and is not required after ``pip install``.
