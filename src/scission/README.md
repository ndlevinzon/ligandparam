# scission (integrated)

AMBER-aware torsion fragment generation, vendored under ``src/scission`` next to
``ligandparam`` and ``ffpopt``.

## Package layout

Flat package (intentional): snake_case modules, thin public ``__init__.py``,
``cli.py`` entrypoint, ``io.py`` / ``models.py`` for I/O and dataclasses.
No ffpopt/ligandparam imports — keep that edge one-way.

Used by ``ffpopt.Workflows.run_fragmented_dihed_twist_workflow`` and by the
standalone CLIs:

```bash
scission fragment --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod --outdir frags
lig-scission fragment --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod --outdir frags

# After lig-getparam (same layout as lig-dihed-correct):
lig-scission fragment -d CHA3 -r CHA --label chaps
```

When merging fragment fits back to the parent, DIHE terms are accumulated from
**all** ``itX.frcmod`` files in each fragment directory (in order). Drop-mode
survivors from earlier iterations are kept unless a later file explicitly
refits the same key.

Requires AmberTools (``tleap``) on ``PATH`` for ``parm7``/``rst7`` writing.
RDKit is already a ligandparam dependency and is used for SMARTS / drawings.

``split_core_budget`` in ``parallel.py`` is kept local (algorithm synced with
``ffpopt.runtime.fast_wavefront.split_core_budget``).

Runtime package is this ``src/scission`` tree. An optional ``scission-main/``
checkout (gitignored) may exist as an upstream reference for historical docs /
examples and is not required after ``pip install``.
