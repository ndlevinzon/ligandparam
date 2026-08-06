# ffpopt (integrated)

Python package for force-field torsion optimization, vendored alongside
``ligandparam`` under ``src/ffpopt``.

Primary API for ligandparam integration:

```python
from ffpopt.Workflows import run_fragmented_dihed_twist_workflow
```

CLI (installed with ligandparam):

```bash
lig-dihed-correct -d CHA3 -r CHA --label chaps
```

See ``GLOSSARY.md`` in this directory and the upstream tree ``ffpopt-main/``
for models, installation notes (scission, AmberTools, ML groups), and examples.
