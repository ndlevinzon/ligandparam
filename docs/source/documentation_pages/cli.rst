Command-line tools
==================

Entry points installed with the package (see ``pyproject.toml``):

* ``lig-getparam`` — run a parameterization recipe
* ``lig-dihed-correct`` — fit / merge dihedral corrections (ffpopt + scission)
* ``lig-scission`` — fragment or merge with ligandparam-friendly ``-d`` / ``-r`` / ``--label`` shortcuts
* ``scission`` — upstream scission CLI (``fragment`` / ``merge`` / ``pick-bond``)
* ``smiles-to-pdb`` — SMILES → 3D PDB
* ``lighfix`` — fix ligand hydrogenation / bonding
* ``lig-to-sage`` — mol2 → OpenFF Sage helpers

Additional ``ffpopt-*.py`` scripts (PrepareInput, GenDihedFit, DihedWavefront,
…) are registered for the torsion-fitting engine.

CLIs print a one-time startup banner (logo, authors, version) at the top of
stdout; fragment workers do not reprint it.

Typical same-session workflow
-----------------------------

.. code-block:: bash

   lig-getparam -i chaps.mol2 -r CHA -d CHA3 -rn freeligand \
       --atom_type gaff2 --charge_model bcc --net_charge 0 -n 10 -mem 32

   # Optional: fragment only
   lig-scission fragment -d CHA3 -r CHA --label chaps

   # Dihedral correction (HL model example: xtb — no qdpi required)
   lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 10 --fast

Notes
-----

* ``-d`` / ``-r`` match between ``lig-getparam`` and the post-processing CLIs.
* ``--label`` is the recipe **file stem** (e.g. ``chaps`` from ``chaps.mol2``),
  not necessarily the residue name (``CHA``).
* Outputs for dihedral correction default to ``{label}.dihed.frcmod`` beside
  the original ``{label}.frcmod``; the ``.lib`` is unchanged.
* ``--fast`` enables coarser wavefront / looser geomeTRIC presets
  (``FFPOPT_FAST_WAVEFRONT=1``). Explicit non-default knobs still win.
* Console logs use a single timestamp and hierarchical ``[tag]`` brackets
  (``ffpopt.runtime.console``).

See :doc:`dihedrals` for models and file flow, and :doc:`examples/07_DihedCorrect`
for a worked narrative.
