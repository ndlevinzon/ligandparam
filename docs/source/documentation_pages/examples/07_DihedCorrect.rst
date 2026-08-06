Example 07: Dihedral correction after parameterization
======================================================

This example describes the recommended **same-session** workflow: run a
recipe with ``lig-getparam``, then correct torsions with
``lig-dihed-correct`` (ffpopt + scission).

Learning outcomes
-----------------

1. Know which files ``lig-getparam`` must produce for dihedral correction.
2. Run ``lig-dihed-correct`` with a non-qdpi HL model (e.g. ``xtb``).
3. Know which outputs to load in LEaP afterward.

Prerequisites
-------------

* Editable install: ``pip install -e ".[dihed]"`` (and ``tblite`` for ``xtb``)
* AmberTools on ``PATH`` (including ``tleap``)
* A completed FreeLigand (or LazyLigand) run for your ligand

Commands
--------

.. code-block:: bash

   # Parameterize (label stem = chaps from chaps.mol2)
   lig-getparam \
       -i chaps.mol2 -r CHA -d CHA3 -rn freeligand \
       -a gaff2 -cm bcc -c 0 -n 10 -mem 32

   # Optional: inspect fragments only
   lig-scission fragment -d CHA3 -r CHA --label chaps

   # Fit torsions (example HL model: xtb)
   lig-dihed-correct \
       -d CHA3 -r CHA --label chaps \
       --model xtb -n 10

Expected files under ``CHA3/CHA/``
---------------------------------

After ``lig-getparam``:

* ``chaps.mol2``, ``chaps.lib``, ``chaps.frcmod``

After ``lig-dihed-correct``:

* ``chaps.dihed.frcmod`` — merged torsion-corrected frcmod
* ``chaps.dihed.frcmod.merge_report.json``
* ``chaps.dihed_fragments/`` — scission + scan/fit intermediates

Use ``chaps.lib`` (unchanged) with ``chaps.dihed.frcmod`` in LEaP.

See :doc:`../dihedrals` and :doc:`../cli` for models and options.
