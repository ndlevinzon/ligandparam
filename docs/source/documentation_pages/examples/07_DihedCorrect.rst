Example 07: Dihedral correction after parameterization
======================================================

This example is the recommended **same-session** workflow: parameterize with
``lig-getparam``, then correct torsions with ``lig-dihed-correct``. ffpopt
has two modes: **fragment** (default, scission + merge) and **whole-ligand**
(``--whole-ligand``, intact parent).

Learning outcomes
-----------------

1. Know which files ``lig-getparam`` must produce for dihedral correction.
2. Run the fragment path with a non-qdpi HL model (e.g. ``xtb``).
3. Know when to switch to ``--whole-ligand`` and which extras are optional.
4. Know which outputs to load in LEaP afterward.

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

   # Fit torsions, fragment path (default; --fast = wall-time presets)
   lig-dihed-correct \
       -d CHA3 -r CHA --label chaps \
       --model xtb -n 10 --fast

   # Alternative: whole-ligand (no scission)
   lig-dihed-correct \
       -d CHA3 -r CHA --label chaps \
       --model xtb -n 44 --fast \
       --whole-ligand --soft-dihed-restraint --fit-full --fit-backend jax

Expected files under ``CHA3/CHA/``
---------------------------------

After ``lig-getparam``:

* ``chaps.mol2``, ``chaps.lib``, ``chaps.frcmod``

After ``lig-dihed-correct`` (fragment):

* ``chaps.dihed.frcmod`` - merged torsion-corrected frcmod
* ``chaps.dihed.frcmod.merge_report.json``
* ``chaps.dihed_fragments/`` - scission + scan/fit intermediates

After ``lig-dihed-correct --whole-ligand``:

* ``chaps.dihed.frcmod`` - parent torsion-corrected frcmod
* per-batch ``whole-twist.log`` and live ``WHOLE_STATUS.txt``

Use ``chaps.lib`` (unchanged) with ``chaps.dihed.frcmod`` in LEaP.

See :doc:`../dihedrals` and :doc:`../cli` for models and options.
