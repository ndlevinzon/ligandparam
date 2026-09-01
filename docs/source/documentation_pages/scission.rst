scission
========

``scission`` is the integrated Amber-aware fragmentation package under
``src/scission``. It builds torsion-scan fragments from a parent
``mol2`` / ``lib`` / ``frcmod`` triplet and can merge fitted fragment DIHE
terms back into a parent frcmod.

ALPS's fragmented dihed-twist workflow calls scission automatically.
You can also run it alone. scission does not import ffpopt or ligandparam.

CLI
---

Upstream-style:

.. code-block:: bash

   scission fragment \
       --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod \
       --outdir frags

After ``lig-getparam`` (same ``-d`` / ``-r`` / ``--label`` layout as
``lig-dihed-correct``):

.. code-block:: bash

   lig-scission fragment -d CHA3 -r CHA --label chaps

Merge fitted fragment frcmods:

.. code-block:: bash

   scission merge \
       --parent-frcmod LIG.frcmod \
       --out LIG.merged.frcmod \
       --fragments-root frags

When collecting a fragment's contribution, DIHE lines are accumulated from
**all** ``itXX.frcmod`` files in order (drop-mode survivors from earlier
iterations are kept unless a later file explicitly refits the same key).
Parent merge still applies scanned-vs-unscanned conflict rules.

Requirements
------------

* AmberTools ``tleap`` on ``PATH`` (writes fragment ``parm7`` / ``rst7``)
* RDKit (already a ligandparam dependency) for SMARTS / drawings when used

Python API
----------

.. code-block:: python

   from scission import InputBundle, FragmentConfig, fragment_ligand

   result = fragment_ligand(
       InputBundle(
           mol2_path="LIG.mol2",
           lib_path="LIG.lib",
           frcmod_path="LIG.frcmod",
       ),
       "frags",
       FragmentConfig(),
   )

Module reference
----------------

.. automodule:: scission
   :members:
   :undoc-members:
   :show-inheritance:

Runtime package is ``src/scission`` unless ``LIGANDPARAM_SCISSION=external``
points at another tree (:doc:`companions`). An optional ``scission-main/``
checkout is upstream reference only. See ``src/scission/README.md``.
