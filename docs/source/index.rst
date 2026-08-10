Welcome to ligandparam's documentation!
========================================

``ligandparam`` **1.5** is a modular Python toolkit for Amber ligand
parameterization. Workflows are expressed as **recipes** (ordered lists of
**stages**) that wrap familiar tools such as Antechamber, Gaussian,
``parmchk2``, and LEaP.

The repository also ships two integrated companion packages under ``src/``:

* ``ffpopt`` — post-hoc torsion (dihedral) fitting (``runtime/``, ``scan/``, …)
* ``scission`` — Amber-aware ligand fragmentation for torsion scans

Quick start
-----------

.. code-block:: python

   from ligandparam.recipes import FreeLigand

   recipe = FreeLigand(
       in_filename="ligand.pdb",
       cwd="output",
       net_charge=0,
       nproc=12,
       mem=8,
       logger="stream",
   )
   recipe.setup()
   recipe.list_stages()
   recipe.execute()

By default, :class:`~ligandparam.recipes.FreeLigand` samples 28 orientations with
the deterministic ``so3_n28`` quaternion pack before multi-RESP fitting, then
writes ``.mol2`` / ``.frcmod`` / ``.lib`` outputs. Pass
``orientation_protocol="legacy_euler"`` to restore the older alpha/beta Euler
grid.

Same-session CLI (parameterize, then optional torsion correction)
------------------------------------------------------------------

.. code-block:: bash

   lig-getparam -i chaps.mol2 -r CHA -d CHA3 -rn freeligand -c 0 -n 10 -mem 32
   lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 10 --fast

``--label`` is the recipe file stem (from the input filename). Fragmentation
alone is available via ``lig-scission`` / ``scission``.

Common recipes
--------------

* :class:`~ligandparam.recipes.LazierLigand` — fast Antechamber (e.g. BCC) path
* :class:`~ligandparam.recipes.LazyLigand` — single-orientation Gaussian RESP
* :class:`~ligandparam.recipes.FreeLigand` — multi-orientation RESP (``so3_n28``)
* :class:`~ligandparam.recipes.DPLigand` / :class:`~ligandparam.recipes.DPFreeLigand`
  — DeepMD-assisted variants
* :class:`~ligandparam.recipes.SQMLigand` — SQM / DeepMD-assisted minimize + RESP

Stages can be inspected and edited after ``setup()``
(``remove_stage``, ``insert_stage``, ``add_stage``).

See the examples directory and the pages below for details.

.. toctree::
   :maxdepth: 4
   :caption: Documentation
   :numbered:
   :hidden:

   ./documentation_pages/overview.rst
   ./documentation_pages/installation.rst
   ./documentation_pages/cli.rst
   ./documentation_pages/dihedrals.rst
   ./documentation_pages/recipes.rst
   ./documentation_pages/stages.rst
   ./documentation_pages/io.rst
   ./documentation_pages/ffpopt.rst
   ./documentation_pages/scission.rst
   ./documentation_pages/multiresp.rst
   ./documentation_pages/examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
