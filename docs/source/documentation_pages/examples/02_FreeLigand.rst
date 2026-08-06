Example 02: FreeLigand
======================

This example runs a multi-orientation RESP parameterization with
:class:`~ligandparam.recipes.FreeLigand`.

Unlike :doc:`01_LazyLigand`, FreeLigand evaluates ESP charges at many ligand
orientations (default: the deterministic ``so3_n28`` quaternion pack) and
combines them with multi-state RESP fitting. That reduces bias from the
initial molecular orientation relative to the ESP grid.

Learning outcomes
-----------------

1. Use FreeLigand for multi-orientation RESP parameterization.
2. List and execute the stage pipeline.

Files
-----

See ``examples/02_FreeLigand`` in the source tree.

Tutorial
--------

Import the recipe and configure Gaussian / machine options as needed:

.. literalinclude :: ../../../../examples/02_FreeLigand/param_thio.py
    :language: python
    :end-at: gaussian_scratch

Construct the recipe (``net_charge`` must match the ligand):

.. literalinclude :: ../../../../examples/02_FreeLigand/param_thio.py
    :language: python
    :start-at: parametrize_ligand =
    :end-before: Build the FreeLigand stage list

Set up and run the stages:

.. literalinclude :: ../../../../examples/02_FreeLigand/param_thio.py
    :language: python
    :start-at: Build the FreeLigand stage list
    :end-at: dry_run=False

Pipeline (abbreviated)
----------------------

1. Convert PDB → mol2 and assign initial charges
2. Gaussian minimization / RESP preparation
3. Multi-orientation ESP (``so3_n28`` by default) and multi-RESP fit
4. Write ``.mol2`` / ``.frcmod`` / ``.lib``

Typical outputs in the working directory include ``*.resp.mol2``,
``*.frcmod``, and ``*.lib``. Charges will be similar but not identical to
LazyLigand, because FreeLigand averages over orientations.

To reproduce the older Euler alpha/beta grid instead of quaternion sampling:

.. code-block:: python

   FreeLigand(..., orientation_protocol="legacy_euler")

After parameterization, optional torsion correction (same session)::

   lig-dihed-correct -d <data_cwd> -r <resname> --label <input_stem> --model xtb

See :doc:`../dihedrals` and :doc:`07_DihedCorrect`.

Full code
---------

.. literalinclude :: ../../../../examples/02_FreeLigand/param_thio.py
    :language: python
