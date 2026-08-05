Example 01: LazyLigand
======================

Single-orientation Gaussian RESP parameterization with
:class:`~ligandparam.recipes.LazyLigand`.

Learning outcomes
-----------------

1. Construct and configure a LazyLigand recipe.
2. Call ``setup()``, ``list_stages()``, and ``execute()``.

Files
-----

See ``examples/01_LazyLigand`` in the source tree.

Tutorial
--------

Import the recipe and (optionally) define Gaussian path overrides:

.. literalinclude :: ../../../../examples/01_LazyLigand/param_thio.py
    :language: python
    :end-at: gaussian_scratch

Construct the recipe. ``net_charge`` is required; ``nproc`` / ``mem`` can be
passed to the constructor or to ``execute()``. Full options are documented on
:class:`~ligandparam.recipes.LazyLigand`.

.. literalinclude :: ../../../../examples/01_LazyLigand/param_thio.py
    :language: python
    :end-before: # Set the pre-initialized stages
    :start-at: parametrize_ligand =

Build the stage list, inspect it, and run:

.. literalinclude :: ../../../../examples/01_LazyLigand/param_thio.py
    :language: python
    :start-at: parametrize_ligand.setup

Rough pipeline:

1. Convert the PDB to mol2 and assign initial charges.
2. Minimize and fit RESP charges with Gaussian.
3. Write ``lib`` / ``frcmod`` via ``parmchk2`` and LEaP.

Typical outputs in the working directory (label = input stem):

* ``{label}.resp.mol2`` — RESP charges
* ``{label}.frcmod`` — missing parameters
* ``{label}.lib`` — Leap library

Full code
---------

.. literalinclude :: ../../../../examples/01_LazyLigand/param_thio.py
    :language: python
