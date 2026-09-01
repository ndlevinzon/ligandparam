Installation
============

Clone the repository and install into a conda / mamba environment
(Miniforge recommended):

.. code-block:: bash

    git clone https://github.com/piskulichz/ligandparam.git
    cd ligandparam
    mamba env create -f env.yaml
    conda activate ligandparam
    # pip uninstall ligandparam   # only if this tree was installed under that name
    pip install .

Editable install for development:

.. code-block:: bash

    pip install -e .

This installs the **ALPS** distribution from ``src/``. ALPS is the package
you install; it ships four import packages and their core Python
dependencies:

* ``alps`` - orchestrator (``lig-dihed-correct``, ``lig-scission``)
* ``ligandparam`` - parameterization (``lig-getparam``)
* ``ffpopt`` - torsion fitting
* ``scission`` - fragmentation and frcmod merge

Those in-tree ffpopt / scission copies are the **internal** companions.
To point ALPS at independent checkouts instead, see :doc:`companions`.
If you previously installed this repo as ``ligandparam``, uninstall that
distribution first (``pip uninstall ligandparam``) so the ``alps``
metadata is the one pip sees.

Validate your install
---------------------

After ``pip install`` / ``pip install -e .``, run the install-validation suite
(no AmberTools or Gaussian required for the core checks):

.. code-block:: bash

    python -m unittest tests.test_install_validation -v

Optional extras (``tblite``, ``geometric``, AmberTools on ``PATH``) are checked
when present and skipped with an explicit reason when absent.

Developer regression tests
--------------------------

After changing code under ``src/``, run:

.. code-block:: bash

    python -m unittest tests.test_developer_regression -v

Both suites:

.. code-block:: bash

    python -m unittest tests.test_install_validation tests.test_developer_regression -v

These two modules are the supported test entry points (recipe wiring, logging,
I/O contracts, and core helpers).

Optional extras
---------------

.. code-block:: bash

    pip install ".[tblite]" # GFN2-xTB (lig-dihed-correct --model xtb)
    pip install ".[aimnet]" # AIMNet2 (Python 3.11-3.13 + PyTorch 2.8+; --model aimnet2)
    pip install ".[dihed]"  # ndfes + geometric (required by ffpopt WaveFront)
    pip install ".[ml]"     # DeepMD (use conda for TensorFlow on HPC)
    pip install ".[sage]"   # OpenFF Sage conversion
    pip install ".[docs]"   # Sphinx documentation build
    pip install ".[all]"

``lig-dihed-correct --model xtb`` needs ``tblite`` (``pip install ".[tblite]"``).
``--model aimnet2`` needs the ``aimnet`` extra and a matching PyTorch
(CPU HPC: install the CPU torch wheel first; see :doc:`dihedrals`).
DeepMD recipes need TensorFlow; on HPC install it from conda, not pip::

    conda install -c conda-forge tensorflow deepmd-kit
    pip install -e ".[ml]"

Heavier HL models (``qdpi2``, ``mace``, ...) need their corresponding stacks;
see :doc:`dihedrals`.

External tools
--------------

Depending on the recipe or CLI you run, you also need these on your ``PATH``
(or configured via recipe kwargs / environment variables):

* AmberTools (``antechamber``, ``parmchk2``, ``tleap``) - parameterization
  and scission fragment ``parm7`` / ``rst7`` writing
* Gaussian (``g16`` or compatible) - FreeLigand / LazyLigand ESP and
  optimization

See the project README for a fuller requirements table.
