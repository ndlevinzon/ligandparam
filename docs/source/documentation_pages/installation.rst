Installation
============

Clone the repository and install into a conda / mamba environment
(Miniforge recommended):

.. code-block:: bash

    git clone https://github.com/piskulichz/ligandparam.git
    cd ligandparam
    mamba env create -f env.yaml
    conda activate ligandparam
    pip install .

Editable install for development:

.. code-block:: bash

    pip install -e .

This installs three packages from ``src/``: ``ligandparam``, ``ffpopt``, and
``scission``, plus the CLI entry points listed in :doc:`cli`.

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
    pip install ".[dihed]"  # ndfes + geometric (required by ffpopt WaveFront)
    pip install ".[ml]"     # DeepMD (use conda for TensorFlow on HPC)
    pip install ".[sage]"   # OpenFF Sage conversion
    pip install ".[docs]"   # Sphinx documentation build
    pip install ".[all]"

``lig-dihed-correct --model xtb`` needs ``tblite`` (``pip install ".[tblite]"``).
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
