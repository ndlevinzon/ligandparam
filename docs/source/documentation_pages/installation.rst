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

Optional extras
---------------

.. code-block:: bash

    pip install ".[ml]"     # DeepMD / SQM / tblite (xtb) related workflows
    pip install ".[dihed]"  # extras useful for dihedral fitting (e.g. ndfes)
    pip install ".[sage]"   # OpenFF Sage conversion
    pip install ".[docs]"   # Sphinx documentation build
    pip install ".[all]"

``lig-dihed-correct`` with ``--model xtb`` typically needs ``tblite``
(``pip install tblite`` or ``".[ml]"``). Heavier HL models (``qdpi2``,
``mace``, …) need their corresponding ML stacks; see :doc:`dihedrals`.

External tools
--------------

Depending on the recipe or CLI you run, you also need these on your ``PATH``
(or configured via recipe kwargs / environment variables):

* AmberTools (``antechamber``, ``parmchk2``, ``tleap``) — parameterization
  and scission fragment ``parm7`` / ``rst7`` writing
* Gaussian (``g16`` or compatible) — FreeLigand / LazyLigand ESP and
  optimization

See the project README for a fuller requirements table.
