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

Optional extras:

.. code-block:: bash

    pip install ".[ml]"     # DeepMD / SQM-related workflows
    pip install ".[sage]"   # OpenFF Sage conversion
    pip install ".[docs]"   # Sphinx documentation build
    pip install ".[all]"

External tools
--------------

Depending on the recipe, you also need these on your ``PATH`` (or configured
via recipe kwargs / environment variables):

* AmberTools (``antechamber``, ``parmchk2``, ``tleap``)
* Gaussian (``g16`` or compatible)

See the project README for a fuller requirements table.
