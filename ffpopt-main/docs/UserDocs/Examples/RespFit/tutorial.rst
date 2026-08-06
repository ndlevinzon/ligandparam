.. _respfit-tutorial:


Restrained Electrostatic Potential Charge Fitting
=================================================

This tutorial will show how to perform RESP calculations to derive MM charges.


Learning Objectives
-------------------


Tutorial
--------

1. Create the molecule using antechamber 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the example shown below, the ffpopt-ConfSearch.py script is used to create create a 3D structure of propanol. The ``--nkeep=1`` causes the script to output only the lowest-energy conformer that it could locate.
The ffpopt-Json2Crds.py script extracts the molecule from mol.json and writes it in PDB format.
The antechamber command writes a mol2 with gaff2 atom types and am1-bcc charges after correcting duplicate atom names

.. code-block::

   ffpopt-ConfSearch.py --out=mol.json --nkeep=1 'InChI=1S/C3H8O/c1-2-3-4/h4H,2-3H2,1H3'
   ffpopt-Json2Crds.py --inp=mol.json --out=mol.pdb
   antechamber -i mol.pdb -fi pdb -o am1bcc.mol2 -fo mol2 -c bcc -nc 0 -rn MOL -at gaff2 -du y -an y -pf y -seq n
   rm mol.pdb mol.json


The ffpopt-RespFit.py script does not try to automatically determine atoms that should have the same charge (for example, all 3 hydrogens on a methyl group should have the same charge).  Instead, it follows the pattern that it reads from the input molecule; if the input molecule has atoms with the same atomic number and same charge, then it will enforce this condition on the output RESP-fitted charges.
If you already have a PDB file of your molecule, you should still run it through antechamber to create am1-bcc (or abcg2) charges for the sole purpose of identifying the charge symmetries within the molecule.


2. Create 1-or-more conformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESP fitting is more robust if multiple conformers are considered during the fit.  We can use the ffpopt-ConfSearch script to search for conformers and output them to confs.json.  More information can be found in the conformer search tutorial:  :ref:`confsearch-tutorial`.

.. code-block::

   ffpopt-ConfSearch.py --out=confs.json am1bcc.mol2


3. Perform the ab initio calculations and the RESP fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ffpopt-RespFit.py script can use several ab initio backends to perform the necessary SCF calculations.
The default is the use psi4, but this can be changed with the ``--program`` option.
The script will first check to see if the ab initio input (*_*.inp) and output (*_*.log) files are present.
If the files are present, then it reads the output without running the ab initio calculation to save time.
It does this without checking if their contents are consistent with your command-line arguments; therefore,
you should delete (or move) these files before rerunning the RESP fit with different options.

.. code-block::

   ffpopt-RespFit.py --model="hf/6-31g*" --inp=confs.json --out=resp.json confs.json
   rm *_*.inp *_*.log
   rm confs.json

4. Extract the mol2 file
~~~~~~~~~~~~~~~~~~~~~~~~

One can extract the mol2 file containing the multi-conformer RESP fit charges using ffpopt-Json2Crds.py/

.. code-block::

   ffpopt-Json2Crds.py --inp=resp.json --out=resp.mol2
   
   
5. Find out more
~~~~~~~~~~~~~~~~
This short example does not highlight all of the command line options.
For the most up-to-date list of options, run ``ffpopt-RespFit.py --help``.
Some of the most used options are listed below.

   a. ``--program="command"``. This sets the command used to run the ab initio softwarem, where the "command" could be "g16", "mpirun -n 4 quick.MPI", "quick.cuda", "psi4", etc.  It supports Gaussian, Psi4, and quick (only for gas phase fits).
   b. ``--respf``. If present, then perform the fit using Kollman's resp.f program rather than ffpopt's internal optimization procedure based on the description in 10.1021/j100142a004
   c. ``--group="mask"``. This option can be used multiple times to enforce constraints on the charges. The sum of fitted charges within a group will be preserved from the input. The mask is an Amber mask; e.g., "@C9,H91,H92,H93".
   d. ``--scosmo=float``. If present, then perform the fit in the gas phase and a second fit in a "COSMO-like" environment and take a linear combination of the two charge arrays.  The default is 0.0, which corresponds to a gas phase electrostatic potential.

      
