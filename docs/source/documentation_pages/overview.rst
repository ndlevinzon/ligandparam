Overview
========

``ligandparam`` provides a stage-based interface for parameterizing nonstandard
ligands and residues for Amber MD. Recipes such as
:class:`~ligandparam.recipes.FreeLigand` and
:class:`~ligandparam.recipes.LazyLigand` assemble a pipeline; each stage wraps
a concrete step (Gaussian ESP, RESP fitting, Leap, …).

Multi-orientation RESP
----------------------

:class:`~ligandparam.recipes.FreeLigand` (and
:class:`~ligandparam.recipes.DPFreeLigand`) sample multiple ligand orientations
before averaging charges. The default ``so3_n28`` protocol uses a fixed
28-point quaternion pack that covers SO(3) more uniformly than the historical
Euler alpha/beta grid (``legacy_euler``). Both protocols keep the same job
count and feed the same multi-RESP → ``parmchk2`` → LEaP path.

See :mod:`ligandparam.io.orientations` and the :doc:`recipes` / :doc:`examples`
sections for details.
