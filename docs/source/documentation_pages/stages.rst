Available Parametrization Stages
================================

The parametrization process is broken down into a series of stages, each of
which is a subclass of :doc:`./stages/abstractstage`. Each stage handles one
step of the pipeline and can be added, removed, or reordered after
``setup()``.

Post-processing stages such as dihedral correction sit after Leap when
enabled (see :doc:`dihedrals`).

.. toctree::
   :maxdepth: 2
   :caption: Available Stages:

   ./stages/initialize.rst
   ./stages/charge.rst
   ./stages/gaussian.rst
   ./stages/resp.rst
   ./stages/parmchk.rst
   ./stages/leap.rst
   ./stages/ffpopt_dihed.rst
   ./stages/build_system.rst
   ./stages/typematching.rst

.. toctree::
   :maxdepth: 2
   :caption: Development Stages:

   ./stages/abstractstage.rst
   ./stages/teststage.rst
