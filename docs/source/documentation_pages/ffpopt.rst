ffpopt
======

``ffpopt`` is the integrated force-field torsion optimizer under ``src/ffpopt``.
ligandparam uses it for post-parameterization dihedral correction
(:doc:`dihedrals`).

Primary API
-----------

.. code-block:: python

   from ffpopt.Workflows import run_fragmented_dihed_twist_workflow

   result = run_fragmented_dihed_twist_workflow(
       mol2="LIG.mol2",
       lib="LIG.lib",
       frcmod="LIG.frcmod",
       out_dir="fragments",
       merged_frcmod="LIG.dihed.frcmod",
       model="xtb",
       geometric_opt=True,
       nproc=8,
       maxiter=2,
   )

Call from an ``if __name__ == "__main__":`` guard (wavefront uses spawn-mode
multiprocessing).

Single-molecule twist (when you already have ``parm7`` / ``rst7`` and explicit
bonds) is :func:`ffpopt.Workflows.run_dihed_twist_workflow`.

ligandparam wrapper
-------------------

:class:`~ligandparam.stages.ffpopt_dihed.StageDihedTwistCorrection` and the
``lig-dihed-correct`` CLI wrap the fragmented workflow. Prefer those for
everyday use after ``lig-getparam``.

Module reference
----------------

.. automodule:: ffpopt
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ffpopt.Workflows
   :members: run_fragmented_dihed_twist_workflow, run_dihed_twist_workflow
   :undoc-members:
   :show-inheritance:

Upstream docs and examples remain in ``ffpopt-main/``. See also
``src/ffpopt/GLOSSARY.md`` and ``src/ffpopt/README.md``.
