Workflows
=========

High-level, in-process orchestration of the relaxed dihedral-scan + fit
sequence used to refit AMBER torsion parameters against a higher-level
reference (QM or ML). Two entry points:

* :func:`ffpopt.Workflows.run_dihed_twist_workflow` — single-molecule
  re-implementation of the bash-script workflow emitted by
  ``ffpopt-DihedTwistWorkflow.py``. Every scan is run in-process via
  :func:`ffpopt.WaveFront.run_dihed_wavefront`; fit / prepare steps still
  shell out to ``ffpopt-GenDihedFit.py`` / ``ffpopt-PrepareInput.py``.
* :func:`ffpopt.Workflows.run_fragmented_dihed_twist_workflow` — fragment
  the parent ligand with ``scission`` (from FragmentMol), run the twist
  workflow on each fragment, then splice the fitted DIHE terms back into
  a unified parent ``frcmod`` via ``scission.merge.merge_fragment_frcmods``.

Both APIs require an ``if __name__ == "__main__":`` guard in the caller's
entry script — the wavefront uses ``multiprocessing.set_start_method('spawn')``.

Single-molecule twist
---------------------

Phases:

#. High-level scan per bond at ``model`` (default ``sander``).
#. Reference ``sander`` scan per bond (``orig`` prefix).
#. *(Optional, Phase 2b.)* Drop bonds whose HL and reference scans already
   agree — gated by ``skip_converged_initial`` (default ``True``).
#. Iterative refinement (1..``maxiter``): write ``itNN.fit.json``, run
   ``ffpopt-GenDihedFit.py`` → ``itNN.py``, apply the fit, rebuild
   ``itNN.json``, sander-scan again, compare HL vs ``itNN`` per bond.
   Bonds that converge can be dropped, kept, or ignored based on
   ``convergence_mode`` (``drop`` / ``all_or_nothing`` / ``off``).

The phase blocks in
:func:`~ffpopt.Workflows.run_dihed_twist_workflow` are kept self-contained
on purpose — copy the function and edit a block to add a verification
stage or reorder phases.

Fragmented twist
----------------

Drives ``scission.fragment_ligand`` on the parent ligand triplet
(``mol2`` + ``lib`` + ``frcmod``), runs
:func:`~ffpopt.Workflows.run_dihed_twist_workflow` independently inside
each fragment directory (``ffpopt-PrepareInput.py`` → twist), and then
merges each fragment's highest-numbered ``itXX.frcmod`` into the parent
via ``scission.merge.merge_fragment_frcmods``.

The on-disk layout under ``out_dir/`` after a run::

    out_dir/
      fragment_index.json          ← scission's top-level index
      summary.json
      fragment_1/
        fragment.mol2, .lib, .frcmod, .parm7, .rst7
        fragment_2d.svg              ← scission overview
        torsion_<label>.svg          ← per-torsion drawings
        manifest.json, fit_torsions.json
        start.json                   ← from PrepareInput
        qdpi2_<idxs>.json/.dat       ← HL scans
        orig_<idxs>.json/.dat        ← reference sander scans
        it01.fit.json, it01.py, it01.parm7, it01.json, it01.frcmod
        it01_<idxs>.json/.dat        ← per-iteration scans
        compare_qdpi2_vs_orig_<idxs>.png   ← comparison plots
        compare_qdpi2_vs_it01_<idxs>.png
        ...
      fragment_2/ ...
    <merged_frcmod>                 ← merged parent frcmod
    <merged_frcmod>.merge_report.json

Re-running
~~~~~~~~~~

Both functions accept ``skip_existing=True`` (default), which short-circuits
every wavefront scan, ``GenDihedFit`` call, and ``apply+PrepareInput`` step
when its outputs already exist. The fragmented workflow extends this to
the scission step itself: when ``out_dir/fragment_index.json`` exists, the
``scission.fragment_ligand`` call is skipped entirely and the per-fragment
loop reads ``fit_torsions.json`` plus ``fragment.parm7/rst7`` directly off
disk. The net effect: re-running with the same args **only regenerates
the comparison plots and the merged frcmod** — no scans rerun, no
``tleap``, no QM/ML calls. Pass ``skip_existing=False`` to force a
fresh run.

Comparison plots
~~~~~~~~~~~~~~~~

When ``plot_comparisons=True`` (default ``False`` for the single-molecule
workflow, default ``True`` for the fragmented workflow), each comparison
is also rendered as a PNG alongside the matching ``.dat`` file. The
fragmented workflow additionally locates the matching scission torsion
SVG via each fragment's ``manifest.json`` and renders it as a top panel
on the plot — see :doc:`ScanAnalysis` for the panel layout and the
failure-criteria text box, and the worked example at
:doc:`../../UserDocs/Examples/ScissionInterface/tutorial` for an
end-to-end walkthrough using the ``examples/scission-interface`` ligand
triplet.

Module reference
----------------

.. automodule:: ffpopt.Workflows
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
