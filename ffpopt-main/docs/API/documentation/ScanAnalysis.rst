ScanAnalysis
============

Comparison utilities for relaxed dihedral scans. The heuristic in this
module looks at a *low-level* (e.g. ``sander``) scan and a *high-level*
(e.g. ``qdpi2``, ``psi4``, ``mace``) scan of the same dihedral and decides
whether the two profiles are close enough that the dihedral does not need
refitting.

Three criteria, all tunable via :class:`ScanCompareConfig`:

* **Location of extrema** — maxima and minima of the two profiles must lie
  at similar angles (``angle_tol``, default 15°, with proper 0°/360° wrap).
* **Identity of extrema** — by default a maximum can only match a maximum
  and a minimum can only match a minimum (``require_extremum_identity``).
* **Barrier height** — both the overall profile barrier delta
  (``barrier_tol``) and each matched-pair energy delta (``energy_tol``)
  must fall under thresholds.

A short-circuit applies first: if the high-level barrier is below
``flat_threshold`` (default 1 kcal/mol), the dihedral is treated as "soft"
and reported as ``is_close=True`` / ``is_flat=True`` with no further
analysis.

Verdicts are returned as a :class:`ScanComparison` dataclass with a boolean
``is_close``, both barrier heights, the detected extrema on each side, the
matched / unmatched pair indices, and a list of human-readable ``reasons``
explaining each criterion failure (empty when ``is_close`` is True).

The heuristic is wired into
:func:`ffpopt.Workflows.run_dihed_twist_workflow` to drop dihedrals that
already agree from the iterative-fit phase (Phase 2b and Phase 3e — see
the ``skip_converged_initial`` and ``convergence_mode`` kwargs there).
Callers can also invoke it directly on raw arrays or on scan files.

Quick usage
-----------

On raw scan arrays from
:func:`ffpopt.WaveFront.run_dihed_wavefront`::

    from ffpopt.ScanAnalysis import compare_scans

    result = compare_scans(
        hl_angles=hl_result["angles"],
        hl_energies=hl_result["energies"],
        ll_angles=ll_result["angles"],
        ll_energies=ll_result["energies"],
    )
    if result.is_close:
        print("dihedral converged — no refit needed")
    else:
        for r in result.reasons:
            print("refit because:", r)

On scan ``.dat`` or ``.json`` files written by the wavefront::

    from ffpopt.ScanAnalysis import compare_scan_files, ScanCompareConfig

    cfg = ScanCompareConfig(angle_tol=15.0, energy_tol=0.5)
    result = compare_scan_files("qdpi2_0-1-2-3.dat", "orig_0-1-2-3.dat", cfg)

Plot output
-----------

:func:`plot_comparison` renders the comparison as a one-, two-, or
three-panel PNG, depending on what's available:

* **Top panel** — 2D fragment structure (when ``structure_image_path`` is
  set; PNG/JPG load directly, SVG is rasterized via ``cairosvg`` if
  importable, otherwise via the ``rsvg-convert`` system binary).
* **Middle panel** — the two min-shifted scan profiles, with extrema
  highlighted (up-triangle = maximum, down-triangle = minimum; filled for
  matched, open red for unmatched), and dotted connectors between matched
  pairs colored by ``|ΔE| / energy_tol`` (green ≤ tol, red > tol).
* **Bottom panel** — a red-bordered list of every failed criterion from
  ``comparison.reasons`` (omitted when the verdict is ``OK``).

The figure title reports the overall verdict (``OK``, ``FLAT``, or
``FAIL``) and both barrier heights.

When invoked indirectly via :func:`compare_scan_files` with the
``plot_path=`` kwarg, the plot is saved beside the scan ``.dat`` files;
the workflow wires this through automatically when
``plot_comparisons=True`` (default ``True`` for the fragmented workflow).

Example output — a high-level ``qdpi2`` scan compared against the untuned
reference ``sander`` scan (``orig``) shows two unmatched extrema, a 4
kcal/mol barrier mismatch, and a max/min count differential:

.. image:: ../../_static/scan_compare_fail.png
   :alt: Three-panel comparison plot: structure top, scans middle, failed criteria bottom.
   :width: 90%

After six iterations of dihedral-parameter fitting, the same scan agrees
within thresholds — no failure criteria, no bottom panel:

.. image:: ../../_static/scan_compare_ok.png
   :alt: Two-panel comparison plot: structure top, agreeing scans below.
   :width: 90%

Both plots are written automatically when running the fragmented twist
workflow (see :doc:`Workflows`); the structure drawings come from
``scission``'s per-torsion SVGs.

Worked example
--------------

The two scans below come from the same fragment, same torsion (atoms
``0-1-2-3``); the high-level reference is ``qdpi2``, and the two
low-level candidates are an untuned ``sander`` scan (``orig``) and a
``sander`` scan after six iterations of dihedral-parameter fitting
(``it06``):

============ ============== ====================== ==========
LL profile   ``is_close``   Why                    Action
============ ============== ====================== ==========
``orig``     ``False``      Wells at 60° vs 120°   Refit
``it06``     ``True``       Same extrema, ΔE<0.2   Drop
============ ============== ====================== ==========

For ``orig`` the heuristic flags two unmatched extrema (the HL minimum at
60° has no LL counterpart within 15°), one unmatched LL extremum (at
120°), and a 2.93 kcal/mol energy delta on the one matched minimum at
240°. For ``it06`` every HL extremum has a 0.0° angle delta to its LL
counterpart, every matched-pair energy delta is under 0.5 kcal/mol, and
the overall barrier delta is 0.07 kcal/mol — well inside every default
threshold.

Tuning
------

Loosen all four numeric thresholds if you want the heuristic to skip more
dihedrals; tighten them if you want it to keep more. The defaults assume
the wavefront's ``--delta`` is in the 10–30° range and that scans are
reported in kcal/mol (which is what
:func:`ffpopt.WaveFront.run_dihed_wavefront` writes to the companion
``.dat`` file). Set ``require_extremum_identity=False`` to allow a
maximum to match a minimum if you only care about angle positions; set
``require_same_count=False`` to ignore differences in the number of
detected extrema (e.g. when one profile has a tiny extra wiggle that
just clears ``prominence``).

Module reference
----------------

.. automodule:: ffpopt.ScanAnalysis
   :members:
   :show-inheritance:
   :special-members: __init__
