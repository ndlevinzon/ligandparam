Companion packages (ffpopt / scission)
======================================

This page is the contract ligandparam uses to talk to ``ffpopt`` and
``scission``. Send it to those teams when they implement an independent
package that ligandparam can load **instead of** the in-tree copies under
``src/ffpopt`` and ``src/scission``.

ligandparam keeps the bundled trees. Switching is a resolver setting, not a
rewrite of stages or CLIs. Import names stay ``ffpopt`` and ``scission``.

How ligandparam chooses a tree
------------------------------

Set these **before** starting the process (``export`` / Slurm ``--export``).
They are read once, on the first ``import ffpopt`` or ``import scission``
after ``import ligandparam``.

=============================== ===========================================
Variable                        Meaning
=============================== ===========================================
``LIGANDPARAM_FFPOPT``          ``internal`` (default) or ``external``
``LIGANDPARAM_SCISSION``        ``internal`` (default) or ``external``
``LIGANDPARAM_FFPOPT_PATH``     Directory for external ffpopt (see below)
``LIGANDPARAM_SCISSION_PATH``   Directory for external scission
=============================== ===========================================

A PATH value is either:

* the directory that **contains** the package (``.../src`` with
  ``src/ffpopt/__init__.py``), or
* the package directory itself (``.../src/ffpopt``).

``internal`` always loads the copy sitting next to ``ligandparam`` (editable:
``<repo>/src``; wheel: ``site-packages``). ``external`` loads the PATH tree
and prepends it to ``sys.path`` and ``PYTHONPATH`` so ffpopt ``bin/``
subprocesses (``ffpopt-GenDihedFit.py``, ...) import the same copy.

Example: independent checkouts, bundled ligandparam still installed::

    export LIGANDPARAM_FFPOPT=external
    export LIGANDPARAM_FFPOPT_PATH=/path/to/ffpopt/src
    export LIGANDPARAM_SCISSION=external
    export LIGANDPARAM_SCISSION_PATH=/path/to/scission/src
    lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 10 --fast

The three ``lig-*`` CLIs print one ASCII confirmation line after the banner::

    companions: ffpopt=external (/path/to/ffpopt/src/ffpopt) scission=internal (...)

Query the same map from Python::

    from ligandparam.companions import companion_status, format_status_line
    print(format_status_line())

Two pip distributions **cannot** both own the top-level name ``ffpopt``.
PATH is how an independent tree coexists with the bundle. Pointing
``LIGANDPARAM_FFPOPT=external`` at the bundled tree is an error (the bundle
sets ``__ligandparam_bundle__ = True`` on ``ffpopt`` / ``scission``).

The ``scission`` console script goes through
``ligandparam.cli.LigScission.scission_console`` so it honors the same env.
Direct ``import ffpopt`` **without** importing ``ligandparam`` first uses
whatever is already on ``sys.path`` (no resolver).

Why there are two layers
------------------------

Do not ship only the torsion workflows. ligandparam already imports ffpopt
for parameterization, even when the user never runs ``lig-dihed-correct``.

**Layer A (science).** Dihedral correction and fragmentation. Required for
``lig-dihed-correct`` / recipe ``dihed_correct`` / ``lig-scission``.

**Layer B (runtime).** Logging, banners, Gaussian orientation boards,
``CopyParm``. Required for ``lig-getparam``. If an external ffpopt omits
these, either keep the names below stable or ligandparam will stop importing
``ffpopt.runtime`` / ``ffpopt.AmberParm`` (that is a ligandparam change, not
an ffpopt one).

scission must **not** import ffpopt or ligandparam. ffpopt workflows **do**
import scission. Fragmented twist therefore needs both packages even when
ligandparam only calls ``ffpopt.workflows``.

Call graph
----------

::

    lig-getparam
        ligandparam.Log                 -> ffpopt.runtime.Console
        stages.Gaussian                 -> split_gaussian_orientation_budget
                                        -> JobProgressStore / JobBoardWatcher
        multiresp.ParmEdUtils.CopyParm  -> ffpopt.AmberParm.CopyParm

    lig-dihed-correct / StageDihedTwistCorrection
        ffpopt.workflows.run_fragmented_dihed_twist_workflow
            scission.Models.InputBundle / FragmentConfig
            scission.fragment_ligand
            scission.Merge.merge_fragment_frcmods
            scission.Writers.safe_name
        ffpopt.workflows.run_whole_ligand_dihed_twist_workflow
            scission.LigandIo.load_ligand_from_mol2
            scission.Torsions.find_rotatable_bonds
        ffpopt.affdo.AffdoLog.describe_affdo_extras / log_affdo
        scission.Models.FragmentConfig  (coerce_fragment_config)

    lig-scission
        expands -d / -r / --label to --mol2 / --lib / --frcmod / --outdir
        scission.Cli.main(argv) -> int

    AmberLigandBundle.to_scission_input
        scission.Models.InputBundle(mol2_path, lib_path, frcmod_path, ligand_name=)

Layer A: ffpopt science API
---------------------------

Call both workflows from ``if __name__ == "__main__":`` (wavefront uses
spawn-mode multiprocessing).

Fragmented (default ``lig-dihed-correct``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def run_fragmented_dihed_twist_workflow(
       *, mol2, lib, frcmod, out_dir, merged_frcmod,
       model, maxiter, nprim, delta, nproc,
       geometric_opt, skip_existing,
       rotatable_bond_smarts=None, fragment_config=None,
       fast_wavefront=None, multi_centroid=0,
       centroid_mol2=None, fit_cli_args=None,
       logger=None, **standard_kwargs,
   ) -> dict

Keyword names ligandparam actually forwards: ``mol2``, ``lib``, ``frcmod``,
``out_dir``, ``merged_frcmod``, ``model``, ``maxiter``, ``nprim``, ``delta``,
``nproc``, ``geometric_opt``, ``skip_existing``, ``rotatable_bond_smarts``,
``fragment_config``, ``fast_wavefront``, ``multi_centroid``, ``centroid_mol2``,
``fit_cli_args``, ``logger``, plus optional ``geometric_maxiter``,
``geometric_converge``, ``ase_opt_tol``, ``soft_dihed_restraint``,
``soft_dihed_k``, ``soft_dihed_kmax``, ``soft_dihed_tol``.

Return keys ligandparam reads:

.. code-block:: python

   {
       "merged_frcmod": str,           # required
       "fragments": [                  # required for the log line
           {"fragment_id": str, "dir": str, "bonds": ..., "twist_result": ...},
           ...
       ],
       "fragmentation": dict | None,
       "merge_report": dict,
   }

Whole-ligand (``--whole-ligand``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def run_whole_ligand_dihed_twist_workflow(
       *, mol2, lib, frcmod, out_dir, out_frcmod,
       model, maxiter, nprim, delta, nproc,
       geometric_opt, skip_existing,
       rotatable_bond_smarts=None, fast_wavefront=None,
       multi_centroid=0, boltzmann_charges=False,
       fit_cli_args=None, logger=None, **standard_kwargs,
   ) -> dict

Note ``out_frcmod``, not ``merged_frcmod``. Return keys ligandparam reads:

.. code-block:: python

   {
       "out_frcmod": str,
       "bonds": list,          # 0-based pairs; logged
       "boltzmann_charges": None | {"out_mol2": ..., "out_lib": ...},
       "out_dir": str,
       "twist": dict,          # unused by ligandparam today
   }

AffdoLog
~~~~~~~~

.. code-block:: python

   describe_affdo_extras(
       *, whole_ligand=False, multi_centroid=0, boltzmann_charges=False,
       soft_dihed_restraint=False, soft_dihed_k=None, soft_dihed_kmax=None,
       soft_dihed_tol=None, fit_cli_args=None,
   ) -> str
   # ASCII one-liner. Soft-dihed units: k in kcal/mol/rad^2, tol in deg.

   log_affdo(logger, msg, *args) -> None
   # logger.info("[affdo] " + msg, *args)

Index convention: ffpopt bonds are **0-based**. Scission ``fit_torsions``
are **1-based**; the bundled fragmented workflow converts at the boundary
(``bonds0_from_scission_fit_torsions``). Keep that conversion on the ffpopt
side so ligandparam can keep passing scission objects through.

Layer A: scission API
---------------------

Types ligandparam constructs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class InputBundle:
       mol2_path: Path
       lib_path: Path
       frcmod_path: Path
       ligand_name: str | None = None

   @dataclass(frozen=True)
   class FragmentConfig:
       angle_step: int = 30
       torsion_scope: str = "acyclic_rotatable"
       include_rigid_single_bonds: bool = True
       rotatable_bond_smarts: tuple[str, ...] = ()
       restrict_to_bond_smarts: tuple[str, ...] = ()
       cap_strategy: str = "chemistry_aware"
       cap_h_min_charge: float = 0.0
       preserve_torsion_neighborhood: bool = True
       torsion_neighborhood_radius: int = 1
       preserve_conjugated_caps: bool = True
       optimization_target: str = "fewest_total_fragments"
       clash_thresholds: ClashThresholds
       preserve_conjugated_neighbors: bool = True
       use_parent_fallback: bool = False
       nproc: int = 1

       @classmethod
       def from_dict(cls, payload: dict) -> FragmentConfig: ...

``from_dict`` must accept a JSON/YAML mapping, including nested
``clash_thresholds``. SMARTS that mark a rotatable bond use ``:1`` and
``:2`` on the two central atoms. ligandparam may pass a live
``FragmentConfig`` **or** a dict (``coerce_fragment_config``).

Functions ffpopt (and thus ligandparam) calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fragment_ligand(bundle: InputBundle, out_dir: Path, config: FragmentConfig)
       -> FragmentationResult
       # .selected_fragments, .to_dict()

   Merge.merge_fragment_frcmods(
       parent_frcmod_path, output_frcmod_path,
       fragment_dirs, report_path=None,
   ) -> dict
       # JSON-serializable. Collect DIHE from all itXX.frcmod in each dir.

   LigandIo.load_ligand_from_mol2(mol2_path, ligand_name=None) -> Ligand

   Torsions.find_rotatable_bonds(
       ligand, include_rigid_single_bonds=True,
       rotatable_bond_smarts=(), graph=None,
   ) -> list[tuple[int, int]]
       # 0-based sorted pairs for whole-ligand twist.

   Writers.safe_name(text) -> str

   Cli.main(argv: list[str] | None = None) -> int

CLI argv after ``lig-scission`` shortcut expansion::

    fragment --mol2 <abs>/LIG.mol2 --lib <abs>/LIG.lib \
             --frcmod <abs>/LIG.frcmod --outdir <abs>/LIG.scission_fragments \
             [...user flags...]

Keep subcommands ``fragment``, ``merge``, ``pick-bond``. Extra flags already
on the in-tree CLI (``--config``, ``--include-bond-smarts``, ``--nproc``,
...) should stay.

Layer B: ffpopt runtime ligandparam already uses
------------------------------------------------

``lig-getparam`` imports these even when dihedral correction is off. Freeze
the names or coordinate a ligandparam change to stop depending on them.

``ffpopt.runtime.Console``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print_startup_banner() -> bool
   ascii_for_stdio(text) -> str
   format_console_line(text, tag=...) -> str
   attach_console_handlers(logger, tag=..., level=...)
   console_formatter(tag) -> logging.Formatter

Console writes must be ASCII (``+/-``, ``deg``, ``chi^2``) so latin-1 Slurm
``.out`` files do not mojibake. Leading ``[scope]`` tokens in log messages
are peeled into a bracket hierarchy
(``[ligandparam] [affdo] ...``).

``ffpopt.runtime.FastWavefront``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   split_gaussian_orientation_budget(nproc, n_jobs, mem_gb, *, min_mem_gb=4)
       -> (n_workers, job_nproc, job_mem_gb)

Invariants: ``n_workers * job_nproc <= nproc`` and
``n_workers * job_mem_gb <= mem_gb``. Do **not** flatten to one core per
orientation with the full ``--mem`` in every Gaussian ``%MEM`` (that OOMs
a 28-orientation ``so3_n28`` pool).

``ffpopt.runtime.ProgressBoard``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   JobProgressStore(
       path, *, collection_key="jobs", id_header="Job",
       title="Live job status", empty_hint=..., detail_hint_label=...,
   )
   store.register(job_id, *, status=, stage=, detail=, log_path=)
   store.update(...)
   store.snapshot() -> dict

   JobBoardWatcher(
       store, *, board_path, logger, interval_sec=5.0,
       log_root_hint=None, thread_name=...,
   ).start() / .stop()

Gaussian rotation uses ``collection_key="orientations"``, files
``ROT_STATUS.txt`` and ``.rot_progress.json``.

``ffpopt.AmberParm``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   CopyParm(parm) -> AmberParm   # shallow copy including coordinates and box

ligandparam re-exports this as ``ligandparam.multiresp.ParmEdUtils.CopyParm``.

Hard rules
----------

* Keep import names ``ffpopt.*`` and ``scission.*``.
* Parent Amber triplet is always ``mol2`` + ``lib`` + ``frcmod``. The
  ``.lib`` is not rewritten by twist; corrected torsions land in a new
  frcmod (``{label}.dihed.frcmod``).
* ffpopt atom indices are 0-based; scission parent indices are 1-based
  **inside** scission.
* Logs stay ASCII.
* scission must not import ffpopt or ligandparam.
* Public CLIs, recipe names, and pickle facades (``WaveFront`` /
  ``WaveFrontND``) stay stable. Specialty tools
  (``ffpopt-specialty``) are not part of this contract.
* Do not require Gaussian RESP / ``ligandparam.multiresp`` internals in
  ffpopt or scission.

Checklist for an external package
---------------------------------

ffpopt
~~~~~~

1. Implement the two ``run_*_workflow`` functions with the keyword names
   and return keys above.
2. Implement ``AffdoLog.describe_affdo_extras`` / ``log_affdo``.
3. Keep Layer B helpers **or** tell ligandparam you are dropping them so
   the runtime imports can move.
4. Do **not** set ``__ligandparam_bundle__ = True``.
5. Workflows that spawn ``ffpopt.bin`` scripts must honor ``sys.executable``
   and the caller's ``PYTHONPATH`` (ligandparam prepends your PATH).
6. Import scission for fragment + whole-ligand rotatable-bond discovery;
   do not vendor a second scission.

scission
~~~~~~~~

1. ``InputBundle``, ``FragmentConfig.from_dict``, ``fragment_ligand``,
   ``merge_fragment_frcmods``, ``load_ligand_from_mol2``,
   ``find_rotatable_bonds``, ``Writers.safe_name``, ``Cli.main``.
2. No reverse imports of ffpopt / ligandparam.
3. Do **not** set ``__ligandparam_bundle__ = True``.

Verify against this tree
~~~~~~~~~~~~~~~~~~~~~~~~

::

    export LIGANDPARAM_FFPOPT=external
    export LIGANDPARAM_FFPOPT_PATH=/path/to/your/ffpopt/src
    python -c "import ligandparam; import ffpopt; from ligandparam.companions import format_status_line; print(format_status_line()); print(ffpopt.__file__)"

    python -m unittest tests.test_install_validation tests.test_developer_regression -v

See also :doc:`ffpopt`, :doc:`scission`, :doc:`dihedrals`, :doc:`installation`.
