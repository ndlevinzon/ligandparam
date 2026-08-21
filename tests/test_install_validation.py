"""Install validation suite for new users.

Run after ``pip install -e .`` (or ``pip install .``)::

    python -m unittest tests.test_install_validation -v

These tests exercise real import paths, CLI entry points, and a few
non-trivial numerical / I/O helpers. They intentionally avoid AmberTools,
Gaussian, and GPU stacks so a basic conda/pip install can pass on a laptop.

Optional extras (``tblite``, ``geometric``, …) are checked when present and
skipped with an explicit reason when absent.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


class TestCorePackageInstall(unittest.TestCase):
    """Top-level packages resolve and report a coherent version."""

    def test_ligandparam_version(self):
        import ligandparam

        self.assertTrue(ligandparam.__version__)
        self.assertRegex(ligandparam.__version__, r"^\d+\.\d+")
        # Packaging metadata should match the in-tree version when installed.
        try:
            dist_ver = importlib.metadata.version("ligandparam")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("ligandparam not installed as a distribution")
        self.assertEqual(dist_ver, ligandparam.__version__)

    def test_import_integrated_packages(self):
        import ffpopt
        import ligandparam
        import scission

        self.assertTrue(hasattr(ligandparam, "__version__"))
        self.assertTrue(hasattr(ffpopt, "Workflows") or callable(getattr(ffpopt, "__getattr__", None)))
        self.assertTrue(hasattr(scission, "fragment_ligand"))

    def test_core_scientific_dependencies(self):
        missing = []
        for name in (
            "numpy",
            "scipy",
            "pandas",
            "parmed",
            "ase",
            "networkx",
            "yaml",
            "MDAnalysis",
            "rdkit",
        ):
            if not _has_module(name):
                missing.append(name)
        self.assertEqual(
            missing,
            [],
            "Missing required dependencies: "
            f"{missing}. Install with: pip install -e . "
            "(use the project conda/mamba env from env.yaml when possible).",
        )

    def test_numpy_runtime_usable(self):
        import numpy as np

        a = np.linspace(0.0, 1.0, 5)
        self.assertAlmostEqual(float(a.sum()), 2.5, places=10)

    def test_rdkit_smiles_roundtrip(self):
        if not _has_module("rdkit"):
            self.fail(
                "rdkit is a required dependency but is not importable. "
                "Install with: pip install -e . (or conda install -c conda-forge rdkit)"
            )
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumAtoms(), 3)


class TestPublicAPISurface(unittest.TestCase):
    """Canonical modules and symbols used by recipes / CLIs must import."""

    def test_ligandparam_stages_and_recipes(self):
        from ligandparam.recipes.registry import available_recipes, get_recipe
        from ligandparam.stages.abstractstage import AbstractStage

        names = available_recipes()
        self.assertIsInstance(names, (list, tuple))
        self.assertIn("freeligand", names)
        self.assertIn("lazyligand", names)
        with self.assertRaises(ValueError):
            get_recipe("not-a-real-recipe")
        self.assertTrue(issubclass(AbstractStage, object))

        # Module files must be present (find_spec does not execute heavy imports).
        for modname in (
            "ligandparam.recipes.freeligand",
            "ligandparam.recipes.lazyligand",
            "ligandparam.stages.gaussian",
        ):
            self.assertIsNotNone(
                importlib.util.find_spec(modname),
                f"missing module {modname}",
            )

        # Import gaussian stages (no RDKit at module top in typical installs).
        m = importlib.import_module("ligandparam.stages.gaussian")
        for name in (
            "GaussianMinimizeRESP",
            "GaussianRESP",
            "StageGaussianRotation",
            "StageGaussianToMol2",
            "StageGaussiantoMol2",
        ):
            self.assertTrue(isinstance(getattr(m, name), type), name)
        self.assertIs(m.StageGaussiantoMol2, m.StageGaussianToMol2)

        if _has_module("rdkit"):
            pdb_names = importlib.import_module("ligandparam.stages.pdb_names")
            self.assertIs(pdb_names.PDB_Name_Fixer, pdb_names.StagePdbNameFixer)

    def test_gaussian_and_smiles_stages(self):
        if not _has_module("rdkit"):
            self.skipTest(
                "rdkit required for smiles stages; install with: pip install -e ."
            )
        smiles = importlib.import_module("ligandparam.stages.smiles_to_pdb")
        self.assertTrue(hasattr(smiles, "StageSmilesToPDB"))

    def test_ffpopt_workflow_surface(self):
        m = importlib.import_module("ffpopt.Workflows")
        for name in (
            "run_dihed_twist_workflow",
            "run_fragmented_dihed_twist_workflow",
            "normalize_bond_pairs0",
            "bonds0_from_scission_fit_torsions",
        ):
            self.assertTrue(callable(getattr(m, name)), name)

    def test_ffpopt_geomopt_and_dihedrals(self):
        geom = importlib.import_module("ffpopt.GeomOpt")
        for name in (
            "GeomOpt",
            "GeomOpt_ASE",
            "GeomOpt_GEOMETRIC",
            "bare_potential_energy",
            "is_soft_opt_recovery",
            "opt_recovery_label",
        ):
            self.assertTrue(hasattr(geom, name), name)

        dihed = importlib.import_module("ffpopt.Dihedrals")
        for name in (
            "FitInputType",
            "NonlinearSolve",
            "WriteParmedScript",
            "align_scan_profiles",
            "shape_match_delta",
            "joint_linear_solve_from_caches",
        ):
            self.assertTrue(hasattr(dihed, name), name)

    def test_wavefront_1d_nd_and_pickle_aliases(self):
        wf = importlib.import_module("ffpopt.scan.WaveFront")
        for name in ("Wavefront", "WavefrontNode", "run_dihed_wavefront", "wavefront_loader"):
            self.assertTrue(hasattr(wf, name), name)

        wnd = importlib.import_module("ffpopt.scan.WaveFrontND")
        for name in ("Wavefront", "WavefrontNode", "run_dihed_wavefront"):
            self.assertTrue(hasattr(wnd, name), name)

        # Pre-scan/ pickle paths must resolve for checkpoint resume.
        legacy = importlib.import_module("ffpopt.WaveFront")
        self.assertIs(legacy.Wavefront, wf.Wavefront)

        runtime = (
            "ffpopt.runtime.console",
            "ffpopt.runtime.cpu_budget",
            "ffpopt.runtime.fast_wavefront",
            "ffpopt.runtime.nondaemon_pool",
            "ffpopt.runtime.progress_board",
            "ffpopt.scan.wavefront_mixins",
            "ffpopt.scan.ScanAnalysis",
            "ffpopt.affdo_log",
            "ffpopt.profile_select",
            "ffpopt.boltzmann_charges",
            "ffpopt.dihed_fit_ext",
        )
        for mod in runtime:
            importlib.import_module(mod)

        pool_mod = importlib.import_module("ffpopt.runtime.nondaemon_pool")
        self.assertTrue(callable(pool_mod.make_nondaemon_spawn_pool))

        writers = importlib.import_module("scission.writers")
        self.assertEqual(writers.safe_name("a/b c"), "a_b_c")
        frcmod = importlib.import_module("scission.frcmod")
        key = frcmod._normalize_param_name_to_key("LIG_ca-c3-c-o")
        self.assertIsNotNone(key)
        self.assertEqual(len(key), 4)
        self.assertEqual(key, frcmod._normalize_dihe_key(key))

    def test_scission_public_api(self):
        import scission

        for name in (
            "fragment_ligand",
            "FragmentConfig",
            "SelectedFragment",
            "match_central_bond_smarts",
        ):
            self.assertTrue(hasattr(scission, name), name)

        merge = importlib.import_module("scission.merge")
        self.assertTrue(callable(merge.merge_fragment_frcmods))
        self.assertTrue(callable(merge.list_iteration_frcmods))


class TestCLIEntrypoints(unittest.TestCase):
    """Console scripts declared in pyproject must be importable callables."""

    EXPECTED = (
        ("ligandparam.cli.lig_dihed_correct", "main"),
        ("ligandparam.cli.lig_scission", "main"),
        ("scission.cli", "main"),
    )
    # These pull RDKit at import time (core dep; skipped only if RDKit absent).
    RDKit_EXPECTED = (
        ("ligandparam.cli.ligandparam_getparam", "main"),
        ("ligandparam.cli.smiles_to_pdb", "main"),
    )

    def test_cli_modules_expose_main(self):
        for modname, attr in self.EXPECTED:
            with self.subTest(module=modname):
                mod = importlib.import_module(modname)
                fn = getattr(mod, attr)
                self.assertTrue(callable(fn))

    def test_cli_modules_requiring_rdkit(self):
        if not _has_module("rdkit"):
            self.skipTest(
                "rdkit required for lig-getparam / smiles-to-pdb; "
                "install with: pip install -e ."
            )
        for modname, attr in self.RDKit_EXPECTED:
            with self.subTest(module=modname):
                mod = importlib.import_module(modname)
                self.assertTrue(callable(getattr(mod, attr)))

    def test_installed_console_scripts_when_available(self):
        try:
            dist = importlib.metadata.distribution("ligandparam")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("ligandparam distribution metadata unavailable")
        ep_names = {ep.name for ep in dist.entry_points if ep.group == "console_scripts"}
        for required in (
            "lig-getparam",
            "lig-dihed-correct",
            "lig-scission",
            "scission",
            "ffpopt-specialty",
        ):
            self.assertIn(required, ep_names, f"missing console script {required}")
        for banned in (
            "ffpopt-DihedTwistAnimate.py",
            "ffpopt-WavefrontAnimate.py",
            "ffpopt-FindSugarPuckers.py",
            "ffpopt-Json2Img.py",
            "ffpopt-JsonJoin.py",
            "ffpopt-JsonSplit.py",
            "ffpopt-Json2Crds.py",
            "ffpopt-DeltaPuckerFit.py",
            "ffpopt-WavefrontToDP.py",
        ):
            self.assertNotIn(banned, ep_names, f"specialty tool should not be a console script: {banned}")


class TestBehavioralSmoke(unittest.TestCase):
    """Small but real computations that catch broken installs."""

    def test_shape_match_delta_invariant_to_constant(self):
        import numpy as np
        from ffpopt.Dihedrals import shape_match_delta

        hl = np.array([1.0, 3.0, 2.0, 4.0])
        ll = np.array([0.5, 1.5, 1.0, 2.0])
        d0 = shape_match_delta(hl, ll)
        d1 = shape_match_delta(hl, ll + 12.0)
        np.testing.assert_allclose(d0, d1)
        self.assertAlmostEqual(float(np.mean(d0)), 0.0, places=12)

    def test_wavefront_evaluate_policy(self):
        from ffpopt.GeomOpt import is_soft_opt_recovery
        from ffpopt.scan.wavefront_mixins import evaluate_wavefront_minimum

        self.assertTrue(is_soft_opt_recovery("loose"))
        self.assertTrue(is_soft_opt_recovery("soft-maxiter"))
        seed = evaluate_wavefront_minimum(
            energy=1.0,
            soft=True,
            has_incumbent=False,
            incumbent_energy=None,
            incumbent_soft=False,
            threshold_ev=0.05,
        )
        self.assertTrue(seed["update_min"])
        self.assertTrue(seed["active"])

        quiet = evaluate_wavefront_minimum(
            energy=0.99,
            soft=False,
            has_incumbent=True,
            incumbent_energy=1.0,
            incumbent_soft=False,
            threshold_ev=0.05,
        )
        self.assertTrue(quiet["update_min"])
        self.assertFalse(quiet["active"])

    def test_align_scan_profiles_on_shared_angles(self):
        if not _has_module("ase"):
            self.skipTest("ase required for Struct import path")
        from ffpopt.Dihedrals import align_scan_profiles
        from ffpopt.Struct import ListOfStruct

        def _frame(name: str, energy: float = 0.0):
            return SimpleNamespace(
                data={
                    "name": name,
                    "energy": energy,
                    "positions": [[0.0, 0.0, 0.0]],
                    "constraints": [],
                },
                constraints=None,
            )

        hl = ListOfStruct.from_structs_shared(
            [_frame("d000"), _frame("d010"), _frame("d020"), _frame("d030")]
        )
        ll = ListOfStruct.from_structs_shared(
            [_frame("d010"), _frame("d020"), _frame("d030"), _frame("d040")]
        )
        ahl, all_, info = align_scan_profiles(hl, ll)
        self.assertEqual(info["n_common"], 3)
        self.assertEqual(len(ahl.structs), 3)
        self.assertEqual(len(all_.structs), 3)

    def test_startup_banner_once(self):
        from ffpopt.runtime import console as console_mod
        from ffpopt.runtime.console import print_startup_banner

        console_mod._BANNER_PRINTED = False
        os.environ.pop("LIGANDPARAM_BANNER_PRINTED", None)
        try:
            buf = io.StringIO()
            self.assertTrue(print_startup_banner(stream=buf))
            self.assertIn("ligandparam", buf.getvalue())
            self.assertIn("Authors:", buf.getvalue())
            console_mod._BANNER_PRINTED = False  # simulate worker re-import
            self.assertFalse(print_startup_banner(stream=buf))
        finally:
            console_mod._BANNER_PRINTED = False
            os.environ.pop("LIGANDPARAM_BANNER_PRINTED", None)

    def test_scission_frcmod_merge_accumulates_iterations(self):
        from scission.merge import _load_fragment_update

        def _frcmod(dihe_lines: list[str]) -> str:
            return (
                "Remark line goes here\n"
                "MASS\n\n"
                "BOND\n\n"
                "ANGLE\n\n"
                "DIHE\n"
                + "".join(f"{line}\n" for line in dihe_lines)
                + "\nIMPROPER\n\nNONB\n\n"
            )

        with tempfile.TemporaryDirectory() as td:
            frag = Path(td)
            (frag / "it01.frcmod").write_text(
                _frcmod(["c3-c3-c3-c3 1 1.00 0.0 1.", "c3-c3-c3-n  1 2.00 0.0 1."])
            )
            (frag / "it02.frcmod").write_text(
                _frcmod(["c3-c3-c3-n  1 3.50 0.0 1."])
            )
            update = _load_fragment_update(frag)
            keys = set(update["dihe_groups"].keys())
            self.assertIn(("c3", "c3", "c3", "c3"), keys)
            self.assertIn(("c3", "c3", "c3", "n"), keys)

    def test_constraints_to_geometric_roundtrip(self):
        from ffpopt.Constraints import Constraint, to_geometric

        cons = [Constraint("dihed", [0, 1, 2, 3], value=90.0)]
        lines = to_geometric(cons)
        self.assertTrue(any("90.0" in ln for ln in lines))
        self.assertTrue(any(ln.startswith("dihedral") for ln in lines))


class TestOptionalExtras(unittest.TestCase):
    """Optional stacks: pass when installed, skip cleanly otherwise."""

    def test_geometric_optional(self):
        if not _has_module("geometric"):
            self.skipTest("optional: geometric (pip install '.[dihed]')")
        import geometric  # noqa: F401

    def test_tblite_optional(self):
        if not _has_module("tblite"):
            self.skipTest("optional: tblite (pip install '.[tblite]')")
        import tblite  # noqa: F401

    def test_ambertools_on_path_optional(self):
        found = {
            name: shutil.which(name)
            for name in ("antechamber", "parmchk2", "tleap")
        }
        if not any(found.values()):
            self.skipTest(
                "optional: AmberTools not on PATH "
                "(needed for lig-getparam / lig-dihed-correct production runs)"
            )
        for name, path in found.items():
            if path is None:
                continue
            self.assertTrue(Path(path).exists(), name)


if __name__ == "__main__":
    unittest.main()
