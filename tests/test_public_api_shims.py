"""Import-compat smoke tests for public facades after modular splits."""

from __future__ import annotations

import importlib
import unittest


class TestFfpoptPublicShims(unittest.TestCase):
    def test_geomopt_facade(self):
        m = importlib.import_module("ffpopt.GeomOpt")
        for name in (
            "GeomOpt",
            "GeomOpt_ASE",
            "GeomOpt_GEOMETRIC",
            "bare_potential_energy",
            "opt_recovery_label",
            "is_soft_opt_recovery",
            "FwdRevDihedScan",
            "ParallelGeomOpt",
        ):
            self.assertTrue(hasattr(m, name), name)

    def test_wavefront_facade(self):
        m = importlib.import_module("ffpopt.WaveFront")
        for name in (
            "Wavefront",
            "WavefrontNode",
            "WavefrontLevel",
            "run_dihed_wavefront",
            "wavefront_loader",
        ):
            self.assertTrue(hasattr(m, name), name)

    def test_wavefront_nd_facade(self):
        m = importlib.import_module("ffpopt.WaveFrontND")
        for name in ("Wavefront", "WavefrontNode", "run_dihed_wavefront", "wavefront_loader"):
            self.assertTrue(hasattr(m, name), name)

    def test_workflows_facade(self):
        m = importlib.import_module("ffpopt.Workflows")
        for name in (
            "run_dihed_twist_workflow",
            "run_fragmented_dihed_twist_workflow",
            "normalize_bond_pairs0",
            "bonds0_from_scission_fit_torsions",
        ):
            self.assertTrue(hasattr(m, name), name)

    def test_dihedrals_facade(self):
        m = importlib.import_module("ffpopt.Dihedrals")
        for name in (
            "FitInputType",
            "NonlinearSolve",
            "WriteParmedScript",
            "FindPuckers",
            "align_scan_profiles",
        ):
            self.assertTrue(hasattr(m, name), name)

    def test_runtime_compat_paths(self):
        for mod in (
            "ffpopt.cpu_budget",
            "ffpopt.fast_wavefront",
            "ffpopt.console",
            "ffpopt.progress_board",
            "ffpopt.fragment_progress",
        ):
            importlib.import_module(mod)


class TestLigandparamStageShims(unittest.TestCase):
    def test_abstract_stage(self):
        from ligandparam.stages.abstractstage import AbstractStage

        self.assertTrue(isinstance(AbstractStage, type))

    def test_gaussian_stage_names(self):
        # Import module directly (stages.__init__ eagerly pulls optional deps).
        import importlib

        m = importlib.import_module("ligandparam.stages.gaussian")
        for name in (
            "GaussianMinimizeRESP",
            "GaussianRESP",
            "StageGaussianRotation",
            "StageGaussiantoMol2",
        ):
            self.assertTrue(isinstance(getattr(m, name), type), name)


if __name__ == "__main__":
    unittest.main()
