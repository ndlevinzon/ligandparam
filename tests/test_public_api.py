"""Smoke tests for canonical public module paths after package layout cleanup."""

from __future__ import annotations

import importlib
import unittest


class TestFfpoptPublicModules(unittest.TestCase):
    def test_geomopt(self):
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

    def test_wavefront(self):
        m = importlib.import_module("ffpopt.scan.WaveFront")
        for name in (
            "Wavefront",
            "WavefrontNode",
            "WavefrontLevel",
            "run_dihed_wavefront",
            "wavefront_loader",
        ):
            self.assertTrue(hasattr(m, name), name)

    def test_wavefront_nd(self):
        m = importlib.import_module("ffpopt.scan.WaveFrontND")
        for name in ("Wavefront", "WavefrontNode", "run_dihed_wavefront", "wavefront_loader"):
            self.assertTrue(hasattr(m, name), name)

    def test_workflows(self):
        m = importlib.import_module("ffpopt.Workflows")
        for name in (
            "run_dihed_twist_workflow",
            "run_fragmented_dihed_twist_workflow",
            "normalize_bond_pairs0",
            "bonds0_from_scission_fit_torsions",
        ):
            self.assertTrue(hasattr(m, name), name)

    def test_dihedrals(self):
        m = importlib.import_module("ffpopt.Dihedrals")
        for name in (
            "FitInputType",
            "NonlinearSolve",
            "WriteParmedScript",
            "FindPuckers",
            "align_scan_profiles",
        ):
            self.assertTrue(hasattr(m, name), name)

    def test_runtime_and_scan_packages(self):
        for mod in (
            "ffpopt.runtime.cpu_budget",
            "ffpopt.runtime.fast_wavefront",
            "ffpopt.runtime.console",
            "ffpopt.runtime.progress_board",
            "ffpopt.scan.wavefront_mixins",
            "ffpopt.scan.ScanAnalysis",
        ):
            importlib.import_module(mod)

    def test_lazy_wavefront_attr(self):
        import ffpopt

        m = ffpopt.WaveFront
        self.assertTrue(hasattr(m, "run_dihed_wavefront"))


class TestLigandparamStages(unittest.TestCase):
    def test_abstract_stage(self):
        from ligandparam.stages.abstractstage import AbstractStage

        self.assertTrue(isinstance(AbstractStage, type))

    def test_gaussian_stage_names(self):
        m = importlib.import_module("ligandparam.stages.gaussian")
        for name in (
            "GaussianMinimizeRESP",
            "GaussianRESP",
            "StageGaussianRotation",
            "StageGaussiantoMol2",
        ):
            self.assertTrue(isinstance(getattr(m, name), type), name)

    def test_smiles_to_pdb_module(self):
        try:
            m = importlib.import_module("ligandparam.stages.smiles_to_pdb")
        except ModuleNotFoundError as exc:
            if "rdkit" in str(exc).lower():
                self.skipTest("rdkit not installed")
            raise
        self.assertTrue(hasattr(m, "StageSmilesToPDB"))


if __name__ == "__main__":
    unittest.main()
