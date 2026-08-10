"""Tests for parallel StageGaussianRotation core splitting and Gaussian IPC safety."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _import_gaussian_stage():
    """Import ``stages.gaussian`` without pulling in ``stages.__init__`` deps."""
    import importlib.util

    # Ensure parent packages exist without executing stages/__init__.py.
    import ligandparam  # noqa: F401

    if "ligandparam.stages" not in sys.modules:
        stages_pkg = types.ModuleType("ligandparam.stages")
        stages_pkg.__path__ = [
            str(Path(__file__).resolve().parents[1] / "src" / "ligandparam" / "stages")
        ]
        sys.modules["ligandparam.stages"] = stages_pkg

    if "ligandparam.stages.gaussian" in sys.modules:
        mod = sys.modules["ligandparam.stages.gaussian"]
        if not hasattr(mod, "StageGaussianRotation"):
            del sys.modules["ligandparam.stages.gaussian"]
        else:
            return (
                mod.StageGaussianRotation,
                mod._run_gaussian_rotation_job,
                mod._orientation_id_from_paths,
            )

    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ligandparam"
        / "stages"
        / "gaussian.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ligandparam.stages.gaussian", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ligandparam.stages.gaussian"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return (
        mod.StageGaussianRotation,
        mod._run_gaussian_rotation_job,
        mod._orientation_id_from_paths,
    )


class TestSplitGaussianJobBudget(unittest.TestCase):
    def test_single_job_uses_all_cores(self):
        from ffpopt.runtime.fast_wavefront import split_core_budget as split_gaussian_job_budget

        self.assertEqual(split_gaussian_job_budget(16, 1), (1, 16))

    def test_so3_n28_with_28_cores(self):
        from ffpopt.runtime.fast_wavefront import split_core_budget as split_gaussian_job_budget

        n_workers, nproc = split_gaussian_job_budget(28, 28)
        self.assertEqual((n_workers, nproc), (28, 1))
        self.assertLessEqual(n_workers * nproc, 28)

    def test_more_jobs_than_cores(self):
        from ffpopt.runtime.fast_wavefront import split_core_budget as split_gaussian_job_budget

        n_workers, nproc = split_gaussian_job_budget(16, 28)
        self.assertEqual((n_workers, nproc), (16, 1))
        self.assertLessEqual(n_workers * nproc, 16)

    def test_fat_jobs_when_few_orientations(self):
        from ffpopt.runtime.fast_wavefront import split_core_budget as split_gaussian_job_budget

        n_workers, nproc = split_gaussian_job_budget(16, 4)
        self.assertEqual((n_workers, nproc), (4, 4))
        self.assertLessEqual(n_workers * nproc, 16)


class TestGaussianParallelSafe(unittest.TestCase):
    def test_unique_script_and_scratch_env_copy(self):
        from ligandparam.interfaces import Gaussian

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            gau = Gaussian(
                cwd=cwd,
                gaussian_root="/opt/g16",
                gauss_exedir="/opt/g16/bin",
                gaussian_binary="g16",
                gaussian_scratch=str(cwd / "shared_scratch"),
                logger=MagicMock(),
            )
            scratch_a = cwd / "tmp" / "scratch_jobA"
            scratch_b = cwd / "tmp" / "scratch_jobB"
            env_before = dict(os.environ)

            path_a = gau.write_bash("g16 < a.com > a.log", script_name="_gau_a.sh")
            path_b = gau.write_bash("g16 < b.com > b.log", script_name="_gau_b.sh")
            self.assertTrue(path_a.exists())
            self.assertTrue(path_b.exists())
            self.assertNotEqual(path_a, path_b)

            env_a = gau.set_environment(scratch=scratch_a)
            env_b = gau.set_environment(scratch=scratch_b)
            self.assertEqual(env_a["GAUSS_SCRDIR"], str(scratch_a))
            self.assertEqual(env_b["GAUSS_SCRDIR"], str(scratch_b))
            self.assertTrue(scratch_a.is_dir())
            self.assertTrue(scratch_b.is_dir())
            # Must not mutate the parent process environment.
            self.assertEqual(dict(os.environ), env_before)

    def test_call_defaults_to_per_job_script(self):
        from ligandparam.interfaces import Gaussian

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            gau = Gaussian(
                cwd=cwd,
                gaussian_root="",
                gauss_exedir="",
                gaussian_binary="g16",
                gaussian_scratch="",
                logger=MagicMock(),
            )
            gau.call(inp_pipe="lig_rot_q000.com", out_pipe="lig_rot_q000.log", dry_run=True)
            self.assertTrue((cwd / "_gau_lig_rot_q000.sh").exists())
            text = (cwd / "_gau_lig_rot_q000.sh").read_text()
            self.assertIn("g16 < lig_rot_q000.com > lig_rot_q000.log", text)


class TestRotationExecutePool(unittest.TestCase):
    def test_execute_uses_pool_when_budget_allows(self):
        StageGaussianRotation, _, _ = _import_gaussian_stage()

        with tempfile.TemporaryDirectory() as td:
            gdir = Path(td) / "gaussianCalcs"
            gdir.mkdir()
            stage = StageGaussianRotation.__new__(StageGaussianRotation)
            stage.nproc = 4
            stage.mem = 8
            stage.logger = MagicMock()
            stage.force_gaussian_rerun = True
            stage.orientation_protocol = "legacy_euler"
            stage.alpha = [0.0, 90.0]
            stage.beta = [0.0]
            stage.gamma = [0.0]
            stage.out_gaussian_label = "lig"
            stage.gaussian_root = ""
            stage.gauss_exedir = ""
            stage.gaussian_binary = "g16"
            stage.gaussian_scratch = ""
            stage.gaussian_cwd = gdir
            stage.in_coms = [
                gdir / "lig_rot_0.00_0.00_0.00.com",
                gdir / "lig_rot_90.00_0.00_0.00.com",
            ]
            stage.out_logs = [
                gdir / "lig_rot_0.00_0.00_0.00.log",
                gdir / "lig_rot_90.00_0.00_0.00.log",
            ]

            fake_pool = MagicMock()
            fake_pool.__enter__.return_value = fake_pool
            fake_pool.__exit__.return_value = False
            fake_pool.imap_unordered.return_value = [
                {"in_com": "a", "status": "ok"},
                {"in_com": "b", "status": "ok"},
            ]

            with (
                patch.object(stage, "_setup_execution"),
                patch.object(stage, "setup"),
                patch.object(stage, "_n_orientation_count", return_value=2),
                patch("multiprocessing.get_context") as get_ctx,
            ):
                ctx = MagicMock()
                ctx.Pool.return_value = fake_pool
                get_ctx.return_value = ctx
                stage.execute(dry_run=False, nproc=4, mem=8)

            self.assertEqual(stage._job_nproc, 2)
            self.assertEqual(stage._rotation_n_workers, 2)
            ctx.Pool.assert_called_once_with(processes=2)
            fake_pool.imap_unordered.assert_called_once()
            jobs = fake_pool.imap_unordered.call_args[0][1]
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0]["job_id"], "0.00_0.00_0.00")
            self.assertTrue(jobs[0]["status_path"])
            self.assertTrue((gdir / "ROT_STATUS.txt").is_file())

    def test_orientation_id_from_paths(self):
        _, _, _orientation_id_from_paths = _import_gaussian_stage()

        self.assertEqual(
            _orientation_id_from_paths("lig_rot_q012.com", "lig_rot_q012.log"),
            "q012",
        )
        self.assertEqual(
            _orientation_id_from_paths(
                "lig_rot_30.00_60.00_0.00.com",
                "lig_rot_30.00_60.00_0.00.log",
            ),
            "30.00_60.00_0.00",
        )


if __name__ == "__main__":
    unittest.main()
