"""Tests for parallel per-conformer ESP in RespFit / CpeFit."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSplitCoreBudget(unittest.TestCase):
    def test_single_job(self):
        from ffpopt.cpefit.parallel_esp import split_core_budget

        self.assertEqual(split_core_budget(8, 1), (1, 8))

    def test_many_conformers(self):
        from ffpopt.cpefit.parallel_esp import split_core_budget

        self.assertEqual(split_core_budget(8, 4), (4, 2))
        self.assertEqual(split_core_budget(8, 20), (8, 1))


class TestNeedsAbinitioEsp(unittest.TestCase):
    def test_missing_log_needs_work(self):
        from ffpopt.cpefit.AbInitioOptions import AbInitioOptions
        from ffpopt.cpefit.parallel_esp import needs_abinitio_esp

        conf = MagicMock()
        conf.GetBasename.return_value = "confA_H000"
        opts = AbInitioOptions(program="psi4", nproc=4)
        with patch.object(Path, "is_file", return_value=False):
            self.assertTrue(needs_abinitio_esp(conf, opts))

    def test_existing_log_skips(self):
        from ffpopt.cpefit.AbInitioOptions import AbInitioOptions
        from ffpopt.cpefit.parallel_esp import needs_abinitio_esp

        conf = MagicMock()
        conf.GetBasename.return_value = "confA_H000"
        opts = AbInitioOptions(program="g16", nproc=4)
        with patch.object(Path, "is_file", return_value=True):
            self.assertFalse(needs_abinitio_esp(conf, opts))


class TestRunAbinitioEspConformers(unittest.TestCase):
    def test_single_pending_stays_serial(self):
        from ffpopt.cpefit.AbInitioOptions import AbInitioOptions
        from ffpopt.cpefit.parallel_esp import run_abinitio_esp_conformers

        conf = MagicMock()
        conf.GetBasename.return_value = "c0"
        opts = AbInitioOptions(program="psi4", nproc=8)

        with patch(
            "ffpopt.cpefit.parallel_esp.needs_abinitio_esp", return_value=True
        ), patch(
            "ffpopt.cpefit.parallel_esp._run_abinitio_esp_job"
        ) as worker, patch(
            "multiprocessing.get_context"
        ) as get_ctx:
            run_abinitio_esp_conformers([conf], opts)

        get_ctx.assert_not_called()
        worker.assert_called_once()
        self.assertEqual(worker.call_args.args[0]["aiopts"]["nproc"], 8)
        # Parent reload after worker.
        self.assertEqual(conf.RunAbInitioEspIfNeeded.call_count, 1)

    def test_multi_conformer_pools_and_splits(self):
        from ffpopt.cpefit.AbInitioOptions import AbInitioOptions
        from ffpopt.cpefit.parallel_esp import run_abinitio_esp_conformers

        confs = []
        for i in range(4):
            c = MagicMock()
            c.GetBasename.return_value = f"c{i}"
            confs.append(c)
        opts = AbInitioOptions(program="psi4", nproc=8)

        fake_pool = MagicMock()
        fake_pool.__enter__.return_value = fake_pool
        fake_pool.__exit__.return_value = False
        fake_pool.map.side_effect = lambda fn, jobs: [j["conf"].GetBasename() for j in jobs]

        ctx = MagicMock()
        ctx.Pool.return_value = fake_pool

        with patch(
            "ffpopt.cpefit.parallel_esp.needs_abinitio_esp", return_value=True
        ), patch("multiprocessing.get_context", return_value=ctx):
            run_abinitio_esp_conformers(confs, opts)

        ctx.Pool.assert_called_once_with(processes=4)
        jobs = fake_pool.map.call_args.args[1]
        self.assertEqual(len(jobs), 4)
        for job in jobs:
            self.assertEqual(job["aiopts"]["nproc"], 2)
        for conf in confs:
            conf.RunAbInitioEspIfNeeded.assert_called()


class TestRunCosmoHarmonicsConformers(unittest.TestCase):
    def test_pools_when_multiple(self):
        from ffpopt.cpefit.AbInitioOptions import AbInitioOptions
        from ffpopt.cpefit.parallel_esp import run_cosmo_harmonics_conformers

        confs = [MagicMock(), MagicMock()]
        for i, c in enumerate(confs):
            c.GetBasename.return_value = f"c{i}"
            c.desps = None
        opts = AbInitioOptions(program="psi4", nproc=4)

        updated = []
        for c in confs:
            u = MagicMock()
            u.desps = [f"desp-{c.GetBasename()}"]
            updated.append(u)

        fake_pool = MagicMock()
        fake_pool.__enter__.return_value = fake_pool
        fake_pool.__exit__.return_value = False
        fake_pool.map.return_value = updated

        ctx = MagicMock()
        ctx.Pool.return_value = fake_pool

        with patch("multiprocessing.get_context", return_value=ctx):
            run_cosmo_harmonics_conformers(confs, 2, [0.1, 0.2], opts)

        ctx.Pool.assert_called_once_with(processes=2)
        jobs = fake_pool.map.call_args.args[1]
        self.assertEqual(jobs[0]["aiopts"]["nproc"], 2)
        self.assertEqual(confs[0].desps, ["desp-c0"])
        self.assertEqual(confs[1].desps, ["desp-c1"])


if __name__ == "__main__":
    unittest.main()
