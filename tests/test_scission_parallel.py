"""Tests for scission parallel screening and fragment writes."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scission.models import ClashThresholds, FragmentConfig, TorsionDefinition
from scission.parallel import (
    screen_torsions,
    split_core_budget,
    write_selected_fragments,
)


class TestSplitCoreBudget(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(split_core_budget(8, 1), (1, 8))
        self.assertEqual(split_core_budget(8, 4), (4, 2))
        self.assertEqual(split_core_budget(8, 20), (8, 1))


class TestScreenTorsionsPool(unittest.TestCase):
    def test_serial_when_nproc_one(self):
        ligand = MagicMock()
        torsion = TorsionDefinition(
            label="A-B-C-D",
            atom_indices=(1, 2, 3, 4),
            bond=(2, 3),
        )
        config = FragmentConfig(nproc=1)

        fake_result = {
            "torsion_label": "A-B-C-D",
            "evaluations": [],
            "valid_for_torsion": False,
            "best_failure": None,
        }

        with patch(
            "scission.parallel._screen_one_torsion", return_value=fake_result
        ) as worker, patch("multiprocessing.get_context") as get_ctx:
            pool, accepted, rejected = screen_torsions(ligand, [torsion], config)

        get_ctx.assert_not_called()
        worker.assert_called_once()
        self.assertIn("A-B-C-D", rejected)
        self.assertEqual(pool, {})
        self.assertEqual(accepted, {})

    def test_pools_when_multiple_torsions(self):
        ligand = MagicMock()
        torsions = [
            TorsionDefinition(
                label=f"T{i}",
                atom_indices=(1, 2, 3, 4),
                bond=(2, 3),
            )
            for i in range(4)
        ]
        config = FragmentConfig(nproc=4, clash_thresholds=ClashThresholds())

        fake_pool = MagicMock()
        fake_pool.__enter__.return_value = fake_pool
        fake_pool.__exit__.return_value = False
        fake_pool.map.side_effect = lambda fn, jobs: [
            {
                "torsion_label": job["torsion"].label,
                "evaluations": [],
                "valid_for_torsion": False,
                "best_failure": None,
            }
            for job in jobs
        ]

        ctx = MagicMock()
        ctx.Pool.return_value = fake_pool

        with patch("multiprocessing.get_context", return_value=ctx):
            _pool, _accepted, rejected = screen_torsions(ligand, torsions, config)

        ctx.Pool.assert_called_once_with(processes=4)
        self.assertEqual(len(rejected), 4)


class TestWriteSelectedFragmentsPool(unittest.TestCase):
    def test_thread_pool_for_multiple_fragments(self):
        ligand = MagicMock()
        cand_a = MagicMock()
        cand_a.candidate_id = "a"
        cand_b = MagicMock()
        cand_b.candidate_id = "b"
        config = FragmentConfig(nproc=2)
        warnings: list[str] = []

        frag_a = MagicMock()
        frag_b = MagicMock()

        with patch(
            "scission.parallel._write_one_fragment",
            side_effect=[
                {"index": 1, "fragment": frag_a, "warnings": ["w1"]},
                {"index": 2, "fragment": frag_b, "warnings": ["w2"]},
            ],
        ), patch("scission.parallel.ThreadPoolExecutor") as executor_cls:
            executor = MagicMock()
            executor.__enter__.return_value = executor
            executor.__exit__.return_value = False
            executor.map.side_effect = lambda fn, jobs: [fn(j) for j in jobs]
            executor_cls.return_value = executor

            out = write_selected_fragments(
                ligand,
                [cand_a, cand_b],
                {"a": ["T1"], "b": ["T2"]},
                {},
                Path("/tmp/out"),
                config,
                warnings,
            )

        executor_cls.assert_called_once_with(max_workers=2)
        self.assertEqual(out, [frag_a, frag_b])
        self.assertEqual(warnings, ["w1", "w2"])


class TestFragmentConfigNproc(unittest.TestCase):
    def test_from_dict(self):
        cfg = FragmentConfig.from_dict({"nproc": 8})
        self.assertEqual(cfg.nproc, 8)


if __name__ == "__main__":
    unittest.main()
