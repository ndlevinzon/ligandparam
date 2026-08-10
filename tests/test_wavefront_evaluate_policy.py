"""Unit tests for shared wavefront min/spawn evaluate policy."""

from __future__ import annotations

import unittest

from ffpopt.scan.wavefront_mixins import evaluate_wavefront_minimum


class TestEvaluateWavefrontMinimum(unittest.TestCase):
    def test_soft_first_seeds_and_spawns(self):
        d = evaluate_wavefront_minimum(
            energy=1.0,
            soft=True,
            has_incumbent=False,
            incumbent_energy=None,
            incumbent_soft=False,
            threshold_ev=0.1,
        )
        self.assertTrue(d["update_min"])
        self.assertTrue(d["active"])
        self.assertEqual(d["reason"], "soft_first_seed")

    def test_soft_improve_updates_no_spawn(self):
        d = evaluate_wavefront_minimum(
            energy=0.5,
            soft=True,
            has_incumbent=True,
            incumbent_energy=1.0,
            incumbent_soft=True,
            threshold_ev=0.1,
        )
        self.assertTrue(d["update_min"])
        self.assertFalse(d["active"])
        self.assertEqual(d["reason"], "soft_improve")

    def test_hard_quiet_improve_within_threshold(self):
        d = evaluate_wavefront_minimum(
            energy=0.95,
            soft=False,
            has_incumbent=True,
            incumbent_energy=1.0,
            incumbent_soft=False,
            threshold_ev=0.1,
        )
        self.assertTrue(d["update_min"])
        self.assertFalse(d["active"])
        self.assertEqual(d["reason"], "hard_quiet_improve")

    def test_hard_does_not_replace_lower_soft(self):
        d = evaluate_wavefront_minimum(
            energy=1.2,
            soft=False,
            has_incumbent=True,
            incumbent_energy=1.0,
            incumbent_soft=True,
            threshold_ev=0.1,
        )
        self.assertFalse(d["update_min"])
        self.assertFalse(d["active"])
        self.assertEqual(d["reason"], "hard_worse_than_soft")

    def test_hard_replaces_soft_when_not_higher(self):
        d = evaluate_wavefront_minimum(
            energy=1.0,
            soft=False,
            has_incumbent=True,
            incumbent_energy=1.0,
            incumbent_soft=True,
            threshold_ev=0.1,
        )
        self.assertTrue(d["update_min"])
        self.assertTrue(d["active"])
        self.assertEqual(d["reason"], "hard_replace_soft")


class TestLooseTreatedAsSoft(unittest.TestCase):
    def test_loose_labels(self):
        from ffpopt.GeomOpt import is_soft_opt_recovery

        self.assertTrue(is_soft_opt_recovery("loose"))
        self.assertTrue(is_soft_opt_recovery("dlc-loose"))
        self.assertTrue(is_soft_opt_recovery("hdlc-loose"))
        self.assertTrue(is_soft_opt_recovery("soft-maxiter"))
        self.assertFalse(is_soft_opt_recovery("primary"))


if __name__ == "__main__":
    unittest.main()
