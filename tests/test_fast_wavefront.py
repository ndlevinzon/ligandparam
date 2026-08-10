"""Tests for fast-wavefront presets and allocation helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ffpopt.fast_wavefront import (
    FAST_WAVEFRONT_PRESETS,
    apply_fast_wavefront_presets,
    fast_recovery_ladder,
    fast_wavefront_enabled,
    prefer_wavefront_depth,
    split_nproc_for_items,
    wf_checkpoint_every,
    write_success_node_pickle,
)
from ffpopt.geometric_inprocess import _recovery_attempts


class TestFastWavefrontPresets(unittest.TestCase):
    def test_enabled_from_env(self):
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}):
            self.assertTrue(fast_wavefront_enabled())
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "0"}):
            self.assertFalse(fast_wavefront_enabled())
        self.assertTrue(fast_wavefront_enabled(True))
        self.assertFalse(fast_wavefront_enabled(False))

    def test_apply_overrides_library_defaults_only(self):
        kw = {
            "delta": 10,
            "geometric_maxiter": 500,
            "geometric_converge": "set GAU",
            "wf_convergence_threshold": 0.01,
            "ase_opt_tol": 0.01,
        }
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}):
            applied = apply_fast_wavefront_presets(kw)
        self.assertEqual(applied, FAST_WAVEFRONT_PRESETS)
        self.assertEqual(kw["delta"], 15)
        self.assertEqual(kw["geometric_maxiter"], 200)

        kw2 = {"delta": 5, "geometric_maxiter": 500}
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}):
            applied2 = apply_fast_wavefront_presets(kw2)
        self.assertNotIn("delta", applied2)
        self.assertEqual(kw2["delta"], 5)
        self.assertEqual(kw2["geometric_maxiter"], 200)


class TestNprocSplit(unittest.TestCase):
    def test_breadth_default(self):
        self.assertEqual(split_nproc_for_items(8, 4), (4, 2))
        self.assertEqual(split_nproc_for_items(4, 8), (4, 1))

    def test_prefer_depth(self):
        # Keep at least 2 inner cores: 8 cores / 4 items -> 4 outer x 2
        # but min_inner=2 limits outer to 8//2=4, same.
        self.assertEqual(
            split_nproc_for_items(8, 4, prefer_depth=True, min_inner=2),
            (4, 2),
        )
        # 8 cores, 8 items, min_inner=2 -> 4 outer x 2 (not 8x1)
        self.assertEqual(
            split_nproc_for_items(8, 8, prefer_depth=True, min_inner=2),
            (4, 2),
        )

    def test_prefer_depth_xtb_fast(self):
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}):
            self.assertTrue(prefer_wavefront_depth(model="xtb"))
            self.assertFalse(prefer_wavefront_depth(model="qdpi2"))


class TestCheckpointAndRecovery(unittest.TestCase):
    def test_checkpoint_every_fast(self):
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}, clear=False):
            os.environ.pop("FFPOPT_WF_CHECKPOINT_EVERY", None)
            self.assertEqual(wf_checkpoint_every(2), 8)
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "0"}, clear=False):
            os.environ.pop("FFPOPT_WF_CHECKPOINT_EVERY", None)
            self.assertEqual(wf_checkpoint_every(2), 2)

    def test_success_node_pickle_off_in_fast(self):
        with patch.dict(os.environ, {"FFPOPT_FAST_WAVEFRONT": "1"}, clear=False):
            os.environ.pop("FFPOPT_WF_NODE_PICKLE", None)
            self.assertFalse(write_success_node_pickle())
        with patch.dict(os.environ, {"FFPOPT_WF_NODE_PICKLE": "1"}):
            self.assertTrue(write_success_node_pickle())

    def test_fast_recovery_skips_alt_coordsys(self):
        with patch.dict(
            os.environ,
            {
                "FFPOPT_FAST_WAVEFRONT": "1",
                "FFPOPT_GEOMOPT_SOFT_MAXITER": "1",
            },
        ):
            self.assertTrue(fast_recovery_ladder())
            labels = [
                a["label"]
                for a in _recovery_attempts(
                    coordsys="tric", maxiter=200, converge="set GAU_LOOSE", enforce=0.1
                )
            ]
        self.assertEqual(labels, ["primary", "loose", "soft-maxiter"])
        self.assertFalse(any("dlc" in x or "hdlc" in x for x in labels))

    def test_full_recovery_includes_alts(self):
        with patch.dict(
            os.environ,
            {
                "FFPOPT_FAST_WAVEFRONT": "0",
                "FFPOPT_GEOMOPT_FAST_RECOVERY": "0",
                "FFPOPT_GEOMOPT_SOFT_MAXITER": "1",
            },
        ):
            labels = [
                a["label"]
                for a in _recovery_attempts(
                    coordsys="tric", maxiter=500, converge="set GAU", enforce=0.1
                )
            ]
        self.assertIn("loose", labels)
        self.assertTrue(any("dlc" in x or "hdlc" in x for x in labels))
        self.assertIn("soft-maxiter", labels)


if __name__ == "__main__":
    unittest.main()
