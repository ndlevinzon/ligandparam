"""Tests for ConfSearch Butina RMS distance helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestConfSearchFastRmsThreshold(unittest.TestCase):
    def test_default_threshold(self):
        from ffpopt.confsearch.ConfSearch import _confsearch_fast_rms_threshold

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_CONFSEARCH_RMS_FAST_N", None)
            self.assertEqual(_confsearch_fast_rms_threshold(), 50)

    def test_env_override(self):
        from ffpopt.confsearch.ConfSearch import _confsearch_fast_rms_threshold

        with patch.dict(os.environ, {"FFPOPT_CONFSEARCH_RMS_FAST_N": "50"}):
            self.assertEqual(_confsearch_fast_rms_threshold(), 50)


class TestButinaRmsDistances(unittest.TestCase):
    def test_legacy_getbestrms_below_threshold(self):
        from ffpopt.confsearch.ConfSearch import _butina_rms_distances

        mol = MagicMock()
        cids = [0, 1, 2]
        with patch(
            "ffpopt.confsearch.ConfSearch._confsearch_fast_rms_threshold",
            return_value=100,
        ), patch("rdkit.Chem.rdMolAlign.GetBestRMS", side_effect=[0.1, 0.2, 0.3]) as gbr:
            dists = _butina_rms_distances(mol, cids, quiet=True)

        self.assertEqual(dists, [0.1, 0.2, 0.3])
        self.assertEqual(gbr.call_count, 3)

    def test_fast_path_vectorized_length(self):
        from ffpopt.confsearch.ConfSearch import _butina_rms_distances

        # Three conformations of a 2-heavy-atom molecule at known positions.
        class _Pos:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class _Conf:
            def __init__(self, pts):
                self._pts = pts

            def GetAtomPosition(self, idx):
                return _Pos(*self._pts[idx])

        mol = MagicMock()
        atom0 = MagicMock()
        atom0.GetIdx.return_value = 0
        atom0.GetAtomicNum.return_value = 6
        atom1 = MagicMock()
        atom1.GetIdx.return_value = 1
        atom1.GetAtomicNum.return_value = 6
        mol.GetAtoms.return_value = [atom0, atom1]
        mol.GetConformer.side_effect = lambda cid: _Conf(
            {
                0: [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                1: [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],  # identical after align
                2: [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],  # farther
            }[cid]
        )

        with patch(
            "ffpopt.confsearch.ConfSearch._confsearch_fast_rms_threshold",
            return_value=2,
        ), patch("rdkit.Chem.rdMolAlign.AlignMol"):
            dists = _butina_rms_distances(mol, [0, 1, 2], quiet=True)

        # Condensed length n*(n-1)/2 = 3
        self.assertEqual(len(dists), 3)
        self.assertAlmostEqual(dists[0], 0.0, places=6)  # 1 vs 0
        self.assertGreater(dists[1], 0.0)  # 2 vs 0
        self.assertGreater(dists[2], 0.0)  # 2 vs 1


if __name__ == "__main__":
    unittest.main()
