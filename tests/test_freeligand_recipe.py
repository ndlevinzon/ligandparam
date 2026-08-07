"""Lightweight FreeLigand / Leap recipe construction tests (no Gaussian)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class TestFreeLigandSetup(unittest.TestCase):
    def test_setup_builds_expected_stage_sequence(self):
        from ligandparam.recipes.freeligand import FreeLigand
        from ligandparam.stages import (
            StageInitialize,
            StageNormalizeCharge,
            StageLeap,
            StageParmChk,
            StageMultiRespFit,
        )

        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            recipe = FreeLigand(
                cwd / "ligand.pdb",
                cwd,
                net_charge=0,
                nproc=2,
                mem=4,
                logger="stream",
            )
            recipe.setup()
            types = [type(s) for s in recipe.stages]
            self.assertEqual(types[0], StageInitialize)
            self.assertIn(StageNormalizeCharge, types)
            self.assertIn(StageMultiRespFit, types)
            self.assertEqual(types[-2], StageParmChk)
            self.assertEqual(types[-1], StageLeap)
            self.assertEqual(recipe.net_charge, 0)

    def test_missing_net_charge_raises(self):
        from ligandparam.recipes.freeligand import FreeLigand

        with self.assertRaises(KeyError):
            FreeLigand("ligand.pdb", "out_dir")

    def test_bad_orientation_protocol_raises(self):
        from ligandparam.recipes.freeligand import FreeLigand

        with self.assertRaises(ValueError):
            FreeLigand(
                "ligand.pdb",
                "out_dir",
                net_charge=0,
                orientation_protocol="not_a_protocol",
            )


class TestChargeNormalize(unittest.TestCase):
    def _stage(self, net_charge=0, precision=0.001, decimals=3):
        from ligandparam.stages.charge import StageNormalizeCharge

        st = StageNormalizeCharge.__new__(StageNormalizeCharge)
        st.net_charge = net_charge
        st.precision = precision
        st.decimals = decimals
        st.logger = MagicMock()
        return st

    def test_nonzero_net_charge(self):
        st = self._stage(net_charge=1, precision=0.001, decimals=3)
        q = [0.4, 0.3, 0.2]
        rounded, total, diff = st.check_charge(q)
        out = st.normalize(rounded, diff)
        _, new_total, _ = st.check_charge(out)
        self.assertTrue(abs(new_total - 1.0) < 0.002)

    def test_zero_count_safe(self):
        st = self._stage(net_charge=0, precision=0.001, decimals=3)
        q = [0.0, 0.0]
        out = st.normalize(q, 0.0)
        self.assertEqual(list(out), [0.0, 0.0])

    def test_large_delta_warns_and_finishes(self):
        st = self._stage(net_charge=1, precision=0.001, decimals=3)
        q = [0.0, 0.0]
        out = st.normalize(q, 0.05)
        self.assertAlmostEqual(float(sum(out)), 0.05, places=6)
        st.logger.warning.assert_called()


if __name__ == "__main__":
    unittest.main()
