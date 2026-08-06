"""Tests for GenDihedFit fixed-geometry LL energy cache (no sander/ASE)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ffpopt.Dihedrals import (
    PrimDihedFcn,
    MultiDihedFcn,
    _analytical_fitted_torsion_kcal,
    ll_energies_kcal_from_cache,
    use_dihed_fit_reopt,
)


class TestUseDihedFitReopt(unittest.TestCase):
    def test_default_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_DIHED_FIT_REOPT", None)
            self.assertFalse(use_dihed_fit_reopt())

    def test_env_on(self):
        with patch.dict(os.environ, {"FFPOPT_DIHED_FIT_REOPT": "1"}):
            self.assertTrue(use_dihed_fit_reopt())


class TestAnalyticalTorsionCache(unittest.TestCase):
    def _system(self, fc=1.0, phase=0.0, per=1):
        prim = PrimDihedFcn(fc, phase, per)
        dfcns = MultiDihedFcn([0, 1, 2, 3], [prim])
        ptype = SimpleNamespace(name="t1", dfcns=dfcns, nprim=1)
        pinst = SimpleNamespace(
            ptype=ptype,
            dihedidxs=[(0, 1, 2, 3), (4, 5, 6, 7)],
        )
        return SimpleNamespace(pinstances=[pinst])

    def test_analytical_sum_over_occurrences(self):
        system = self._system(fc=2.0, phase=0.0, per=1)
        # CptEne(0°) = 2*(1+cos(0)) = 4; two diheds → 8
        ang_tables = [[0.0, 0.0]]
        e = _analytical_fitted_torsion_kcal(system, ang_tables)
        self.assertAlmostEqual(e, 8.0)

    def test_ll_energies_add_base_and_torsion(self):
        system = self._system(fc=1.0, phase=0.0, per=1)
        # one geom: base 10, two diheds at 0° → +2 each = 4 → total 14
        cache = {
            "profiles": [
                {
                    "base_kcal": np.array([10.0, 20.0]),
                    "angles": [
                        [[0.0, 0.0]],
                        [[180.0, 180.0]],  # 1+cos(180)=0 → torsion 0
                    ],
                }
            ]
        }
        ll = ll_energies_kcal_from_cache(system, cache)
        self.assertEqual(len(ll), 1)
        self.assertAlmostEqual(ll[0][0], 14.0)
        self.assertAlmostEqual(ll[0][1], 20.0)

    def test_fc_change_updates_energy_without_new_base(self):
        system = self._system(fc=1.0)
        cache = {
            "profiles": [
                {
                    "base_kcal": np.array([5.0]),
                    "angles": [[[0.0, 0.0]]],
                }
            ]
        }
        e1 = ll_energies_kcal_from_cache(system, cache)[0][0]
        system.pinstances[0].ptype.dfcns.SetFCs([3.0])
        e2 = ll_energies_kcal_from_cache(system, cache)[0][0]
        # torsion at 0°: fc*(1+1)*2 diheds = 4*fc
        self.assertAlmostEqual(e1, 5.0 + 4.0)
        self.assertAlmostEqual(e2, 5.0 + 12.0)


if __name__ == "__main__":
    unittest.main()
