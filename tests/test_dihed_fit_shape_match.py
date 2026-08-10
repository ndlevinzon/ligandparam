"""χ² shape-match and joint LS numerics for GenDihedFit."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

for _name in (
    "ase",
    "ase.calculators",
    "ase.calculators.calculator",
    "ase.calculators.amber",
    "ase.io",
    "ase.optimize",
    "ase.units",
):
    sys.modules.setdefault(_name, MagicMock())

from ffpopt.Dihedrals import (  # noqa: E402
    MultiDihedFcn,
    PrimDihedFcn,
    joint_design_matrix_from_caches,
    joint_linear_solve_from_caches,
    shape_match_delta,
)


class TestShapeMatchDelta(unittest.TestCase):
    def test_invariant_to_ll_constant(self):
        hl = np.array([1.0, 2.0, 4.0, 3.0])
        ll = np.array([0.5, 1.0, 2.5, 2.0])
        d0 = shape_match_delta(hl, ll)
        d1 = shape_match_delta(hl, ll + 7.5)
        np.testing.assert_allclose(d0, d1)
        self.assertAlmostEqual(float(np.mean(d0)), 0.0, places=12)


class TestJointLinearMatchesNL(unittest.TestCase):
    def test_joint_ls_recovers_fc_and_matches_residual_at_x0(self):
        # Synthetic one-param Fourier: E_term = 1+cos(per*ang), per=1, phase=0.
        angs = np.linspace(0.0, 330.0, 12)
        true_fc = 1.7
        prim = PrimDihedFcn(0.0, 0.0, 1)
        terms = np.array([float(prim.CptEterm(a)) for a in angs])
        base = np.zeros_like(angs)
        hl = base + true_fc * terms + 3.0  # constant offset must not matter

        dfcns = MultiDihedFcn([0, 1, 2, 3], [PrimDihedFcn(0.0, 0.0, 1)])
        ptype = SimpleNamespace(name="t1", dfcns=dfcns, nprim=1)
        pinst = SimpleNamespace(ptype=ptype, dihedidxs=[(0, 1, 2, 3)])
        structs_hl = [SimpleNamespace(data={"energy": float(e)}) for e in hl]
        # Energy in eV in Struct; joint path multiplies by kcal_per_ev.
        # Pass kcal directly by mocking conversion factor = 1 via hl already in kcal
        # and kcal_per_ev=1 in the design builder call below.
        # joint_linear_solve_from_caches uses real constants — store energies in eV
        # so that eV * (kcal/eV) = hl_kcal. Easiest: set energy = hl / kcal_per_ev.
        from ffpopt.constants import AU_PER_ELECTRON_VOLT, AU_PER_KCAL_PER_MOL

        kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
        structs_hl = [
            SimpleNamespace(data={"energy": float(e) / kcal_per_ev}) for e in hl
        ]

        class _Los:
            def __init__(self, structs):
                self.structs = structs

            def __iter__(self):
                return iter(self.structs)

            def __len__(self):
                return len(self.structs)

        loshl = _Los(structs_hl)
        prof = SimpleNamespace(name="p", loshl=loshl, losll=loshl)
        system = SimpleNamespace(pinstances=[pinst], profiles=[prof])
        finp = SimpleNamespace(
            ptypedict={"t1": ptype},
            systems=[system],
        )
        cache = {
            "profiles": [
                {
                    "base_kcal": base.copy(),
                    "angles": [[[float(a)]] for a in angs],
                }
            ]
        }

        A, y, nparam = joint_design_matrix_from_caches(finp, [cache], kcal_per_ev)
        self.assertEqual(nparam, 1)
        x, info = joint_linear_solve_from_caches(finp, [cache])
        self.assertEqual(info["rank"], 1)
        self.assertAlmostEqual(float(x[0]), true_fc, places=6)

        # Residual of mean-centered model at x0 matches shape_match of NL energies.
        dfcns.SetFCs([float(x[0])])
        ll = base + true_fc * terms  # exact
        d_nl = shape_match_delta(hl, ll)
        resid = y - A @ x
        np.testing.assert_allclose(resid, d_nl, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
