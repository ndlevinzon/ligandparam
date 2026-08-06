"""Tests for bare_potential_energy (no SCF / ASE required)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ffpopt.GeomOpt import bare_potential_energy


class TestBarePotentialEnergy(unittest.TestCase):
    def test_no_restraints(self):
        struct = SimpleNamespace(
            data={"energy": 1.25, "positions": [[0.0, 0.0, 0.0]]},
            restraints=None,
        )
        self.assertEqual(bare_potential_energy(struct), 1.25)

    def test_empty_restraint_list(self):
        rests = MagicMock()
        rests.__len__.return_value = 0
        struct = SimpleNamespace(
            data={"energy": -3.0, "positions": [[0.0, 0.0, 0.0]]},
            restraints=rests,
        )
        self.assertEqual(bare_potential_energy(struct), -3.0)

    def test_subtracts_restraint_penalties(self):
        rst1 = MagicMock()
        rst1.GetValueAndGradients.return_value = (0.5, np.zeros((2, 3)))
        rst2 = MagicMock()
        rst2.GetValueAndGradients.return_value = (0.25, np.zeros((2, 3)))
        rests = [rst1, rst2]
        struct = SimpleNamespace(
            data={
                "energy": 10.0,
                "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            },
            restraints=rests,
        )
        self.assertAlmostEqual(bare_potential_energy(struct), 9.25)
        self.assertEqual(rst1.GetValueAndGradients.call_count, 1)
        self.assertEqual(rst2.GetValueAndGradients.call_count, 1)

    def test_missing_energy_raises(self):
        struct = SimpleNamespace(data={}, restraints=None)
        with self.assertRaises(ValueError):
            bare_potential_energy(struct)


if __name__ == "__main__":
    unittest.main()
