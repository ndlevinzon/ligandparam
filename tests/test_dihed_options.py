"""Tests for dihedral-correction option helpers (no ffpopt required)."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from ligandparam.recipes.dihed_options import apply_dihed_options, pop_dihed_options


class TestDihedOptions(unittest.TestCase):
    def test_pop_dihed_options_defaults(self):
        kwargs = {"atom_type": "gaff2"}
        opts = pop_dihed_options(kwargs)
        self.assertFalse(opts["dihed_correct"])
        self.assertEqual(opts["dihed_model"], "qdpi2")
        self.assertEqual(opts["dihed_maxiter"], 2)
        self.assertEqual(kwargs, {"atom_type": "gaff2"})

    def test_apply_dihed_options(self):
        obj = SimpleNamespace()
        kwargs = {"dihed_correct": True, "dihed_model": "xtb", "keep": 1}
        apply_dihed_options(obj, kwargs)
        self.assertTrue(obj.dihed_correct)
        self.assertEqual(obj.dihed_model, "xtb")
        self.assertEqual(kwargs, {"keep": 1})


if __name__ == "__main__":
    unittest.main()
