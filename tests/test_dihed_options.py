"""Tests for dihedral-correction option helpers (no ffpopt required)."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ligandparam.recipes.dihed_options import (
    append_dihed_twist_stage,
    apply_dihed_options,
    coerce_fragment_config,
    pop_dihed_options,
)


class TestDihedOptions(unittest.TestCase):
    def test_pop_dihed_options_defaults(self):
        kwargs = {"atom_type": "gaff2"}
        opts = pop_dihed_options(kwargs)
        self.assertFalse(opts["dihed_correct"])
        self.assertEqual(opts["dihed_model"], "qdpi2")
        self.assertEqual(opts["dihed_maxiter"], 2)
        self.assertEqual(opts["dihed_delta"], 10)
        self.assertIsNone(opts["dihed_fragment_config"])
        self.assertEqual(kwargs, {"atom_type": "gaff2"})

    def test_pop_dihed_delta_and_fragment_config(self):
        kwargs = {
            "dihed_delta": 5,
            "dihed_fragment_config": {"angle_step": 15},
            "other": True,
        }
        opts = pop_dihed_options(kwargs)
        self.assertEqual(opts["dihed_delta"], 5)
        self.assertEqual(opts["dihed_fragment_config"], {"angle_step": 15})
        self.assertEqual(kwargs, {"other": True})

    def test_apply_dihed_options(self):
        obj = SimpleNamespace()
        kwargs = {
            "dihed_correct": True,
            "dihed_model": "xtb",
            "dihed_delta": 5,
            "keep": 1,
        }
        apply_dihed_options(obj, kwargs)
        self.assertTrue(obj.dihed_correct)
        self.assertEqual(obj.dihed_model, "xtb")
        self.assertEqual(obj.dihed_delta, 5)
        self.assertEqual(kwargs, {"keep": 1})

    def test_coerce_fragment_config_none(self):
        self.assertIsNone(coerce_fragment_config(None))

    def test_coerce_fragment_config_dict(self):
        try:
            from scission.models import FragmentConfig
        except ImportError:
            self.skipTest("scission not importable")
        cfg = coerce_fragment_config({"angle_step": 15, "cap_strategy": "hydrogen"})
        self.assertIsInstance(cfg, FragmentConfig)
        self.assertEqual(cfg.angle_step, 15)
        self.assertEqual(cfg.cap_strategy, "hydrogen")

    def test_coerce_fragment_config_rejects_bad_type(self):
        with self.assertRaises(TypeError):
            coerce_fragment_config("nope")

    def test_append_passes_delta_and_fragment_config(self):
        stages = []
        recipe = SimpleNamespace(
            dihed_correct=True,
            dihed_out_frcmod=None,
            dihed_out_dir=None,
            cwd=Path("/tmp/out"),
            label="LIG",
            dihed_model="xtb",
            dihed_maxiter=2,
            dihed_nprim=3,
            dihed_delta=5,
            nproc=4,
            dihed_geometric_opt=True,
            dihed_skip_existing=True,
            dihed_rotatable_bond_smarts=None,
            dihed_fragment_config={"angle_step": 20},
            logger=None,
        )
        fake_stage = MagicMock()
        with patch(
            "ligandparam.stages.ffpopt_dihed.StageDihedTwistCorrection",
            return_value=fake_stage,
        ) as ctor, patch(
            "ligandparam.recipes.dihed_options.coerce_fragment_config",
            return_value="COERCED",
        ) as coerce:
            append_dihed_twist_stage(
                stages,
                recipe=recipe,
                mol2=Path("/tmp/out/LIG.mol2"),
                lib=Path("/tmp/out/LIG.lib"),
                frcmod=Path("/tmp/out/LIG.frcmod"),
            )
        self.assertEqual(stages, [fake_stage])
        coerce.assert_called_once_with({"angle_step": 20})
        kwargs = ctor.call_args.kwargs
        self.assertEqual(kwargs["delta"], 5)
        self.assertEqual(kwargs["fragment_config"], "COERCED")
        self.assertEqual(kwargs["model"], "xtb")


if __name__ == "__main__":
    unittest.main()
