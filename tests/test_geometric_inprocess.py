"""Tests for in-process geomeTRIC helpers (no geomeTRIC required)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ffpopt.geometric_inprocess import (
    calc_cache_key,
    get_persistent_calc,
    use_geometric_subprocess,
    _normalize_converge,
)


class TestUseGeometricSubprocess(unittest.TestCase):
    def test_default_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_GEOMETRIC_SUBPROCESS", None)
            self.assertFalse(use_geometric_subprocess())

    def test_env_on(self):
        for val in ("1", "true", "YES", "on"):
            with patch.dict(os.environ, {"FFPOPT_GEOMETRIC_SUBPROCESS": val}):
                self.assertTrue(use_geometric_subprocess(), msg=val)

    def test_env_off(self):
        for val in ("0", "false", ""):
            with patch.dict(os.environ, {"FFPOPT_GEOMETRIC_SUBPROCESS": val}):
                self.assertFalse(use_geometric_subprocess(), msg=repr(val))


class TestCalcCache(unittest.TestCase):
    def _los_struct(self, model="xtb", charge=0, parm=None):
        los = SimpleNamespace(args=SimpleNamespace(model=model))
        struct = MagicMock()
        struct.GetCharge.return_value = charge
        struct.data = {"parm": parm}
        return los, struct

    def test_cache_key_stable(self):
        los, struct = self._los_struct(model="xtb", charge=-1, parm="/a.parm7")
        k1 = calc_cache_key(los, struct)
        k2 = calc_cache_key(los, struct)
        self.assertEqual(k1, k2)
        self.assertEqual(k1[0], "XTB")
        self.assertEqual(k1[1], -1)

    def test_cache_key_changes_with_model(self):
        los, struct = self._los_struct(model="xtb")
        k1 = calc_cache_key(los, struct)
        los.args.model = "sander"
        k2 = calc_cache_key(los, struct)
        self.assertNotEqual(k1, k2)

    def test_get_persistent_calc_reuses_base(self):
        los, struct = self._los_struct()
        base = object()
        los.BuildCalc = MagicMock(return_value=base)

        c1 = get_persistent_calc(los, struct, reslist=None)
        c2 = get_persistent_calc(los, struct, reslist=None)
        self.assertIs(c1, base)
        self.assertIs(c2, base)
        los.BuildCalc.assert_called_once_with(struct)

    def test_get_persistent_calc_wraps_restraints(self):
        los, struct = self._los_struct()
        base = object()
        los.BuildCalc = MagicMock(return_value=base)
        reslist = object()
        wrapper = object()

        with patch(
            "ffpopt.geometric_inprocess._wrap_restrained",
            return_value=wrapper,
        ) as wrap:
            c = get_persistent_calc(los, struct, reslist=reslist)
            self.assertIs(c, wrapper)
            wrap.assert_called_once_with(base, reslist)

        # Second call still only one BuildCalc; fresh wrap.
        with patch(
            "ffpopt.geometric_inprocess._wrap_restrained",
            return_value=object(),
        ):
            get_persistent_calc(los, struct, reslist=reslist)
        los.BuildCalc.assert_called_once_with(struct)


class TestNormalizeConverge(unittest.TestCase):
    def test_string_split(self):
        self.assertEqual(_normalize_converge("set GAU"), ["set", "GAU"])

    def test_list_passthrough(self):
        self.assertEqual(_normalize_converge(["set", "GAU_TIGHT"]), ["set", "GAU_TIGHT"])

    def test_none(self):
        self.assertIsNone(_normalize_converge(None))


if __name__ == "__main__":
    unittest.main()
