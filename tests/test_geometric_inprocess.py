"""Tests for in-process geomeTRIC helpers (no geomeTRIC required)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ffpopt.geometric_inprocess import (
    _normalize_converge,
    _recovery_attempts,
    calc_cache_key,
    get_persistent_calc,
    is_geomopt_not_converged,
    run_geometric_robust,
    use_geometric_robust,
    use_geometric_subprocess,
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


class TestRobustHelpers(unittest.TestCase):
    def test_not_converged_detection(self):
        class GeomOptNotConvergedError(Exception):
            pass

        self.assertTrue(
            is_geomopt_not_converged(
                GeomOptNotConvergedError(
                    "Optimizer.optimizeGeometry() failed to converge."
                )
            )
        )
        self.assertFalse(is_geomopt_not_converged(ValueError("bad input")))

    def test_robust_default_on(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_GEOMOPT_ROBUST", None)
            self.assertTrue(use_geometric_robust())
        with patch.dict(os.environ, {"FFPOPT_GEOMOPT_ROBUST": "0"}):
            self.assertFalse(use_geometric_robust())

    def test_recovery_attempts_include_soft(self):
        with patch.dict(os.environ, {"FFPOPT_GEOMOPT_SOFT_MAXITER": "1"}):
            atts = _recovery_attempts(
                coordsys="tric", maxiter=500, converge="set GAU", enforce=0.1
            )
        labels = [a["label"] for a in atts]
        self.assertEqual(labels[0], "primary")
        self.assertIn("loose", labels)
        self.assertIn("soft-maxiter", labels)
        self.assertTrue(any("dlc" in x or "hdlc" in x for x in labels))

    def test_run_geometric_robust_recovers(self):
        import numpy as np

        class _Atoms:
            def __init__(self, positions):
                self._pos = np.asarray(positions, dtype=float)

            def get_positions(self):
                return self._pos.copy()

            def set_positions(self, positions):
                self._pos = np.asarray(positions, dtype=float)

            def __deepcopy__(self, memo):
                return _Atoms(self._pos)

        atoms = _Atoms([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        calls = {"n": 0}

        def fake_run(work, *_a, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Optimizer.optimizeGeometry() failed to converge.")
            return {
                "coords": work.get_positions(),
                "energy_ha": -1.0,
                "progress": None,
            }

        with patch.dict(
            os.environ,
            {
                "FFPOPT_GEOMOPT_ROBUST": "1",
                "FFPOPT_GEOMOPT_SOFT_MAXITER": "1",
            },
        ):
            with patch(
                "ffpopt.geometric_inprocess.run_geometric_inprocess",
                side_effect=fake_run,
            ):
                with patch(
                    "ffpopt.geometric_inprocess.read_last_optim_xyz",
                    return_value=None,
                ):
                    out = run_geometric_robust(
                        atoms,
                        calc=object(),
                        prefix="/tmp/ffpopt-test-geo",
                        coordsys="tric",
                        maxiter=10,
                        converge="set GAU",
                    )
        self.assertEqual(calls["n"], 2)
        self.assertEqual(out["recovery"], "loose")


class TestWritePlainXyz(unittest.TestCase):
    def test_no_charge_column(self):
        import tempfile
        from pathlib import Path

        from ase import Atoms

        from ffpopt.geometric_inprocess import write_plain_xyz

        atoms = Atoms(
            "CH",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            charges=[-0.0026, 0.0026],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tmp.xyz"
            write_plain_xyz(path, atoms)
            lines = path.read_text().strip().splitlines()
        # Header + comment + 2 atoms; each atom line is symbol + 3 floats.
        self.assertEqual(lines[0].strip(), "2")
        for line in lines[2:]:
            parts = line.split()
            self.assertEqual(len(parts), 4, msg=line)


if __name__ == "__main__":
    unittest.main()
