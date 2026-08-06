"""Tests for robust tblite SCF helpers (no real tblite / ase required)."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _load_tblite_scf():
    """Load tblite_scf.py without importing ffpopt.ase (needs ase)."""
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "ffpopt" / "ase" / "tblite_scf.py"
    # Register parent packages as namespaces so patch targets resolve.
    for name, pkg_path in (
        ("ffpopt", root / "src" / "ffpopt"),
        ("ffpopt.ase", root / "src" / "ffpopt" / "ase"),
    ):
        if name not in sys.modules:
            mod = type(sys)(name)
            mod.__path__ = [str(pkg_path)]
            sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("ffpopt.ase.tblite_scf", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ffpopt.ase.tblite_scf"] = mod
    spec.loader.exec_module(mod)
    return mod


tblite_scf = _load_tblite_scf()
_is_scf_failure = tblite_scf._is_scf_failure
_scf_retry_configs = tblite_scf._scf_retry_configs
run_tblite_with_scf_retries = tblite_scf.run_tblite_with_scf_retries
tblite_kwargs_from_env = tblite_scf.tblite_kwargs_from_env


class TestTbliteKwargs(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in (
                "FFPOPT_XTB_MAX_ITER",
                "FFPOPT_XTB_ETEMP",
                "FFPOPT_XTB_MIXER_DAMPING",
                "FFPOPT_XTB_GUESS",
            ):
                os.environ.pop(key, None)
            kwargs = tblite_kwargs_from_env()
        self.assertEqual(kwargs["max_iterations"], 500)
        self.assertEqual(kwargs["electronic_temperature"], 500.0)
        self.assertEqual(kwargs["mixer_damping"], 0.25)
        self.assertEqual(kwargs["initial_guess"], "eeq")
        self.assertEqual(kwargs["method"], "GFN2-xTB")

    def test_env_overrides(self):
        with patch.dict(
            os.environ,
            {
                "FFPOPT_XTB_MAX_ITER": "800",
                "FFPOPT_XTB_ETEMP": "750.5",
                "FFPOPT_XTB_MIXER_DAMPING": "0.2",
                "FFPOPT_XTB_GUESS": "sad",
            },
        ):
            kwargs = tblite_kwargs_from_env()
        self.assertEqual(kwargs["max_iterations"], 800)
        self.assertEqual(kwargs["electronic_temperature"], 750.5)
        self.assertEqual(kwargs["mixer_damping"], 0.2)
        self.assertEqual(kwargs["initial_guess"], "sad")

    def test_explicit_overrides_win(self):
        with patch.dict(os.environ, {"FFPOPT_XTB_MAX_ITER": "800"}):
            kwargs = tblite_kwargs_from_env({"max_iterations": 900})
        self.assertEqual(kwargs["max_iterations"], 900)


class TestScfFailureDetect(unittest.TestCase):
    def test_direct_message(self):
        self.assertTrue(
            _is_scf_failure(RuntimeError("SCF not converged in 250 cycles"))
        )

    def test_wrapped_cause(self):
        root = RuntimeError("SCF not converged in 250 cycles")
        outer = RuntimeError("CalculationFailed")
        outer.__cause__ = root
        self.assertTrue(_is_scf_failure(outer))

    def test_unrelated(self):
        self.assertFalse(_is_scf_failure(RuntimeError("Not bracketed")))


class TestRetryConfigs(unittest.TestCase):
    def test_ladder_length_and_escalation(self):
        configs = _scf_retry_configs({"charge": 0, "mixer_damping": 0.25})
        self.assertEqual(len(configs), 3)
        self.assertEqual(configs[0]["electronic_temperature"], 1000.0)
        self.assertEqual(configs[0]["max_iterations"], 750)
        self.assertIn("annealing", configs[1])
        self.assertEqual(configs[2]["max_iterations"], 1000)
        self.assertEqual(configs[2]["mixer_damping"], 0.15)


class TestRunWithRetries(unittest.TestCase):
    def test_succeeds_first_try(self):
        atoms = MagicMock()
        atoms.get_potential_energy.return_value = -1.0
        atoms.get_forces.return_value = np.zeros((2, 3))
        calc = MagicMock()
        energy, forces, out_calc = run_tblite_with_scf_retries(atoms, calc)
        self.assertEqual(energy, -1.0)
        self.assertIs(out_calc, calc)
        self.assertEqual(atoms.calc, calc)

    def test_retries_then_succeeds(self):
        atoms = MagicMock()
        fail = RuntimeError("SCF not converged in 250 cycles")
        atoms.get_potential_energy.side_effect = [fail, fail, -2.5]
        atoms.get_forces.return_value = np.ones((2, 3))
        calc = MagicMock()
        calc.parameters = {"charge": 0, "method": "GFN2-xTB"}
        rebuilt = MagicMock()

        with patch.object(
            tblite_scf, "make_tblite_calculator", return_value=rebuilt
        ) as make:
            energy, forces, out_calc = run_tblite_with_scf_retries(atoms, calc)

        self.assertEqual(energy, -2.5)
        self.assertIs(out_calc, rebuilt)
        self.assertGreaterEqual(make.call_count, 1)

    def test_non_scf_error_not_retried(self):
        atoms = MagicMock()
        atoms.get_potential_energy.side_effect = RuntimeError("Not bracketed")
        with self.assertRaises(RuntimeError) as ctx:
            run_tblite_with_scf_retries(atoms, MagicMock())
        self.assertIn("Not bracketed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
