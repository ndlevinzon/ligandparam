"""Antechamber argv must not treat a negative net charge as another flag."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _paths  # noqa: E402


def setUpModule():
    _paths.ensure_ligandparam()


class TestAntechamberNetChargeArgv(unittest.TestCase):
    def test_negative_nc_is_not_a_bare_minus_token(self):
        from ligandparam.Interfaces import _format_program_flag

        flag, value = _format_program_flag("nc", -1.0)
        self.assertEqual(flag, "-nc")
        self.assertFalse(value.startswith("-"), value)
        self.assertEqual(int(value), -1)

    def test_zero_and_positive_nc_stay_plain_ints(self):
        from ligandparam.Interfaces import _format_program_flag

        self.assertEqual(_format_program_flag("nc", 0.0), ["-nc", "0"])
        self.assertEqual(_format_program_flag("nc", 2), ["-nc", "2"])

    def test_antechamber_dry_run_logs_spaced_negative_nc(self):
        from ligandparam.Interfaces import Antechamber

        logger = MagicMock()
        ante = Antechamber(cwd=".", logger=logger, nproc=1)
        ante.call(i="lig.mol2", fi="mol2", o="out.mol2", fo="mol2", c="bcc", nc=-1.0, dry_run=True)
        logged = " ".join(str(c) for call in logger.info.call_args_list for c in call.args)
        self.assertIn("-nc", logged)
        self.assertNotIn("-nc -1", logged)
        self.assertIn("-1", logged)
