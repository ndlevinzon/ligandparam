"""Tests for Gaussian resume / skip-complete / -O force-rerun helpers."""

from __future__ import annotations

import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock


def _import_skip_helper():
    """Import gaussian helpers without pulling in ``stages.__init__`` deps."""
    import importlib.util

    import ligandparam  # noqa: F401

    if "ligandparam.stages" not in sys.modules:
        stages_pkg = types.ModuleType("ligandparam.stages")
        stages_pkg.__path__ = [
            str(Path(__file__).resolve().parents[1] / "src" / "ligandparam" / "stages")
        ]
        sys.modules["ligandparam.stages"] = stages_pkg

    if "ligandparam.stages.gaussian" in sys.modules:
        return sys.modules["ligandparam.stages.gaussian"]._should_skip_gaussian_job

    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ligandparam"
        / "stages"
        / "gaussian.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ligandparam.stages.gaussian", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ligandparam.stages.gaussian"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._should_skip_gaussian_job


def _write_log(path: Path, *, complete: bool) -> None:
    body = "Gaussian progress...\n"
    if complete:
        body += " Normal termination of Gaussian 16 at Fri Aug  7 12:00:00 2026.\n"
    else:
        body += " Error termination via Lnk1e.\n"
    path.write_text(body, encoding="utf-8")


class TestShouldSkipGaussianJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.should_skip = staticmethod(_import_skip_helper())

    def test_skips_complete_final_log(self):
        with tempfile.TemporaryDirectory() as td:
            final_log = Path(td) / "lig.gaussian.log"
            _write_log(final_log, complete=True)
            logger = MagicMock()
            self.assertTrue(
                self.should_skip(
                    force_rerun=False,
                    final_log=final_log,
                    cwd_log=Path(td) / "missing.log",
                    logger=logger,
                )
            )

    def test_reruns_incomplete_log(self):
        with tempfile.TemporaryDirectory() as td:
            final_log = Path(td) / "lig.gaussian.log"
            _write_log(final_log, complete=False)
            logger = MagicMock()
            self.assertFalse(
                self.should_skip(
                    force_rerun=False,
                    final_log=final_log,
                    logger=logger,
                )
            )
            logger.info.assert_any_call(
                "Incomplete Gaussian log found (%s); will re-run this job",
                final_log,
            )

    def test_force_rerun_overrides_complete(self):
        with tempfile.TemporaryDirectory() as td:
            final_log = Path(td) / "lig.gaussian.log"
            _write_log(final_log, complete=True)
            logger = MagicMock()
            self.assertFalse(
                self.should_skip(
                    force_rerun=True,
                    final_log=final_log,
                    logger=logger,
                )
            )

    def test_promotes_complete_cwd_log_to_final(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cwd_log = root / "gaussianCalcs" / "lig.log"
            cwd_log.parent.mkdir()
            final_log = root / "lig.gaussian.log"
            _write_log(cwd_log, complete=True)
            logger = logging.getLogger("test.gaussian.skip")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            self.assertTrue(
                self.should_skip(
                    force_rerun=False,
                    final_log=final_log,
                    cwd_log=cwd_log,
                    logger=logger,
                    promote_cwd_to_final=True,
                )
            )
            self.assertTrue(final_log.is_file())
            self.assertIn("Normal termination", final_log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
