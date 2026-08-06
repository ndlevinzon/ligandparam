"""Tests for geomeTRIC watchdog helpers (no geomeTRIC required)."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ffpopt.GeomOpt import (
    _geometric_stall_timeout_sec,
    _path_tree_mtime,
)


class TestGeometricStallHelpers(unittest.TestCase):
    def test_stall_timeout_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFPOPT_GEOMETRIC_STALL_SEC", None)
            self.assertEqual(_geometric_stall_timeout_sec(), 1800.0)

    def test_stall_timeout_env_disable(self):
        with patch.dict(os.environ, {"FFPOPT_GEOMETRIC_STALL_SEC": "0"}):
            self.assertEqual(_geometric_stall_timeout_sec(), 0.0)

    def test_stall_timeout_env_override(self):
        with patch.dict(os.environ, {"FFPOPT_GEOMETRIC_STALL_SEC": "120"}):
            self.assertEqual(_geometric_stall_timeout_sec(), 120.0)

    def test_path_tree_mtime_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            t0 = _path_tree_mtime(str(base))
            time.sleep(0.05)
            (base / "child.txt").write_text("x")
            t1 = _path_tree_mtime(str(base))
            self.assertGreaterEqual(t1, t0)


if __name__ == "__main__":
    unittest.main()
