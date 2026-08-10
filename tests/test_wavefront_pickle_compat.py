"""Pickle compat for pre-scan/ wavefront module paths."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestWaveFrontPickleCompat(unittest.TestCase):
    def test_old_module_path_resolves(self):
        import ffpopt.WaveFront as legacy
        from ffpopt.scan.WaveFront import Wavefront as canonical

        self.assertIs(legacy.Wavefront, canonical)

    def test_roundtrip_under_legacy_module_name(self):
        import ffpopt.WaveFront as legacy
        from ffpopt.scan.wavefront_mixins import pickle_load_compat

        # Simulate an object whose class was recorded as ffpopt.WaveFront.*
        obj = SimpleNamespace(marker="ckpt")
        # Pickle a real WavefrontNode-less stand-in by forcing __module__.
        class _Node:
            pass

        _Node.__module__ = "ffpopt.WaveFront"
        _Node.__qualname__ = "WavefrontNode"
        # Register on the legacy module so find_class succeeds.
        legacy.WavefrontNode = _Node  # type: ignore[attr-defined]
        node = _Node()
        node.payload = 42

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "node.pkl"
            with open(path, "wb") as f:
                pickle.dump(node, f)
            loaded = pickle_load_compat(path)
        self.assertEqual(loaded.payload, 42)
        self.assertEqual(loaded.__class__.__module__, "ffpopt.WaveFront")


if __name__ == "__main__":
    unittest.main()
