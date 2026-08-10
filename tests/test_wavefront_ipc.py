"""Tests for slim wavefront IPC payloads (no geomeTRIC / pool required)."""

import copy
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

# WaveFront imports Struct, which pulls ASE calculators at import time.
for _name in (
    "ase",
    "ase.calculators",
    "ase.calculators.calculator",
    "ase.calculators.amber",
    "ase.io",
    "ase.optimize",
    "ase.units",
):
    sys.modules.setdefault(_name, MagicMock())

from ffpopt.scan.WaveFront import (  # noqa: E402
    WavefrontNode,
    _init_worker,
    _run_node_job,
    _WORKER,
)


class _FakeStruct:
    def __init__(self, n=3):
        self.data = {
            "elements": ["C"] * n,
            "positions": np.zeros((n, 3), dtype=float).tolist(),
            "bonds": [],
        }
        self.constraints = None
        self.restraints = None

    def Update(self, ene, crds, frcs):
        self.data["energy"] = ene
        self.data["positions"] = np.asarray(crds, dtype=float).tolist()
        self.data["forces"] = frcs


class _FakeCon:
    def __init__(self):
        self.value = None
        self.idxs = [0, 1, 2, 3]


class TestWavefrontIPC(unittest.TestCase):
    def setUp(self):
        _WORKER.clear()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.workdir = self.tmpdir.name
        self.los = SimpleNamespace(calc=None, args=SimpleNamespace())
        self.struct = _FakeStruct()
        self.con = _FakeCon()
        _init_worker(self.los, self.con, self.struct)

    def tearDown(self):
        self.tmpdir.cleanup()
        _WORKER.clear()

    def _node(self, angle=30.0):
        return WavefrontNode(
            los=self.los,
            struct=copy.deepcopy(self.struct),
            con=self.con,
            angle=angle,
            level=1,
            node_id=0,
            workdir=self.workdir,
        )

    def test_to_job_omits_los(self):
        node = self._node()
        job = node.to_job()
        self.assertNotIn("los", job)
        self.assertNotIn("struct", job)
        self.assertIn("coords", job)
        self.assertEqual(job["angle"], 30.0)

    def test_apply_result_builds_opt_geom(self):
        node = self._node()
        coords = np.ones((3, 3), dtype=float)
        node.apply_result(
            {
                "energy": -1.5,
                "forces": None,
                "coords": coords,
                "complete": True,
                "error": None,
                "active": True,
            }
        )
        self.assertTrue(node.complete)
        self.assertEqual(node.energy, -1.5)
        self.assertIsNotNone(node.opt_geom)
        np.testing.assert_allclose(
            np.asarray(node.opt_geom.data["positions"]), coords
        )

    def test_checkpoint_omits_los(self):
        node = self._node()
        node.energy = 0.1
        node.complete = True
        node._write_checkpoint()
        self.assertIsNotNone(node.los)
        loaded = Path(node.node_pkl)
        self.assertTrue(loaded.is_file())
        with open(loaded, "rb") as fh:
            dumped = pickle.load(fh)
        self.assertIsNone(dumped.los)

    def test_run_node_job_uses_worker_los(self):
        node = self._node(angle=10.0)
        job = node.to_job()

        def fake_calculate(self):
            self.opt_geom = copy.deepcopy(self.struct)
            self.opt_geom.Update(-2.0, np.asarray(self.struct.data["positions"]), None)
            self.energy = -2.0
            self.complete = True

        # Patch calculate to avoid GeomOpt
        orig = WavefrontNode.calculate
        WavefrontNode.calculate = fake_calculate
        try:
            result = _run_node_job(job)
        finally:
            WavefrontNode.calculate = orig

        self.assertTrue(result["complete"])
        self.assertEqual(result["energy"], -2.0)
        self.assertNotIn("los", result)


if __name__ == "__main__":
    unittest.main()
