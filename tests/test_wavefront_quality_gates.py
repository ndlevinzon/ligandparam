"""WaveFront soft-opt demotion, failure summary, and related helpers."""

from __future__ import annotations

import ast
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Avoid pulling real ASE at import time.
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

from ffpopt.GeomOpt import is_soft_opt_recovery, opt_recovery_label  # noqa: E402
from ffpopt.WaveFront import Wavefront, WavefrontNode  # noqa: E402


class TestConstraintsPy310FString(unittest.TestCase):
    def test_constraints_module_parses(self):
        src = (ROOT / "src" / "ffpopt" / "Constraints.py").read_text(encoding="utf-8")
        ast.parse(src)


class TestGeometricOptHelp(unittest.TestCase):
    def test_help_mentions_geometric_not_ase_as_primary(self):
        from ffpopt.Options import AddGeomOptOptions
        import argparse

        p = argparse.ArgumentParser()
        AddGeomOptOptions(p)
        help_text = p.format_help()
        self.assertIn("geomeTRIC", help_text)
        # Old inverted wording.
        self.assertNotIn(
            "use the BFGS optimizer in ASE rather than geometric-optimize",
            help_text,
        )


class TestSoftOptHelpers(unittest.TestCase):
    def test_labels(self):
        self.assertTrue(is_soft_opt_recovery("soft-maxiter"))
        self.assertTrue(is_soft_opt_recovery("BFGS-soft"))
        self.assertFalse(is_soft_opt_recovery("loose"))
        self.assertFalse(is_soft_opt_recovery("primary"))
        self.assertFalse(is_soft_opt_recovery("BFGS"))

        struct = SimpleNamespace(data={"geometric_recovery": "soft-maxiter"})
        self.assertEqual(opt_recovery_label(struct), "soft-maxiter")
        self.assertTrue(is_soft_opt_recovery(struct))


class _FakeStruct:
    def __init__(self, n=3):
        self.data = {
            "elements": ["C"] * n,
            "positions": np.zeros((n, 3), dtype=float).tolist(),
            "bonds": [],
            "energy": None,
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


def _make_run():
    run = Wavefront.__new__(Wavefront)
    run.convergence_threshold = 0.01  # kcal/mol
    run.min_energies = {}
    run.min_structures = {}
    run.min_nodes = {}
    run.levels = []
    return run


def _make_node(angle, energy, *, soft=False, recovery=None):
    node = WavefrontNode.__new__(WavefrontNode)
    node.angle = angle
    node.energy = energy
    node.active = True
    node.soft_opt = soft
    node.opt_recovery = recovery
    node.opt_geom = _FakeStruct()
    node.opt_geom.data["energy"] = energy
    if recovery:
        if str(recovery).endswith("-soft") or recovery in ("BFGS", "LBFGS", "FIRE"):
            node.opt_geom.data["ase_opt_recovery"] = recovery
        else:
            node.opt_geom.data["geometric_recovery"] = recovery
    node.error = None
    node.node_id = 0
    return node


class TestSoftOptEvaluate(unittest.TestCase):
    def test_soft_fills_but_does_not_stay_active(self):
        run = _make_run()
        node = _make_node(30.0, -1.0, soft=True, recovery="soft-maxiter")
        run._evaluate_node(node)
        self.assertFalse(node.active)
        self.assertEqual(run.min_energies[30.0], -1.0)

    def test_soft_does_not_replace_hard(self):
        run = _make_run()
        hard = _make_node(30.0, -1.0, soft=False, recovery="primary")
        run._evaluate_node(hard)
        soft = _make_node(30.0, -2.0, soft=True, recovery="soft-maxiter")
        run._evaluate_node(soft)
        self.assertEqual(run.min_energies[30.0], -1.0)
        self.assertFalse(soft.active)
        self.assertIs(run.min_nodes[30.0], hard)

    def test_hard_replaces_soft(self):
        run = _make_run()
        soft = _make_node(30.0, -2.0, soft=True, recovery="BFGS-soft")
        run._evaluate_node(soft)
        hard = _make_node(30.0, -1.0, soft=False, recovery="BFGS")
        run._evaluate_node(hard)
        self.assertEqual(run.min_energies[30.0], -1.0)
        self.assertIs(run.min_nodes[30.0], hard)
        self.assertTrue(hard.active)


class TestPrintSummary(unittest.TestCase):
    def test_reports_failures_not_always_success(self):
        run = _make_run()
        level = SimpleNamespace(nodes=[])
        bad = _make_node(10.0, np.inf)
        bad.error = "clash_precheck"
        bad.active = False
        soft = _make_node(20.0, -1.0, soft=True, recovery="soft-maxiter")
        level.nodes.extend([bad, soft])
        run.levels = [level]
        buf = io.StringIO()
        with redirect_stdout(buf):
            run.print_summary()
        out = buf.getvalue()
        self.assertIn("Failed nodes: 1", out)
        self.assertIn("Soft-accepted nodes (no spawn): 1", out)
        self.assertIn("finished with 1 failed", out)
        self.assertNotIn("completed successfully.", out)


class TestIpcSoftTags(unittest.TestCase):
    def test_apply_result_restores_recovery_tag(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            node = WavefrontNode(
                los=SimpleNamespace(),
                struct=_FakeStruct(),
                con=_FakeCon(),
                angle=40.0,
                level=1,
                node_id=1,
                workdir=td,
            )
            node.apply_result(
                {
                    "energy": -1.5,
                    "forces": np.zeros((3, 3)),
                    "coords": np.zeros((3, 3)),
                    "complete": True,
                    "error": None,
                    "active": False,
                    "soft_opt": True,
                    "opt_recovery": "soft-maxiter",
                }
            )
            self.assertTrue(node.soft_opt)
            self.assertEqual(node.opt_recovery, "soft-maxiter")
            self.assertEqual(
                node.opt_geom.data.get("geometric_recovery"), "soft-maxiter"
            )


if __name__ == "__main__":
    unittest.main()
