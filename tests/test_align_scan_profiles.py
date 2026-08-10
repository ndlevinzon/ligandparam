"""Tests for HL/LL scan profile angle alignment in GenDihedFit."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

# Struct import pulls ASE calculators; stub for unit tests.
for _name in (
    "ase",
    "ase.calculators",
    "ase.calculators.calculator",
    "ase.calculators.amber",
    "ase.io",
    "ase.optimize",
    "ase.units",
    "ase.build",
):
    sys.modules.setdefault(_name, MagicMock())

from ffpopt.Dihedrals import (  # noqa: E402
    _normalize_scan_angle,
    align_scan_profiles,
    struct_scan_angle,
)
from ffpopt.Struct import ListOfStruct  # noqa: E402


def _frame(name: str, energy: float = 0.0):
    return SimpleNamespace(
        data={
            "name": name,
            "energy": energy,
            "positions": [[0.0, 0.0, 0.0]],
            "constraints": [],
        },
        constraints=None,
    )


class TestStructScanAngle(unittest.TestCase):
    def test_from_name(self):
        self.assertEqual(struct_scan_angle(_frame("d030")), 30.0)
        self.assertEqual(struct_scan_angle(_frame("d000")), 0.0)
        self.assertEqual(_normalize_scan_angle(360.0), 0.0)

    def test_from_constraint_dict(self):
        s = _frame("s000")
        s.data["constraints"] = [{"type": "dihedral", "value": 120.0}]
        self.assertEqual(struct_scan_angle(s), 120.0)


class TestAlignScanProfiles(unittest.TestCase):
    def test_intersection_keeps_shared_sorted(self):
        hl = ListOfStruct.from_structs_shared(
            [_frame("d000", 1.0), _frame("d010", 2.0), _frame("d020", 3.0)]
        )
        ll = ListOfStruct.from_structs_shared(
            [
                _frame("d000", 10.0),
                _frame("d010", 20.0),
                _frame("d020", 30.0),
                _frame("d030", 40.0),  # LL-only
            ]
        )
        ahl, all_, info = align_scan_profiles(hl, ll, hl_path="hl.json", ll_path="ll.json")
        self.assertEqual(info["n_common"], 3)
        self.assertEqual(info["ll_only"], [30.0])
        self.assertEqual(info["hl_only"], [])
        self.assertEqual([s.data["name"] for s in ahl.structs], ["d000", "d010", "d020"])
        self.assertEqual([s.data["name"] for s in all_.structs], ["d000", "d010", "d020"])
        self.assertEqual(len(ahl.structs), len(all_.structs))

    def test_too_few_shared_raises(self):
        hl = ListOfStruct.from_structs_shared([_frame("d000"), _frame("d010")])
        ll = ListOfStruct.from_structs_shared([_frame("d100"), _frame("d110")])
        with self.assertRaises(Exception) as ctx:
            align_scan_profiles(hl, ll, min_points=3)
        self.assertIn("shared points", str(ctx.exception))

    def test_equal_length_different_angles_still_aligns(self):
        """Same n frames but shifted angle labels must still angle-align."""
        hl = ListOfStruct.from_structs_shared(
            [
                _frame("d000", 1.0),
                _frame("d010", 2.0),
                _frame("d020", 3.0),
                _frame("d040", 3.5),
            ]
        )
        ll = ListOfStruct.from_structs_shared(
            [
                _frame("d010", 4.0),
                _frame("d020", 5.0),
                _frame("d030", 6.0),
                _frame("d040", 6.5),
            ]
        )
        ahl, all_, info = align_scan_profiles(hl, ll)
        self.assertEqual(info["n_common"], 3)
        self.assertEqual(
            [s.data["name"] for s in ahl.structs], ["d010", "d020", "d040"]
        )
        self.assertEqual(
            [s.data["name"] for s in all_.structs], ["d010", "d020", "d040"]
        )


if __name__ == "__main__":
    unittest.main()
