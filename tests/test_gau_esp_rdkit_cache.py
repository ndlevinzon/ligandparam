"""Tests for RDKit mol / SMARTS caches and faster Gaussian ESP parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ligandparam.io.gaussian_io import GaussianReader
from ligandparam.multiresp.respfunctions import ReadGauEsp, _coords_after_is_at


class TestCoordsAfterIsAt(unittest.TestCase):
    def test_parses_standard_line(self):
        line = "       Atomic Center    1 is at   -0.123456    1.234567    0.000000\n"
        self.assertEqual(
            _coords_after_is_at(line),
            [-0.123456, 1.234567, 0.0],
        )

    def test_rejects_garbage(self):
        self.assertIsNone(_coords_after_is_at("no coordinates here"))


class TestReadGauEspFast(unittest.TestCase):
    def test_roundtrip_minimal_log(self):
        text = """\
 Gaussian log header
 Atomic Center    1 is at    0.000000    0.000000    0.000000
 Atomic Center    2 is at    1.000000    0.000000    0.000000
 ESP Fit Center    1 is at    0.500000    0.500000    0.000000
 ESP Fit Center    2 is at    0.500000   -0.500000    0.000000
     1 Fit     -0.100000
     2 Fit      0.200000
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "esp.log"
            path.write_text(text)
            crds, pts, esp = ReadGauEsp(str(path))
        self.assertEqual(crds, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(pts, [0.5, 0.5, 0.0, 0.5, -0.5, 0.0])
        self.assertEqual(esp, [-0.1, 0.2])


class TestCheckCompleteTail(unittest.TestCase):
    def test_finds_marker_in_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.log"
            path.write_bytes(b"x" * 20000 + b"\n Normal termination of Gaussian\n")
            self.assertTrue(GaussianReader(path).check_complete())

    def test_missing_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.log"
            path.write_bytes(b"still running\n" * 1000)
            self.assertFalse(GaussianReader(path).check_complete())

    def test_missing_file(self):
        self.assertFalse(GaussianReader("no_such_gaussian.log").check_complete())


class TestRdkitMolCache(unittest.TestCase):
    def test_mol_and_smarts_cached(self):
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            self.skipTest("RDKit not installed")

        from scission import torsions as torsions_mod
        from scission.models import Atom, Bond, Ligand
        from scission.rdkit_mol import build_rdkit_mol as real_build
        from scission.torsions import (
            _COMPILED_CENTRAL_BOND_SMARTS,
            _build_rdkit_mol,
            _compiled_central_bond_smarts,
            match_central_bond_smarts,
        )

        atoms = [
            Atom(1, "C1", "C", "c3", -0.1, (0.0, 0.0, 0.0)),
            Atom(2, "C2", "C", "c3", -0.1, (1.5, 0.0, 0.0)),
            Atom(3, "C3", "C", "c3", -0.1, (2.0, 1.4, 0.0)),
            Atom(4, "C4", "C", "c3", -0.1, (3.5, 1.4, 0.0)),
        ]
        bonds = [
            Bond(1, 1, 2, "1"),
            Bond(2, 2, 3, "1"),
            Bond(3, 3, 4, "1"),
        ]
        ligand = Ligand(
            name="BUT",
            atoms=atoms,
            bonds=bonds,
            lib_atom_names=[a.name for a in atoms],
            lib_atom_types={a.name: a.atom_type for a in atoms},
            frcmod_text="",
            mol2_path=Path("but.mol2"),
            lib_path=Path("but.lib"),
            frcmod_path=Path("but.frcmod"),
        )
        with patch.object(torsions_mod, "build_rdkit_mol", wraps=real_build) as mock_build:
            m1 = _build_rdkit_mol(ligand)
            m2 = _build_rdkit_mol(ligand)
        self.assertIs(m1, m2)
        self.assertEqual(mock_build.call_count, 1)

        pattern = "[C:1]-[C:2]"
        _COMPILED_CENTRAL_BOND_SMARTS.pop(pattern, None)
        with patch.object(torsions_mod.Chem, "MolFromSmarts", wraps=Chem.MolFromSmarts) as mock_smarts:
            _compiled_central_bond_smarts(pattern)
            _compiled_central_bond_smarts(pattern)
        self.assertEqual(mock_smarts.call_count, 1)

        matched = match_central_bond_smarts(ligand, (pattern,))
        self.assertTrue(matched)


if __name__ == "__main__":
    unittest.main()
