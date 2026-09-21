"""Open Babel SDS-style PDBs must not gain a sulfate hydrogen."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _paths  # noqa: E402


def setUpModule():
    _paths.ensure_ligandparam()


# Anionic sulfate oxygen as Open Babel writes it (element "O1-") plus a
# duplicate CONECT for S=O. 4 atoms: C-O-S-O-
_SDS_TAIL_PDB = """\
HETATM    1  C   UNL     1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  O   UNL     1       1.400   0.000   0.000  1.00  0.00           O
HETATM    3  S   UNL     1       2.100   0.000   1.200  1.00  0.00           S
HETATM    4  O   UNL     1       3.500   0.000   1.200  1.00  0.00           O1-
CONECT    1    2
CONECT    2    1    3
CONECT    3    2    4    4
CONECT    4    3    3
END
"""


class TestPdbSanitize(unittest.TestCase):
    def test_o1minus_becomes_oxygen_with_charge(self):
        from ligandparam.io.Coordinates import _rewrite_pdb_atom_line

        line = (
            "HETATM   17  O   UNL     1      13.945   5.161   6.844  1.00  0.00"
            "           O1-\n"
        )
        out = _rewrite_pdb_atom_line(line)
        self.assertEqual(out[76:78].strip(), "O")
        self.assertEqual(out[78:80].strip(), "1-")
        self.assertNotIn("O1-", out)

    def test_sanitize_keeps_atom_count_and_unique_conect(self):
        from ligandparam.io.Coordinates import count_structure_atoms, sanitize_pdb_ligand

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "sds.pdb"
            dst = Path(td) / "sds.sanitized.pdb"
            src.write_text(_SDS_TAIL_PDB, encoding="utf-8")
            sanitize_pdb_ligand(src, dst)
            self.assertEqual(count_structure_atoms(src), 4)
            self.assertEqual(count_structure_atoms(dst), 4)
            text = dst.read_text(encoding="utf-8")
            self.assertIn("O1-", src.read_text(encoding="utf-8"))
            self.assertNotIn("O1-", text)
            conect = [ln for ln in text.splitlines() if ln.startswith("CONECT")]
            s_line = [ln for ln in conect if ln.split()[1] == "3"][0]
            partners = s_line.split()[2:]
            self.assertEqual(partners, ["2", "4"])

    def test_split_element_charge_tokens(self):
        from ligandparam.io.Coordinates import _split_pdb_element_charge

        self.assertEqual(_split_pdb_element_charge("O1-"), ("O", "1-"))
        self.assertEqual(_split_pdb_element_charge("O-1"), ("O", "1-"))
        self.assertEqual(_split_pdb_element_charge("N+"), ("N", "1+"))
        self.assertEqual(_split_pdb_element_charge("Cl"), ("Cl", ""))
        self.assertEqual(_split_pdb_element_charge("C"), ("C", ""))

    def test_drop_extra_hydrogen_against_reference(self):
        import numpy as np

        from ligandparam.io.Coordinates import match_current_to_reference

        ref = [
            ("C", np.array([0.0, 0.0, 0.0])),
            ("O", np.array([1.4, 0.0, 0.0])),
            ("S", np.array([2.1, 0.0, 1.2])),
            ("O", np.array([3.5, 0.0, 1.2])),
            ("H", np.array([-0.9, 0.0, 0.0])),
        ]
        extra_h = ("H", np.array([4.1, 0.4, 0.8]))
        current = list(ref) + [extra_h]
        elems, coords = match_current_to_reference(ref, current)
        self.assertEqual(len(elems), 5)
        self.assertEqual(elems.count("H"), 1)
        self.assertEqual(len(coords), 5)
