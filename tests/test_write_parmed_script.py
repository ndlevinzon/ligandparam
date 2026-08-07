"""Ensure WriteParmedScript emits parseable Python (progress prints safe)."""

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ffpopt.Dihedrals import MultiDihedFcn, PrimDihedFcn, WriteParmedScript


class _Atom:
    def __init__(self, idx, name):
        self.idx = idx
        self.name = name
        self.residue = SimpleNamespace(name="LIG")


class _Parm:
    def __init__(self):
        self.atoms = [_Atom(i, n) for i, n in enumerate(["C11", "C12", "C13", "S1"])]


class TestWriteParmedScript(unittest.TestCase):
    def test_generated_script_parses(self):
        prims = [
            PrimDihedFcn(1.0, 0.0, 1),
            PrimDihedFcn(0.5, 180.0, 2),
            PrimDihedFcn(0.1, 0.0, 3),
        ]
        dfcn = MultiDihedFcn([0, 1, 2, 3], prims)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "it01.py"
            WriteParmedScript(str(path), _Parm(), [dfcn])
            src = path.read_text()
        ast.parse(src)
        self.assertIn("delete+add idxs=0-1-2-3", src)
        self.assertIn('f":{rname}@C11"', src)
        # Broken pattern from the prior progress-print bug.
        self.assertNotIn('delete+add f":{rname}', src)


if __name__ == "__main__":
    unittest.main()
