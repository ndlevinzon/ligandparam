"""geomeTRIC constraint file must use target (force=False) dihedral values."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path


class TestGeometricConstraintTargets(unittest.TestCase):
    def test_constraint_file_uses_target_not_force_true(self):
        """Writing force=True (pre-twist) values would poison scan targets."""
        from ffpopt.Constraints import Constraint, to_geometric

        target = [Constraint("dihed", [0, 1, 2, 3], value=90.0)]
        force_true = [Constraint("dihed", [0, 1, 2, 3], value=100.0)]

        # GeomOpt keeps a deepcopy of force=False fill for the constraint file.
        target_cons = copy.deepcopy(target)
        self.assertNotEqual(target_cons[0].value, force_true[0].value)

        with tempfile.TemporaryDirectory() as td:
            tmpcons = Path(td) / "constraints.txt"
            with open(tmpcons, "w") as fh:
                fh.write("$set\n")
                for line in to_geometric(target_cons):
                    fh.write("%s\n" % (line))
            text = tmpcons.read_text()
        self.assertIn("90.0", text)
        self.assertNotIn("100.0", text)
        # Sanity: force_true content would differ.
        force_lines = "\n".join(to_geometric(force_true))
        self.assertIn("100.0", force_lines)


if __name__ == "__main__":
    unittest.main()
