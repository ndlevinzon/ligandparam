"""Scission writers / merge smoke tests (no AmberTools)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scission.models import SelectedFragment
from scission.writers import write_fragment_index


class TestWriteFragmentIndex(unittest.TestCase):
    def test_writes_json_index(self):
        frag = SelectedFragment(
            fragment_id="frag_0001",
            source_candidate_id="cand_a",
            retained_atoms=[0, 1, 2],
            cut_bonds=[(2, 3)],
            cap_atoms=[],
            torsions=["t1"],
            fit_torsions=[],
            parent_atom_map={0: 0, 1: 1, 2: 2},
            manifest_path=Path("frags/frag_0001/manifest.json"),
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            path = write_fragment_index([frag], out)
            self.assertTrue(path.is_file())
            data = json.loads(path.read_text())
            self.assertEqual(data["fragments"][0]["fragment_id"], "frag_0001")
            self.assertEqual(data["fragments"][0]["torsions"], ["t1"])


class TestMergeHelpers(unittest.TestCase):
    def test_normalize_param_name(self):
        from scission.merge import _normalize_param_name_to_key

        key = _normalize_param_name_to_key("c3-c3-c3-c3")
        self.assertEqual(key, ("c3", "c3", "c3", "c3"))
        self.assertIsNone(_normalize_param_name_to_key("not-a-torsion"))

    def test_merge_skips_empty_fragment_dir(self):
        from scission.frcmod import FrcmodFile
        from scission.merge import merge_fragment_frcmods

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent.frcmod"
            # Minimal frcmod text FrcmodFile can read.
            parent.write_text(
                "Remark line goes here\n"
                "MASS\n"
                "\n"
                "BOND\n"
                "\n"
                "ANGLE\n"
                "\n"
                "DIHE\n"
                "c3-c3-c3-c3 1 0.00 0.0 1.\n"
                "\n"
                "IMPROPER\n"
                "\n"
                "NONB\n"
                "\n"
            )
            empty_frag = root / "frag_empty"
            empty_frag.mkdir()
            out = root / "merged.frcmod"
            with patch("scission.merge.warnings.warn"):
                report = merge_fragment_frcmods(parent, out, [empty_frag])
            self.assertTrue(out.is_file())
            self.assertGreaterEqual(len(report.get("skipped_fragments", [])), 1)


if __name__ == "__main__":
    unittest.main()
