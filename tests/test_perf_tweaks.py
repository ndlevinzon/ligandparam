"""Tests for clash vectorization, GetGraph cache, compact JSON, SANDER scratch."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr
from unittest.mock import MagicMock

import numpy as np
import pytest

from ffpopt.Constraints import has_nonbonded_clash
from ffpopt.GeomOpt import _geomopt_fallback_note
from ffpopt.Struct import ListOfStruct, Struct
from ffpopt.ase.calculator import _scratch_atoms_energy_forces


def _water_like_struct() -> Struct:
    return Struct(
        {
            "elements": ["O", "H", "H"],
            "positions": [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
            "bonds": [[0, 1], [0, 2]],
            "charges": [0.0, 0.0, 0.0],
            "names": ["O", "H1", "H2"],
            "types": ["oh", "ho", "ho"],
            "spin": 1,
            "constraints": [],
            "restraints": [],
            "name": "wat",
            "energy": None,
            "forces": None,
        }
    )


def test_get_graph_returns_cached_object():
    s = _water_like_struct()
    g1 = s.GetGraph()
    g2 = s.GetGraph()
    assert g1 is g2
    assert g1 is s.graph


def test_has_nonbonded_clash_detects_and_skips_bonds():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # bonded close — ok
            [0.1, 0.1, 0.0],  # nonbonded clash with 0
        ],
        dtype=float,
    )
    bonds = [[0, 1]]
    clashed, i, j, dist = has_nonbonded_clash(pos, bonds, min_dist=0.8)
    assert clashed
    assert {i, j} == {0, 2}
    assert dist < 0.8

    pos2 = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float)
    ok, *_ = has_nonbonded_clash(pos2, [[0, 1]], min_dist=0.8)
    assert not ok


def test_list_of_struct_save_compact(tmp_path):
    s = _water_like_struct()
    los = ListOfStruct([s])
    path = tmp_path / "out.json"
    los.save(path)
    text = path.read_text()
    assert "\n    " not in text
    data = json.loads(text)
    assert data[0]["name"] == "wat"

    los.save(path, indent=4)
    assert "\n    " in path.read_text()


def test_clear_runtime_caches():
    los = ListOfStruct([])
    calc = MagicMock()
    los.calc = calc
    los._ffpopt_calc_cache = ("key", object())
    los.clear_runtime_caches()
    assert los.calc is None
    assert los._ffpopt_calc_cache is None
    calc.reset.assert_called_once()


def test_geomopt_fallback_note_quiet_by_default(monkeypatch):
    monkeypatch.delenv("FFPOPT_GEOMOPT_TRACEBACK", raising=False)
    buf = io.StringIO()
    with redirect_stderr(buf):
        _geomopt_fallback_note("ASE", RuntimeError("boom"), "geomeTRIC")
    out = buf.getvalue()
    assert "ASE geomopt failed" in out
    assert "RuntimeError: boom" in out
    assert "Traceback" not in out


def test_scratch_atoms_reused():
    import ase
    from ase.calculators.calculator import Calculator, all_changes

    class ConstCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            self.results["energy"] = 1.23
            self.results["forces"] = np.zeros((len(self.atoms), 3))

    class Wrapper:
        def __init__(self):
            self.atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
            self.atoms.set_initial_charges([0.0, 0.0])
            self._scratch_atoms = None

    wrapper = Wrapper()
    const = ConstCalc()
    e1, f1 = _scratch_atoms_energy_forces(wrapper, const)
    scratch1 = wrapper._scratch_atoms
    wrapper.atoms.set_positions([[0, 0, 0], [0.8, 0, 0]])
    e2, f2 = _scratch_atoms_energy_forces(wrapper, const)
    assert wrapper._scratch_atoms is scratch1
    assert e1 == pytest.approx(1.23)
    assert e2 == pytest.approx(1.23)
    assert f1.shape == (2, 3)
