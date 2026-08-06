"""Tests for shared Amber ligand bundle resolution."""

from pathlib import Path

import pytest

from ligandparam.io.amber_bundle import AmberLigandBundle, resolve_getparam_bundle


def _touch_triplet(work_dir: Path, stem: str) -> None:
    (work_dir / f"{stem}.mol2").write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    (work_dir / f"{stem}.lib").write_text("!entry\n", encoding="utf-8")
    (work_dir / f"{stem}.frcmod").write_text("Remark line\n", encoding="utf-8")


def test_resolve_explicit_paths(tmp_path: Path):
    _touch_triplet(tmp_path, "chaps")
    bundle = resolve_getparam_bundle(
        mol2=tmp_path / "chaps.mol2",
        lib=tmp_path / "chaps.lib",
        frcmod=tmp_path / "chaps.frcmod",
    )
    assert isinstance(bundle, AmberLigandBundle)
    assert bundle.stem == "chaps"
    assert bundle.work_dir == tmp_path.resolve()
    assert bundle.mol2.name == "chaps.mol2"


def test_resolve_getparam_layout_with_label(tmp_path: Path):
    work = tmp_path / "CHA3" / "CHA"
    work.mkdir(parents=True)
    _touch_triplet(work, "chaps")
    bundle = resolve_getparam_bundle(
        cwd=tmp_path,
        data_cwd="CHA3",
        resname="CHA",
        label="chaps",
    )
    assert bundle.stem == "chaps"
    assert bundle.work_dir == work.resolve()


def test_resolve_unique_mol2_fallback(tmp_path: Path):
    work = tmp_path / "OUT" / "LIG"
    work.mkdir(parents=True)
    _touch_triplet(work, "ligand")
    (work / "ligand.initial.mol2").write_text("x\n", encoding="utf-8")
    bundle = resolve_getparam_bundle(
        cwd=tmp_path,
        data_cwd="OUT",
        resname="LIG",
    )
    assert bundle.stem == "ligand"


def test_to_scission_input(tmp_path: Path):
    pytest.importorskip("scission")
    _touch_triplet(tmp_path, "LIG")
    bundle = resolve_getparam_bundle(
        mol2=tmp_path / "LIG.mol2",
        lib=tmp_path / "LIG.lib",
        frcmod=tmp_path / "LIG.frcmod",
    )
    inp = bundle.to_scission_input()
    assert inp.mol2_path == bundle.mol2
    assert inp.lib_path == bundle.lib
    assert inp.frcmod_path == bundle.frcmod


def test_missing_triplet_raises(tmp_path: Path):
    work = tmp_path / "A" / "B"
    work.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Missing"):
        resolve_getparam_bundle(cwd=tmp_path, data_cwd="A", resname="B", label="x")
