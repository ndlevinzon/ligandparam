from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import networkx as nx

from scission.capping import plan_caps
from scission.fragments import _cut_bonds_for_retained, build_candidate_fragments
from scission.graph import build_graph
from scission.io import load_ligand, load_ligand_from_mol2
from scission.merge import (
    MergeWarning,
    discover_fragment_dirs,
    find_latest_iteration_frcmod,
    merge_fragment_frcmods,
)
from scission.models import Atom, CapSite, FragmentConfig, InputBundle
from scission.pipeline import fragment_ligand
from scission.screen import screen_candidate
from scission.torsions import enumerate_torsions, match_central_bond_smarts


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_ligand"
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "jmc2025-1"


def load_fixture(bundle_name: str = "ligand.lib"):
    """Load the small ligand fixture used across the test suite.

    Args:
        bundle_name: Name of the Amber LIB fixture to pair with the MOL2 file.

    Returns:
        Parsed ligand fixture used by the tests.
    """

    return load_ligand(
        InputBundle(
            mol2_path=FIXTURE_DIR / "ligand.mol2",
            lib_path=FIXTURE_DIR / bundle_name,
            frcmod_path=FIXTURE_DIR / "ligand.frcmod",
        )
    )


def load_amide_fixture(tmp_path: Path):
    """Create and load a tiny amide-like ligand fixture.

    Args:
        tmp_path: Temporary pytest directory used to write the input triplet.

    Returns:
        Parsed ligand containing an amide-like C-N single bond.
    """

    mol2_path = tmp_path / "amide.mol2"
    lib_path = tmp_path / "amide.lib"
    frcmod_path = tmp_path / "amide.frcmod"
    tmp_path.mkdir(parents=True, exist_ok=True)

    mol2_path.write_text(
        """@<TRIPOS>MOLECULE
AMIDE
 5 4 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 c3      1 LIG       -0.1000
      2 C2          1.5300    0.0000    0.0000 c       1 LIG        0.5000
      3 O1          2.3000    1.1000    0.0000 o       1 LIG       -0.5000
      4 N1          2.9000   -0.1000    0.0000 n       1 LIG       -0.3000
      5 C3          4.4300   -0.1000    0.0000 c3      1 LIG        0.4000
@<TRIPOS>BOND
     1    1    2 1
     2    2    3 2
     3    2    4 am
     4    4    5 1
"""
    )
    lib_path.write_text(
        """!entry.AMIDE.unit.atoms table  str name  str type
"C1" "c3"
"C2" "c"
"O1" "o"
"N1" "n"
"C3" "c3"
"""
    )
    frcmod_path.write_text("Remark line goes here\n")

    return load_ligand(
        InputBundle(
            mol2_path=mol2_path,
            lib_path=lib_path,
            frcmod_path=frcmod_path,
        )
    )


def load_imine_fixture(tmp_path: Path):
    """Create and load a tiny ligand with an exocyclic C=N bond."""

    mol2_path = tmp_path / "imine.mol2"
    lib_path = tmp_path / "imine.lib"
    frcmod_path = tmp_path / "imine.frcmod"
    tmp_path.mkdir(parents=True, exist_ok=True)

    mol2_path.write_text(
        """@<TRIPOS>MOLECULE
IMINE
 6 5 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1         -1.5300    0.0000    0.0000 c3      1 LIG       -0.1000
      2 C2          0.0000    0.0000    0.0000 c2      1 LIG        0.2000
      3 N1          1.3000    0.0000    0.0000 n2      1 LIG       -0.2000
      4 C3          2.7500    0.0000    0.0000 c3      1 LIG        0.1000
      5 O1         -0.4000    1.2000    0.0000 o       1 LIG       -0.3000
      6 C4          4.2800    0.0000    0.0000 c3      1 LIG        0.3000
@<TRIPOS>BOND
     1    1    2 1
     2    2    3 2
     3    3    4 1
     4    2    5 1
     5    4    6 1
"""
    )
    lib_path.write_text(
        """!entry.IMINE.unit.atoms table  str name  str type
"C1" "c3"
"C2" "c2"
"N1" "n2"
"C3" "c3"
"O1" "o"
"C4" "c3"
"""
    )
    frcmod_path.write_text("Remark line goes here\n")

    return load_ligand(
        InputBundle(
            mol2_path=mol2_path,
            lib_path=lib_path,
            frcmod_path=frcmod_path,
        )
    )


def load_ring_imine_fixture(tmp_path: Path):
    """Create and load a tiny ligand with a ring C=N bond."""

    mol2_path = tmp_path / "ring_imine.mol2"
    lib_path = tmp_path / "ring_imine.lib"
    frcmod_path = tmp_path / "ring_imine.frcmod"
    tmp_path.mkdir(parents=True, exist_ok=True)

    mol2_path.write_text(
        """@<TRIPOS>MOLECULE
RINGIMINE
 5 5 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1         -1.0000    0.0000    0.0000 c3      1 LIG       -0.1000
      2 C2          0.0000    0.0000    0.0000 c2      1 LIG        0.2000
      3 N1          1.2000    0.0000    0.0000 n       1 LIG       -0.2000
      4 C3          0.2000   -1.1000    0.0000 c3      1 LIG        0.1000
      5 O1          0.0000    1.2000    0.0000 o       1 LIG       -0.3000
@<TRIPOS>BOND
     1    1    2 1
     2    2    3 2
     3    3    4 1
     4    4    1 1
     5    2    5 1
"""
    )
    lib_path.write_text(
        """!entry.RINGIMINE.unit.atoms table  str name  str type
"C1" "c3"
"C2" "c2"
"N1" "n"
"C3" "c3"
"O1" "o"
"""
    )
    frcmod_path.write_text("Remark line goes here\n")

    return load_ligand(
        InputBundle(
            mol2_path=mol2_path,
            lib_path=lib_path,
            frcmod_path=frcmod_path,
        )
    )


def test_load_ligand_accepts_consistent_triplet():
    """The parser should accept matching MOL2, LIB, and FRCMOD inputs."""

    ligand = load_fixture()
    assert ligand.name == "SAMPLE"
    assert len(ligand.atoms) == 8
    assert ligand.lib_atom_names[:3] == ["C1", "C2", "C3"]


def test_load_ligand_rejects_atom_order_mismatch():
    """The loader should reject mismatched atom ordering between files."""

    with pytest.raises(ValueError, match="atom order mismatch"):
        load_fixture("ligand_bad_order.lib")


def test_enumerate_torsions_finds_acyclic_rotatable_bonds():
    """Rotatable torsions should be enumerated without hydrogen termini."""

    ligand = load_fixture()
    torsions = enumerate_torsions(ligand)
    labels = [torsion.label for torsion in torsions]
    assert "C1-C2-C3-C5" in labels
    assert "C2-C3-C5-C6" in labels
    assert all("H1" not in label for label in labels)


def test_enumerate_torsions_includes_amide_like_single_bonds_by_default(tmp_path: Path):
    """Amide-like acyclic single bonds should be included by default."""

    ligand = load_amide_fixture(tmp_path)
    labels = [torsion.label for torsion in enumerate_torsions(ligand)]
    assert "O1-C2-N1-C3" in labels


def test_enumerate_torsions_can_use_strict_legacy_rotatable_definition(tmp_path: Path):
    """The opt-out flag should restore the stricter legacy amide exclusion."""

    ligand = load_amide_fixture(tmp_path)
    labels = [
        torsion.label
        for torsion in enumerate_torsions(
            ligand,
            include_rigid_single_bonds=False,
        )
    ]
    assert "O1-C2-N1-C3" not in labels


def test_smarts_override_can_restore_excluded_amide_like_bonds(tmp_path: Path):
    """SMARTS overrides should re-include otherwise excluded non-ring bonds."""

    pytest.importorskip("rdkit")
    ligand = load_amide_fixture(tmp_path)
    labels = [
        torsion.label
        for torsion in enumerate_torsions(
            ligand,
            include_rigid_single_bonds=False,
            rotatable_bond_smarts=("[C:1](=[O])[N:2]",),
        )
    ]
    assert "O1-C2-N1-C3" in labels


def test_smarts_override_can_include_non_single_non_ring_bonds(tmp_path: Path):
    """SMARTS overrides should admit valid non-ring bonds like exocyclic imines."""

    pytest.importorskip("rdkit")
    ligand = load_imine_fixture(tmp_path)
    assert "O1-C2-N1-C3" not in [torsion.label for torsion in enumerate_torsions(ligand)]
    labels = [
        torsion.label
        for torsion in enumerate_torsions(
            ligand,
            rotatable_bond_smarts=("[C:1]=[N:2]",),
        )
    ]
    assert "O1-C2-N1-C3" in labels


def test_smarts_override_cannot_force_ring_bonds(tmp_path: Path):
    """SMARTS overrides should still reject ring bonds as scan targets."""

    pytest.importorskip("rdkit")
    ligand = load_ring_imine_fixture(tmp_path)
    labels = [
        torsion.label
        for torsion in enumerate_torsions(
            ligand,
            rotatable_bond_smarts=("[C:1]=[N:2]",),
        )
    ]
    assert "O1-C2-N1-C3" not in labels


def test_smarts_override_requires_mapped_central_bond_atoms(tmp_path: Path):
    """SMARTS overrides should require :1 and :2 on the central bond."""

    pytest.importorskip("rdkit")
    ligand = load_amide_fixture(tmp_path)
    with pytest.raises(ValueError, match=":1 and :2"):
        enumerate_torsions(
            ligand,
            rotatable_bond_smarts=("[C](=[O])[N]",),
        )


def test_match_central_bond_smarts_returns_normalized_parent_bonds(tmp_path: Path):
    """The public matcher should map :1/:2 patterns to parent atom pairs."""

    pytest.importorskip("rdkit")
    ligand = load_imine_fixture(tmp_path)
    matched = match_central_bond_smarts(ligand, ("[C:1]=[N:2]",))
    # C2 (index 2) is double-bonded to N1 (index 3).
    assert (2, 3) in matched


def test_from_dict_parses_restrict_to_bond_smarts():
    """from_dict should accept restrict_to_bond_smarts as a string or list."""

    single = FragmentConfig.from_dict({"restrict_to_bond_smarts": "[C:1]=[N:2]"})
    assert single.restrict_to_bond_smarts == ("[C:1]=[N:2]",)
    many = FragmentConfig.from_dict(
        {"restrict_to_bond_smarts": ["[C:1]=[N:2]", "[c:1][c:2]"]}
    )
    assert many.restrict_to_bond_smarts == ("[C:1]=[N:2]", "[c:1][c:2]")
    assert FragmentConfig.from_dict({}).restrict_to_bond_smarts == ()


def test_restrict_to_bond_smarts_keeps_only_matching_torsions(tmp_path: Path):
    """Restriction should fit only allow-listed bonds and reject the rest."""

    pytest.importorskip("rdkit")
    ligand = load_imine_fixture(tmp_path / "input")
    result = fragment_ligand(
        InputBundle(
            mol2_path=ligand.mol2_path,
            lib_path=ligand.lib_path,
            frcmod_path=ligand.frcmod_path,
        ),
        tmp_path / "out",
        FragmentConfig(
            rotatable_bond_smarts=("[C:1]=[N:2]",),
            restrict_to_bond_smarts=("[C:1]=[N:2]",),
            use_parent_fallback=True,
        ),
    )
    # The C2=N1 torsion is allow-listed and fitted ...
    assert "O1-C2-N1-C3" in result.covered_torsions
    # ... while the default-rotatable N1-C3 torsion is dropped by restriction.
    assert "C2-N1-C3-C4" not in result.covered_torsions
    assert (
        result.rejected_torsions.get("C2-N1-C3-C4")
        == "excluded_by_restrict_to_bond_smarts"
    )


def test_restrict_to_bond_smarts_with_no_match_warns_and_selects_nothing(tmp_path: Path):
    """A restriction that matches no torsion should warn and select no fragments."""

    pytest.importorskip("rdkit")
    ligand = load_imine_fixture(tmp_path / "input")
    result = fragment_ligand(
        InputBundle(
            mol2_path=ligand.mol2_path,
            lib_path=ligand.lib_path,
            frcmod_path=ligand.frcmod_path,
        ),
        tmp_path / "out",
        FragmentConfig(restrict_to_bond_smarts=("[Cl:1][Cl:2]",)),
    )
    assert result.selected_fragments == []
    assert any(
        "restrict_to_bond_smarts matched no" in warning for warning in result.warnings
    )


def test_candidate_generation_produces_caps_and_full_parent():
    """Candidate generation should include capped fragments and the fallback parent."""

    ligand = load_fixture()
    torsion = enumerate_torsions(ligand)[0]
    candidates = build_candidate_fragments(ligand, torsion)
    assert candidates
    assert any(candidate.cut_bonds for candidate in candidates[:-1])
    assert candidates[-1].retained_atoms == set(range(1, len(ligand.atoms) + 1))


def test_clash_screen_rejects_too_small_candidate_and_accepts_larger_one():
    """Clash screening should prefer the larger, less-crowded fragment."""

    ligand = load_fixture()
    torsion = enumerate_torsions(ligand)[1]
    candidates = build_candidate_fragments(ligand, torsion)
    small = candidates[0]
    large = candidates[-1]
    small_result = screen_candidate(
        ligand,
        torsion,
        small,
        angle_step=30,
        thresholds=FragmentConfig().clash_thresholds,
    )
    large_result = screen_candidate(
        ligand,
        torsion,
        large,
        angle_step=30,
        thresholds=FragmentConfig().clash_thresholds,
    )
    assert large_result.accepted
    assert small_result.worst_margin <= large_result.worst_margin


def test_pipeline_writes_expected_outputs(tmp_path: Path):
    """A full pipeline run should emit the expected fragment artifacts.

    Args:
        tmp_path: Temporary pytest directory for fragment outputs.
    """

    result = fragment_ligand(
        InputBundle(
            mol2_path=FIXTURE_DIR / "ligand.mol2",
            lib_path=FIXTURE_DIR / "ligand.lib",
            frcmod_path=FIXTURE_DIR / "ligand.frcmod",
        ),
        tmp_path,
        FragmentConfig(),
    )
    assert result.summary_path and result.summary_path.exists()
    assert result.fragment_index_path and result.fragment_index_path.exists()
    summary = json.loads(result.summary_path.read_text())
    fragment_index = json.loads(result.fragment_index_path.read_text())
    assert "selected_fragments" in summary
    assert "warnings" in summary
    assert isinstance(summary["warnings"], list)
    assert "fragments" in fragment_index
    for fragment in result.selected_fragments:
        assert fragment.fragment_id.startswith("fragment_")
        assert fragment.source_candidate_id
        assert fragment.manifest_path.exists()
        assert fragment.mol2_path.exists()
        assert fragment.xyz_path.exists()
        assert fragment.lib_path.exists()
        assert fragment.frcmod_path.exists()
        assert fragment.overview_image_path is not None
        assert fragment.overview_image_path.exists()
        assert fragment.torsion_image_paths
        for image_path in fragment.torsion_image_paths.values():
            assert image_path.exists()
        if fragment.parm7_path is not None:
            assert fragment.parm7_path.exists()
        if fragment.rst7_path is not None:
            assert fragment.rst7_path.exists()


def test_pipeline_honors_smarts_overrides_from_config(tmp_path: Path):
    """The full pipeline should honor SMARTS-nominated scan bonds."""

    pytest.importorskip("rdkit")
    ligand = load_imine_fixture(tmp_path / "input")
    result = fragment_ligand(
        InputBundle(
            mol2_path=ligand.mol2_path,
            lib_path=ligand.lib_path,
            frcmod_path=ligand.frcmod_path,
        ),
        tmp_path / "out",
        FragmentConfig(
            rotatable_bond_smarts=("[C:1]=[N:2]",),
            use_parent_fallback=True,
        ),
    )
    assert "O1-C2-N1-C3" in result.covered_torsions


def test_full_parent_fallback_is_not_counted_as_success_by_default(tmp_path: Path):
    """Parent fallback fragments should remain opt-in during selection.

    Args:
        tmp_path: Temporary pytest directory for fragment outputs.
    """

    result = fragment_ligand(
        InputBundle(
            mol2_path=EXAMPLE_DIR / "binder_jmc2025-1.mol2",
            lib_path=EXAMPLE_DIR / "binder_jmc2025-1.lib",
            frcmod_path=EXAMPLE_DIR / "binder_jmc2025-1.frcmod",
        ),
        tmp_path,
        FragmentConfig(),
    )
    assert all(not fragment.is_parent_fallback for fragment in result.selected_fragments)
    assert "C1-C2-O1-C3" in result.rejected_torsions
    assert "C11-S1-C12-C13" not in result.rejected_torsions
    assert "S1-C12-C13-C18" not in result.rejected_torsions


def test_example_candidates_do_not_break_rings_apart():
    """Generated example candidates should keep any touched ring intact."""

    ligand = load_ligand(
        InputBundle(
            mol2_path=EXAMPLE_DIR / "binder_jmc2025-1.mol2",
            lib_path=EXAMPLE_DIR / "binder_jmc2025-1.lib",
            frcmod_path=EXAMPLE_DIR / "binder_jmc2025-1.frcmod",
        )
    )
    cycles = [set(cycle) for cycle in nx.cycle_basis(build_graph(ligand))]
    for torsion in enumerate_torsions(ligand):
        for candidate in build_candidate_fragments(ligand, torsion):
            for cycle in cycles:
                assert not (candidate.retained_atoms & cycle) or cycle.issubset(candidate.retained_atoms)


def test_find_latest_iteration_frcmod_uses_highest_iteration_number(tmp_path: Path):
    """The merge helper should choose the highest-numbered iteration frcmod.

    Args:
        tmp_path: Temporary pytest directory for synthetic fragment outputs.
    """

    fragment_dir = tmp_path / "fragment_1"
    fragment_dir.mkdir()
    (fragment_dir / "it01.frcmod").write_text("Created by ParmEd\nDIHE\nold\n")
    (fragment_dir / "it12.frcmod").write_text("Created by ParmEd\nDIHE\nnew\n")
    (fragment_dir / "it03.frcmod").write_text("Created by ParmEd\nDIHE\nmid\n")

    assert find_latest_iteration_frcmod(fragment_dir) == fragment_dir / "it12.frcmod"


def test_discover_fragment_dirs_finds_direct_and_coupling_subdir_runs(tmp_path: Path):
    """Discovery must handle both ffpopt run layouts.

    A single-coupling-group fragment keeps its ``itX.frcmod`` directly in the
    fragment directory; a multi-coupling-group fragment keeps each group's
    fits in ``coupling_NN/`` subdirectories. Both must be discovered, and a
    fragment directory with no fits at all must be ignored.

    Args:
        tmp_path: Temporary pytest directory for synthetic fragment outputs.
    """

    # Single coupling group: fits directly in the fragment dir.
    single = tmp_path / "fragment_1"
    single.mkdir()
    (single / "it01.frcmod").write_text("Created by ParmEd\nDIHE\nx\n")

    # Multiple coupling groups: fits live one level down.
    multi = tmp_path / "fragment_2"
    (multi / "coupling_00").mkdir(parents=True)
    (multi / "coupling_00" / "it02.frcmod").write_text("Created by ParmEd\nDIHE\nx\n")
    (multi / "coupling_01").mkdir()
    (multi / "coupling_01" / "it01.frcmod").write_text("Created by ParmEd\nDIHE\nx\n")

    # Fragment with no fittable torsions: contributes nothing, but must say so.
    (tmp_path / "fragment_3").mkdir()

    with pytest.warns(MergeWarning, match="fragment_3"):
        discovered = discover_fragment_dirs(tmp_path)

    assert discovered == [
        single,
        multi / "coupling_00",
        multi / "coupling_01",
    ]


def test_merge_fragment_frcmods_replaces_parent_dihedrals_and_writes_report(tmp_path: Path):
    """Latest fitted fragment frcmods should overwrite matching parent DIHE terms.

    Args:
        tmp_path: Temporary pytest directory for synthetic merge inputs.
    """

    parent_frcmod = tmp_path / "parent.frcmod"
    parent_frcmod.write_text(
        "\n".join(
            [
                "Remark line goes here",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "c3-ca-ce-o    4    2.800       180.000           2.000      parent",
                "ca-ce-nf-cd   1    0.400         0.000           3.000      parent",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    fragment_one = tmp_path / "fragment_1"
    fragment_one.mkdir()
    (fragment_one / "fit_torsions.json").write_text(
        json.dumps(
            [
                {
                    "label": "C1-C2-C3-C4",
                    "parent_dihedral_atoms": [1, 2, 3, 4],
                    "parent_rotatable_bond": [2, 3],
                }
            ]
        )
    )
    (fragment_one / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "c3-ca-ce-o    1    -0.24275516    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "c3-ca-ce-o    1    -0.28317361    0.000  -2.0    SCEE=1.2 SCNB=2.0",
                "c3-ca-ce-o    1    -0.83848409    0.000   3.0    SCEE=1.2 SCNB=2.0",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    fragment_two = tmp_path / "fragment_2"
    fragment_two.mkdir()
    (fragment_two / "fit_torsions.json").write_text("[]")
    (fragment_two / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ce-nf-cd-ss    1    -1.05903010    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    output_frcmod = tmp_path / "merged.frcmod"
    report_path = tmp_path / "merge_report.json"

    report = merge_fragment_frcmods(
        parent_frcmod_path=parent_frcmod,
        output_frcmod_path=output_frcmod,
        fragment_dirs=[fragment_one, fragment_two],
        report_path=report_path,
    )

    merged_text = output_frcmod.read_text()
    assert "c3-ca-ce-o    4    2.800       180.000           2.000      parent" not in merged_text
    assert "c3-ca-ce-o    1    -0.243        0.000           -1.000     SCEE=1.2 SCNB=2.0  fragment=1" in merged_text
    assert "c3-ca-ce-o    1    -0.838        0.000            3.000     SCEE=1.2 SCNB=2.0  fragment=1" in merged_text
    assert "ca-ce-nf-cd   1    0.400         0.000           3.000      parent" in merged_text
    assert "ce-nf-cd-ss   1    -1.059        0.000           -1.000     SCEE=1.2 SCNB=2.0  fragment=2" in merged_text
    assert report["replacement_counts"] == {"replaced": 1, "added": 1}
    assert report_path.exists()
    written_report = json.loads(report_path.read_text())
    assert len(written_report["fragment_reports"]) == 2
    assert written_report["merged_dihedral_groups"][0]["fragment_label"] == "fragment=1"


def test_merge_fragment_frcmods_keeps_dihe_parameters_contiguous(tmp_path: Path):
    """Merged DIHE entries should not contain internal blank separators.

    Args:
        tmp_path: Temporary pytest directory for synthetic merge inputs.
    """

    parent_frcmod = tmp_path / "parent.frcmod"
    parent_frcmod.write_text(
        "\n".join(
            [
                "Remark line goes here",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ca-ce-nf-cd   1    0.400         0.000           3.000      parent",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    fragment_dir = tmp_path / "fragment_1"
    fragment_dir.mkdir()
    (fragment_dir / "fit_torsions.json").write_text("[]")
    (fragment_dir / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ce-nf-cd-ss    1    -1.05903010    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    output_frcmod = tmp_path / "merged.frcmod"
    merge_fragment_frcmods(
        parent_frcmod_path=parent_frcmod,
        output_frcmod_path=output_frcmod,
        fragment_dirs=[fragment_dir],
    )

    merged_text = output_frcmod.read_text()
    assert (
        "ca-ce-nf-cd   1    0.400         0.000           3.000      parent\n"
        "ce-nf-cd-ss   1    -1.059        0.000           -1.000     SCEE=1.2 SCNB=2.0  fragment=1"
    ) in merged_text


def test_merge_fragment_frcmods_falls_back_to_profiled_fit_keys(
    tmp_path: Path,
):
    """Plotted families gate the merge when a fit file lists no ``params``.

    Args:
        tmp_path: Temporary pytest directory for synthetic merge inputs.
    """

    parent_frcmod = tmp_path / "parent.frcmod"
    parent_frcmod.write_text(
        "\n".join(
            [
                "Remark line goes here",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ca-ca-ns-c    4    1.800       180.000           2.000      parent",
                "nb-ca-ns-c    4    1.800       180.000           2.000      parent",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    fragment_one = tmp_path / "fragment_1"
    fragment_one.mkdir()
    (fragment_one / "fit_torsions.json").write_text("[]")
    (fragment_one / "it01.fit.json").write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "profiles": [
                            {"plots": ["LIG_ca-ca-ns-c"]},
                        ]
                    }
                ]
            }
        )
    )
    (fragment_one / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ca-ca-ns-c     1     0.38739315    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "nb-ca-ns-c     1     0.79133730    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    fragment_two = tmp_path / "fragment_2"
    fragment_two.mkdir()
    (fragment_two / "fit_torsions.json").write_text("[]")
    (fragment_two / "it01.fit.json").write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "profiles": [
                            {"plots": ["LIG_nb-ca-ns-c"]},
                        ]
                    }
                ]
            }
        )
    )
    (fragment_two / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ca-ca-ns-c     1    -9.99999999    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "nb-ca-ns-c     1     0.55555555    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    output_frcmod = tmp_path / "merged.frcmod"

    report = merge_fragment_frcmods(
        parent_frcmod_path=parent_frcmod,
        output_frcmod_path=output_frcmod,
        fragment_dirs=[fragment_one, fragment_two],
    )

    merged_text = output_frcmod.read_text()
    assert "0.387" in merged_text
    assert "0.556" in merged_text
    assert "-9.99999999" not in merged_text
    assert "ca-ca-ns-c    1     0.387        0.000           -1.000     SCEE=1.2 SCNB=2.0  fragment=1" in merged_text
    assert "nb-ca-ns-c    1     0.556        0.000           -1.000     SCEE=1.2 SCNB=2.0  fragment=2" in merged_text
    assert report["replacement_counts"] == {"replaced": 2, "added": 0}


def _write_amidate_fragment(fragment_dir: Path) -> None:
    """Write a synthetic amidate fragment fit that couples four DIHE families.

    The fit optimizes the whole conjugated family around the scanned bonds but
    only plots two of them, mirroring real ``ffpopt`` output.

    Args:
        fragment_dir: Directory to populate with ``it01`` fit artifacts.
    """

    fragment_dir.mkdir()
    (fragment_dir / "fit_torsions.json").write_text("[]")
    (fragment_dir / "it01.fit.json").write_text(
        json.dumps(
            {
                "params": {
                    "LIG_ca-ce-nf-cd": {"nprim": 3},
                    "LIG_o-ce-nf-cd": {"nprim": 3},
                    "LIG_ce-nf-cd-nc": {"nprim": 3},
                    "LIG_ce-nf-cd-ss": {"nprim": 3},
                },
                "systems": [
                    {
                        "profiles": [
                            {"plots": ["LIG_ca-ce-nf-cd"]},
                            {"plots": ["LIG_ce-nf-cd-nc"]},
                        ]
                    }
                ],
            }
        )
    )
    (fragment_dir / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ca-ce-nf-cd    1     0.11111111    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "ca-ce-nf-cd    1     0.11222222    0.000   2.0    SCEE=1.2 SCNB=2.0",
                "o -ce-nf-cd    1     0.22111111    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "o -ce-nf-cd    1     0.22222222    0.000   2.0    SCEE=1.2 SCNB=2.0",
                "ce-nf-cd-nc    1     0.33111111    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "ce-nf-cd-nc    1     0.33222222    0.000   2.0    SCEE=1.2 SCNB=2.0",
                "ce-nf-cd-ss    1     0.44111111    0.000  -1.0    SCEE=1.2 SCNB=2.0",
                "ce-nf-cd-ss    1     0.44222222    0.000   2.0    SCEE=1.2 SCNB=2.0",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )


def test_merge_fragment_frcmods_promotes_whole_fitted_dihe_family(tmp_path: Path):
    """Every fitted DIHE family must be promoted, not just the scanned ones.

    A fragment fit relaxes the entire coupled torsion family around the scanned
    bonds. Merging only the plotted subset leaves the parent with a mixed
    torsional surface: part new fit, part old generic parameters.

    Args:
        tmp_path: Temporary pytest directory for synthetic merge inputs.
    """

    parent_frcmod = tmp_path / "parent.frcmod"
    parent_frcmod.write_text(
        "\n".join(
            [
                "Remark line goes here",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ca-ce-nf-cd   1    9.100         0.000           2.000      parent",
                "o -ce-nf-cd   1    9.200         0.000           2.000      parent",
                "ce-nf-cd-nc   1    9.300         0.000           2.000      parent",
                # Written in reverse order relative to the fitted key.
                "ss-cd-nf-ce   1    9.400         0.000           2.000      parent",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    fragment_dir = tmp_path / "fragment_1"
    _write_amidate_fragment(fragment_dir)

    output_frcmod = tmp_path / "merged.frcmod"
    report = merge_fragment_frcmods(
        parent_frcmod_path=parent_frcmod,
        output_frcmod_path=output_frcmod,
        fragment_dirs=[fragment_dir],
    )

    merged_text = output_frcmod.read_text()
    # All four coupled families come from the fit, including the unplotted ones.
    for value in ("0.111", "0.222", "0.331", "0.441"):
        assert value in merged_text
    # No generic parent value survives in the fitted family.
    for value in ("9.100", "9.200", "9.300", "9.400"):
        assert value not in merged_text
    # The reversed parent key is replaced, not duplicated alongside the fit.
    assert "ss-cd-nf-ce" not in merged_text
    assert merged_text.count("ce-nf-cd-ss") == 2
    assert report["replacement_counts"] == {"replaced": 4, "added": 0}
    assert report["conflicts"] == []

    merged = {entry["dihedral_key"]: entry for entry in report["merged_dihedral_groups"]}
    assert set(merged) == {
        "ca-ce-nf-cd",
        "ce-nf-cd-nc",
        "ce-nf-cd-ss",
        "cd-nf-ce-o",  # canonical orientation of o-ce-nf-cd
    }
    assert merged["ca-ce-nf-cd"]["directly_scanned"] is True
    assert merged["ce-nf-cd-ss"]["directly_scanned"] is False


def test_merge_fragment_frcmods_prefers_the_fragment_that_scanned_a_shared_family(
    tmp_path: Path,
):
    """A directly scanned fit outranks a coupled fit of the same family.

    Args:
        tmp_path: Temporary pytest directory for synthetic merge inputs.
    """

    parent_frcmod = tmp_path / "parent.frcmod"
    parent_frcmod.write_text(
        "\n".join(
            [
                "Remark line goes here",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ce-nf-cd-nc   1    9.300         0.000           2.000      parent",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    # Fragment 1 only picks up ce-nf-cd-nc as a coupled term; fragment 2 scans it.
    fragment_one = tmp_path / "fragment_1"
    fragment_one.mkdir()
    (fragment_one / "fit_torsions.json").write_text("[]")
    (fragment_one / "it01.fit.json").write_text(
        json.dumps(
            {
                "params": {"LIG_ce-nf-cd-nc": {"nprim": 3}},
                "systems": [{"profiles": [{"plots": ["LIG_ca-ce-nf-cd"]}]}],
            }
        )
    )
    (fragment_one / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ce-nf-cd-nc    1     0.11111111    0.000   2.0    SCEE=1.2 SCNB=2.0",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    fragment_two = tmp_path / "fragment_2"
    fragment_two.mkdir()
    (fragment_two / "fit_torsions.json").write_text("[]")
    (fragment_two / "it01.fit.json").write_text(
        json.dumps(
            {
                "params": {"LIG_ce-nf-cd-nc": {"nprim": 3}},
                "systems": [{"profiles": [{"plots": ["LIG_ce-nf-cd-nc"]}]}],
            }
        )
    )
    (fragment_two / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ce-nf-cd-nc    1     0.22222222    0.000   2.0    SCEE=1.2 SCNB=2.0",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    output_frcmod = tmp_path / "merged.frcmod"
    report = merge_fragment_frcmods(
        parent_frcmod_path=parent_frcmod,
        output_frcmod_path=output_frcmod,
        fragment_dirs=[fragment_one, fragment_two],
    )

    merged_text = output_frcmod.read_text()
    assert "0.222" in merged_text
    assert "0.111" not in merged_text
    assert report["conflicts"] == [
        {
            "dihedral_key": "ce-nf-cd-nc",
            "kept": str(fragment_two),
            "kept_scanned": True,
            "dropped": str(fragment_one),
            "dropped_scanned": False,
        }
    ]


def test_merge_fragment_frcmods_rejects_two_scans_of_the_same_family(tmp_path: Path):
    """Two fragments scanning one family is ambiguous and must fail loudly.

    Args:
        tmp_path: Temporary pytest directory for synthetic merge inputs.
    """

    parent_frcmod = tmp_path / "parent.frcmod"
    parent_frcmod.write_text(
        "\n".join(
            [
                "Remark line goes here",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ca-ce-nf-cd   1    9.100         0.000           2.000      parent",
                "o -ce-nf-cd   1    9.200         0.000           2.000      parent",
                "ce-nf-cd-nc   1    9.300         0.000           2.000      parent",
                "ss-cd-nf-ce   1    9.400         0.000           2.000      parent",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    _write_amidate_fragment(tmp_path / "fragment_1")
    _write_amidate_fragment(tmp_path / "fragment_2")

    with pytest.raises(ValueError, match="directly scanned the same DIHE family"):
        merge_fragment_frcmods(
            parent_frcmod_path=parent_frcmod,
            output_frcmod_path=tmp_path / "merged.frcmod",
            fragment_dirs=[tmp_path / "fragment_1", tmp_path / "fragment_2"],
        )


def _write_simple_parent(path: Path) -> None:
    """Write a minimal parent frcmod with one generic DIHE family.

    Args:
        path: Destination frcmod path.
    """

    path.write_text(
        "\n".join(
            [
                "Remark line goes here",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ca-ce-nf-cd   2    1.600       180.000           2.000      parent",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )


def test_merge_fragment_frcmods_skips_fragments_without_iterations(tmp_path: Path):
    """A fragment with no ``itXX.frcmod`` warns instead of aborting the merge.

    ffpopt writes no iteration output when a fragment needed no refit — its
    starting parameters were already good, or the high-level profile came back
    flat. The other fragments must still merge.

    Args:
        tmp_path: Temporary pytest directory for synthetic merge inputs.
    """

    parent_frcmod = tmp_path / "parent.frcmod"
    _write_simple_parent(parent_frcmod)

    fitted = tmp_path / "fragment_1"
    fitted.mkdir()
    (fitted / "fit_torsions.json").write_text("[]")
    (fitted / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "ca-ce-nf-cd    1     3.33518856    0.000   2.0    SCEE=1.2 SCNB=2.0",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    # No iteration output at all: the fit never needed to run.
    empty = tmp_path / "fragment_2"
    empty.mkdir()
    (empty / "fit_torsions.json").write_text("[]")

    output_frcmod = tmp_path / "merged.frcmod"
    with pytest.warns(MergeWarning, match="fragment_2"):
        report = merge_fragment_frcmods(
            parent_frcmod_path=parent_frcmod,
            output_frcmod_path=output_frcmod,
            fragment_dirs=[fitted, empty],
        )

    merged_text = output_frcmod.read_text()
    assert "3.335" in merged_text
    assert report["replacement_counts"] == {"replaced": 1, "added": 0}
    assert report["skipped_fragments"] == [
        {"fragment_dir": str(empty), "reason": "no itXX.frcmod in the fragment directory"}
    ]


def test_merge_fragment_frcmods_skips_iterations_without_dihe_terms(tmp_path: Path):
    """An iteration frcmod with an empty DIHE section warns and is skipped.

    Args:
        tmp_path: Temporary pytest directory for synthetic merge inputs.
    """

    parent_frcmod = tmp_path / "parent.frcmod"
    _write_simple_parent(parent_frcmod)

    flat = tmp_path / "fragment_1"
    flat.mkdir()
    (flat / "fit_torsions.json").write_text("[]")
    (flat / "it01.frcmod").write_text(
        "\n".join(
            [
                "Created by ParmEd",
                "MASS",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "",
                "IMPROPER",
                "",
                "NONB",
                "",
            ]
        )
        + "\n"
    )

    output_frcmod = tmp_path / "merged.frcmod"
    with pytest.warns(MergeWarning) as records:
        report = merge_fragment_frcmods(
            parent_frcmod_path=parent_frcmod,
            output_frcmod_path=output_frcmod,
            fragment_dirs=[flat],
        )

    messages = [str(record.message) for record in records]
    assert any("defines no fitted DIHE terms" in message for message in messages)
    assert any("copy of the parent frcmod" in message for message in messages)
    assert report["replacement_counts"] == {"replaced": 0, "added": 0}
    assert report["skipped_fragments"][0]["reason"] == (
        "it01.frcmod defines no fitted DIHE terms"
    )
    # The parent's own terms survive untouched.
    assert (
        "ca-ce-nf-cd   2    1.600       180.000           2.000      parent"
        in output_frcmod.read_text()
    )


def load_minimal_fixture():
    """Load the sample ligand from its MOL2 alone (no lib/frcmod).

    Returns:
        A :class:`Ligand` suitable for RDKit mol construction and central-bond
        SMARTS matching.
    """

    return load_ligand_from_mol2(FIXTURE_DIR / "ligand.mol2")


def test_load_ligand_from_mol2_minimal_matches_parse_mol2():
    """The mol2-only loader should mirror parse_mol2 and fill lib placeholders."""

    from scission.io import parse_mol2

    _, atoms, bonds = parse_mol2(FIXTURE_DIR / "ligand.mol2")
    ligand = load_minimal_fixture()
    assert len(ligand.atoms) == len(atoms) == 8
    assert len(ligand.bonds) == len(bonds) == 7
    assert ligand.lib_atom_names == [atom.name for atom in atoms]
    assert ligand.lib_atom_types == {atom.name: atom.atom_type for atom in atoms}
    assert ligand.frcmod_text == ""


def test_generate_bond_smarts_radius0_is_minimal_and_mapped():
    """Radius-0 SMARTS should keep only the two :1/:2 mapped, bonded atoms."""

    pytest.importorskip("rdkit")
    from rdkit import Chem

    from scission.pickbond import generate_bond_smarts
    from scission.torsions import _build_rdkit_mol

    mol = _build_rdkit_mol(load_minimal_fixture())
    # RDKit indices 4,5 are C5-C6 (a single bond in the fixture).
    smarts = generate_bond_smarts(mol, 4, 5, 0)
    query = Chem.MolFromSmarts(smarts)
    assert query is not None
    mapped = {a.GetAtomMapNum() for a in query.GetAtoms() if a.GetAtomMapNum()}
    assert mapped == {1, 2}
    assert query.GetNumAtoms() == 2


def test_generate_bond_smarts_passes_library_matcher_roundtrip():
    """A generated SMARTS should match the clicked bond via the real matcher."""

    pytest.importorskip("rdkit")
    from scission.pickbond import generate_bond_smarts
    from scission.torsions import _build_rdkit_mol

    ligand = load_minimal_fixture()
    mol = _build_rdkit_mol(ligand)
    # RDKit indices 4,5 -> one-based parent atoms 5,6 (C5-C6).
    smarts = generate_bond_smarts(mol, 4, 5, 1)
    matched = match_central_bond_smarts(ligand, (smarts,))
    assert (5, 6) in matched


def test_generate_bond_smarts_radius_does_not_widen_match_count():
    """Increasing the radius should never increase the number of matches."""

    pytest.importorskip("rdkit")
    from scission.pickbond import generate_bond_smarts
    from scission.torsions import _build_rdkit_mol

    ligand = load_minimal_fixture()
    mol = _build_rdkit_mol(ligand)

    def count(radius: int) -> int:
        smarts = generate_bond_smarts(mol, 4, 5, radius)
        return len(match_central_bond_smarts(ligand, (smarts,)))

    broad = count(0)
    narrow = count(2)
    # Radius 0 is a generic C-C bond (several matches); radius 2 is far tighter.
    assert broad > 1
    assert narrow <= broad


def _plan_one_cap(elements, strategy="chemistry_aware", retained_net=0.0, h_min=0.0, sites=None, steric_rank_of=None, force_matched_of=None):
    """Resolve caps for a tiny synthetic cut-site set via ``plan_caps``.

    Args:
        elements: Mapping from parent atom index to element symbol.
        strategy: Cap strategy to exercise.
        retained_net: Net charge of the retained atoms.
        h_min: Minimum allowed bare-hydrogen charge.
        sites: Optional explicit cap-site list (defaults to a single ``1-2`` cut).
        steric_rank_of: Optional steric-rank callable forwarded to ``plan_caps``.

    Returns:
        The ``(resolved_caps, fragment_net_charge)`` tuple from ``plan_caps``.
    """

    if sites is None:
        sites = [CapSite(retained_atom=1, removed_atom=2, bond_type="1")]
    return plan_caps(
        sites,
        element_of=elements.get,
        position_of=lambda idx: (0.0, 0.0, 0.0),
        direction_of=lambda retained, removed: np.array([1.0, 0.0, 0.0]),
        strategy=strategy,
        retained_net_charge=retained_net,
        existing_names=set(),
        h_min_charge=h_min,
        steric_rank_of=steric_rank_of,
        force_matched_of=force_matched_of,
    )


def test_chemistry_aware_caps_carbon_carbon_with_bare_hydrogen():
    """A neutral C-C cut should be capped with a bare hydrogen, not -OH."""

    caps, net = _plan_one_cap({1: "C", 2: "C"})
    assert len(caps) == 1
    cap = caps[0]
    assert cap.heavy is None
    assert len(cap.hydrogens) == 1
    assert cap.hydrogens[0].atom_type == "hc"
    assert cap.hydrogens[0].charge == pytest.approx(0.0)
    assert net == pytest.approx(0.0)


def test_chemistry_aware_caps_carbon_oxygen_with_hydroxyl():
    """A severed C-O bond should be recreated as C-OH."""

    caps, _ = _plan_one_cap({1: "C", 2: "O"})
    cap = caps[0]
    assert cap.heavy is not None
    assert cap.heavy.element == "O"
    assert cap.heavy.atom_type == "oh"
    assert len(cap.hydrogens) == 1
    assert cap.hydrogens[0].atom_type == "ho"


def test_chemistry_aware_caps_carbon_nitrogen_with_amine():
    """A severed C-N single bond should be recreated as C-NH2."""

    caps, _ = _plan_one_cap({1: "C", 2: "N"})
    cap = caps[0]
    assert cap.heavy is not None
    assert cap.heavy.element == "N"
    assert cap.heavy.atom_type == "n3"
    assert len(cap.hydrogens) == 2
    assert all(hydrogen.atom_type == "hn" for hydrogen in cap.hydrogens)


def test_chemistry_aware_double_bond_uses_matched_order_and_fewer_hydrogens():
    """A severed C=N bond should become C=NH with sp2 typing."""

    sites = [CapSite(retained_atom=1, removed_atom=2, bond_type="2")]
    caps, _ = _plan_one_cap({1: "C", 2: "N"}, sites=sites)
    cap = caps[0]
    assert cap.parent_bond_order == 2
    assert cap.heavy.atom_type == "n2"
    assert len(cap.hydrogens) == 1


def test_chemistry_aware_escalates_carbon_cap_when_hydrogen_would_be_negative():
    """A C-C cut whose bare H would go negative escalates to a CH3 cap."""

    caps, net = _plan_one_cap({1: "C", 2: "C"}, retained_net=0.3)
    cap = caps[0]
    # The bare hydrogen would have to carry the -0.3 correction, so it is
    # promoted to a methyl cap instead.
    assert cap.heavy is not None
    assert cap.heavy.element == "C"
    assert cap.heavy.atom_type == "c3"
    assert len(cap.hydrogens) == 3
    assert all(hydrogen.charge >= 0.0 for hydrogen in cap.hydrogens)
    assert net == pytest.approx(0.0)


def test_chemistry_aware_preserves_integer_charge_with_mixed_caps():
    """Bare-H and matched caps together still reach an integer net charge."""

    sites = [
        CapSite(retained_atom=1, removed_atom=2, bond_type="1"),
        CapSite(retained_atom=3, removed_atom=4, bond_type="1"),
    ]
    caps, net = _plan_one_cap(
        {1: "C", 2: "C", 3: "C", 4: "O"},
        retained_net=0.2,
        sites=sites,
    )
    assert net == pytest.approx(0.0)
    cap_hydrogens = [atom for cap in caps for atom in cap.atoms if atom.element == "H"]
    assert all(hydrogen.charge >= 0.0 for hydrogen in cap_hydrogens)


def test_hydroxyl_strategy_reproduces_legacy_oh_cap():
    """The legacy strategy keeps the OX01/HX01 oh/ho pair and base charges."""

    caps, _ = _plan_one_cap({1: "C", 2: "C"}, strategy="hydroxyl")
    cap = caps[0]
    assert cap.heavy.element == "O" and cap.heavy.atom_type == "oh"
    assert cap.heavy.name == "OX01"
    assert cap.hydrogens[0].atom_type == "ho" and cap.hydrogens[0].name == "HX01"
    assert cap.heavy.charge == pytest.approx(-0.54)
    assert cap.hydrogens[0].charge == pytest.approx(0.54)


def test_hydrogen_strategy_always_uses_a_bare_hydrogen():
    """The hydrogen strategy caps even a severed heteroatom with a bare H."""

    caps, _ = _plan_one_cap({1: "C", 2: "O"}, strategy="hydrogen")
    cap = caps[0]
    assert cap.heavy is None
    assert len(cap.hydrogens) == 1


def test_cut_bonds_for_retained_emits_one_cap_per_cut_neighbor():
    """A retained atom with several cut neighbors yields several cap sites."""

    graph = nx.Graph()
    for idx in (1, 2, 3, 4):
        graph.add_node(
            idx,
            atom=Atom(index=idx, name=f"C{idx}", element="C", atom_type="c3", charge=0.0, coords=(0.0, 0.0, 0.0)),
        )
    for nbr in (2, 3, 4):
        graph.add_edge(1, nbr, bond_type="1")

    cut_bonds, cap_sites = _cut_bonds_for_retained(graph, {1}, set())
    assert len(cut_bonds) == 3
    assert len(cap_sites) == 3
    assert all(site.retained_atom == 1 for site in cap_sites)
    assert {site.removed_atom for site in cap_sites} == {2, 3, 4}


def test_pipeline_chemistry_aware_keeps_integer_charge_and_nonnegative_cap_hydrogens(tmp_path: Path):
    """End-to-end default capping should never emit a negative cap hydrogen."""

    result = fragment_ligand(
        InputBundle(
            mol2_path=FIXTURE_DIR / "ligand.mol2",
            lib_path=FIXTURE_DIR / "ligand.lib",
            frcmod_path=FIXTURE_DIR / "ligand.frcmod",
        ),
        tmp_path,
        FragmentConfig(),
    )
    assert result.selected_fragments
    for fragment in result.selected_fragments:
        manifest = json.loads(fragment.manifest_path.read_text())
        assert abs(manifest["net_charge"] - round(manifest["net_charge"])) < 1.0e-6
        for cap in manifest["cap_atoms"]:
            if cap["element"] == "H":
                assert cap["charge"] >= -1.0e-9


def test_chemistry_aware_bare_h_keeps_realistic_charge_when_heteroatom_absorbs():
    """A carbon cut next to a heteroatom cap stays a normal H; the heteroatom sinks the residual."""

    sites = [
        CapSite(retained_atom=1, removed_atom=2, bond_type="1"),  # C-C (flexible)
        CapSite(retained_atom=3, removed_atom=4, bond_type="1"),  # C-N (heteroatom)
    ]
    caps, net = _plan_one_cap({1: "C", 2: "C", 3: "C", 4: "N"}, retained_net=0.0055, sites=sites)
    by_ret = {cap.retained_atom: cap for cap in caps}

    # The C-C cut stays a bare H at its realistic base charge, not a methyl.
    cc = by_ret[1]
    assert cc.heavy is None
    assert cc.reason == "bare_hydrogen"
    assert cc.hydrogens[0].atom_type == "hc"
    assert cc.hydrogens[0].charge == pytest.approx(0.06, abs=1e-6)

    # The NH2 absorbs the residual, so its group is NOT forced neutral.
    nh2 = by_ret[3]
    group_charge = nh2.heavy.charge + sum(h.charge for h in nh2.hydrogens)
    assert abs(group_charge) > 1e-3
    assert net == pytest.approx(0.0)


def test_chemistry_aware_puts_methyl_sink_on_sterically_roomy_side():
    """When a methyl sink is needed, the roomier carbon cut becomes it and the tight one stays H."""

    sites = [
        CapSite(retained_atom=1, removed_atom=2, bond_type="1"),
        CapSite(retained_atom=3, removed_atom=4, bond_type="1"),
    ]
    # Site at retained-atom 3 is roomier (higher rank); atom 1 is crowded.
    rank = {1: 0.1, 3: 5.0}
    caps, net = _plan_one_cap(
        {1: "C", 2: "C", 3: "C", 4: "C"},
        retained_net=0.2,  # forces one methyl sink
        sites=sites,
        steric_rank_of=lambda site: rank[site.retained_atom],
    )
    by_ret = {cap.retained_atom: cap for cap in caps}
    assert by_ret[3].heavy is not None and by_ret[3].cap_label == "CH3"  # roomy -> methyl
    assert by_ret[1].heavy is None  # crowded -> bare H
    assert by_ret[3].reason == "charge_escalation"
    assert net == pytest.approx(0.0)


def test_from_dict_parses_torsion_neighborhood_options():
    """from_dict should accept the cap neighborhood-preservation knobs."""

    default = FragmentConfig.from_dict({})
    assert default.preserve_torsion_neighborhood is True
    assert default.torsion_neighborhood_radius == 1
    assert default.preserve_conjugated_caps is True

    custom = FragmentConfig.from_dict(
        {
            "preserve_torsion_neighborhood": False,
            "torsion_neighborhood_radius": 2,
            "preserve_conjugated_caps": False,
        }
    )
    assert custom.preserve_torsion_neighborhood is False
    assert custom.torsion_neighborhood_radius == 2
    assert custom.preserve_conjugated_caps is False


def test_force_matched_keeps_substituent_near_fitted_torsion():
    """A carbon cut next to the fitted bond is forced to a matched cap, not a bare H."""

    # A lone C-C cut that would otherwise be a bare hydrogen...
    bare, _ = _plan_one_cap({1: "C", 2: "C"})
    assert bare[0].heavy is None

    # ...is forced to a methyl when the site is flagged as near the fitted torsion.
    forced, net = _plan_one_cap(
        {1: "C", 2: "C"},
        force_matched_of=lambda site: "near_fitted_torsion",
    )
    assert forced[0].heavy is not None
    assert forced[0].cap_label == "CH3"
    assert forced[0].reason == "near_fitted_torsion"
    assert net == pytest.approx(0.0)


def test_cap_decisions_record_reason_for_each_outcome():
    """Each cap should carry a reason code and human-readable label."""

    bare, _ = _plan_one_cap({1: "C", 2: "C"})
    assert bare[0].reason == "bare_hydrogen"
    assert bare[0].cap_label == "bare_hydrogen"

    matched, _ = _plan_one_cap({1: "C", 2: "O"})
    assert matched[0].reason == "heteroatom_severed"
    assert matched[0].cap_label == "OH"
    assert matched[0].bare_h_charge is None

    escalated, _ = _plan_one_cap({1: "C", 2: "C"}, retained_net=0.3)
    assert escalated[0].reason == "charge_escalation"
    assert escalated[0].cap_label == "CH3"
    # The hypothetical bare-H charge that triggered escalation is recorded.
    assert escalated[0].bare_h_charge is not None
    assert escalated[0].bare_h_charge < 0.0


def test_pipeline_writes_cap_decisions_and_summary_rollup(tmp_path: Path):
    """Manifests should record per-cut decisions and the summary should tally them."""

    result = fragment_ligand(
        InputBundle(
            mol2_path=EXAMPLE_DIR / "binder_jmc2025-1.mol2",
            lib_path=EXAMPLE_DIR / "binder_jmc2025-1.lib",
            frcmod_path=EXAMPLE_DIR / "binder_jmc2025-1.frcmod",
        ),
        tmp_path,
        FragmentConfig(),
    )
    valid_reasons = {
        "bare_hydrogen",
        "charge_escalation",
        "heteroatom_severed",
        "retained_not_carbon",
        "near_fitted_torsion",
        "conjugated_center",
        "forced_hydrogen_strategy",
        "legacy_hydroxyl_strategy",
    }
    total_from_manifests = 0
    for fragment in result.selected_fragments:
        manifest = json.loads(fragment.manifest_path.read_text())
        assert "cap_decisions" in manifest
        for decision in manifest["cap_decisions"]:
            assert decision["reason"] in valid_reasons
            assert set(decision) >= {"parent_atom", "removed_atom", "bond_order", "cap", "reason"}
            total_from_manifests += 1

    summary = json.loads(result.summary_path.read_text())
    assert "cap_decision_counts" in summary
    assert sum(summary["cap_decision_counts"].values()) == total_from_manifests


def test_generate_bond_smarts_rejects_non_bonded_pair():
    """Selecting two atoms that are not directly bonded should raise clearly."""

    pytest.importorskip("rdkit")
    from scission.pickbond import generate_bond_smarts
    from scission.torsions import _build_rdkit_mol

    mol = _build_rdkit_mol(load_minimal_fixture())
    # RDKit indices 0 and 6 are C1 and O1 -- not directly bonded.
    with pytest.raises(ValueError, match="not directly bonded"):
        generate_bond_smarts(mol, 0, 6, 1)
