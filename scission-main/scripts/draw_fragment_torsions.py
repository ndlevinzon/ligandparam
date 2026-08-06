from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from scission.io import parse_mol2


def _safe_name(text: str) -> str:
    """Normalize a label so it can be used safely as part of a filename.

    Args:
        text: Source label to sanitize.

    Returns:
        A filename-safe version of ``text``.
    """

    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _atom_name_map(mol: Chem.Mol) -> dict[str, int]:
    """Map Tripos atom names to RDKit atom indices.

    Args:
        mol: RDKit molecule containing ``_TriposAtomName`` properties.

    Returns:
        Mapping from atom name to RDKit atom index.
    """

    mapping: dict[str, int] = {}
    for atom in mol.GetAtoms():
        name = atom.GetProp("_TriposAtomName") if atom.HasProp("_TriposAtomName") else atom.GetSymbol()
        mapping[name] = atom.GetIdx()
    return mapping


def _rdkit_bond_type(bond_type: str) -> Chem.BondType:
    """Translate stored bond labels into RDKit bond types.

    Args:
        bond_type: Bond label stored in the source topology.

    Returns:
        The corresponding RDKit bond type.
    """

    normalized = bond_type.lower()
    if normalized in {"1", "1.0", "single", "am"}:
        return Chem.BondType.SINGLE
    if normalized in {"2", "2.0"}:
        return Chem.BondType.DOUBLE
    if normalized in {"3", "3.0"}:
        return Chem.BondType.TRIPLE
    if normalized == "ar":
        return Chem.BondType.AROMATIC
    return Chem.BondType.SINGLE


def _load_fragment_mol(mol2_path: Path) -> Chem.Mol:
    """Load a fragment MOL2 file into RDKit while preserving atom names.

    Args:
        mol2_path: Path to the fragment MOL2 file.

    Returns:
        The reconstructed RDKit molecule.
    """

    _, atoms, bonds = parse_mol2(mol2_path)
    editable = Chem.RWMol()
    for atom in atoms:
        rd_atom = Chem.Atom(atom.element)
        rd_atom.SetFormalCharge(0)
        rd_idx = editable.AddAtom(rd_atom)
        editable.GetAtomWithIdx(rd_idx).SetProp("_TriposAtomName", atom.name)

    aromatic_atoms: set[int] = set()
    for bond in bonds:
        atom1 = bond.atom1 - 1
        atom2 = bond.atom2 - 1
        editable.AddBond(atom1, atom2, _rdkit_bond_type(bond.bond_type))
        if bond.bond_type.lower() == "ar":
            aromatic_atoms.add(atom1)
            aromatic_atoms.add(atom2)

    mol = editable.GetMol()
    for atom_idx in aromatic_atoms:
        mol.GetAtomWithIdx(atom_idx).SetIsAromatic(True)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(True)
    Chem.SanitizeMol(mol)
    return mol


def _highlight_for_torsion(mol: Chem.Mol, torsion_label: str) -> tuple[list[int], list[int]]:
    """Locate the atoms and bonds that define a named torsion.

    Args:
        mol: RDKit molecule to inspect.
        torsion_label: Torsion label formatted as joined atom names.

    Returns:
        Atom indices and bond indices to highlight in the drawing.
    """

    atom_names = torsion_label.split("-")
    name_to_idx = _atom_name_map(mol)
    atom_ids = [name_to_idx[name] for name in atom_names if name in name_to_idx]
    bond_ids: list[int] = []
    for left, right in zip(atom_names, atom_names[1:]):
        if left not in name_to_idx or right not in name_to_idx:
            continue
        bond = mol.GetBondBetweenAtoms(name_to_idx[left], name_to_idx[right])
        if bond is not None:
            bond_ids.append(bond.GetIdx())
    return atom_ids, bond_ids


def draw_fragments(summary_path: Path, output_dir: Path) -> None:
    """Render per-torsion SVG depictions from a summary JSON file.

    Args:
        summary_path: Path to a workflow summary JSON file.
        output_dir: Directory where drawings and the README index should go.
    """

    data = json.loads(summary_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    index_lines = ["# Fragment Torsion Drawings", ""]
    for fragment in data["selected_fragments"]:
        mol2_path = Path(fragment["mol2_path"])
        mol = _load_fragment_mol(mol2_path)
        AllChem.Compute2DCoords(mol)

        fragment_tag = _safe_name(fragment["fragment_id"])[:80]
        index_lines.append(f"## {fragment['fragment_id']}")
        index_lines.append("")
        for torsion in fragment["torsions"]:
            atom_ids, bond_ids = _highlight_for_torsion(mol, torsion)
            drawer = Draw.MolDraw2DSVG(900, 600)
            options = drawer.drawOptions()
            options.addAtomIndices = False
            legend = f"{torsion} | charge={fragment['net_charge']:.3f}"
            drawer.DrawMolecule(
                mol,
                highlightAtoms=atom_ids,
                highlightBonds=bond_ids,
                legend=legend,
            )
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            out_name = f"{fragment_tag}__{_safe_name(torsion)}.svg"
            out_path = output_dir / out_name
            out_path.write_text(svg)
            index_lines.append(f"- [{torsion}]({out_name})")
        index_lines.append("")

    (output_dir / "README.md").write_text("\n".join(index_lines) + "\n")


def main(argv: list[str]) -> int:
    """Run the standalone torsion-drawing script.

    Args:
        argv: Raw command-line argument vector.

    Returns:
        Process exit code suitable for ``SystemExit``.
    """

    if len(argv) != 3:
        print("usage: draw_fragment_torsions.py SUMMARY_JSON OUTPUT_DIR", file=sys.stderr)
        return 2
    summary_path = Path(argv[1])
    output_dir = Path(argv[2])
    draw_fragments(summary_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
