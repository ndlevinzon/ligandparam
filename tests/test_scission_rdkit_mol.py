"""Tests for scission RDKit molecule construction / formal charges."""

from scission.models import Atom, Bond
from scission.rdkit_mol import build_rdkit_mol, infer_formal_charge


def _atom(index: int, name: str, element: str, atom_type: str, charge: float = 0.0) -> Atom:
    return Atom(
        index=index,
        name=name,
        element=element,
        atom_type=atom_type,
        charge=charge,
        coords=(float(index), 0.0, 0.0),
    )


def test_infer_quaternary_nitrogen_formal_charge():
    assert infer_formal_charge("N", 4, atom_type="n4", partial_charge=0.7) == 1
    assert infer_formal_charge("N", 4, atom_type="c3", partial_charge=0.2) == 1


def test_build_rdkit_mol_quaternary_nitrogen_sanitizes():
    """Tetramethylammonium-like N (GAFF n4) must sanitize as N+."""
    atoms = [
        _atom(1, "N1", "N", "n4", 0.8),
        _atom(2, "C1", "C", "c3", -0.2),
        _atom(3, "C2", "C", "c3", -0.2),
        _atom(4, "C3", "C", "c3", -0.2),
        _atom(5, "C4", "C", "c3", -0.2),
    ]
    bonds = [
        Bond(1, 1, 2, "1"),
        Bond(2, 1, 3, "1"),
        Bond(3, 1, 4, "1"),
        Bond(4, 1, 5, "1"),
    ]
    mol = build_rdkit_mol(atoms, bonds)
    n_atom = mol.GetAtomWithIdx(0)
    assert n_atom.GetSymbol() == "N"
    assert n_atom.GetFormalCharge() == 1
    assert n_atom.GetDegree() == 4
