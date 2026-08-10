import rdkit
import re
import shutil
import MDAnalysis as mda
import numpy as np

from rdkit.Chem import rdFMCS

class PDBFromSMILES:
    """Generate a PDB file from a SMILES string.

    Parameters
    ----------
    resname : str
        Residue name of the molecule.
    smiles : str
        SMILES string of the molecule.

    Attributes
    ----------
    resname : str
        Residue name of the molecule.
    smiles : str
        SMILES string of the molecule.
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    pdb_filename : str
        Path to the written PDB file.
    """
    def __init__(self, resname, smiles):
        self.resname = resname
        self.smiles = smiles
        self.mol = None
        return

    
    def write_pdb(self, filename, randomSeed=0xf00d, *, minimize: bool = False):
        """Embed the molecule and write a cleaned PDB file.

        Parameters
        ----------
        filename : str
            Output PDB path.
        randomSeed : int, optional
            Random seed for the embedding algorithm.
        minimize : bool, optional
            If True, run MMFF (or UFF) optimization after embedding.
        """
        params = rdkit.Chem.AllChem.ETKDGv3()
        params.randomSeed = randomSeed
        rdkit.Chem.AllChem.EmbedMolecule(self.mol, params)
        if minimize:
            if rdkit.Chem.AllChem.MMFFHasAllMoleculeParams(self.mol):
                rdkit.Chem.AllChem.MMFFOptimizeMolecule(self.mol)
            else:
                rdkit.Chem.AllChem.UFFOptimizeMolecule(self.mol)
        rdkit.Chem.rdmolfiles.MolToPDBFile(self.mol, filename)
        self.pdb_filename = filename
        clean_pdb(self.pdb_filename, self.resname)
        return
    
    def mol_from_smiles(self, addHs=True):
        """Build an RDKit molecule from the stored SMILES string.

        Parameters
        ----------
        addHs : bool, optional
            Whether to add hydrogens to the molecule.
        """
        mol = rdkit.Chem.MolFromSmiles(self.smiles)
        if addHs:
            mol = rdkit.Chem.rdmolops.AddHs(mol)
        self.mol = mol
        return 
    
class MolFromPDB:
    """Load a PDB file into both RDKit and MDAnalysis representations."""

    def __init__(self, pdb_filename, removeHs=False):
        self.remove_Hs = removeHs
        self.pdb_filename = pdb_filename
        self._rdkit_representation()
        self._mda_representation()
        return

    
    def _rdkit_representation(self):
        """Generate an RDKit molecule from the PDB file."""
        self.rdkit_mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(self.pdb_filename, removeHs=self.remove_Hs)
        return
    
    def _mda_representation(self):
        """Generate an MDAnalysis Universe from the PDB file."""
        self.mda_universe = mda.Universe(self.pdb_filename)
        return
    
    def resname(self):
        """Return the residue name from the PDB file."""
        return self.mda_universe.atoms.residues.resnames[0]
    
    def names(self):
        """Return atom names from the MDAnalysis Universe."""
        return self.mda_universe.atoms.names
    
    def elements(self):
        """Return element symbols from the MDAnalysis Universe."""
        return self.mda_universe.atoms.elements
    
    def write_pdb(self, filename):
        """Write the Universe to a PDB file and clean residue names.

        Parameters
        ----------
        filename : str
            Output PDB path.
        """
        self.mda_universe.atoms.write(filename)
        clean_pdb(filename, self.resname())
        return
    
    
class RenamePDBTypes:
    """Rename atom types in a PDB file using a reference structure.

    Parameters
    ----------
    primary_pdb : str
        Path to the primary PDB file to rename.
    resname : str
        Residue name of the molecule.

    Attributes
    ----------
    primary_pdb : str
        Path to the primary PDB file.
    resname : str
        Residue name of the molecule.
    mols : list
        List of :class:`MolFromPDB` objects.
    mcs_mol : rdkit.Chem.Mol
        RDKit molecule of the common substructure.
    """
    def __init__(self, primary_pdb, resname):
        self.primary_pdb = primary_pdb
        self.mols = []
        self.mols.append(MolFromPDB(primary_pdb, removeHs=False))
        self.resname = resname
        return
    
    def add_mol(self, mol_pdb):
        """Add a reference molecule from a PDB path."""
        self.mols.append(MolFromPDB(mol_pdb))
        return
    
    def rename_by_reference(self):
        """Rename atoms in the primary PDB to match the reference molecule."""
        if len(self.mols) != 2:
            raise ValueError("ERROR: Only two molecules can be compared for reference.")
        st_comm, rf_comm = self.common_atoms()
        codes = [f"C{i}" for i in range(1, 41)]
        available_names = set(self.mols[0].names()) | set(self.mols[1].names()) | set(codes)
        new_names = np.zeros_like(self.mols[0].names())
        print(self.mols[0].names()[st_comm])
        print(self.mols[1].names()[rf_comm])
        for i, j in zip(st_comm, rf_comm):
            new_names[i] = self.mols[1].names()[j]
            available_names.remove(self.mols[1].names()[j])
        for atom in self.mols[0].mda_universe.atoms:
            if atom.index not in st_comm:
                renamed=False
                for key in available_names:
                    cleaned_key = self._split_letters_numbers(key)
                    if self.mols[0].elements()[atom.index] == cleaned_key[0]:
                        new_names[atom.index] = key
                        available_names.remove(key)
                        renamed=True
                        break
                if not renamed:
                    raise ValueError("ERROR: Could not rename atom. ")
        self.mols[0].mda_universe.atoms.names = new_names
        shutil.copyfile(self.mols[0].pdb_filename, 'original_'+self.mols[0].pdb_filename)
        self.mols[0].write_pdb(f"{self.mols[0].resname()}.pdb")
                    
        return
    
    def find_mcs(self):
        """Find the maximum common substructure between loaded molecules."""
        mcs = rdFMCS.FindMCS([mol.rdkit_mol for mol in self.mols])
        self.mcs_mol = rdkit.Chem.rdmolfiles.MolFromSmarts(mcs.smartsString)
        return
    
    def common_atoms(self):
        """Return atom-index arrays for the common substructure in each molecule.

        Returns
        -------
        list of np.ndarray
            Matching atom indices per molecule.
        """
        self.find_mcs()
        return [np.array(mol.rdkit_mol.GetSubstructMatch(self.mcs_mol)) for mol in self.mols]

    def _split_letters_numbers(self, s):
        """Split a string into leading letters and trailing digits.

        Parameters
        ----------
        s : str
            Atom-name-like string to split.

        Returns
        -------
        tuple or None
            ``(letters, numbers)`` if matched, else None.
        """
        match = re.match(r"([a-zA-Z]+)([0-9]+)", s)
        if match:
            return match.groups()
        else:
            return None


def clean_pdb(pdb_filename, resname):
    """Replace UNL residue names and strip SYST tags from a PDB file.

    Parameters
    ----------
    pdb_filename : str
        PDB file to clean in place.
    resname : str
        Three-character residue name to substitute for ``UNL``.
    """
    if len(resname) != 3:
        raise ValueError("Resname must be 3 characters")
    lines = []
    with open(pdb_filename, "r") as f:
        lines = f.readlines()
    with open(pdb_filename, "w") as f:
        for line in lines:
            line = line.replace("UNL", resname)
            line = line.replace("SYST", "    ")
            if line.startswith("ATOM") or line.startswith("HETATM"):
                f.write(line)

    return
    
if __name__ == "__main__":
    # Create the PDBFromSMILES object
    pdb = PDBFromSMILES("F3G", "O=C1NC(C(F)(F)F)=NC2=C1N=CN2")
    
    # Generate the molecule
    mol = pdb.mol_from_smiles()
    
    # Write the PDB file
    pdb.write_pdb(mol, "test.pdb")

    new = RenamePDBTypes("test.pdb", "F3G")
    new.add_mol("align.pdb")
    new.rename_by_reference()
