from pathlib import Path
from typing import Optional, Union, Any

from ligandparam.stages import AbstractStage
from rdkit import Chem
from rdkit.Chem.AllChem import ETKDGv3, EmbedMolecule
from typing_extensions import override

from ligandparam.io.smiles import (
    get_available_names_per_element,
    get_element_name_and_number,
    get_mcs_mol,
    normalize_to_reference as normalize_mol_to_reference,
    pad_atom_name,
)
from ligandparam.stages import set_atom_pdb_info


class PDB_Name_Fixer(AbstractStage):
    """
    Stage for fixing PDB atom names and related metadata in ligand files.

    Parameters
    ----------
    stage_name : str
        Name of the stage.
    main_input : Union[Path, str]
        Path to the input PDB file.
    cwd : Union[Path, str]
        Current working directory.
    out_pdb : str
        Path to the output PDB file (from kwargs).
    resname : str, optional
        Residue name to use (default is 'LIG').
    reduce : bool, optional
        Whether to reduce hydrogens (default is True).
    add_conect : bool, optional
        Whether to add CONECT records (default is True).
    random_seed : int, optional
        Random seed for coordinate generation (default is None).
    reference_pdb : str, optional
        Path to reference PDB for atom name normalization (from kwargs).
    align : bool, optional
        Whether to align molecule to reference (default is False).
    """

    @override
    def __init__(self, stage_name: str, main_input: Union[Path, str], cwd: Union[Path, str], *args, **kwargs) -> None:
        """
        Initialize the PDB_Name_Fixer stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage.
        main_input : Union[Path, str]
            Path to the input PDB file.
        cwd : Union[Path, str]
            Current working directory.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments. Must include 'out_pdb'.
        """
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_pdb = main_input
        self.out_pdb = Path(kwargs["out_pdb"])
        self.resname = kwargs.get("resname", "LIG")
        self.reduce = kwargs.get("reduce", True)
        self.add_conect = kwargs.get("add_conect", True)
        self.random_seed = kwargs.get("random_seed", None)

        try:
            self.reference_pdb = Path(kwargs["reference_pdb"]).resolve()
            self.add_required(self.reference_pdb)
            self.normalize_atom_names = True
            self.align = kwargs.get("align", False)
        except KeyError:
            self.normalize_atom_names = False
            self.align = False

    def execute(self, dry_run=False, nproc: Optional[int] = None, mem: Optional[int] = None) -> Any:
        """
        Execute the stage: fix atom names, normalize to reference, and write output PDB.

        Parameters
        ----------
        dry_run : bool, optional
            If True, do not perform actual execution (default is False).
        nproc : int, optional
            Number of processors to use (default is None).
        mem : int, optional
            Memory to use in MB (default is None).

        Returns
        -------
        Any
            Output of the execution (None).
        """
        super()._setup_execution(dry_run=dry_run, nproc=nproc, mem=mem)
        # First, create the molecule
        try:
            mol = Chem.MolFromPDBFile(str(self.in_pdb), removeHs=False)
        except Exception as e:
            err_msg = f"Failed to generate an rdkit molecule from input SMILES {self.in_pdb}. Got exception: {e}"
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)

        
        #if self.reduce:
        #    mol = Chem.rdmolops.AddHs(mol)
        # All the atoms have their coordinates set to zero. Come up with some values
        #params = ETKDGv3()
        #if self.random_seed:
        #    params.randomSeed = self.random_seed
        #EmbedMolecule(mol, params)

        # Set metadata
        mol = set_atom_pdb_info(mol, self.resname)

        # Normalize the molecule to match the reference PDB
        if self.normalize_atom_names:
            mol = self.normalize_to_reference(mol, self.reference_pdb, self.align)

        flavor = 0 if self.add_conect else 2
        self.logger.info(f"Writing {self.in_pdb} to {self.out_pdb}")

        try:
            Chem.MolToPDBFile(mol, str(self.out_pdb), flavor=flavor)
        except Exception as e:
            self.logger.error(
                f"Failed to write to  {self.out_pdb}. Got exception: {e}")

    def _append_stage(self, stage: "AbstractStage") -> "AbstractStage":
        """
        Not implemented. Appends a stage to the workflow.

        Parameters
        ----------
        stage : AbstractStage
            Stage to append.

        Returns
        -------
        AbstractStage
            The appended stage.
        """
        raise NotImplementedError

    def _clean(self):
        """
        Not implemented. Cleans up after stage execution.
        """
        raise NotImplementedError

    def normalize_to_reference(self, mol: Chem.Mol, reference_pdb: Path, align: bool = False) -> Chem.Mol:
        """Normalize atom names to a reference PDB (see ``ligandparam.io.smiles``)."""
        return normalize_mol_to_reference(
            mol,
            reference_pdb,
            align=align,
            remove_hs=False,
            logger=self.logger,
        )

    pad_atom_name = staticmethod(pad_atom_name)
    get_element_name_and_number = staticmethod(get_element_name_and_number)
    get_mcs_mol = staticmethod(get_mcs_mol)

    def get_available_names_per_element(self, ref_mol: Chem.Mol, ref_match, mol: Chem.Mol) -> dict[int, list[str]]:
        return get_available_names_per_element(ref_mol, ref_match, mol)
