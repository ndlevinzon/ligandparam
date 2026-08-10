from pathlib import Path
from typing import Optional, Union, Any

from ligandparam.stages.abstractstage import AbstractStage
from rdkit import Chem
from typing_extensions import override

from ligandparam.io.smiles import (
    PDBFromSMILES,
    get_available_names_per_element,
    get_element_name_and_number,
    get_mcs_mol,
    normalize_to_reference as normalize_mol_to_reference,
    pad_atom_name,
)
from ligandparam.stages.utilsstages import set_atom_pdb_info


class StageSmilesToPDB(AbstractStage):
    """Stage for converting SMILES strings to PDB files."""

    @override
    def __init__(self, stage_name: str, main_input: Union[Path, str], cwd: Union[Path, str], *args, **kwargs) -> None:
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_smiles = main_input
        self.out_pdb = Path(kwargs["out_pdb"])
        self.resname = kwargs.get("resname", "LIG")
        self.reduce = kwargs.get("reduce", True)
        self.add_conect = kwargs.get("add_conect", True)
        self.random_seed = kwargs.get("random_seed", None)
        self.minimize = kwargs.get("minimize", False)

        try:
            self.reference_pdb = Path(kwargs["reference_pdb"]).resolve()
            self.add_required(self.reference_pdb)
            self.normalize_atom_names = True
            self.align = kwargs.get("align", False)
        except KeyError:
            self.normalize_atom_names = False
            self.align = False

    def execute(self, dry_run=False, nproc: Optional[int] = None, mem: Optional[int] = None) -> Any:
        super()._setup_execution(dry_run=dry_run, nproc=nproc, mem=mem)
        try:
            builder = PDBFromSMILES(self.resname, self.in_smiles)
            seed = self.random_seed if self.random_seed is not None else 0xF00D
            mol = builder.build_embedded_mol(
                seed,
                minimize=self.minimize,
                addHs=self.reduce,
            )
        except Exception as e:
            err_msg = (
                f"Failed to generate an rdkit molecule from input SMILES "
                f"{self.in_smiles}. Got exception: {e}"
            )
            self.logger.error(err_msg)
            raise RuntimeError(err_msg) from e

        mol = set_atom_pdb_info(mol, self.resname)

        if self.normalize_atom_names:
            mol = self.normalize_to_reference(mol, self.reference_pdb, self.align)

        flavor = 0 if self.add_conect else 2
        self.logger.info(f"Writing {self.in_smiles} to {self.out_pdb}")

        try:
            Chem.MolToPDBFile(mol, str(self.out_pdb), flavor=flavor)
        except Exception as e:
            self.logger.error(f"Failed to write to  {self.out_pdb}. Got exception: {e}")

    def _append_stage(self, stage: "AbstractStage") -> "AbstractStage":
        raise NotImplementedError

    def _clean(self):
        raise NotImplementedError

    def normalize_to_reference(self, mol: Chem.Mol, reference_pdb: Path, align: bool = False) -> Chem.Mol:
        """Normalize atom names to a reference PDB (see ``ligandparam.io.smiles``)."""
        return normalize_mol_to_reference(
            mol,
            reference_pdb,
            align=align,
            remove_hs=True,
            logger=self.logger,
        )

    pad_atom_name = staticmethod(pad_atom_name)
    get_element_name_and_number = staticmethod(get_element_name_and_number)
    get_mcs_mol = staticmethod(get_mcs_mol)

    def get_available_names_per_element(self, ref_mol: Chem.Mol, ref_match, mol: Chem.Mol) -> dict[int, list[str]]:
        return get_available_names_per_element(ref_mol, ref_match, mol)


StageSmilestoPDB = StageSmilesToPDB
