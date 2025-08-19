import warnings
from typing import Optional, Union

from pathlib import Path
from openff.toolkit import Molecule, ForceField, Topology, Quantity
from openff.interchange import Interchange
from openff.units import unit
from rdkit import Chem
from rdkit.Chem import AllChem

from ligandparam.stages import AbstractStage 


class StageSageCreate(AbstractStage):
    """ Converts the final resulting mol2 to the SAGE forcefield.

    Parameters
    ----------
    stage_name: str
        The name of the stage.
    main_input: Union[Path, str]
        The main input file (mol2 format).
    cwd: Union[Path, str]
        The current working directory.
    out_parm: str
        The output parm file (parm7 format).

    Attributes
    ----------
    in_mol2: Path
        The input mol2 file.
    out_parm: Path
        The output parm file.
    out_rst7: Path
        The output rst7 file.
    ff_name: str
        The name of the force field.
    
    Notes
    -----
    This stage is responsible for converting the final resulting mol2 file to the SAGE forcefield format.

    """
    def __init__(self, stage_name: str, main_input: Union[Path, str], cwd: Union[Path, str], *args, **kwargs) -> None:
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_mol2 = Path(main_input)
        print(self.in_mol2)
        self.out_parm = Path(kwargs["out_parm"])
        self.out_rst7 = Path(kwargs["out_parm"].replace(".parm7", ".rst7"))
        if ".parm7" not in self.out_parm.name:
            raise ValueError("Output parameter file must have .parm7 extension")
        self.ff_name = kwargs.get("ff_name", "openff-2.2.0.offxml")
        self.add_required(self.in_mol2)

    def _append_stage(self, stage: AbstractStage) -> "AbstractStage":
        """
        Append a stage to the current stage.

        Parameters
        ----------
        stage : AbstractStage
            The stage to append.

        Returns
        -------
        AbstractStage
            The appended stage.
        """
        return stage
    
    def execute(self, dry_run=False, nproc: Optional[int]=None, mem: Optional[int]=None) -> None:
        mol = Chem.MolFromMol2File(str(self.in_mol2), removeHs=False)
        molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        topology = Topology.from_molecules(molecule)
        ff = ForceField(self.ff_name)
        interchange = Interchange.from_smirnoff(
            force_field = ff,
            topology = topology
        )
        interchange.to_prmtop(f"{self.out_parm}")
        interchange.to_inpcrd(f"{self.out_rst7}")

        return 
    
    def _clean(self):
        """Clean up temporary files created during the stage."""
        raise NotImplementedError("clean method not implemented.")
