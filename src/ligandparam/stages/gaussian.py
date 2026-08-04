import os
from typing import Optional,  Union, Any
import logging
import warnings
from itertools import product

import MDAnalysis as mda

from pathlib import Path
import shutil as sh

from ligandparam.stages.abstractstage import AbstractStage
from ligandparam.io.coordinates import Coordinates, SimpleXYZ, Mol2Writer
from ligandparam.io.gaussianIO import GaussianWriter, GaussianInput, GaussianReader
from ligandparam.interfaces import Gaussian, Antechamber
from ligandparam.log import get_logger

#
logger = logging.getLogger("ligandparam.gaussian")

# Use Opt(CalcFC) below this atom count; plain Opt at or above it.
_CALCFC_MAX_ATOMS = 50


def _gaussian_opt_keyword(n_atoms: int) -> str:
    """Choose ``Opt(CalcFC)`` or ``Opt`` from ligand size.

    Opt=CalcFC computes the full force-constant matrix (Hessian) at the
    initial geometry. Plain Opt starts from an inexpensive approximate
    Hessian and updates it using gradients from later optimization steps.

    The additional cost of CalcFC is roughly one frequency calculation at
    the same method and basis set:

        T_Opt ≈ n_steps * T_gradient
        T_Opt(CalcFC) ≈ T_Hessian + n_steps' * T_gradient

    CalcFC is worthwhile only when the better starting Hessian saves enough
    optimization steps to offset T_Hessian.

    Scaling (M basis functions, N atoms):

    - The Cartesian Hessian has (3N)^2 elements → storage scales as O(N^2).
    - For HF and many DFT methods, Gaussian has analytical second
      derivatives. Formal scaling is broadly similar to the gradient, but
      with a much larger prefactor, memory, and disk footprint.
    - Without analytical seconds, a numerical Hessian may need up to ~6N
      gradient calculations and becomes prohibitive quickly.
    - Post-HF Hessians (MP2 and higher) get expensive at much smaller
      sizes than ordinary DFT Hessians.

    Policy here: ``Opt(CalcFC)`` when ``N < 50``, otherwise ``Opt``.
    """
    if n_atoms < _CALCFC_MAX_ATOMS:
        return "Opt(CalcFC)"
    return "Opt"


class GaussianMinimizeRESP(AbstractStage):
    """
    Run a basic Gaussian calculation on the ligand, including minimization and ESP calculation for RESP charges.

    Parameters
    ----------
    stage_name : str
        The name of the stage.
    main_input : Union[Path, str]
        Path to the input mol2 file.
    cwd : Union[Path, str]
        Current working directory.
    out_gaussian_log : str
        Path to the output Gaussian log file.
    opt_theory : str, optional
        Theory for optimization (default: 'PBE1PBE/6-31G*').
    resp_theory : str, optional
        Theory for RESP calculation (default: 'HF/6-31G*').
    net_charge : float, optional
        Net charge for the molecule (default: 0.0).
    force_gaussian_rerun : bool, optional
        Whether to force rerun of Gaussian (default: False).
    minimize : bool, optional
        Whether to perform minimization (default: True).

    Attributes
    ----------
    in_mol2 : Path
        Path to the input mol2 file.
    out_gaussian_log : Path
        Path to the output Gaussian log file.
    opt_theory : str
        Theory for optimization.
    resp_theory : str
        Theory for RESP calculation.
    net_charge : float
        Net charge for the molecule.
    force_gaussian_rerun : bool
        Whether to force rerun of Gaussian.
    gaussian_cwd : Path
        Directory for Gaussian calculations.
    minimize : bool
        Whether to perform minimization.
    label : str
        Label for the calculation.
    """

    def __init__(self, stage_name: str, main_input: Union[Path, str], cwd: Union[Path, str], *args, **kwargs) -> None:
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_mol2 = Path(main_input)
        self.out_gaussian_log = Path(kwargs["out_gaussian_log"])

        self._validate_input_paths(**kwargs)
        self.opt_theory = kwargs.get("opt_theory", "PBE1PBE/6-31G*")
        self.resp_theory = kwargs.get("resp_theory", "HF/6-31G*")
        self.net_charge = kwargs.get("net_charge", 0.0)
        self.force_gaussian_rerun = kwargs.get("force_gaussian_rerun", False)
        self.gaussian_cwd = Path(self.cwd, "gaussianCalcs")
        self.minimize = kwargs.get("minimize", True)

        self.label = self.out_gaussian_log.stem

        return

    def _validate_input_paths(self, **kwargs):
        """
        Validate and set input paths for Gaussian execution.

        Parameters
        ----------
        **kwargs
            Keyword arguments containing Gaussian path options.

        Raises
        ------
        ValueError
            If a required option is missing.
        """
        for opt in ("gaussian_root", "gauss_exedir", "gaussian_binary", "gaussian_scratch"):
            try:
                setattr(self, opt, kwargs.get(opt, ""))
            except KeyError:
                raise ValueError(f"ERROR: Please provide {opt} option as a keyword argument.")
        if self.gaussian_binary is None:
            self.gaussian_binary = "g16"

    def _append_stage(self, stage: "AbstractStage") -> "AbstractStage":
        """Append a stage to the current stage.

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

    def setup(self, name_template: str) -> bool:
        """
        Set up Gaussian input and output files for the calculation.

        Parameters
        ----------
        name_template : str
            Template name for input/output files.

        Returns
        -------
        bool
            True if Gaussian calculation is already complete, False otherwise.
        """
        self.in_com = self.gaussian_cwd / f"{name_template}.com"
        self.out_log = self.gaussian_cwd / f"{name_template}.log"
        self._add_outputs(self.out_log)

        # __init__ tries to set up the coordinates object, but it may not have been available at init time.
        print(f"Setting up Gaussian calculations in {self.gaussian_cwd}")
        self.logger.info(f"Setting up Gaussian calculations in {self.gaussian_cwd}")
        if not getattr(self, "coord_object", None):
            self.coord_object = Coordinates(self.in_mol2, filetype="pdb")
        self.gaussian_cwd.mkdir(exist_ok=True)

        stageheader = [f"%NPROC={self.nproc}"]
        
        stageheader.append(f"%MEM={self.mem}GB")

        stageheader.append(f"%chk={self.in_mol2.stem}.antechamber.chk")

        # Set up the Gaussian Block - it does not yet write anything,
        # so this part can be set up before the Gaussian calculations are run.
        gau = GaussianWriter(self.in_com)
        if self.minimize:
            n_atoms = len(self.coord_object.get_elements())
            opt_keyword = _gaussian_opt_keyword(n_atoms)
            self.logger.info(
                f"Gaussian optimization keyword: {opt_keyword} "
                f"({n_atoms} atoms; CalcFC if N < {_CALCFC_MAX_ATOMS})"
            )
            gau.add_block(
                GaussianInput(
                    command=f"#P {self.opt_theory} {opt_keyword}",
                    initial_coordinates=self.coord_object.get_coordinates(),
                    elements=self.coord_object.get_elements(),
                    charge=self.net_charge,
                    header=stageheader,
                )
            )
            gau.add_block(
                GaussianInput(
                    command=f"#P {self.resp_theory} GEOM(AllCheck) Guess(Read) NoSymm Pop=mk IOp(6/33=2) GFInput GFPrint",
                    charge=self.net_charge,
                    header=stageheader,
                )
            )
        else:
            gau.add_block(
                GaussianInput(
                    command=f"#P {self.resp_theory} NoSymm Pop=mk IOp(6/33=2) GFInput GFPrint",
                    initial_coordinates=self.coord_object.get_coordinates(),
                    elements=self.coord_object.get_elements(),
                    charge=self.net_charge,
                    header=stageheader,
                )
            )

        gau_complete = False
        # Check if the Gaussian calculation has already been run
        if os.path.exists(self.out_gaussian_log):
            reader = GaussianReader(self.out_gaussian_log)
            if reader.check_complete():
                self.logger.info("Gaussian calculation already complete")
                gau_complete = True

        # Check if the Gaussian calculation should be rerun
        if self.force_gaussian_rerun:
            gau_complete = False

        if not gau_complete:
            gau.write(dry_run=False)

        return gau_complete

    def execute(self, dry_run=False, nproc: Optional[int]=None, mem: Optional[int]=None) -> Any:
        """Execute the Gaussian minimization and RESP calculations.

        Parameters
        ----------
        dry_run : bool, optional
            If True, log the commands that would be run without executing them.
        nproc : int, optional
            Number of processors to use.
        mem : int, optional
            Amount of memory to use (in GB).
        """
        super()._setup_execution(dry_run=dry_run, nproc=nproc, mem=mem)
        gau_complete = self.setup(self.label)

        # Run the Gaussian calculations in the gaussianCalcs directory
        if not gau_complete:
            gau_run = Gaussian(
                cwd=self.gaussian_cwd,
                logger=self.logger,
                gaussian_root=self.gaussian_root,
                gauss_exedir=self.gauss_exedir,
                gaussian_binary=self.gaussian_binary,
                gaussian_scratch=self.gaussian_scratch,
            )
            gau_run.call(inp_pipe=self.in_com.name, out_pipe=self.out_log.name, dry_run=dry_run)

            # Move the Gaussian log file to the output location
            sh.move(self.out_log, self.out_gaussian_log)

        return

    def _clean(self):
        """Clean the files generated during the stage."""
        raise NotImplementedError("clean method not implemented")

class GaussianRESP(AbstractStage):
    """
    Run a basic Gaussian calculation on the ligand (RESP calculation only).

    Parameters
    ----------
    stage_name : str
        The name of the stage.
    main_input : Union[Path, str]
        Path to the input mol2 file.
    cwd : Union[Path, str]
        Current working directory.
    out_gaussian_log : str
        Path to the output Gaussian log file.
    resp_theory : str, optional
        Theory for RESP calculation (default: 'HF/6-31G*').
    net_charge : float, optional
        Net charge for the molecule (default: 0.0).
    force_gaussian_rerun : bool, optional
        Whether to force rerun of Gaussian (default: False).

    Attributes
    ----------
    in_mol2 : Path
        Path to the input mol2 file.
    out_gaussian_log : Path
        Path to the output Gaussian log file.
    resp_theory : str
        Theory for RESP calculation.
    net_charge : float
        Net charge for the molecule.
    force_gaussian_rerun : bool
        Whether to force rerun of Gaussian.
    gaussian_cwd : Path
        Directory for Gaussian calculations.
    label : str
        Label for the calculation.
    """

    def __init__(self, stage_name: str, main_input: Union[Path, str], cwd: Union[Path, str], *args, **kwargs) -> None:
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_mol2 = Path(main_input)
        self.out_gaussian_log = Path(kwargs["out_gaussian_log"])

        self._validate_input_paths(**kwargs)
        self.resp_theory = kwargs.get("resp_theory", "HF/6-31G*")
        self.net_charge = kwargs.get("net_charge", 0.0)
        self.force_gaussian_rerun = kwargs.get("force_gaussian_rerun", False)
        self.gaussian_cwd = Path(self.cwd, "gaussianCalcs")

        self.label = self.out_gaussian_log.stem

        return

    def _validate_input_paths(self, **kwargs):
        """
        Validate and set input paths for Gaussian execution.

        Parameters
        ----------
        **kwargs
            Keyword arguments containing Gaussian path options.

        Raises
        ------
        ValueError
            If a required option is missing.
        """
        for opt in ("gaussian_root", "gauss_exedir", "gaussian_binary", "gaussian_scratch"):
            try:
                setattr(self, opt, kwargs.get(opt, ""))
            except KeyError:
                raise ValueError(f"ERROR: Please provide {opt} option as a keyword argument.")
        if self.gaussian_binary is None:
            self.gaussian_binary = "g16"

    def _append_stage(self, stage: "AbstractStage") -> "AbstractStage":
        """Append a stage to the current stage.

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

    def setup(self, name_template: str) -> bool:
        """
        Set up Gaussian input and output files for the RESP calculation.

        Parameters
        ----------
        name_template : str
            Template name for input/output files.

        Returns
        -------
        bool
            True if Gaussian calculation is already complete, False otherwise.
        """
        self.in_com = self.gaussian_cwd / f"{name_template}.com"
        self.out_log = self.gaussian_cwd / f"{name_template}.log"
        self._add_outputs(self.out_log)
        print(f"Setting up Gaussian calculations in {self.gaussian_cwd}")
        self.logger.info(f"Setting up Gaussian calculations in {self.gaussian_cwd}")
        self.logger.info(f"Writing Gaussian input file: {self.in_com}")

        # __init__ tries to set up the coordinates object, but it may not have been available at init time.
        if not getattr(self, "coord_object", None):
            self.coord_object = Coordinates(self.in_mol2, filetype="pdb")
        self.gaussian_cwd.mkdir(exist_ok=True)

        stageheader = [f"%NPROC={self.nproc}"]
        
        stageheader.append(f"%MEM={self.mem}GB")

        stageheader.append(f"%chk={self.in_mol2.stem}.antechamber.chk")

        # Set up the Gaussian Block - it does not yet write anything,
        # so this part can be set up before the Gaussian calculations are run.
        gau = GaussianWriter(self.in_com)

        gau.add_block(
            GaussianInput(
                command=f"#P {self.resp_theory} GEOM(AllCheck) Guess(Read) NoSymm Pop=mk IOp(6/33=2) GFInput GFPrint",
                initial_coordinates=self.coord_object.get_coordinates(),
                elements=self.coord_object.get_elements(),
                charge=self.net_charge,
                header=stageheader,
            )
        )

        gau_complete = False
        # Check if the Gaussian calculation has already been run
        if os.path.exists(self.out_gaussian_log):
            reader = GaussianReader(self.out_gaussian_log)
            if reader.check_complete():
                self.logger.info("Gaussian calculation already complete")
                gau_complete = True

        # Check if the Gaussian calculation should be rerun
        if self.force_gaussian_rerun:
            gau_complete = False

        if not gau_complete:
            gau.write(dry_run=False)

        return gau_complete

    def execute(self, dry_run=False, nproc: Optional[int]=None, mem: Optional[int]=None) -> Any:
        """Execute the Gaussian RESP calculation.

        Parameters
        ----------
        dry_run : bool, optional
            If True, log the commands that would be run without executing them.
        nproc : int, optional
            Number of processors to use.
        mem : int, optional
            Amount of memory to use (in GB).
        """
        super()._setup_execution(dry_run=dry_run, nproc=nproc, mem=mem)
        gau_complete = self.setup(self.label)

        # Run the Gaussian calculations in the gaussianCalcs directory
        if not gau_complete:
            gau_run = Gaussian(
                cwd=self.gaussian_cwd,
                logger=self.logger,
                gaussian_root=self.gaussian_root,
                gauss_exedir=self.gauss_exedir,
                gaussian_binary=self.gaussian_binary,
                gaussian_scratch=self.gaussian_scratch,
            )
            gau_run.call(inp_pipe=self.in_com.name, out_pipe=self.out_log.name, dry_run=dry_run)

            # Move the Gaussian log file to the output location
            sh.move(self.out_log, self.out_gaussian_log)

        return

    def _clean(self):
        """Clean the files generated during the stage."""
        raise NotImplementedError("clean method not implemented")

class StageGaussianRotation(AbstractStage):
    """
    Rotate the ligand and run Gaussian RESP calculations for each orientation.

    Rotations use Euler angles in degrees applied as Rx then Ry then Rz about
    the center of mass, matching :meth:`Coordinates.rotate`.

    Parameters
    ----------
    stage_name : str
        The name of the stage.
    main_input : Union[Path, str]
        Path to the input mol2 file.
    cwd : Union[Path, str]
        Current working directory.
    out_gaussian_label : str
        Label for the output Gaussian files.
    alpha : list of float
        Alpha (x-axis) rotation angles in degrees.
    beta : list of float
        Beta (y-axis) rotation angles in degrees.
    gamma : list of float
        Gamma (z-axis) rotation angles in degrees.
    opt_theory : str, optional
        Theory for optimization (default: 'HF/6-31G*').
    resp_theory : str, optional
        Theory for RESP calculation (default: 'HF/6-31G*').
    net_charge : float, optional
        Net charge for the molecule (default: 0.0).
    force_gaussian_rerun : bool, optional
        Whether to force rerun of Gaussian (default: False).

    Attributes
    ----------
    in_mol2 : Path
        Path to the input mol2 file.
    out_gaussian_label : str
        Label for the output Gaussian files.
    alpha : list of float
        Alpha (x-axis) rotation angles in degrees.
    beta : list of float
        Beta (y-axis) rotation angles in degrees.
    gamma : list of float
        Gamma (z-axis) rotation angles in degrees.
    opt_theory : str
        Theory for optimization.
    resp_theory : str
        Theory for RESP calculation.
    net_charge : float
        Net charge for the molecule.
    force_gaussian_rerun : bool
        Whether to force rerun of Gaussian.
    gaussian_cwd : Path
        Directory for Gaussian calculations.
    in_com_template : Path
        Template for input Gaussian .com files.
    xyz : Path
        Path to the output XYZ file for rotations.
    """

    def __init__(self, stage_name: str, main_input: Union[Path, str], cwd: Union[Path, str], *args, **kwargs) -> None:
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_mol2 = Path(main_input)
        self.out_gaussian_label = kwargs["out_gaussian_label"]

        self._validate_input_paths(**kwargs)
        self.opt_theory = kwargs.get("opt_theory", "HF/6-31G*")
        self.resp_theory = kwargs.get("resp_theory", "HF/6-31G*")
        self.net_charge = kwargs.get("net_charge", 0.0)
        self.force_gaussian_rerun = kwargs.get("force_gaussian_rerun", False)
        self.gaussian_cwd = Path(self.cwd, "gaussianCalcs")

        if "alpha" not in kwargs or "beta" not in kwargs or "gamma" not in kwargs:
            raise ValueError("Please provide the alpha, beta, and gamma angles as lists")

        self.alpha = [float(a) for a in kwargs["alpha"]]
        self.beta = [float(b) for b in kwargs["beta"]]
        self.gamma = [float(g) for g in kwargs["gamma"]]

        self.in_com_template = Path(self.gaussian_cwd, f"{self.out_gaussian_label}.com")
        self.xyz = Path(self.gaussian_cwd, f"{self.out_gaussian_label}_rotations.xyz")

        return

    def _validate_input_paths(self, **kwargs):
        """
        Validate and set input paths for Gaussian execution.

        Parameters
        ----------
        **kwargs
            Keyword arguments containing Gaussian path options.

        Raises
        ------
        ValueError
            If a required option is missing.
        """
        for opt in ("gaussian_root", "gauss_exedir", "gaussian_binary", "gaussian_scratch"):
            try:
                setattr(self, opt, kwargs.get(opt, ""))
            except KeyError:
                raise ValueError(f"ERROR: Please provide {opt} option as a keyword argument.")
        if self.gaussian_binary is None:
            self.gaussian_binary = "g16"

    def _append_stage(self, stage: "AbstractStage") -> "AbstractStage":
        """Append a stage to the current stage.

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

    def setup(self, name_template: str) -> bool:
        """
        Set up Gaussian input and output files for the rotation calculations.

        Parameters
        ----------
        name_template : str or Path
            Template name for input/output files. Accepted as either a string
            or a Path (callers may pass either).

        Returns
        -------
        bool
            Always returns False (rotation calculations are not pre-completed).
        """
        self.header = [f"%NPROC={self.nproc}",
                       f"%MEM={self.mem}GB"]

        # __init__ tries to set up the coordinates object, but it may not have been available at init time.
        if not getattr(self, "coord_object", None):
            self.coord_object = Coordinates(self.in_mol2, filetype="pdb")
        self.gaussian_cwd.mkdir(exist_ok=True)
        logger.info(f"Setting up Gaussian calculations in {self.gaussian_cwd}")
        print(f"Setting up Gaussian calculations in {self.gaussian_cwd}")

        # Work-around: some callers pass name_template as a Path, others as a str.
        if not isinstance(name_template, Path):
            name_label = name_template
        else:
            name_label = name_template.name

        store_coords = []
        self.in_coms = []
        self.out_logs = []
        elements = self.coord_object.get_elements()
        for a, b, g in product(self.alpha, self.beta, self.gamma):
            test_rotation = self.coord_object.rotate(alpha=a, beta=b, gamma=g)
            store_coords.append(test_rotation)
            in_com = self.gaussian_cwd / f"{name_label}_rot_{a:0.2f}_{b:0.2f}_{g:0.2f}.com"
            print(f"--> Writing Gaussian input file: {in_com}")
            self.in_coms.append(in_com)
            newgau = GaussianWriter(in_com)
            newgau.add_block(
                GaussianInput(
                    command=f"#P {self.resp_theory} SCF(Conver=6) NoSymm Test Pop=mk IOp(6/33=2) GFInput GFPrint",
                    initial_coordinates=test_rotation,
                    elements=elements,
                    charge=self.net_charge,
                    header=self.header,
                )
            )
            # Always write the Gaussian input file
            newgau.write(dry_run=False)

            out_log = self.gaussian_cwd / f"{name_label}_rot_{a:0.2f}_{b:0.2f}_{g:0.2f}.log"
            self.out_logs.append(out_log)
            self._add_outputs(out_log)

        # Write the coordinates to a "trajectory" file
        self.write_rotation(store_coords, name_label)

        return False

    def execute(self, dry_run=False, nproc: Optional[int]=None, mem: Optional[int]=None) -> Any:
        """Execute Gaussian RESP calculations for each rotated ligand.

        Parameters
        ----------
        dry_run : bool, optional
            If True, log the commands that would be run without executing them.
        nproc : int, optional
            Number of processors to use.
        mem : int, optional
            Amount of memory to use (in GB).
        """
        super()._setup_execution(dry_run=dry_run, nproc=nproc, mem=mem)
        self.setup(self.out_gaussian_label)

        for i, (in_com, out_log) in enumerate(zip(self.in_coms, self.out_logs)):
            gau_run = Gaussian(
                cwd=self.gaussian_cwd,
                logger=self.logger,
                gaussian_root=self.gaussian_root,
                gauss_exedir=self.gauss_exedir,
                gaussian_binary=self.gaussian_binary,
                gaussian_scratch=self.gaussian_scratch,
            )
            gau_run.call(inp_pipe=in_com.name, out_pipe=out_log.name, dry_run=dry_run)
            self._print_status(i, self.alpha, self.beta, self.gamma)

        return

    def _print_rotation(self, alpha, beta, gamma):
        """
        Log the current rotation angles.

        Parameters
        ----------
        alpha : float
            Alpha (x-axis) rotation angle in degrees.
        beta : float
            Beta (y-axis) rotation angle in degrees.
        gamma : float
            Gamma (z-axis) rotation angle in degrees.
        """
        self.logger.info(f"---> Rotation: alpha={alpha}, beta={beta}, gamma={gamma}")
        return

    def _print_status(self, count, alphas, betas, gammas):
        """Log progress through the rotation grid.

        Parameters
        ----------
        count : int
            Index of the current rotation (0-based).
        alphas : list of float
            Alpha (x-axis) angles in degrees.
        betas : list of float
            Beta (y-axis) angles in degrees.
        gammas : list of float
            Gamma (z-axis) angles in degrees.
        """
        total_count = len(alphas) * len(betas) * len(gammas)
        percent = count / total_count * 100
        self.logger.info(f"Current Rotation Progress: {percent:.2f}%")
        return

    def write_rotation(self, coords, name_template: str):
        """
        Write the rotated coordinates to a trajectory file.

        Parameters
        ----------
        coords : list
            List of rotated coordinates.
        name_template : str
            Template name for the output file.
        """
        self.logger.info(f"--> Writing rotations to file: gaussianCalcs/{name_template}_rotations.xyz")
        with open(self.xyz, "w") as file_obj:
            for frame in coords:
                SimpleXYZ(file_obj, frame)
        return

    def _clean(self):
        return


class StageGaussiantoMol2(AbstractStage):
    """
    Convert Gaussian output to mol2 format and assign charges to the mol2 file.

    Parameters
    ----------
    stage_name : str
        The name of the stage.
    main_input : Union[Path, str]
        Path to the input Gaussian log file.
    cwd : Union[Path, str]
        Current working directory.
    template_mol2 : str
        Path to the template mol2 file.
    out_mol2 : str
        Path to the output mol2 file.
    net_charge : float, optional
        Net charge for the molecule (default: 0.0).
    atom_type : str, optional
        Atom type (default: 'gaff2').
    force_gaussian_rerun : bool, optional
        Whether to force rerun of Gaussian (default: False).

    Attributes
    ----------
    in_log : Path
        Path to the input Gaussian log file.
    template_mol2 : Path
        Path to the template mol2 file.
    out_mol2 : Path
        Path to the output mol2 file.
    temp1_mol2 : Path
        Path to the first temporary mol2 file.
    temp2_mol2 : Path
        Path to the second temporary mol2 file.
    net_charge : float
        Net charge for the molecule.
    atom_type : str
        Atom type.
    force_gaussian_rerun : bool
        Whether to force rerun of Gaussian.
    gaussian_cwd : Path
        Directory for Gaussian calculations.
    """

    def __init__(self, stage_name: str, main_input: Union[Path, str], cwd: Union[Path, str], *args, **kwargs) -> None:
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_log = Path(main_input)
        self.template_mol2 = Path(kwargs["template_mol2"])
        self.out_mol2 = Path(kwargs["out_mol2"])
        self.temp1_mol2 = Path(self.cwd, f"{self.out_mol2.stem}.tmp1.mol2")
        self.temp2_mol2 = Path(self.cwd, f"{self.out_mol2.stem}.tmp2.mol2")
        self.net_charge = kwargs.get("net_charge", 0.0)
        self.atom_type = kwargs.get("atom_type", "gaff2")

        self._validate_input_paths(**kwargs)
        self.net_charge = kwargs.get("net_charge", 0.0)
        self.force_gaussian_rerun = kwargs.get("force_gaussian_rerun", False)
        self.gaussian_cwd = Path(self.cwd, "gaussianCalcs")

        self._add_outputs(self.out_mol2)

    def _validate_input_paths(self, **kwargs) -> None:
        """
        Validate and set input paths for Gaussian execution.

        Parameters
        ----------
        **kwargs
            Keyword arguments containing Gaussian path options.

        Raises
        ------
        ValueError
            If a required option is missing.
        """
        for opt in ("gaussian_root", "gauss_exedir", "gaussian_binary", "gaussian_scratch"):
            try:
                setattr(self, opt, kwargs.get(opt, ""))
            except KeyError:
                raise ValueError(f"ERROR: Please provide {opt} option as a keyword argument.")
        if self.gaussian_binary is None:
            self.gaussian_binary = "g16"

    def _append_stage(self, stage: "AbstractStage") -> "AbstractStage":
        """Append the stage to the current stage."""
        return stage

    def setup(self, name_template: str) -> bool:
        """
        Set up required files and headers for Gaussian to mol2 conversion.

        Parameters
        ----------
        name_template : str
            Template name for input/output files.
        """
        self.add_required(self.in_log)

        self.header = [f"%NPROC={self.nproc}",
                       f"%MEM={self.mem}GB"]

    def execute(self, dry_run=False, nproc: Optional[int]=None, mem: Optional[int]=None) -> Any:
        """Execute the Gaussian to mol2 conversion.

        Parameters
        ----------
        dry_run : bool, optional
            If True, log the commands that would be run without executing them.
        nproc : int, optional
            Number of processors to use.
        mem : int, optional
            Amount of memory to use (in GB).
        """
        super()._setup_execution(dry_run=dry_run, nproc=nproc, mem=mem)

        warnings.filterwarnings("ignore")
        self.setup(self.in_log.stem)

        # Convert from gaussian to mol2
        ante = Antechamber(cwd=self.cwd, logger=self.logger, nproc=self.nproc)
        ante.call(i=self.in_log, fi="gout", o=self.temp1_mol2, fo="mol2", pf="y", at=self.atom_type, an="no", nc=self.net_charge, dry_run=dry_run)

        # Assign the charges
        if not dry_run:
            u1 = mda.Universe(self.template_mol2)
            u2 = mda.Universe(self.temp1_mol2)
            assert len(u1.atoms) == len(u2.atoms), "Number of atoms in the two files do not match"

            u2.atoms.charges = u1.atoms.charges
            """
            ag = u2.select_atoms("all")
            ag.write(self.name+'.tmp2.mol2')
            # This exists because for some reason antechamber misinterprets
            # the mol2 file's blank lines in the atoms section.
            self.remove_blank_lines(self.name+'.tmp2.mol2')
            """
            Mol2Writer(u2, self.temp2_mol2, selection="all").write()

        # Use antechamber to clean up the mol2 format
        ante = Antechamber(cwd=self.cwd, logger=self.logger, nproc=self.nproc)
        ante.call(i=self.temp2_mol2, fi="mol2", o=self.out_mol2, fo="mol2", pf="y", at=self.atom_type, an="no", nc=self.net_charge, dry_run=dry_run)

        return

    def _clean(self):
        return

    def remove_blank_lines(self, file_path):
        """Remove blank lines from a file.

        Parameters
        ----------
        file_path : str
            Path to the file to clean.
        """
        if Path(file_path).exists():
            # Read the file and filter out blank lines
            with open(file_path, "r") as file:
                lines = file.readlines()
                non_blank_lines = [line for line in lines if line.strip()]

            # Write the non-blank lines back to the file
            with open(file_path, "w") as file:
                file.writelines(non_blank_lines)
