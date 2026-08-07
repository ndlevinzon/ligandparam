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
from ligandparam.io.orientations import (
    get_quaternion_pack,
    minimum_pairwise_rotation_angle,
    quaternion_to_matrix,
)
from ligandparam.interfaces import Gaussian, Antechamber
from ligandparam.log import get_logger
from ligandparam.gaussian_budget import split_gaussian_job_budget

#
logger = logging.getLogger("ligandparam.gaussian")

# Use Opt(CalcFC) below this atom count; plain Opt at or above it.
_CALCFC_MAX_ATOMS = 50


def _orientation_id_from_paths(in_com: str | Path, out_log: str | Path) -> str:
    """Stable board id: ``q012`` or ``0.00_30.00_0.00`` from ``*_rot_<id>.*``."""
    stem = Path(in_com).stem
    marker = "_rot_"
    if marker in stem:
        return stem.split(marker, 1)[1]
    return Path(out_log).stem


def _run_gaussian_rotation_job(payload: dict) -> dict:
    """Run one rotation ESP job (spawn-pool worker; must be picklable)."""
    from ffpopt.progress_board import JobProgressStore

    cwd = Path(payload["cwd"])
    in_com = payload["in_com"]
    out_log = payload["out_log"]
    force = bool(payload.get("force", False))
    dry_run = bool(payload.get("dry_run", False))
    log_path = cwd / out_log
    job_id = payload.get("job_id") or _orientation_id_from_paths(in_com, out_log)
    store = None
    status_path = payload.get("status_path")
    if status_path:
        store = JobProgressStore(
            status_path,
            collection_key="orientations",
            id_header="Angle",
            title="Gaussian orientation ESP - live status",
        )

    def _set(**kwargs):
        if store is not None:
            store.update(job_id, **kwargs)

    if not force and GaussianReader(log_path).check_complete():
        _set(status="skipped", stage="finished", detail=f"already complete | {out_log}")
        return {"in_com": in_com, "status": "skipped", "job_id": job_id}

    _set(
        status="running",
        stage="gaussian",
        detail=f"{out_log} | %NProc={payload.get('job_nproc', '?')}",
        log_path=str(log_path),
    )
    try:
        gau = Gaussian(
            cwd=cwd,
            gaussian_root=payload.get("gaussian_root", ""),
            gauss_exedir=payload.get("gauss_exedir", ""),
            gaussian_binary=payload.get("gaussian_binary", "g16"),
            gaussian_scratch=payload.get("gaussian_scratch", ""),
            logger=logging.getLogger("ligandparam.gaussian.worker"),
        )
        stem = Path(in_com).stem
        gau.call(
            inp_pipe=in_com,
            out_pipe=out_log,
            dry_run=dry_run,
            script_name=f"_gau_{stem}.sh",
            scratch=str(cwd / "tmp" / f"scratch_{stem}"),
        )
        if not dry_run and not GaussianReader(log_path).check_complete():
            raise RuntimeError(f"Gaussian did not complete normally: {log_path}")
        _set(status="done", stage="finished", detail=f"ok | {out_log}")
        return {"in_com": in_com, "status": "ok", "job_id": job_id}
    except Exception as exc:
        _set(
            status="failed",
            stage="failed",
            detail=type(exc).__name__,
            error=str(exc)[:200],
        )
        raise


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
    """Rotate the ligand and run a Gaussian ESP job at each orientation.

    Supports two protocols:

    * ``so3_n28`` — 28 deterministic quaternion-packed SO(3) orientations
    * ``legacy_euler`` — historical Rx/Ry Euler grid (requires ``alpha``,
      ``beta``, ``gamma`` lists)

    Output logs are named ``{out_gaussian_label}_rot_*.log`` so
    :class:`~ligandparam.stages.resp.StageMultiRespFit` can discover them
    regardless of protocol.

    Parameters
    ----------
    stage_name : str
        Stage name.
    main_input : path-like
        Input mol2 used for the rotated ESP jobs.
    cwd : path-like
        Working directory (Gaussian files go under ``cwd/gaussianCalcs``).
    out_gaussian_label : str
        Filename prefix for ``.com`` / ``.log`` outputs.
    orientation_protocol : {"legacy_euler", "so3_n28"}, optional
        Orientation generator. Default ``legacy_euler`` when used as a
        standalone stage; recipes such as FreeLigand typically pass ``so3_n28``.
    alpha, beta, gamma : list of float, optional
        Euler angles in degrees (Rx, Ry, Rz). Required for ``legacy_euler``.
    resp_theory : str, optional
        Theory for the ESP / RESP single-point jobs.
    net_charge : float, optional
        Net molecular charge.
    force_gaussian_rerun : bool, optional
        If False (default), skip orientation logs that already show
        ``Normal termination``. If True, rerun every orientation.
    nproc : int, optional
        Total core budget for this stage. Concurrent jobs and per-job
        ``%NProc`` are chosen so ``n_workers * %NProc <= nproc``.
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

        self.orientation_protocol = kwargs.get("orientation_protocol", "legacy_euler")
        if self.orientation_protocol == "legacy_euler":
            if "alpha" not in kwargs or "beta" not in kwargs or "gamma" not in kwargs:
                raise ValueError(
                    "legacy_euler requires alpha, beta, and gamma angle lists"
                )
            self.alpha = [float(a) for a in kwargs["alpha"]]
            self.beta = [float(b) for b in kwargs["beta"]]
            self.gamma = [float(g) for g in kwargs["gamma"]]
        elif self.orientation_protocol == "so3_n28":
            self.alpha = []
            self.beta = []
            self.gamma = []
        else:
            raise ValueError(
                "orientation_protocol must be 'legacy_euler' or 'so3_n28'"
            )

        self.in_com_template = Path(self.gaussian_cwd, f"{self.out_gaussian_label}.com")
        self.xyz = Path(self.gaussian_cwd, f"{self.out_gaussian_label}_rotations.xyz")

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

    def _orientation_coordinates(self):
        """Yield stable filename suffixes and coordinates for each orientation."""
        if self.orientation_protocol == "so3_n28":
            quaternions = get_quaternion_pack("so3_n28")
            minimum_angle = minimum_pairwise_rotation_angle(quaternions)
            self.logger.info(
                f"Using {len(quaternions)}-point SO(3) quaternion pack "
                f"(minimum pairwise angle {minimum_angle:.2f} degrees)"
            )
            for index, quaternion in enumerate(quaternions):
                rotation = quaternion_to_matrix(quaternion)
                yield f"q{index:03d}", self.coord_object.rotate_matrix(rotation)
            return

        for alpha, beta, gamma in product(self.alpha, self.beta, self.gamma):
            suffix = f"{alpha:0.2f}_{beta:0.2f}_{gamma:0.2f}"
            yield suffix, self.coord_object.rotate(
                alpha=alpha, beta=beta, gamma=gamma
            )

    def _n_orientation_count(self) -> int:
        """Number of orientation jobs this stage will write."""
        if self.orientation_protocol == "so3_n28":
            return len(get_quaternion_pack("so3_n28"))
        return len(self.alpha) * len(self.beta) * len(self.gamma)

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
        job_nproc = getattr(self, "_job_nproc", None) or self.nproc
        self.header = [f"%NPROC={job_nproc}",
                       f"%MEM={self.mem}GB"]

        # __init__ tries to set up the coordinates object, but it may not have been available at init time.
        if not getattr(self, "coord_object", None):
            self.coord_object = Coordinates(self.in_mol2, filetype="pdb")
        self.gaussian_cwd.mkdir(exist_ok=True)
        logger.info(f"Setting up Gaussian calculations in {self.gaussian_cwd}")
        print(f"Setting up Gaussian calculations in {self.gaussian_cwd}")

        # Some recipes pass a Path while others pass a string.
        name_label = Path(name_template).name

        store_coords = []
        self.in_coms = []
        self.out_logs = []
        elements = self.coord_object.get_elements()
        for orientation_suffix, test_rotation in self._orientation_coordinates():
            store_coords.append(test_rotation)
            # Keep "<rotation label>_*.log" stable: StageMultiRespFit discovers
            # these files by that prefix regardless of the orientation protocol.
            in_com = self.gaussian_cwd / f"{name_label}_rot_{orientation_suffix}.com"
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

            out_log = self.gaussian_cwd / f"{name_label}_rot_{orientation_suffix}.log"
            self.out_logs.append(out_log)
            self._add_outputs(out_log)

        # Write the coordinates to a "trajectory" file
        self.write_rotation(store_coords, name_label)

        return False

    def execute(self, dry_run=False, nproc: Optional[int]=None, mem: Optional[int]=None) -> Any:
        """Execute Gaussian RESP calculations for each rotated ligand.

        Pools over ``.com`` jobs. ``nproc`` is the total core budget: workers and
        per-job ``%NProc`` satisfy ``n_workers * %NProc <= nproc``.

        Parameters
        ----------
        dry_run : bool, optional
            If True, log the commands that would be run without executing them.
        nproc : int, optional
            Total processor budget for concurrent orientation jobs.
        mem : int, optional
            Amount of memory to use (in GB) written into each ``%MEM`` header.
        """
        import multiprocessing as mp

        from ffpopt.progress_board import JobBoardWatcher, JobProgressStore

        # Prefer self. so subclasses / tests can override or patch setup.
        self._setup_execution(dry_run=dry_run, nproc=nproc, mem=mem)
        n_orients = self._n_orientation_count()
        n_workers, job_nproc = split_gaussian_job_budget(self.nproc, n_orients)
        self._rotation_n_workers = n_workers
        self._job_nproc = job_nproc
        self.logger.info(
            f"Gaussian rotation parallel plan: {n_orients} job(s), "
            f"nproc={self.nproc} -> {n_workers} worker(s) x %NProc={job_nproc}"
        )
        self.setup(self.out_gaussian_label)

        status_path = self.gaussian_cwd / ".rot_progress.json"
        board_path = self.gaussian_cwd / "ROT_STATUS.txt"
        store = JobProgressStore(
            status_path,
            collection_key="orientations",
            id_header="Angle",
            title="Gaussian orientation ESP - live status",
            empty_hint="no orientations registered yet",
            detail_hint_label="Per-orientation Gaussian logs",
        )

        pending = []
        for in_com, out_log in zip(self.in_coms, self.out_logs):
            job_id = _orientation_id_from_paths(in_com, out_log)
            already_done = (
                not self.force_gaussian_rerun
                and GaussianReader(out_log).check_complete()
            )
            if already_done:
                store.register(
                    job_id,
                    status="skipped",
                    stage="finished",
                    detail=f"already complete | {out_log.name}",
                    log_path=str(out_log),
                )
                self.logger.info(f"Skipping complete {out_log.name}")
                continue
            store.register(
                job_id,
                status="queued",
                stage="queued",
                detail=f"{out_log.name} | %NProc={job_nproc}",
                log_path=str(out_log),
            )
            pending.append(
                {
                    "cwd": str(self.gaussian_cwd),
                    "in_com": in_com.name,
                    "out_log": out_log.name,
                    "job_id": job_id,
                    "status_path": str(status_path),
                    "job_nproc": int(job_nproc),
                    "force": bool(self.force_gaussian_rerun),
                    "dry_run": bool(dry_run),
                    "gaussian_root": self.gaussian_root,
                    "gauss_exedir": self.gauss_exedir,
                    "gaussian_binary": self.gaussian_binary,
                    "gaussian_scratch": self.gaussian_scratch,
                }
            )

        total = len(self.in_coms)
        finished = total - len(pending)
        self.logger.info(
            f"Gaussian rotation status board: {board_path} "
            f"({finished} already complete, {len(pending)} pending)"
        )
        watcher = JobBoardWatcher(
            store,
            board_path=board_path,
            logger=self.logger,
            interval_sec=5.0,
            log_root_hint=str(self.gaussian_cwd / "*_rot_*.log"),
            thread_name="rot-progress-board",
        )
        watcher.start()
        try:
            if not pending:
                return

            workers = min(n_workers, len(pending))
            if dry_run or workers <= 1:
                for i, job in enumerate(pending):
                    _run_gaussian_rotation_job(job)
                    self._print_status(finished + i + 1, total)
                return

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for i, _result in enumerate(
                    pool.imap_unordered(_run_gaussian_rotation_job, pending)
                ):
                    self._print_status(finished + i + 1, total)
        finally:
            watcher.stop()

        return

    def _print_status(self, count, total_count):
        """Log progress through the orientation set."""
        percent = count / total_count * 100
        self.logger.info(f"Current Rotation Progress: {percent:.2f}%")

    def write_rotation(self, coords, name_template: str):
        """Write all rotated frames to ``{label}_rotations.xyz``."""
        self.logger.info(f"--> Writing rotations to file: gaussianCalcs/{name_template}_rotations.xyz")
        with open(self.xyz, "w") as file_obj:
            for frame in coords:
                SimpleXYZ(file_obj, frame)

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
