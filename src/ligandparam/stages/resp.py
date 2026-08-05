import glob
from typing import Optional,  Union, Any
from pathlib import Path

from ligandparam.stages.abstractstage import AbstractStage
from ligandparam.interfaces import Antechamber
from ligandparam.io.gaussianIO import GaussianReader

from ligandparam.multiresp import parmhelper
from ligandparam.multiresp.residueresp import ResidueResp


class StageLazyResp(AbstractStage):
    """
    Runs a 'lazy' RESP calculation based on a single Gaussian output file.

    Parameters
    ----------
    stage_name : str
        Name of the stage.
    main_input : Union[Path, str]
        Path to the input Gaussian log file.
    cwd : Union[Path, str]
        Current working directory.
    out_mol2 : str
        Path to the output mol2 file (from kwargs).
    net_charge : float, optional
        Net charge for the molecule (default is 0.0).
    atom_type : str, optional
        Atom type for antechamber (default is 'gaff2').
    molname : str, optional
        Molecule name (from kwargs).
    """

    def __init__(self, stage_name: str, main_input: Union[Path, str], cwd: Union[Path, str], *args, **kwargs) -> None:
        """
        Initialize the StageLazyResp.

        Parameters
        ----------
        stage_name : str
            Name of the stage.
        main_input : Union[Path, str]
            Path to the input Gaussian log file.
        cwd : Union[Path, str]
            Current working directory.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments. Must include 'out_mol2'.
        """
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_gaussian_log = Path(main_input)
        self.add_required(self.in_gaussian_log)
        self.out_mol2 = Path(kwargs["out_mol2"])

        self.net_charge = kwargs.get("net_charge", 0.0)
        self.atom_type = kwargs.get("atom_type", "gaff2")

        if "molname" in kwargs:
            self.additional_args = {"rn": kwargs["molname"]}
        else:
            self.additional_args = {}

    def _append_stage(self, stage: "AbstractStage") -> "AbstractStage":
        """
        Appends the stage to the workflow.

        Parameters
        ----------
        stage : AbstractStage
            Stage to append.

        Returns
        -------
        AbstractStage
            The appended stage.
        """
        return stage

    def execute(self, dry_run=False, nproc: Optional[int]=None, mem: Optional[int]=None) -> Any:
        """
        Execute antechamber to convert the Gaussian output to a mol2 file.

        Parameters
        ----------
        dry_run : bool, optional
            If True, the stage will not be executed, but the function will print the commands that would be run.
        nproc : int, optional
            Number of processors to use (default is None).
        mem : int, optional
            Memory to use in MB (default is None).

        Returns
        -------
        Any
            None
        """
        ante = Antechamber(cwd=self.cwd, logger=self.logger, nproc=self.nproc)
        ante.call(
            i=self.in_gaussian_log,
            fi="gout",
            o=self.out_mol2,
            fo="mol2",
            gv=0,
            c="resp",
            nc=self.net_charge,
            at=self.atom_type,
            an="no",
            dry_run=dry_run,
            **self.additional_args,
        )
        return

    def _clean(self):
        """
        Clean the files generated during the stage.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("clean method not implemented")


class StageMultiRespFit(AbstractStage):
    """
    Runs a multi-state RESP fitting calculation based on multiple Gaussian output files.

    Parameters
    ----------
    stage_name : str
        Name of the stage.
    main_input : Union[Path, str]
        Path to the input mol2 file.
    cwd : Union[Path, str]
        Directory containing Gaussian output files.
    in_gaussian_label : str
        Label for Gaussian output files (from kwargs).
    out_respfit : str
        Path to the output RESP fit file (from kwargs).
    net_charge : float, optional
        Net charge for the molecule (default is 0.0).
    expected_gaussian_logs : int, optional
        Required number of rotation logs. If provided, fitting stops rather
        than silently using a partial or mixed orientation set.
    """

    def __init__(self, stage_name: str, main_input: Union[Path, str], cwd: Union[Path, str], *args, **kwargs) -> None:
        """
        Initialize the StageMultiRespFit.

        Parameters
        ----------
        stage_name : str
            Name of the stage.
        main_input : Union[Path, str]
            Path to the input mol2 file.
        cwd : Union[Path, str]
            Directory containing Gaussian output files.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments. Must include 'in_gaussian_label' and 'out_respfit'.
        """
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_gaussian_label = kwargs["in_gaussian_label"]
        self.in_mol2 = Path(main_input)
        self.in_gaussian_dir = Path(cwd)
        self.glob_str = str(self.in_gaussian_dir / f"*{self.in_gaussian_label}_*.log")
        # self.add_required(self.in_gaussian_log)
        self.out_respfit = Path(kwargs["out_respfit"])

        self.net_charge = kwargs.get("net_charge", 0.0)
        self.expected_gaussian_logs = kwargs.get("expected_gaussian_logs")

    def _append_stage(self, stage: "AbstractStage") -> "AbstractStage":
        """
        Appends the stage to the workflow.

        Parameters
        ----------
        stage : AbstractStage
            Stage to append.

        Returns
        -------
        AbstractStage
            The appended stage.
        """
        return stage

    def execute(self, dry_run=False, nproc: Optional[int]=None, mem: Optional[int]=None) -> Any:
        """
        Execute a multi-state RESP fitting calculation.

        Parameters
        ----------
        dry_run : bool, optional
            If True, the stage will not be executed, but the function will print the commands that would be run.
        nproc : int, optional
            Number of processors to use (default is None).
        mem : int, optional
            Memory to use in MB (default is None).

        Returns
        -------
        Any
            None
        """
        # Both Euler and quaternion rotation jobs preserve the
        # "<in_gaussian_label>_*.log" naming contract. Sorting makes the
        # multi-state fit deterministic (q000, q001, ... for quaternion packs).
        gaussian_logs = sorted(glob.glob(self.glob_str))
        if not gaussian_logs:
            raise FileNotFoundError(
                f"No Gaussian rotation logs matched {self.glob_str!r}"
            )
        if (
            self.expected_gaussian_logs is not None
            and len(gaussian_logs) != self.expected_gaussian_logs
        ):
            raise RuntimeError(
                f"Expected {self.expected_gaussian_logs} Gaussian rotation logs "
                f"for {self.in_gaussian_label!r}, found {len(gaussian_logs)}. "
                "Refusing to perform a partial or mixed-orientation RESP fit."
            )
        incomplete_logs = [
            filename
            for filename in gaussian_logs
            if not GaussianReader(filename).check_complete()
        ]
        if incomplete_logs:
            names = ", ".join(Path(filename).name for filename in incomplete_logs)
            raise RuntimeError(
                f"{len(incomplete_logs)} Gaussian rotation job(s) did not "
                f"terminate normally: {names}"
            )

        comp = parmhelper.BASH(12)
        model = ResidueResp(comp, 1)
        model.add_state(
            self.in_gaussian_label,
            str(self.in_mol2),
            gaussian_logs,
            qmmask="@*",
        )
        model.multimolecule_fit(True)
        model.perform_fit("@*", unique_residues=False)
        with open(self.out_respfit, "w") as f:
            model.print_resp(fh=f)

        return

    def _clean(self):
        """
        Clean the files generated during the stage.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("clean method not implemented")
