from pathlib import Path
from typing import Optional, Union, Any

from typing_extensions import override

from ligandparam.parametrization import Recipe, apply_option_defaults
from ligandparam.stages import (
    StageInitialize,
    StageDisplaceMol,
    StageNormalizeCharge,
    GaussianMinimizeRESP,
    StageLazyResp,
    StageUpdate,
    StageParmChk,
    StageLeap,
    DPMinimize,
)


class DPLigand(Recipe):
    """Parameterize a ligand with DeepMD minimization and Gaussian RESP.

    Uses DeepMD for geometry relaxation, then Gaussian RESP fitting and Leap
    to produce mol2/lib/frcmod outputs.

    Parameters
    ----------
    in_filename : path-like
        Input ligand structure (typically PDB).
    cwd : path-like
        Working directory for intermediate and output files.
    net_charge : int
        Net molecular charge.
    theory : dict, optional
        Mapping with ``low`` and ``high`` Gaussian theory levels.
    leaprc : list of str, optional
        Leaprc files for the Leap stage. Default ``["leaprc.gaff2"]``.
    force_gaussian_rerun : bool, optional
        Rerun Gaussian even if output logs already exist.
    nproc, mem : int, optional
        Gaussian processor count and memory in GB.
    gaussian_root, gauss_exedir, gaussian_binary, gaussian_scratch : optional
        Gaussian environment overrides; otherwise environment variables are used.
    **kwargs
        Extra options forwarded to stages (for example ``logger``).

    Raises
    ------
    KeyError
        If ``net_charge`` is not provided.

    Examples
    --------
    >>> from ligandparam.recipes import DPLigand
    >>> recipe = DPLigand(
    ...     "ligand.pdb", "output", net_charge=0, nproc=4, mem=8, logger="stream"
    ... )
    """

    @override
    def __init__(self, in_filename: Union[Path, str], cwd: Union[Path, str], *args, **kwargs):
        super().__init__(in_filename, cwd, *args, **kwargs)
        # logger will be passed manually to each stage
        kwargs.pop("logger", None)

        # required options
        for opt in ("net_charge",):
            try:
                setattr(self, opt, kwargs[opt])
                del kwargs[opt]
            except KeyError:
                raise KeyError(f"Missing {opt}")
        # required options with defaults (mutable defaults are created fresh)
        apply_option_defaults(
            self,
            kwargs,
            ("theory", "leaprc", "force_gaussian_rerun", "nproc", "mem"),
        )

        # optional options, without defaults
        for opt in ("gaussian_root", "gauss_exedir", "gaussian_binary", "gaussian_scratch"):
            setattr(self, opt, kwargs.pop(opt, None))

        self.kwargs = kwargs

    def setup(self):
        """Build the ordered DPLigand stage list on ``self.stages``.

        Stages cover initialization, DeepMD minimization, Gaussian RESP,
        charge/name updates, and Leap library generation.
        """
        initial_mol2 = self.cwd / f"{self.label}.initial.mol2"
        centered_mol2 = self.cwd / f"{self.label}.centered.mol2"
        lowtheory_minimization_gaussian_log = self.cwd / f"{self.label}.lowtheory.minimization.log"
        hightheory_minimization_gaussian_log = self.cwd / f"{self.label}.hightheory.minimization.log"
        resp_mol2_low = self.cwd / f"{self.label}.lowtheory.mol2"
        resp_mol2_high = self.cwd / f"{self.label}.minimized.mol2"
        resp_mol2 = self.cwd / f"{self.label}.resp.mol2"
        final_mol2 = self.cwd / f"final_{self.label}.mol2"
        nonminimized_mol2 = self.cwd / f"{self.label}.mol2"
        frcmod = self.cwd / f"{self.label}.frcmod"
        lib = self.cwd / f"{self.label}.lib"

        self.stages = [
            StageInitialize(
                "Initialize",
                main_input=self.in_filename,
                cwd=self.cwd,
                out_mol2=initial_mol2,
                net_charge=self.net_charge,
                logger=self.logger,
                **self.kwargs,
            ),
            StageNormalizeCharge(
                "Normalize1",
                main_input=initial_mol2,
                cwd=self.cwd,
                net_charge=self.net_charge,
                out_mol2=initial_mol2,
                logger=self.logger,
                **self.kwargs,
            ),
            StageDisplaceMol(
                "Centering",
                main_input=initial_mol2,
                cwd=self.cwd,
                out_mol=centered_mol2,
                logger=self.logger,
            ),
            DPMinimize(
                "DPMinimize",
                main_input=centered_mol2,
                cwd=self.cwd,
                out_xyz=centered_mol2.with_suffix(".xyz"),
                model=self.kwargs.get("model", "deepmd_model.pb"),
                ftol=self.kwargs.get("ftol", 0.01),
                steps=self.kwargs.get("steps", 50000),
                out_mol2=resp_mol2_low,
                logger=self.logger,
            ),
            GaussianMinimizeRESP(
                "MinimizeHighTheory",
                main_input=resp_mol2_low,
                cwd=self.cwd,
                nproc=self.nproc,
                mem=self.mem,
                gaussian_root=self.gaussian_root,
                gauss_exedir=self.gauss_exedir,
                gaussian_binary=self.gaussian_binary,
                gaussian_scratch=self.gaussian_scratch,
                net_charge=self.net_charge,
                resp_theory=self.theory["low"],
                force_gaussian_rerun=self.force_gaussian_rerun,
                out_gaussian_log=hightheory_minimization_gaussian_log,
                logger=self.logger,
                minimize=False,
                **self.kwargs,
            ),
            StageLazyResp(
                "LazyRespHigh",
                main_input=hightheory_minimization_gaussian_log,
                cwd=self.cwd,
                out_mol2=resp_mol2_high,
                net_charge=self.net_charge,
                logger=self.logger,
                **self.kwargs,
            ),
            StageNormalizeCharge(
                "Normalize2",
                main_input=resp_mol2_high,
                cwd=self.cwd,
                net_charge=self.net_charge,
                out_mol2=resp_mol2,
                logger=self.logger,
                **self.kwargs,
            ),
            StageUpdate(
                "UpdateNames",
                main_input=resp_mol2,
                cwd=self.cwd,
                source_mol2=initial_mol2,
                out_mol2=final_mol2,
                net_charge=self.net_charge,
                update_names=True,
                update_types=False,
                update_resname=True,
                logger=self.logger,
                **self.kwargs,
            ),
            # Create a `nonminimized_mol2` with `initial_mol2` coordinates and  `final_mol2` charges
            StageUpdate(
                "UpdateCharges",
                main_input=initial_mol2,
                cwd=self.cwd,
                source_mol2=final_mol2,
                out_mol2=nonminimized_mol2,
                update_charges=True,
                net_charge=self.net_charge,
                logger=self.logger,
                **self.kwargs,
            ),
            StageParmChk("ParmChk", main_input=nonminimized_mol2, cwd=self.cwd, out_frcmod=frcmod,
                         logger=self.logger,
                         **self.kwargs),
            StageLeap("Leap", main_input=nonminimized_mol2, cwd=self.cwd, in_frcmod=frcmod, out_lib=lib,
                      logger=self.logger, **self.kwargs),
        ]

    @override
    def execute(self, dry_run=False, nproc: Optional[int] = None, mem: Optional[int] = None) -> Any:
        """Run all stages defined by :meth:`setup`.

        Parameters
        ----------
        dry_run : bool, optional
            If True, log planned commands without running external programs.
        nproc : int, optional
            Override the recipe processor count for this run.
        mem : int, optional
            Override the recipe memory allocation in GB for this run.
        """
        self.logger.info(f"Starting the DPLigand recipe at {self.cwd}")
        super().execute(dry_run=dry_run, nproc=nproc, mem=mem)
        self.logger.info("Done with the DPLigand recipe")
