from pathlib import Path
from typing import Optional, Union, Any

from typing_extensions import override

from ligandparam.parametrization import Recipe, apply_option_defaults
from ligandparam.io.orientations import (
    DEFAULT_ORIENTATION_PROTOCOL,
    N_ORIENTATIONS_SO3_N28,
    legacy_euler_kwargs,
)
from ligandparam.recipes.common import charge_update_parmchk_leap_stages
from ligandparam.recipes.dihed_options import apply_dihed_options, append_dihed_twist_stage
from ligandparam.stages import (
    StageInitialize,
    StageNormalizeCharge,
    StageDisplaceMol,
    GaussianMinimizeRESP,
    StageGaussiantoMol2,
    StageGaussianRotation,
    StageLazyResp,
    StageMultiRespFit,
    StageUpdateCharge,
    StageUpdate,
    StageParmChk,
    StageLeap,
)

class FreeLigand(Recipe):
    """Parameterize a ligand with multi-orientation Gaussian RESP fitting.

    Pipeline: initialize → normalize → center → low/high Gaussian minimize →
    multi-orientation ESP (default: ``so3_n28`` quaternion pack) → multi-RESP
    fit → charge/name/type updates → ``parmchk2`` / LEaP (``.frcmod`` / ``.lib``).

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
        If True, re-run Gaussian stages even when logs already show
        ``Normal termination``. Exposed on ``lig-getparam`` as ``-O`` /
        ``--force-gaussian-rerun``. Default False (skip complete jobs;
        incomplete orientation ESP jobs are re-run individually).
    orientation_protocol : {"so3_n28", "legacy_euler"}, optional
        Multi-RESP orientation set. Default ``so3_n28`` (28 quaternion-packed
        orientations). ``legacy_euler`` restores the older alpha/beta grid.
    dihed_correct : bool, optional
        If True, append an ffpopt fragmented dihed-twist stage after Leap.
        For interactive sessions after ``lig-getparam``, prefer the separate
        ``lig-dihed-correct`` CLI instead.
    dihed_model : str, optional
        High-level model for dihedral fitting. Default ``"qdpi2"``.
    dihed_maxiter : int, optional
        Fit-then-rescan iterations. Default ``2``.
    dihed_delta : int, optional
        Wavefront dihedral step in degrees (same as CLI ``--delta``).
        Default ``10``.
    dihed_fragment_config : FragmentConfig or dict, optional
        Scission fragmentation settings (``FragmentConfig`` or a dict for
        ``FragmentConfig.from_dict``). Default ``None``.
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
    >>> from ligandparam.recipes import FreeLigand
    >>> recipe = FreeLigand(
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

        # Default is so3_n28; only legacy_euler opts into the old Euler grid.
        self.orientation_protocol = kwargs.pop(
            "orientation_protocol", DEFAULT_ORIENTATION_PROTOCOL
        )
        if self.orientation_protocol not in ("so3_n28", "legacy_euler"):
            raise ValueError(
                "orientation_protocol must be 'so3_n28' or 'legacy_euler', "
                f"got {self.orientation_protocol!r}"
            )
        # Angle lists belong only to legacy_euler; drop them for so3_n28 so they
        # cannot leak into StageGaussianRotation via **self.kwargs.
        if self.orientation_protocol == "so3_n28":
            for key in ("alpha", "beta", "gamma"):
                kwargs.pop(key, None)
        apply_dihed_options(self, kwargs)
        self.kwargs = kwargs

    def setup(self):
        """Build the ordered FreeLigand stage list on ``self.stages``."""
        initial_mol2 = self.cwd / f"{self.label}.initial.mol2"
        centered_mol2 = self.cwd / f"{self.label}.centered.mol2"
        lowtheory_minimization_gaussian_log = self.cwd / f"{self.label}.lowtheory.minimization.log"
        hightheory_minimization_gaussian_log = self.cwd / f"{self.label}.hightheory.minimization.log"
        resp_mol2_low = self.cwd / f"{self.label}.minimized.lowtheory.mol2"
        resp_mol2_high = self.cwd / f"{self.label}.minimized.mol2"
        # Namespace orientation labels so leftover logs from another protocol
        # cannot enter the same multi-RESP fit.
        if self.orientation_protocol == "so3_n28":
            rotation_label = f"{self.label}.rotation.so3_n28"
        else:
            rotation_label = f"{self.label}.rotation"
        out_respfit = self.cwd / f"respfit.charges.{self.label}"
        resp_mol2 = self.cwd / f"{self.label}.resp.mol2"
        final_mol2 = self.cwd / f"final_{self.label}.mol2"
        nonminimized_mol2 = self.cwd / f"{self.label}.mol2"
        frcmod = self.cwd / f"{self.label}.frcmod"
        lib = self.cwd / f"{self.label}.lib"

        # Recipe protocol wins over any leftover stage kwargs.
        rotation_kwargs = {
            **self.kwargs,
            "orientation_protocol": self.orientation_protocol,
        }
        if self.orientation_protocol == "legacy_euler":
            rotation_kwargs.update(legacy_euler_kwargs())

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
            GaussianMinimizeRESP(
                "MinimizeLowTheory",
                main_input=centered_mol2,
                cwd=self.cwd,
                nproc=self.nproc,
                mem=self.mem,
                gaussian_root=self.gaussian_root,
                gauss_exedir=self.gauss_exedir,
                gaussian_binary=self.gaussian_binary,
                gaussian_scratch=self.gaussian_scratch,
                net_charge=self.net_charge,
                opt_theory=self.theory["low"],
                resp_theory=self.theory["low"],
                force_gaussian_rerun=self.force_gaussian_rerun,
                out_gaussian_log=lowtheory_minimization_gaussian_log,
                logger=self.logger,
                minimize=self.kwargs.get("minimize", True),
                **self.kwargs,
            ),
            StageLazyResp(
                "Resp",
                main_input=lowtheory_minimization_gaussian_log,
                cwd=self.cwd,
                out_mol2=resp_mol2_low,
                net_charge=self.net_charge,
                logger=self.logger,
                **self.kwargs,
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
                opt_theory=self.theory["high"],
                resp_theory=self.theory["low"],
                force_gaussian_rerun=self.force_gaussian_rerun,
                out_gaussian_log=hightheory_minimization_gaussian_log,
                logger=self.logger,
                minimize=self.kwargs.get("minimize", True),
                **self.kwargs,
            ),
            StageGaussiantoMol2(
                "GrabGaussianCharge",
                main_input=hightheory_minimization_gaussian_log,
                cwd=self.cwd,
                nproc=self.nproc,
                mem=self.mem,
                gaussian_root=self.gaussian_root,
                gauss_exedir=self.gauss_exedir,
                gaussian_binary=self.gaussian_binary,
                gaussian_scratch=self.gaussian_scratch,
                net_charge=self.net_charge,
                theory=self.theory,
                force_gaussian_rerun=self.force_gaussian_rerun,
                template_mol2=initial_mol2,
                out_mol2=resp_mol2_high,
                logger=self.logger,
                **self.kwargs,
            ),
            StageGaussianRotation(
                "Rotate",
                main_input=resp_mol2_high,
                cwd=self.cwd,
                nproc=self.nproc,
                mem=self.mem,
                gaussian_root=self.gaussian_root,
                gauss_exedir=self.gauss_exedir,
                gaussian_binary=self.gaussian_binary,
                gaussian_scratch=self.gaussian_scratch,
                net_charge=self.net_charge,
                theory=self.theory,
                force_gaussian_rerun=self.force_gaussian_rerun,
                out_gaussian_label=rotation_label,
                logger=self.logger,
                **rotation_kwargs,
            ),
            # Gaussian stages write under cwd/gaussianCalcs
            StageMultiRespFit(
                "MultiRespFit",
                main_input=resp_mol2_high,
                cwd=self.cwd / "gaussianCalcs",
                in_gaussian_label=rotation_label,
                out_respfit=out_respfit,
                net_charge=self.net_charge,
                expected_gaussian_logs=N_ORIENTATIONS_SO3_N28,
                logger=self.logger,
                **self.kwargs,
            ),
            StageUpdateCharge(
                "UpdateCharge",
                main_input=resp_mol2_high,
                cwd=self.cwd,
                out_mol2=resp_mol2,
                charge_column=3,
                charge_source=out_respfit,
                net_charge=self.net_charge,
                logger=self.logger,
                **self.kwargs,
            ),
            StageNormalizeCharge(
                "Normalize2",
                main_input=resp_mol2,
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
                out_mol2=resp_mol2,
                update_names=True,
                update_types=False,
                update_resname=True,
                net_charge=self.net_charge,
                logger=self.logger,
                **self.kwargs,
            ),
            StageUpdate(
                "UpdateTypes",
                main_input=resp_mol2,
                cwd=self.cwd,
                source_mol2=initial_mol2,
                out_mol2=final_mol2,
                update_names=False,
                update_types=True,
                net_charge=self.net_charge,
                logger=self.logger,
                **self.kwargs,
            ),
            *charge_update_parmchk_leap_stages(
                recipe=self,
                initial_mol2=initial_mol2,
                final_mol2=final_mol2,
                nonminimized_mol2=nonminimized_mol2,
                frcmod=frcmod,
                lib=lib,
            ),
        ]
        append_dihed_twist_stage(
            self.stages,
            recipe=self,
            mol2=nonminimized_mol2,
            lib=lib,
            frcmod=frcmod,
        )

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
        self.logger.info(f"Starting the FreeLigand recipe at {self.cwd}")
        super().execute(dry_run=dry_run, nproc=nproc, mem=mem)
        self.logger.info("Done with the FreeLigand recipe")
