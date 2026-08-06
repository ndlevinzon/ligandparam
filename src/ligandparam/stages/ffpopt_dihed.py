"""Optional ffpopt dihedral-twist correction stage.

Wraps ``ffpopt.Workflows.run_fragmented_dihed_twist_workflow`` (package under
``src/ffpopt``) so recipes and the ``lig-dihed-correct`` CLI can apply torsion
corrections to a parent ``mol2`` / ``lib`` / ``frcmod`` triplet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from ligandparam.stages.abstractstage import AbstractStage


class StageDihedTwistCorrection(AbstractStage):
    """Fit and merge dihedral corrections into a parent Amber frcmod.

    Requires an installed ``ffpopt`` package (with ``scission`` / FragmentMol
    and AmberTools on ``PATH``). The parent ``lib`` is left unchanged; use the
    merged ``out_frcmod`` together with the original library in LEaP.

    Parameters
    ----------
    stage_name : str
        Stage name.
    main_input : path-like
        Parent ligand ``mol2``.
    cwd : path-like
        Working directory (fragment outputs go under ``out_dir``).
    in_lib : path-like
        Parent Amber ``lib``.
    in_frcmod : path-like
        Parent ``frcmod`` produced by ``parmchk2`` / the recipe.
    out_frcmod : path-like
        Destination for the merged, torsion-corrected frcmod.
    out_dir : path-like, optional
        Directory for scission fragments and per-fragment twist outputs.
        Default ``{cwd}/dihed_fragments``.
    model : str, optional
        High-level ffpopt model chemistry. Default ``"qdpi2"``.
    maxiter : int, optional
        Fit-then-rescan iterations per fragment. Default ``2``.
    nprim : int, optional
        Cosine primitives per torsion family. Default ``3``.
    geometric_opt : bool, optional
        Use geomeTRIC for constrained optimizations. Default ``True``.
    skip_existing : bool, optional
        Restart-friendly reuse of on-disk artifacts. Default ``True``.
    """

    def __init__(
        self,
        stage_name: str,
        main_input: Union[Path, str],
        cwd: Union[Path, str],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(stage_name, main_input, cwd, *args, **kwargs)
        self.in_mol2 = Path(main_input)
        self.in_lib = Path(kwargs["in_lib"])
        self.in_frcmod = Path(kwargs["in_frcmod"])
        self.out_frcmod = Path(kwargs["out_frcmod"])
        self.out_dir = Path(kwargs.get("out_dir", self.cwd / "dihed_fragments"))
        self.model = kwargs.get("model", "qdpi2")
        self.maxiter = int(kwargs.get("maxiter", 2))
        self.nprim = int(kwargs.get("nprim", 3))
        self.delta = int(kwargs.get("delta", 10))
        self.geometric_opt = bool(kwargs.get("geometric_opt", True))
        self.skip_existing = bool(kwargs.get("skip_existing", True))
        self.rotatable_bond_smarts = kwargs.get("rotatable_bond_smarts")
        self.add_required(self.in_mol2)
        self.add_required(self.in_lib)
        self.add_required(self.in_frcmod)

    def _append_stage(self, stage: "AbstractStage") -> "AbstractStage":
        return stage

    def _clean(self) -> None:
        raise NotImplementedError

    def execute(
        self,
        dry_run: bool = False,
        nproc: Optional[int] = None,
        mem: Optional[int] = None,
    ) -> Any:
        """Run the fragmented dihed-twist workflow and write ``out_frcmod``."""
        self._setup_execution(dry_run=dry_run, nproc=nproc, mem=mem)
        nproc_eff = self.nproc if nproc is None else nproc

        self.logger.info(
            "Dihed twist correction: mol2=%s lib=%s frcmod=%s → %s (model=%s)",
            self.in_mol2,
            self.in_lib,
            self.in_frcmod,
            self.out_frcmod,
            self.model,
        )
        if dry_run:
            self.logger.info(
                "Dry run: would call run_fragmented_dihed_twist_workflow "
                "(out_dir=%s, maxiter=%s, nproc=%s)",
                self.out_dir,
                self.maxiter,
                nproc_eff,
            )
            return None

        try:
            from ffpopt.Workflows import run_fragmented_dihed_twist_workflow
        except ImportError as exc:
            raise ImportError(
                "StageDihedTwistCorrection requires the integrated 'ffpopt' "
                "package (src/ffpopt) plus 'scission' / FragmentMol and "
                "AmberTools on PATH. Reinstall ligandparam editable and "
                "install scission into this environment."
            ) from exc

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_frcmod.parent.mkdir(parents=True, exist_ok=True)

        result = run_fragmented_dihed_twist_workflow(
            mol2=str(self.in_mol2.resolve()),
            lib=str(self.in_lib.resolve()),
            frcmod=str(self.in_frcmod.resolve()),
            out_dir=str(self.out_dir.resolve()),
            merged_frcmod=str(self.out_frcmod.resolve()),
            model=self.model,
            maxiter=self.maxiter,
            nprim=self.nprim,
            delta=self.delta,
            nproc=int(nproc_eff),
            geometric_opt=self.geometric_opt,
            skip_existing=self.skip_existing,
            rotatable_bond_smarts=self.rotatable_bond_smarts,
        )
        self.logger.info(
            "Dihed twist complete: merged_frcmod=%s fragments=%s",
            result.get("merged_frcmod"),
            [f.get("fragment_id") for f in result.get("fragments", [])],
        )
        return result


def dihed_twist_stage_kwargs(
    *,
    mol2: Path,
    lib: Path,
    frcmod: Path,
    out_frcmod: Path,
    out_dir: Path,
    model: str,
    maxiter: int,
    nproc: int,
    logger,
    geometric_opt: bool = True,
    nprim: int = 3,
    delta: int = 10,
    skip_existing: bool = True,
    rotatable_bond_smarts=None,
) -> dict:
    """Keyword arguments for constructing :class:`StageDihedTwistCorrection`."""
    return {
        "main_input": mol2,
        "in_lib": lib,
        "in_frcmod": frcmod,
        "out_frcmod": out_frcmod,
        "out_dir": out_dir,
        "model": model,
        "maxiter": maxiter,
        "nproc": nproc,
        "nprim": nprim,
        "delta": delta,
        "geometric_opt": geometric_opt,
        "skip_existing": skip_existing,
        "rotatable_bond_smarts": rotatable_bond_smarts,
        "logger": logger,
    }
