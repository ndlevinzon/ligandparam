#!/usr/bin/env python3
from __future__ import annotations

import copy
import os
import sys
import numpy as np

from typing import Generator, Optional
from pathlib import Path

from ffpopt.geom.GeomOpt import GeomOpt, is_mpi_worker, is_soft_opt_recovery, opt_recovery_label
from ffpopt.geom.Constraints import Constraint, ConstraintList
from ffpopt.geom.Restraints import RestraintList
from ffpopt.Struct import ListOfStruct, Struct

from .WavefrontMixins import (
    apply_slim_node_result,
    apply_wavefront_minimum_to_node,
    atomic_pickle_dump,
    bin_needs_rescue,
    clear_los_calc,
    clone_struct_geometry,
    demote_redundant_spawn,
    dihed_seed_targets,
    discrete_energy_spike,
    ensure_soft_opt_attrs,
    finalize_successful_node_opt,
    finite_profile_energy,
    format_wavefront_progress,
    geomopt_mm_then_hl,
    kcal_threshold_to_ev,
    load_wavefront_pickle,
    mark_node_failed,
    merge_standard_wavefront_kwargs,
    pickle_checkpoint_keep_calc_cache,
    pickle_load_compat,
    precheck_geometry_clash,
    print_wavefront,
    replace_node_with_pickle,
    require_main_guard_for_spawn,
    run_mp_spawn_drain_loop,
    run_mpi_spawn_drain_loop,
    seed_struct_rigid_dihed_rotates,
    select_rescue_seed,
    slim_completed_nodes_for_checkpoint,
    slim_node_result,
    write_node_pickle,
    cleanup_wavefront_geometric_scratch,
)



# Per-process worker state (set once via Pool initializer / MPI bcast).
_WORKER: dict = {}

# Reuse wavefront spawn workers across sequential bond scans in this process
# (same model/parm). Constraint indices travel with each job.
_REUSED_POOL: dict = {"pool": None, "key": None}


def close_reused_wavefront_pool() -> None:
    """Terminate a process-local reused wavefront pool, if any."""
    pool = _REUSED_POOL.get("pool")
    if pool is None:
        _REUSED_POOL["pool"] = None
        _REUSED_POOL["key"] = None
        return
    print_wavefront("closing reused spawn pool")
    try:
        pool.terminate()
        pool.join()
    except Exception:
        pass
    _REUSED_POOL["pool"] = None
    _REUSED_POOL["key"] = None
    print_wavefront("reused spawn pool closed")


def _pool_reuse_key(los: ListOfStruct, struct: Struct, nproc: int):
    from ffpopt.geom.Geometric import calc_cache_key

    return (int(nproc),) + tuple(calc_cache_key(los, struct))


def _acquire_wavefront_pool(nproc: int, los, con, struct):
    """Return ``(pool, owns_pool)``; reuse workers when model/parm match."""
    from ffpopt.runtime.NondaemonPool import make_wavefront_spawn_pool

    key = _pool_reuse_key(los, struct, nproc)
    if _REUSED_POOL["pool"] is not None and _REUSED_POOL["key"] == key:
        return _REUSED_POOL["pool"], False
    close_reused_wavefront_pool()
    pool = make_wavefront_spawn_pool(
        nproc,
        initializer=_init_worker,
        initargs=(los, con, struct),
    )
    _REUSED_POOL["pool"] = pool
    _REUSED_POOL["key"] = key
    # Drain must not terminate; close_reused_wavefront_pool() owns teardown.
    return pool, False



def _init_worker(los, con, template_struct, reslist=None) -> None:
    """Pool initializer: share los / constraint templates once per worker.

    1-D jobs pass a single ``Constraint`` and ``reslist is None``.
    N-D jobs pass a ``ConstraintList`` plus a ``RestraintList``.
    """
    from ffpopt.ase.Aimnet import configure_aimnet_spawn_worker

    configure_aimnet_spawn_worker(los)
    clear_los_calc(los)
    _WORKER["los"] = los
    _WORKER["template"] = copy.deepcopy(template_struct)
    if reslist is not None:
        _WORKER["con"] = None
        _WORKER["conlist"] = copy.deepcopy(con)
        _WORKER["reslist"] = copy.deepcopy(reslist)
    else:
        _WORKER["con"] = copy.deepcopy(con)
        _WORKER["conlist"] = None
        _WORKER["reslist"] = None



def _struct_from_coords(coords) -> Struct:
    return clone_struct_geometry(
        _WORKER["template"], coords, ene=0.0, frcs=None
    )


def _run_node_job(job: dict) -> dict:
    """Worker entry: slim coords job in -> slim result out (no ``los``)."""
    if "rcs" in job:
        node = WavefrontNode.from_job(
            job, _WORKER["los"], _WORKER["conlist"], _WORKER["reslist"]
        )
    else:
        if "con" in job and job["con"] is not None:
            from ffpopt.geom.Constraints import Constraint

            con = Constraint.from_dict(job["con"])
        else:
            con = _WORKER["con"]
        node = WavefrontNode.from_job(job, _WORKER["los"], con)
    node.calculate()
    return node.to_result()


def _run_node(node: "WavefrontNode") -> "WavefrontNode":
    """In-process entry (serial path); mutates and returns ``node``."""
    node.calculate()
    return node


def GetGridNeighbors(bidx, grid, validbins=None, *, stencil: str = "von_neumann"):
    """Return neighbor bin indices of ``bidx``.

    Default ``von_neumann`` is axis-aligned only (2 * ndim bins): enough to
    fill an N-D grid. ``moore`` is the full 3**ndim - 1 stencil, including
    diagonals (extra multi-starts, not required for coverage).
    """
    bidx = [int(round(x)) for x in bidx]
    ndim = len(grid.dims)

    def _keep(b):
        if validbins is None:
            return True
        gidx = grid.CptGlbIdxFromBinIdx(b)
        return gidx in validbins

    if stencil == "moore":
        from ndfes.GridUtils import LinearPtsToMeshPts

        lol = []
        for idim in range(ndim):
            if grid.dims[idim].isper:
                ilo = bidx[idim] - 1
                ihi = bidx[idim] + 2
                lol.append([i % grid.dims[idim].size for i in range(ilo, ihi)])
            else:
                ilo = max(bidx[idim] - 1, 0)
                ihi = min(bidx[idim] + 2, grid.dims[idim].size)
                lol.append([i for i in range(ilo, ihi)])
        pts = LinearPtsToMeshPts(lol)
        newpts = []
        for pt in pts:
            b = [int(round(x)) for x in pt]
            if b != bidx and _keep(b):
                newpts.append(b)
        return newpts

    if stencil not in ("von_neumann", "axis"):
        raise ValueError(
            f"Unknown neighbor stencil {stencil!r}; use 'von_neumann' or 'moore'"
        )

    newpts = []
    for idim in range(ndim):
        dim = grid.dims[idim]
        for step in (-1, 1):
            b = list(bidx)
            j = bidx[idim] + step
            if dim.isper:
                b[idim] = j % dim.size
            elif 0 <= j < dim.size:
                b[idim] = j
            else:
                continue
            if _keep(b):
                newpts.append(b)
    return newpts


class WavefrontNode:
    """ This is a node in the wavefront algorithm. It represents a single geometry optimization.

    It contains the atoms, the energy, the angle, and the constraints.

    Parameters
    ----------
    los : ListOfStruct
        Shared calculator / arg source.
    struct : Struct
        Starting geometry for this node.
    con : Constraint
        The constraint to apply to the optimization.
    angle : float, optional
        The angle to optimize. If not provided, it will be set to the initial angle of
        the constraint.
    level : int, optional
        The level of the node in the wavefront algorithm. Default is None.
    node_id : int, optional
        Unique id within the level.
    workdir : path-like, optional
        Directory for per-node pickle checkpoints.

    Attributes
    ----------
    energy : float
        The energy of the optimized geometry.
    angle : float
        The angle to optimize.
    active : bool
        Whether the node is active (i.e., whether it should be optimized).
    constraints : list of Constraint
        The constraints to apply to the optimization.
    opt_geom : Struct
        The optimized geometry.
    level : int
        The level of the node in the wavefront algorithm.

    """
    def __init__(
        self,
        los: ListOfStruct,
        struct: Struct,
        con: Constraint = None,
        angle: float = None,
        level: int = None,
        node_id: int = None,
        workdir: str | Path = ".",
        conlist: ConstraintList = None,
        reslist: RestraintList = None,
        rcs: list[float] = None,
    ) -> None:
        self.los = los
        self.struct = struct
        self.energy = None
        self.forces = np.zeros((len(struct.data["elements"]), 3))
        self.active = True
        self.opt_geom = None
        self.level = level
        self.node_id = node_id
        self.complete = False
        self.error = None
        self.soft_opt = False
        self.opt_recovery = None
        self.conlist = copy.deepcopy(conlist) if conlist is not None else None
        self.reslist = copy.deepcopy(reslist) if reslist is not None else None
        self.rcs = list(rcs) if rcs is not None else None
        if self.rcs is not None:
            self.angle = None
            self.constraints = list(self.conlist.cons) if self.conlist is not None else []
            self._assign_rcs()
            self.node_pkl = str(Path(workdir) / self.get_pkl_name())
        else:
            self.angle = angle
            self.constraints = [copy.deepcopy(con)]
            self.node_pkl = str(
                Path(workdir)
                / f"level_{self.level}_angle_{self.angle}_id_{self.node_id}_node.pckl"
            )

    @property
    def is_nd(self) -> bool:
        return getattr(self, "rcs", None) is not None

    def _assign_rcs(self):
        i = 0
        if self.conlist is not None:
            for c in self.conlist:
                c.value = self.rcs[i]
                i += 1
        if self.reslist is not None:
            for c in self.reslist:
                c.value = self.rcs[i]
                i += 1

    def get_pkl_name(self) -> str:
        """Return the N-D sidecar pickle filename for this node."""
        s = "~".join(["%.2f" % (x) for x in self.rcs])
        return f"level_{self.level}_rcs_{s}_id_{self.node_id}_node.pckl"

    @classmethod
    def from_job(cls, job: dict, los: ListOfStruct, con=None, reslist=None) -> "WavefrontNode":
        """Rebuild a node from a slim IPC job (coords + loc; no pickled ``los``)."""
        if "coords" in job and job["coords"] is not None:
            struct = _struct_from_coords(job["coords"])
        else:
            struct = job["struct"]
        workdir = Path(job["node_pkl"]).parent
        if "rcs" in job:
            node = cls(
                los=los,
                struct=struct,
                conlist=con,
                reslist=reslist,
                rcs=list(job["rcs"]),
                level=job["level"],
                node_id=job["node_id"],
                workdir=workdir,
            )
        else:
            node = cls(
                los=los,
                struct=struct,
                con=con,
                angle=job["angle"],
                level=job["level"],
                node_id=job["node_id"],
                workdir=workdir,
            )
        node.node_pkl = job["node_pkl"]
        node.complete = bool(job.get("complete", False))
        return node

    def to_job(self) -> dict:
        """Slim payload for spawn workers: loc + coords (not ``los``)."""
        coords = np.asarray(self.struct.data["positions"], dtype=float)
        if self.is_nd:
            return {
                "rcs": list(self.rcs),
                "coords": coords,
                "level": self.level,
                "node_id": self.node_id,
                "node_pkl": self.node_pkl,
                "complete": self.complete,
            }
        con0 = self.constraints[0] if self.constraints else None
        return {
            "angle": self.angle,
            "coords": coords,
            "level": self.level,
            "node_id": self.node_id,
            "node_pkl": self.node_pkl,
            "complete": self.complete,
            "con": con0.to_dict() if con0 is not None else None,
        }

    def to_result(self) -> dict:
        """Slim result payload: energy + optimized coords (not ``los`` / full node)."""
        return slim_node_result(self)

    def apply_result(self, result: dict) -> None:
        """Merge a slim worker result into this parent-side node."""
        apply_slim_node_result(self, result, clone_fn=clone_struct_geometry)

    def _ensure_soft_opt_attrs(self) -> None:
        """Fill soft-opt fields missing from older node pickles / checkpoints."""
        ensure_soft_opt_attrs(self)
    def replace_with_pickle(self) -> None:
        """Replace node fields from a sidecar pickle if present (restores ``los``)."""
        replace_node_with_pickle(
            self, found_msg=f"Found existing pickle file for node: {self.node_id}"
        )

    def calculate(self) -> None:
        """Calculate the energy of the atoms."""
        from ffpopt.runtime.Console import ensure_ascii_stdio

        ensure_ascii_stdio()
        if not self.complete:
            if self.is_nd:
                ncon = len(self.conlist) if self.conlist is not None else 0
                for ic in range(ncon):
                    self.conlist.cons[ic].value = self.rcs[ic]
                nres = len(self.reslist) if self.reslist is not None else 0
                for ic in range(nres):
                    self.reslist.rests[ic].value = self.rcs[ncon + ic]
            else:
                self.constraints[0].value = self.angle
            self.struct = seed_struct_rigid_dihed_rotates(
                self.struct,
                dihed_seed_targets(self),
                node_id=self.node_id,
            )
            precheck_err = self._precheck_geometry()
            if precheck_err is not None:
                self._mark_failed(precheck_err)
                return
            try:
                # Stable geomeTRIC basename beside the node pickle so a killed
                # mid-minimize can warm-start from ``*_optim.xyz`` on restart.
                geom_prefix = str(Path(self.node_pkl).with_suffix("")) + "_geom"
                soft = (not self.is_nd) and bool(
                    getattr(self.los.args, "soft_dihed_restraint", False)
                )
                if soft:
                    from ffpopt.scan.WavefrontMixins import run_soft_dihed_opt

                    con0 = self.constraints[0]
                    self.opt_geom = run_soft_dihed_opt(
                        self.los,
                        self.struct,
                        self.constraints,
                        list(con0.idxs),
                        float(self.angle),
                        geom_prefix,
                        node_id=self.node_id,
                    )
                else:
                    cons = self.constraints
                    rest = None
                    if self.is_nd:
                        cons = self.conlist.cons if self.conlist is not None else None
                        rest = self.reslist.rests if self.reslist is not None else None
                    self.opt_geom = geomopt_mm_then_hl(
                        self.los,
                        self.struct,
                        constraints=cons,
                        restraints=rest,
                        geom_prefix=geom_prefix,
                        node_id=self.node_id,
                    )
                self.opt_recovery = opt_recovery_label(self.opt_geom)
                self.soft_opt = is_soft_opt_recovery(self.opt_geom)
                if self.soft_opt:
                    print_wavefront(
                        f"Node {self.node_id} soft-accepted opt "
                        f"(recovery={self.opt_recovery}); will not spawn neighbors"
                    )
                finalize_successful_node_opt(self)
            except Exception as e:
                print_wavefront(
                    f"Node {self.node_id} optimization error: "
                    f"{type(e).__name__}: {e}"
                )
                self._mark_failed("optimization_error", e)

    def _write_checkpoint(self) -> None:
        """Write the node's data to a pickle file (without ``los``)."""
        write_node_pickle(self)

    def cleanup(self) -> None:
        """Remove the node pickle and geomeTRIC scratch for this node.

        Missing files are ignored (NFS/VAST: ``is_file()`` then ``remove``
        is a race; pickle write may also have been skipped).
        """
        filename = Path(f"{self.node_pkl}")
        try:
            existed = filename.is_file()
            filename.unlink(missing_ok=True)
            if existed:
                print_wavefront(f"Cleaning up node pickle file: {self.node_pkl}")
        except OSError as exc:
            print_wavefront(
                f"could not remove node pickle {self.node_pkl}: "
                f"{type(exc).__name__}: {exc}"
            )
        from ffpopt.geom.Geometric import (
            cleanup_geometric_scratch,
            geometric_prefix_from_node_pkl,
        )

        try:
            cleanup_geometric_scratch(
                geometric_prefix_from_node_pkl(self.node_pkl), keep_optim=False
            )
        except OSError as exc:
            print_wavefront(
                f"could not remove geomeTRIC scratch for {self.node_pkl}: "
                f"{type(exc).__name__}: {exc}"
            )

    def _mark_failed(self, reason: str, error: Optional[Exception] = None) -> None:
        mark_node_failed(
            self, reason, error, where=self.rcs if self.is_nd else self.angle
        )

    def _precheck_geometry(self, min_dist: float = 0.8) -> Optional[str]:
        """Return a failure reason, or ``None`` if the geometry looks usable.

        Distinguishes real clashes from precheck exceptions (imports, constraint
        apply failures, ...) so failure reports are not all labeled as clashes.
        """
        def _atoms():
            from ffpopt.geom.Constraints import FillConstraints, ApplyConstraints
            from ffpopt.scan.WavefrontMixins import uses_soft_dihed_restraint

            myatoms = self.struct.GetASEAtoms()
            if self.is_nd:
                return ApplyConstraints(
                    myatoms, self.conlist.cons, graph=self.struct.GetGraph()
                )
            # Soft restraint: do not hard-snap the scanned dihedral; that
            # clash-rejects bulky whole-ligand seeds before the optimizer runs.
            if uses_soft_dihed_restraint(self.los):
                return myatoms
            cons = FillConstraints(myatoms, copy.deepcopy(self.constraints))
            return ApplyConstraints(myatoms, cons)

        return precheck_geometry_clash(
            get_atoms=_atoms,
            bonds=self.struct.data["bonds"],
            min_dist=min_dist,
        )


class WavefrontLevel:
    """ This class represents a level in the wavefront algorithm, containing multiple nodes. 
    
    It is responsible for managing the nodes, optimizing them, and checking if the level is complete. 
    
    Attributes
    ----------
    nodes : list of WavefrontNode
        The nodes in the level.
    
    """
    def __init__(self, level_id: int = 0):
        self.nodes = []
        self.level_id = level_id

    def add_node(self,
                 los: ListOfStruct, struct: Struct,
                 con: Constraint = None, angle: float = None,
                 *, workdir: str | Path = ".",
                 conlist: ConstraintList = None,
                 reslist: RestraintList = None,
                 grid=None,
                 rcs: list[float] = None,
                 node_id=None):
        """Add a 1-D (angle) or N-D (rcs/grid) node to this level."""
        if rcs is not None or grid is not None:
            from ffpopt.geom.Constraints import FillConstraints
            if node_id is None:
                atoms = struct.GetASEAtoms()
                crds = atoms.get_positions()
                conrcs = FillConstraints(atoms, conlist.cons, force=True)
                conrcs = [c.value for c in conrcs]
                resrcs = [restraint.GetCrdValue(crds) for restraint in reslist]
                inprcs = conrcs + resrcs
                bidx = grid.GetBinIdx(inprcs)
                gidx = grid.CptGlbIdxFromBinIdx(bidx)
                node_id = f"{len(self.nodes)}_{gidx}"
            node = WavefrontNode(
                los=los,
                struct=struct,
                conlist=conlist,
                reslist=reslist,
                rcs=rcs,
                level=self.level_id,
                node_id=node_id,
                workdir=workdir,
            )
            self.nodes.append(node)
            return node
        node_id = len(self.nodes)
        node = WavefrontNode(
            los=los,
            struct=struct,
            con=con,
            angle=angle,
            level=self.level_id,
            node_id=node_id,
            workdir=workdir,
        )
        self.nodes.append(node)
        return node

    def check_node_checkpoints(self) -> None:
        """Check for existing checkpoints for all nodes in the level."""
        for node in self.nodes:
            node.replace_with_pickle()


class Wavefront:
    """ This class implements the wavefront algorithm for dihedral scans.
    
    It manages the levels, nodes, and the optimization process.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The atoms to optimize.
    stdargs
        The standard arguments for the optimization.
    con : Constraint
        The constraint to apply to the optimization.
    delta : int, optional
        The step size for the dihedral scan in degrees. Default is 10.
    max_levels : int, optional
        The maximum number of levels to explore in the wavefront. Default is 1.
    nproc : int, optional
        The number of optimizations to run at a time. Default is 1.
    starting_nodes : int, optional
        The number of initial nodes to start with. Default is 1.
    extra_dih : list of int, optional
        A list of additional dihedral angles for conformer generations.
    convergence_threshold : float, optional
        Energy convergence threshold (kcal/mol) for wavefront calculation. A
        revisited angle must lower its running minimum by at least this much to
        stay active and spawn another level. Default is 0.01.
    
    Attributes
    ----------
    atoms : ase.Atoms
        The atoms to optimize.
    stdargs
        The standard arguments for the optimization.
    con : Constraint
        The constraint to apply to the optimization.
    delta : int
        The step size for the dihedral scan in degrees.
    min_geom : GeomOpt
        The initial geometry optimization.
    min_geom_ang : float
        The initial dihedral angle.
    levels : list of WavefrontLevel
        The levels in the wavefront algorithm.
    min_energies : dict
        A dictionary mapping angles to their minimum energies.
    min_structures : dict
        A dictionary mapping angles to their minimum structures.
    max_levels : int
        The maximum number of levels to explore in the wavefront.
    nproc : int
        The number of optimizations to run at a time.
    
    """
    def __init__(self, los: ListOfStruct, struct: Struct = None, con: Constraint = None, delta: int = 10, max_levels: int = 1, nproc: int = 1, starting_nodes: int = 1, extra_dih: list[int] = None, num_conformers: int = 1, checkpoint: str = "wavefront_checkpoint.pkl", convergence_threshold: float = 0.01, conlist: ConstraintList = None, reslist: RestraintList = None, grid=None, use_mpi: bool = False) -> None:
        # N-D positional: Wavefront(los, conlist, reslist, grid, ...)
        if grid is None and hasattr(delta, "GetRegGridCenterPts"):
            grid = delta
            delta = 10
            conlist = struct
            reslist = con
            struct = None
            con = None
        elif conlist is None and struct is not None and hasattr(struct, "cons"):
            conlist = struct
            reslist = con
            struct = None
            con = None
        self.los = copy.deepcopy(los) if grid is not None else los
        self.struct = struct
        self.con = con
        self.delta = delta
        self.min_geom = None
        self.min_geom_ang = None
        self.extra_dih = extra_dih
        self.levels = []
        self.min_energies = {}
        self.min_structures = {}
        self.min_nodes = {}
        self.max_levels = max_levels
        self.nproc = nproc
        self.starting_nodes = starting_nodes
        self.num_conformers = num_conformers
        self.level_energies = []
        self.checkpoint = checkpoint
        self.workdir = str(Path(checkpoint).resolve().parent)
        self.restarted = []
        self.verbose = False
        self.convergence_threshold = convergence_threshold
        self._resume_queue = None
        self._pending_by_loc = {}
        self._inflight_locs = set()
        self._deferred_seeds = {}
        self._expand_count = {}
        self._recent_spawns = []
        self._rescue_count = {}
        self._completed_locs = set()
        self.grid = copy.deepcopy(grid) if grid is not None else None
        self.conlist = copy.deepcopy(conlist) if conlist is not None else None
        self.reslist = copy.deepcopy(reslist) if reslist is not None else None
        self.use_mpi = use_mpi
        self.bins = {}
        self.min_bins = {}
        if self.grid is not None:
            from ndfes import SpatialBin
            allrcs = self.grid.GetRegGridCenterPts()
            for rcs in allrcs:
                bidx = self.grid.GetBinIdx(rcs)
                gidx = self.grid.CptGlbIdxFromBinIdx(bidx)
                self.bins[gidx] = SpatialBin(bidx)
                self.bins[gidx].center = self.grid.GetBinCenter(bidx)

    @property
    def is_nd(self) -> bool:
        return getattr(self, "grid", None) is not None

    def restart_options(self, los: ListOfStruct, #stdargs: StandardArgs,
                        max_levels: int = -1, nproc: int = 1, checkpoint: str = None,
                        use_mpi: bool = False) -> None:
        """ This function is used to set the options for a restarted wavefront calculation.

        This lets you use slightly different options for a restarted calculation, such as changing the number of processors or the maximum number of levels.

        Parameters
        ----------
        stdargs
            The standard arguments for the optimization.
        max_levels : int, optional
            The maximum number of levels to explore in the wavefront. Default is -1, which
            means unlimited levels.
        nproc : int, optional
            The number of optimizations to run at a time. Default is 1.
        checkpoint : str, optional
            The path to the checkpoint file to use for the restarted calculation. If not provided, it will use the current checkpoint.

        Returns
        -------
        None
            This function does not return anything, but it sets the options for the wavefront calculation.
        
        """
        prev_options = {"los": self.los, "max_levels": self.max_levels, "nproc": self.nproc}
        if not self.levels:
            raise ValueError("Cannot restart a wavefront calculation that has not been initialized.")
        self.max_levels = max_levels
        self.nproc = nproc
        self.use_mpi = use_mpi
        #self.stdargs = stdargs
        self.los = copy.deepcopy(los) if self.is_nd else los

        if checkpoint is not None:
            self.checkpoint = checkpoint
            self.workdir = str(Path(checkpoint).resolve().parent)

        # Rebind the (possibly re-themed) calculator source onto every node. The
        # queue scheduler can leave incomplete nodes at any level, not just the
        # last one, so update them all rather than only self.levels[-1].
        for level in self.levels:
            for node in level.nodes:
                node.los = los
                if hasattr(node, "_ensure_soft_opt_attrs"):
                    node._ensure_soft_opt_attrs()
        for node in getattr(self, "_resume_queue", None) or []:
            if hasattr(node, "_ensure_soft_opt_attrs"):
                node._ensure_soft_opt_attrs()
        print_wavefront(f"number of times restarted: {len(self.restarted)}")
        self.restarted.append(prev_options)
        
        print_wavefront("restart options updated")

    def theory_change(self, #stdargs: StandardArgs,
                      los: ListOfStruct,
                      stride: int = 1) -> None:
        if self.is_nd:
            return self._theory_change_nd(los, stride)
        """ This function is used to change the theory of the calculator for all nodes in the wavefront calculation.
        
        This is useful if you want to change the level of theory for a restarted calculation, for example from a lower level of theory to a higher level of theory. This will take the current level, and change the calculator for all nodes in that level.
        
        Parameters
        ----------
        None
            This function does not take any parameters, but it changes the calculator for all nodes in the current level.
            
        
        Returns
        -------
        None
            This function does not return anything, but it changes the calculator for all nodes in the current level.
            
        """
        if not self.levels:
            raise ValueError("Cannot change the theory of a wavefront calculation that has not been initialized.")
        
        self.levels = []
        new_starting_level = WavefrontLevel(level_id=0)
        dihedral_angles = list(self.min_energies.keys())
        dihedral_angles.sort()
        for angle in dihedral_angles[::stride]:
            print_wavefront(f"theory_change add node at angle {angle}")
            old = self.min_structures[angle]
            # Re-bind geometry onto the new ListOfStruct template so parm/topology
            # match the updated theory (critical when seeding itNN from orig).
            template = los.structs[0] if getattr(los, "structs", None) else self.struct
            coords = np.asarray(old.data["positions"], dtype=float)
            clone = getattr(template, "clone_geometry", None)
            if callable(clone):
                seeded = clone(coords=coords, ene=0.0, frcs=None)
            else:
                seeded = copy.deepcopy(template)
                seeded.Update(0.0, coords, None)
            new_starting_level.add_node(
                los,
                seeded,
                self.con,
                angle=angle,
                workdir=self.workdir,
            )
        self.min_energies = {}
        self.min_structures = {}
        self.min_nodes = {}
        self.levels.append(new_starting_level)
        self.los = los
        clear_los_calc(los)
        print_wavefront("theory changed and new starting level added")





    def init_calculation(self) -> None:
        if self.is_nd:
            return self._init_calculation_nd()
        """ This initializes the wavefront calculation by setting up the first level and adding nodes to that level.
        
        This function is called at the beginning of the wavefront calculation to set up the initial conditions.
        It performs an initial geometry optimization to find the minimum geometry and sets up the first level with nodes.
        
        
        Returns
        -------
        None
            This function does not return anything, but it initializes the wavefront calculation.
            
        """
        print_wavefront("starting wavefront calculation")
        self.init_min()
        #Add the initial level
        self.add_level()
        chosen_first, chosen_second = find_adjacent_dihedrals(self.con, self.los)
        extra_dih = None
        if chosen_first is not None:
            extra_dih = chosen_first
        if chosen_second is not None:
            extra_dih = chosen_second
        
        print_wavefront("generating conformers and adding initial nodes")
        min_geoms = []
        
        for i in range(self.num_conformers):
            angle_increment = (360 // self.num_conformers) * i
            print_wavefront(
                f"generating conformer {i+1} with angle increment {angle_increment} deg"
            )
            min_geoms.append(self.init_conformer(self.struct, self.con, extra_dih, angle_increment))

        for i in range(self.starting_nodes):
            # Add the initial node with the minimum geometry angle
            add_ang = (360//self.delta//self.starting_nodes * i *self.delta)
            print_wavefront(
                "adding starting node at angle: "
                f"{self.nearest_angle(self.min_geom_ang + add_ang, self.delta) % 360}"
            )
            if self.init_check(self.struct, self.con, self.nearest_angle(self.min_geom_ang + add_ang, self.delta)% 360):
                self.levels[0].add_node(
                    self.los,
                    self.struct,
                    self.con,
                    angle=self.nearest_angle(self.min_geom_ang + add_ang, self.delta)% 360,
                    workdir=self.workdir,
                )
            
            if self.num_conformers > 0:
                for min_geom in min_geoms:
                    if self.init_check(min_geom, self.con, self.nearest_angle(self.min_geom_ang + add_ang, self.delta)% 360):
                        self.levels[0].add_node(
                            self.los,
                            min_geom,
                            self.con,
                            angle=self.nearest_angle(self.min_geom_ang + add_ang, self.delta) % 360,
                            workdir=self.workdir,
                        )

        if not self.levels[0].nodes:
            native = self.nearest_angle(self.min_geom_ang, self.delta) % 360
            print_wavefront(
                "all seed angles failed the hard-twist clash check; "
                f"seeding native angle {native} deg so the scan can start"
            )
            self.levels[0].add_node(
                self.los,
                self.struct,
                self.con,
                angle=native,
                workdir=self.workdir,
            )

    def init_conformer(self, struct: Struct, con: Constraint, extra_con_in: list[int], angle_increment: float, iter_count: int = 0) -> GeomOpt:
        """Initialize a conformer based on the initial geometry optimization.
        
        This function uses the extra dihedrals in addition to the main dihedral to generate a conformer
        where that dihedral is set to a random value between 0 and 360 degrees. If there is a clash, it will find it, and it will recall this function
        with the angle incremented by the delta value.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms to optimize.
        con : Constraint
            The constraint to apply during optimization.
        extra_con : list of int, optional
            A list of additional dihedral angles for conformer generations.
        angle_increment : float
            The angle increment to apply to the dihedral angle.
        iter_count : int, optional
            The current iteration count for the conformer generation. Default is 0.
        
        Returns
        -------
        GeomOpt
            A GeomOpt object representing the optimized geometry of the conformer.
            
        """
        from ffpopt.geom.Constraints import Constraint
        if iter_count > 5:
            raise ValueError("Maximum iteration count exceeded for conformer generation. This likely indicates a persistent clash that cannot be resolved.")
        
        new_atoms = copy.deepcopy(struct)
        extra_con = ",".join([str(x) for x in extra_con_in]) if extra_con_in else None
        if extra_con is not None:
            print_wavefront(
                f"generating conformer with extra dihedral {extra_con} "
                f"at angle {angle_increment}"
            )
            extra_con = Constraint.from_str(extra_con, graph=self.struct.GetGraph())
            extra_con.value = angle_increment
            if self.init_check(new_atoms, con, con.value, extra_con=extra_con, extra_angle=angle_increment):
                conf_min_geom = GeomOpt(self.los, new_atoms, constraints=[con, extra_con])
            else:
                print_wavefront(
                    f"skipping conformer generation for angle {angle_increment} "
                    "(invalid initial geometry)"
                )
                conf_min_geom = self.init_conformer(struct, con, extra_con_in, angle_increment=angle_increment + self.delta, iter_count=iter_count + 1)
        else:
            print_wavefront(
                f"generating conformer without extra dihedral at angle {angle_increment}"
            )
            print_wavefront("no adjacent dihedrals found for extra-conformer generation")
            if self.init_check(new_atoms, con, con.value):
                conf_min_geom = GeomOpt(self.los, new_atoms, constraints=[con])
            else:
                print_wavefront(
                    f"skipping conformer generation for angle {angle_increment} "
                    "(invalid initial geometry)"
                )
                conf_min_geom = self.init_conformer(struct, con, extra_con_in, angle_increment=angle_increment + self.delta, iter_count=iter_count + 1)
        
        return conf_min_geom
    



    def init_check(self, struct: Struct, con: Constraint, angle: float, extra_con: Constraint = None, extra_angle: float = None) -> bool:
        """ This function checks if the initial geometry is valid for the wavefront calculations.

        This function applies the constraints and then checks whether with those constraints in place, any atoms are too close to each other (less than 0.8 Anstroms apart) that are not bonded to each other.
        If they are too close, it returns False, indicating that the geometry is not valid for the wavefront calculations. If all atoms are sufficiently far apart, it returns True. The primary reasoning behind this is that if atoms are too close, 
        the geometry optimization will likely fail or take a very long time to converge, which is not desirable for the wavefront algorithm. This is especially important for the initial geometry, as it sets the stage for all subsequent optimizations in the wavefront algorithm.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms to check.
        con : Constraint
            The constraint to apply to the atoms.
        stdargs
            The standard arguments for the optimization.
        angle : float
            The angle to check.
        
        Returns
        -------
        bool
            Returns True if the initial geometry is valid, False otherwise.
            
        """
        from ffpopt.geom.Constraints import FillConstraints, ApplyConstraints
        from ffpopt.scan.WavefrontMixins import uses_soft_dihed_restraint

        if uses_soft_dihed_restraint(self.los):
            return True
        myatoms = struct.GetASEAtoms()
        con = copy.deepcopy(con)
        con.value = angle
        if extra_con is not None:
            if extra_angle is None:
                raise ValueError("extra_angle must be provided if extra_con is provided.")
            extra_con = copy.deepcopy(extra_con)
            extra_con.value = extra_angle
            cons = FillConstraints(myatoms, [con, extra_con])
        else:
            cons = FillConstraints(myatoms, [con])
        myatoms = ApplyConstraints(myatoms, cons)
        from ffpopt.geom.Constraints import has_nonbonded_clash
        clashed, i, j, dist = has_nonbonded_clash(
            myatoms.get_positions(), struct.data["bonds"], min_dist=0.8
        )
        if clashed:
            print_wavefront(
                f"warning: distance between atom {i} and atom {j} is {dist:.3f} Ang "
                "(< 0.8 Ang); omitting this node"
            )
            return False
        return True

    @staticmethod
    def nearest_angle(number: float, delta: float) -> float:
        """ Round a number to the nearest angle based on the delta.
        
        This function rounds a given number to the nearest angle based on the delta value.
        
        Parameters
        ----------
        number : float
            The number to round.
        delta : float
            The delta value to round to.
        
        Returns
        -------
        float
            The rounded number.

        Examples
        --------
        >>> from ffpopt.scan.WaveFront import Wavefront
        >>> Wavefront.nearest_angle(16, 30)
        30
        >>> Wavefront.nearest_angle(44, 30)
        30
        >>> Wavefront.nearest_angle(46, 30)
        60
        >>> Wavefront.nearest_angle(74, 30)
        60
            
        """
        return round(number / delta) * delta

    def calculate(self) -> None:
        """Apply the wavefront algorithm (1-D queue or N-D threads/MPI)."""
        from ffpopt.ase.Aimnet import aimnet_gpu_plan_message, cap_aimnet_nproc

        model = getattr(getattr(self.los, "args", None), "model", None)
        old = max(1, int(self.nproc))
        new = cap_aimnet_nproc(old, model)
        if new != old:
            print_wavefront(aimnet_gpu_plan_message(old, new, model))
            self.nproc = new
        if self.is_nd:
            if getattr(self, "use_mpi", False):
                return self.calculate_mpi()
            return self.calculate_threads()
        """Apply the wavefront algorithm to optimize a dihedral scan.

        This runs the wavefront as a single calculation queue: a persistent pool
        of ``nproc`` workers pulls nodes off the queue, and each finished node
        immediately enqueues its active neighbors. Levels are kept only as a
        post-processing label (every node keeps its ``level``), not as a
        synchronization barrier, so a slow node no longer stalls the rest of its
        level. The scan stops when the queue drains with no work in flight.

        Worker results return in completion order, so which redundant neighbor
        nodes get spawned (and therefore the exact per-angle minima) can vary
        between runs at ``nproc > 1``. Also note that ``nproc`` workers each may
        launch a geomeTRIC/psi4 subprocess, so the effective core usage is
        ``nproc`` times the per-worker thread count.

        1 - - - - - - - - o - - -
        2 - - - - - - - o x o - -
        3 - - - - - - o x o x o - 
        4 - - - - - o x o x x x o 
        5 o - - - o x o x x x x x
        6 x o - o x x x x x x x x 
        7 x x o x x x x x x x x x
        0          180        360

        The wavefront algorithm looks something like the above, where each o represents and activate node in the wavefront, and each x 
        represents an inactive node that has been calculated. Once there are no more active nodes, the algorithm stops.
        
        Returns
        -------
        tuple
            A tuple containing the angles, energies, and structures of the optimized nodes.
            
        """
        import multiprocessing
        from collections import deque
        from ffpopt.runtime.Console import ensure_ascii_stdio

        ensure_ascii_stdio()

        # Seed the queue: a fresh run initializes level 1; a restart re-enqueues
        # the work the checkpoint recorded as pending/in-flight (falling back to
        # any active, incomplete node for checkpoints predating _resume_queue).
        if not self.levels:
            self.init_calculation()
            pending = deque(self.levels[0].nodes)
        elif self._resume_queue:
            pending = deque(self._resume_queue)
        else:
            pending = deque(node for level in self.levels
                            for node in level.nodes
                            if node.active and not node.complete)

        self._rebuild_occupancy(pending)
        self._resume_queue = list(pending)
        self.save_checkpoint()
        cleanup_wavefront_geometric_scratch(self, keep_incomplete_optim=True)

        pool = None
        owns_pool = False
        external = getattr(self, "_external_mp_pool", None)
        if self.nproc > 1:
            if external is not None:
                pool = external
            else:
                pool, owns_pool = _acquire_wavefront_pool(
                    self.nproc, self.los, self.con, self.struct
                )

        from ffpopt.runtime.FastWavefront import wf_checkpoint_every
        checkpoint_every = wf_checkpoint_every(self.nproc)
        run_mp_spawn_drain_loop(
            pending=pending,
            nproc=self.nproc,
            pool=pool,
            run_node_job=_run_node_job,
            on_complete=self._on_complete,
            on_dispatch=self._mark_dispatch,
            on_skip=self._finish_loc,
            set_resume_queue=lambda q: setattr(self, "_resume_queue", q),
            save_checkpoint=self.save_checkpoint,
            cleanup_completed=self._cleanup_completed,
            print_progress=self._print_progress,
            checkpoint_every=checkpoint_every,
            terminate_pool=owns_pool,
        )

        self._resume_queue = []
        self.save_checkpoint()
        self._cleanup_completed()
        cleanup_wavefront_geometric_scratch(self, keep_incomplete_optim=False)
        self._print_progress(0, 0)
        results = self.sort_results()
        print_wavefront(
            f"finished this scan (angles={len(self.min_energies)}, "
            f"checkpoint={self.checkpoint})"
        )
        return results

    def _print_progress(self, pending: int, in_flight: int) -> None:
        if self.is_nd:
            return self._print_progress_nd(pending, in_flight)
        """ Print a one-line live progress summary for the calculation queue.

        Replaces the old per-level banner: since several levels are in flight at
        once there is no clean level boundary to print at, so this is emitted at
        each checkpoint and at the end instead.

        Parameters
        ----------
        pending : int
            Number of nodes queued but not yet started.
        in_flight : int
            Number of nodes currently being optimized by workers.

        """
        total_angles = 360 // self.delta if self.delta else 0
        print_wavefront(
            format_wavefront_progress(
                self,
                pending,
                in_flight,
                extra=f"angles={len(self.min_energies)}/{total_angles}",
            )
        )

    def _cleanup_completed(self) -> None:
        """ Delete per-node checkpoint pickles already captured in the checkpoint.

        Mirrors the old per-level cleanup: a node writes a ``*_node.pckl`` while
        it runs (so a crash mid-node can resume), which becomes redundant once
        the node's result is folded into a saved wavefront checkpoint. Called
        right after ``save_checkpoint`` so a delete never races ahead of saved
        state; ``WavefrontNode.cleanup`` is a no-op when the file is already gone.

        """
        for level in self.levels:
            for node in level.nodes:
                if node.complete:
                    try:
                        node.cleanup()
                    except OSError as exc:
                        print_wavefront(
                            f"Node {getattr(node, 'node_id', '?')} cleanup: "
                            f"{type(exc).__name__}: {exc}"
                        )

    def _get_or_create_level(self, level_id: int) -> WavefrontLevel:
        """ Return the level with ``level_id``, creating and appending it if new.

        Levels are a post-processing label under the queue scheduler; they are
        built lazily as nodes are spawned into them.

        Parameters
        ----------
        level_id : int
            The 1-based level identifier.

        Returns
        -------
        WavefrontLevel
            The existing or newly created level.

        """
        for level in self.levels:
            if level.level_id == level_id:
                return level
        level = WavefrontLevel(level_id=level_id)
        self.levels.append(level)
        return level

    def _store_result(self, node: WavefrontNode) -> None:
        """ Write a worker-returned node back into its level.

        The node round-tripped through a pool worker, so it is a copy; the
        placeholder in ``self.levels`` (located by level id and node id) is
        replaced, and its ``los`` is re-pointed at the shared one so the
        checkpoint does not accumulate a separate calculator source per node.

        Parameters
        ----------
        node : WavefrontNode
            The completed node returned by a pool worker.

        """
        node.los = self.los
        level = self._get_or_create_level(node.level)
        for i, existing in enumerate(level.nodes):
            if existing.node_id == node.node_id:
                level.nodes[i] = node
                return
        level.nodes.append(node)

    def _on_complete(self, node: WavefrontNode) -> list:
        """ Fold a finished node into the running minima and spawn its neighbors.

        Parameters
        ----------
        node : WavefrontNode
            A node whose calculation has finished.

        Returns
        -------
        list of WavefrontNode
            New neighbor nodes to enqueue (empty if the node is not active).

        """
        self._evaluate_node(node)
        spawned = []
        if node.active:
            if self.max_levels > 0 and node.level + 1 > self.max_levels:
                print_wavefront(
                    f"max_levels={self.max_levels}; not spawning from "
                    f"{getattr(node, 'angle', getattr(node, 'rcs', '?'))}"
                )
                node.active = False
            else:
                spawned.extend(self.spawn_neighbors(node))
        spawned.extend(self._rescue_outlier_bins(node))
        extra = self._finish_loc(node)
        if extra is not None:
            spawned.append(extra)
        return spawned

    def _ensure_occupancy(self) -> None:
        if getattr(self, "_pending_by_loc", None) is None:
            self._pending_by_loc = {}
        if getattr(self, "_inflight_locs", None) is None:
            self._inflight_locs = set()
        if getattr(self, "_deferred_seeds", None) is None:
            self._deferred_seeds = {}
        if getattr(self, "_expand_count", None) is None:
            self._expand_count = {}
        if getattr(self, "_recent_spawns", None) is None:
            self._recent_spawns = []
        if getattr(self, "_rescue_count", None) is None:
            self._rescue_count = {}
        if getattr(self, "_completed_locs", None) is None:
            self._completed_locs = set()

    def _n_grid_bins(self) -> int:
        if self.is_nd:
            return max(0, len(self.bins or {}))
        return max(1, int(360 // max(1, int(self.delta))))

    def _n_hard_bins(self) -> int:
        if self.is_nd:
            n = 0
            for b in (self.min_bins or {}).values():
                e = getattr(b, "energy", None)
                if e is not None and np.isfinite(e):
                    n += 1
            return n
        n = 0
        for loc, e in (self.min_energies or {}).items():
            if e is None or not np.isfinite(e):
                continue
            prev = (self.min_nodes or {}).get(loc)
            if prev is not None and getattr(prev, "soft_opt", False):
                continue
            n += 1
        return n

    def _spawn_guard_key(self, loc):
        if self.is_nd:
            return self._loc_key(rcs=loc)
        return self._loc_key(angle=loc)

    def _spawn_guard(self, decision, loc, node, old):
        """Demote BFS re-expansion after the 1-D profile is filled."""
        from ffpopt.runtime.EnvDefaults import env_float, env_int

        self._ensure_occupancy()
        key = self._spawn_guard_key(loc)
        improve = None
        if old is not None:
            try:
                improve = float(old) - float(node.energy)
            except (TypeError, ValueError):
                improve = None
        why = demote_redundant_spawn(
            reason=str(decision.get("reason") or ""),
            loc=key,
            prior_expands=int(self._expand_count.get(key, 0)),
            max_expand_per_loc=max(0, env_int("FFPOPT_WF_MAX_EXPAND", 3)),
            n_hard_bins=self._n_hard_bins(),
            n_grid=self._n_grid_bins(),
            improve_ev=improve,
            threshold_ev=kcal_threshold_to_ev(self.convergence_threshold),
            coverage_spawn_factor=float(
                env_float("FFPOPT_WF_COVERAGE_SPAWN_FACTOR", 4.0)
            ),
            recent_spawn_locs=self._recent_spawns,
            pingpong_window=max(0, env_int("FFPOPT_WF_PINGPONG_WINDOW", 8)),
        )
        if why is None:
            if decision.get("reason") == "hard_significant_improve":
                self._expand_count[key] = int(self._expand_count.get(key, 0)) + 1
                self._recent_spawns.append(key)
                if len(self._recent_spawns) > 32:
                    del self._recent_spawns[:-32]
            return decision
        out = dict(decision)
        out["active"] = False
        out["reason"] = why
        return out

    def _rescue_outlier_bins(self, node) -> list:
        """Reseed Laplacian spikes / failed bins from a better neighbor.

        Inactive BFS nodes never spawn, so a 6 kcal hole (DDM 240 deg vs
        230/250) stays forever. After each completion, inspect this angle
        and its two cycle neighbors. If a stored min sits above the
        discrete interpolant (or the bin failed), enqueue a visit seeded
        from the lower neighbor, Kabsch-lerping both when they agree.
        """
        if self.is_nd:
            return []
        from ffpopt.runtime.EnvDefaults import env_float, env_int

        max_rescue = int(env_int("FFPOPT_WF_RESCUE_MAX", 2))
        if max_rescue <= 0:
            return []
        rescue_kcal = float(env_float("FFPOPT_WF_RESCUE_KCAL", 2.0))
        if rescue_kcal <= 0.0:
            return []
        threshold_ev = kcal_threshold_to_ev(rescue_kcal)
        self._ensure_occupancy()
        delta = float(self.delta)
        center = float(self.nearest_angle(node.angle, self.delta) % 360)
        spawned = []
        kcal_per_ev = 1.0 / kcal_threshold_to_ev(1.0) if threshold_ev else 0.0
        here = self._loc_key(node)
        for ang in (
            center,
            float(self.nearest_angle(center - delta, self.delta) % 360),
            float(self.nearest_angle(center + delta, self.delta) % 360),
        ):
            loc = self._loc_key(angle=ang)
            if int(self._rescue_count.get(loc, 0)) >= max_rescue:
                continue
            left = float(self.nearest_angle(ang - delta, self.delta) % 360)
            right = float(self.nearest_angle(ang + delta, self.delta) % 360)
            energy = finite_profile_energy(self.min_energies, ang)
            left_e = finite_profile_energy(self.min_energies, left)
            right_e = finite_profile_energy(self.min_energies, right)
            completed = (
                loc in self._completed_locs
                or loc in self._inflight_locs
                or loc == here
            )
            if not bin_needs_rescue(
                energy,
                left_e,
                right_e,
                threshold_ev=threshold_ev,
                completed=completed,
            ):
                continue
            picked = select_rescue_seed(
                ang,
                left,
                right,
                energies=self.min_energies,
                structures=self.min_structures,
                lerp_ev=threshold_ev,
            )
            if picked is None:
                continue
            struct, seed_e, src = picked
            if energy is not None and self._seed_rank(seed_e) >= self._seed_rank(energy) - 1e-9:
                continue
            spike = discrete_energy_spike(energy, left_e, right_e)
            spike_kcal = spike * kcal_per_ev if energy is not None else float("inf")
            self._rescue_count[loc] = int(self._rescue_count.get(loc, 0)) + 1
            if energy is None:
                print_wavefront(
                    f"rescue angle {ang:g} from {src} "
                    f"(failed bin; retry {self._rescue_count[loc]}/{max_rescue})"
                )
            else:
                print_wavefront(
                    f"rescue angle {ang:g} from {src} "
                    f"(spike {spike_kcal:.2f} kcal; "
                    f"retry {self._rescue_count[loc]}/{max_rescue})"
                )
            child = self._enqueue_visit(
                loc,
                struct=struct,
                seed_energy=seed_e,
                level_id=(node.level or 0) + 1,
                angle=ang,
            )
            if child is not None:
                spawned.append(child)
        return spawned

    @staticmethod
    def _seed_rank(energy) -> float:
        if energy is None:
            return float("inf")
        try:
            val = float(energy)
        except (TypeError, ValueError):
            return float("inf")
        if not np.isfinite(val):
            return float("inf")
        return val

    def _loc_key(self, node=None, *, angle=None, rcs=None):
        """Stable occupancy key: 1-D snapped angle, N-D global bin index."""
        if self.is_nd:
            coords = rcs if rcs is not None else node.rcs
            bidx = self.grid.GetBinIdx(coords)
            return self.grid.CptGlbIdxFromBinIdx(bidx)
        ang = angle if angle is not None else node.angle
        return float(self.nearest_angle(ang, self.delta) % 360)

    def _rebuild_occupancy(self, pending) -> None:
        """Index pending nodes by loc; drop duplicate locs from the deque."""
        from collections import deque as _deque

        self._ensure_occupancy()
        self._pending_by_loc = {}
        self._inflight_locs = set()
        kept = _deque()
        for node in list(pending):
            loc = self._loc_key(node)
            if loc in self._pending_by_loc:
                continue
            if getattr(node, "seed_energy", None) is None:
                node.seed_energy = float("inf")
            self._pending_by_loc[loc] = node
            kept.append(node)
        pending.clear()
        pending.extend(kept)
        for level in getattr(self, "levels", None) or []:
            for node in getattr(level, "nodes", None) or []:
                if getattr(node, "complete", False):
                    self._completed_locs.add(self._loc_key(node))

    def _mark_dispatch(self, node: WavefrontNode) -> None:
        self._ensure_occupancy()
        loc = self._loc_key(node)
        self._pending_by_loc.pop(loc, None)
        self._inflight_locs.add(loc)

    def _finish_loc(self, node: WavefrontNode):
        """Free occupancy for ``node`` and enqueue the best deferred seed, if any."""
        self._ensure_occupancy()
        loc = self._loc_key(node)
        self._pending_by_loc.pop(loc, None)
        self._inflight_locs.discard(loc)
        self._completed_locs.add(loc)
        deferred = self._deferred_seeds.pop(loc, None)
        if deferred is None:
            return None
        return self._enqueue_visit(
            loc,
            struct=deferred["struct"],
            seed_energy=deferred["energy"],
            level_id=deferred.get("level", (node.level or 0) + 1),
            angle=node.angle,
            rcs=getattr(node, "rcs", None),
        )

    def _enqueue_visit(
        self,
        loc,
        *,
        struct,
        seed_energy,
        level_id,
        angle=None,
        rcs=None,
    ):
        """Return a new pending node, or None if loc is already queued/in-flight.

        A better (lower) ``seed_energy`` replaces the pending seed in place, or
        is stored as a deferred seed if that loc is already running.
        """
        self._ensure_occupancy()
        rank = self._seed_rank(seed_energy)
        pending = self._pending_by_loc.get(loc)
        if pending is not None:
            old = self._seed_rank(getattr(pending, "seed_energy", None))
            if rank < old:
                pending.struct = struct
                pending.seed_energy = rank
                print_wavefront(
                    f"Coalesce pending loc={loc}: better seed E={rank} < {old}"
                )
            return None
        if loc in self._inflight_locs:
            cur = self._deferred_seeds.get(loc)
            if cur is None or rank < self._seed_rank(cur.get("energy")):
                self._deferred_seeds[loc] = {
                    "struct": struct,
                    "energy": rank,
                    "level": level_id,
                }
                print_wavefront(f"Defer seed loc={loc}: in-flight, E={rank}")
            return None
        level = self._get_or_create_level(level_id)
        if self.is_nd:
            node = level.add_node(
                self.los,
                struct,
                conlist=self.conlist,
                reslist=self.reslist,
                grid=self.grid,
                rcs=rcs,
            )
        else:
            node = level.add_node(
                self.los,
                struct,
                self.con,
                angle=angle,
                workdir=self.workdir,
            )
        node.seed_energy = rank
        self._pending_by_loc[loc] = node
        return node

    def _rebuild_level_energies(self) -> None:
        if self.is_nd:
            return self._rebuild_level_energies_nd()
        """ Rebuild ``self.level_energies`` as a per-level convergence history.

        Produces one ``{angle: energy}`` snapshot per level, where level ``k``
        holds every angle whose final minimum was reached at level ``k`` or
        earlier. This is derived deterministically from the recorded minima
        (independent of worker completion order); the final snapshot equals
        ``self.min_energies``. ``ffpopt-WavefrontAnimate.py`` consumes it.

        """
        max_level = max((level.level_id for level in self.levels), default=0)
        snapshots = []
        for k in range(1, max_level + 1):
            snapshots.append({angle: energy
                              for angle, energy in self.min_energies.items()
                              if self.min_nodes[angle].level <= k})
        self.level_energies = snapshots

    def save_checkpoint(self) -> None:
        """Save the current state of the wavefront calculation to a file.
        
        This function saves the current state of the wavefront calculation to a file, which can be used to resume the calculation later.
        
        Parameters
        ----------
        filename : str
            The name of the file to save the checkpoint to.
            
        """
        # Refresh the derived per-level convergence history so any checkpoint is
        # immediately consumable by ffpopt-WavefrontAnimate.py.
        self._rebuild_level_energies()
        slim_completed_nodes_for_checkpoint(self)
        pickle_checkpoint_keep_calc_cache(self, self.checkpoint, self.los)
        print_wavefront(f"checkpoint saved to {self.checkpoint}")

    def _slim_nodes_for_checkpoint(self) -> None:
        """Drop bulky redundant arrays from completed nodes before pickling."""
        slim_completed_nodes_for_checkpoint(self)

    def init_min(self) -> None:
        """Initial geometry optimization and angle calculation.
        
        This does a geometry optimization of the initial atoms without using any constraints. This is used to find
        the initial minimized geometry. 
        
        """

        self.min_geom = GeomOpt(self.los, self.struct)
        self.min_geom_ang = self.min_geom.get_dihedral(self.con.idxs[0], 
                                                       self.con.idxs[1],
                                                       self.con.idxs[2], 
                                                       self.con.idxs[3])



    def add_level(self) -> WavefrontLevel:
        """Add a new level to the wavefront.
        
        This creates a new WavefrontLevel and appends it to the levels list.
        
        Returns
        -------
        WavefrontLevel
            The newly created WavefrontLevel object.
        
        """
        level_id = len(self.levels) + 1
        level = WavefrontLevel(level_id=level_id)
        self.levels.append(level)
        return level
    
    def _evaluate_node(self, node: WavefrontNode) -> None:
        if self.is_nd:
            return self._evaluate_node_nd(node)
        """Update per-angle minima and set ``node.active`` (spawn) via shared policy.

        See :func:`ffpopt.scan.WavefrontMixins.apply_wavefront_minimum_to_node`.
        ``convergence_threshold`` is kcal/mol; node energies are eV.
        """
        has_incumbent = node.angle in self.min_energies
        incumbent_energy = self.min_energies.get(node.angle)
        incumbent_soft = False
        if node.angle in self.min_nodes:
            prev = self.min_nodes[node.angle]
            incumbent_soft = bool(getattr(prev, "soft_opt", False))
            if not incumbent_soft:
                incumbent_soft = is_soft_opt_recovery(
                    self.min_structures.get(node.angle)
                )

        def on_update(n, _reason, _old):
            self.min_energies[n.angle] = n.energy
            self.min_structures[n.angle] = n.opt_geom
            self.min_nodes[n.angle] = n

        apply_wavefront_minimum_to_node(
            node,
            loc=node.angle,
            threshold_kcal=self.convergence_threshold,
            has_incumbent=has_incumbent,
            incumbent_energy=incumbent_energy,
            incumbent_soft=incumbent_soft,
            on_update=on_update,
            noun="angle",
            spawn_guard=self._spawn_guard,
        )

    def determine_active_nodes(self, current_level: WavefrontLevel) -> None:
        """ Evaluate every node in a level via :meth:`_evaluate_node`.

        Retained for backward compatibility; the queue scheduler calls
        :meth:`_evaluate_node` per node as results arrive rather than per level.

        Parameters
        ----------
        current_level : WavefrontLevel
            The level whose nodes to evaluate.

        """
        for node in current_level.nodes:
            self._evaluate_node(node)

    def spawn_neighbors(self, node: WavefrontNode) -> list:
        if self.is_nd:
            return self._spawn_neighbors_nd(node)
        """ Spawn the two neighbor nodes of an active node in the next level.

        For a node at 180 degrees with delta 10, this creates nodes at 170 and
        190 degrees in level ``node.level + 1`` (created lazily) and returns them
        so the caller can enqueue them.

        Parameters
        ----------
        node : WavefrontNode
            The active node for which to spawn neighbors.

        Returns
        -------
        list of WavefrontNode
            The newly created neighbor nodes.

        Raises
        ------
        ValueError
            If the node is inactive, it cannot spawn neighbors.

        """
        if not node.active:
            raise ValueError("Cannot spawn neighbors for an inactive node.")
        next_level_id = node.level + 1
        lower = self.nearest_angle(node.angle - self.delta, self.delta) % 360
        upper = self.nearest_angle(node.angle + self.delta, self.delta) % 360
        spawned = []
        for ang in (lower, upper):
            loc = self._loc_key(angle=ang)
            child = self._enqueue_visit(
                loc,
                struct=node.opt_geom,
                seed_energy=node.energy,
                level_id=next_level_id,
                angle=ang,
            )
            if child is not None:
                spawned.append(child)
        return spawned

    def calculate_threads(self) -> None:
            """Apply the wavefront algorithm to optimize a dihedral scan.
            
            This runs the wavefront as a single calculation queue: a persistent pool
            of ``nproc`` workers pulls nodes off the queue, and each finished node
            immediately enqueues its active neighbors. Levels are kept only as a
            post-processing label (every node keeps its ``level``), not as a
            synchronization barrier, so a slow node no longer stalls the rest of its
            level. The scan stops when the queue drains with no work in flight.

            Worker results return in completion order, so which redundant neighbor
            nodes get spawned (and therefore the exact per-angle minima) can vary
            between runs at ``nproc > 1``. Also note that ``nproc`` workers each may
            launch a geomeTRIC/psi4 subprocess, so the effective core usage is
            ``nproc`` times the per-worker thread count.

            1 - - - - - - - - o - - -
            2 - - - - - - - o x o - -
            3 - - - - - - o x o x o - 
            4 - - - - - o x o x x x o 
            5 o - - - o x o x x x x x
            6 x o - o x x x x x x x x 
            7 x x o x x x x x x x x x
            0          180        360

            The wavefront algorithm looks something like the above, where each o represents and activate node in the wavefront, and each x 
            represents an inactive node that has been calculated. Once there are no more active nodes, the algorithm stops.
            
            Returns
            -------
            tuple
                A tuple containing the angles, energies, and structures of the optimized nodes.
                
            """
            import multiprocessing
            from collections import deque

            # Seed the queue: a fresh run initializes level 1; a restart re-enqueues
            # the work the checkpoint recorded as pending/in-flight (falling back to
            # any active, incomplete node for checkpoints predating _resume_queue).

            print_wavefront("entered calculate_threads")
            
            if not self.levels:
                self.init_calculation()
                pending = deque(self.levels[0].nodes)
            elif self._resume_queue:
                pending = deque(self._resume_queue)
            else:
                pending = deque(node for level in self.levels
                                for node in level.nodes
                                if node.active and not node.complete)

            self._rebuild_occupancy(pending)
            self._resume_queue = list(pending)
            self.save_checkpoint()
            cleanup_wavefront_geometric_scratch(self, keep_incomplete_optim=True)

            pool = None
            owns_pool = False
            external = getattr(self, "_external_mp_pool", None)
            if self.nproc > 1:
                if external is not None:
                    pool = external
                else:
                    from ffpopt.runtime.NondaemonPool import make_wavefront_spawn_pool

                    template = self.los.structs[0] if getattr(self.los, "structs", None) else None
                    if template is None:
                        for level in self.levels:
                            for n in level.nodes:
                                template = n.struct
                                break
                            if template is not None:
                                break
                    pool = make_wavefront_spawn_pool(
                        self.nproc,
                        initializer=_init_worker,
                        initargs=(self.los, self.conlist, template, self.reslist),
                    )
                    owns_pool = True

            from ffpopt.runtime.FastWavefront import wf_checkpoint_every
            checkpoint_every = wf_checkpoint_every(self.nproc)
            run_mp_spawn_drain_loop(
                pending=pending,
                nproc=self.nproc,
                pool=pool,
                run_node_job=_run_node_job,
                on_complete=self._on_complete,
                on_dispatch=self._mark_dispatch,
                on_skip=self._finish_loc,
                set_resume_queue=lambda q: setattr(self, "_resume_queue", q),
                save_checkpoint=self.save_checkpoint,
                cleanup_completed=self._cleanup_completed,
                print_progress=self._print_progress,
                checkpoint_every=checkpoint_every,
                terminate_pool=owns_pool,
            )

            self._resume_queue = []
            self.save_checkpoint()
            self._cleanup_completed()
            cleanup_wavefront_geometric_scratch(self, keep_incomplete_optim=False)
            self._print_progress(0, 0)
            results = self.sort_results()
            print_wavefront(
                "finished this scan "
                f"(angles={len(getattr(self, 'min_energies', {}) or getattr(self, 'min_bins', {}))}, "
                f"checkpoint={getattr(self, 'checkpoint', None)})"
            )
            return results

    def calculate_mpi(self) -> None:
            """Apply the wavefront algorithm to optimize a dihedral scan using MPI."""
            from mpi4py import MPI
            from collections import deque

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            if size == 1:
                #print(f"calculate_mpi invoked by rank/size is {rank}/{size}")
                #return self.calculate_threads()
                raise Exception(f"calculate_mpi invoked by rank/size is {rank}/{size}")

            # Define communication tags
            TAG_TASK = 1
            TAG_RESULT = 2
            TAG_STOP = 3

            # Broadcast shared worker state once (los + templates); tasks are slim.
            if rank == 0:
                template = self.los.structs[0] if getattr(self.los, "structs", None) else None
                if template is None:
                    for level in self.levels:
                        for n in level.nodes:
                            template = n.struct
                            break
                        if template is not None:
                            break
                setup = (self.los, self.conlist, template, self.reslist)
            else:
                setup = None
            setup = comm.bcast(setup, root=0)

            # ---------------------------------------------------------
            # WORKER LOGIC (Ranks 1 to N-1)
            # ---------------------------------------------------------
            if rank > 0:
                _init_worker(*setup)
                while True:
                    status = MPI.Status()
                    # Block and wait for a task or stop signal from Rank 0
                    job = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                    tag = status.Get_tag()

                    if tag == TAG_STOP:
                        break  # Exit the worker loop

                    result = _run_node_job(job)
                    comm.send(result, dest=0, tag=TAG_RESULT)

                # Workers return None, leaving the main process to handle results
                return None

            # ---------------------------------------------------------
            # MASTER LOGIC (Rank 0)
            # ---------------------------------------------------------

            # 1. Seed the Queue
            if not self.levels:
                self.init_calculation()
                pending = deque(self.levels[0].nodes)
            elif self._resume_queue:
                pending = deque(self._resume_queue)
            else:
                pending = deque(node for level in self.levels
                                for node in level.nodes
                                if node.active and not node.complete)

            self._rebuild_occupancy(pending)
            self._resume_queue = list(pending)
            self.save_checkpoint()
            cleanup_wavefront_geometric_scratch(self, keep_incomplete_optim=True)

            from ffpopt.runtime.FastWavefront import wf_checkpoint_every

            checkpoint_every = max(wf_checkpoint_every(max(size - 1, 1)), 1)
            run_mpi_spawn_drain_loop(
                pending=pending,
                comm=comm,
                size=size,
                tag_task=TAG_TASK,
                tag_result=TAG_RESULT,
                tag_stop=TAG_STOP,
                on_complete=self._on_complete,
                on_dispatch=self._mark_dispatch,
                on_skip=self._finish_loc,
                set_resume_queue=lambda q: setattr(self, "_resume_queue", q),
                save_checkpoint=self.save_checkpoint,
                cleanup_completed=self._cleanup_completed,
                print_progress=self._print_progress,
                checkpoint_every=checkpoint_every,
            )

            self._resume_queue = []
            self.save_checkpoint()
            self._cleanup_completed()
            cleanup_wavefront_geometric_scratch(self, keep_incomplete_optim=False)
            self._print_progress(0, 0)

            results = self.sort_results()
            print_wavefront(
                "finished this scan "
                f"(angles={len(getattr(self, 'min_energies', {}) or getattr(self, 'min_bins', {}))}, "
                f"checkpoint={getattr(self, 'checkpoint', None)})"
            )
            return results

    def plot_wavefront(self,
                           pngfile: str='wavefront_workflow.png',
                           xmlfile: str='wavefront_workflow.xml') -> None:
            """  Plot the wavefront workflow.

            This function visualizes the wavefront algorithm by creating a grid where each row represents a level
            and each column represents an angle. The colors indicate the status of the nodes:
            
            - White: Not present
            - Orange: Present and active
            - Red: Present and inactive
            - Blue: Present and inactive but was active in the previous level
        
            Parameters
            ----------
            levels : list of WavefrontLevel
                The levels of the wavefront algorithm, each containing nodes with angles.
            nbins : int
                The number of bins in the grid
            filename : str, optional
                The name of the file to save the plot to. Default is 'wavefront_workflow.png'.
            
            Returns
            -------
            None
            This function does not return anything, but it saves the plot to a file.
            """
            import ndfes
            import copy
            from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT
            KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()

            # Determine all possible angles
            n_levels = len(self.levels)
            n_angles = len(self.bins)
            # 0: not present (white), 1: present & active (orange), 2: present & inactive (blue)
            grid = np.zeros((n_levels, n_angles), dtype=int)
            counts = np.zeros((n_levels, n_angles), dtype=int)

            for i, level in enumerate(self.levels):
                if i > 0:
                    for idx in range(len(grid[i,:])):
                        if grid[i-1,idx] == 2: # if red,  make orange
                            grid[i, idx] = 1
                        if grid[i-1, idx] == 3: # if blue, make blue
                            grid[i, idx] = 3
                        if grid[i-1, idx] == 1: # if orange, make blue
                            grid[i, idx] = 3
                idxs = sorted( self.bins.keys() )
                for node in level.nodes:
                    # Find the closest angle index
                    bidx = self.grid.GetBinIdx(node.rcs)
                    gidx = self.grid.CptGlbIdxFromBinIdx(bidx)
                    idx = idxs.index( gidx )
                    grid[i, idx] = 2  # red
                    counts[i, idx] += 1

            # Color map: 0=white, 1=orange, 2=red, 3=blue
            import matplotlib
            if not os.environ.get("MPLBACKEND"):
                matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            from matplotlib.colors import ListedColormap

            cmap = ListedColormap(['white', 'orange', 'red', 'dodgerblue'])

            plt.figure(figsize=(n_angles/2, n_levels/2))
            plt.imshow(grid, aspect='auto', cmap=cmap, origin='upper')
            plt.xlabel('Angle (deg)')
            plt.ylabel('Level')
            plt.xticks(np.arange(n_angles), np.arange(n_angles), rotation=90)
            plt.yticks(np.arange(n_levels), np.arange(1, n_levels+1))
            plt.title('Wavefront Workflow')

            # Add counts as text
            for i in range(n_levels):
                for j in range(n_angles):
                    if counts[i, j] > 0:
                        plt.text(j, i, str(counts[i, j]), va='center',
                                 ha='center', color='black', fontsize=8)

            from ffpopt.scan.WavefrontMixins import save_wavefront_figure

            save_wavefront_figure(pngfile)

            bins = copy.deepcopy(self.min_bins)
            for gidx in bins:
                bins[gidx].value = bins[gidx].energy * KCAL_PER_EV
                bins[gidx].stderr = 0
                bins[gidx].entropy = 1
                

            fes = ndfes.MBAR( self.grid, bins )
            ndfes.SaveXml(xmlfile,[fes])

    def _spawn_neighbors_nd(self, node: WavefrontNode) -> list:
            """ Spawn the two neighbor nodes of an active node in the next level.

            For a node at 180 degrees with delta 10, this creates nodes at 170 and
            190 degrees in level ``node.level + 1`` (created lazily) and returns them
            so the caller can enqueue them.

            Parameters
            ----------
            node : WavefrontNode
                The active node for which to spawn neighbors.

            Returns
            -------
            list of WavefrontNode
                The newly created neighbor nodes.

            Raises
            ------
            ValueError
                If the node is inactive, it cannot spawn neighbors.

            """
            if not node.active:
                raise ValueError("Cannot spawn neighbors for an inactive node.")
            next_level_id = node.level + 1
            bidx = self.grid.GetBinIdx(node.rcs)
            bins = GetGridNeighbors(bidx, self.grid, validbins=self.bins)
            nodes = []
            for newbin in bins:
                gidx = self.grid.CptGlbIdxFromBinIdx(newbin)
                rcs = self.bins[gidx].center
                child = self._enqueue_visit(
                    gidx,
                    struct=node.opt_geom,
                    seed_energy=node.energy,
                    level_id=next_level_id,
                    rcs=rcs,
                )
                if child is not None:
                    print_wavefront(
                        f"Spawn node {child.node_pkl} from {node.node_pkl}"
                    )
                    nodes.append(child)
            return nodes

    def _evaluate_node_nd(self, node: WavefrontNode) -> None:
            """Update per-bin minima and set ``node.active`` (spawn) via shared policy.

            See :func:`ffpopt.scan.WavefrontMixins.apply_wavefront_minimum_to_node`.
            """
            if not node.active:
                return
            bidx = self.grid.GetBinIdx(node.rcs)
            gidx = self.grid.CptGlbIdxFromBinIdx(bidx)

            has_incumbent = gidx in self.min_bins
            incumbent_energy = (
                self.min_bins[gidx].energy if has_incumbent else None
            )
            incumbent_soft = False
            if gidx in self.min_nodes:
                prev = self.min_nodes[gidx]
                incumbent_soft = bool(getattr(prev, "soft_opt", False))
                if not incumbent_soft and gidx in self.min_bins:
                    incumbent_soft = is_soft_opt_recovery(self.min_bins[gidx].struct)

            def on_update(n, _reason, _old):
                if gidx not in self.min_bins:
                    self.min_bins[gidx] = self.bins[gidx]
                self.min_bins[gidx].energy = n.energy
                self.min_bins[gidx].struct = n.opt_geom
                self.min_nodes[gidx] = n

            apply_wavefront_minimum_to_node(
                node,
                loc=node.rcs,
                threshold_kcal=self.convergence_threshold,
                has_incumbent=has_incumbent,
                incumbent_energy=incumbent_energy,
                incumbent_soft=incumbent_soft,
                on_update=on_update,
                noun="node",
                spawn_guard=self._spawn_guard,
            )

    def _init_calculation_nd(self) -> None:
            """ This initializes the wavefront calculation by setting up the first level and adding nodes to that level.
            
            This function is called at the beginning of the wavefront calculation to set up the initial conditions.
            It performs an initial geometry optimization to find the minimum geometry and sets up the first level with nodes.
            
            
            Returns
            -------
            None
                This function does not return anything, but it initializes the wavefront calculation.
                
            """

            if is_mpi_worker():
                return
            
            print_wavefront("starting wavefront calculation")
            #Add the initial level
            self.add_level()

            num_added = 0
            for iconf,s in enumerate(self.los):
                atoms = s.GetASEAtoms()
                pos = atoms.get_positions()
                rc = []
                if self.conlist is not None:
                    fcons = self.conlist.FillConstraints(atoms,force=True)
                    rc.extend( [ c.value for c in fcons ] )
                if self.reslist is not None:
                    for c in self.reslist:
                        v = c.GetCrdValue( pos )
                        rc.append(v)
                        
                bidx = self.grid.GetBinIdx(rc)
                oldrc = [x for x in rc]
                if None in bidx:
                    print_wavefront(
                        f"skipping initial structure {s.data['name']} "
                        f"(reaction coordinates {rc} out of bounds)"
                    )
                    continue
                rc   = self.grid.GetBinCenter(bidx)
                self.levels[0].add_node(
                    self.los,
                    s,
                    conlist=self.conlist,
                    reslist=self.reslist,
                    grid=self.grid,
                    rcs=rc,
                    node_id=s.data["name"],
                )
                num_added += 1
                print_wavefront(
                    f"init_node {bidx} {rc} {self.levels[0].nodes[-1].node_pkl}"
                )
            if num_added == 0:
                from mpi4py import MPI
                print_wavefront(
                    "failed to initialize the wavefront from the initial conformers"
                )
                MPI.COMM_WORLD.Abort()

    def _theory_change_nd(self, 
                          los: ListOfStruct,
                          stride: int = 1) -> None:
            """ This function is used to change the theory of the calculator for all nodes in the wavefront calculation.
            
            This is useful if you want to change the level of theory for a restarted calculation, for example from a lower level of theory to a higher level of theory. This will take the current level, and change the calculator for all nodes in that level.
            
            Parameters
            ----------
            los: ListOfStruct
                Builds calculators
            
            Returns
            -------
            None
                This function does not return anything, but it changes the calculator for all nodes in the current level.
            """
            if not self.levels:
                raise ValueError("Cannot change the theory of a wavefront calculation that has not been initialized.")
            
            self.levels = []

            if is_mpi_worker():
                return
            
            new_starting_level = WavefrontLevel(level_id=0)
            gidxs = list(self.min_bins.keys())
            gidxs.sort()
            for gidx in gidxs[::stride]:
                print_wavefront(
                    f"theory_change add node {gidx}, {self.min_bins[gidx].center}"
                )
                new_starting_level.add_node(
                    los,
                    self.min_bins[gidx].struct,
                    conlist=self.conlist,
                    reslist=self.reslist,
                    grid=self.grid,
                    rcs=self.min_bins[gidx].center,
                )
            self.min_bins = {}
            #self.min_energies = {}
            #self.min_structures = {}
            #self.min_nodes = {}
            self.levels.append(new_starting_level)
            print_wavefront("theory changed and new starting level added")

    def _print_progress_nd(self, pending: int, in_flight: int) -> None:
            """ Print a one-line live progress summary for the calculation queue.

            Replaces the old per-level banner: since several levels are in flight at
            once there is no clean level boundary to print at, so this is emitted at
            each checkpoint and at the end instead.

            Parameters
            ----------
            pending : int
                Number of nodes queued but not yet started.
            in_flight : int
                Number of nodes currently being optimized by workers.

            """
            total_num_rcs = len(self.bins)
            print_wavefront(
                format_wavefront_progress(
                    self,
                    pending,
                    in_flight,
                    extra=f"rcs={len(self.min_bins)}/{total_num_rcs}",
                )
            )

    def _rebuild_level_energies_nd(self) -> None:
            """ Rebuild ``self.level_energies`` as a per-level convergence history.

            Produces one ``{angle: energy}`` snapshot per level, where level ``k``
            holds every angle whose final minimum was reached at level ``k`` or
            earlier. This is derived deterministically from the recorded minima
            (independent of worker completion order); the final snapshot equals
            ``self.min_energies``. ``ffpopt-WavefrontAnimate.py`` consumes it.

            """
            max_level = max((level.level_id for level in self.levels), default=0)
            snapshots = []
            for k in range(1, max_level + 1):
                snapshots.append({gidx: sbin.energy
                                  for gidx, sbin in self.min_bins.items()
                                  if self.min_nodes[gidx].level <= k})
            self.level_energies = snapshots

    def _sort_results_nd(self) -> tuple[list[float], list[float], list[ase.Atoms]]:
            """Sort the results by angle."""
            from ffpopt.Struct import ListOfStruct
            
            #angles = sorted(self.min_energies.keys())
            #sorted_energies = [self.min_energies[angle] for angle in angles]
            #sorted_structures = [self.min_structures[angle] for angle in angles]

            gidxs = sorted(self.min_bins.keys())
            rcs   = [self.min_bins[gidx].center for gidx in gidxs]
            sorted_energies   = [self.min_bins[gidx].energy for gidx in gidxs]
            sorted_structures = [self.min_bins[gidx].struct for gidx in gidxs]

            ss = []
            for i in range(len(gidxs)):
                t = sorted_structures[i]
                t.data["name"] = "~".join( [ "%.2f"%(x) for x in rcs[i] ] )
                t.data["energy"] = sorted_energies[i]
                ss.append(t)
            
            #return angles, sorted_energies, sorted_structures
            return rcs, sorted_energies, ListOfStruct( ss )

    def sort_results(self) -> tuple[list[float], list[float], ListOfStruct]:
        if self.is_nd:
            return self._sort_results_nd()
        """Sort the results by angle."""
        from ffpopt.Struct import ListOfStruct
        angles = sorted(self.min_energies.keys())
        sorted_energies = [self.min_energies[angle] for angle in angles]
        sorted_structures = [self.min_structures[angle] for angle in angles]
        #print(f"angles={angles}")
        #print(f"sorted_energies={sorted_energies}")
        #print(f"sorted_structures={sorted_structures}")

        ss = []
        for i in range(len(angles)):
            t = sorted_structures[i]
            t.data["name"] = "d%03i"%(angles[i])
            t.data["energy"] = sorted_energies[i]
            ss.append( t )
        
        #return angles, sorted_energies, sorted_structures
        return angles, sorted_energies, ListOfStruct( ss )

    def print_summary(self) -> None:
        """Print a summary of the wavefront results."""
        print_wavefront("summary")
        print_wavefront(f"total levels: {len(self.levels)}")
        print_wavefront("number of nodes per level:")
        for i, level in enumerate(self.levels):
            print_wavefront(f"  level {i+1}: {len(level.nodes)} nodes")
        total = sum(len(level.nodes) for level in self.levels)
        print_wavefront(f"total nodes: {total}")
        if self.levels:
            print_wavefront(
                f"average nodes per level: {total / len(self.levels)}"
            )

        failed = []
        soft = []
        for level in self.levels:
            for node in level.nodes:
                if getattr(node, "error", None):
                    failed.append(node)
                elif getattr(node, "soft_opt", False):
                    soft.append(node)
                elif node.energy is None or not np.isfinite(node.energy):
                    failed.append(node)

        print_wavefront(f"failed nodes: {len(failed)}")
        if failed:
            for node in failed[:20]:
                print_wavefront(
                    f"  loc={node.rcs if getattr(node, 'is_nd', False) else node.angle} "
                    f"id={node.node_id} "
                    f"error={getattr(node, 'error', None)}"
                )
            if len(failed) > 20:
                print_wavefront(f"  ... and {len(failed) - 20} more")
        print_wavefront(f"soft-accepted nodes (no spawn): {len(soft)}")
        if failed:
            print_wavefront(
                f"finished with {len(failed)} failed node(s)"
            )
        else:
            print_wavefront(
                f"summary: no failed nodes ({len(soft)} soft-accepted)"
            )
    
    

def plot_wavefront(levels: list[WavefrontLevel], delta: int = 10, filename: str = 'wavefront_workflow.png') -> None:
    """  Plot the wavefront workflow.

    This function visualizes the wavefront algorithm by creating a grid where each row represents a level
    and each column represents an angle. The colors indicate the status of the nodes:
    
    - White: Not present
    - Orange: Present and active
    - Red: Present and inactive
    - Blue: Present and inactive but was active in the previous level
    
    Parameters
    ----------
    levels : list of WavefrontLevel
        The levels of the wavefront algorithm, each containing nodes with angles.
    delta : int, optional
        The step size for the angles in degrees. Default is 10.
    filename : str, optional
        The name of the file to save the plot to. Default is 'wavefront_workflow.png'.
        
    Returns
    -------
    None
        This function does not return anything, but it saves the plot to a file.
    
    """
    # Determine all possible angles
    angles = np.arange(0, 360, delta)
    n_levels = len(levels)
    n_angles = len(angles)
    # 0: not present (white), 1: present & active (orange), 2: present & inactive (blue)
    grid = np.zeros((n_levels, n_angles), dtype=int)
    counts = np.zeros((n_levels, n_angles), dtype=int)

    for i, level in enumerate(levels):
        if i > 0:
            for idx in range(len(grid[i,:])):
                if grid[i-1,idx] == 2: # if red,  make orange
                    grid[i, idx] = 1
                if grid[i-1, idx] == 3: # if blue, make blue
                    grid[i, idx] = 3
                if grid[i-1, idx] == 1: # if orange, make blue
                    grid[i, idx] = 3
        for node in level.nodes:
            # Find the closest angle index
            idx = int(round(node.angle / delta)) % (360 // delta)
            grid[i, idx] = 2  # red
            counts[i, idx] += 1

    # Color map: 0=white, 1=orange, 2=red, 3=blue
    import matplotlib
    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['white', 'orange', 'red', 'dodgerblue'])

    plt.figure(figsize=(n_angles/2, n_levels/2))
    plt.imshow(grid, aspect='auto', cmap=cmap, origin='upper')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Level')
    plt.xticks(np.arange(n_angles), angles, rotation=90)
    plt.yticks(np.arange(n_levels), np.arange(1, n_levels+1))
    plt.title('Wavefront Workflow')

    # Add counts as text
    for i in range(n_levels):
        for j in range(n_angles):
            if counts[i, j] > 0:
                plt.text(j, i, str(counts[i, j]), va='center', ha='center', color='black', fontsize=8)

    from ffpopt.scan.WavefrontMixins import save_wavefront_figure

    save_wavefront_figure(filename)

def find_adjacent_dihedrals(con: Constraint, los: ListOfStruct) -> tuple[list[int], list[int]]:
    """ Generates initial conformers based on the initial geometry optimization.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The atoms to optimize.
    con : Constraints
        The constraints to apply during optimization.
    stdargs
        The standard arguments for the optimization."""
    
    compare_dih = con.idxs
    first = compare_dih[:2]
    second = compare_dih[2:]
    chosen_first, chosen_second = None, None
    mol = los[0].GetParmedAtoms()
    
    for d in mol.dihedrals:
        if d.improper:
            continue
        idxs = [d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx]
        # Check that neither atom 2 nor atom 3 is a carbon atom with less than 4 bonds
        flag_value=False
        for atom_idx in [idxs[1], idxs[2]]:
            atom = mol.atoms[atom_idx]
            # Make sure that the atom is not a carbon with less than 4 bonds or a nitrogen with less than 3 bonds
            # if found, it will likely force a planar bond.
            if atom.atomic_number == 6 and len(atom.bonds) < 4:
                flag_value=True
            if atom.atomic_number == 7 and len(atom.bonds) < 3:
                flag_value=True
        if flag_value:
            continue
        if idxs[1] == first[0] and idxs[2] == first[1]:
            chosen_first = idxs
        elif idxs[2] == first[0] and idxs[1] == first[1]:
            chosen_first = idxs[::-1]
        elif idxs[1] == second[0] and idxs[2] == second[1]:
            chosen_second = idxs
        elif idxs[2] == second[0] and idxs[1] == second[1]:
            chosen_second = idxs[::-1]
        else:
            continue
        if chosen_first and chosen_second:
            break
    if chosen_first is None and chosen_second is None:
        print_wavefront("no adjacent dihedrals found for conformer generation")
    else:
        print_wavefront(f"adjacent dihedrals: {chosen_first} {chosen_second}")
    return chosen_first, chosen_second

def wavefront_loader(filename: str) -> Wavefront:
    """Load a Wavefront object from a pickle file (see ``wavefront_mixins``)."""
    return load_wavefront_pickle(filename, restore_soft_opt=True)


def run_dihed_wavefront(
    *,
    inp: str,
    out: str,
    dihed: str,
    delta: int = 10,
    nproc: int = 1,
    wf_max_levels: int = -1,
    wf_starting_nodes: int = 1,
    wf_num_conformers: int = 0,
    wf_change_theory: bool = False,
    wf_theory_stride: int = 1,
    wf_alt_starting_checkpoint: Optional[str] = None,
    wf_convergence_threshold: float = 0.01,
    **standard_kwargs,
) -> dict:
    """Run a relaxed dihedral wavefront scan from Python kwargs.

    Required: ``inp`` (input json), ``out`` (output json), ``dihed`` (e.g.
    ``"4,5,6,7"``). All other wavefront-specific options match the
    ``ffpopt-DihedWavefront.py`` CLI flags with underscores.

    ``standard_kwargs`` accepts anything declared by
    ``Options.AddStandardOptions`` (``model``, ``mfile``, ``no_opt``,
    ``geometric_opt``, ``ase_opt_tol``, ``geometric_maxiter``, ``cpu``, ...).
    Anything not supplied falls back to the same default the CLI would use.

    Side effects mirror the CLI:
      * writes ``out`` (json), ``out`` with ``.dat`` / ``.pkl`` suffixes,
        and a ``wf_workflow_<name>.png`` plot;
      * checkpoints under ``checkpoint_<out>.pkl``;
      * prints progress to stdout.

    Returns
    -------
    dict
        ``{'angles', 'energies', 'energies_noshift'}``.
        ``energies`` is min-shifted (kcal/mol); ``energies_noshift`` is the raw
        kcal/mol energies before shifting. The wavefront object and
        ``ListOfStruct`` frames stay on disk (``.pkl`` / ``.json``) so bond-pool
        IPC does not pickle them.
    """
    from types import SimpleNamespace
    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

    require_main_guard_for_spawn()
    std = merge_standard_wavefront_kwargs(standard_kwargs)

    if std.get("soft_dihed_restraint"):
        k = std.get("soft_dihed_k", 500.0)
        kmax = std.get("soft_dihed_kmax", 8000.0)
        tol = std.get("soft_dihed_tol", 0.5)
        print(
            f"[affdo] wavefront soft harmonic dihedral: k={k:g} kcal/mol/rad^2 "
            f"kmax={kmax:g} (double until in-band, then hard IC from last coords) "
            f"tol={tol:g} deg dihed={dihed}",
            flush=True,
        )

    args = SimpleNamespace(
        inp=inp,
        out=out,
        dihed=dihed,
        delta=delta,
        nproc=nproc,
        wf_max_levels=wf_max_levels,
        wf_starting_nodes=wf_starting_nodes,
        wf_num_conformers=wf_num_conformers,
        wf_change_theory=wf_change_theory,
        wf_theory_stride=wf_theory_stride,
        wf_alt_starting_checkpoint=wf_alt_starting_checkpoint,
        wf_convergence_threshold=wf_convergence_threshold,
        **std,
    )

    inps = ListOfStruct.from_file(args.inp)
    inps.SetArgs(args)
    inps.structs = [inps.structs[0]]

    con = Constraint.from_str(args.dihed, graph=inps[0].GetGraph())
    con.value = None

    idel = None
    for i in range(len(inps[0].data["constraints"])):
        x = inps[0].data["constraints"][i]
        c = Constraint.from_dict(x)
        if c.is_same(con):
            idel = i
            break
    if idel is not None:
        del inps.structs[0].data["constraints"][idel]

    if args.delta < 1:
        raise ValueError("Delta must be at least 1 degree")

    if args.wf_max_levels > 0:
        minlevels = np.round((360 // args.delta) / 2, 0)
        if args.wf_max_levels < minlevels:
            args.wf_max_levels = int(minlevels)

    out_path = Path(args.out)
    # Keep checkpoint / plot beside ``out`` so absolute ``out`` paths work
    # without relying on process cwd (fragmented workflows).
    checkpoint_path = out_path.parent / f"checkpoint_{out_path.with_suffix('.pkl').name}"

    if args.wf_alt_starting_checkpoint:
        starting_checkpoint_path = Path(args.wf_alt_starting_checkpoint)
    else:
        starting_checkpoint_path = checkpoint_path

    if starting_checkpoint_path.exists():
        print_wavefront(
            f"checkpoint {starting_checkpoint_path} exists; loading previous run"
        )
        wf_run = pickle_load_compat(starting_checkpoint_path)
        wf_run.restart_options(
            inps,
            max_levels=int(args.wf_max_levels),
            nproc=args.nproc,
            checkpoint=checkpoint_path,
        )
        if not isinstance(wf_run, Wavefront):
            raise TypeError("Checkpoint file does not contain a valid Wavefront object.")
        print_wavefront("wavefront run loaded successfully")
    else:
        print_wavefront(
            f"no checkpoint file found at {starting_checkpoint_path}; "
            "starting a new wavefront run"
        )
        wf_run = Wavefront(
            inps, inps[0], con,
            delta=args.delta,
            max_levels=int(args.wf_max_levels),
            nproc=args.nproc,
            starting_nodes=args.wf_starting_nodes,
            num_conformers=args.wf_num_conformers,
            checkpoint=checkpoint_path,
            convergence_threshold=args.wf_convergence_threshold,
        )

    if args.wf_change_theory:
        print_wavefront("changing theory of the calculator for all nodes")
        wf_run.theory_change(inps, stride=args.wf_theory_stride)

    wf_run.calculate()
    wf_run.print_summary()
    angles, energies, structures = wf_run.sort_results()

    if not angles:
        from ffpopt.scan.WavefrontMixins import empty_scan_error_message

        raise RuntimeError(empty_scan_error_message(wf_run, args.out))

    KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    energies_noshift = [e * KCAL_PER_EV for e in energies]
    energies = np.asarray(energies_noshift, dtype=float) - np.amin(energies_noshift)

    structures.save(args.out)

    for i, angle in enumerate(angles):
        print_wavefront(f"angle={angle} energy={energies[i]}")
    print_wavefront(f"finished this scan -> {args.out}")

    dat = out_path.with_suffix(".dat")
    with open(dat, "w") as fh:
        for a, e, n in zip(angles, energies, energies_noshift):
            fh.write(f"{a} {e} {n}\n")
    print_wavefront(f"data written to {dat}")

    pkl_path = out_path.with_suffix(".pkl")
    atomic_pickle_dump(wf_run, pkl_path)
    print_wavefront(f"wavefront run saved to {pkl_path}")
    wf_fname = str(out_path.parent / f"wf_workflow_{out_path.with_suffix('.png').name}")
    plot_wavefront(wf_run.levels, delta=args.delta, filename=wf_fname)
    wf_run.print_summary()
    print_wavefront(f"wavefront plot saved as {wf_fname}")
    del wf_run, structures

    return {
        "angles": angles,
        "energies": energies,
        "energies_noshift": energies_noshift,
    }


def run_ndim_dihed_wavefront(
    *,
    inp: str,
    out: str,
    condim: list,
    resdim: list,
    nproc: int = 1,
    mpi: bool = False,
    wf_max_levels: int = -1,
    wf_change_theory: bool = False,
    wf_theory_stride: int = 1,
    wf_alt_starting_checkpoint: Optional[str] = None,
    wf_convergence_threshold: float = 0.01,
    **standard_kwargs
) -> dict:
    """Run a relaxed N-D wavefront scan from Python kwargs.

    Required: ``inp`` (input json), ``out`` (output json), ``condim`` (e.g.
    ``[xlo,xhi,nbins]``). Signature matches ``ffpopt-NDimWavefront.py``.
    """
    from types import SimpleNamespace
    from ffpopt.Options import AddConstraintAndRestraintOptions
    from ffpopt.Options import ParseConstraintAndRestraintOptions
    from ffpopt.Options import DeleteConstraintAndRestraintFromStruct
    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT
    from ndfes import VirtualGrid, SpatialDim

    require_main_guard_for_spawn()
    std = merge_standard_wavefront_kwargs(
        standard_kwargs, extra_adders=(AddConstraintAndRestraintOptions,)
    )

    args = SimpleNamespace(
        inp=inp,
        out=out,
        condim=condim,
        resdim=resdim,
        nproc=nproc,
        mpi=mpi,
        wf_max_levels=wf_max_levels,
        wf_change_theory=wf_change_theory,
        wf_theory_stride=wf_theory_stride,
        wf_alt_starting_checkpoint=wf_alt_starting_checkpoint,
        wf_convergence_threshold=wf_convergence_threshold,
        **std,
    )

    los = ListOfStruct.from_file(args.inp)
    los.SetArgs(args)

    conlist, reslist = ParseConstraintAndRestraintOptions(args)
    for s in los:
        DeleteConstraintAndRestraintFromStruct(s, conlist, reslist)

    if len(conlist) != len(condim):
        raise Exception(
            f"Constraint dimension mismatch {len(conlist)} {len(condim)}"
        )
    if len(reslist) != len(resdim):
        raise Exception(
            f"Restraint dimension mismatch {len(reslist)} {len(resdim)}"
        )

    dims = []
    maxsize = 0
    for i in range(len(condim)):
        isper = conlist[i].isper()
        cs = condim[i].split(",")
        if len(cs) != 3:
            raise Exception(f"Expected 2 floats and an int, but found {condim[i]}")
        xlo = float(cs[0])
        xhi = float(cs[1])
        size = int(cs[2])
        maxsize = max(maxsize, size)
        if isper and int(round(xhi - xlo)) != 360:
            isper = False
        dims.append(SpatialDim(xlo, xhi, size, isper))
    for i in range(len(resdim)):
        isper = reslist[i].isper()
        cs = resdim[i].split(",")
        if len(cs) != 3:
            raise Exception(f"Expected 2 floats and an int, but found {resdim[i]}")
        xlo = float(cs[0])
        xhi = float(cs[1])
        size = int(cs[2])
        maxsize = max(maxsize, size)
        if isper and int(round(xhi - xlo)) != 360:
            isper = False
        dims.append(SpatialDim(xlo, xhi, size, isper))

    grid = VirtualGrid(dims)

    if args.wf_max_levels > 0:
        minlevels = np.round(maxsize, 0)
        if args.wf_max_levels < minlevels:
            args.wf_max_levels = int(minlevels)

    checkpoint_path = Path(args.out).resolve().parent / (
        f"checkpoint_{Path(args.out).resolve().with_suffix('.pkl').name}"
    )

    if args.wf_alt_starting_checkpoint:
        starting_checkpoint_path = Path(args.wf_alt_starting_checkpoint)
    else:
        starting_checkpoint_path = checkpoint_path

    is_worker = is_mpi_worker()

    if starting_checkpoint_path.exists():
        if not is_worker:
            print_wavefront(
                f"checkpoint {starting_checkpoint_path} exists; loading previous run"
            )
        wf_run = pickle_load_compat(starting_checkpoint_path)
        wf_run.restart_options(
            los,
            max_levels=int(args.wf_max_levels),
            nproc=args.nproc,
            use_mpi=args.mpi,
            checkpoint=checkpoint_path,
        )
        if not isinstance(wf_run, Wavefront):
            raise TypeError("Checkpoint file does not contain a valid Wavefront object.")
        if not is_worker:
            print_wavefront("wavefront run loaded successfully")
    else:
        if not is_worker:
            print_wavefront(
                f"no checkpoint file found at {starting_checkpoint_path}; "
                "starting a new wavefront run"
            )
        wf_run = Wavefront(
            los,
            conlist=conlist,
            reslist=reslist,
            grid=grid,
            max_levels=int(args.wf_max_levels),
            nproc=args.nproc,
            use_mpi=args.mpi,
            checkpoint=checkpoint_path,
            convergence_threshold=args.wf_convergence_threshold,
        )

    if args.wf_change_theory:
        if not is_worker:
            print_wavefront("changing theory of the calculator for all nodes")
        wf_run.theory_change(los, stride=args.wf_theory_stride)

    wf_run.calculate()

    if is_worker:
        return None

    rcs, energies, structures = wf_run.sort_results()

    if not rcs:
        from ffpopt.scan.WavefrontMixins import empty_scan_error_message

        raise RuntimeError(empty_scan_error_message(wf_run, args.out))

    KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    energies_noshift = [e * KCAL_PER_EV for e in energies]
    if len(energies_noshift) > 0:
        emin = np.amin(energies_noshift)
        energies = energies_noshift - emin
    else:
        energies = np.array([])

    structures.save(args.out)

    for i, loc in enumerate(rcs):
        print_wavefront(f"angle={loc} energy={energies[i]}")
    print_wavefront(f"finished this scan -> {args.out}")

    out_path = Path(args.out).resolve()
    dat = out_path.with_suffix(".dat")
    with open(dat, "w") as fh:
        for a, e, n in zip(rcs, energies, energies_noshift):
            fh.write(f"{a} {e} {n}\n")
    print_wavefront(f"data written to {dat}")

    pkl_path = out_path.with_suffix(".pkl")
    atomic_pickle_dump(wf_run, pkl_path)
    print_wavefront(f"wavefront run saved to {pkl_path}")
    wf_pngfile = str(out_path.parent / f"wf_workflow_{out_path.with_suffix('.png').name}")
    wf_xmlfile = str(Path(wf_pngfile).with_suffix(".xml"))
    wf_run.plot_wavefront(pngfile=wf_pngfile, xmlfile=wf_xmlfile)
    wf_run.print_summary()
    print_wavefront(f"wavefront plot saved as {wf_pngfile}")
    print_wavefront(f"energies saved as {wf_xmlfile}")
    del wf_run, structures

    return {
        "rcs": rcs,
        "energies": energies,
        "energies_noshift": energies_noshift,
    }


