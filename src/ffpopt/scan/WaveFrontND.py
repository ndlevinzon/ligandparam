#!/usr/bin/env python3
from __future__ import annotations

#
# To launch this with MPI on amarel,
#    mpirun -n $SLURM_NTASKS python3 -m mpi4py `which ffpopt-NDimWavefront.py`
#
# In principle, you should be able to submit this with
#    srun --mpi=pmi2 python3 -m mpi4py `which ffpopt-NDimWavefront.py`
# or
#    srun --mpi=pmix python3 -m mpi4py `which ffpopt-NDimWavefront.py`
# but on amarel this ends up launching multiple serial copies of the script
# without using mpi.

import copy
import os
import sys
import pickle
import numpy as np
from pathlib import Path

from typing import Generator, Optional

from ffpopt.Struct import ListOfStruct, Struct
from ffpopt.GeomOpt import GeomOpt, bare_potential_energy, is_soft_opt_recovery, opt_recovery_label
from ffpopt.Constraints import ConstraintList
from ffpopt.Restraints import RestraintList


# Per-process worker state (Pool initializer / MPI bcast).
_WORKER: dict = {}


def is_mpi_worker():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return size > 1 and rank > 0


def GetGridNeighbors(bidx, grid, validbins=None):
    from ndfes.GridUtils import LinearPtsToMeshPts
    lol = []
    for idim in range(len(grid.dims)):
        if grid.dims[idim].isper:
            ilo = bidx[idim]-1
            ihi = bidx[idim]+2
            lol.append( [i % grid.dims[idim].size for i in range(ilo,ihi) ] )
        else:
            ilo = max( bidx[idim]-1, 0 )
            ihi = min( bidx[idim]+2, grid.dims[idim].size )
            lol.append( [i for i in range(ilo,ihi) ] )
    pts = LinearPtsToMeshPts(lol)
    newpts = []
    for pt in pts:
        b = [int(round(x)) for x in pt]
        if b != bidx:
            if validbins is not None:
                gidx = grid.CptGlbIdxFromBinIdx(b)
                if gidx in validbins:
                    newpts.append(b)
            else:
                newpts.append(b)
    return newpts


def _clear_los_calc(los: ListOfStruct) -> None:
    from .wavefront_mixins import clear_los_calc

    clear_los_calc(los)


def _init_worker(los, conlist, reslist, template_struct) -> None:
    """Pool/MPI initializer: share los + templates once per worker."""
    _clear_los_calc(los)
    _WORKER["los"] = los
    _WORKER["conlist"] = copy.deepcopy(conlist)
    _WORKER["reslist"] = copy.deepcopy(reslist)
    _WORKER["template"] = copy.deepcopy(template_struct)


def _clone_struct_geometry(struct, coords, ene=0.0, frcs=None):
    """Prefer ``Struct.clone_geometry``; fall back to deepcopy for test doubles."""
    from .wavefront_mixins import clone_struct_geometry

    return clone_struct_geometry(struct, coords, ene=ene, frcs=frcs)


def _struct_from_coords(coords) -> Struct:
    return _clone_struct_geometry(
        _WORKER["template"], coords, ene=0.0, frcs=None
    )


def _run_node_job(job: dict) -> dict:
    """Worker entry: slim rcs/coords job in -> slim result out (no ``los``)."""
    node = WavefrontNode.from_job(
        job,
        _WORKER["los"],
        _WORKER["conlist"],
        _WORKER["reslist"],
    )
    node.calculate()
    return node.to_result()


def _run_node(node: "WavefrontNode") -> "WavefrontNode":
    """In-process entry (serial path); mutates and returns ``node``."""
    node.calculate()
    return node


class WavefrontNode(object):
    """ This is a node in the wavefront algorithm. It represents a single geometry optimization.

    It contains the atoms, the energy, the angle, and the constraints.

    Parameters
    ----------
    los : ListOfStruct
        The objet used to store input structures and build calculators
    struct
        The structure to modify
    conlist : ConstraintList
        The list of constraints to apply to the optimization.
        If there are 2 constraints, these correspond to the first 2 reaction coordinates
    rcs : list of float, optional
        The angle to optimize. If not provided, it will be set to the initial angle of
        the constraint.
    level : int, optional
        The level of the node in the wavefront algorithm. Default is None.
    
    Attributes
    ----------
    atoms : ase.Atoms
        The atoms to optimize.
    energy : float
        The energy of the optimized geometry.
    rcs : list of float
        The angle to optimize.
    active : bool
        Whether the node is active (i.e., whether it should be optimized).
    conlist : ListOfConstraint
        The constraints to apply to the optimization.
    reslist : ListOfConstraint
        The restraints to apply to the optimization.
    opt_geom : GeomOpt
        The optimized geometry.
    stdargs
        The standard arguments for the optimization.
    level : int
        The level of the node in the wavefront algorithm.
    
    """
    def __init__(self,
                 los: ListOfStruct,
                 struct: Struct,
                 conlist: ConstraintList,
                 reslist: RestraintList,
                 rcs: list[float],
                 level: int = None,
                 node_id: str = None) -> None:
        self.los = los
        self.struct = struct
        self.energy = None
        self.forces = np.zeros((len(struct.data["elements"]), 3))
        self.rcs = rcs
        self.active = True
        self.conlist = copy.deepcopy(conlist)
        self.reslist = copy.deepcopy(reslist)

        self.opt_geom = None
        self.level = level
        self.node_id = node_id
        
        self.node_pkl = self.get_pkl_name()
        self.complete = False
        self.error = None
        self.soft_opt = False
        self.opt_recovery = None
        self._assign_rcs()
        

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
        """Return the name of the pckl file

        Parameters
        ----------
        None

        Returns
        -------
        str
            The name of the pckl file
        """
        s = "~".join( [ "%.2f"%(x) for x in self.rcs ] )
        return f"level_{self.level}_rcs_{s}_id_{self.node_id}_node.pckl"

    @classmethod
    def from_job(cls, job: dict, los, conlist, reslist) -> "WavefrontNode":
        """Rebuild a node from a slim IPC job (rcs + coords; no pickled ``los``)."""
        if job.get("coords") is not None:
            struct = _struct_from_coords(job["coords"])
        else:
            struct = job["struct"]
        node = cls(
            los=los,
            struct=struct,
            conlist=conlist,
            reslist=reslist,
            rcs=list(job["rcs"]),
            level=job["level"],
            node_id=job["node_id"],
        )
        node.node_pkl = job["node_pkl"]
        node.complete = bool(job.get("complete", False))
        return node

    def to_job(self) -> dict:
        """Slim payload for spawn/MPI workers: rcs + coords (not ``los``)."""
        coords = np.asarray(self.struct.data["positions"], dtype=float)
        return {
            "rcs": list(self.rcs),
            "coords": coords,
            "level": self.level,
            "node_id": self.node_id,
            "node_pkl": self.node_pkl,
            "complete": self.complete,
        }

    def to_result(self) -> dict:
        """Slim result: energy + optimized coords (not ``los`` / full node)."""
        from .wavefront_mixins import slim_node_result

        return slim_node_result(self)

    def apply_result(self, result: dict) -> None:
        """Merge a slim worker result into this parent-side node."""
        from .wavefront_mixins import apply_slim_node_result

        apply_slim_node_result(self, result, clone_fn=_clone_struct_geometry)

    def _ensure_soft_opt_attrs(self) -> None:
        """Fill soft-opt fields missing from older node pickles / checkpoints."""
        from .wavefront_mixins import ensure_soft_opt_attrs

        ensure_soft_opt_attrs(self)

    def replace_with_pickle(self) -> None:
        """Replace node fields from a sidecar pickle if present (restores ``los``)."""
        filename = Path(f"{self.node_pkl}")
        if filename.is_file():
            print("EXISTING NODE PICKLE", self.node_pkl)
            los = self.los
            from .wavefront_mixins import pickle_load_compat

            loaded_node = pickle_load_compat(filename)
            self.__dict__.update(loaded_node.__dict__)
            if self.los is None:
                self.los = los
            self._ensure_soft_opt_attrs()
            print("Node data replaced with pickle data.")

    def calculate(self) -> None:
        """Calculate the energy of the atoms."""
        if not self.complete:

            #print("calculate node ",self.node_pkl)
            
            #self.constraints[0].value = self.angle
            ncon = len(self.conlist)
            for ic in range(ncon):
                self.conlist.cons[ic].value = self.rcs[ic]
            nres = len(self.reslist)
            for ic in range(nres):
                self.reslist.rests[ic].value = self.rcs[ncon+ic]
            precheck_err = self._precheck_geometry()
            if precheck_err is not None:
                self._mark_failed(precheck_err)
                return
            try:
                cons = None
                rest = None
                if self.conlist is not None:
                    cons = self.conlist.cons
                if self.reslist is not None:
                    rest = self.reslist.rests

                self.opt_geom = GeomOpt(self.los, self.struct, constraints=cons, restraints=rest)
                # Opt energy already includes a final SCF; strip restraint
                # penalties analytically (legacy path re-ran SinglePoint bare).
                self.opt_recovery = opt_recovery_label(self.opt_geom)
                self.soft_opt = is_soft_opt_recovery(self.opt_geom)
                if self.soft_opt:
                    print(
                        f"Node {self.node_id} soft-accepted opt "
                        f"(recovery={self.opt_recovery}); will not spawn neighbors"
                    )
                self.energy = np.round(bare_potential_energy(self.opt_geom), 6)
                self.forces = self.opt_geom.data.get("forces", self.forces)
                from .wavefront_mixins import maybe_write_success_checkpoint

                maybe_write_success_checkpoint(self)
                self.complete = True
                
            except Exception as e:
                print(f"Node {self.node_id} optimization error: {type(e).__name__}: {e}")
                self._mark_failed("optimization_error", e)


    def _write_checkpoint(self) -> None:
        """Write the node's data to a pickle file (without ``los``)."""
        from .wavefront_mixins import write_node_pickle

        write_node_pickle(self, verbose=True)

    def cleanup(self) -> None:
        """Clean up the node's pickle file."""
        filename = Path(f"{self.node_pkl}")
        if filename.is_file():
            try:
                print(f"Remove node {self.node_pkl}")
                os.remove(filename)
            except Exception:
                print(f"Failed to remove {self.node_pkl} because it disappeared")

    def _mark_failed(self, reason: str, error: Optional[Exception] = None) -> None:
        from .wavefront_mixins import mark_node_failed

        mark_node_failed(self, reason, error, where=self.rcs)

    def _precheck_geometry(self, min_dist: float = 0.8) -> Optional[str]:
        """Return a failure reason, or ``None`` if the geometry looks usable."""
        try:
            from ffpopt.Constraints import ApplyConstraints, has_nonbonded_clash
            myatoms = self.struct.GetASEAtoms()
            myatoms = ApplyConstraints(
                myatoms, self.conlist.cons, graph=self.struct.GetGraph()
            )
            clashed, i, j, dist = has_nonbonded_clash(
                myatoms.get_positions(), self.struct.data["bonds"], min_dist=min_dist
            )
            if clashed:
                print(f"Precheck clash: atom {i} and atom {j} at {dist:.3f} Ang (< {min_dist} Ang)")
                return "clash_precheck"
        except Exception as e:
            print(f"Precheck failed due to error: {e}")
            return f"precheck_error: {e}"
        return None

    
class WavefrontLevel(object):
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

    def add_node(self, #atoms: ase.Atoms, stdargs: StandardArgs,
                 los: ListOfStruct,
                 struct: Struct,
                 conlist: ConstraintList,
                 reslist: RestraintList,
                 grid: ndfes.VirtualGrid,
                 rcs: list[float] = None,
                 node_id = None):
        """Add a node to the level.
        
        This creates a new WavefrontNode with the given atoms, standard arguments, constraint, and angle.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms to optimize.
        stdargs
            The standard arguments for the optimization.
        conlist : ConstraintList
            The constraints to apply to the optimization.
        reslist : RestraintList
            The restraints to apply to the optimization.
        grid : ndfes.VirtualGrid
        rcs : float, optional
            The reaction coordinates to optimize. If not provided, it will be set to the initial angle of the constraint.
        
        Returns
        -------
        None
        
        """
        from ffpopt.Constraints import FillConstraints
        #node_id = len(self.nodes)
        if node_id is None:
            #crds = struct.get_positions()
            atoms = struct.GetASEAtoms()
            crds = atoms.get_positions()
            conrcs = FillConstraints( atoms, conlist.cons, force=True )
            conrcs = [ c.value for c in conrcs ]
            resrcs = [ restraint.GetCrdValue(crds) for restraint in reslist ]
            inprcs = conrcs + resrcs
            bidx = grid.GetBinIdx(inprcs)
            gidx = grid.CptGlbIdxFromBinIdx(bidx)
            node_id = f"{len(self.nodes)}_{gidx}"

        names = [ node.node_pkl for node in self.nodes ]
        
        node = WavefrontNode(los=los,struct=struct,
                             conlist=conlist,
                             reslist=reslist,
                             rcs=rcs,
                             level=self.level_id,
                             node_id=node_id)

        if node.node_pkl in names:
            print("NODE ALREADY IN THE NODE LIST {node.node_pkl}")
        
        self.nodes.append(node)
        return node

    # def check_node_checkpoints(self) -> None:
    #     """Check for existing checkpoints for all nodes in the level."""
    #     for node in self.nodes:
    #         node.replace_with_pickle()


class Wavefront(object):
    """ This class implements the wavefront algorithm for dihedral scans.
    
    It manages the levels, nodes, and the optimization process.
    
    Parameters
    ----------
    los : ListOfStruct
        Constructs calculators and stores the list of starting structures
    conlist : ConstraintList
        The constraint definitions to apply to the optimization.
    reslist : RestraintList
        The restraint definitions to apply to the optimization.
    grid : ndfes.VirtualGrid
        The grid to fill
    max_levels : int, optional
        The maximum number of levels to explore in the wavefront. Default is 1.
    nproc : int, optional
        The number of optimizations to run at a time. Default is 1.
    convergence_threshold : float, optional
        Energy convergence threshold (kcal/mol) for wavefront calculation. A
        revisited bin must lower its running minimum by at least this much to
        stay active and spawn another level. Default is 0.01.
    use_mpi : bool, optional, default=False
        If true, then mpi4pi is used rather than single-node threading
    
    Attributes
    ----------
    los : ListOfStruct
        Constructs calculators and stores the list of starting structures
    conlist : ConstraintList
        The constraint definitions to apply to the optimization.
    reslist : RestraintList
        The restraint definitions to apply to the optimization.
    grid : ndfes.VirtualGrid
        The grid to fill
    bins : dict of ndfes.SpatialBin
        The keys are integers (global bin index)
    
    min_geom : GeomOpt
        The initial geometry optimization.
    min_geom_ang : float
        The initial dihedral angle.
    
    levels : list of WavefrontLevel
        The levels in the wavefront algorithm.
    min_energies : dict
        A dictionary mapping bin to their minimum energies.
    min_structures : dict
        A dictionary mapping bin to their minimum structures.
    max_levels : int
        The maximum number of levels to explore in the wavefront.
    nproc : int
        The number of optimizations to run at a time.
    use_mpi : bool
        Whether to use mpi or threading
    
    """
    def __init__(self,
                 los: ListOfStruct,
                 conlist: ConstraintList,
                 reslist: RestraintList,
                 grid: ndfes.VirtualGrid,
                 max_levels: int = 1,
                 nproc: int = 1,
                 checkpoint: str = "wavefront_checkpoint.pkl",
                 convergence_threshold: float = 0.01,
                 use_mpi: bool = False ) -> None:

        from ndfes import SpatialBin
        
        self.los     = copy.deepcopy(los)
        self.conlist = copy.deepcopy(conlist)
        self.reslist = copy.deepcopy(reslist)
        self.grid    = copy.deepcopy(grid)
        self.bins    = {}
        self.use_mpi = use_mpi
        allrcs       = self.grid.GetRegGridCenterPts()
        for rcs in allrcs:
            bidx = self.grid.GetBinIdx(rcs)

            gidx = self.grid.CptGlbIdxFromBinIdx(bidx)
            self.bins[gidx] = SpatialBin(bidx)
            self.bins[gidx].center = self.grid.GetBinCenter(bidx)

            
        self.min_bins = {}
        self.min_nodes = {}

        #self.min_geom = None
        #self.min_geom_ang = None
        #self.extra_dih = extra_dih
        self.levels = []
        #self.min_energies = {}
        #self.min_structures = {}
        self.max_levels = max_levels
        self.nproc = nproc
        
        #self.starting_nodes = starting_nodes 
        #self.num_conformers = num_conformers
        self.level_energies = []
        self.checkpoint = checkpoint
        self.restarted = []
        self.verbose = False
        self.convergence_threshold = convergence_threshold
        # Nodes created but not yet completed. Persisted in the checkpoint so a
        # restart re-enqueues the in-flight/pending work (see calculate).
        self._resume_queue = None

    def restart_options(self,
                        los: ListOfStruct,
                        max_levels: int = -1,
                        nproc: int = 1,
                        use_mpi: bool = False,
                        checkpoint: str = None) -> None:
        """ This function is used to set the options for a restarted wavefront calculation.

        This lets you use slightly different options for a restarted calculation, such as changing the number of processors or the maximum number of levels.

        Parameters
        ----------
        los : ListOfStruct
            Builds calculators
        max_levels : int, optional
            The maximum number of levels to explore in the wavefront. Default is -1, which
            means unlimited levels.
        nproc : int, optional
            The number of optimizations to run at a time. Default is 1.
        use_mpi : bool, optional, default=False
            Use MPI rather than threads
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
        self.los = copy.deepcopy(los)

        if checkpoint is not None:
            self.checkpoint = checkpoint

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
        print("Number of times restarted:", len(self.restarted))
        self.restarted.append(prev_options)
        
        print("Restart options updated.")

    def theory_change(self, 
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
            print(f"theory_change add node {gidx}, {self.min_bins[gidx].center}")
            new_starting_level.add_node(los, self.min_bins[gidx].struct,
                                        self.conlist, self.reslist, self.grid,
                                        rcs=self.min_bins[gidx].center)
        self.min_bins = {}
        #self.min_energies = {}
        #self.min_structures = {}
        #self.min_nodes = {}
        self.levels.append(new_starting_level)
        print("Theory changed and new starting level added.")





    def init_calculation(self) -> None:
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
        
        print("Starting wavefront calculation...")
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
                print(f"Skipping initial structure {s.data['name']} because reaction coordinates {rc} are out of bounds")
                continue
            rc   = self.grid.GetBinCenter(bidx)
            self.levels[0].add_node(self.los, s, self.conlist, self.reslist, self.grid, rc, node_id=s.data["name"])
            num_added += 1
            print(f"init_node {bidx} {rc} {self.levels[0].nodes[-1].node_pkl}")
        if num_added == 0:
            from mpi4py import MPI
            print("Failed to initialize the wavefront method from the initial conformers")
            MPI.COMM_WORLD.Abort()
        


    @staticmethod
    def nearest_angle( rcs: list[float]) -> float:
        """ Round a reaction coordinate to the nearest bin center.
        
        Parameters
        ----------
        rcs : list of float
            The reaction coordinates
        
        Returns
        -------
        newrcs : list of float
            The bin center            
        """
        return self.grid.GetBinCenter(self.grid.GetBinIdx(rc))


    
    def calculate(self) -> None:
        if self.use_mpi:
            self.calculate_mpi()
        else:
            self.calculate_threads()

    
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
        import time
        from collections import deque

        # Seed the queue: a fresh run initializes level 1; a restart re-enqueues
        # the work the checkpoint recorded as pending/in-flight (falling back to
        # any active, incomplete node for checkpoints predating _resume_queue).

        print("Entered calculate_threads")
        
        if not self.levels:
            self.init_calculation()
            pending = deque(self.levels[0].nodes)
        elif self._resume_queue:
            pending = deque(self._resume_queue)
        else:
            pending = deque(node for level in self.levels
                            for node in level.nodes
                            if node.active and not node.complete)

        self._resume_queue = list(pending)
        self.save_checkpoint()

        pool = None
        if self.nproc > 1:
            ctx = multiprocessing.get_context("spawn")
            template = self.los.structs[0] if getattr(self.los, "structs", None) else None
            if template is None:
                # Fall back: any completed node's struct, else first pending seed.
                for level in self.levels:
                    for n in level.nodes:
                        template = n.struct
                        break
                    if template is not None:
                        break
            pool = ctx.Pool(
                processes=self.nproc,
                initializer=_init_worker,
                initargs=(self.los, self.conlist, self.reslist, template),
            )

        from ffpopt.runtime.fast_wavefront import wf_checkpoint_every

        checkpoint_every = wf_checkpoint_every(self.nproc)
        try:
            in_flight = {}
            since_checkpoint = 0
            while pending or in_flight:
                # Top up the pool (or run one node inline when serial).
                while pending and len(in_flight) < self.nproc:
                    node = pending.popleft()
                    node.replace_with_pickle()
                    if node.complete:
                        pending.extend(self._on_complete(node))
                        continue
                    if not node.active:
                        continue
                    if pool is None:
                        node.calculate()
                        pending.extend(self._on_complete(node))
                        since_checkpoint += 1
                        break
                    in_flight[pool.apply_async(_run_node_job, (node.to_job(),))] = node

                # Harvest finished workers; sleep briefly only when none are
                # ready so the pool is not polled in a busy loop.
                if in_flight:
                    progressed = False
                    for async_result in list(in_flight):
                        if async_result.ready():
                            node = in_flight.pop(async_result)
                            result = async_result.get()
                            node.apply_result(result)
                            pending.extend(self._on_complete(node))
                            since_checkpoint += 1
                            progressed = True
                    if not progressed:
                        time.sleep(0.1)

                if since_checkpoint >= checkpoint_every:
                    self._resume_queue = list(pending) + list(in_flight.values())
                    self.save_checkpoint()
                    self._cleanup_completed()
                    self._print_progress(len(pending), len(in_flight))
                    since_checkpoint = 0
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()

        self._resume_queue = []
        self.save_checkpoint()
        self._cleanup_completed()
        self._print_progress(0, 0)
        results = self.sort_results()
        print("[wavefront] finished this scan "
              f"(angles={len(getattr(self, 'min_energies', {}) or getattr(self, 'min_bins', {}))}, "
              f"checkpoint={getattr(self, 'checkpoint', None)})")
        return results


    def calculate_mpi(self) -> None:
        """Apply the wavefront algorithm to optimize a dihedral scan using MPI."""
        from mpi4py import MPI
        import time
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
            setup = (self.los, self.conlist, self.reslist, template)
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

        self._resume_queue = list(pending)
        self.save_checkpoint()

        from ffpopt.runtime.fast_wavefront import wf_checkpoint_every

        checkpoint_every = max(wf_checkpoint_every(max(size - 1, 1)), 1)
        since_checkpoint = 0

        # Track available workers and tasks in flight
        idle_workers = set(range(1, size))
        in_flight = {}  # Mapping of worker_rank -> node

        try:
            while pending or in_flight:

                # 2. Dispatch work to idle ranks
                while pending and idle_workers:
                    worker = idle_workers.pop()
                    node = pending.popleft()

                    node.replace_with_pickle()

                    if node.complete:
                        pending.extend(self._on_complete(node))
                        idle_workers.add(worker) # Worker never actually got the task
                        continue
                    if not node.active:
                        idle_workers.add(worker)
                        continue

                    # Slim job: rcs + coords (not full node / los)
                    comm.send(node.to_job(), dest=worker, tag=TAG_TASK)
                    in_flight[worker] = node

                # 3. Harvest finished workers
                if in_flight:
                    status = MPI.Status()

                    result = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
                    worker = status.Get_source()

                    node = in_flight.pop(worker)
                    idle_workers.add(worker)

                    node.apply_result(result)
                    new_nodes = self._on_complete(node)
                    pending.extend(new_nodes)
                    since_checkpoint += 1

                # 4. Handle checkpoints and progress updates
                if since_checkpoint >= checkpoint_every:
                    self._resume_queue = list(pending) + list(in_flight.values())
                    self.save_checkpoint()
                    self._cleanup_completed()
                    self._print_progress(len(pending), len(in_flight))
                    since_checkpoint = 0

        finally:
            # 5. Shut down the worker pool gracefully
            for worker in range(1, size):
                comm.send(None, dest=worker, tag=TAG_STOP)

        self._resume_queue = []
        self.save_checkpoint()
        self._cleanup_completed()
        self._print_progress(0, 0)

        results = self.sort_results()
        print("[wavefront] finished this scan "
              f"(angles={len(getattr(self, 'min_energies', {}) or getattr(self, 'min_bins', {}))}, "
              f"checkpoint={getattr(self, 'checkpoint', None)})")
        return results


        
    
    def _print_progress(self, pending: int, in_flight: int) -> None:
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
        completed = sum(1 for level in self.levels
                        for node in level.nodes if node.complete)
        highest = max((level.level_id for level in self.levels), default=0)
        #total_angles = 360 // self.delta if self.delta else 0
        total_num_rcs = len(self.bins)
        print(f"[wavefront] completed={completed} pending={pending} "
              f"in-flight={in_flight} highest-level={highest} "
              f"rcs={len(self.min_bins)}/{total_num_rcs}")

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
                    node.cleanup()

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
        if not node.active:
            return []
        if self.max_levels > 0 and node.level + 1 > self.max_levels:
            print(f"Reached maximum levels: {self.max_levels}. Stopping calculation.")
            raise ValueError("Too many levels, something is wrong with the wavefront algorithm.")
        return self.spawn_neighbors(node)

    def _rebuild_level_energies(self) -> None:
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
        if self.los is not None:
            self.los.clear_runtime_caches()
        self._slim_nodes_for_checkpoint()
        from .wavefront_mixins import atomic_pickle_dump

        atomic_pickle_dump(self, self.checkpoint)
        print(f"Checkpoint saved to {self.checkpoint}.")

    def _slim_nodes_for_checkpoint(self) -> None:
        """Drop bulky redundant arrays from completed nodes before pickling."""
        for level in getattr(self, "levels", []) or []:
            for node in getattr(level, "nodes", []) or []:
                if not getattr(node, "complete", False):
                    continue
                if getattr(node, "opt_geom", None) is not None:
                    if "forces" in node.opt_geom.data:
                        node.opt_geom.data["forces"] = None
                n_atoms = len(node.struct.data["elements"])
                node.forces = np.zeros((n_atoms, 3))


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
        """Update per-bin minima and set ``node.active`` (spawn) via shared policy.

        See :func:`ffpopt.scan.wavefront_mixins.evaluate_wavefront_minimum`.
        """
        from ffpopt.scan.wavefront_mixins import (
            evaluate_wavefront_minimum,
            kcal_threshold_to_ev,
        )

        threshold_ev = kcal_threshold_to_ev(self.convergence_threshold)
        if not node.active:
            return

        bidx = self.grid.GetBinIdx(node.rcs)
        gidx = self.grid.CptGlbIdxFromBinIdx(bidx)

        if node.energy is None or not np.isfinite(node.energy):
            print(f"Node {node.rcs} is inactive due to failed optimization.")
            node.active = False
            return

        soft = bool(getattr(node, "soft_opt", False))
        if not soft and node.opt_geom is not None:
            soft = is_soft_opt_recovery(node.opt_geom)
            node.soft_opt = soft

        existing_soft = False
        has_incumbent = gidx in self.min_bins
        incumbent_energy = (
            self.min_bins[gidx].energy if has_incumbent else None
        )
        if gidx in self.min_nodes:
            prev = self.min_nodes[gidx]
            existing_soft = bool(getattr(prev, "soft_opt", False))
            if not existing_soft and gidx in self.min_bins:
                existing_soft = is_soft_opt_recovery(self.min_bins[gidx].struct)

        decision = evaluate_wavefront_minimum(
            energy=node.energy,
            soft=soft,
            has_incumbent=has_incumbent,
            incumbent_energy=incumbent_energy,
            incumbent_soft=existing_soft,
            threshold_ev=threshold_ev,
        )
        reason = decision["reason"]
        if decision["update_min"]:
            old = incumbent_energy
            if gidx not in self.min_bins:
                self.min_bins[gidx] = self.bins[gidx]
            self.min_bins[gidx].energy = node.energy
            self.min_bins[gidx].struct = node.opt_geom
            self.min_nodes[gidx] = node
            if reason == "soft_first_seed":
                print(
                    f"New reaction coordinate (soft-opt seed): {node.rcs}, "
                    f"Energy: {node.energy} "
                    f"(recovery={getattr(node, 'opt_recovery', None)}; spawn once)"
                )
            elif reason == "soft_improve":
                print(
                    f"Updating soft-opt node: {node.rcs}, "
                    f"Old Energy: {old}, New Energy: {node.energy} (no spawn)"
                )
            elif reason == "hard_first":
                print(
                    f"New reaction coordinate detected: {node.rcs}, "
                    f"Energy: {node.energy}"
                )
            elif reason == "hard_replace_soft":
                print(
                    f"Replacing soft-opt node {node.rcs} with hard-converged "
                    f"Energy: {node.energy} (was {old})"
                )
            elif reason == "hard_significant_improve":
                print(
                    f"Updating node: {node.rcs}, Old Energy: {old}, "
                    f"New Energy: {node.energy}"
                )
            elif reason == "hard_quiet_improve":
                print(
                    f"Quiet update node {node.rcs}: {old} -> {node.energy} "
                    f"(within threshold; no spawn)"
                )
        else:
            if reason == "soft_demoted":
                print(
                    f"Node {node.rcs} soft-opt demoted "
                    f"(recovery={getattr(node, 'opt_recovery', None)}); "
                    f"not replacing hard-converged / lower soft minimum."
                )
            elif reason == "hard_worse_than_soft":
                print(
                    f"Node {node.rcs} hard-opt higher than soft min "
                    f"({node.energy} > {incumbent_energy}); keeping soft profile."
                )
            elif reason == "hard_not_lower":
                print(
                    f"Node {node.rcs} is not active, energy {node.energy} "
                    f"is not lower than minimum {incumbent_energy}."
                )
            else:
                print(f"Node {node.rcs} inactive ({reason}): energy {node.energy}.")

        node.active = bool(decision["active"])

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
        next_level = self._get_or_create_level(node.level + 1)

        # TIM RIGHT HERE. THIS NEEDS A N-D LAYER ALGORTHIM WITH CHECKS TO MAKE
        # SURE THINGS ARE IN RANGE *AND* THE PROPOSED BINS ARE WITHIN self.bins
        # BECAUSE IN THE FUTURE THE BINS DICT MAY ONLY HAVE A SUBSECTION
        
        # lower = self.nearest_angle(node.angle - self.delta, self.delta) % 360
        # upper = self.nearest_angle(node.angle + self.delta, self.delta) % 360
        # lower_node = next_level.add_node(self.los,
        #                                  node.opt_geom,
        #                                  self.con,
        #                                  angle=lower)
        # upper_node = next_level.add_node(self.los,
        #                                  node.opt_geom,
        #                                  self.con,
        #                                  angle=upper)
        # return [lower_node, upper_node]

        bidx = self.grid.GetBinIdx(node.rcs)
        bins = GetGridNeighbors(bidx,self.grid,validbins=self.bins)
        nodes = []
        for newbin in bins:
            gidx = self.grid.CptGlbIdxFromBinIdx(newbin)
            rcs = self.bins[gidx].center
            #print(f"spawn_neighbors {gidx} {newbin} {rcs}")
            nodes.append( next_level.add_node( self.los, node.opt_geom,
                                               self.conlist, self.reslist,
                                               self.grid,
                                               rcs ) )
            print(f"Spawn  node {nodes[-1].node_pkl} from {node.node_pkl}")
            
        return nodes

    def sort_results(self) -> tuple[list[float], list[float], list[ase.Atoms]]:
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

    def print_summary(self) -> None:
        """Print a summary of the wavefront results."""
        print("Wavefront Summary:")
        print(f"Total Levels: {len(self.levels)}")
        print("Number of Nodes per Level:")
        for i, level in enumerate(self.levels):
            print(f"Level {i+1}: {len(level.nodes)} nodes")
        total = sum(len(level.nodes) for level in self.levels)
        print("Total Nodes: ", total)
        if self.levels:
            print(
                "Average number of nodes per level: ",
                total / len(self.levels),
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

        print(f"Failed nodes: {len(failed)}")
        if failed:
            for node in failed[:20]:
                print(
                    f"  rcs={node.rcs} id={node.node_id} "
                    f"error={getattr(node, 'error', None)}"
                )
            if len(failed) > 20:
                print(f"  ... and {len(failed) - 20} more")
        print(f"Soft-accepted nodes (no spawn): {len(soft)}")
        if failed:
            print(
                f"Wavefront calculation finished with {len(failed)} failed "
                f"node(s)."
            )
        else:
            print(
                "[wavefront] summary: no failed nodes "
                f"({len(soft)} soft-accepted)."
            )
    
    

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

        plt.tight_layout()
        plt.savefig(pngfile)

        bins = copy.deepcopy(self.min_bins)
        for gidx in bins:
            bins[gidx].value = bins[gidx].energy * KCAL_PER_EV
            bins[gidx].stderr = 0
            bins[gidx].entropy = 1
            

        fes = ndfes.MBAR( self.grid, bins )
        ndfes.SaveXml(xmlfile,[fes])
        
        
        
# def find_adjacent_dihedrals(con: Constraint, los: ListOfStruct) -> tuple[list[int], list[int]]:
#     """ Generates initial conformers based on the initial geometry optimization.
    
#     Parameters
#     ----------
#     atoms : ase.Atoms
#         The atoms to optimize.
#     con : Constraints
#         The constraints to apply during optimization.
#     stdargs
#         The standard arguments for the optimization."""
    
#     compare_dih = con.idxs
#     first = compare_dih[:2]
#     second = compare_dih[2:]
#     chosen_first, chosen_second = None, None
#     mol = los[0].GetParmedAtoms()
    
#     for d in mol.dihedrals:
#         if d.improper:
#             continue
#         idxs = [d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx]
#         # Check that neither atom 2 nor atom 3 is a carbon atom with less than 4 bonds
#         flag_value=False
#         for atom_idx in [idxs[1], idxs[2]]:
#             atom = mol.atoms[atom_idx]
#             # Make sure that the atom is not a carbon with less than 4 bonds or a nitrogen with less than 3 bonds
#             # if found, it will likely force a planar bond.
#             if atom.atomic_number == 6 and len(atom.bonds) < 4:
#                 flag_value=True
#             if atom.atomic_number == 7 and len(atom.bonds) < 3:
#                 flag_value=True
#         if flag_value:
#             continue
#         if idxs[1] == first[0] and idxs[2] == first[1]:
#             chosen_first = idxs
#         elif idxs[2] == first[0] and idxs[1] == first[1]:
#             chosen_first = idxs[::-1]
#         elif idxs[1] == second[0] and idxs[2] == second[1]:
#             chosen_second = idxs
#         elif idxs[2] == second[0] and idxs[1] == second[1]:
#             chosen_second = idxs[::-1]
#         else:
#             continue
#         if chosen_first and chosen_second:
#             break
#     if chosen_first is None and chosen_second is None:
#         print("No adjacent dihedrals found for conformer generation.")
#     else:
#         print(chosen_first, chosen_second)
#     return chosen_first, chosen_second

def wavefront_loader(filename: str) -> Wavefront:
    """Load a Wavefront object from a pickle file (see ``wavefront_mixins``)."""
    from .wavefront_mixins import load_wavefront_pickle

    return load_wavefront_pickle(filename, restore_soft_opt=True)


def run_dihed_wavefront(
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
    """Run a relaxed dihedral wavefront scan from Python kwargs.

    Required: ``inp`` (input json), ``out`` (output json), ``condim`` (e.g.
    ``[xlo,xhi,nbins]``). All other wavefront-specific options match the
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
        ``{'wf_run', 'angles', 'energies', 'energies_noshift', 'structures'}``.
        ``energies`` is min-shifted (kcal/mol); ``energies_noshift`` is the raw
        kcal/mol energies before shifting.
    """
    import argparse
    import sys as _sys
    from types import SimpleNamespace
    from ffpopt.Options import AddStandardOptions
    from ffpopt.Options import AddConstraintAndRestraintOptions
    from ffpopt.Options import ParseConstraintAndRestraintOptions
    from ffpopt.Options import DeleteConstraintAndRestraintFromStruct
        
    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

    from ndfes import VirtualGrid, SpatialDim

    # Spawn workers re-import the caller's entry script with __name__ == '__mp_main__'.
    # If we see that name on our caller's frame, the caller didn't wrap the call in
    # `if __name__ == '__main__':` and we'd otherwise blow up inside multiprocessing
    # with a confusing traceback. Fail loudly with a fix-it message instead.
    _caller_globals = _sys._getframe(1).f_globals
    if _caller_globals.get("__name__") == "__mp_main__":
        raise RuntimeError(
            "run_dihed_wavefront was re-invoked by a multiprocessing spawn worker "
            "re-importing the calling script. Wrap the call in "
            "`if __name__ == '__main__':` so the worker re-import doesn't "
            "re-execute it. See "
            "https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming"
        )

    _p = argparse.ArgumentParser(add_help=False)
    AddStandardOptions(_p)
    AddConstraintAndRestraintOptions(_p)
    std_defaults = vars(_p.parse_args([]))
    unknown = set(standard_kwargs) - set(std_defaults)
    if unknown:
        raise TypeError(
            f"run_dihed_wavefront got unexpected keyword argument(s): {sorted(unknown)}"
        )
    std = {**std_defaults, **standard_kwargs}

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
    #los.structs = [los.structs[0]]

    #conlist,reslist = ParseConstraintAndRestraintOptions(args,struct=los[0])
    conlist,reslist = ParseConstraintAndRestraintOptions(args)
    for s in los:
        DeleteConstraintAndRestraintFromStruct(s,conlist,reslist)

    if len(conlist) != len(condim):
        raise Exception(f"Constraint dimension mismatch {len(conlist)} "
                        f" {len(condim)}")
    
    if len(reslist) != len(resdim):
        raise Exception(f"Restraint dimension mismatch {len(reslist)} "
                        f" {len(resdim)}")

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
        maxsize = max(maxsize,size)
        if isper and int(round(xhi-xlo)) != 360:
            isper=False
        dims.append( SpatialDim(xlo,xhi,size,isper) )
    for i in range(len(resdim)):
        isper = reslist[i].isper()
        cs = resdim[i].split(",")
        if len(cs) != 3:
            raise Exception(f"Expected 2 floats and an int, but found {resdim[i]}")
        xlo = float(cs[0])
        xhi = float(cs[1])
        size = int(cs[2])
        maxsize = max(maxsize,size)
        if isper and int(round(xhi-xlo)) != 360:
            isper=False
        dims.append( SpatialDim(xlo,xhi,size,isper) )

        
    grid = VirtualGrid( dims )
    
        
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

    #
    # Need to first check to see if this is a MPI run
    # If so, then it should only init Wavefront and immediately call calculate_mpi()
    #

    is_worker = is_mpi_worker()
        
    if starting_checkpoint_path.exists():
        if not is_worker:
            print(f"Checkpoint file {starting_checkpoint_path} exists. Loading previous wavefront run.")
        from .wavefront_mixins import pickle_load_compat

        wf_run = pickle_load_compat(starting_checkpoint_path)
        wf_run.restart_options(
            los,
            max_levels=int(args.wf_max_levels),
            nproc=args.nproc,
            use_mpi=args.mpi,
            checkpoint=checkpoint_path
        )
        if not isinstance(wf_run, Wavefront):
            raise TypeError("Checkpoint file does not contain a valid Wavefront object.")
        if not is_worker:
            print("Wavefront run loaded successfully.")
    else:
        if not is_worker:
            print(f"No checkpoint file found at {starting_checkpoint_path}. Starting a new wavefront run.")
        wf_run = Wavefront(
            los, conlist, reslist, grid,
            max_levels=int(args.wf_max_levels),
            nproc=args.nproc,
            use_mpi=args.mpi,
            checkpoint=checkpoint_path,
            convergence_threshold=args.wf_convergence_threshold
        )

    if args.wf_change_theory:
        if not is_worker:
            print("Changing theory of the calculator for all nodes in the wavefront calculation.")
        wf_run.theory_change(los, stride=args.wf_theory_stride)

    wf_run.calculate()

    if is_worker:
        return None
    
    rcs, energies, structures = wf_run.sort_results()

    KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    energies_noshift = [e * KCAL_PER_EV for e in energies]
    emin = 0
    if len(energies_noshift) > 0:
        emin = np.amin(energies_noshift)
        energies = energies_noshift - emin
    else:
        energies = np.array([])

    structures.save(args.out)

    for i, rcs in enumerate(rcs):
        print(f"Angle: {rcs}, Energy: {energies[i]}")
    print(f"[wavefront] finished this scan -> {args.out}")

    out_path = Path(args.out).resolve()
    dat = out_path.with_suffix(".dat")
    with open(dat, "w") as fh:
        for a, e, n in zip(rcs, energies, energies_noshift):
            fh.write(f"{a} {e} {n}\n")
    print(f"Data written to {dat}.")

    pkl_path = out_path.with_suffix(".pkl")
    pickle.dump(wf_run, open(pkl_path, "wb"))
    print(f"Wavefront run saved to {pkl_path}.")
    wf_pngfile = str(out_path.parent / f"wf_workflow_{out_path.with_suffix('.png').name}")
    wf_xmlfile = str(Path(wf_pngfile).with_suffix(".xml"))
    wf_run.plot_wavefront(pngfile=wf_pngfile, xmlfile=wf_xmlfile)
    wf_run.print_summary()
    print(f"Wavefront plot saved as '{wf_pngfile}'.")
    print(f"Energies saved as '{wf_xmlfile}'.")

    return {
        "wf_run": wf_run,
        "rcs": rcs,
        "energies": energies,
        "energies_noshift": energies_noshift,
        "structures": structures,
    }

