#!/usr/bin/env python3
from __future__ import annotations

import copy
import os
import sys
import pickle
import numpy as np

from typing import Generator, Optional
from pathlib import Path

from . GeomOpt import GeomOpt, bare_potential_energy, is_soft_opt_recovery, opt_recovery_label
from . Constraints import Constraint
from . Struct import ListOfStruct, Struct


# Per-process worker state (set once via Pool initializer / MPI bcast).
_WORKER: dict = {}


def _clear_los_calc(los: ListOfStruct) -> None:
    """Drop live calculators so workers rebuild (and cache) in-process."""
    clearer = getattr(los, "clear_runtime_caches", None)
    if callable(clearer):
        clearer()
        return
    calc = getattr(los, "calc", None)
    if calc is not None:
        try:
            calc.reset()
        except Exception:
            pass
        los.calc = None
    if hasattr(los, "_ffpopt_calc_cache"):
        los._ffpopt_calc_cache = None


def _init_worker(los: ListOfStruct, con: Constraint, template_struct: Struct) -> None:
    """Pool initializer: share los / constraint / topology once per worker."""
    _clear_los_calc(los)
    _WORKER["los"] = los
    _WORKER["con"] = copy.deepcopy(con)
    _WORKER["template"] = copy.deepcopy(template_struct)


def _struct_from_coords(coords) -> Struct:
    struct = copy.deepcopy(_WORKER["template"])
    struct.Update(0.0, np.asarray(coords, dtype=float), None)
    return struct


def _run_node_job(job: dict) -> dict:
    """Worker entry: slim angle/coords job in -> slim result out (no ``los``)."""
    node = WavefrontNode.from_job(job, _WORKER["los"], _WORKER["con"])
    node.calculate()
    return node.to_result()


def _run_node(node: "WavefrontNode") -> "WavefrontNode":
    """In-process entry (serial path); mutates and returns ``node``."""
    node.calculate()
    return node


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
        con: Constraint,
        angle: float = None,
        level: int = None,
        node_id: int = None,
        workdir: str | Path = ".",
    ) -> None:
        self.los = los
        self.struct = struct
        self.energy = None
        self.forces = np.zeros((len(struct.data["elements"]), 3))
        self.angle = angle
        self.active = True
        self.constraints = [ copy.deepcopy(con) ]
        self.opt_geom = None
        self.level = level
        self.node_id = node_id
        # Keep node pickles beside the scan outputs (not process cwd) so
        # fragmented workflows can avoid os.chdir with multiprocessing.
        self.node_pkl = str(
            Path(workdir)
            / f"level_{self.level}_angle_{self.angle}_id_{self.node_id}_node.pckl"
        )
        self.complete = False
        self.error = None
        self.soft_opt = False
        self.opt_recovery = None

    @classmethod
    def from_job(cls, job: dict, los: ListOfStruct, con: Constraint) -> "WavefrontNode":
        """Rebuild a node from a slim IPC job (coords + angle; no pickled ``los``)."""
        if "coords" in job and job["coords"] is not None:
            struct = _struct_from_coords(job["coords"])
        else:
            struct = job["struct"]
        node = cls(
            los=los,
            struct=struct,
            con=con,
            angle=job["angle"],
            level=job["level"],
            node_id=job["node_id"],
            workdir=Path(job["node_pkl"]).parent,
        )
        # __init__ rebuilds node_pkl from workdir; restore exact path for checkpoints.
        node.node_pkl = job["node_pkl"]
        node.complete = bool(job.get("complete", False))
        return node

    def to_job(self) -> dict:
        """Slim payload for spawn workers: angle + coords (not ``los``)."""
        coords = np.asarray(self.struct.data["positions"], dtype=float)
        return {
            "angle": self.angle,
            "coords": coords,
            "level": self.level,
            "node_id": self.node_id,
            "node_pkl": self.node_pkl,
            "complete": self.complete,
        }

    def to_result(self) -> dict:
        """Slim result payload: energy + optimized coords (not ``los`` / full node)."""
        coords = None
        if self.opt_geom is not None:
            coords = np.asarray(self.opt_geom.data["positions"], dtype=float)
        return {
            "energy": self.energy,
            "forces": self.forces,
            "coords": coords,
            "complete": self.complete,
            "error": self.error,
            "active": self.active,
            "soft_opt": self.soft_opt,
            "opt_recovery": self.opt_recovery,
        }

    def apply_result(self, result: dict) -> None:
        """Merge a slim worker result into this parent-side node."""
        self.energy = result.get("energy")
        if result.get("forces") is not None:
            self.forces = result["forces"]
        self.complete = bool(result.get("complete", self.complete))
        self.error = result.get("error", self.error)
        if "active" in result:
            self.active = bool(result["active"])
        self.soft_opt = bool(result.get("soft_opt", False))
        self.opt_recovery = result.get("opt_recovery", self.opt_recovery)
        coords = result.get("coords")
        if coords is not None:
            self.opt_geom = copy.deepcopy(self.struct)
            self.opt_geom.Update(self.energy, coords, result.get("forces"))
            if self.opt_recovery:
                tag = str(self.opt_recovery)
                if tag in ("BFGS", "LBFGS", "FIRE") or tag.endswith("-soft"):
                    self.opt_geom.data["ase_opt_recovery"] = tag
                else:
                    self.opt_geom.data["geometric_recovery"] = tag

    def replace_with_pickle(self) -> None:
        """Replace node fields from a sidecar pickle if present (restores ``los``)."""
        filename = Path(f"{self.node_pkl}")
        if Path.is_file(filename):
            print("Found existing pickle file for node:", self.node_id)
            los = self.los
            with open(filename, 'rb') as f:
                loaded_node = pickle.load(f)
                self.__dict__.update(loaded_node.__dict__)
            if self.los is None:
                self.los = los
            print("Node data replaced with pickle data.")

    def calculate(self) -> None:
        """Calculate the energy of the atoms."""
        if not self.complete:
            self.constraints[0].value = self.angle
            precheck_err = self._precheck_geometry()
            if precheck_err is not None:
                self._mark_failed(precheck_err)
                return
            try:
                self.opt_geom = GeomOpt(self.los, self.struct, constraints=self.constraints)
                self.opt_recovery = opt_recovery_label(self.opt_geom)
                self.soft_opt = is_soft_opt_recovery(self.opt_geom)
                if self.soft_opt:
                    print(
                        f"Node {self.node_id} soft-accepted opt "
                        f"(recovery={self.opt_recovery}); will not spawn neighbors"
                    )
                self.energy = np.round(bare_potential_energy(self.opt_geom), 6)
                self.forces = self.opt_geom.data.get("forces", self.forces)
                self._write_checkpoint()
                self.complete = True
            except Exception as e:
                print(f"Node {self.node_id} optimization error: {type(e).__name__}: {e}")
                self._mark_failed("optimization_error", e)

    def _write_checkpoint(self) -> None:
        """Write the node's data to a pickle file (without ``los``)."""
        los = self.los
        self.los = None
        try:
            with open(self.node_pkl, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self.los = los

    def cleanup(self) -> None:
        """Clean up the node's pickle file."""
        filename = Path(f"{self.node_pkl}")
        if Path.is_file(filename):
            print("Cleaning up node pickle file:", self.node_pkl)
            os.remove(filename)

    def _mark_failed(self, reason: str, error: Optional[Exception] = None) -> None:
        msg = reason
        if error is not None:
            msg = f"{reason}: {error}"
        self.error = msg
        self.active = False
        self.complete = True
        self.energy = np.inf
        self.forces = np.zeros((len(self.struct.data["elements"]), 3))
        print(f"Node {self.node_id} failed at angle {self.angle}: {msg}")
        self._write_checkpoint()

    def _precheck_geometry(self, min_dist: float = 0.8) -> Optional[str]:
        """Return a failure reason, or ``None`` if the geometry looks usable.

        Distinguishes real clashes from precheck exceptions (imports, constraint
        apply failures, …) so failure reports are not all labeled as clashes.
        """
        try:
            from ffpopt.Constraints import FillConstraints, ApplyConstraints, has_nonbonded_clash
            myatoms = self.struct.GetASEAtoms()
            cons = FillConstraints(myatoms, copy.deepcopy(self.constraints))
            myatoms = ApplyConstraints(myatoms, cons)
            clashed, i, j, dist = has_nonbonded_clash(
                myatoms.get_positions(), self.struct.data["bonds"], min_dist=min_dist
            )
            if clashed:
                print(f"Precheck clash: atom {i} and atom {j} at {dist:.3f} Å (< {min_dist} Å)")
                return "clash_precheck"
        except Exception as e:
            print(f"Precheck failed due to error: {e}")
            return f"precheck_error: {e}"
        return None

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

    def add_node(self, #atoms: ase.Atoms, stdargs: StandardArgs,
                 los: ListOfStruct, struct: Struct,
                 con: Constraint, angle: float = None,
                 *, workdir: str | Path = "."):
        """Add a node to the level.
        
        This creates a new WavefrontNode with the given atoms, standard arguments, constraint, and angle.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms to optimize.
        stdargs
            The standard arguments for the optimization.
        con : Constraint
            The constraint to apply to the optimization.
        angle : float, optional
            The angle to optimize. If not provided, it will be set to the initial angle of the constraint.
        workdir : path-like, optional
            Directory for per-node pickle checkpoints. Default ``"."``.
        
        Returns
        -------
        None
        
        """
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
    def __init__(self, los: ListOfStruct, struct: Struct, con: Constraint, delta: int = 10, max_levels: int = 1, nproc: int = 1, starting_nodes: int = 1, extra_dih: list[int] = None, num_conformers: int = 1, checkpoint: str = "wavefront_checkpoint.pkl", convergence_threshold: float = 0.01) -> None:
        self.los = los
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
        # Sidecar node pickles live next to the checkpoint (absolute when
        # ``checkpoint`` is absolute), not the process cwd.
        self.workdir = str(Path(checkpoint).resolve().parent)
        self.restarted = []
        self.verbose = False
        self.convergence_threshold = convergence_threshold
        # Nodes created but not yet completed. Persisted in the checkpoint so a
        # restart re-enqueues the in-flight/pending work (see calculate).
        self._resume_queue = None

    def restart_options(self, los: ListOfStruct, #stdargs: StandardArgs,
                        max_levels: int = -1, nproc: int = 1, checkpoint: str = None) -> None:
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
        #self.stdargs = stdargs
        self.los = los

        if checkpoint is not None:
            self.checkpoint = checkpoint
            self.workdir = str(Path(checkpoint).resolve().parent)

        # Rebind the (possibly re-themed) calculator source onto every node. The
        # queue scheduler can leave incomplete nodes at any level, not just the
        # last one, so update them all rather than only self.levels[-1].
        for level in self.levels:
            for node in level.nodes:
                node.los = los
        print("Number of times restarted:", len(self.restarted))
        self.restarted.append(prev_options)
        
        print("Restart options updated.")

    def theory_change(self, #stdargs: StandardArgs,
                      los: ListOfStruct,
                      stride: int = 1) -> None:
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
            print(angle)
            new_starting_level.add_node(
                self.min_structures[angle],
                los,
                self.con,
                angle=angle,
                workdir=self.workdir,
            )
        self.min_energies = {}
        self.min_structures = {}
        self.min_nodes = {}
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
        print("Starting wavefront calculation...")
        self.init_min()
        #Add the initial level
        self.add_level()
        chosen_first, chosen_second = find_adjacent_dihedrals(self.con, self.los)
        extra_dih = None
        if chosen_first is not None:
            extra_dih = chosen_first
        if chosen_second is not None:
            extra_dih = chosen_second
        
        print("Generating conformers and adding initial nodes...")
        min_geoms = []
        
        for i in range(self.num_conformers):
            angle_increment = (360 // self.num_conformers) * i
            print(f"Generating conformer {i+1} with angle increment {angle_increment} degrees.")
            min_geoms.append(self.init_conformer(self.struct, self.con, extra_dih, angle_increment))

        for i in range(self.starting_nodes):
            # Add the initial node with the minimum geometry angle
            add_ang = (360//self.delta//self.starting_nodes * i *self.delta)
            print(f"Adding starting node at angle: {self.nearest_angle(self.min_geom_ang + add_ang, self.delta)% 360}")
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
        from ffpopt.Constraints import Constraint
        if iter_count > 5:
            raise ValueError("Maximum iteration count exceeded for conformer generation. This likely indicates a persistent clash that cannot be resolved.")
        
        new_atoms = copy.deepcopy(struct)
        extra_con = ",".join([str(x) for x in extra_con_in]) if extra_con_in else None
        if extra_con is not None:
            print(f"Generating conformer with extra dihedral constraint: {extra_con} at angle {angle_increment}")
            extra_con = Constraint.from_str(extra_con, graph=self.struct.GetGraph())
            extra_con.value = angle_increment
            if self.init_check(new_atoms, con, con.value, extra_con=extra_con, extra_angle=angle_increment):
                conf_min_geom = GeomOpt(self.los, new_atoms, constraints=[con, extra_con])
            else:
                print(f"Skipping conformer generation for angle {angle_increment} due to invalid initial geometry.")
                conf_min_geom = self.init_conformer(struct, con, extra_con_in, angle_increment=angle_increment + self.delta, iter_count=iter_count + 1)
        else:
            print(f"Generating conformer without extra dihedral constraint at angle {angle_increment}")
            print(f"This typically means that there were no adjacent dihedrals found.")
            if self.init_check(new_atoms, con, con.value):
                conf_min_geom = GeomOpt(self.los, new_atoms, constraints=[con])
            else:
                print(f"Skipping conformer generation for angle {angle_increment} due to invalid initial geometry.")
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
        from ffpopt.Constraints import FillConstraints, ApplyConstraints
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
        from ffpopt.Constraints import has_nonbonded_clash
        clashed, i, j, dist = has_nonbonded_clash(
            myatoms.get_positions(), struct.data["bonds"], min_dist=0.8
        )
        if clashed:
            print(f"Warning: Distance between atom {i} and atom {j} is {dist:.3f} Å (< 0.8 Å)")
            print(f"Ommitting this node due to close proximity of atoms.")
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
        >>> from ffpopt.WaveFront import Wavefront
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
            # Share los/con/template once per worker; jobs carry only angle/coords.
            pool = ctx.Pool(
                processes=self.nproc,
                initializer=_init_worker,
                initargs=(self.los, self.con, self.struct),
            )

        checkpoint_every = max(self.nproc, 1)
        try:
            in_flight = {}        # AsyncResult -> node; transient, never pickled
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
        print("Wavefront calculation completed.")
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
        total_angles = 360 // self.delta if self.delta else 0
        print(f"[wavefront] completed={completed} pending={pending} "
              f"in-flight={in_flight} highest-level={highest} "
              f"angles={len(self.min_energies)}/{total_angles}")

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
        if self.los is not None:
            self.los.clear_runtime_caches()
        self._slim_nodes_for_checkpoint()
        with open(self.checkpoint, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Checkpoint saved to {self.checkpoint}.")

    def _slim_nodes_for_checkpoint(self) -> None:
        """Drop bulky redundant arrays from completed nodes before pickling."""
        for level in getattr(self, "levels", []) or []:
            for node in getattr(level, "nodes", []) or []:
                if not getattr(node, "complete", False):
                    continue
                # Forces are not needed to resume; energy + opt_geom coords are.
                if getattr(node, "opt_geom", None) is not None:
                    if "forces" in node.opt_geom.data:
                        node.opt_geom.data["forces"] = None
                n_atoms = len(node.struct.data["elements"])
                node.forces = np.zeros((n_atoms, 3))

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
        """ Update the per-angle minima for one node and set its active flag.

        Applies the historical per-level energy filter to a single node so it
        can run as each result arrives. A node stays active (and will spawn
        neighbors) when it is the first seen at its angle or strictly lowers that
        angle's minimum by at least ``convergence_threshold``; otherwise it is
        marked inactive. ``convergence_threshold`` is in kcal/mol but node
        energies are stored in eV, so the threshold is converted to eV before
        the comparison.

        Soft-accepted optimizations (``soft-maxiter`` / ASE ``*-soft``) may fill
        or improve soft minima for the energy profile, but never displace a
        hard-converged minimum and never spawn neighbors.

        Parameters
        ----------
        node : WavefrontNode
            The completed node to evaluate.

        """
        from . constants import AU_PER_ELECTRON_VOLT, AU_PER_KCAL_PER_MOL
        # Node energies are eV; the threshold is kcal/mol. KCAL_PER_EV is
        # kcal/mol per eV, so dividing converts the threshold into eV.
        kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
        threshold_ev = self.convergence_threshold / kcal_per_ev
        if not node.active:
            return
        if node.energy is None or not np.isfinite(node.energy):
            print(f"Angle {node.angle} is inactive due to failed optimization.")
            node.active = False
            return

        soft = bool(getattr(node, "soft_opt", False))
        if not soft and node.opt_geom is not None:
            soft = is_soft_opt_recovery(node.opt_geom)
            node.soft_opt = soft

        existing_soft = False
        if node.angle in self.min_nodes:
            prev = self.min_nodes[node.angle]
            existing_soft = bool(getattr(prev, "soft_opt", False))
            if not existing_soft:
                existing_soft = is_soft_opt_recovery(
                    self.min_structures.get(node.angle)
                )

        if soft:
            if node.angle not in self.min_energies:
                print(
                    f"New angle (soft-opt): {node.angle}, Energy: {node.energy} "
                    f"(recovery={getattr(node, 'opt_recovery', None)}; no spawn)"
                )
                self.min_energies[node.angle] = node.energy
                self.min_structures[node.angle] = node.opt_geom
                self.min_nodes[node.angle] = node
            elif existing_soft and node.energy < self.min_energies[node.angle]:
                print(
                    f"Updating soft-opt angle: {node.angle}, "
                    f"Old Energy: {self.min_energies[node.angle]}, "
                    f"New Energy: {node.energy} (no spawn)"
                )
                self.min_energies[node.angle] = node.energy
                self.min_structures[node.angle] = node.opt_geom
                self.min_nodes[node.angle] = node
            else:
                print(
                    f"Angle {node.angle} soft-opt demoted "
                    f"(recovery={getattr(node, 'opt_recovery', None)}); "
                    f"not replacing hard-converged / lower soft minimum."
                )
            node.active = False
            return

        if node.angle not in self.min_energies:
            print(f"New angle detected: {node.angle}, Energy: {node.energy}")
            self.min_energies[node.angle] = node.energy
            self.min_structures[node.angle] = node.opt_geom
            self.min_nodes[node.angle] = node
        elif existing_soft:
            print(
                f"Replacing soft-opt angle {node.angle} with hard-converged "
                f"Energy: {node.energy} (was {self.min_energies[node.angle]})"
            )
            self.min_energies[node.angle] = node.energy
            self.min_structures[node.angle] = node.opt_geom
            self.min_nodes[node.angle] = node
        elif (abs(node.energy - self.min_energies[node.angle]) < threshold_ev):
            print(f"Angle {node.angle} is not active, energy {node.energy} is not sufficiently different from minimum {self.min_energies[node.angle]}.")
            node.active = False
        elif node.energy < self.min_energies[node.angle]:
            print(f"Updating angle: {node.angle}, Old Energy: {self.min_energies[node.angle]}, New Energy: {node.energy}")
            self.min_energies[node.angle] = node.energy
            self.min_structures[node.angle] = node.opt_geom
            self.min_nodes[node.angle] = node
        else:
            print(f"Angle {node.angle} is not active, energy {node.energy} is not lower than minimum {self.min_energies[node.angle]}.")
            node.active = False

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
        lower = self.nearest_angle(node.angle - self.delta, self.delta) % 360
        upper = self.nearest_angle(node.angle + self.delta, self.delta) % 360
        lower_node = next_level.add_node(
            self.los,
            node.opt_geom,
            self.con,
            angle=lower,
            workdir=self.workdir,
        )
        upper_node = next_level.add_node(
            self.los,
            node.opt_geom,
            self.con,
            angle=upper,
            workdir=self.workdir,
        )
        return [lower_node, upper_node]

    def sort_results(self) -> tuple[list[float], list[float], ListOfStruct]:
        """Sort the results by angle."""
        from . Struct import ListOfStruct
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
                    f"  angle={node.angle} id={node.node_id} "
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
            print("Wavefront calculation completed successfully.")
    
    

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

    plt.tight_layout()
    plt.savefig(filename)

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
        print("No adjacent dihedrals found for conformer generation.")
    else:
        print(chosen_first, chosen_second)
    return chosen_first, chosen_second

def wavefront_loader(filename: str) -> Wavefront:
    """ Load a Wavefront object from a pickle file.
    
    This function loads a Wavefront object from a pickle file. This is useful for restarting calculations
    that were previously interrupted.
    
    Parameters
    ----------
    filename : str
        The name of the pickle file to load the Wavefront object from.
        
    Returns
    -------
    Wavefront
        The loaded Wavefront object.
        
    """
    with open(filename, 'rb') as f:
        wavefront = pickle.load(f)
    print("Wavefront object loaded from", filename)
    return wavefront


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
        ``{'wf_run', 'angles', 'energies', 'energies_noshift', 'structures'}``.
        ``energies`` is min-shifted (kcal/mol); ``energies_noshift`` is the raw
        kcal/mol energies before shifting.
    """
    import argparse
    import sys as _sys
    from types import SimpleNamespace
    from . Options import AddStandardOptions
    from . constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

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
        print(f"Checkpoint file {starting_checkpoint_path} exists. Loading previous wavefront run.")
        wf_run = pickle.load(open(starting_checkpoint_path, "rb"))
        wf_run.restart_options(
            inps,
            max_levels=int(args.wf_max_levels),
            nproc=args.nproc,
            checkpoint=checkpoint_path,
        )
        if not isinstance(wf_run, Wavefront):
            raise TypeError("Checkpoint file does not contain a valid Wavefront object.")
        print("Wavefront run loaded successfully.")
    else:
        print(f"No checkpoint file found at {starting_checkpoint_path}. Starting a new wavefront run.")
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
        print("Changing theory of the calculator for all nodes in the wavefront calculation.")
        wf_run.theory_change(inps, stride=args.wf_theory_stride)

    wf_run.calculate()
    angles, energies, structures = wf_run.sort_results()

    KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    energies_noshift = [e * KCAL_PER_EV for e in energies]
    energies = energies_noshift - np.amin(energies_noshift)

    structures.save(args.out)

    for i, angle in enumerate(angles):
        print(f"Angle: {angle}, Energy: {energies[i]}")
    print(f"Wavefront scan completed. Results written to {args.out}.")

    dat = out_path.with_suffix(".dat")
    with open(dat, "w") as fh:
        for a, e, n in zip(angles, energies, energies_noshift):
            fh.write(f"{a} {e} {n}\n")
    print(f"Data written to {dat}.")

    pkl_path = out_path.with_suffix(".pkl")
    pickle.dump(wf_run, open(pkl_path, "wb"))
    print(f"Wavefront run saved to {pkl_path}.")
    wf_fname = str(out_path.parent / f"wf_workflow_{out_path.with_suffix('.png').name}")
    plot_wavefront(wf_run.levels, delta=args.delta, filename=wf_fname)
    wf_run.print_summary()
    print(f"Wavefront plot saved as '{wf_fname}'.")

    return {
        "wf_run": wf_run,
        "angles": angles,
        "energies": energies,
        "energies_noshift": energies_noshift,
        "structures": structures,
    }

