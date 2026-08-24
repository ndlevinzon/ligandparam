"""Parallel / MPI geometry optimization drivers (CalcNode + pools)."""

from __future__ import annotations

def ParallelGeomOpt(los,norestene,nproc):
    out = None
    if is_mpi():
        out = ParallelGeomOpt_mpi(los, norestene)
    else:
        out = ParallelGeomOpt_threads(los,norestene,nproc)
    return out
        

###########################################################################################
###########################################################################################
###########################################################################################
        
class CalcNode(object):
    def __init__(self,los,s,norestene):
        self.los = los
        self.s = s
        self.norestene = norestene
        self.out = None

    def calculate(self):
        from ffpopt.geom.GeomOpt import GeomOpt,GeomOpt_SinglePoint
        import copy
        if self.los.args.no_opt:
            self.out = copy.deepcopy(self.s)
        else:
            self.out = GeomOpt(self.los,self.s)
        tmp = copy.deepcopy(self.out)
        if self.norestene:
            tmp.restraints = None
            tmp.constraints = None
        tmp = GeomOpt_SinglePoint(self.los,tmp)
        self.out.Update( tmp.get_potential_energy(), tmp.get_positions(), tmp.get_forces() )
        self.los.calc = None
        #self.los = None

        
def _run_node( node ):
    node.calculate()
    return node


def is_mpi_worker():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return size > 1 and rank > 0


def is_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    size = comm.Get_size()
    return size > 1


def ParallelGeomOpt_threads(los,norestene,nproc):
    import concurrent.futures
    import multiprocessing
    from ffpopt.Struct import ListOfStruct

    nodes = [ CalcNode(los,s,norestene) for s in los ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
        results = list(executor.map(_run_node, nodes))
    return ListOfStruct( [ node.out for node in results ] )


# -------------------------------------------------------------------------
# WORKER SIDE CAR ENVIRONMENT
# -------------------------------------------------------------------------
# Worker-level global storage variables
_WORKER_LOS = None
_WORKER_NORESTENE = None

def _worker_init(los, norestene):
    """
    Runs ONCE per worker process when it joins the cluster.
    Safely stores context metadata in worker memory space.
    """
    global _WORKER_LOS, _WORKER_NORESTENE
    _WORKER_LOS = los
    _WORKER_NORESTENE = norestene

def _run_node_mpi(s):
    """
    Executes on a single structure using pre-cached environment context.
    """
    global _WORKER_LOS, _WORKER_NORESTENE
    
    # Instantiate the node locally using the cached background variables
    node = CalcNode(_WORKER_LOS, s, _WORKER_NORESTENE)
    node.calculate()
    
    # Return ONLY the structure output payload to minimize MPI data footprint
    return node.out

# -------------------------------------------------------------------------
# TARGET MPI FUNCTION
# -------------------------------------------------------------------------
def ParallelGeomOpt_mpi(los, norestene):
    """
    Asynchronous streaming MPI implementation.
    Safely captures the existing mpirun worker pool.
    """
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    from ffpopt.Struct import ListOfStruct
    
    # MPICommExecutor partitions COMM_WORLD.
    # Workers enter a passive processing loop inside the 'with' block context.
    # Only Rank 0 exits the block to submit jobs via the executor.
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            # Set up global contextual environments on worker memory pools
            # Note: MPICommExecutor does not support the 'initializer' parameter, 
            # so we map the initialization function across workers manually.
            num_workers = MPI.COMM_WORLD.Get_size() - 1
            los.calc = None
            if num_workers > 0:
                list(executor.map(_worker_init, [los]*num_workers, [norestene]*num_workers))

            # Dynamically stream data chunks to achieve perfect load balancing
            results_iterator = executor.map(_run_node_mpi, list(los), chunksize=1)
            final_outputs = list(results_iterator)
            out = ListOfStruct(final_outputs)
            return out
    return None
