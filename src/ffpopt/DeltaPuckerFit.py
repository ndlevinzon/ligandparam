#!/usr/bin/env python3

from . GeomOpt import is_mpi_worker

###########################################################################################
###########################################################################################
###########################################################################################
        
# class Node(object):
#     def __init__(self,los,s,norestene):
#         self.los = los
#         self.s = s
#         self.norestene = norestene
#         self.out = None

#     def calculate(self):
#         from ffpopt.GeomOpt import GeomOpt,GeomOpt_SinglePoint
#         import copy
#         if self.los.args.no_opt:
#             self.out = copy.deepcopy(self.s)
#         else:
#             self.out = GeomOpt(self.los,self.s)
#         tmp = copy.deepcopy(self.out)
#         if self.norestene:
#             tmp.restraints = None
#             tmp.constraints = None
#         tmp = GeomOpt_SinglePoint(self.los,tmp)
#         self.out.Update( tmp.get_potential_energy(), tmp.get_positions(), tmp.get_forces() )
#         self.los.calc = None
#         #self.los = None

        
# def _run_node( node ):
#     node.calculate()
#     return node


# def is_mpi_worker():
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     return size > 1 and rank > 0


# def is_mpi():
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     #rank = comm.Get_rank()
#     size = comm.Get_size()
#     return size > 1




# def RerunLOS(los,norestene,nproc):
#     out = None
#     if is_mpi():
#         out = RerunLOS_mpi(los, norestene)
#     else:
#         out = RerunLOS_threads(los,norestene,nproc)
#     return out
        

# def RerunLOS_threads(los,norestene,nproc):
#     import concurrent.futures
#     import multiprocessing
#     from . Struct import ListOfStruct

#     nodes = [ Node(los,s,norestene) for s in los ]
#     with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
#         results = list(executor.map(_run_node, nodes))
#     return ListOfStruct( [ node.out for node in results ] )


# # -------------------------------------------------------------------------
# # WORKER SIDE CAR ENVIRONMENT
# # -------------------------------------------------------------------------
# # Worker-level global storage variables
# _WORKER_LOS = None
# _WORKER_NORESTENE = None

# def _worker_init(los, norestene):
#     """
#     Runs ONCE per worker process when it joins the cluster.
#     Safely stores context metadata in worker memory space.
#     """
#     global _WORKER_LOS, _WORKER_NORESTENE
#     _WORKER_LOS = los
#     _WORKER_NORESTENE = norestene

# def _run_node_mpi(s):
#     """
#     Executes on a single structure using pre-cached environment context.
#     """
#     global _WORKER_LOS, _WORKER_NORESTENE
    
#     # Instantiate the node locally using the cached background variables
#     node = Node(_WORKER_LOS, s, _WORKER_NORESTENE)
#     node.calculate()
    
#     # Return ONLY the structure output payload to minimize MPI data footprint
#     return node.out

# # -------------------------------------------------------------------------
# # TARGET MPI FUNCTION
# # -------------------------------------------------------------------------
# def RerunLOS_mpi(los, norestene):
#     """
#     Asynchronous streaming MPI implementation.
#     Safely captures the existing mpirun worker pool.
#     """
#     from mpi4py import MPI
#     from mpi4py.futures import MPICommExecutor
#     from . Struct import ListOfStruct
    
#     # MPICommExecutor partitions COMM_WORLD.
#     # Workers enter a passive processing loop inside the 'with' block context.
#     # Only Rank 0 exits the block to submit jobs via the executor.
#     with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
#         if executor is not None:
#             # Set up global contextual environments on worker memory pools
#             # Note: MPICommExecutor does not support the 'initializer' parameter, 
#             # so we map the initialization function across workers manually.
#             num_workers = MPI.COMM_WORLD.Get_size() - 1
#             los.calc = None
#             if num_workers > 0:
#                 list(executor.map(_worker_init, [los]*num_workers, [norestene]*num_workers))

#             # Dynamically stream data chunks to achieve perfect load balancing
#             results_iterator = executor.map(_run_node_mpi, list(los), chunksize=1)
#             final_outputs = list(results_iterator)
#             out = ListOfStruct(final_outputs)
#             return out
#     return None

###########################################################################################
###########################################################################################
###########################################################################################



def PrepareGrid(los):
    import ndfes
    
    xs = []
    ys = []
    for s in los:
        x,y = s.data["name"].split("~")
        xs.append(float(x))
        ys.append(float(y))
        
    xs = list(set(xs))
    ys = list(set(ys))
    xs.sort()
    ys.sort()

    dx  = xs[1]  - xs[0]
    xlo = xs[0]  - dx/2
    xhi = xs[-1] + dx/2
    
    dy  = ys[1]  - ys[0]
    ylo = ys[0]  - dy/2
    yhi = ys[-1] + dy/2

    dims = [ ndfes.SpatialDim(xlo,xhi,len(xs),False),
             ndfes.SpatialDim(ylo,yhi,len(ys),False) ]

    return ndfes.VirtualGrid( dims )


def GetSurfaceBins(los,grid,vals):
    import ndfes

    if len(los) != len(vals):
        raise Exception(f"size mismatch {len(los)} {len(vals)}")

    bins = {}
    for s,v in zip(los,vals):
        x,y = s.data["name"].split("~")
        x=float(x)
        y=float(y)
        bidx = grid.GetBinIdx([x,y])
        gidx = grid.CptGlbIdxFromBinIdx(bidx)
        bins[gidx] = ndfes.SpatialBin(bidx,value=v,stderr=0,entropy=0)
        
    return bins


def SaveSurface(los,grid,vals,fname):
    import ndfes
    bins = GetSurfaceBins(los,grid,vals)
    tmp = ndfes.MBAR( grid, bins )
    ndfes.SaveXml( fname, [tmp] )



def GetTmpParmFilename(tmpdir=None):
    import os
    from pathlib import Path
    from tempfile import mkstemp

    if tmpdir is None:
        tmpfile_loc = "./tmpfiles"
    else:
        tmpfile_loc = tmpdir
        
    if not Path(tmpfile_loc).is_dir():
        os.makedirs(tmpfile_loc, exist_ok=True)
        
    fd,tmpfname = mkstemp(dir=tmpfile_loc,prefix="tmp.",suffix=".parm7")
    if not os.isatty(fd):  # Check if fd is still valid
        os.close(fd)

    return tmpfname
    

def SaveTmpParm(p,tmpdir=None):
    """Save Amber parameter object to a temporary file within ./tmpfiles

    Parameters
    ----------
    p : parmed.AmberParm
        The parameter object

    Returns
    -------
    tmpfname : str
        The name of the (closed) temporary filename
    """
    
    # import os
    # from pathlib import Path
    # from tempfile import mkstemp
    
    # tmpfile_loc = "./tmpfiles"
    # if not Path(tmpfile_loc).is_dir():
    #     os.makedirs(tmpfile_loc, exist_ok=True)
        
    # fd,tmpfname = mkstemp(dir=tmpfile_loc,prefix="tmp.",suffix=".parm7")
    # if not os.isatty(fd):  # Check if fd is still valid
    #     os.close(fd)

    tmpfname = GetTmpParmFilename(tmpdir=tmpdir)
    p.save(tmpfname,overwrite=True)
    
    return tmpfname



def PrepareLLStructs(llfile,loshl,args,parmfile,save_llnative):
    import copy
    from . Struct import ListOfStruct
    from . GeomOpt import ParallelGeomOpt
    
    los = None
    llok = False
    if llfile is not None:
        if len(llfile) > 0:
            llok = True
            los = ListOfStruct.from_file( llfile )
            
    if los is None:
        los = copy.deepcopy(loshl)
        
    for s in los:
        s.data["parm"] = parmfile

    los.SetArgs(args)

    if llok and save_llnative is None:
        out = los
    else:
        out = ParallelGeomOpt(los,True,args.nproc)
        if save_llnative is not None and not is_mpi_worker():
            if isinstance(save_llnative, str):
                if out is None:
                    raise Exception("PrepareLLStructs tries to save a None object")
                out.save(save_llnative)
         
    return out
        

def EV2KCAL(x):
    from . constants import AU_PER_ELECTRON_VOLT
    from . constants import AU_PER_KCAL_PER_MOL
    kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    return x * kcal_per_ev




class DeltaPuckerFitType(object):
    def __init__(self,args,modparm=None,save_llnative=None):
        
        import numpy as np
        
        from . Struct import ListOfStruct
        from . Dihedrals import ParamType
        from . Dihedrals import ParamInstance
        from . Dihedrals import GetDihedClasses
        from . Dihedrals import FindDihedrals
        import ndfes

        self.args = args
        
        self.nprim = 1
        self.opt_phase = True
        self.fit_ddu   = True
        #self.trust_ll  = False

        self.los_hlnative = ListOfStruct.from_file( args.hlnative )
        self.los_hlmod    = ListOfStruct.from_file( args.hlmod )


        if True:
            self.los_llnative = PrepareLLStructs\
                (args.llnative,self.los_hlnative,args,
                 self.los_hlnative[0].data["parm"],
                 save_llnative)
            
            if modparm is None:
                modparm = self.los_hlmod[0].data["parm"]

            self.los_llmod = PrepareLLStructs\
                (args.llmod,self.los_hlmod,args,
                 modparm,False)
        else:
            self.los_llnative = ListOfStruct.from_file( args.llnative )
            self.los_llmod    = ListOfStruct.from_file( args.llmod )


        
        if is_mpi_worker():
            return
            
        ref = []
        self.llnat_enes = []
        self.hlnat_enes = []
        self.hlmod_enes = []
        for s in self.los_hlmod:
            
            name = s.data["name"]
            
            hlnative = self.los_hlnative.GetByName(name)
            if hlnative is None:
                raise Exception(f"Failed to find {name} in {args.hlnative}")
            
            llnative = self.los_llnative.GetByName(name)
            if llnative is None:
                raise Exception(f"Failed to find {name} in {args.llnative}")

            self.llnat_enes.append( llnative.data["energy"] )
            self.hlnat_enes.append( hlnative.data["energy"] )
            self.hlmod_enes.append( s.data["energy"] )

        
        self.llnat_enes = np.array( self.llnat_enes )
        self.hlnat_enes = np.array( self.hlnat_enes )
        self.hlmod_enes = np.array( self.hlmod_enes )

        self.llnat_enes -= min(self.llnat_enes)
        self.hlnat_enes -= min(self.hlnat_enes)
        self.hlmod_enes -= min(self.hlmod_enes)


        self.grid = PrepareGrid(self.los_hlmod)


        mods = []
        for ref in self.los_hlmod:
            name = ref.data["name"]
            s = self.los_llmod.GetByName(name)
            if s is None:
                raise Exception("Could not find structure named "
                                f"{name} in {args.llmod}")
            mods.append(s)

        self.los_llmod = ListOfStruct( mods )
        self.los_llmod.SetArgs(self.args)
        
        puckerx = None
        puckery = None
        for s in self.los_hlmod:
            name = s.data["name"]
            myx = None
            myy = None
            for r in s.restraints:
                if r.name == "puckerx":
                    if myx is None:
                        myx = r
                    else:
                        raise Exception(f"Multiple puckerx restraints on "
                                        +f"the same structure {name}")
                elif r.name == "puckery":
                    if myy is None:
                        myy = r
                    else:
                        raise Exception(f"Multiple puckerx restraints on "
                                        +f"the same structure {name}")
            if myx is None:
                raise Exception(f"Missing puckerx restraint {name}")
            if myy is None:
                raise Exception(f"Missing puckery restraint {name}")
            if puckerx is None and puckery is None:
                puckerx = myx
                puckery = myy
            else:
                if not puckerx.is_same(myx):
                    raise Exception(f"Inconsistent puckerx restraint {name}")
                if not puckery.is_same(myy):
                    raise Exception(f"Inconsistent puckery restraint {name}")

        idxs = puckerx.idxs

        d_idxs = [ [idxs[4],idxs[0],idxs[1],idxs[2]],
                   [idxs[1],idxs[2],idxs[3],idxs[4]],
                   [idxs[2],idxs[3],idxs[4],idxs[0]],
                   [idxs[3],idxs[4],idxs[0],idxs[1]],
                   [idxs[0],idxs[1],idxs[2],idxs[3]] ]

        if False:
            d_idxs = [ [idxs[4],idxs[0],idxs[1],idxs[2]],
                       [idxs[1],idxs[2],idxs[3],idxs[4]] ]

        
        self.parm = self.los_llmod[0].ReadAmberParm()

        d_masks = [ [ "@%s"%( self.parm.atoms[i].name ) for i in idxs ]
                    for idxs in d_idxs ]
        
        d_ptype = [ ParamType( "_".join([ "%s"%( self.parm.atoms[i].name )
                                         for i in d_idxs[k] ]),
                               self.nprim, d_masks[k] )
                    for k in range(len(d_idxs)) ]
        
        origparams = []

        d_pinstances = [ ParamInstance( d_ptype[k], [ d_masks[k] ], self.parm, False )
                         for k in range(len(d_idxs)) ]
                
        for k in range(len(d_idxs)):
            dfcns  = GetDihedClasses(idxs=d_idxs[k])[self.nprim][0]
            diheds = FindDihedrals(self.parm,d_idxs[k],impropers=False)
            for d in dfcns.prims:
                found=False
                for o in diheds:
                    if o.type.per == d.per:
                        origparams.append(o.type.phi_k)
                        found=True
                        break
                if not found:
                    origparams.append(0.)
            if self.opt_phase:
                for d in dfcns.prims:
                    found=False
                    for o in diheds:
                        if o.type.per == d.per:
                            origparams.append(o.type.phase)
                            found=True
                            break
                    if not found:
                        origparams.append(0.)

            d_pinstances[k].ptype.dfcns = dfcns
            
        self.pinstances = d_pinstances
        self.llmod_enes = self.GetEnergiesFromSander([0]*len(origparams))
        self.set_params(origparams)


        
    def SaveSurface(self,fname,vals):
        SaveSurface(self.los_hlmod,self.grid,vals,fname)
        

        
    def GetBeta(self):
        from . constants import AU_PER_ELECTRON_VOLT
        from . constants import BOLTZMANN_CONSTANT_AU
        kbT_au = BOLTZMANN_CONSTANT_AU() * 600
        kbT_eV = kbT_au / AU_PER_ELECTRON_VOLT()
        beta = 1 / kbT_eV
        return beta


    def GetWts(self):
        import numpy as np
        beta = self.GetBeta()
        wts = np.array([ np.exp(-beta*e) for e in self.hlmod_enes ])
        return wts


    def GetRefEnes(self):
        if self.fit_ddu:
            refenes = self.hlmod_enes - self.hlnat_enes
        else:
            refenes = self.hlmod_enes
        return refenes

    
    def GetModelEnes(self,x):
        import numpy as np
        from pathlib import Path
        if self.args.shm:
            try:
                #fname = Path("/dev/shm/foo.parm7")
                #fname = Path("/root/deleteme.parm7")
                fname = GetTmpParmFilename(tmpdir="/dev/shm")
                enes = self.GetSanderEnes(x,filename=fname)
            except Exception as e:
                print(e)
                fname = None
                enes = self.GetSanderEnes(x)
            if fname is not None:
                if Path(fname).is_file():
                    Path(fname).unlink()
            return enes
        else:
            enes = self.GetSanderEnes(x)

        return enes

    
    def GetSanderEnes(self,x,filename=None):
        import numpy as np
        enes = np.array(self.GetEnergiesFromSander(x,filename=filename))
        enes -= min(enes)
        if self.fit_ddu:
            enes -= self.llnat_enes
        return enes

    
    def GetDeltaEnes(self,x):
        refenes = self.GetRefEnes()
        enes    = self.GetModelEnes(x)
        delta   = enes-refenes
        return delta

    
    def GetWeightedDeltaEnes(self,x):
        delta = self.GetDeltaEnes(x)
        wts   = self.GetWts()
        return wts * delta
    

    def CptChisq(self,x):
        import numpy as np
        delta = self.GetDeltaEnes(x)
        wts   = self.GetWts()
        dsum = np.dot(wts,delta) / sum(wts)
        delta -= dsum
        dsum2 = np.dot(wts,delta) / sum(wts)
        return np.dot( delta, wts * delta )
  
        
    def GetEnergiesFromSander(self,x,filename=None):
        import os
        import copy
        import numpy as np
        from pathlib import Path
        from . GeomOpt import GeomOpt_SinglePoint

        origparms = []
        for s in self.los_llmod:
            origparms.append( s.data["parm"] )

        fname = None
        
        if x is not None:
            self.set_params(x)
            p = self.make_new_parm()
            if filename is None:
                fname = SaveTmpParm(p)
            else:
                p.save(filename,overwrite=True)
                fname = filename
            for s in self.los_llmod:
                s.data["parm"] = fname
            
        self.los_llmod.calc = None

        enes = []
        for i in range(len(self.los_llmod)):
            s = copy.deepcopy( self.los_llmod.structs[i] )
            s.data["restraints"] = None
            s.data["constraints"] = None
            s.restraints = None
            s.constraints = None
            t = GeomOpt_SinglePoint(self.los_llmod,s)
            enes.append( t.get_potential_energy() )
            
        for i,s in enumerate(self.los_llmod):
            s.data["parm"] = origparms[i]

        if filename is None and fname is not None:
            if Path(fname).is_file():
                Path(fname).unlink()
            
        return np.array(enes)

    
        
    def GetDihedralEnergies(self,x):
        import copy
        import numpy as np
        from . Geometry import CptDihed
        from . constants import AU_PER_ELECTRON_VOLT, AU_PER_KCAL_PER_MOL
        
        KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
        EV_PER_KCAL = 1/KCAL_PER_EV
        
        self.set_params(x)

        enes = []
        rcs = []
        for i in range(len(self.los_llmod)):
            crds = self.los_llmod[i].get_positions()
            e = 0.
            rc=[]
            for pinst in self.pinstances:
                for idxs in pinst.dihedidxs:
                    d = CptDihed(crds[idxs[0],:], crds[idxs[1],:],
                                 crds[idxs[2],:], crds[idxs[3],:])
                    rc.append(d)
                    e += pinst.ptype.dfcns.CptEne(d) * EV_PER_KCAL
            #e += self.llmod_enes[i] - self.llnat_enes[i])
            enes.append(e)
            rcs.append(rc)

        enes = np.array(enes)
        enes = (self.llmod_enes + enes)
        
        return enes

        
        
    def make_new_parm(self):
        """ Create a new Parm object with the dihedral parameters set according to the instances.
        
        Returns
        -------
        parmed.AmberParm
            A new Parm object with the dihedral parameters set according to the instances.
        
        """
        from parmed import DihedralType
        from . Dihedrals import ChangeDihedrals
        from . AmberParm import CopyParm

        p = CopyParm(self.parm)
        
        scee = 1.2
        scnb = 2.0
        for pinst in self.pinstances:
            xs = []
            for prim in pinst.ptype.dfcns.prims:
                per = prim.per
                ph = prim.phase
                fc = prim.fc
                x = DihedralType(fc, per, ph, scee, scnb)
                xs.append(x)
            for idxs in pinst.dihedidxs:
                ChangeDihedrals(p,idxs,xs)
        return p

    
    def find_pinstance(self,pname):
        """ Find a ParamInstance by its name.
        
        Parameters
        ----------
        pname : str
            The name of the parameter type to find.
        
        Returns
        -------
        ParamInstance or None
            The ParamInstance object if found, otherwise None.
            
        """
        p=None
        for pinst in self.pinstances:
            if pinst.ptype.name == pname:
                p = pinst
        return p

    
    def write_parmed(self,fname):
        """ Write the output file with the dihedral parameters.
        
        This method creates a new Parm object with the dihedral parameters set according to the instances,
        and writes the parameters to the specified output file using the WriteParmedScript function.
        
        Returns
        -------
        None
        
        """
        import copy
        from . Dihedrals import WriteParmedScript
        dfcns = []
        for pinst in self.pinstances:
            for idxs in pinst.dihedidxs:
                dfcn = copy.deepcopy(pinst.ptype.dfcns)
                dfcn.idxs = idxs
                dfcns.append(dfcn)
        WriteParmedScript(fname,self.parm,dfcns)

        
    def get_num_params(self):
        """ Get the total number of primitive parameters across all parameter types.
        
        Returns
        -------
        int
            The total number of primitive parameters across all parameter types.

        """

        n = 0
        for pinst in self.pinstances:
            n += pinst.ptype.nprim
        if self.opt_phase:
            for pinst in self.pinstances:
                n += pinst.ptype.nprim
        return n

    
    def get_params(self):
        """ Get the current values of the dihedral function coefficients.
        
        Returns
        -------
        numpy.ndarray
            A numpy array containing the current values of the dihedral function coefficients for all parameter types.
        
        """
        import numpy as np
        x=[]
        for pinst in self.pinstances:
            ptype = pinst.ptype
            dfcns = ptype.dfcns
            for prim in dfcns.prims:
                x.append( prim.fc )
            if self.opt_phase:
                for prim in dfcns.prims:
                    x.append( prim.phase )
        return np.array(x)


    def set_params(self,x):
        """ Set the dihedral function coefficients based on the provided array.
        
        Parameters
        ----------
        x : numpy.ndarray
            A numpy array containing the new values for the dihedral function coefficients.
        
        """
        ipar = 0
        for pinst in self.pinstances:
            ptype = pinst.ptype
            nprim = ptype.nprim
            ptype.dfcns.SetFCs( x[ipar:ipar+nprim] )
            ipar += nprim
            if self.opt_phase:
                ptype.dfcns.SetPhases( x[ipar:ipar+nprim] )
                ipar += nprim


    

def NonlinearSolve(args,finp):
    """ Perform a nonlinear optimization to fit dihedral parameters.
    
    Parameters
    ----------
    finp : FitInputType
        An instance of FitInputType containing the systems and profiles for fitting.

    Returns
    -------
    None
        The function modifies the FitInputType instance in place, setting the optimized dihedral parameters.
    
    """
    from scipy.optimize import minimize
    import numpy as np
    import copy

    #objfcn = NonlinearObjective(stdargs,udscans)
    

    x0 = finp.get_params()
    n = len(x0)

    xlo = np.array(x0[:],copy=True)
    xhi = np.array(x0[:],copy=True)

    ipar = 0
    for pinst in finp.pinstances:
        ptype = pinst.ptype
        nprim = ptype.nprim
        xlo[ipar:ipar+nprim] = x0[ipar:ipar+nprim] - 120.
        xhi[ipar:ipar+nprim] = x0[ipar:ipar+nprim] + 120.
        ipar += nprim
        if finp.opt_phase:
            xlo[ipar:ipar+nprim] = x0[ipar:ipar+nprim] - 120.
            xhi[ipar:ipar+nprim] = x0[ipar:ipar+nprim] + 120.
            ipar += nprim

    bounds = [ (lo,hi) for lo,hi in zip(xlo,xhi) ]

    for x,b in zip(x0,bounds):
        print("%12.3f [%12.3f %12.3f]"%(x,b[0],b[1]))
    
    finp.glb_niter = 1
    def ObjFcn(x,self):
        #print(x)
        chisq = self.CptChisq(x)
        print("%5i %20.11e"%(self.glb_niter,chisq))
        self.glb_niter += 1
        return chisq

    if True:
        res = minimize( ObjFcn, x0, args=(finp,),
                        method='COBYLA',
                        bounds=bounds,
                        options={ "rhobeg": args.nlrhobeg,
                                  "tol": args.nltol,
                                  "maxiter": args.nlmaxiter,
                                  "disp": True })
    else:
        res = minimize( ObjFcn, x0, args=(finp,),
                        method='L-BFGS-B',
                        bounds=bounds,
                        jac='3-point',
                        tol=1.e-15,
                        options={"maxiter": args.nlmaxiter,
                                 "disp": True })

    print(res)

    finp.set_params(res.x)
    return res
            


def RunDeltaPuckerFit\
        (*,
         hlnative: str,
         hlmod: str,
         llnative: str,
         llmod: str,
         nlrhobeg: float = 0.25,
         nlmaxiter: int = 300,
         nltol: float = 0.01,
         nproc: int = 1,
         shm: bool = False,
         **standard_kwargs):

    import os
    import shutil
    import argparse
    import subprocess
    from types import SimpleNamespace
    from . Options import AddStandardOptions
    from . Struct import ListOfStruct
    from . Dihedrals import FindDihedrals
    
    _p = argparse.ArgumentParser(add_help=False)
    AddStandardOptions(_p)
    std_defaults = vars(_p.parse_args([]))
    std_defaults["prefix"] = ""
    unknown = set(standard_kwargs) - set(std_defaults)
    if unknown:
        raise TypeError(
            f"Unexpected keyword argument(s): {sorted(unknown)}"
        )
    std = {**std_defaults, **standard_kwargs}


    args = SimpleNamespace(
        hlnative = hlnative,
        hlmod = hlmod,
        llnative = llnative,
        llmod = llmod,
        nlrhobeg = nlrhobeg,
        nlmaxiter = nlmaxiter,
        nltol = nltol,
        nproc = nproc,
        shm = shm,
        **std,
    )

    args.model = "sander"

    modparm = None
    save_llnative = "it000.json"

    for it in range(3):

        fitobj = DeltaPuckerFitType(args,modparm=modparm,
                                    save_llnative=save_llnative)

        args.llnative = save_llnative
        save_llnative = None
        
        base = "it%03i"%(it+1)
        modparm = f"{base}.parm7"

        if is_mpi_worker():
            continue
        
        res = NonlinearSolve(args,fitobj)

        fitobj.SaveSurface("plot_wts.xml",fitobj.GetWts()*20)
        fitobj.SaveSurface("plot_ref.xml",EV2KCAL(fitobj.GetRefEnes()))
        #fitobj.SaveSurface("plot_ene.xml",EV2KCAL(fitobj.GetModelEnes(res.x)))
        fitobj.SaveSurface("plot_san.xml",EV2KCAL(fitobj.GetSanderEnes(res.x,filename="opt.parm7")))
        fitobj.SaveSurface("plot_del.xml",abs(EV2KCAL(fitobj.GetDeltaEnes(res.x))))
        fitobj.SaveSurface("plot_wdl.xml",abs(EV2KCAL(fitobj.GetWeightedDeltaEnes(res.x))))


        shutil.copy(fitobj.los_llmod[0].data["parm"],f"{base}.inp.parm7")
        
        fitobj.write_parmed(f"{base}.py")
        
        subprocess.run(["python3",
                        f"{base}.py",
                        f"{base}.inp.parm7",
                        modparm],
                       check=True)

        os.remove(f"{base}.inp.parm7")
