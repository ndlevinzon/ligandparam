#!/usr/bin/env python3

def AddGeomOptOptions(parser):
   """Add options for geometry optimization to the parser.

   Parameters
   ----------
   parser : argparse.ArgumentParser
         The argument parser to which the options will be added.
   
   Returns
   -------
   None

   """
   
   parser.add_argument \
      ("--no-opt",
       help="Skip all geometry optimizations and simply perform an energy evaluation",
       action='store_true')
   
   parser.add_argument \
      ("--geometric-opt",
       help="If present, prefer geomeTRIC for geometry optimization "
            "(falls back to ASE BFGS/LBFGS/FIRE on failure). "
            "Without this flag, ASE is tried first.",
       action='store_true')

   parser.add_argument \
      ("--ase-opt-tol",
       help="The ASE geometry optimization tolerance. Default: 0.01 eV/A",
       default=0.01,
       type=float)
   
   parser.add_argument \
       ("--geometric-maxiter",
        help="Maximum number of optimization steps. Default: 500",
        default=500,
        type=int)
   
   parser.add_argument \
       ("--geometric-coordsys",
        help="Coordinate system. Default: tric",
        default='tric',
        type=str)
   
   parser.add_argument \
       ("--geometric-converge",
        help="Optimization tolerance(s). Default: 'set GAU_TIGHT'. Other named options include: 'set GAU_LOOSE', 'set GAU', 'set GAU_TIGHT', 'set GAU_VERYTIGHT'",
        default='set GAU',
        type=str)

   parser.add_argument \
       ("--geometric-enforce",
        help="Constraint enforcement tolerance. Default: 0.1",
        default=0.1,
        type=float)

   parser.add_argument \
       ("--geometric-ini",
        help="Path to the geometric INI file.",
        default=None,
        type=str)

   parser.add_argument \
       ("--soft-dihed-restraint",
        help=(
            "Use an AFFDO-style soft harmonic dihedral restraint "
            "(default k=500 kcal/mol/rad^2, +/-0.5 deg tolerance band) instead of "
            "a hard IC dihedral constraint during wavefront opts. Works with "
            "geomeTRIC via ASE RestrainedCalculator."
        ),
        action="store_true")

   parser.add_argument \
       ("--soft-dihed-k",
        help="Soft dihedral spring constant in kcal/mol/rad^2. Default: 500",
        default=500.0,
        type=float)

   parser.add_argument \
       ("--soft-dihed-kmax",
        help=(
            "Cap for k-doubling when the soft dihedral is out of band "
            "(kcal/mol/rad^2). Default: 8000. After kmax, one hard-IC opt "
            "starts from the last soft coordinates."
        ),
        default=8000.0,
        type=float)

   parser.add_argument \
       ("--soft-dihed-tol",
        help="Soft dihedral tolerance band in degrees. Default: 0.5",
        default=0.5,
        type=float)



def AddModelOptions(parser):
   """ Add options for the model/energy calculator to the parser.
   
   Parameters
   ----------
   parser : argparse.ArgumentParser
         The argument parser to which the options will be added.
      
   Returns
   -------
   None
   
   """
   
   parser.add_argument \
       ("-m","--model",
        help="Energy calculator: sander, xtb, qdpi2, dpmlp, mace, aimnet2, aimnet2_wb97m(aimnet2 is an alias for aimnet2_wb97m), aimnet2_b973c, aimnet2_qr, ani1x, ani2x, ani1ccx, pyscfneo, theory/basis (psi4). dpmlp looks for dp_test.pt in the current directory. Default: sander",
        default="sander",
        type=str)

   parser.add_argument \
       ("--mfile",
        help="Parameter file / network parameter file. Default: None",
        type=str,
        default=None)
   
   parser.add_argument \
       ("--psi4-memory",
        help="The available memory. This is sent to the psi4 calculator. Default: '1gb'",
        default='1gb',
        type=str)

   parser.add_argument \
       ("--psi4-num-threads",
        help="Total CPU-core budget for ab initio ESP (split across concurrent "
             "conformers). Also used for Gaussian %%nproc. Default: 4",
        default=4,
        type=int)

   parser.add_argument \
      ("--cpu",
       help="This will adjust the environmental variables to force machine learning model evaluation on the cpus. Specifically, it exports JAX_PLATFORMS='cpu' and CUDA_VISIBLE_DEVICES=-1",
       action='store_true')



def AddStandardOptions(parser):
   """ Add standard options to the parser.
   
   Parameters
   ----------
   parser : argparse.ArgumentParser
         The argument parser to which the options will be added.
      
   Returns
   -------
   None
   
   """
   AddModelOptions(parser)
   AddGeomOptOptions(parser)




def AddConstraintAndRestraintOptions(parser):
   """ Add options for constraints and restraints

   Parameters
   ----------
   parser : argparse.ArgumentParser
         The argument parser to which the options will be added.
      
   Returns
   -------
   None
   
   """
   
   parser.add_argument \
      ("--constrain",
       help="comma-separated list of 2,3,or 4 0-based integers. The list can be appended with =value to specify a value. If not given, then the input coordinates are used to assign the value. This can be used multiple times.",
       required=False,
      type=str,
      action='append')

   parser.add_argument \
      ("--restrain-bond",
       help="str, format:'k,idx1,idx2[=value]', where idx1,idx2 are 0-based integers and value can be missing",
       required=False,
       type=str,
       action='append')

   parser.add_argument \
      ("--restrain-angle",
       help="str, format:'k,idx1,idx2,idx3[=value]', where idx1,idx2,idx3 are 0-based integers and value can be missing. Value should be degrees",
       required=False,
       type=str,
       action='append')

   parser.add_argument \
      ("--restrain-dihed",
       help="str, format:'k,idx1,idx2,idx3,idx4[=value]', where idx1,idx2,idx3,idx4 are 0-based integers and value can be missing. Value should be degrees",
       required=False,
       type=str,
       action='append')
   
   parser.add_argument \
      ("--restrain-r12",
       help="str, format:'k,idx1,idx2,idx3,idx4[=value]', where idx1,idx2,idx3,idx4 are 0-based integers and value can be missing",
       required=False,
       type=str,
       action='append')
   
   parser.add_argument \
      ("--restrain-puckerx",
       help="str, format:'k,idx1,idx2,idx3,idx4,idx5[=value]', where idx1,idx2,idx3,idx4,idx5 are 0-based integers and value can be missing",
       required=False,
       type=str,
       action='append')
   
   parser.add_argument \
      ("--restrain-puckery",
       help="str, format:'k,idx1,idx2,idx3,idx4,idx5[=value]', where idx1,idx2,idx3,idx4,idx5 are 0-based integers and value can be missing",
       required=False,
       type=str,
       action='append')
   
   parser.add_argument \
      ("--restrain-rms",
       help="str, format:'[filename,]k,idx1,idx2,[...]', where idx1,idx2,[...] are 0-based integers. If filename is missing, then the input coordinates are adopted. If filename is present, it must be a XYZ file. The indexes reflect those atoms that contribute as nonzero mass-weighted RMS contribution.",
       required=False,
       type=str,
       action='append')
   
   parser.add_argument \
      ("--restrain-twist",
       help="str, format:'[filename,]k,idx1,idx2', where idx1,idx2 are 0-based integers. If filename is missing, then the input coordinates are adopted. If filename is present, it must be a XYZ file. The indexes reflect the bond that is being twisted, which will create separate rms restraints for each half of the bond.",
       required=False,
       type=str,
       action='append')

   

def ParseConstraintAndRestraintOptions(args,struct=None):
   """ Return ConstraintList and RestraintList objects created from the argument list
   
   Parameters
   ----------
   args : argparse.Namespace
      The parsed arguments from the command line.
      
   Returns
   -------
   conlist : ffpopt.geom.Constraints.ConstraintList
      The object containing the list of constraints

   reslist : ffpopt.geom.Restraints.RestraintList
      The object containing the list of restraints
   """
   from pathlib import Path
   import numpy as np
   import ase
   import ase.io
   from ffpopt.geom.Constraints import ConstraintList
   from ffpopt.geom.Restraints import RestraintList, RmsRestraint, TwistRestraint
   from . AmberParm   import RotateBondMask
   
   atoms = None
   graph = None
   crds = None
   if struct is not None:
      atoms = struct.GetASEAtoms()
      graph = struct.GetGraph()
      crds  = struct.get_positions()

   conlist = ConstraintList.from_list_of_str(args.constrain)
   if atoms is not None:
      conlist.FillConstraints(atoms,force=False)

   nbond = 0
   if args.restrain_bond is not None:
      nbond  = len(args.restrain_bond)

   nangle = 0
   if args.restrain_angle is not None:
      nangle = len(args.restrain_angle)

   ndihed = 0
   if args.restrain_dihed is not None:
      ndihed = len(args.restrain_dihed)

   nr12 = 0
   if args.restrain_r12 is not None:
      nr12   = len(args.restrain_r12)

   npuckerx = 0
   if args.restrain_puckerx is not None:
      npuckerx= len(args.restrain_puckerx)
        
   npuckery = 0
   if args.restrain_puckery is not None:
      npuckery= len(args.restrain_puckery)
        
   nrms = 0
   if args.restrain_rms is not None:
      nrms   = len(args.restrain_rms)

   ntwist = 0
   if args.restrain_twist:
      ntwist = len(args.restrain_twist)

   nres = nbond+nangle+ndihed+nr12+nrms+ntwist+npuckerx+npuckery
   rlist = RestraintList([])
   if nbond > 0:
      tmp = RestraintList.from_list_of_str( [ "bond,"+x for x in args.restrain_bond ] )
      if tmp.rests[0].value is None and crds is not None:
         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
      rlist.extend(tmp.rests)
   if nangle > 0:
      tmp = RestraintList.from_list_of_str( [ "angle,"+x for x in args.restrain_angle ] )
      if tmp.rests[0].value is None and crds is not None:
         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
      rlist.extend(tmp.rests)
   if ndihed > 0:
      tmp = RestraintList.from_list_of_str( [ "dihed,"+x for x in args.restrain_dihed ] )
      if tmp.rests[0].value is None and crds is not None:
         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
      rlist.extend(tmp.rests)
   if nr12 > 0:
      tmp = RestraintList.from_list_of_str( [ "r12,"+x for x in args.restrain_r12 ] )
      if tmp.rests[0].value is None and crds is not None:
         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
      rlist.extend(tmp.rests)
   if npuckerx > 0:
      tmp = RestraintList.from_list_of_str( [ "puckerx,"+x for x in args.restrain_puckerx ] )
      if tmp.rests[0].value is None and crds is not None:
         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
      rlist.extend(tmp.rests)
   if npuckery > 0:
      tmp = RestraintList.from_list_of_str( [ "puckery,"+x for x in args.restrain_puckery ] )
      if tmp.rests[0].value is None and crds is not None:
         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
      rlist.extend(tmp.rests)
        
   if nrms > 0:
      for x in args.restrain_rms:
         xs = x.split(",")
         if Path(xs[0]).exists():
                
            geoms = ase.io.read(xs[0],index=":")
            atoms = geoms[0]
            rcrds = atoms.get_positions()
            k     = xs[1]
            idxs  = xs[2:]
         else:
            rcrds = np.array(crds,copy=True)
            k     = xs[0]
            idxs  = xs[1:]
         k  = float(k)
         idxs = [ int(i) for i in idxs ]
         wts = np.zeros( (len(eles),) )
         rcrds = rcrds.tolist()
         for i in idxs:
            wts[i] = ms[i]
         wts = wts.tolist()
         rlist.append( RmsRestraint(k,rcrds,wts) )

   if ntwist > 0:

      if graph is None:
         raise Exception("A struct is needed if twist restraint are used")

      for x in args.restrain_twist:
         xs = x.split(",")
         if Path(xs[0]).exists():
            geoms = ase.io.read(xs[0],index=":")
            atoms = geoms[0]
            rcrds = atoms.get_positions().tolist()
            k = float(xs[1])
            idxs = [int(xs[2]),int(xs[3])]
         else:
            rcrds = np.array(crds,copy=True).tolist()
            k = float(xs[0])
            idxs = [int(xs[1]),int(xs[2])]

         mask = RotateBondMask(graph,idxs)
         wts1 = np.array( ms, copy=True )
         wts2 = np.array( ms, copy=True )
         for i in range( len(mask) ):
            if i == idxs[0]:
               continue
            elif i == idxs[1]:
               continue
            elif mask[i] > 0:
               wts2[i] = 0
            else:
               wts1[i] = 0

         wts1 = wts1.tolist()
         wts2 = wts2.tolist()
         rest = TwistRestraint(k,rcrds,wts1,wts2)
         rlist.append(rest)
   return conlist,rlist


def DeleteConstraintAndRestraintFromStruct(struct,conlist,reslist):
   """Delete the constraints and restraints within the struct

   Parameters
   ----------
   struct : ffpopt.Struct.Struct
       The structure object to modify
   conlist : ffpopt.geom.Constraints.ConstraintList
       The list of constraints that should not appear within the Struct object
   reslist : ffpopt.geom.Restraints.RestraintList
       The list of restraints that should not appear within the Struct object

   Returns
   -------
   None
   """
   from ffpopt.geom.Constraints import Constraint
   from ffpopt.geom.Restraints import RestraintList
   idels = []
   for i in range(len(struct.data["constraints"])):
      x = struct.data["constraints"][i]
      c = Constraint.from_dict(x)
      for con in conlist:
         if c.is_same(con):
            idels.append(i)
            break
   idels.sort(reverse=True)
   for idel in idels:
      del struct.data["constraints"][idel]

   iress = []
   cs = RestraintList.from_list_of_dict(struct.data["restraints"])
   for i in range(len(cs)):
      x = cs[i]
      for con in reslist:
         if x.is_same(con):
            iress.append(i)
            break
   iress.sort(reverse=True)
   for ires in iress:
      del struct.data["restraints"][ires]
   
   
   
def GetStandardOptions(args):
   """ Get the standard options from the parsed arguments.
   
   Parameters
   ----------
   args : argparse.Namespace
      The parsed arguments from the command line.
      
   Returns
   -------
   dict
      A dictionary containing the standard options.
      
   """

   geometric_kwargs = {}
   if all(
      hasattr(args, name)
      for name in (
         "geometric_coordsys",
         "geometric_maxiter",
         "geometric_converge",
         "geometric_enforce",
      )
   ):
      geometric_kwargs = {
         "coordsys": str(args.geometric_coordsys),
         "maxiter": str(args.geometric_maxiter),
         "converge": str(args.geometric_converge),
         "enforce": str(args.geometric_enforce),
      }

   
   psi4_kwargs = {}
   if hasattr(args, "psi4_memory") and hasattr(args, "psi4_num_threads"):
      psi4_kwargs = {
         "memory": str(args.psi4_memory),
         "num_threads": str(args.psi4_num_threads),
      }


   extra_args = { "geometric": geometric_kwargs,
                  "psi4": psi4_kwargs }

   return extra_args




def ModelIsPsi4(model):
   """ Check if the model is a Psi4 model.

   Parameters
   ----------
   model : str
      The model name to check.

   Returns
   -------
   bool
      True if the model is a Psi4 model, False otherwise.
   
   """
   m = model.upper()
   ispsi4 = False
   if ".pb" in m:
      pass
   elif ".pt" in m:
      pass
   elif ".model" in m:
      pass
   elif "XTB" in m:
      pass
   elif "QDPI2" in m:
      pass
   elif "DPMLP" in m:
      pass
   elif "MACE" in m:
      pass
   elif "AIMNET" in m:
      pass
   elif "PSYCFNEO" in m:
      pass
   elif "/" in m:
      ispsi4 = True
   return ispsi4

def configure_geometric_logging(ini_path: str = None):
    """Locate the packaged geometric_log.ini or return a caller-provided path.

    Prefer importlib.resources.files (modern API). Fall back to pkg_resources.resource_filename
    for older installations. Raises FileNotFoundError if not found and no ini_path provided.
    """
    from pathlib import Path
    
    # If caller supplied an explicit path, return it (allow non-existent path to be handled by caller)
    if ini_path:
        return Path(ini_path)

    # Try importlib.resources (preferred)
    try:
        import importlib.resources as resources  # Python 3.9+
        candidate = Path(resources.files("ffpopt") / "pkgdata" / "files" /"geometric_log.ini")
        if candidate.exists():
            return candidate
        # if resources.files returned a Traversable inside a zip, convert to a temp file if necessary
        # but normally .exists() will be True for installed package data on filesystem.
    except Exception:
        pass

    # Fallback to pkg_resources.resource_filename if available
    try:
        import pkg_resources
        candidate_path = pkg_resources.resource_filename("ffpopt", "pkgdata/files/geometric_log.ini")
        return Path(candidate_path)
    except Exception:
        pass

    raise FileNotFoundError("geometric_log.ini not found in package; pass explicit ini_path to configure_geometric_logging()")


 
def argparse2geometric(jsonfname,args):
   """ Convert argparse arguments to a geometric-optimize command.
   
   Parameters
   ----------
   jsonfname : str
      The json filename
   args : argparse.Namespace
      The parsed arguments from the command line.

   Returns
   -------
   list
      A list of command-line arguments for the geometric-optimize command.
   
   """
   #import json
   
   kwargs = GetStandardOptions(args)
   geok = kwargs["geometric"]
   psik = kwargs["psi4"]
   model = args.model.upper()

   ini_path = configure_geometric_logging(args.geometric_ini)
   
   if args.geometric_ini is not None:
      if len(str(args.geometric_ini)) == 0:
         ini_path = ""
         
   geoopts = []
   for key in geok:
      val = geok[key]
      geoopts.append( "--%s"%(key) )
      if key == "converge":
         geoopts.extend( geok[key].split() )
      else:
         geoopts.append( geok[key] )

   asek = { "mode": model, "inp": jsonfname }
   
   if args.mfile is not None:
      asek["mfile"] = args.mfile

      
   if ModelIsPsi4(model):
      for key in psik:
         asek[key] = str(psik[key])
         

   # if restraintfile is not None and len(restraints) > 0:
   #    data = {}
   #    twistrst = []
   #    for rst in restraints:
   #       if rst.rsttype == "twistrst":
   #          twistrst.append( [rst.filename,rst.wts1,rst.wts2] )
   #    if len(twistrst) > 0:
   #       data["twistrst"] = twistrst
   #    with open(str(restraintfile), "w") as json_file:
   #       json.dump(data, json_file, indent=4)
   #    asek["restraintfile"] = str(restraintfile)
      
   
   asestr = ",".join( [ '"%s": "%s"'%(key,asek[key]) for key in asek ])
   asestr = "{%s}"%(asestr)

   # Invoke via ffpopt.geom.Geometric so constrained IC recovery cannot
   # abort on geomeTRIC's unsupported Cartesian fallback (see that module).
   import sys
   cmds = [sys.executable, "-m", "ffpopt.geom.Geometric",
           "--engine", "ase",
           "--ase-class", "ffpopt.Struct.RestCalculator",
           "--ase-kwargs", asestr ] + geoopts
   
   if ini_path is not None:
      ini_path=str(ini_path)
      if len(ini_path) > 0:
         cmds += [ "--logINI", ini_path]

   return cmds
