#!/usr/bin/env python3

if __name__ == "__main__":
    from ffpopt.geom.GeomOpt import FwdRevDihedScan
    from ffpopt.AmberParm import parmed2graph
    from ffpopt.Options import AddStandardOptions
    from ffpopt.geom.Constraints import Constraint
    import ase.io
    import numpy as np
    import argparse
    from pathlib import Path
    from ffpopt.constants import AU_PER_KCAL_PER_MOL
    from ffpopt.constants import AU_PER_ELECTRON_VOLT
    from ffpopt.Struct import ListOfStruct

    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Perform a relaxed dihedral scan""" )
    
    parser.add_argument \
        ("-i","--inp",
         help="Input json file",
         required=False,
         type=str)

    parser.add_argument \
        ("-o","--out",
         help="Output json file",
         required=True,
         type=str)
    
    parser.add_argument \
        ("-d","--dihed",
         help="4 comma-separated 0-based integers defining the torsion",
         required=True,
         type=str)
    
    parser.add_argument \
        ("--delta",
         help="Scan delta (degrees). Default: 10",
         default=10,
         type=int)

    parser.add_argument \
        ("--parallel",
         help="Run foward and reverse scans in parallel",
         action='store_true')

    
    
    AddStandardOptions(parser)

    args = parser.parse_args()
    inps = ListOfStruct.from_file(args.inp)
    inps.SetArgs(args)

    inps.structs = [ inps.structs[0] ]
    
    con = Constraint.from_str(args.dihed,graph=inps.structs[0].GetGraph())
    con.value = None

    idel = None
    for i in range(len(inps.structs[0].data["constraints"])):
        x = inps.structs[0].data["constraints"][i]
        c = Constraint.from_dict(x)
        if c.is_same(con):
            idel = i
            break
    if idel is not None:
        del inps.structs[0].data["constraints"][idel]
    #print(inps.structs[0].data["constraints"])
    
    # extracons = []
    # if args.constrain is not None:
    #     for c in args.constrain:
    #         extracons.append( Constraint.from_str(c,graph=stdargs.graph) )
    
    if args.delta < 1:
        raise Exception("--delta must be > 0")
    
    sched = np.arange(0,360,args.delta)

    # parallel = False
    # if args.nproc > 1:
    #     parallel = True
        
    #mingeom,scan = FwdRevDihedScan(stdargs.atoms,stdargs,con,extracons,sched,parallel=parallel)

    scan = FwdRevDihedScan(inps,inps.structs[0],con,sched,parallel=args.parallel)
    
    scan.save(args.out)
    
    #ase.io.write(args.oscan,scan,format="extxyz")

    KCAL_PER_EV = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()

    angles=[]
    enes = []
    for s in scan:
        angles.append(int(s.data["name"].replace("ang","")))
        enes.append(s.get_potential_energy() * KCAL_PER_EV)
    enes_noshift = np.array(enes)
    enes = enes_noshift - np.amin(enes_noshift)
    
    dat = Path(args.out).with_suffix(".dat")
    fh = open(dat,"w")
    for a,e,n in zip(angles,enes,enes_noshift):
        fh.write("%12.2f %20.12e %20.12e\n"%(a,e,n) )
        
    fh.write("%12.2f %20.12e %20.12e\n"%(angles[0]+360,enes[0],enes_noshift[0]) )
    fh.close()


    
