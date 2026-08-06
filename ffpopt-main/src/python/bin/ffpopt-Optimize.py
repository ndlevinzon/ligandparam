#!/usr/bin/env python3


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
#         if self.norestene:
#             self.out.restraints = None
#             self.out.constraints = None
#         out = GeomOpt_SinglePoint(self.los,self.out)
#         self.out.Update( out.get_potential_energy(), out.get_positions(), out.get_forces() )


# def _run_node( node ):
#     node.calculate()
#     return node



if __name__ == "__main__":
    import concurrent.futures
    
    from ffpopt.Options import AddStandardOptions
    from ffpopt.Struct import ListOfStruct
    from ffpopt.GeomOpt import ParallelGeomOpt, is_mpi_worker
    
    import argparse
    import multiprocessing
    from pathlib import Path

    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Perform a geometry optimization

recommended options:
  ffpopt-Optimize.py --geometric-opt --geometric-ini="" --model=qdpi2 --inp=inp.json --out=out.json""" )
    
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
        ("-n","--nproc",
         help="Number of optimizations to run at a time. Default 1",
         default=1,
         type=int)

    parser.add_argument \
        ("--test",
         help="Finitie difference test of forces",
         action='store_true')
    
    parser.add_argument \
        ("--test-delta",
         help="Finitie difference displacement (Angstroms) Default:1.e-3",
         default=1.e-3,
         type=float)

    parser.add_argument \
        ("--norestene",
         help="If present, then run a single point energy after optimization that excludes the restraint contributions",
         action='store_true')


    
    AddStandardOptions(parser)
    args = parser.parse_args()

    inps = None
    if not is_mpi_worker():
        inps = ListOfStruct.from_file(args.inp)
        inps.SetArgs(args)

    # if False:
    #     from io import StringIO
    #     mol2str = StringIO()
    #     inp = inps.structs[0]
    #     inp.SaveCrds(mol2str,fmt="xyz")
    #     mol2str = mol2str.getvalue()
    #     print(mol2str)
    #     exit(0)

    if args.test:
        from ffpopt.GeomOpt import CheckForces
        for inp in inps:
            print("")
            CheckForces(inps,inp,delta=args.test_delta)
            print("")
    else:

        # nodes = [ Node(inps,inp,args.norestene) for inp in inps ]
        # with concurrent.futures.ProcessPoolExecutor(max_workers=args.nproc) as executor:
        #     results = list(executor.map(_run_node, nodes))
        # outs = ListOfStruct( [ node.out for node in results ] )

        outs = ParallelGeomOpt(inps,args.norestene,args.nproc)
        if not is_mpi_worker():
            outs.save(args.out)
    
        
    
