#!/usr/bin/env python3

if __name__ == "__main__":

    import argparse
    import numpy as np
    from ffpopt.Options import AddStandardOptions
    from ffpopt.DeltaPuckerFit import RunDeltaPuckerFit

    
    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="Perform a delta pucker fit\n\n"
          +"X^2 = sum_{scan} | (U_{mod,LL}(p+dp)-U_{native,LL}(p)) - (U_{mod,HL}-U_{native,HL}) |^2\n"
          +"U_{native,HL} : high-level energy of native pucker scan\n"
          +"U_{mod,HL} : high-level energy of modified pucker scan\n"
          +"U_{native,LL}(p) : low-level energy of native pucker scan using the standard torsion parameters, p\n"
          +"U_{mod,LL}(p+dp) : low-level energy of modified pucker scan using the modified parameters\n\n"
          +"X^2 = sum_{scan} | U_{mod,LL}(p+dp) - U_{ref} |^2\n"
          +"U_{ref} = U_{native,HL}-U_{mod,HL}-U_{native,LL}(p)\n\n")

    parser.add_argument\
        ("--hlnative",
         type=str,
         required=True,
         help="json file containing the native residue high-level scan")

    parser.add_argument\
        ("--hlmod",
         type=str,
         required=True,
         help="json file containing the modified residue high-level scan. The json file must also contain the name of the amber parm file for the modified system")

    parser.add_argument\
        ("--llnative",
         type=str,
         required=False,
         help="json file containing the native residue low-level scan")

    parser.add_argument\
        ("--llmod",
         type=str,
         required=False,
         help="json file containing the native residue low-level scan")
    
    parser.add_argument \
        ("--nlmaxiter",
         help="Maximum number of nonlinear optimization steps. Default: 1000",
         default=1000,
         type=int)

    parser.add_argument \
        ("--nlrhobeg",
         help="Initial parameter displacements. Default: 0.25 kcal/mol",
         default=0.25,
         type=float)
    
    parser.add_argument \
        ("--nltol",
         help="Tolerance on the parameter optimization. Default: 1.e-3",
         default=1.e-3,
         type=float)

    parser.add_argument \
        ("--nproc",
         help="Number of processors to use in parallel optimizations. Default: 1",
         default=1,
         type=int)

    parser.add_argument \
        ("--shm",
         help="Write temporary parm7 files to /dev/shm/foo.parm7",
         action='store_true')


    AddStandardOptions(parser)
    args = parser.parse_args()
    args.model="sander"

    #hlnative = ListOfStruct.from_file(args.hlnative)
    #hlmod    = ListOfStruct.from_file(args.hlmod)
    #llnative = ListOfStruct.from_file(args.llnative)

    RunDeltaPuckerFit(**vars(args))

    
    
