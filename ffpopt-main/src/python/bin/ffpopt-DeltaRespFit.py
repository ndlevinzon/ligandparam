#!/usr/bin/env python3


if __name__ == "__main__":

    from ffpopt.Options import AddModelOptions
    from ffpopt.RespFit import RunDeltaRespFit
    import argparse

    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Perform a geometry optimization""" )

    
    parser.add_argument \
        ("--native",
         type=str,
         required=True,
         help="The json (or mol2) file describing the native system. If it is a json, then all structures within the json are assumed to be conformations used to perform a multi-conformer fit.")

    
    parser.add_argument \
        ("--modified",
         type=str,
         required=True,
         help="The json (or mol2) file describing the native system. If it is a json, then all structures within the json are assumed to be conformations used to perform a multi-conformer fit.")

    
    parser.add_argument \
        ("--out",
         type=str,
         required=True,
         help="Output json or mol2 file.")

    
    parser.add_argument \
        ("--respf",
         action='store_true',
         help="If present, perform resp fit using the 'resp' program (expected to be in path)")

    
    parser.add_argument \
        ("--espaloma",
         action='store_true',
         help="If present, then use conformer-independent espaloma charges from https://doi.org/10.1021/acs.jpca.4c01287")

    
    parser.add_argument \
        ("--hilfiker",
         action='store_true',
         help="If present, then use conformer-independent hilfiker charges from https://doi.org/10.48550/arXiv.2512.13579")

    
    parser.add_argument \
        ("--program",
         type=str,
         required=False,
         default="psi4",
         help="Ab initio executable. Default: psi4. This could also be gaussian; e.g., --program=g16")


    parser.add_argument \
        ("--resp-a",
         type=float,
         required=False,
         default=0.001,
         help="Hyperbolic penalty prefactor. Default: 0.001. The penalty is pen= a * sum_i ( sqrt( qi**2 + b**2 ) - b ), where i loops over all heavy atoms.")

    
    parser.add_argument \
        ("--resp-b",
         type=float,
         required=False,
         default=0.1,
         help="Hyperbolic penalty width. Default: 0.1. The penalty is pen= a * sum_i ( sqrt( qi**2 + b**2 ) - b ), where i loops over all heavy atoms.")
    
    
    parser.add_argument \
        ("--density",
         type=float,
         required=False,
         default=6,
         help="The density of surface points. Default: 6 pts/Ang**2.")


    parser.add_argument \
        ("--digits",
         type=int,
         default=4,
         help="Round charges to this number of digits. Default: 4")

    
    parser.add_argument \
        ("--scosmo",
         type=float,
         default=0.0,
         help="Calculate fits in gas and cosmo environments and take a weighted average of the 2 charge vectors. The SCOSMO perturbation is a response to the MM charges fit to the gas phase ESP. Default: 0.0, which returns only the gas phase charges.")
    
    
    parser.add_argument \
        ("--ext-scale",
         type=float,
         required=False,
         default=1.1,
         help="UFF radius scale factor used to generate the external potential surface. Default: 1.1")

    
    parser.add_argument \
        ("--ext-density",
         type=float,
         required=False,
         default=2,
         help="Density of external potential points. Default: 2 pts/Ang**2")


    parser.add_argument \
        ("--native-cap",
         type=str,
         default=[],
         action='append',
         help="Amber-style atom selection string indicating a group of atoms whose charge-sum should not change. This can be used multiple times. This could potentially fail if an atom exists in multiple groups.")
    
    parser.add_argument \
        ("--modified-cap",
         type=str,
         default=[],
         action='append',
         help="Amber-style atom selection string indicating a group of atoms whose charge-sum should not change. This can be used multiple times. This could potentially fail if an atom exists in multiple groups.")
    
    parser.add_argument \
        ("--mcss-allow-mismatch",
         action='store_true',
         help="Default does not allow MCSS mapping between different elements. If present, then a carbon can map to a flourine, for example")
    
    parser.add_argument \
        ("--dont-modify-cap-charges",
         action='store_true',
         help="If present, then the atomic charges in the caps are the exact same as the input. The default is to replace the cap charges with the result of the constrained RESP fit. In both cases, the net charge of each cap is preserved.")
    
    parser.add_argument \
        ("--mcss-map",
         type=str,
         required=False,
         help="File containing lines like: AT1 => AT2 where AT1 is an atom name in native and AT2 is an atom name in 2. If present, then this mapping is used instead of MCSS")
    
    
    AddModelOptions(parser)
    args = parser.parse_args()

    print(dict(**vars(args)))

    los = RunDeltaRespFit(**vars(args))

    if ".mol2" in args.out:
        los[0].SaveCrds(args.out,fmt="MOL2")
    else:
        los.save(args.out)

    
