#!/usr/bin/env python3


            
if __name__ == "__main__":
    import argparse
    import numpy as np
    from ffpopt.Options import AddStandardOptions
    from ffpopt.dihed.Dihedrals import FitInputType
    from ffpopt.dihed.Dihedrals import NonlinearSolve

    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Perform a general, multidihedral fit""" )

    parser.add_argument\
        ("inp",
         help="Json input file")

    parser.add_argument\
        ("--stride",
         default=1,
         type=int,
         help="Stride when reading structures. Default: 1")
    
    parser.add_argument \
        ("--nlmaxiter",
         help="Maximum number of nonlinear optimization steps. Default: 300",
         default=300,
         type=int)

    parser.add_argument \
        ("--nlrhobeg",
         help="Initial parameter displacements. Default: 0.25 kcal/mol",
         default=0.25,
         type=float)
    
    
    parser.add_argument \
        ("--nltol",
         help="Tolerance on the parameter optimization. Default: 0.01",
         default=0.01,
         type=float)

    parser.add_argument(
        "--fit-mode",
        choices=("barrier", "torsion", "full"),
        default=None,
        help=(
            "barrier=FCs only (default); torsion=FC+phase+period; "
            "full=FC+phase+period+scee/scnb. Overrides FFPOPT_FIT_MODE."
        ),
    )
    parser.add_argument(
        "--fit-backend",
        choices=("lsq", "lbfgsb", "jax"),
        default=None,
        help="Solver backend (default lsq for barrier; lbfgsb/jax for extended).",
    )
    parser.add_argument(
        "--fit-full",
        action="store_true",
        help="Shorthand for --fit-mode full",
    )
    parser.add_argument(
        "--barrier-only",
        action="store_true",
        help="Force FC-only fit (default behavior)",
    )
    parser.add_argument("--fit-phases", action="store_true", help="Also optimize phases")
    parser.add_argument(
        "--fit-periods", action="store_true", help="Also optimize periodicities"
    )
    parser.add_argument(
        "--fit-scee-scnb",
        action="store_true",
        help="Also optimize 1-4 scee/scnb scaling factors",
    )
    parser.add_argument("--scee", type=float, default=None, help="Initial scee (default 1.2)")
    parser.add_argument("--scnb", type=float, default=None, help="Initial scnb (default 2.0)")

    AddStandardOptions(parser)
    args = parser.parse_args()
    args.model="sander"

    from ffpopt.affdo.AffdoLog import print_affdo
    from ffpopt.dihed.ExtendedFit import apply_fit_flags_to_args

    apply_fit_flags_to_args(args)
    print_affdo(
        f"GenDihedFit flags: mode={args.fit_mode} backend={args.fit_backend} "
        f"opt_phase={args.opt_phase} opt_periods={args.opt_periods} "
        f"opt_scee_scnb={args.opt_scee_scnb} scee={args.scee:g} scnb={args.scnb:g}"
    )
    
    finp = FitInputType.from_file(args,args.inp)
    
    NonlinearSolve(args,finp)
    finp.write_output()
