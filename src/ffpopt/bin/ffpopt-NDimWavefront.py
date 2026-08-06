#!python

# import warnings

# warnings.filterwarnings(
#     "ignore", 
#     category=FutureWarning, 
#     message=r'.*ignore_bad_restart_file.*'
# )

    
if __name__ == "__main__":
    import argparse
    from ffpopt.Options import AddStandardOptions
    from ffpopt.Options import AddConstraintAndRestraintOptions
    from ffpopt.WaveFrontND import run_dihed_wavefront

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Perform a relaxed N-dimensional scan""",
    )

    parser.add_argument("-i", "--inp",
        help="Input json file",
        required=True, type=str)

    parser.add_argument("-o", "--out",
        help="Output json file",
        required=True, type=str)

    parser.add_argument\
        ("--condim",
         help="Constraint dimension range: 2 floats and an integer cooresponding to xmin, xmax, and dimsize '0.,360.,15'",
         type=str,
         default=[],
         action='append')

    parser.add_argument\
        ("--resdim",
         help="Restraint dimension range: 2 floats and an integer cooresponding to xmin, xmax, and dimsize '0.,360.,15'",
         type=str,
         default=[],
         action='append')
    
    parser.add_argument("-n", "--nproc",
        help="Number of optimizations to run at a time. Default: 1 (maximum: 2)",
        default=1, type=int)

    parser.add_argument("--mpi",
        help="Use mpi rather than threads",
        action='store_true')

    parser.add_argument("--wf-max-levels",
        help="Maximum number of levels to explore in the wavefront. Default: -1. Set to -1 for unlimited.",
        default=-1, type=int)

    parser.add_argument("--wf-change-theory",
        help="Change the theory of the calculator for all nodes in the wavefront calculation. Starts from previous highest level. Use with caution.",
        action="store_true")

    parser.add_argument("--wf-theory-stride",
        help="Stride for creating new starting nodes (default: 1, every node). Only used with --wf-change-theory.",
        default=1, type=int)

    parser.add_argument("--wf-alt-starting-checkpoint",
        help="Use an alternative checkpoint file to start the wavefront calculation. Default: None.",
        default=None, type=str)

    parser.add_argument("--wf-convergence-threshold",
        help="Energy convergence threshold (kcal/mol) a revisited angle must beat to keep spawning levels. Default: 0.01",
        default=0.01, type=float)

    AddStandardOptions(parser)
    AddConstraintAndRestraintOptions(parser)

    args = parser.parse_args()

    run_dihed_wavefront(**vars(args))
