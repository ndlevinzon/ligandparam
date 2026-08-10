#!python

if __name__ == "__main__":
    import argparse
    from ffpopt.Options import AddStandardOptions
    from ffpopt.scan.WaveFront import run_dihed_wavefront

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Perform a relaxed dihedral scan""",
    )

    parser.add_argument("-d", "--dihed",
        help="4 comma-separated 0-based integers defining the torsion",
        required=True, type=str)

    parser.add_argument("-D", "--delta",
        help="Scan delta (degrees). Default: 10",
        default=10, type=int)

    parser.add_argument("-i", "--inp",
        help="Input json file",
        required=True, type=str)

    parser.add_argument("-o", "--out",
        help="Output json file",
        required=True, type=str)

    parser.add_argument("-n", "--nproc",
        help="Number of optimizations to run at a time. Default: 1 (maximum: 2)",
        default=1, type=int)

    parser.add_argument("--wf_max_levels",
        help="Maximum number of levels to explore in the wavefront. Default: -1. Set to -1 for unlimited.",
        default=-1, type=int)

    parser.add_argument("--wf_starting_nodes",
        help="Number of initial nodes to start with. Default: 1.",
        default=1, type=int)

    parser.add_argument("--wf_num_conformers",
        help="Number of conformers to generate at each level. Default: 0 (no conformers).",
        default=0, type=int)

    parser.add_argument("--wf_change_theory",
        help="Change the theory of the calculator for all nodes in the wavefront calculation. Starts from previous highest level. Use with caution.",
        action="store_true")

    parser.add_argument("--wf_theory_stride",
        help="Stride for creating new starting nodes (default: 1, every node). Only used with --wf_change_theory.",
        default=1, type=int)

    parser.add_argument("--wf_alt_starting_checkpoint",
        help="Use an alternative checkpoint file to start the wavefront calculation. Default: None.",
        default=None, type=str)

    parser.add_argument("--wf_convergence_threshold",
        help="Energy convergence threshold (kcal/mol) a revisited angle must beat to keep spawning levels. Default: 0.01",
        default=0.01, type=float)

    AddStandardOptions(parser)
    args = parser.parse_args()

    run_dihed_wavefront(**vars(args))
