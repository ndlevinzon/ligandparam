#!/usr/bin/env python3


if __name__ == "__main__":

    import argparse
    from pathlib import Path
    from ffpopt.confsearch import ConformerSearch
    
    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Read or create a structure and search for conformations""")
    

    parser.add_argument \
        ("--nconf",
         help="Initial number of conformations to geometry optimize. (default: 100)",
         type=int,
         default=100,
         required=False)

    parser.add_argument \
        ("--nkeep",
         help="The number of low-energy clustered conformations to save. (default: 5)",
         type=int,
         default=5,
         required=False)

    parser.add_argument \
        ("--searchmaxit",
         help="The number of UFF/MMFF94 geometry optimization steps. (default: 250)",
         type=int,
         default=250,
         required=False)

    
    parser.add_argument \
        ("-t","--searchrmstol",
         help="The RMS tolerance used to cluster the structures in Angstrom (default: 0.5)",
         type=float,
         default=0.5,
         required=False)


    parser.add_argument \
        ("--uff",
         help="If present, optimize with UFF (recommended if metals are present). The default is MMFF94 (recommended for small molecule organic drug-like molecules)",
         action='store_true')

    parser.add_argument \
        ("--quiet",
         help="If present, do not print informative messages to stdout",
         action='store_true')

    
    parser.add_argument \
        ("-o","--out",
         help="Output json file (default: confs.json)",
         type=str,
         default="confs",
         required=False)


    parser.add_argument \
        ('name_or_string',
         metavar='name_or_string',
         type=str,
         help='Either a filename, Inchi string, or smiles string')
    
    args = parser.parse_args()

    if Path(args.out).suffix != ".json":
        raise Exception("output file must end in .json")
    
    
    confs = ConformerSearch\
        (args.name_or_string,
         args.out,
         args.nconf,args.nkeep,
         not args.uff,
         args.searchmaxit,
         args.searchrmstol,
         args.quiet)

    
