#!/usr/bin/env python3

if __name__ == "__main__":

    import argparse
    from ffpopt.Reader import ConvertXyz2Mol2
    
    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Convert a xyz file to a mol2 file""" )
    

    parser.add_argument \
        ("-i","--inp",
         help="xyz file",
         type=str,
         required=True)

    parser.add_argument \
        ("-o","--out",
         help="mol2 file",
         type=str,
         required=True)

    parser.add_argument \
        ("--charge",
         help="net charge (default: 0)",
         type=int,
         default=0)

    args = parser.parse_args()

    if not args.inp.endswith(".xyz"):
        raise Exception(f"{args.inp} should end with .xyz")

    if not args.out.endswith(".mol2"):
        raise Exception(f"{args.out} should end with .mol2")

    ConvertXyz2Mol2(args.inp,args.charge,mol2_filename=args.out)

    
    
    

    
    
