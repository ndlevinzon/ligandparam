#!/usr/bin/env python3

if __name__ == "__main__":

    from pathlib import Path
    import argparse
    from ffpopt.Struct import ListOfStruct
    
    
    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Split systems within a json file into separate files (or extract a molecule by name)""" )

    parser.add_argument \
        ("--name",
         help="Extract structure with a given name",
         required=False,
         default=None,
         type=str)

    parser.add_argument \
        ("inp",
         help="Input json file",
         type=str)

    args = parser.parse_args()

    los = ListOfStruct.from_file(args.inp)

    dname = Path(args.inp).parent

    found=False
    names = []
    for s in los:
        name = s.data["name"]
        names.append(name)
        if args.name is not None:
            if name != args.name:
                continue
            found=True
        fname = Path(dname) / Path( name + ".json" )
        t = ListOfStruct( [s] )
        t.save(str(fname))
        if found:
            break
        
    if args.name is not None and not found:
        raise Exception(f"Structure named {args.name} not found in {names}")
        
