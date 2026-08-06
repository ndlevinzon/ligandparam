#!/usr/bin/env python3

if __name__ == "__main__":

    from pathlib import Path
    import argparse
    from ffpopt.Struct import ListOfStruct,Struct
    from collections import defaultdict as ddict
    
    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Split systems within a json file into separate files (or extract a molecule by name)""" )

    
    parser.add_argument \
        ("--strict",
         help="Do not rename molecules",
         action='store_true')
    
    parser.add_argument \
        ("--out","-o",
         required=True,
         type=str,
         help="Output json file")

    parser.add_argument \
        ('inpjson',
         nargs='+',
         help='One or more json files to join')

    args = parser.parse_args()

    s=ListOfStruct([])
    for inp in args.inpjson:
        if not Path(inp).exists:
            raise Exception(f"File not found {inp}")
        x = ListOfStruct.from_file(inp)
        s.structs.extend(x)

    seen = ddict(int)
    for x in s.structs:
        name = x.data["name"]

        if name in seen:
            if args.strict:
                raise Exception(f"repeated name: {name}")
            if not args.strict:
                for i in range(998):
                    t = "s%03i"%(i)
                    if t not in seen:
                        name = t
                        break
        x.data["name"] = name
        seen[name] += 1

    s.save(args.out)

    
    
