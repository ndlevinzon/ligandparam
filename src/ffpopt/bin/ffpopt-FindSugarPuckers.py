#!/usr/bin/env python3

def IdentifySugarAtoms(s,ring):
    from ndfes.constants import GetAtomicNumber
    
    if ring is None:
        return None
    if len(ring[1]) != 5:
        return None

    graph = s.GetGraph()
    
    seen = ring[1]
    atoms = {}
    atoms["C1'"] = seen[0]
    atoms["C2'"] = seen[1]
    atoms["C3'"] = seen[2]
    atoms["C4'"] = seen[3]
    atoms["O4'"] = seen[4]

    
    
    seen = [ str(x) for x in seen ]
    
    # Find the C5' position
    atoms["C5'"] = None   
    for edge in graph.edges[str(atoms["C4'"])]:
        name = s.data["elements"][int(edge)]
        if name == "C5'":
            atoms["C5'"] = int(edge)
            seen.append(edge)
            break
    if atoms["C5'"] is None:
        for edge in graph.edges[str(atoms["C4'"])]:
            if edge in seen:
                continue
            else:
                ele = s.data["elements"][int(edge)]
                z = GetAtomicNumber(ele)
                if z > 1:
                    atoms["C5'"] = int(edge)
                    seen.append(edge)
                    break

    # Find the O5' position
    atoms["O5'"] = None
    if atoms["C5'"] is not None:
        for edge in graph.edges[str(atoms["C5'"])]:
            name = s.data["elements"][int(edge)]
            if name == "O5'":
                atoms["O5'"] = int(edge)
                seen.append(edge)
                break
        if atoms["O5'"] is None:
            for edge in graph.edges[str(atoms["C5'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    if z > 1:
                        atoms["O5'"] = int(edge)
                        seen.append(edge)
                        break
                   
    # Find the HO5' position
    atoms["HO5'"] = None
    if atoms["O5'"] is not None:
        for edge in graph.edges[str(atoms["O5'"])]:
            name = s.data["elements"][int(edge)]
            if name == "HO5'" or name == "P":
                atoms["HO5'"] = int(edge)
                seen.append(edge)
                break
        if atoms["HO5'"] is None:
            for edge in graph.edges[str(atoms["O5'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    if z == 1 or z == 15:
                        atoms["HO5'"] = int(edge)
                        seen.append(edge)
                        break
              
    # Find the O3' position
    atoms["O3'"] = None
    for edge in graph.edges[str(atoms["C3'"])]:
        name = s.data["elements"][int(edge)]
        if name == "O3'":
            atoms["O3'"] = int(edge)
            seen.append(edge)
            break
    if atoms["O3'"] is None:
        for edge in graph.edges[str(atoms["C3'"])]:
            if edge in seen:
                continue
            else:
                ele = s.data["elements"][int(edge)]
                z = GetAtomicNumber(ele)
                if z > 1:
                    atoms["O3'"] = int(edge)
                    seen.append(edge)
                    break

    # Find the HO3' position
    atoms["HO3'"] = None
    if atoms["O3'"] is not None:
        for edge in graph.edges[str(atoms["O3'"])]:
            name = s.data["elements"][int(edge)]
            if name == "HO3'":
                atoms["HO3'"] = int(edge)
                seen.append(edge)
                break
        if atoms["HO3'"] is None:
            for edge in graph.edges[str(atoms["O3'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    if z == 1:
                        atoms["HO3'"] = int(edge)
                        seen.append(edge)
                        break

    # Find the O3' position
    atoms["O2'"] = None
    for edge in graph.edges[str(atoms["C2'"])]:
        name = s.data["elements"][int(edge)]
        if name == "O2'":
            atoms["O2'"] = int(edge)
            seen.append(edge)
            break
    if atoms["O2'"] is None:
        for edge in graph.edges[str(atoms["C2'"])]:
            if edge in seen:
                continue
            else:
                ele = s.data["elements"][int(edge)]
                z = GetAtomicNumber(ele)
                if z > 1:
                    atoms["O2'"] = int(edge)
                    seen.append(edge)
                    break

    # Find the HO2' position
    atoms["HO2'"] = None
    if atoms["O2'"] is not None:
        for edge in graph.edges[str(atoms["O2'"])]:
            name = s.data["elements"][int(edge)]
            if name == "HO2'":
                atoms["HO2'"] = int(edge)
                seen.append(edge)
                break
        if atoms["HO2'"] is None:
            for edge in graph.edges[str(atoms["O2'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    if z == 1:
                        atoms["HO2'"] = int(edge)
                        seen.append(edge)
                        break

      
    # Find the N9/N1 position
    atoms["N9"] = None
    if atoms["C1'"] is not None:
        for edge in graph.edges[str(atoms["C1'"])]:
            name = s.data["elements"][int(edge)]
            if name == "N9" or name == "N1":
                atoms["N9"] = int(edge)
                seen.append(edge)
                break
        if atoms["N9"] is None:
            for edge in graph.edges[str(atoms["C1'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    # Hmm, why was this outside z>1 check?
                    #if atoms["N9"] is None:
                    #    atoms["N9"] = int(edge)
                    #    seen.append(edge)
                    if z > 1:
                        atoms["N9"] = int(edge)
                        seen.append(edge)
                        break

    # Find the C4/C2 position
    atoms["C4"] = None
    if atoms["N9"] is not None:
        for edge in graph.edges[str(atoms["N9"])]:
            name = s.data["elements"][int(edge)]
            if name == "C4" or name == "C2":
                atoms["C4"] = int(edge)
                seen.append(edge)
                break
        if atoms["C4"] is None:
            for edge in graph.edges[str(atoms["N9"])]:
                if edge in seen:
                    continue
                else:
                    # Count the number of heavy atoms connected to "edge"
                    nheavy = 0
                    for nbor in graph.edges[edge]:
                        ele = s.data["elements"][int(nbor)]
                        z = GetAtomicNumber(ele)
                        if z > 1:
                            nheavy += 1
                    if nheavy > 2:
                        atoms["C4"] = int(edge)
                        seen.append(edge)
                        break
    isdna = False
    if atoms["O2'"] is None and atoms["O3'"] is not None:
        isdna = True

    return atoms,isdna


def PrintConstraints(s,ring,forcerna,forcedna):
    import sys
    from ffpopt.Restraints import PuckerXRestraint, PuckerYRestraint
    from ffpopt.FindFuncGrps import __IdentifySugarAtoms_backend
    
    missing = "incomplete identification"
    
    if ring[1] is not None:
        atoms,isdna = __IdentifySugarAtoms_backend(s,ring)

        if forcerna:
            isdna = False
        if forcedna:
            isdna = True
        
        sys.stdout.write("\n Atom position map\n")
        sys.stdout.write("-------------------\n")
        sys.stdout.write("%8s %5s %4s\n"%("Position","0-idx","Name"))
        for pos in atoms:
            if atoms[pos] is None:
                sys.stdout.write("%8s %5s %4s\n"%(pos,"None","None"))
            else:
                sys.stdout.write("%8s %5s %4s\n"%(pos,atoms[pos],s.data["names"][atoms[pos]]))
        cons = []
        if isdna:
            sys.stdout.write("\n B-DNA Constraints\n")
            sys.stdout.write("-------------------\n")
            if None not in [ atoms[pos] for pos in ["HO5'","O5'","C5'","C4'"] ]:
                val = '%i,%i,%i,%i=-151.5'%(atoms["HO5'"],atoms["O5'"],atoms["C5'"],atoms["C4'"])
                sys.stdout.write("beta    HO5'-O5'C5'-C4'  : %s\n"%(val))
                cons.append(val)
            else:
                sys.stdout.write("beta    HO5'-O5'C5'-C4'  : %s\n"%(missing))
            if None not in [ atoms[pos] for pos in ["O5'","C5'","C4'","C3'"] ]:
                val = '%i,%i,%i,%i=30.9'%(atoms["O5'"],atoms["C5'"],atoms["C4'"],atoms["C3'"])
                sys.stdout.write("gamma   O5'-C5'-C4'-C3'  : %s\n"%(val))
                cons.append(val)
            else:
                sys.stdout.write("gamma   O5'-C5'-C4'-C3'  : %s\n"%(missing))
            if None not in [ atoms[pos] for pos in ["C4'","C3'","O3'","HO3'"] ]:
                val = '%i,%i,%i,%i=-159.1'%(atoms["C4'"],atoms["C3'"],atoms["O3'"],atoms["HO3'"])
                sys.stdout.write("epsilon C4'-C3'-O3'-HO3' : %s\n"%(val))
                cons.append(val)
            else:
                sys.stdout.write("epsilon C4'-C3'-O3'-HO3' : %s\n"%(missing))
            if None not in [ atoms[pos] for pos in ["O4'","C1'","N9","C4"] ]:
                val = '%i,%i,%i,%i=-99.4'%(atoms["O4'"],atoms["C1'"],atoms["N9"],atoms["C4"])
                sys.stdout.write("chi     O4'-C1'-N9-C4    : %s\n"%(val))
                cons.append(val)
            else:
                sys.stdout.write("chi     O4'-C1'-N9'-C4   : %s\n"%(missing))
        else:
            sys.stdout.write("\n A-RNA Constraints\n")
            sys.stdout.write("-------------------\n")
            if None not in [ atoms[pos] for pos in ["HO5'","O5'","C5'","C4'"] ]:
                val = '%i,%i,%i,%i=-179.9'%(atoms["HO5'"],atoms["O5'"],atoms["C5'"],atoms["C4'"])
                sys.stdout.write("beta    HO5'-O5'C5'-C4'  : %s\n"%(val))
                cons.append(val)
            else:
                sys.stdout.write("beta    HO5'-O5'C5'-C4'  : %s\n"%(missing))
            if None not in [ atoms[pos] for pos in ["O5'","C5'","C4'","C3'"] ]:
                val = '%i,%i,%i,%i=47.4'%(atoms["O5'"],atoms["C5'"],atoms["C4'"],atoms["C3'"])
                sys.stdout.write("gamma   O5'-C5'-C4'-C3'  : %s\n"%(val))
                cons.append(val)
            else:
                sys.stdout.write("gamma   O5'-C5'-C4'-C3'  : %s\n"%(missing))
            if None not in [ atoms[pos] for pos in ["C4'","C3'","O3'","HO3'"] ]:
                val = '%i,%i,%i,%i=-151.7'%(atoms["C4'"],atoms["C3'"],atoms["O3'"],atoms["HO3'"])
                sys.stdout.write("epsilon C4'-C3'-O3'-HO3' : %s\n"%(val))
                cons.append(val)
            else:
                sys.stdout.write("epsilon C4'-C3'-O3'-HO3' : %s\n"%(missing))
            if None not in [ atoms[pos] for pos in ["C3'","C2'","O2'","HO2'"] ]:
                val = '%i,%i,%i,%i=-169.7'%(atoms["C3'"],atoms["C2'"],atoms["O2'"],atoms["HO2'"])
                sys.stdout.write("        C3'-C2'-O2'-HO2' : %s\n"%(val))
                cons.append(val)
            else:
                sys.stdout.write("        C3'-C2'-O2'-HO2' : %s\n"%(missing))
            if None not in [ atoms[pos] for pos in ["O4'","C1'","N9","C4"] ]:
                val = '%i,%i,%i,%i=-166.1'%(atoms["O4'"],atoms["C1'"],atoms["N9"],atoms["C4"])
                sys.stdout.write("chi     O4'-C1'-N9-C4    : %s\n"%(val))
                cons.append(val)
            else:
                sys.stdout.write("chi     O4'-C1'-N9'-C4   : %s\n"%(missing))

        allcons = " ".join( [ "--restrain-dihed='20.,%s'"%(x) for x in cons ] )
        sys.stdout.write("\nBackbone flags (ideal values):\n%s\n\n"%(allcons))


        idxs = [ atoms[pos] for pos in ["C1'","C2'","C3'","C4'","O4'"] ]
        sidx = ",".join([str(x) for x in idxs])
        vx   = PuckerXRestraint(0,idxs,None).GetCrdValue(s.get_positions())
        vy   = PuckerYRestraint(0,idxs,None).GetCrdValue(s.get_positions())
        res  = " ".join([ "--restrain-puckerx='20.,%s=%.2f'"%(sidx,vx),
                          "--restrain-puckery='20.,%s=%.2f'"%(sidx,vy) ])
        sys.stdout.write("Pucker flags (observed values):\n%s\n\n"%(res))
        

        

if __name__ == "__main__":

    import sys
    import argparse
    from ffpopt.Dihedrals import FindPuckers
    from ffpopt.Struct import ListOfStruct

    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Read or create a structure and search for conformations""")
    
    parser.add_argument \
        ("--rna",
         action='store_true',
         help="Force applicable A-RNA constraint definitions")

    parser.add_argument \
        ("--dna",
         action='store_true',
         help="Force applicable B-DNA constraint definitions")

    parser.add_argument \
        ("inp",
         type=str,
         help="Input json (or mol2) file")

    
    args = parser.parse_args()

    if args.rna and args.dna:
        raise Exception("Cannot use both --rna and --dna")
    
    los = ListOfStruct.from_file(args.inp)
    rings = FindPuckers(los[0])

    sys.stdout.write(f"\nFound {len(rings)} 5-membered rings\n\n")
    for iring,ring in enumerate(rings):
        a = ring[0]
        b = ring[1]

        bok = True
        if b is None:
            bok = False
        else:
            if len(b) == 0:
                bok = False
        
        if not bok:
            names = [ los[0].data["names"][int(i)] for i in a ]
            idxs = ",".join( [x for x in a] )
            mask = "@" + ",".join(names)
            sys.stdout.write("Ring %2i: FAILED TO IDENTIFY ORDER\n"%(iring+1))
            sys.stdout.write(f" 0-idxs: {idxs}\n")
            sys.stdout.write(f"   mask: {mask}\n")
            sys.stderr.write("\n")
        else:
            names = [ los[0].data["names"][i] for i in b ]
            idxs = ",".join( [str(x) for x in b] )
            mask = "@" + ",".join(names)
            sys.stdout.write("Ring %2i: CONSISTENT WITH BEING A SUGAR\n"%(iring+1))
            sys.stdout.write(f" 0-idxs: {idxs}\n")
            sys.stdout.write(f"   mask: {mask}\n")
            PrintConstraints(los[0],ring,args.rna,args.dna)
