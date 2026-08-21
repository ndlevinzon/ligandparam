#!/usr/bin/env python3


def GetRestraintList(args,bonds):
    from pathlib import Path
    from ffpopt.geom.Constraints import ConstraintList
    from ffpopt.geom.Restraints  import RestraintList, RmsRestraint, TwistRestraint
    from ffpopt.AmberParm   import bonds2graph, RotateBondMask
    import numpy as np
    import ase
    import ase.io
    
    nbond = 0
    if args.restrain_bond is not None:
        nbond  = len(args.restrain_bond)

    nangle = 0
    if args.restrain_angle is not None:
        nangle = len(args.restrain_angle)

    ndihed = 0
    if args.restrain_dihed is not None:
        ndihed = len(args.restrain_dihed)

    nr12 = 0
    if args.restrain_r12 is not None:
        nr12   = len(args.restrain_r12)

    npuckerx = 0
    if args.restrain_puckerx is not None:
        npuckerx= len(args.restrain_puckerx)
        
    npuckery = 0
    if args.restrain_puckery is not None:
        npuckery= len(args.restrain_puckery)
        
    nrms = 0
    if args.restrain_rms is not None:
        nrms   = len(args.restrain_rms)

    ntwist = 0
    if args.restrain_twist:
        ntwist = len(args.restrain_twist)

    nres = nbond+nangle+ndihed+nr12+nrms+ntwist+npuckerx+npuckery
    rlist = None
    if nres > 0:
        rlist = RestraintList([])
    if nbond > 0:
        tmp = RestraintList.from_list_of_str( [ "bond,"+x for x in args.restrain_bond ] )
        if tmp.rests[0].value is None:
            tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
        rlist.extend(tmp.rests)
    if nangle > 0:
        tmp = RestraintList.from_list_of_str( [ "angle,"+x for x in args.restrain_angle ] )
        if tmp.rests[0].value is None:
            tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
        rlist.extend(tmp.rests)
    if ndihed > 0:
        tmp = RestraintList.from_list_of_str( [ "dihed,"+x for x in args.restrain_dihed ] )
        if tmp.rests[0].value is None:
            tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
        rlist.extend(tmp.rests)
    if nr12 > 0:
        tmp = RestraintList.from_list_of_str( [ "r12,"+x for x in args.restrain_r12 ] )
        if tmp.rests[0].value is None:
            tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
        rlist.extend(tmp.rests)
    if npuckerx > 0:
        tmp = RestraintList.from_list_of_str( [ "puckerx,"+x for x in args.restrain_puckerx ] )
        if tmp.rests[0].value is None:
            tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
        rlist.extend(tmp.rests)
    if npuckery > 0:
        tmp = RestraintList.from_list_of_str( [ "puckery,"+x for x in args.restrain_puckery ] )
        if tmp.rests[0].value is None:
            tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
        rlist.extend(tmp.rests)
        
    if nrms > 0:
        for x in args.restrain_rms:
            xs = x.split(",")
            if Path(xs[0]).exists():
                
                geoms = ase.io.read(xs[0],index=":")
                atoms = geoms[0]
                rcrds = atoms.get_positions()
                k     = xs[1]
                idxs  = xs[2:]
            else:
                rcrds = np.array(crds,copy=True)
                k     = xs[0]
                idxs  = xs[1:]
            k  = float(k)
            idxs = [ int(i) for i in idxs ]
            wts = np.zeros( (len(eles),) )
            rcrds = rcrds.tolist()
            for i in idxs:
                wts[i] = ms[i]
            wts = wts.tolist()
            rlist.append( RmsRestraint(k,rcrds,wts) )
            
    if ntwist > 0:

        graph = bonds2graph(bonds)

        for x in args.restrain_twist:
            xs = x.split(",")
            if Path(xs[0]).exists():
                geoms = ase.io.read(xs[0],index=":")
                atoms = geoms[0]
                rcrds = atoms.get_positions().tolist()
                k = float(xs[1])
                idxs = [int(xs[2]),int(xs[3])]
            else:
                rcrds = np.array(crds,copy=True).tolist()
                k = float(xs[0])
                idxs = [int(xs[1]),int(xs[2])]

            mask = RotateBondMask(graph,idxs)
            wts1 = np.array( ms, copy=True )
            wts2 = np.array( ms, copy=True )
            for i in range( len(mask) ):
                if i == idxs[0]:
                    continue
                elif i == idxs[1]:
                    continue
                elif mask[i] > 0:
                    wts2[i] = 0
                else:
                    wts1[i] = 0

            wts1 = wts1.tolist()
            wts2 = wts2.tolist()
            rest = TwistRestraint(k,rcrds,wts1,wts2)
            rlist.append(rest)
    return rlist


if __name__ == "__main__":

    
    from pathlib import Path
    import numpy as np
    import argparse
    import json
    from collections import defaultdict as ddict
    from copy import deepcopy
    
    import parmed
    import ase
    import ase.io

    
    from ffpopt.Reader      import ReadMol2, ReadXyz
    from ffpopt.constants   import GetAtomicSymbol, GetAtomicNumber, GetAtomicMass
    from ffpopt.geom.Constraints import ConstraintList
    from ffpopt.geom.Restraints  import RestraintList, RmsRestraint, TwistRestraint
    from ffpopt.AmberParm   import bonds2graph, RotateBondMask
    from ffpopt.Struct      import ListOfStruct, Struct
    
    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Prepare na ASE XYZ file that contains charge, multiplicity, constraint, and restraint information""" )

    
    parser.add_argument \
        ("--charge","-q",
         help="Charge, int. Default: None (reads from parm7 or mol2 file).",
         required=False,
         default=None,
         type=int)

    
    parser.add_argument \
        ("--spin","-m",
         help="Spin multiplicity, int. Default: 1",
         required=False,
         type=int,
         default=1)

    
    parser.add_argument \
        ("--parm","-p",
         help="Amber parm7 file, only needed if crd is a rst7 file or you want to use sander in the future",
         required=False,
         type=str,
         default=None)

    
    parser.add_argument \
        ("--crd","-c",
         help="xyz, mol2, or amber rst7 file; if --update is present, then this should be a json file",
         required=True,
         type=str)

    
    parser.add_argument \
        ("--name","-n",
         help="Name of the structure. If blank, then it is sXXX, where XXX is a zero-padded number",
         type=str,
         default=None)

    
    parser.add_argument \
        ("--append","-a",
         help="append structure to output json",
         action='store_true')

    
    parser.add_argument \
        ("--update","-u",
         help="update an existing json file. This is used to supple a --parm value or replace the constraints/restraints.  If --parm is supplied and --name is not given, then all structures with the same list of elements are modified. If --parm is not supplied, then only the constraints/restraints are replaced (or cleared).",
         action='store_true')

    
    parser.add_argument \
        ("--clear",
         help="if --update and --clear, then it removes all restraint and constraint definitions.",
         action='store_true')

    
    parser.add_argument \
        ("--out","-o",
         help="output json file",
         required=True,
         type=str)
    

    parser.add_argument \
        ("--constrain",
         help="comma-separated list of 2,3,or 4 0-based integers. The list can be appended with =value to specify a value. If not given, then the input coordinates are used to assign the value. This can be used multiple times.",
         required=False,
         type=str,
         action='append')


    parser.add_argument \
        ("--restrain-bond",
         help="str, format:'k,idx1,idx2[=value]', where idx1,idx2 are 0-based integers and value can be missing",
         required=False,
         type=str,
         action='append')


    parser.add_argument \
        ("--restrain-angle",
         help="str, format:'k,idx1,idx2,idx3[=value]', where idx1,idx2,idx3 are 0-based integers and value can be missing. Value should be degrees",
         required=False,
         type=str,
         action='append')

    
    parser.add_argument \
        ("--restrain-dihed",
         help="str, format:'k,idx1,idx2,idx3,idx4[=value]', where idx1,idx2,idx3,idx4 are 0-based integers and value can be missing. Value should be degrees",
         required=False,
         type=str,
         action='append')

    
    parser.add_argument \
        ("--restrain-r12",
         help="str, format:'k,idx1,idx2,idx3,idx4[=value]', where idx1,idx2,idx3,idx4 are 0-based integers and value can be missing",
         required=False,
         type=str,
         action='append')

    parser.add_argument \
        ("--restrain-puckerx",
         help="str, format:'k,idx1,idx2,idx3,idx4,idx5[=value]', where idx1,idx2,idx3,idx4,idx5 are 0-based integers and value can be missing",
         required=False,
         type=str,
         action='append')

    parser.add_argument \
        ("--restrain-puckery",
         help="str, format:'k,idx1,idx2,idx3,idx4,idx5[=value]', where idx1,idx2,idx3,idx4,idx5 are 0-based integers and value can be missing",
         required=False,
         type=str,
         action='append')

    
    parser.add_argument \
        ("--restrain-rms",
         help="str, format:'[filename,]k,idx1,idx2,[...]', where idx1,idx2,[...] are 0-based integers. If filename is missing, then the input coordinates are adopted. If filename is present, it must be a XYZ file. The indexes reflect those atoms that contribute as nonzero mass-weighted RMS contribution.",
         required=False,
         type=str,
         action='append')

    
    parser.add_argument \
        ("--restrain-twist",
         help="str, format:'[filename,]k,idx1,idx2', where idx1,idx2 are 0-based integers. If filename is missing, then the input coordinates are adopted. If filename is present, it must be a XYZ file. The indexes reflect the bond that is being twisted, which will create separate rms restraints for each half of the bond.",
         required=False,
         type=str,
         action='append')

    
    args = parser.parse_args()

    
    cons = args.constrain

    parm = None
    if args.parm is not None:
        parm = Path(args.parm)
        
    crd = None
    if args.crd is not None:
        crd = Path(args.crd)


    ############################################################################
    if args.update:
        if crd.suffix != ".json":
            raise Exception("--crd must be a json file if --update is present")

        los = ListOfStruct.from_file(args.crd)
        ss = los.structs
        if args.name is not None:
            found=False
            for x in los.structs:
                if x.data["name"] == args.name:
                    found=True
                    ss=[x]
                    break
            if not found:
                names = [ x.data["name"] for x in los.structs ]
                raise Exception(f"structure {args.name} not in {names}")

        keepss = ss
        if parm is not None:
            keepss = []
            mol = parmed.load_file(str(parm))
            names = [ a.name for a in mol.atoms ]
            types = [ a.type for a in mol.atoms ]
            elems = [ GetAtomicSymbol(a.element) for a in mol.atoms ]
            qs = [ float("%.8f"%(a.charge)) for a in mol.atoms ]
            netq = int(round(sum(qs)))
            bonds = [ [x.atom1.idx,x.atom2.idx] for x in mol.bonds ]
            residxs = [ a.residue.idx for a in mol.atoms ]
            resnames = { a.residue.idx: a.residue.name for a in mol.atoms }
            for s in ss:
                #print(elems)
                if s.data["elements"] == elems:
                    s.data["names"] = names
                    s.data["types"] = types
                    s.data["charges"] = qs
                    #s.data["charge"] = netq
                    s.data["bonds"] = bonds
                    s.data["parm"] = str(parm)
                    s.data["residxs"] = residxs
                    s.data["resnames"] = resnames
                    keepss.append(s)
                    #print(s.data)

        if args.clear:
            for s in keepss:
                s.data["restraints"] = []
                s.data["constraints"] = []
                    
        if cons is not None:
            for s in keepss:
                eles = s.data["elements"]
                crds = s.data["positions"]
                qs = s.data["charges"]
                netcharge = int(round(sum(qs)))
                m = s.data["spin"]
                atlist = "".join( ["%s1"%(ele) for ele in eles ] )
                catoms = ase.Atoms(atlist,positions=crds,charges=qs)
                catoms.info["charge"] = netcharge
                catoms.info["spin"] = m
                conlist = ConstraintList.from_list_of_str(cons)
                conlist.FillConstraints(catoms,force=False)
                s.data["constraints"] = conlist.to_list_of_dict()

        for s in keepss:
            rlist = GetRestraintList(args,s.data["bonds"])
            if rlist is not None:
                s.data["restraints"] = rlist.to_list_of_dict()

        los.save(args.out)
            
        exit(0)
    ############################################################################

        
    crds = None
    eles = None
    netcharge = None
    mol  = None
    mult = args.spin
    qs   = None
    bonds = []
    names = None
    residxs = None
    resnames = None
    

    atypes = None
    tmpparm = None
    if parm is not None:
        if crd is not None:
            if crd.suffix == ".rst7" or crd.suffix == ".crd7":
                mol = parmed.load_file(str(parm),xyz=str(crd))
            elif crd.suffix == ".mol2":
                mol = ReadMol2(str(crd))
            for a in mol.atoms:
                a.charge = float("%.8f"%(a.charge))
        if mol is None:
            tmpparm = parmed.load_file(parm,structure=True)
            for a in tmpparm.atoms:
                a.charge = float("%.8f"%(a.charge))
            qs = []
            for a in tmpparm.atoms:
                qs.append( a.charge )
            eles = [ GetAtomicSymbol(a.atomic_number) for a in tmpparm.atoms ]
            qsum = sum(qs)
            netcharge = int(round(qsum))
            dq = (netcharge-qsum)/len(eles)
            qs = [ q + dq for q in qs ]
            

    elif crd is not None:
        if crd.suffix == ".mol2":
            mol = ReadMol2(str(crd))
        elif crd.suffix == ".pdb":
            if args.charge is None:
                raise Exception("--charge=Q must be used when reading an PDB file")
            mylos = ListOfStruct.from_file(str(crd))
            mol = mylos[0].GetParmedAtoms()
            deltaq = args.charge / len(mol.atoms)
            qs = []
            for a in mol.atoms:
                qs.append( deltaq )
            eles = [ GetAtomicSymbol(a.atomic_number) for a in mol.atoms ]
            qsum = sum(qs)
            netcharge = int(round(qsum))
            dq = (netcharge-qsum)/len(eles)
            qs = [ q + dq for q in qs ]
            for ia,a in enumerate(mol.atoms):
                a.charge = qs[ia]
                a.type = eles[ia]

    if mol is not None:
        tmpparm = mol
        names = [ a.name for a in mol.atoms ]
    elif tmpparm is not None:
        pass
    else:
        if args.charge is None:
            raise Exception("--charge=Q must be used when reading an XYZ file")
        tmpparm = ReadXyz(str(crd),args.charge)
        
    bonds = []
    for x in tmpparm.bonds:
        bonds.append( [x.atom1.idx,x.atom2.idx] )

    atypes = []
    for a in tmpparm.atoms:
        atypes.append( a.type )
            
    if mol is not None:
        for a in mol.atoms:
            a.charge = float("%.8f"%(a.charge))
        qs = []
        for a in mol.atoms:
            qs.append( a.charge )
        if eles is None:
            eles = [ GetAtomicSymbol(a.atomic_number) for a in mol.atoms ]
        if crds is None:
            crds=[]
            for a in mol.atoms:
                crds.append( [ a.xx, a.xy, a.xz ] )
            crds = crds
        qsum = sum(qs)
        netcharge = int(round(qsum))
        dq = (netcharge-qsum)/len(eles)
        qs = [ q + dq for q in qs ]

        residxs = [ a.residue.idx for a in mol.atoms ]
        resnames = { a.residue.idx: a.residue.name for a in mol.atoms }
        
    else:
        atoms = ase.io.read(str(crd),index=0)
        if "charge" in atoms.info:
            netcharge = atoms.info["charge"]
        if "spin" in atoms.info:
            mult = atoms.info["spin"]
        if eles is None:
            eles = atoms.get_chemical_symbols()
        crds = atoms.get_positions().tolist()

    
        if netcharge is None:
            netcharge = args.charge
        
        qs = atoms.get_initial_charges().tolist()
        qsum = sum(qs)
        if qsum != netcharge:
            dq = (netcharge-qsum)/len(eles)
            qs = [ q+dq for q in qs ]

        residxs = [0]*len(eles)
        resnames = {0:"MOL"}

    if mult is None:
        mult = args.spin
        
    if qs is None:
        qs = [ netcharge / len(eles) ]*len(eles)

    if names is None:
        seen = ddict(int)
        names = []
        for e in eles:
            seen[e] += 1
            if seen[e] > 1:
                names.append("%s%i"%(e,seen[e]))
            else:
                names.append(e)
        
    if names is not None:
        seen = ddict(int)
        for iname in range(len(names)):
            name = names[iname]
            if name in seen:
                for i in range(98):
                    j=i+1
                    tname = "%s%i"%(name,j)
                    if tname in seen or tname in names:
                        continue
                    else:
                        break
                seen[tname] += 1
                names[iname] = tname
            
    #atoms = ase.Atoms(eles,positions=crds,charges=qs)
    #atoms.info["charge"] = int(round(sum(qs)))
    #atoms.info["spin"] = mult
    #constraints2info(atoms,cons)
    #ase.io.write(args.out,atoms,format="extxyz")

    zs = []
    ms = []
    for e in eles:
        z = GetAtomicNumber(e)
        zs.append(z)
        m = GetAtomicMass(z)
        ms.append(m)


    
        
    data = {}
    data["elements"] = eles
    data["types"] = atypes
    data["names"] = names
    data["residxs"] = residxs
    data["resnames"] = resnames
    data["positions"] = crds
    data["charges"] = qs
    #data["charge"] = int(netcharge)
    data["spin"]  = int(mult)
    data["bonds"] = bonds
    
    
    if args.parm is not None:
        data["parm"] = str(parm)
    else:
        data["parm"] = None
        
    data["constraints"] = []
    
    if cons is not None:
        atlist = "".join( ["%s1"%(ele) for ele in eles ] )
        catoms = ase.Atoms(atlist,positions=crds,charges=qs)
        catoms.info["charge"] = netcharge
        catoms.info["spin"] = m
        conlist = ConstraintList.from_list_of_str(cons)
        conlist.FillConstraints(catoms,force=False)
        data["constraints"] = conlist.to_list_of_dict()
        
    data["restraints"] = []

    # nbond = 0
    # if args.restrain_bond is not None:
    #     nbond  = len(args.restrain_bond)

    # nangle = 0
    # if args.restrain_angle is not None:
    #     nangle = len(args.restrain_angle)

    # ndihed = 0
    # if args.restrain_dihed is not None:
    #     ndihed = len(args.restrain_dihed)

    # nr12 = 0
    # if args.restrain_r12 is not None:
    #     nr12   = len(args.restrain_r12)

    # nrms = 0
    # if args.restrain_rms is not None:
    #     nrms   = len(args.restrain_rms)

    # ntwist = 0
    # if args.restrain_twist:
    #     ntwist = len(args.restrain_twist)

    # nres = nbond+nangle+ndihed+nr12+nrms+ntwist
    # if nres > 0:
    #     rlist = RestraintList([])
    # if nbond > 0:
    #     tmp = RestraintList.from_list_of_str( [ "bond,"+x for x in args.restrain_bond ] )
    #     if tmp.rests[0].value is None:
    #         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
    #     rlist.extend(tmp.rests)
    # if nangle > 0:
    #     tmp = RestraintList.from_list_of_str( [ "angle,"+x for x in args.restrain_angle ] )
    #     if tmp.rests[0].value is None:
    #         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
    #     rlist.extend(tmp.rests)
    # if ndihed > 0:
    #     tmp = RestraintList.from_list_of_str( [ "dihed,"+x for x in args.restrain_dihed ] )
    #     if tmp.rests[0].value is None:
    #         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
    #     rlist.extend(tmp.rests)
    # if nr12 > 0:
    #     tmp = RestraintList.from_list_of_str( [ "r12,"+x for x in args.restrain_r12 ] )
    #     if tmp.rests[0].value is None:
    #         tmp.rests[0].value = tmp.rests[0].GetCrdValue(crds)
    #     rlist.extend(tmp.rests)
    # if nrms > 0:
    #     for x in args.restrain_rms:
    #         xs = x.split(",")
    #         if Path(xs[0]).exists():
                
    #             geoms = ase.io.read(xs[0],index=":")
    #             atoms = geoms[0]
    #             rcrds = atoms.get_positions()
    #             k     = xs[1]
    #             idxs  = xs[2:]
    #         else:
    #             rcrds = np.array(crds,copy=True)
    #             k     = xs[0]
    #             idxs  = xs[1:]
    #         k  = float(k)
    #         idxs = [ int(i) for i in idxs ]
    #         wts = np.zeros( (len(eles),) )
    #         rcrds = rcrds.tolist()
    #         for i in idxs:
    #             wts[i] = ms[i]
    #         wts = wts.tolist()
    #         rlist.append( RmsRestraint(k,rcrds,wts) )
            
    # if ntwist > 0:

    #     graph = bonds2graph(bonds)

    #     for x in args.restrain_twist:
    #         xs = x.split(",")
    #         if Path(xs[0]).exists():
    #             geoms = ase.io.read(xs[0],index=":")
    #             atoms = geoms[0]
    #             rcrds = atoms.get_positions().tolist()
    #             k = float(xs[1])
    #             idxs = [int(xs[2]),int(xs[3])]
    #         else:
    #             rcrds = np.array(crds,copy=True).tolist()
    #             k = float(xs[0])
    #             idxs = [int(xs[1]),int(xs[2])]

    #         mask = RotateBondMask(graph,idxs)
    #         wts1 = np.array( ms, copy=True )
    #         wts2 = np.array( ms, copy=True )
    #         for i in range( len(mask) ):
    #             if i == idxs[0]:
    #                 continue
    #             elif i == idxs[1]:
    #                 continue
    #             elif mask[i] > 0:
    #                 wts2[i] = 0
    #             else:
    #                 wts1[i] = 0

    #         wts1 = wts1.tolist()
    #         wts2 = wts2.tolist()
    #         rest = TwistRestraint(k,rcrds,wts1,wts2)
    #         rlist.append(rest)

    rlist = GetRestraintList(args,bonds)
    #if nres > 0:
    if rlist is not None:
        data["restraints"] = rlist.to_list_of_dict()

    do_append = False
    tname = args.name
    if Path(args.out).exists and args.append:
        from collections import defaultdict
        do_append = True
        seen_names = defaultdict(int)
        fh = open(args.out,'r')
        old = json.load(fh)
        for s in old:
            if "name" in s:
                seen_names[s["name"]] += 1
        seen_names = [ k for k in seen_names ]
        if tname is not None:
            if tname in seen_names:
                pass
            else:
                tname = None
        if tname is None:
            for i in range(1000):
                tname = "s%03i"%(i)
                if tname not in seen_names:
                    break
        data["name"] = tname
        data["energy"] = None
        data["forces"] = None
        old.append(data)
        fh.close()
        
        with open(args.out,'w') as f:
            json.dump(old,f,indent=4)

    else:
        if tname is None:
            tname = "s000"
        data["name"] = tname
        data["energy"] = None
        data["forces"] = None

        with open(args.out,'w') as f:
            json.dump([data],f,indent=4)

    
