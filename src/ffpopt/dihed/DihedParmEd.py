"""ParmEd Amber dihedral delete/change/find/apply and parmed script writer."""

from __future__ import annotations

from ffpopt.dihed.DihedFourier import (
    MultiDihedFcn,
    PrimDihedFcn,
    amber_dihed_period,
    merge_duplicate_period_prims,
    parmed_dihedral_types_from_prims,
)

def DeleteDihedrals(p,list_of_idxs):
    """
    Delete dihedrals from the Parm object p.
    
    Parameters
    ----------
    p : parmed.AmberParm
        The Parm object from which dihedrals will be deleted.
    list_of_idxs : list of tuples
        A list of tuples, where each tuple contains 4 integers representing the indices of the atoms forming the dihedral to be deleted.
    
    Returns
    -------
    None
    
    """
    from parmed.tools.actions import deleteDihedral, addDihedral
    for idxs in list_of_idxs:
        cmd = deleteDihedral(p,
                             f"@{idxs[0]+1}",
                             f"@{idxs[1]+1}",
                             f"@{idxs[2]+1}",
                             f"@{idxs[3]+1}")
    
        cmd.execute()
        
    
    

def ChangeDihedrals(p,idxs,xs,fc=None,bytype=False):
    """ Change dihedrals in the Parm object p.

    Parameters
    ----------
    p : parmed.AmberParm
        The Parm object in which dihedrals will be changed.
    idxs : list of int
        A list of 4 integers representing the indices of the atoms forming the dihedral to be changed.
    xs : list of DihedralType
        A list of DihedralType objects representing the new dihedral parameters.
    fc : float, optional
        A float value to set the force constant for all dihedrals. If None, the force constants from xs will be used.
    bytype : bool, optional
        If True, the function will change dihedrals by type instead of by indices. Default is False.
    
    Returns
    -------
    None

    """

    from collections import defaultdict as ddict
    from parmed.tools.actions import deleteDihedral, addDihedral

    if xs:
        scee = float(getattr(xs[0], "scee", 1.2))
        scnb = float(getattr(xs[0], "scnb", 2.0))
        pers = [amber_dihed_period(x.per) for x in xs]
        if len(pers) != len(set(pers)):
            xs = parmed_dihedral_types_from_prims(
                [PrimDihedFcn(x.phi_k, x.phase, x.per) for x in xs],
                scee=scee,
                scnb=scnb,
            )

    if bytype:

        ftypes = tuple([p.atoms[idx].type for idx in idxs])
        rtypes = tuple(list(ftypes)[::-1])
        allidxs = []
        for x in p.dihedrals:
            kidxs = (x.atom1.idx,x.atom2.idx,x.atom3.idx,x.atom4.idx)
            tidxs = (x.atom1.type,x.atom2.type,x.atom3.type,x.atom4.type)
            if tidxs == ftypes:
                allidxs.append(kidxs)
            elif tidxs == rtypes:
                allidxs.append( tuple(list(kidxs)[::-1]) )
        allseles = [ [ f"@{idx[i]+1}" for i in range(4) ]
                     for idx in allidxs ]

    else:
        allseles = [ [ f"@{i+1}" for i in idxs ] ]

        
    for seles in allseles:
        
        cmd = deleteDihedral(p,seles[0],seles[1],seles[2],seles[3])    
        cmd.execute()

        for x in xs:
            per = x.per
            phase = x.phase
            scee = x.scee
            scnb = x.scnb
            
            k = x.phi_k
            if fc is not None:
                k = fc
    
            cmd = addDihedral\
                (p,
                 seles[0],seles[1],seles[2],seles[3],
                 k,per,phase,scee,scnb)
            cmd.execute()

    

    


def FindDihedrals(p,idxs,impropers=False):
    """ Find dihedrals in the Parm object p that match the given indices.
    
    Parameters
    ----------
    p : parmed.AmberParm
        The Parm object containing the molecular structure.
    idxs : list of int
        A list of 4 integers representing the indices of the atoms forming the dihedral to be found.
    impropers : bool, optional
        If True, only find improper dihedrals. Default is False.
    
    Returns
    -------
    list of Dihedral
        A list of Dihedral objects that match the given indices.
    
    """
    tkey = tuple(idxs)
    xs=[]
    for x in p.dihedrals:
        if x.improper and not impropers:
            continue
        fkey = (x.atom1.idx,x.atom2.idx,x.atom3.idx,x.atom4.idx)
        rkey = tuple(list(fkey)[::-1])
        #print(tkey,fkey,rkey,fkey == tkey, rkey == tkey, x.atom1.type, x.atom2.type,x.atom3.type,x.atom4.type)
        if fkey == tkey or rkey == tkey:
            xs.append(x)
    return xs




def GetMultiDihedFcnFromIdxs(p,idxs):
    """ Get a MultiDihedFcn object from the Parm object p using the given indices.
    
    Parameters
    ----------
    p : parmed.AmberParm
        The Parm object containing the molecular structure.
    idxs : list of int
        A list of 4 integers representing the indices of the atoms forming the dihedral.
    
    Returns
    -------
    MultiDihedFcn
        A MultiDihedFcn object representing the dihedral function for the given indices.
    
    """
    xs = FindDihedrals(p,idxs)
    prims = []
    for x in xs:
        t = x.type
        prims.append( PrimDihedFcn(t.phi_k,t.phase,t.per) )
    return MultiDihedFcn(idxs,prims)


def ChangeParmFromMultiDihedFcn(p,fcn):
    """ Change parameters from a MultiDihedFcn object to a new Parm object. 
    
    Parameters
    ----------
    p : parmed.AmberParm
        The Parm object to be modified.
    fcn : MultiDihedFcn
        The MultiDihedFcn object containing the new dihedral parameters.
    
    Returns
    -------
    parmed.AmberParm
        A new Parm object with the dihedral parameters changed according to the MultiDihedFcn object.
        
    """
    out = CopyParm(p)
    xs = parmed_dihedral_types_from_prims(fcn.prims, scee=1.2, scnb=2.0)
    ChangeDihedrals(out,fcn.idxs,xs)
    return out




# def GetDihedClasses(idxs=None):
    
#     class1 = [MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,2)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,3)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,180,1)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,180,2)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,180,3)] ) ]
               
#     class2 = [MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1),
#                           PrimDihedFcn(1,  0,2)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,180,1),
#                           PrimDihedFcn(1,180,2)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1),
#                           PrimDihedFcn(1,180,2)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,180,1),
#                           PrimDihedFcn(1, 0,2)] ) ]

#     class3 = [MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1),
#                           PrimDihedFcn(1,  0,2),
#                           PrimDihedFcn(1,  0,3)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,180,1),
#                           PrimDihedFcn(1,180,2),
#                           PrimDihedFcn(1,180,3)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1),
#                           PrimDihedFcn(1,180,2),
#                           PrimDihedFcn(1,180,3)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,180,1),
#                           PrimDihedFcn(1,  0,2),
#                           PrimDihedFcn(1,180,3)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,180,1),
#                           PrimDihedFcn(1,180,2),
#                           PrimDihedFcn(1,  0,3)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1),
#                           PrimDihedFcn(1,  0,2),
#                           PrimDihedFcn(1,180,3)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,  0,1),
#                           PrimDihedFcn(1,180,2),
#                           PrimDihedFcn(1,  0,3)] ),
#               MultiDihedFcn( idxs, [PrimDihedFcn(1,180,1),
#                           PrimDihedFcn(1,  0,2),
#                           PrimDihedFcn(1,  0,3)] ) ]


#     return { 1: class1,
#              2: class2,
#              3: class3 }





def WriteParmedScript(fname,p,dfcns,scee=1.2,scnb=2.0): #,bytype):
    """ Write a Parmed script to modify dihedral parameters in a Parm object.
    
    Parameters
    ----------
    fname : str
        The name of the output file where the script will be written.
    p : parmed.AmberParm
        The Parm object containing the molecular structure.
    dfcns : list of MultiDihedFcn
        A list of MultiDihedFcn objects representing the dihedral functions to be modified.
    scee, scnb : float, optional
        1-4 electrostatic and VDW scaling factors written into the script.
    
    Returns
    -------
    None
    
    """
    from collections import defaultdict as ddict
    from ffpopt.dihed.DihedParmEd import FindDihedrals
    
    aidxs = [ idx for dfcn in dfcns for idx in dfcn.idxs ]
    aidxs = list(set(aidxs))
    resname = p.atoms[aidxs[0]].residue.name
    
    fh = open(fname,"w")
    fh.write("#!/usr/bin/env python3\n")
    fh.write("import sys\n")
    fh.write("import argparse\n")
    fh.write("from parmed import load_file\n")
    fh.write("from parmed.tools.actions import deleteDihedral, addDihedral\n")
    fh.write("from parmed.amber.mask import AmberMask\n")

    fh.write("parser = argparse.ArgumentParser(\"replace dihedral parameters\")\n")
    fh.write(f"parser.add_argument(\"--resname\",default=\"{resname}\",help=\"Name of the residue, default: {resname}\",type=str)\n")
    fh.write("parser.add_argument(\"iparm\",help=\"Input parm7\")\n")
    fh.write("parser.add_argument(\"oparm\",help=\"Output parm7\")\n")
    fh.write("args = parser.parse_args()\n")
    fh.write("rname = args.resname\n")

    fh.write(f"scee = {float(scee)}\n")
    fh.write(f"scnb = {float(scnb)}\n")

    
    fh.write("if args.iparm == args.oparm:\n")
    fh.write("    raise Exception(\"The 2 filenames must be different\")\n\n")
    
    fh.write("print(f\"[fit-apply] loading {args.iparm}\", flush=True)\n")
    fh.write("p = load_file( args.iparm )\n")
    fh.write("print(f\"[fit-apply] loaded {len(p.atoms)} atoms, {len(p.dihedrals)} dihedrals\", flush=True)\n\n")
    

    #if not bytype:
    if True:
        for aidx in aidxs:
            res = p.atoms[aidx].residue.name
            name = p.atoms[aidx].name
            mask = ":%s@%s"%("{rname}",name)
            fh.write(f"mask=f\"{mask}\"\n")
            fh.write("res = [i for i in AmberMask(p,mask).Selected()]\n")
            fh.write("if len(res) == 0:\n")
            fh.write("    raise Exception(f\"No atoms matching {mask}\")\n")
            

    fh.write("\n\n")
    n_ops = len(dfcns)
    fh.write(f"print(\"[fit-apply] updating {n_ops} dihedral(s)\", flush=True)\n")
    for idfcn, dfcn in enumerate(dfcns):
        allmasks = [ [ ":%s@%s"%("{rname}",p.atoms[idx].name)
                       for idx in dfcn.idxs ] ]
        idxs_label = "-".join(str(i) for i in dfcn.idxs)

        for masks in allmasks:
            mstr = ",".join(["f\"%s\""%(mask) for mask in masks])
            # Do not embed mstr in the print string - it contains f\"...\" and
            # would produce a SyntaxError in the generated script.
            fh.write(
                f"print(\"[fit-apply]   {idfcn+1}/{n_ops} delete+add "
                f"idxs={idxs_label}\", flush=True)\n"
            )
            fh.write(f"deleteDihedral(p,{mstr}).execute()\n")
            for prim in merge_duplicate_period_prims(dfcn.prims, label=idxs_label):
                fh.write(f"addDihedral(p,{mstr},{prim.fc},{prim.per},{prim.phase},scee,scnb).execute()\n")
            fh.write("\n\n")

    fh.write("print(f\"[fit-apply] saving {args.oparm}\", flush=True)\n")
    fh.write("p.save(args.oparm,overwrite=True)\n")
    fh.write("print(f\"[fit-apply] finished applying -> {args.oparm}\", flush=True)\n")
    fh.close()

    







    
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################



