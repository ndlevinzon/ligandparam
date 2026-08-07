#!/usr/bin/env python3

def ReadMol2(fname):
    import parmed
    pmol = parmed.load_file(fname)
    
    for a in pmol.atoms:
        if isinstance(a.type,int):
            for elem,num in parmed.periodic_table.AtomicNum.items():
                if num == a.atomic_number:
                    a.type = elem
                    break
    return pmol

    
def ConvertMol2toRDKIT(pmol):
    import copy
    from io import StringIO
    from rdkit.Chem import MolFromMol2Block
    from rdkit import rdBase
    from rdkit.Chem import SanitizeMol
    import parmed

    rdBase.DisableLog('rdApp.warning')
    rdBase.DisableLog('rdApp.error')

    tmol = copy.deepcopy(pmol)
    
    for a in tmol.atoms:
        if isinstance(a.type,int):
            for elem,num in parmed.periodic_table.AtomicNum.items():
                if num == a.atomic_number:
                    a.type = elem
                    break

    
    mol2str = StringIO()
    tmol.save(mol2str,format="MOL2")
    mol2str = mol2str.getvalue()
    #print(mol2str)
    mol = MolFromMol2Block(mol2str, removeHs=False, sanitize=False )
    #print(mol)
    SanitizeMol(mol)
    
    rdBase.EnableLog('rdApp.warning')
    rdBase.EnableLog('rdApp.error')
    
    return mol



def ReadMolecule(fnameormol,quiet=False):
    from rdkit.Chem.inchi import MolFromInchi
    from rdkit.Chem import SanitizeFlags
    from rdkit.Chem import SanitizeMol
    from rdkit.Chem import AddHs
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem import MolFromMol2Block
    from rdkit import rdBase,Chem
    import parmed
    from io import StringIO

    rdBase.DisableLog('rdApp.warning')
    rdBase.DisableLog('rdApp.error')

    #################################
    try:
        if not quiet:
            print(f"Trying to interpret {fnameormol} as an inchi string...")
        mol = MolFromInchi\
            (fnameormol, removeHs=False, sanitize=False,
             treatWarningAsError=True)
        mol = AddHs(mol)
        if not quiet:
            print("Success!")
    except Exception as exc_inchi:
        try:
            if not quiet:
                print(f"...Failed ({type(exc_inchi).__name__}: {exc_inchi})")
                print(f"Trying to interpret {fnameormol} as a smiles string...")
            params = Chem.SmilesParserParams()
            params.sanitize = False
            
            cansmi = Chem.CanonSmiles(fnameormol)
            mol = Chem.MolFromSmiles(cansmi, params)
            mol = AddHs(mol)
            if not quiet:
                print("Success!")
        except Exception as exc_smi:
            try:
                if not quiet:
                    print(f"...Failed ({type(exc_smi).__name__}: {exc_smi})")
                    print(f"Reading {fnameormol} using parmed...")
                pmol = parmed.load_file(fnameormol)
            
                for a in pmol.atoms:
                    if isinstance(a.type,int):
                        for elem,num in parmed.periodic_table.AtomicNum.items():
                            if num == a.atomic_number:
                                a.type = elem
                                break

                mol2str = StringIO()
                pmol.save(mol2str,format="MOL2")
                mol2str = mol2str.getvalue()
                mol = MolFromMol2Block(mol2str, removeHs=False, sanitize=False )
                SanitizeMol(mol)
                if not quiet:
                    print("Success!")
            except Exception as exc_parm:
                if not quiet:
                    print(f"...Failed ({type(exc_parm).__name__}: {exc_parm})")
                    print(f"Reading {fnameormol} as json...")
                from .. Struct import ListOfStruct
                s = ListOfStruct.from_file(fnameormol)
                mol = s.structs[0].GetRDKitAtoms()
            #sanitizeOps=SanitizeFlags.SANITIZE_ALL^SanitizeFlags.SANITIZE_KEKULIZE^SanitizeFlags.SANITIZE_SETAROMATICITY^SanitizeFlags.SANITIZE_ADJUSTHS)
    #################################
    rdBase.EnableLog('rdApp.warning')
    rdBase.EnableLog('rdApp.error')
    return mol


def _confsearch_fast_rms_threshold() -> int:
    """Conformer count at which Condensed RMS switches to the fast path.

    ``FFPOPT_CONFSEARCH_RMS_FAST_N`` (default ``50``). Set to ``0`` to always
    use legacy per-pair ``GetBestRMS``.
    """
    import os

    try:
        return max(0, int(os.environ.get("FFPOPT_CONFSEARCH_RMS_FAST_N", "50")))
    except ValueError:
        return 50


def _butina_rms_distances(mol, cids, *, quiet: bool = False):
    """Build condensed pairwise RMS distances for Butina clustering.

    For modest ensembles, uses RDKit ``GetBestRMS`` (symmetry-aware). When the
    conformer count is at or above :func:`_confsearch_fast_rms_threshold`,
    aligns every conformer to the first and uses vectorized heavy-atom RMS —
    much cheaper for large ``nconf`` while remaining adequate for clustering.
    """
    import numpy as np
    from rdkit.Chem.rdMolAlign import AlignMol, GetBestRMS

    n = len(cids)
    thr = _confsearch_fast_rms_threshold()
    use_fast = thr > 0 and n >= thr

    if not use_fast:
        dists = []
        for i in range(n):
            for j in range(i):
                dists.append(GetBestRMS(mol, mol, int(cids[i]), int(cids[j])))
        return dists

    if not quiet:
        print(
            f"ConfSearch: nconf={n} >= {thr} - using fast heavy-atom RMS "
            f"(set FFPOPT_CONFSEARCH_RMS_FAST_N=0 for GetBestRMS)"
        )

    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        heavy = list(range(mol.GetNumAtoms()))

    ref_cid = int(cids[0])
    for cid in cids[1:]:
        AlignMol(mol, mol, int(cid), ref_cid)

    coords = np.empty((n, len(heavy), 3), dtype=float)
    for i, cid in enumerate(cids):
        conf = mol.GetConformer(int(cid))
        for k, atom_idx in enumerate(heavy):
            pos = conf.GetAtomPosition(atom_idx)
            coords[i, k, 0] = pos.x
            coords[i, k, 1] = pos.y
            coords[i, k, 2] = pos.z

    dists = []
    inv_m = 1.0 / float(coords.shape[1])
    for i in range(1, n):
        diff = coords[i] - coords[:i]
        rms = np.sqrt(np.sum(diff * diff, axis=(1, 2)) * inv_m)
        dists.extend(float(x) for x in rms)
    return dists


def GetConformers(mol,nconf,nkeep,mmff94=True,maxiter=250,rmstol=0.5,quiet=False):
    from rdkit.Chem.AllChem import EmbedMultipleConfs
    from rdkit.Chem.AllChem import ETKDGv3
    from rdkit.Chem.AllChem import MMFFOptimizeMoleculeConfs
    from rdkit.Chem.AllChem import UFFOptimizeMoleculeConfs
    from rdkit.ML.Cluster import Butina

    #
    # Generate nconf conformations using the ETDGv3 method
    #
    params = ETKDGv3()
    cids = EmbedMultipleConfs(mol, nconf, params)

    if not quiet:
        print(f"EmbedMultipleConfs produced {len(cids)} conformations")
        print(f"Performing geometry optimizations. This may take a moment...")
    
    #
    # Minimize each structure
    # The results array is length nconf
    # Each element is a tuple.
    # The first value in the tuple is an integer (zero) -- ignore it.
    # The second value is the minimized energy
    #
    if mmff94:
        results = MMFFOptimizeMoleculeConfs(mol,maxIters=maxiter)
    else:
        results = UFFOptimizeMoleculeConfs(mol,maxIters=maxiter)

    #
    # Cluster the optimized geometries based on RMS overlay
    # clusts is a list. The length of the list is the number
    # of clusters.
    # Each element is a tuple containing all integer indexes
    # of equivalent conformations
    #
    dists = _butina_rms_distances(mol, cids, quiet=quiet)
        
    clusts = Butina.ClusterData(dists, len(cids), rmstol,
                                isDistData=True, reordering=False)

    if not quiet:
        print(f"Butina clustering produced {len(clusts)} clusters")

    #
    # For each cluster, find the conformation that has the minimum
    # optimized energy.
    # The length of the clustered_results list is the number of clusters.
    # An element in the list is a tuple.
    # The first value of the tuple is the integer index of the minimum
    # energy conformation.
    # The second value is the
    #
    ncluster = len(clusts)
    clustered_results = []
    for icluster in range(ncluster):
        minene=1000000
        mineneidx = 0
        cres = [ results[i] for i in clusts[icluster] ]
        cids = [ i for i in clusts[icluster] ]

        #print( ["%6i"%(c) for c in clusts[icluster]] )
        #print( ["%6.3f"%(res[1]) for res in cres] )
        
        for index, result in enumerate(cres):
            if(minene>result[1]):       
                minene=result[1]
                mineneidx=index
        clustered_results.append( (cids[mineneidx],cres[mineneidx]) )

    #
    # Sort the conformers by the minimized energy
    #
        
    clustered_results = sorted(clustered_results, key=lambda x: x[1])

    #
    # Only keep the nkeep-lowest conformations
    #
    
    if nkeep < len(clustered_results):
        clustered_results = clustered_results[:nkeep]

    if not quiet:
        print(f"Keeping {len(clustered_results)} clusters")
        
    #
    # Delete the conformations that are not being kept
    #
    cids = [ c.GetId() for c in mol.GetConformers() ]
    kids = [ c[0] for c in clustered_results ]
    for c in cids:
        if c not in kids:
            mol.RemoveConformer(c)

    # We do not need a return value, the mol object is
    # modified as a side effect




def ConformerSearch(fnameormol,outbasename,nconf,nkeep,mmff94,maxiter,rmstol,quiet):
    """
    Performs a conformational search using rdkit.

    Step 1: Read a structure from file or build one from inchi or smiles
    Step 2: Use the ETKDGv3 method to create many possible conformations
    Step 3: Perform MM geometry optimizations of each conformation
    Step 4: Use the Butina to cluster the optimized conformations
    Step 5: Extract the lowest-energy conformation from each custer
    Step 6: Sort the clustered conformations by energy
    Step 7: Return the nkeep lowest-energy conformations

    Parameters
    ----------
    fnameormol : str
        Either a Inchi or smiles string, or the name of a file that can
        be read via parmed.load_file

    outbasename : str or None
        If not None, then this is the basename used to write xyz files
        for each conformer. For example, if the outbase="foo/bar" and
        2 conformers are detected, then it will write the files:
        foo/bar_c01.xyz and foo/bar_c02.xyz

    nconf : int
        The number of initial conformations to search for. This is usually
        50 or larger.  The number of conformations will be reduced by
        performing geometry optimizations and clustering.

    nkeep : int
        The number of conformations to keep after performing geometry
        optimizations, clustering, and sorting by energy. The output
        will be upto nkeep lowest-energy clustered configurations

    mmff94 : bool
        If True, then perform geometry optimizations with the mmff94
        potential; otherwise use the UFF potential

    maxiter : int
        The number of geometry optimization steps per configuration.
        There likely isn't much benefit to setting this higher than
        250.

    rmstol : float
        The RMS tolerance used by the clustering algorithm in Angstroms.

    quiet : bool
         If True, then do not print informative messages to stdout.

    Returns
    -------
    confs : list of list of list
        The conformations. Each element of the list is a conformation.
        The sublist is length natom.
        The subsublist has 4 elements: [element,x,y,z]
    """
    
    from pathlib import Path
    import os
    from copy import deepcopy
    from .. Struct import Struct, ListOfStruct

    #mol2 = None
    inplos = None
    if Path(fnameormol).suffix == ".mol2":
        #mol2 = ReadMol2(fnameormol)
        #m = ConvertMol2toRDKIT(mol2)
        inplos = ListOfStruct( [ Struct.from_mol2(fnameormol) ] )
        m = inplos.structs[0].GetRDKitAtoms()
    elif Path(fnameormol).suffix == ".json":
        #mol2 = ReadMol2(fnameormol)
        #m = ConvertMol2toRDKIT(mol2)
        inplos = ListOfStruct.from_file(fnameormol)
        m = inplos.structs[0].GetRDKitAtoms()
    else:
        m = ReadMolecule(fnameormol,quiet=quiet)
        #print(m)
        inplos = ListOfStruct( [ Struct.from_rdkit(m) ] )

  
    GetConformers(m,nconf,nkeep,mmff94=mmff94,maxiter=maxiter,rmstol=rmstol,quiet=quiet)
    
    nat = m.GetNumAtoms()
    symbols = [a.GetSymbol() for a in m.GetAtoms()]
    confs = []
    for i,conf in enumerate(m.GetConformers()):
        myconf = []
        for atom,symbol in enumerate(symbols):
            p = conf.GetAtomPosition(atom)
            myconf.append( [symbol,p.x,p.y,p.z] )
        confs.append(myconf)

    if outbasename is not None:
        # if mol2 is not None:
        #if Path(fnameormol).suffix == ".mol2":
        #     for i,conf in enumerate(confs):
        #         ofile = Path("%s_c%02i.mol2"%(outbasename,i+1))
        #         dname = str(ofile.parent)
        #         if dname != ".":
        #             os.makedirs(dname,exist_ok=True)
        #         file_name = str(ofile)
        #         if not quiet:
        #             print("Writing",file_name)
        #         for a,data in enumerate(conf):
        #             mol2.atoms[a].xx = data[1]
        #             mol2.atoms[a].xy = data[2]
        #             mol2.atoms[a].xz = data[3]
        #         mol2.save(str(ofile),format="MOL2")
        #if inplos is not None:
        if True:
            ss = ListOfStruct([])
            for i,conf in enumerate(confs):
                #ofile = Path("%s_c%02i.json"%(outbasename,i+1))
                #dname = str(ofile.parent)
                #if dname != ".":
                #    os.makedirs(dname,exist_ok=True)
                #file_name = str(ofile)
                #if not quiet:
                #    print("Writing",file_name)
                o = deepcopy(inplos.structs[0])
                #o.structs = [ o.structs[0] ]
                crds = [ c[1:] for c in conf ]
                o.Update(None,crds,None)
                o.data["name"] = "s%03i"%(i)
                ss.structs.append(o)
                #o.save(file_name)
            ss.save(outbasename)
        # else:
        #     for i,conf in enumerate(m.GetConformers()):
        #         ofile = Path("%s_c%02i.xyz"%(outbasename,i+1))
        #         dname = str(ofile.parent)
        #         if dname != ".":
        #             os.makedirs(dname,exist_ok=True)
        #         file_name = str(ofile)
        #         if not quiet:
        #             print("Writing",file_name)
        #         with open(file_name, "w") as fh:
        #             fh.write(str(nat)+"\n")
        #             fh.write("%s_c%02i\n"%(outbasename,i+1))
        #             for data in confs[i]:
        #                 fh.write("%2s %15.8f %15.8f %15.8f\n"%\
        #                          (data[0],data[1],data[2],data[3]))

    return confs


                

# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser \
#         ( formatter_class=argparse.RawDescriptionHelpFormatter,
#           description="""Read or create a structure and search for conformations""")
    

#     parser.add_argument \
#         ("-n","--nconf",
#          help="Initial number of conformations to geometry optimize. (default: 100)",
#          type=int,
#          default=100,
#          required=False)

#     parser.add_argument \
#         ("-k","--nkeep",
#          help="The number of low-energy clustered conformations to save. (default: 5)",
#          type=int,
#          default=5,
#          required=False)

#     parser.add_argument \
#         ("-m","--maxit",
#          help="The number of geometry optimization steps. (default: 250)",
#          type=int,
#          default=250,
#          required=False)

    
#     parser.add_argument \
#         ("-t","--rmstol",
#          help="The RMS tolerance used to cluster the structures in Angstrom (default: 0.5)",
#          type=float,
#          default=0.5,
#          required=False)


#     parser.add_argument \
#         ("--uff",
#          help="If present, optimize with UFF (recommended if metals are present). The default is MMFF94 (recommended for small molecule organic drug-like molecules)",
#          action='store_true')

#     parser.add_argument \
#         ("--quiet",
#          help="If present, do not print informative messages to stdout",
#          action='store_true')

    
#     parser.add_argument \
#         ("-o","--out",
#          help="Basename of the output xyz files. If --out=\"foo\", then the outputs are foo_c01.xyz, foo_c02.xyz, ... (default: \"mol\")",
#          type=str,
#          default="mol",
#          required=False)


#     parser.add_argument \
#         ('name_or_string',
#          metavar='name_or_string',
#          type=str,
#          help='Either a filename, Inchi string, or smiles string')
    
#     args = parser.parse_args()
    
#     ConformerSearch(args.name_or_string,
#                     args.out,
#                     args.nconf,args.nkeep,
#                     not args.uff,
#                     args.maxit,args.rmstol,
#                     args.quiet)

    
