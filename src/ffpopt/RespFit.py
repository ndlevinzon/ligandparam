#!/usr/bin/env python3

def espaloma_charge(molecule, total_charge=None):
    """
    Computes rigorous QEq partial charges using espaloma-charge and RDKit.
    Bypasses the OpenFF wrapper entirely and natively fixes the total_charge parameter bug.
    """
    import os
    import torch
    from rdkit import Chem
    from espaloma_charge.utils import from_rdkit_mol

    # 1. Handle charge defaulting exactly like the source package function
    if total_charge is None:
        total_charge = int(round(Chem.GetFormalCharge(molecule)))

    if False:
        
        MODEL_URL = "https://github.com"
        MODEL_PATH = ".espaloma_charge_model.pt"
    
        # 2. Handle model caching and loading
        if not os.path.exists(MODEL_PATH):
            from urllib import request
            request.urlretrieve(MODEL_URL, MODEL_PATH)
        model = torch.load(MODEL_PATH)
        graph = from_rdkit_mol(molecule)
        
    else:
        
        import importlib
        import importlib.resources
        from pathlib import Path
        data_file_name = "pkgdata/espaloma/model.pt"
        data_path = importlib.resources.files("ffpopt") / data_file_name
        model = torch.load(str(data_path))
        graph = from_rdkit_mol(molecule)

    
    # 3. Handle GPU routing identically to the package function
    if torch.cuda.is_available():
        graph = graph.to("cuda:0")
        model = model.cuda()

    for layer in model:
        # Pass total_charge to ChargeEquilibrium layer, which will distribute it across atoms
        if layer.__class__.__name__ == "ChargeEquilibrium":
            graph = layer(graph, total_charge=total_charge)
        else:
            graph = layer(graph)
        
    # 7. Extract the finished charges array from the captured output graph
    charges = graph.ndata["q"].cpu().detach().flatten().numpy()
    
    #print("Rigorous charges.sum() =", charges.sum())
    return charges





def hilfiker_charges(aseatoms, total_charge=0):
    """
    Compute hilfiker charges
    """
    import os
    import sys
    import logging
    import warnings

    # 1. SAVE LOGGER STATES AND REDUCT NOISE
    root_logger = logging.getLogger()
    named_logger = logging.getLogger("root")
    original_root_level = root_logger.level
    original_named_level = named_logger.level
    
    root_logger.setLevel(logging.ERROR)
    named_logger.setLevel(logging.ERROR)

    # Save original environment variable for warnings if it exists
    old_pywarns = os.environ.get("PYTHONWARNINGS", None)
    os.environ["PYTHONWARNINGS"] = "ignore"

    # 2. OPEN A PROTECTED WARNING CONTEXT
    with warnings.catch_warnings():
        # Block Python-level warnings during this entire execution block
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore")

        # 3. IMPORT LIBRARIES LATE (Inside the active filter context)
        import numpy as np
        import importlib
        import importlib.resources
        import xgboost as xgb
        from mace.calculators import MACECalculator

        # 4. RUN MODEL PIPELINE
        ffpoptloc = importlib.resources.files("ffpopt")
        mace_path = ffpoptloc / "pkgdata/mace-off/mace_off23/MACE-OFF23_large.model"
        hilf_path = ffpoptloc / "pkgdata/ml_for_charges/xgb_chgs.json"

        calculator = MACECalculator(model_paths=str(mace_path), device='cpu') 
        descriptor_mace = calculator.get_descriptors(aseatoms)
        
        model = xgb.Booster()
        model.load_model(str(hilf_path))
        
        feature_names = [str(i) for i in range(descriptor_mace.shape[1])]
        dmatrix_descriptors = xgb.DMatrix(descriptor_mace, feature_names=feature_names)
        qs = model.predict(dmatrix_descriptors)
        
        delta = np.sum(qs) - total_charge
        new_q = np.array(qs) - (delta / len(qs))

    # 5. RESTORE LOGGING AND SYSTEM ENVIRONMENT SETTINGS OUTSIDE CONTEXT
    root_logger.setLevel(original_root_level)
    named_logger.setLevel(original_named_level)
    
    if old_pywarns is not None:
        os.environ["PYTHONWARNINGS"] = old_pywarns
    else:
        os.environ.pop("PYTHONWARNINGS", None)

    return new_q





def GetChargeEquivGroups(mc):
    from collections import defaultdict as ddict
    m = mc.mols[0]
    eqgrps = ddict(list)
    for i,pidx in enumerate(m.paridxs):
        eqgrps[pidx].append(i)
        #print("%2i %2i"%(i,pidx))
    #print("")
    return eqgrps

def ReadMap(fname):
    from collections import defaultdict as ddict
    from pathlib import Path
    p = Path(fname)
    if not p.is_file():
        raise Exception(f"File not found: {fname}")
    fwdmap = ddict(str)
    revmap = ddict(str)
    fh = open(fname,"r")
    for line in fh:
        if "=>" in line:
            cs = [ x.strip() for x in line.split("=>") ]
            if len(cs) == 2:
                fwdmap[cs[0]] = cs[1]
                revmap[cs[1]] = cs[0]
    return fwdmap,revmap

def Map2IdxMap(namemap,mol0,mol1):
    from collections import defaultdict as ddict
    names0 = [ a.name for a in mol0 ]
    names1 = [ a.name for a in mol1 ]
    idxmap = ddict(int)
    for name0 in namemap:
        name1 = namemap[name0]
        if name0 in names0:
            idx0 = names0.index(name0)
        else:
            raise Exception(f"Atom name {name0} not in list {names0}")
        if name1 in names1:
            idx1 = names1.index(name1)
        else:
            raise Exception(f"Atom name {name1} not in list {names1}")
        idxmap[idx0] = idx1
    return idxmap




def RunRespFit(*,
               inp: str,
               out: str,
               confs: list, # list of str
               respf: bool = False,
               espaloma: bool = False,
               hilfiker: bool = False,
               nofit: bool = False,
               program: str = "psi4",
               resp_a: float = 0.001,
               resp_b: float = 0.1,
               density: int = 6,
               digits: int = 4,
               scosmo: float = 0.0,
               ext_scale: float = 1.1,
               ext_density: int = 2,
               group: list = [],
               freeze: str = None,
               update_only_grouped_atoms: bool = False,
               update_only_ungrouped_atoms: bool = False,
               **standard_kwargs):
    
    import argparse
    from types import SimpleNamespace
    from ffpopt.Options import AddModelOptions

    import copy
    import numpy as np
    import parmed
    from pathlib import Path
    
    from ffpopt import cpefit
    from ffpopt.cpefit import FixCharges
    from ffpopt.constants import AU_PER_ANGSTROM
    from ffpopt.Struct import Struct, ListOfStruct

    
    _p = argparse.ArgumentParser(add_help=False)
    AddModelOptions(_p)
    std_defaults = vars(_p.parse_args([]))
    std_defaults["prefix"] = ""
    unknown = set(standard_kwargs) - set(std_defaults)
    if unknown:
        raise TypeError(
            f"Unexpected keyword argument(s): {sorted(unknown)}"
        )
    std = {**std_defaults, **standard_kwargs}

    args = SimpleNamespace(
        inp=inp,
        out=out,
        confs=confs,
        respf=respf,
        espaloma=espaloma,
        hilfiker=hilfiker,
        nofit=nofit,
        program=program,
        resp_a=resp_a,
        resp_b=resp_b,
        density=density,
        digits=digits,
        scosmo=scosmo,
        ext_scale=ext_scale,
        ext_density=ext_density,
        group=group,
        freeze=freeze,
        update_only_grouped_atoms=update_only_grouped_atoms,
        update_only_ungrouped_atoms=update_only_ungrouped_atoms,
        **std,
    )


    if args.inp == args.out:
        raise Exception("--inp and --out cannot be the same file")

    if len(args.confs) < 1:
        raise Exception("There must be at least 1 conformer")

    if "quick" in args.program and args.scosmo > 0:
        raise Exception("Cannot use quick if scosmo > 0 due to limitations in the quick input file format")

    if args.scosmo < 0:
        raise Exception("--scosmo value must be > 0")
    
    if args.scosmo > 0:
        if args.espaloma:
            raise Exception("--scosmo must be 0 if --espaloma is set")
        if args.hilfiker:
            raise Exception("--scosmo must be 0 if --hilfiker is set")
        if args.nofit:
            raise Exception("--scosmo must be 0 if --nofit is set")

    
    scfopts = cpefit.AbInitioOptions\
        (program=args.program,
         theory=args.model,
         mem=args.psi4_memory,
         nproc=args.psi4_num_threads)
    
    esppar = cpefit.SurfaceParameters\
        ([1.4,1.6,1.8,2.0],
         args.density)
    
    extpar = None

    if args.scosmo > 0:
        extpar = cpefit.SurfaceParameters\
            ([args.ext_scale],
             args.ext_density)

    
    confs = ReadListOfConformers(args.inp,args.confs,esppar,extpar,prefix=std["prefix"])

    m  = cpefit.Molecule(args.inp,confs,groups=args.group,freeze=args.freeze)
    mc = cpefit.MoleculeCollection([m])

    copymask = [ False ] * len(m.parm.atoms)

    if args.update_only_ungrouped_atoms or args.update_only_grouped_atoms:
        count = m.count_groups_foreach_atom()
        maxcount = max(count)
        if maxcount > 1:
            alist = " ".join( [ name for i,name in enumerate(m.atnames) if count[i] > 1 ] )
            raise Exception("Cannot enforce --update_only_ungrouped_atoms "
                            "because atoms found in multiple groups: %s"%(alist))

    if args.update_only_grouped_atoms:
        copymask = m.get_mask_atom_is_ungrouped()
    elif args.update_only_ungrouped_atoms:
        copymask = m.get_mask_atom_is_grouped()
    origqs = [ atom.charge for atom in m.parm.atoms ]
    
    params = mc.MakeParams()
    params.resp_a = args.resp_a
    params.resp_b = args.resp_b
    
    m     = mc.mols[0]
    confs = m.conformers

    oldq = params.q[ m.paridxs ]


    if args.espaloma:
        los = ListOfStruct.from_file( args.inp )
        tmpq = los[0].GetEspalomaCharges()
        newq = m.clean_loc_charges(tmpq)
    elif args.hilfiker:
        los = ListOfStruct.from_file( args.inp )
        newq = None
        for s in los:
            tmpq = s.GetHilfikerCharges()
            if newq is None:
                newq = np.array(tmpq,copy=True)
            else:
                newq += tmpq
        newq /= len(los)
        newq = m.clean_loc_charges(newq)
    elif args.nofit:
        los = ListOfStruct.from_file( args.inp )
        tmpq = np.array(los[0].data["charges"],copy=True)
        newq = m.clean_loc_charges(tmpq)
    else:
        from ffpopt.cpefit.parallel_esp import run_abinitio_esp_conformers

        run_abinitio_esp_conformers(confs, scfopts)
        if args.respf:
            newq = m.RunRespF(args.inp)
        else:
            mc.OptimizeFixedCharge(params)
            newq = params.q[ m.paridxs ]

            
    print("gas charge constraint check")
    m.check_loc_constraints(newq)
    #print("original charge constraint check")
    #m.check_loc_constraints(origqs)

    gdq_old = 0
    udq_old = 0
    gdq_new = 0
    udq_new = 0
    # O(natoms) mask / charge bookkeeping — intentionally serial (ESP was the
    # parallel cost; see ffpopt.cpefit.parallel_esp).
    for ia,masked in enumerate(copymask):
        print("%3i %12s %6s %12.6f %12.6f"%(ia,m.atnames[ia],masked,newq[ia],origqs[ia]))
        if masked:
            gdq_old += origqs[ia]
            gdq_new += newq[ia]
        else:
            udq_old += origqs[ia]
            udq_new += newq[ia]

        
    if abs(gdq_old-gdq_new) > 1.e-5:
        raise Exception("Could not copy original charges "
                        "because the original and gas fitted charge "
                        f"sums are inconsistent: old={gdq_old} new={gdq_new} "
                        f"diff={abs(gdq_old-gdq_new)}"
                        " This likely happens because an atom appears in multiple groups")
    for ia,masked in enumerate(copymask):
        if masked:
            newq[ia] = origqs[ia]

            
    gasq = np.array(newq,copy=True)
    newq = FixCharges(newq,args.digits)

    
    print("Gas phase charges")
    print("%4s%12s %12s"%("idx","oldQ","newQ"))
    for i in range(len(oldq)):
        print("%3i %12.6f %12.6f"%(i,oldq[i],newq[i]))


    if args.scosmo > 0:

        from ffpopt.cpefit.parallel_esp import (
            run_abinitio_esp_conformers,
            run_cosmo_harmonics_conformers,
        )

        run_cosmo_harmonics_conformers(
            confs, 0, newq, scfopts, onlypos=True
        )
            
        print("build cosmo_confs")
        
        cosmo_confs = []
        for c in confs:
            tmp_pertid = c.pertid
            c.pertid = "%sH000pos"%(c.pertid)
            base = c.GetBasename()
            #print(tmp_pertid,base)
            c.pertid = tmp_pertid
            cosmo_confs.append( ConformerFromAbInitioOutput(base + ".log",esppar,extpar) )

        mccosmo = cpefit.MoleculeCollection\
            ([cpefit.Molecule(args.inp,cosmo_confs,
                              groups=args.group,
                              freeze=args.freeze)])

        newparams = copy.deepcopy( params )
        newparams.opt_q = True
        newparams.opt_hardness = False
        newparams.opt_chempot  = False
        newparams.opt_zetascl  = False


        run_abinitio_esp_conformers(mccosmo.mols[0].conformers, scfopts)

        if args.respf:
            solvq = mccosmo.mols[0].RunRespF(args.inp)
        else:
            mccosmo.OptimizeFixedCharge(newparams)
            solvq = newparams.q[ mccosmo.mols[0].paridxs ]



        print("solvated charge constraint check")
        m.check_loc_constraints(solvq)
    
        gdq_old = 0
        udq_old = 0
        gdq_new = 0
        udq_new = 0
        for ia,masked in enumerate(copymask):
            if masked:
                gdq_old += origqs[ia]
                gdq_new += solvq[ia]
            else:
                udq_old += origqs[ia]
                udq_new += solvq[ia]

        if abs(gdq_old-gdq_new) > 1.e-5:
            raise Exception("Could not apply --update_only_grouped_atoms "
                            "because the original and cosmo fitted charge "
                            f"sums are inconsistent: old={gdq_old} new={gdq_new} "
                            f"diff={abs(gdq_old-gdq_new)}"
                            " This likely happens because an atom appears in multiple groups")
        for ia,masked in enumerate(copymask):
            if masked:
                solvq[ia] = origqs[ia]

                
            
        tmpq = FixCharges(solvq,args.digits)

        print("Charges in a static implicit solvent-like response environment")
        print("%4s%12s %12s"%("idx","oldQ","newQ"))
        for i in range(len(oldq)):
            print("%3i %12.6f %12.6f"%(i,oldq[i],tmpq[i]))


        newq = (1-args.scosmo) * gasq + args.scosmo * solvq
        newq = FixCharges(newq,args.digits)

        
        print("Charges after linear combination")
        print("%4s%12s %12s"%("idx","oldQ","newQ"))
        for i in range(len(oldq)):
            print("%3i %12.6f %12.6f"%(i,oldq[i],newq[i]))

        
    mol = m.parm
    qsum = 0
    for i in range(len(mol.atoms)):
        mol.atoms[i].charge = newq[i] #float("%.6f"%(newq[i]))
        qsum += mol.atoms[i].charge

    if args.out is not None:
        print(f"Writing {args.out}")
        if Path(args.out).suffix == ".mol2":
            mol.save( args.out, overwrite=True )
        elif Path(args.out).suffix == ".json":
            if Path(args.inp).suffix == ".json":
                los = ListOfStruct.from_file(args.inp)
                los.structs = [los.structs[0]]
                los.structs[0].data["charges"] = [ a.charge for a in mol.atoms ]
                los.save(args.out)
            elif Path(args.inp).suffix == ".mol2":
                s = Struct.from_mol2(args.inp)
                s.data["charges"] = [ a.charge for a in mol.atoms ]
                los = ListOfStruct( [s] )
                los.save(args.out)
            else:
                raise Exception(f"--out is json, so --inp must be mol2 or json, "
                                f"but --inp={args.inp}")
        else:
            raise Exception(f"--out should be json or mol2, but --out={args.out}")

    try:
        los = ListOfStruct.from_file(args.inp)
    except:
        s = Struct.from_mol2(args.inp)
        los = ListOfStruct( [s] )
    los.structs = [los.structs[0]]
    los.structs[0].data["charges"] = [ a.charge for a in mol.atoms ]
        
    return los

    


def ConformerFromAbInitioOutput(c,esppar,extpar):
    from pathlib import Path
    from ffpopt import cpefit

    conf = None
    ok=False

    if not Path(c).exists:
        raise Exception(f"File not found: {c}")
    
    if not ok:
        try:
            conf = cpefit.Conformer.FromGaussian(c,esppar,extpar)
            ok=True
        except:
            ok=False
    if not ok:
        try:
            conf = cpefit.Conformer.FromPsi4(c,esppar,extpar)
            ok=True
        except:
            ok=False
    if not ok:
        try:
            conf = cpefit.Conformer.FromQuick(c,esppar,extpar)
            ok=True
        except:
            ok=False
            
    if not ok:
        raise Exception(f"Could not figure out how to create a conformer from {c}")
    
    return conf



def ReadListOfConformers(template,filenamelist,esppar,extpar,prefix=""):
    import numpy as np
    from pathlib import Path
    import traceback
    import parmed
    from ffpopt import cpefit
    from ffpopt.Struct import ListOfStruct

    generic = None
    confs = []
    for c in filenamelist:
        ok=False
        
        if ".mol2" in c:
            try:
                confs.append( cpefit.Conformer.FromMol2(c,esppar,extpar) )
                ok=True
            except:
                ok=False
        elif ".json" in c:
            try:
                los = ListOfStruct.from_file(c)
                for s in los:
                    if len(prefix) > 0:
                        s.data["name"] = "%s%s"%(prefix,s.data["name"])
                    confs.append( cpefit.Conformer.FromStruct(s,esppar,extpar) )
                ok=True
            except Exception as e:
                print(e)
                traceback.print_exc()
                ok=False

        if not ok:
            try:
                confs.append( ConformerFromAbInitioOutput(c,esppar,extpar) )
                ok = True
            except Exception as e:
                print(e)
                traceback.print_exc()
                ok = False
                
        if not ok:
            try:
                if generic is None:
                    if Path(template).suffix == ".mol2":
                        generic = cpefit.Conformer.FromMol2(template,esppar,extpar)
                    elif Path(template).suffix == ".json":
                        los = ListOfStruct.from_file(template)
                        generic = cpefit.Conformer.FromStruct(los.structs[0],esppar,extpar)

                pname    = Path(c)
                if not pname.is_file():
                    raise Exception(f"File not found: {fname}")
                basename = pname.with_suffix("").name
                parts    = basename.split("_")
                name     = parts[0]
                pertid   = None
                if len(parts) > 1:
                    pertid = "_".join( parts[1:] )
                p = parmed.load_file(c,structure=True)
                crds = np.array([ [a.xx,a.xy,a.xz] for a in p.atoms ]) * AU_PER_ANGSTROM()
                atnums = generic.atnums
                charge  = int(round(sum([a.charge for a in p.atoms])))
                extpts  = None
                extvals = None
                esppts  = None
                espvals = None
                confs.append( cpefit.Conformer\
                              ( name,pertid,crds,atnums,charge,
                                esppts,espvals,extpts,extvals,
                                esppar,extpar ) )
                ok=True
            except:
                ok=False
        if not ok:
            raise Exception(f"Could not figure out how to create a conformer from {c}")

    return confs














def ReadJsonOrMol2(fname):
    from ffpopt.Struct import Struct, ListOfStruct
    los = None
    try:
        los = ListOfStruct.from_file(fname)
    except:
        s = Struct.from_mol2(fname)
        los = ListOfStruct( [s] )
    return los
        

def RunDeltaRespFit\
        (*,
         native: str,
         modified: str,
         out: str,
         respf: bool = False,
         espaloma: bool = False,
         hilfiker: bool = False,
         program: str = "psi4",
         resp_a: float = 0.001,
         resp_b: float = 0.1,
         density: int = 6,
         digits: int = 4,
         scosmo: float = 0.0,
         ext_scale: float = 1.1,
         ext_density: int = 2,
         native_cap: list = [],
         modified_cap: list = [],
         mcss_allow_mismatch: bool = False,
         dont_modify_cap_charges: bool = False,
         mcss_map: str = None,
         **standard_kwargs):

    import argparse
    from pathlib import Path
    import copy
    from types import SimpleNamespace
    from parmed.amber.mask import AmberMask
    from . Options import AddModelOptions
    from . RespFit import RunRespFit
    from . cpefit.FixCharges import FixCharges
    from . cpefit.Molecule import Molecule
    from . cpefit.MoleculeCollection import MoleculeCollection
    
    
    _p = argparse.ArgumentParser(add_help=False)
    AddModelOptions(_p)
    std_defaults = vars(_p.parse_args([]))
    unknown = set(standard_kwargs) - set(std_defaults)
    if unknown:
        raise TypeError(
            f"Unexpected keyword argument(s): {sorted(unknown)}"
        )
    std = {**std_defaults, **standard_kwargs}

    args = SimpleNamespace(
        native=native,
        modified=modified,
        out=out,
        respf=respf,
        espaloma=espaloma,
        hilfiker=hilfiker,
        program=program,
        resp_a=resp_a,
        resp_b=resp_b,
        density=density,
        digits=digits,
        scosmo=scosmo,
        ext_scale=ext_scale,
        ext_density=ext_density,
        native_cap=native_cap,
        modified_cap=modified_cap,
        mcss_allow_mismatch=mcss_allow_mismatch,
        dont_modify_cap_charges=dont_modify_cap_charges,
        mcss_map=mcss_map,
        **std,
    )

    #print(dict(**vars(args)))

    from ffpopt.MCSS import OriginalMCSSModel
    from ffpopt.MCSS import OriginalMCSSModelNoMismatchingElements
    from ffpopt.Struct import ListOfStruct

    los_native   = ReadJsonOrMol2(native)
    los_modified = ReadJsonOrMol2(modified)

    mol1 = los_native[0].GetParmedAtoms()
    mol2 = los_modified[0].GetParmedAtoms()

    #
    # cap1 is a list of caps on the native structure
    # caps1 is a flattened list of all native atoms in at least 1 cap
    #
    cap1 = []
    caps1 = []
    for cap in native_cap:
        sidxs = [idx for idx in AmberMask(mol1,cap).Selected()]
        sidxs.sort()
        caps1.append(sidxs)
        cap1.extend(sidxs)

    #
    # cap2 is a list of caps on the modified structure
    # caps2 is a flattened list of all modified atoms in at least 1 cap
    #
    cap2 = []
    caps2 = []
    for cap in modified_cap:
        sidxs = [idx for idx in AmberMask(mol2,cap).Selected()]
        sidxs.sort()
        caps2.append(sidxs)
        cap2.extend(sidxs)

    cap1.sort()
    cap2.sort()

    if len(caps1) != len(caps2):
        raise Exception(f"The native residue has {len(caps1)} "
                        f"caps, but the modified has {len(caps2)}")

    #
    # Perform the MCSS
    # i1toi2 is a dict that takes an atom index from the native structure
    # and returns the matching index in the modified structure.  The keys
    # only contain atoms that have a matching index
    #

    domcss = True
    if args.mcss_map is not None:
        if len(args.mcss_map) > 0:
            domcss = False
            fwdmap,revmap = ReadMap(args.mcss_map)
            i1toi2 = Map2IdxMap(fwdmap,mol1,mol2)
            i2toi1 = Map2IdxMap(revmap,mol2,mol1)
    if domcss:
        if args.mcss_allow_mismatch:
            i1toi2,i2toi1 = OriginalMCSSModel(mol1,mol2,None)
        else:
            i1toi2,i2toi1 = OriginalMCSSModelNoMismatchingElements(mol1,mol2,None)


    mc1 = MoleculeCollection( [ Molecule(native,None,verbose=False) ], verbose=False )
    mc2 = MoleculeCollection( [ Molecule(modified,None,verbose=False) ], verbose=False )
    mc1_equivs = GetChargeEquivGroups(mc1)
    mc2_equivs = GetChargeEquivGroups(mc2)

    sc1=[]
    sc2=[]
    
    for g in mc1_equivs:
        ats = mc1_equivs[g]
        ismapped = [ i in i1toi2 for i in ats ]
        all_same = len(set(ismapped)) <= 1
        #print("mc1",g,all_same,ats,ismapped,[i for i in i1toi2])
        if not all_same:
            sc1.extend(ats)
            for key in ats:
                i1toi2.pop(key, None)
      
    for g in mc2_equivs:
        ats = mc2_equivs[g]
        ismapped = [ i in i2toi1 for i in ats ]
        all_same = len(set(ismapped)) <= 1
        #print("mc2",g,all_same,ats,ismapped,[i for i in i2toi1])
        if not all_same:
            sc2.extend(ats)
            for key in ats:
                i1toi2.pop(key, None)
        
    for c in sc1:
        if c in i1toi2:
            del i1toi2[c]
        for d in i2toi1:
            if i2toi1[d] == c:
                del i2toi1[d]
                break
    for c in sc2:
        if c in i2toi1:
            del i2toi1[c]
        for d in i1toi2:
            if i1toi2[d] == c:
                del i1toi2[d]
                break
        
    #
    # sc1,sc2 the list of softcore atoms in the native and modified
    # structures
    #
    
    #sc1 = []
    for a in mol1.atoms:
        if a.idx not in i1toi2 and a.idx not in cap1:
            sc1.append( a.idx )
    #sc2 = []
    for a in mol2.atoms:
        if a.idx not in i2toi1 and a.idx not in cap2:
            sc2.append( a.idx )

    #
    # Remove the capped atoms from the common core mapping
    # because we will not apply delta fitting to the caps
    #

    sc1 = list(set(sc1))
    sc2 = list(set(sc2))
    
    for c in cap1:
        if c in i1toi2:
            del i1toi2[c]
        for d in i2toi1:
            if i2toi1[d] == c:
                del i2toi1[d]
                break
            
    for c in cap2:
        if c in i2toi1:
            del i2toi1[c]
        for d in i1toi2:
            if i1toi2[d] == c:
                del i1toi2[d]
                break


    testsc2 = []
    for a in mol2.atoms:
        if a.idx not in cap2 and a.idx not in i2toi1:
            testsc2.append(a.idx)
            
    sc2.sort()
    testsc2.sort()
    if sc2 != testsc2:
        non_matching = list(set(sc2) ^ set(testsc2))
        errmsg = (
            "DeltaRespFit cannot continue because there's an "
            "inconsistency in the mapping, likely due to an "
            "error in the cap definitions.\n"
            "The testsc2 array is list of softcore atoms in the modified structure "
            "that aren't in a cap and aren't in the common-core mapping.\n"
            "The sc2 array is the expected list of softcore atoms if one assumes the "
            "native caps have corresponding matching atoms in the modified structure."
            "\n\n"
            "    sc2 = " + " ".join(["%4s"%(mol2.atoms[i].name) for i in sc2])  + "\n"
            "testsc2 = " + " ".join(["%4s"%(mol2.atoms[i].name) for i in testsc2])  + "\n\n" +
            "Should the following atoms from the modified structure be involved in a cap?\n" +
            " ".join(["%4s"%(mol2.atoms[i].name) for i in non_matching])  + "\n"
        )
        raise Exception(errmsg)
              
            

    fmt = f"%.{args.digits}f"
    net2 = int(round(sum( [a.charge for a in mc2.mols[0].parm.atoms ] )))
    target2 = net2
    cc2 = [i for i in range(len(mc2.mols[0].parm.atoms))]
    for c1,c2 in zip(caps1,caps2):
        target = sum( [a.charge for a in mc1.mols[0].parm.atoms
                       if a.idx in c1] )

        qs = [a.charge for a in mc2.mols[0].parm.atoms
              if a.idx in c2]

        dtarget = float(fmt%(target))
        if abs(target-dtarget) < args.digits/10:
            target=dtarget
        target2 -= target

        qs = FixCharges( qs, args.digits, target=target )
        for q,a in zip(qs,c2):
            mc2.mols[0].parm.atoms[a].charge = q

        s2 = set(c2)
        cc2 = [ i for i in cc2 if i not in s2 ]

    if len(cc2) > 0:
        dtarget2 = float(fmt%(target2))
        if abs(target2-dtarget2) < args.digits/10:
            target2=dtarget2
        
        cc2.sort()
        qs = [a.charge for a in mc2.mols[0].parm.atoms
              if a.idx in cc2]
        qs = FixCharges( qs, args.digits, target=target2 )
        for q,a in zip(qs,cc2):
            mc2.mols[0].parm.atoms[a].charge = q

    qs = [a.charge for a in mc2.mols[0].parm.atoms]
    los_modified.structs[0].data["charges"] = qs


    
    for s in los_modified.structs:
        if s.data["types"] == los_modified.structs[0].data["types"]:
            s.data["charges"] = [q for q in qs]
            #s.data["charge"] = sum(qs)

    newmodified = Path(Path(modified).name).with_suffix(".tmp.json")
    los_modified.save( newmodified )
    los_modified = ReadJsonOrMol2(newmodified)
    mol2 = los_modified[0].GetParmedAtoms()
    oldmodified = modified
    modified = str(newmodified)


    #
    # RESP fit of native system
    #
    targs = copy.deepcopy(vars(args))
    targs["inp"] = native
    targs["confs"] = [native]
    targs["out"] = None
    targs["digits"] = 15
    targs["group"] = native_cap
    targs["prefix"] = "native_"
    for d in ["modified","modified_cap","native","native_cap",
              "mcss_allow_mismatch","dont_modify_cap_charges",
              "mcss_map"]:
        if d in targs:
            del targs[d]
    fit1 = RunRespFit(**targs)

    #
    # RESP fit of the modified system
    #
    targs = copy.deepcopy(vars(args))
    targs["inp"] = modified
    targs["confs"] = [modified]
    targs["out"] = None
    targs["digits"] = 15
    targs["group"] = modified_cap
    targs["prefix"] = "modified_"
    for d in ["modified","modified_cap","native","native_cap",
              "mcss_allow_mismatch","dont_modify_cap_charges",
              "mcss_map"]:
        if d in targs:
            del targs[d]
    fit2 = RunRespFit(**targs)

    # modified system original charges
    oq2s = [ a.charge for a in mol2.atoms ]
    # modified system RESP charges
    fq2s = [ q for q in fit2[0].data["charges"] ]
    # modified system sum of original charges
    to2  = int(round(sum(oq2s)))
    # native system original charges
    oq1s = [ a.charge for a in mol1.atoms ]
    # native system RESP charges
    fq1s = [ q for q in fit1[0].data["charges"] ]
    # native system sum of original charges
    to1  = int(round(sum(oq1s)))

    # Sum of charges in each modified system cap
    sumcap2 = sum( [ oq2s[i] for i in cap2 ] )

    # Difference of charge sums 
    qsc1 = sum( [oq1s[i]-fq1s[i] for i in sc1] )
    ncommon = len(oq2s) - len(sc2) - len(cap2)
    dq = qsc1 / ncommon
    #print("qsc1=",qsc1,"dq=",dq,"ncommon=",ncommon,len(sc2),len(cap2),len(i2toi1),len(i1toi2))

    print("\n\n")
    print("-"*80)
    print("Cap definition consistency between native and modified residues")
    has_error = False
    for icap in range(len(caps1)):
        q1 = sum( [ mol1.atoms[idx].charge for idx in caps1[icap] ] )
        q2 = sum( [ mol2.atoms[idx].charge for idx in caps2[icap] ] )
        error_msg = ""
        if abs(q1-q2) > 1.e-6:
            has_error=True
            error_msg = "   Error: cap charge mismatch"
        print("Cap1: %20s q:%9.6f   Cap 2: %20s q:%9.6f%s"%\
              ( native_cap[icap], q1,
                modified_cap[icap], q2,
                error_msg ))
    if has_error:
        raise Exception("Cap definition mismatch")
    

    print("\n")
    print("-"*80)
    print("MCSS non-cap softcore selections")
    print("sc1:",[mol1.atoms[idx].name for idx in sc1])
    print("sc2:",[mol2.atoms[idx].name for idx in sc2])
    print("\n")
    print("-"*80)
    print("MCSS non-cap common-core mapping")
    for a in i1toi2:
        b = i1toi2[a]
        print("%4s (%4s) => %4s (%4s)"%(a,mol1.atoms[a].name,b,mol2.atoms[b].name))

    print("\n")
    print("-"*80)
    print("Conservation of cap charges (native)")
    for i in range(len(caps1)):
        cap  = caps1[i]
        mask = native_cap[i]
        osum = sum([ oq1s[idx] for idx in cap ])
        fsum = sum([ fq1s[idx] for idx in cap ])
        print("orig sum=%12.8f  fit sum=%12.8f"%(osum,fsum))
    if len(caps1) == 0:
        print("No caps")
    print("\n")
    print("-"*80)
    print("Conservation of cap charges (modified)")
    for i in range(len(caps2)):
        cap  = caps2[i]
        mask = modified_cap[i]
        osum = sum([ oq2s[idx] for idx in cap ])
        fsum = sum([ fq2s[idx] for idx in cap ])
        print("orig sum=%12.8f  fit sum=%12.8f"%(osum,fsum))
    if len(caps2) == 0:
        print("No caps")
    print("\n")

    if not args.dont_modify_cap_charges:
        print("Conservation of cap charges (modified, after digitization)")
        for i in range(len(caps2)):
            cap  = caps2[i]
            osum = sum([ oq2s[idx] for idx in cap ])
            qs = [ fq2s[idx] for idx in cap ]
            qs = FixCharges( qs, args.digits, target=osum )
            fsum = sum(qs)
            for q,idx in zip(qs,cap):
                fq2s[idx] = q
            print("orig sum=%12.8f  fit sum=%12.8f"%(osum,fsum))

    nq2s = [0] * len(oq2s)
    o1sum = 0
    q1sum = 0
    ncount = 0
    for at2 in mol2.atoms:
        i2 = at2.idx
        q2 = fit2[0].data["charges"][i2]
        nq2s[i2] = q2
        
        if i2 in cap2:
            if not args.dont_modify_cap_charges:
                nq2s[i2] = fq2s[i2]
            continue
        
        if i2 in i2toi1:
            i1 = i2toi1[i2]
            at1 = mol1.atoms[i1]
            q1 = fit1[0].data["charges"][i1]
            o1 = at1.charge + dq
            ncount += 1
        else:
            i1=None
            at1=None
            q1 = 0
            o1 = 0
        o1sum += o1
        q1sum += q1
        nq2s[i2] = (q2-q1)+o1

    errmsg = None
    if ncount != ncommon:
        errmsg = ( "Logical error in delta calculation: "
                   f"dq shift applied to {ncount} atoms, "
                   f"but expected {ncommon} atoms. "
                   "This can occur if your cap definitions "
                   "are inconsistent because there is an "
                   "unexpected change in the naming schemes "
                   "between the native and modified molecules "
                   "that you failed to recognize" )
        
    #print("ncount=",ncount)

    common_sum = 0
    common_idx = []
    petite_charges = []
    for at2 in mol2.atoms:
        i2 = at2.idx
        if i2 not in cap2:
            common_sum += nq2s[i2]
            common_idx.append(i2)
            petite_charges.append( nq2s[i2] )

    final = FixCharges( petite_charges, args.digits, target=common_sum )

    #tmp = [ a.charge for a in mol2.atoms ]
    tmp = [nq2s[i] for i in range(len(mol2.atoms))]
    for i,q in zip(common_idx,final):
        tmp[i] = q
    final = tmp
    
    #print(common_sum,sum(petite_charges))
    #for i,q in zip(common_idx,petite_charges):
    #    nq2s[i] = q

    
        
    # freemask = [True]*len(oq2s)
    # #for i in cap2:
    # #    freemask[i] = False
    # fixedqs = FixMaskedCharges( nq2s, args.digits, freemask )

    # print(fixedqs)
    # print(sum(oq2s))
    # print(sum(fq2s))
    # print(sum(nq2s))
    # print(sum(fixedqs))


    print("\n")
    print("-"*80)
    print("Charge changes of non-capped atoms")
    osum=0
    f1sum=0
    f2sum=0
    newsum=0
    finsum=0
    for at2 in mol2.atoms:
        i2 = at2.idx
        q2 = fit2[0].data["charges"][i2]
        if i2 in cap2:
            continue
        if i2 in i2toi1:
            i1 = i2toi1[i2]
            at1 = mol1.atoms[i1]
            q1 = fit1[0].data["charges"][i1]
            o1 = at1.charge
        else:
            i1=None
            at1=None
            q1 = 0
            o1 = 0
        qnew=nq2s[i2]
        qfin=final[i2]
        osum += o1
        f1sum += q1
        f2sum += q2
        newsum += qnew
        finsum += qfin
        print("%4s o1=%10.6f f1=%10.6f f2=%10.6f qnew=%10.6f final=%10.6f"%\
              (at2.name, o1, q1, q2, qnew,qfin))
    
    print("- "*40)
    print("%4s o1=%10.6f f1=%10.6f f2=%10.6f qnew=%10.6f final=%10.6f"%\
          ("sum", osum, f1sum, f2sum, newsum, finsum))

    c1sum = sum([ oq1s[idx] for idx in cap1 ])
    c2sum = sum([ oq2s[idx] for idx in cap2 ])

    print("%4s o1=%10.6f f1=%10.6f f2=%10.6f qnew=%10.6f final=%10.6f"%\
          ("cap", c1sum, c1sum, c2sum, c2sum, c2sum))

    osc1 = sum( [oq1s[i] for i in sc1] )
    fsc1 = sum( [fq1s[i] for i in sc1] )

    print("%4s o1=%10.6f f1=%10.6f f2=%10.6f qnew=%10.6f final=%10.6f"%\
          ("sc", osc1, fsc1, 0, 0, 0))
    
    print("%4s o1=%10.6f f1=%10.6f f2=%10.6f qnew=%10.6f final=%10.6f"%\
          ("net", osum+c1sum+osc1, f1sum+c1sum+fsc1, f2sum+c2sum, newsum+c2sum, finsum+c2sum))

    
    for at2 in mol2.atoms:
        i2 = at2.idx
        fit2[0].data["charges"][i2] = final[i2]

    if errmsg is not None:
        raise Exception(errmsg)
        
    return fit2


