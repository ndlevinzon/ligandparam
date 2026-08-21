#!/usr/bin/env python3

def ReadConformer(c,esppar,extpar):
    conf = None
    ok=False
    if not ok:
        try:
            conf = cpefit.Conformer.FromMol2(c,esppar,extpar)
            ok=True
        except:
            ok=False
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
        raise Exception(f"Could not figure out how to create a conformer from {c}")
    return conf




def GetCPEResponse(params,m,c):
    import numpy as np
    #m = mc.mols[0]
    #c = m.conformers[0]
    
    myz = params.zetascl
    myh = params.hardness[ m.paridxs ]
    myu = params.chempot[ m.paridxs ]

    surfqs = np.array([ elem.q for elem in c.extsurf.elems ])
    atomzs = myz * myh**2 * np.pi * 0.5
    gauB,gauBdzeta = c.extsurf.CptGaussianInteractionMatrixAndGrd(atomzs)
    myb = myu + gauB @ surfqs
    E,dq,dqdb,dqdh,dqdz = c.SolveInterCPE(myb,myh,myz)
    
    return dq



if __name__ == "__main__":
    
    import argparse
    import copy
    import numpy as np
    from pathlib import Path
    from ffpopt import cpefit
    import parmed
    
    
    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Read or create a structure and search for conformations""")
    
    
    parser.add_argument \
        ("--inp",
         type=str,
         required=True,
         help="Input mol2 file. The output will be the same as this file but with different charges.")
    
    
    parser.add_argument \
        ("--out",
         type=str,
         required=True,
         help="Output python script")

    
    parser.add_argument \
        ("--program",
         type=str,
         required=False,
         default="psi4",
         help="Ab initio executable. Default: psi4. This could also be gaussian; e.g., --program=g16")

    
    parser.add_argument \
        ("--model",
         type=str,
         required=False,
         default="hf/6-31g*",
         help="Ab initio method used to calculate the electrostatic potential. Default='hf/6-31g*'")

    parser.add_argument \
        ("--psi4-num-threads",
         type=int,
         required=False,
         default=4,
         help="Total CPU-core budget for ab initio ESP (split across concurrent "
              "conformers). Despite the name, this is also used for Gaussian. Default: 4.")

    parser.add_argument \
        ("--psi4-memory",
         type=str,
         required=False,
         default='1gb',
         help="Amount of RAM. Default: '1gb'. Despite the name, this is also used for Gaussian calculations.")

    
    parser.add_argument \
        ("--resp-a",
         type=float,
         required=False,
         default=0.001,
         help="Hyperbolic penalty prefactor. Default: 0.001. The penalty is pen= a * sum_i ( sqrt( qi**2 + b**2 ) - b ), where i loops over all heavy atoms.")

    
    parser.add_argument \
        ("--resp-b",
         type=float,
         required=False,
         default=0.1,
         help="Hyperbolic penalty width. Default: 0.1. The penalty is pen= a * sum_i ( sqrt( qi**2 + b**2 ) - b ), where i loops over all heavy atoms.")
    
    
    parser.add_argument \
        ("--density",
         type=float,
         required=False,
         default=6,
         help="The density of surface points. Default: 6 pts/Ang**2.")
    

    parser.add_argument \
        ("--ext-scale",
         type=float,
         required=False,
         default=1.1,
         help="UFF radius scale factor used to generate the external potential surface. Default: 1.1")

    
    parser.add_argument \
        ("--ext-density",
         type=float,
         required=False,
         default=2,
         help="Density of external potential points. Default: 2 pts/Ang**2")

    
    parser.add_argument \
        ("--ext-lmax",
         type=int,
         required=False,
         default=2,
         help="Maximum spherical harmonic order used to generate perturbations. Default: 2")

    
    parser.add_argument \
        ("--digits",
         type=int,
         required=False,
         default=4,
         help="Round charges to X digits. Default: 4")

    # parser.add_argument \
    #     ("--reopt-cosmo",
    #      action='store_true',
    #      help="Reoptimize the fixed charges after getting the hardness values so the fixed+response charges reproduce the ab initio ESP in the cosmo enfironment")

    
    # parser.add_argument \
    #     ("--reopt-both",
    #      action='store_true',
    #      help="Reoptimize the fixed charges after getting the hardness values so the fixed+response charges reproduce the ab initio ESP in the gas and cosmo environments in an average way")

    
    parser.add_argument \
        ("--opt-zetascl",
         action='store_true',
         help="Optimize the CPE zetascl parameter.")


    
    parser.add_argument \
        ("confs",
         type=str,
         nargs='+',
         help="1-or-more conformers. Either xyz or mol2 files")

    
    args = parser.parse_args()

    
    if args.inp == args.out:
        raise Exception("--inp and --out cannot be the same file")

    if len(args.confs) < 1:
        raise Exception("There must be at least 1 conformer")
    
    scfopts = cpefit.AbInitioOptions\
        (program=args.program,
         theory=args.model,
         mem=args.psi4_memory,
         nproc=args.psi4_num_threads)
    
    esppar = cpefit.SurfaceParameters\
        ([1.4,1.6,1.8,2.0],
         args.density)
    
    extpar = cpefit.SurfaceParameters\
        ([args.ext_scale],
         args.ext_density)


    confs = []
    for c in args.confs:
        confs.append( ReadConformer(c,esppar,extpar) )


    mc = cpefit.MoleculeCollection([cpefit.Molecule(args.inp,confs)])

    params = mc.MakeParams()
    params.resp_a = args.resp_a
    params.resp_b = args.resp_b
        
    m     = mc.mols[0]
    confs = m.conformers
    resname = m.parm.atoms[0].residue.name
    

    oldq = params.q[ m.paridxs ]

    from ffpopt.cpefit.ParallelEsp import (
        run_abinitio_esp_conformers,
        run_cosmo_harmonics_conformers,
    )

    for conf in confs:
        print(len(conf.espsurf.elems),len(conf.extsurf.elems))
    run_abinitio_esp_conformers(confs, scfopts)

    mc.OptimizeFixedCharge(params)
    gasq = params.q[ m.paridxs ]
    newq = cpefit.FixCharges(gasq,args.digits)
    
    for i in range(len(oldq)):
        print("%3i %12.6f %12.6f"%(i,oldq[i],newq[i]))

    mol = m.parm
    for i in range(len(mol.atoms)):
        mol.atoms[i].charge = float("%.6f"%(newq[i]))
    #print(f"Writing {args.out}")
    #mol.save( args.out, overwrite=True )


    run_cosmo_harmonics_conformers(confs, args.ext_lmax, newq, scfopts)

    refhs = np.array(m.refhardness,copy=True)
    mc.OptimizeHardness(params,args.opt_zetascl)

    opths = np.array(mc.hardness,copy=True)
    optcp = np.zeros( opths.shape )
    optzscl = mc.zetascl

    print("\n\n")
    print("-"*75)
    print("Optimized Hardness Parameters")
    print("%-8s %-8s %3s %9s %9s %10s\n"%("AtName","ParName","I","Ref.Hard","Opt.Hard","GasQ"))
    print("-"*75)
    for a in range(len(m.parnames)):
        k = m.paridxs[a]
        print("%-8s %-8s %3i %9.4f %9.4f %10.6f"%\
              (m.atnames[a],m.parnames[a],k,
               refhs[a],mc.hardness[k],newq[a]))
    print("zetascl: %.5f"%(mc.zetascl))
    print("-"*75)
    print("Polarizabilities (au)")
    
    for iconf,conf in enumerate(confs):
        polar = conf.CptPolarizability\
            (mc.hardness[m.paridxs],
             mc.zetascl,
             sym=True)
        print("")
        print("Conformer: %s"%(args.confs[iconf]))
        print("Isotropic: %9.2f"%((polar[0,0]+polar[1,1]+polar[2,2])/3 ))
        for i in range(3):
            print(" ".join(["%16.2f"%(x) for x in polar[i,:]]))
    print("")

    
    #if args.reopt_cosmo or args.reopt_both:
    
    cosmo_confs = []
    for c in args.confs:
        pname = Path(c)
        name = str(pname.with_suffix("")) + "H000pos.log"
        cosmo_confs.append( ReadConformer(name,esppar,extpar) )
            
            
    gas_confs = []
    
    for c in args.confs:
        pname = Path(c)
        if "_" in c:
            name = str(pname.with_suffix(".log"))
        else:
            name = str(pname.with_suffix("")) + "_None.log"
        gas_confs.append( ReadConformer(name,esppar,extpar) )

    
        
    both_confs = cosmo_confs + gas_confs
    
    mcboth = cpefit.MoleculeCollection\
        ([cpefit.Molecule(args.inp,both_confs)])

    mccosmo = cpefit.MoleculeCollection\
        ([cpefit.Molecule(args.inp,cosmo_confs)])

    
    newparams = copy.deepcopy( params )
    newparams.opt_q = True
    newparams.opt_hardness = False
    newparams.opt_chempot  = False
    newparams.opt_zetascl  = False
    
    mccosmo.OptimizeFixedChargeAndCPE(newparams)
    cosmoq  = newparams.q[ m.paridxs ]
    cosmodqs = [ GetCPEResponse(newparams,mccosmo.mols[0],c)
                 for c in mccosmo.mols[0].conformers ]


    
    newparams = copy.deepcopy( params )
    newparams.opt_q = True
    newparams.opt_hardness = False
    newparams.opt_chempot  = False
    newparams.opt_zetascl  = False

    mcboth.OptimizeFixedChargeAndCPE(newparams)
    bothq  = newparams.q[ m.paridxs ]
    bothdq = [ GetCPEResponse(newparams,mcboth.mols[0],c)
               for c in mcboth.mols[0].conformers ]

    
    newparams = copy.deepcopy( params )
    newparams.opt_q = True
    newparams.opt_hardness = False
    newparams.opt_chempot  = False
    newparams.opt_zetascl  = False
    
    mccosmo.OptimizeFixedCharge(newparams)
    onlycosmoq  = newparams.q[ m.paridxs ]
    

    
    fh = open(args.out,"w")
    fh.write("%s%s%s\n"%("#","!","/usr/bin/env python3"))
    fh.write("import sys\n")
    fh.write("import numpy as np\n")
    fh.write("import argparse\n")
    fh.write("from parmed import load_file\n")
    fh.write("from parmed.amber.mask import AmberMask\n")

    fh.write("parser = argparse.ArgumentParser(\"replace charges and add lrch parameters\")\n")
    fh.write(f"parser.add_argument(\"--resname\",default=\"{resname}\",help=\"Name of the residue, default: {resname}\",type=str)\n")
    fh.write("parser.add_argument(\"--cosmo\",help=\"Use charges fit to reproduce the ESP in a COSMO environment\",action='store_true')\n")
    fh.write("parser.add_argument(\"--cpecosmo\",help=\"Use charges fit to reproduce the CPE ESP polarized by COSMO\",action='store_true')\n")
    fh.write("parser.add_argument(\"--cpeboth\",help=\"Use charges fit to reproduce the CPE ESP polarized by COSMO and gas phase in an average way\",action='store_true')\n")

    fh.write("parser.add_argument(\"iparm\",help=\"Input parm7\")\n")
    fh.write("parser.add_argument(\"oparm\",help=\"Output parm7\")\n")
    
    fh.write("args = parser.parse_args()\n")
    fh.write("rname = args.resname\n")
    
    fh.write("if args.iparm == args.oparm:\n")
    fh.write("    raise Exception(\"The 2 filenames must be different\")\n\n")
    
    fh.write("p = load_file( args.iparm )\n\n")

    anames = [ ":%s@%s"%("{rname}",a.name) for a in m.parm.atoms ]

    fh.write("anames = [%s]\n"%\
             ( ",".join(["f\"%s\""%(name)
                                           for name in anames]) ))
    fh.write("qgas = [%s]\n"%\
             ( ",".join(["%.6f"%(q)
                         for q in cpefit.FixCharges(gasq,args.digits)]) ))

    
    
    fh.write("qcosmo = [%s]\n"%\
             ( ",".join(["%.6f"%(q)
                         for q in cpefit.FixCharges(onlycosmoq,args.digits)]) ))
    
    fh.write("qcpecosmo = [%s]\n"%\
             ( ",".join(["%.6f"%(q)
                         for q in cpefit.FixCharges(cosmoq,args.digits)]) ))

    fh.write("qcpeboth = [%s]\n"%\
             ( ",".join(["%.6f"%(q)
                         for q in cpefit.FixCharges(bothq,args.digits)]) ))

    fh.write("hardness = [%s]\n"%\
             ( ",".join(["%.8f"%(opths[ m.paridxs[a] ])
                         for a in range(len(gasq))])))

    fh.write("chempot = [%s]\n"%\
             ( ",".join(["%.8f"%(optcp[ m.paridxs[a] ])
                         for a in range(len(gasq))])))
    
    fh.write("zetascl = %.8f\n\n"%(optzscl))

    
    fh.write("idxs = [ [ i for i in AmberMask(p,mask).Selected() ][0] for mask in anames ]\n\n")
    fh.write("qs = qgas\n")
    fh.write("if args.cosmo:\n")
    fh.write("    qs = qcosmo\n")
    fh.write("elif args.cpecosmo:\n")
    fh.write("    qs = qcpecosmo\n")
    fh.write("elif args.cpeboth:\n")
    fh.write("    qs = qcpeboth\n\n")
    
    fh.write("nat = len(p.atoms)\n")
    fh.write("glbcp = np.array([0.0]*nat)\n")
    fh.write("glbhs = np.array([0.0]*nat)\n")
    fh.write("for i,a in enumerate(idxs):\n")
    fh.write("    p.atoms[a].charge = qs[i]\n")
    fh.write("    glbcp[a] = chempot[i]\n")
    fh.write("    glbhs[a] = hardness[i]\n\n")

    fh.write("flag=\"LRCH_ELECNEG_LIST\"\n")
    fh.write("p.add_flag(flag,'5E16.8',data=glbcp)\n")

    fh.write("flag=\"LRCH_HARDNESS_LIST\"\n")
    fh.write("p.add_flag(flag,'5E16.8',data=glbhs)\n")

    fh.write("p.save(args.oparm,overwrite=True)\n")

    
    

