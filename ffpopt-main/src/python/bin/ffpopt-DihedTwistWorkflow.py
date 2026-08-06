#!/usr/bin/env python3

def FindDefaultValue(key,parser):
    default_value_of_my_arg = None
    for action in parser._actions:
        if action.dest == key:
            default_value_of_my_arg = action.default
            break
    return default_value_of_my_arg

def GetNondefaultArgs(args,parser,skip=[]):
    s = []
    for name,value in vars(args).items():
        #print(name,value)
        dvalue = FindDefaultValue(name,parser)
        myname = name.replace("_","-")
        if myname in skip:
            continue
        if dvalue != value:
            if isinstance(value, list):
                for v in value:
                    s.append( [f"--{myname}",f"\"{v}\""] )
            elif value != True:
                if isinstance(value, str):
                    s.append( [f"--{myname}",f"'{value}'"] )
                else:
                    s.append( [f"--{myname}",f"{value}"] )
            else:
                s.append( [f"--{myname}"] )

    #return s
    return " ".join( [ " ".join(f) for f in s ] )




class Parameter(object):
    def __init__(self,mol,idxs):
        self.idxs=idxs
        self.res=mol.atoms[idxs[0]].residue.name
        self.names=[mol.atoms[i].name for i in idxs]
        self.types=[mol.atoms[i].type for i in idxs]
        self.instances = []
        
    def GetIdxStr(self):
        return "-".join(["%i"%(x) for x in self.idxs])

    def GetNameStr(self):
        return "-".join(self.names)

    def GetTypeStr(self):
        return "-".join(self.types)

    
    def GetParamByName(self):
        return "%s_%s"%(self.res,self.GetNameStr())
     
    def GetParamByType(self):
        return "%s_%s"%(self.res,self.GetTypeStr())


    
    def GetNameMasks(self):
        return [ f"@{name}" for name in self.names ]

    def GetTypeMasks(self):
        return [ f"@%%{name}" for name in self.types ]

    def AsDict(self):
        return { self.GetParamByType():
                 { "nprim": args.nprim,
                   "masks": None } }
    


if __name__ == "__main__":
    import sys
    import json
    import copy
    import shlex
    import argparse
    from pathlib import Path
    from ffpopt.Struct import ListOfStruct
    from ffpopt.Options import AddStandardOptions
    
    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Perform sets up a workflow script for dihedral scans and fits""" )


    
    parser.add_argument\
        ("--inp",
         help="Input json file. Only the first structure is used.",
         type=str,
         required=True)
    
    parser.add_argument\
        ("--bond",
         help="Two 0-based integers separated by a comma. This option can be used more than once",
         action='append',
         required=True)

    parser.add_argument\
        ("--delta",
         help="Scan step size. Default: 10 degrees",
         type=int,
         default=10)
    
    parser.add_argument\
        ("--nprim",
         help="Number of dihedral primitives. Default: 3",
         type=int,
         default=3)
    
    parser.add_argument\
        ("--maxiter",
         help="Number of training iterations. Each iteration repeats the sander scans. Default: 2",
         type=int,
         default=2)
    
    parser.add_argument\
        ("--bytype",
         help="If present, then",
         action='store_true')

    
    parser.add_argument\
        ("--nlmaxiter",
         help="Maximum number of nonlinear optimization steps. Default: 300",
         type=int,
         default=300)
    
    parser.add_argument\
        ("--seqscan",
         action='store_true',
         help="If present, then the script will generate a sequential scan script. Default: False",
         default=False)
    
    parser.add_argument\
        ("--nproc",
         help="Number of processes to use for wavefront scans. Ideally set for 10-16. Default: 1",
         type=int,
         default=1)
    parser.add_argument\
        ("--wf_starting_nodes",
         help="Number of starting nodes to use for wavefront scans. Default: 4",
         type=int,
         default=4)        
    parser.add_argument\
        ("--wf_num_conformers",
         help="Number of conformers to generate at each level. Default: 0 (no conformers).",
         type=int,
         default=0)
    
    parser.add_argument\
        ("--wf_max_levels",
         help="Maximum number of levels to explore in the wavefront. -1 means infinite. Default: -1.",
         type=int,
         default=-1)

    parser.add_argument \
        ("--wf_convergence_threshold",
         help="Energy convergence threshold (kcal/mol) a revisited angle must beat to keep spawning levels. Default: 0.01",
         default=0.01,
         type=float)
    
    parser.add_argument\
        ("--twistrst_sp_opt",
         help="If present, then the script will optimize during single points.",
         action='store_true',
         default=False
         )

    parser.add_argument\
        ("--quiet",
         help="If present, append '2>> error.log' to each emitted scan command so backend stderr (xTB SCF cycles, BFGS step log, framework warnings) is appended to a single error.log file instead of the terminal.",
         action='store_true',
         default=False)
        
    
    
    
    AddStandardOptions(parser)

    args = parser.parse_args()

    los = ListOfStruct.from_file( args.inp )
    los.structs = [ los.structs[0] ]
    los.SetArgs(args)

    mol = los.structs[0].ReadAmberParm()
    origparm = los.structs[0].data["parm"]
    
    useropts = GetNondefaultArgs(args,parser,
                                 skip=["bond","delta","bytype",
                                       "nprim","maxiter",
                                       "inp","model",
                                       "nlmaxiter", "nproc", "wf-starting-nodes",
                                       "wf-num-conformers", "wf-max-levels",
                                       "wf-convergence-threshold",
                                       "quiet",
                                       "scantype", "seqscan"])

    redir = " 2>> error.log" if args.quiet else ""
    
    bonds = [ [int(x) for x in bond.split(",") ] for bond in args.bond ]

    output = "global.frcmod"
    params = {}
    systems = []

    s = { #"parm": None,
          #"crd": args.crd,
          "output": None,
          "params": {},
          "profiles": [] }

    scans = []

    allparams = []
    ps = {}
    for bond in bonds:
        made_scan = False
        for d in mol.dihedrals:
            if d.improper:
                continue
            idxs = [d.atom1.idx,d.atom2.idx,d.atom3.idx,d.atom4.idx]
            myidxs = []
            if idxs[1] == bond[0] and idxs[2] == bond[1]:
                myidxs = idxs
            elif idxs[2] == bond[0] and idxs[1] == bond[1]:
                myidxs = idxs[::-1]
            else:
                continue
            idxs = myidxs
            p = Parameter(mol,idxs)
            name = p.GetParamByType()
            if name not in ps:
                ps[name] = p
            ps[name].instances.append( p.GetNameMasks() )
            allparams.append( name )

            if not made_scan:
                scans.append(p)
                made_scan=True
        if not made_scan:
            raise ValueError(
                f"--bond {bond[0]},{bond[1]} has no proper dihedral with that pair as the central bond. "
                f"This usually means at least one of the two atoms is terminal (no bonded neighbors beyond the other). "
                f"Check your bond indices (0-based) against the parm topology.")

    uparams = list(set(allparams))
    if not args.bytype:
        for name in ps:
            s["params"][name] = ps[name].instances

        for name in uparams:
            params[name] = { "nprim": args.nprim, "masks": None }
    else:
        
        for name in uparams:
            typestr = name.split("_")[1]
            ts = [ f"@%{t}" for t in typestr.split("-") ]
            params[name] = { "nprim": args.nprim,
                             "masks": [ts]  }

    fh = sys.stdout

    fh.write("%s%s%s\n"%("#","!","/bin/bash"))
    fh.write("set -e\n")
    fh.write("set -u\n")

    for scan in scans:
        hlname = args.model.replace("/","_")
        oscan = hlname + "_" + scan.GetIdxStr()
        idxs = ",".join(["%i"%(x) for x in scan.idxs])
        opts = useropts


        fh.write("\n")
        fh.write(f"if [ ! -e {oscan}.json ]; then\n")
        fh.write(f"    echo Creating {oscan}.json\n")
        if args.seqscan:
            fh.write(f"    ffpopt-DihedScan.py --inp {args.inp} --model {args.model} --dihed=\"{idxs}\" \\\n")
            fh.write(f"        --out {oscan}.json --delta {args.delta} {opts}{redir}\n\n")
        else:
            fh.write(f"    ffpopt-DihedWavefront.py --inp {args.inp} --model {args.model} --dihed=\"{idxs}\" \\\n")
            fh.write(f"        --out {oscan}.json --delta {args.delta} --nproc {args.nproc} --wf_starting_nodes {args.wf_starting_nodes} --wf_max_levels {args.wf_max_levels} --wf_num_conformers {args.wf_num_conformers} --wf_convergence_threshold {args.wf_convergence_threshold} {opts}{redir}\n\n")
        fh.write(f"fi\n")
        fh.write("\n")



    for scan in scans:
        oscan = "orig" +"_" + scan.GetIdxStr()
        idxs = ",".join(["%i"%(x) for x in scan.idxs])
        opts=useropts

        fh.write("\n")
        fh.write(f"if [ ! -e {oscan}.json ]; then\n")
        fh.write(f"    echo Creating {oscan}.json\n")
        if args.seqscan:
            fh.write(f"    ffpopt-DihedScan.py --inp {args.inp} --model=sander --dihed=\"{idxs}\" \\\n")
            fh.write(f"        --out {oscan}.json --delta {args.delta}  {opts}{redir}\n\n")
        else:
            fh.write(f"    ffpopt-DihedWavefront.py --inp {args.inp} --model=sander --dihed=\"{idxs}\" \\\n")
            fh.write(f"        --out {oscan}.json --delta {args.delta} --nproc {args.nproc} --wf_starting_nodes {args.wf_starting_nodes} --wf_max_levels {args.wf_max_levels} --wf_num_conformers {args.wf_num_conformers} --wf_convergence_threshold {args.wf_convergence_threshold} {opts}{redir}\n\n")
        fh.write(f"fi\n")
        fh.write("\n")

        
    
    
    for it in range(args.maxiter):
    
        
        pitname = "it%02i"%(it)
        citname = "it%02i"%(it+1)
        ss = copy.deepcopy(s)
        if it == 0:
            parm = origparm
        else:
            parm = pitname + ".parm7"
        scanname = citname
        ss["output"] = citname + ".py"
        ss["parm"] = parm


        hlname = args.model.replace("/","_")

        ss["profiles"] = []
        for scan in scans:
            if it == 0:
                llscan = "orig"
            else:
                llscan = pitname
            prof = { "hl": "%s_%s.json"%(hlname,scan.GetIdxStr()),
                     "ll": "%s_%s.json"%(llscan,scan.GetIdxStr()),
                     "name": citname,
                     "plots": [ f"{scan.GetParamByType()}" ] }
            ss["profiles"].append(prof)
        
        datadict = { "params": params,
                     "output": f"{citname}.frcmod",
                     "systems": [ss] }

        jfh = open(f"{citname}.fit.json","w")
        json.dump(datadict,jfh,indent=4)
        jfh.close()

        fh.write("\n")
        fh.write(f"if [ ! -e {citname}.py ]; then\n")
        fh.write(f"    echo Creating {citname}.py\n")

        fh.write(f"    ffpopt-GenDihedFit.py --nlmaxiter={args.nlmaxiter} {citname}.fit.json\n")
        fh.write("fi\n")
        fh.write("\n")

        fh.write("\n")
        fh.write(f"if [ ! -e {citname}.parm7 -o ! -e {citname}.json ]; then\n")
        fh.write(f"    echo Creating {citname}.parm7\n")
        fh.write(f"    python3 {citname}.py {origparm} {citname}.parm7\n")
        fh.write(f"    ffpopt-PrepareInput.py --update --parm={citname}.parm7 --crd={args.inp} --out={citname}.json\n")

        fh.write("fi\n")
        fh.write("\n")

        
        for scan in scans:
            
            oscan = scanname +"_" + scan.GetIdxStr()
            idxs = ",".join(["%i"%(x) for x in scan.idxs])
            #opts = " ".join([ " ".join(useropts)])
            opts=useropts

            fh.write("\n")
            fh.write(f"if [ ! -e {oscan}.json ]; then\n")
            fh.write(f"    echo Creating {oscan}.json\n")
            if args.seqscan:
                fh.write(f"    ffpopt-DihedScan.py --inp {citname}.json --model sander --dihed=\"{idxs}\" \\\n")
                fh.write(f"        --out {oscan}.json --delta {args.delta}  {opts}{redir}\n\n")
            else:
                fh.write(f"    ffpopt-DihedWavefront.py --inp {citname}.json --model sander --dihed=\"{idxs}\" \\\n")
                fh.write(f"        --out {oscan}.json --delta {args.delta} --nproc {args.nproc} --wf_starting_nodes {args.wf_starting_nodes} --wf_max_levels {args.wf_max_levels} --wf_num_conformers {args.wf_num_conformers} --wf_convergence_threshold {args.wf_convergence_threshold} {opts}{redir}\n\n")
            fh.write(f"fi\n")
            fh.write("\n")


    fh.close()
