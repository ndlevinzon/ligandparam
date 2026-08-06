#!/usr/bin/env python3



def ReadPsi4Output(fname):
    #from ffpopt.constants import GetAtomicNumber
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import GetAtomicNumber
    from .. constants import AU_PER_ANGSTROM
    import os
    import re
    if not os.path.exists(fname):
        raise Exception(f"File not found: {fname}")
    fh = open(fname,"r")
    search = re.compile(r" +Geometry \(in ([A-Za-z]+)\), charge = *([\-0-9]+), multiplicity = *([0-9]+):.*")
    units = None
    charge = None
    multiplicity = None
    crds = []
    atnums = []
    
    for line in fh:
        m = search.match(line)
        if m is not None:
            units = m.group(1)
            charge = int(m.group(2))
            multiplicity = int(m.group(3))
            line = next(fh)
            line = next(fh)
            line = next(fh)
            while True:
                line = next(fh)
                if "----" in line:
                    break
                cs = line.strip().split()
                if len(cs) != 5:
                    break
                atnums.append( GetAtomicNumber(cs[0]) )
                conv = None
                if units == "Bohr":
                    conv = 1
                elif units == "Angstrom":
                    conv = AU_PER_ANGSTROM()
                else:
                    raise Exception(f"Unknown units: {units}")
                    
                crd = [ conv*float(cs[1]), conv*float(cs[2]), conv*float(cs[3])  ]
                crds.append(crd)
    return atnums,crds,charge,multiplicity


def WriteExtLine(pts,val):
    import psi4.core as p4c
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import AU_PER_ANGSTROM
    x = AU_PER_ANGSTROM()
    p4c.print_out(" XYZ= %14.4f %14.4f %14.4f Q= %14.4f A= %14.4f R= %14.4f C= %14.4f\n"%(pts[0]/x,pts[1]/x,pts[2]/x,val,0,0,0))

    
def WriteAtomicCenterLine(i,pts):
    import psi4.core as p4c
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import AU_PER_ANGSTROM
    x = AU_PER_ANGSTROM()
    p4c.print_out(" Atomic Center %4i is at %14.6f %14.6f %14.6f\n"%(i,pts[0]/x,pts[1]/x,pts[2]/x))

    
def WriteFitCenterLine(i,pts):
    import psi4.core as p4c
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import AU_PER_ANGSTROM
    x = AU_PER_ANGSTROM()
    p4c.print_out(" ESP Fit Center %4i is at %14.6f %14.6f %14.6f\n"%(i,pts[0]/x,pts[1]/x,pts[2]/x))


def WriteReadCenterLine(i,pts):
    import psi4.core as p4c
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import AU_PER_ANGSTROM
    x = AU_PER_ANGSTROM()
    p4c.print_out(" Read-in Center %4i is at %14.6f %14.6f %14.6f\n"%(i,pts[0]/x,pts[1]/x,pts[2]/x))


def WriteESPValues(vals):
    import psi4.core as p4c
    p4c.print_out(" -----------------------------------------------------------------\n")
    p4c.print_out("    Center     Electric         -------- Electric Field --------\n")
    p4c.print_out("               Potential          X             Y             Z\n")
    p4c.print_out(" -----------------------------------------------------------------\n")
    for i in range(len(vals)):
        p4c.print_out("%6i %17.6f\n"%(i+1,vals[i]))
    p4c.print_out(" -----------------------------------------------------------------\n\n")


def WriteESPPts(pts):
    import psi4.core as p4c
    p4c.print_out("\n")
    p4c.print_out(" **********************************************************************\n\n")
    p4c.print_out("            Electrostatic Properties Using The SCF Density\n\n")
    p4c.print_out(" **********************************************************************\n\n")
    for i in range(pts.shape[0]):
        WriteReadCenterLine(i+1,pts[i,:])
    p4c.print_out("\n")

    
def WriteExtPts(pts,vals):
    import psi4.core as p4c
    p4c.print_out("\n\n")
    p4c.print_out(" Background charge distribution read from input stream:\n")
    p4c.print_out(" Point Charges:\n")
    for i in range(pts.shape[0]):
        WriteExtLine(pts[i,:],vals[i])
    p4c.print_out("\n")


def WriteESPOutput(extpts,extvals,esppts,espvals):
    
    if extpts is not None and extvals is not None:
        WriteExtPts(extpts,extvals)
    if esppts is not None and espvals is not None:
        WriteESPPts(esppts)
        WriteESPValues(espvals)
        
    

def CalcPsi4Esp(fname,
                crds,atnums,esppts,
                program,nproc,mem,
                theory,
                charge,mult,
                extpts,extvals):


    import os
    scratch_unset = False
    if "PSI_SCRATCH" not in os.environ:
        scratch_unset = True
        os.environ["PSI_SCRATCH"] = os.getcwd()

    import numpy as np
    from .. constants import GetAtomicSymbol
    from pathlib import Path
    import subprocess as subp

    RUNHERE=False
    
    nat = len(atnums)
    crdlines = [f"{charge} {mult}\n"]
    for i in range(nat):
        ele = GetAtomicSymbol(atnums[i])
        xx,xy,xz = crds[i,0],crds[i,1],crds[i,2]
        line = "%2s %22.13e %22.13e %22.13e\n"%\
            (ele,xx,xy,xz)
        crdlines.append(line)
    crdlines.append("units bohr\n")
    crdlines.append("symmetry c1\n")
    crdlines.append("no_reorient\n")
    crdlines.append("no_com\n")
    geom = "".join(crdlines)


    xpts = None
    xptslines = []
    if extvals is not None and extpts is not None:
        #
        # stack the extvals vector to be the new first column
        # of the xpts matrix
        #
        xpts = np.hstack((extvals.reshape(-1,1), extpts))
        for i in range(xpts.shape[0]):
            if i == 0:
                xptslines.append("[%s,\n"%(xpts[i,:].tolist()))
            elif i == xpts.shape[0]-1:
                xptslines.append("%s]"%(xpts[i,:].tolist()))
            else:
                xptslines.append("%s,\n"%(xpts[i,:].tolist()))
        xptslines = "".join(xptslines)



    if RUNHERE:
        import psi4
        import psi4.core as p4c
    
        psi4.set_output_file(fname)
        p4c.set_num_threads(nproc)
        psi4.set_memory(mem)

        psi4mol = psi4.geometry(geom)

        props=[]
        if esppts is not None:
            props=['GRID_ESP']
        
        e,wfn = psi4.energy\
            (theory,
             molecule=psi4mol,
             properties=props,
             return_wfn=True,
             external_potentials=xpts)

        espvals = None
        if esppts is not None:
            myepc = p4c.ESPPropCalc(wfn)
            psi4_matrix = p4c.Matrix.from_array(esppts)
            espvals = np.array(myepc.compute_esp_over_grid_in_memory(psi4_matrix))

        p4c.flush_outfile()
        WriteESPOutput(extpts,extvals,esppts,espvals)
        p4c.flush_outfile()
        p4c.close_outfile()

    else:
        ifname = Path(fname).with_suffix(".inp")
        ifh = open(ifname,"w")
        ifh.write(f"""
import numpy as np
import psi4
import psi4.core as p4c
from ffpopt.cpefit.Psi4Esp import WriteESPOutput

psi4.set_output_file(\"{fname}\")
p4c.set_num_threads({nproc})
psi4.set_memory(\"{mem}\")

""")
    
        ifh.write("esppts=np.array([\n")
        for i in range(esppts.shape[0]):
            if i < esppts.shape[0]-1:
                ifh.write("%s,\n"%(esppts[i,:].tolist()))
            else:
                ifh.write("%s])\n"%(esppts[i,:].tolist()))

        if xpts is None:
            ifh.write("xpts=None\n")
        else:
            ifh.write(f"xpts=np.array({xptslines})\n")
        
        ifh.write(f"""
geom = \"\"\"{geom}\"\"\"

psi4mol = psi4.geometry(geom)
props=['GRID_ESP']
        
""")

#         cols = theory.split(r"//")
#         cols = cols[0].split(r"/")
#         basis = None
#         if len(cols) > 1:
#             basis = cols[1]

#         halogenbasis = None
#         if basis is not None:
#             halogenbasis = basis.replace("+","")
#             if halogenbasis == basis:
#                 halogenbasis = None

#         if halogenbasis is not None:
#             ifh.write("""
# basis_block = \"\"\"
# basis {
#   assign %s
#   assign I %s
#   assign Br %s
# }
# \"\"\"

# psi4.set_options({'basis': basis_block})

# """%(basis,halogenbasis,halogenbasis))
        
        ifh.write(f"""
e,wfn = psi4.energy\\
        (\"{theory}\",
         molecule=psi4mol,
         properties=props,
         return_wfn=True,
         external_potentials=xpts)

myepc = p4c.ESPPropCalc(wfn)
psi4_matrix = p4c.Matrix.from_array(esppts)
espvals = np.array(myepc.compute_esp_over_grid_in_memory(psi4_matrix))

p4c.flush_outfile()
if xpts is not None:
    WriteESPOutput(xpts[:,1:4],xpts[:,0],esppts,espvals)
else:
    WriteESPOutput(None,None,esppts,espvals)
p4c.flush_outfile()

""")
        ifh.close()

        print(f"Running: {program} {ifname}")
        subp.run(f"{program} {ifname}",shell=True,check=True)
        pname = Path(fname)
        out = pname.with_suffix(".out")
        npy = pname.with_suffix(".*.npy")
        for f in [str(out),str(npy)]: #,fname+".p4c"]:
            if os.path.exists(f):
                os.remove(f)
    
    if scratch_unset:
        del os.environ["PSI_SCRATCH"]


def ReadPsi4Esp(fname):
    import os
    if not os.path.exists(fname):
        raise Exception("ReadPsi4Esp file not found: %s"%(fname))

    from . GaussianEsp import ReadGaussianEsp

    return ReadGaussianEsp(fname)

