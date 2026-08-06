#!/usr/bin/env python3

def WriteExtLine(fh,pts,val):
    #import psi4.core as p4c
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import AU_PER_ANGSTROM
    x = AU_PER_ANGSTROM()
    fh.write(" XYZ= %14.4f %14.4f %14.4f Q= %14.4f A= %14.4f R= %14.4f C= %14.4f\n"%(pts[0]/x,pts[1]/x,pts[2]/x,val,0,0,0))

    
def WriteAtomicCenterLine(fh,i,pts):
    #import psi4.core as p4c
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import AU_PER_ANGSTROM
    x = AU_PER_ANGSTROM()
    fh.write(" Atomic Center %4i is at %14.6f %14.6f %14.6f\n"%(i,pts[0]/x,pts[1]/x,pts[2]/x))

    
def WriteFitCenterLine(fh,i,pts):
    #import psi4.core as p4c
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import AU_PER_ANGSTROM
    x = AU_PER_ANGSTROM()
    fh.write(" ESP Fit Center %4i is at %14.6f %14.6f %14.6f\n"%(i,pts[0]/x,pts[1]/x,pts[2]/x))


def WriteReadCenterLine(fh,i,pts):
    #import psi4.core as p4c
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import AU_PER_ANGSTROM
    x = AU_PER_ANGSTROM()
    fh.write(" Read-in Center %4i is at %14.6f %14.6f %14.6f\n"%(i,pts[0]/x,pts[1]/x,pts[2]/x))


def WriteESPValues(fh,vals):
    #import psi4.core as p4c
    fh.write(" -----------------------------------------------------------------\n")
    fh.write("    Center     Electric         -------- Electric Field --------\n")
    fh.write("               Potential          X             Y             Z\n")
    fh.write(" -----------------------------------------------------------------\n")
    for i in range(len(vals)):
        fh.write("%6i %17.6f\n"%(i+1,vals[i]))
    fh.write(" -----------------------------------------------------------------\n\n")


def WriteESPPts(fh,pts):
    #import psi4.core as p4c
    fh.write("\n")
    fh.write(" **********************************************************************\n\n")
    fh.write("            Electrostatic Properties Using The SCF Density\n\n")
    fh.write(" **********************************************************************\n\n")
    for i in range(pts.shape[0]):
        WriteReadCenterLine(fh,i+1,pts[i,:])
    fh.write("\n")

    
def WriteExtPts(fh,pts,vals):
    #import psi4.core as p4c
    fh.write("\n\n")
    fh.write(" Background charge distribution read from input stream:\n")
    fh.write(" Point Charges:\n")
    for i in range(pts.shape[0]):
        WriteExtLine(fh,pts[i,:],vals[i])
    fh.write("\n")


def WriteESPOutput(fh,extpts,extvals,esppts,espvals):
    
    if extpts is not None and extvals is not None:
        WriteExtPts(fh,extpts,extvals)
    if esppts is not None and espvals is not None:
        WriteESPPts(fh,esppts)
        WriteESPValues(fh,espvals)
        




def CalcQuickEsp(fname,
                 crds,atnums,esppts,
                 program,nproc,mem,
                 theory,
                 charge,mult,
                 extpts,extvals):

    import os
    import numpy as np
    from .. constants import GetAtomicSymbol, AU_PER_ANGSTROM
    from pathlib import Path
    import subprocess as subp
    import shutil
    
    u = theory.upper()
    cs = u.split("/")
    if len(cs) != 2:
        raise Exception(f"theory '{theory}' expected to have a single '/' char")
    th = cs[0]
    ba = cs[1]
    dft="DFT "
    if th == "HF":
        dft=""
    
    ifname = Path(fname).with_suffix(".inp")
    ioname = Path(fname).with_suffix(".out")
    ilname = Path(fname).with_suffix(".log")
    iename = Path(fname).with_suffix(".esp")

    ifh = open(ifname,"w")
    ifh.write(f"{dft}{th} BASIS={ba} xccutoff=1.0e-12 basiscutoff=1.0e-12 cutoff=1.0e-12 denserms=1.0e-8\n")
    ifh.write(f"CHARGE={charge} MULT={mult} ESP_GRID")
    if extvals is not None and extpts is not None:
        ifh.write(" EXTCHARGES")
    ifh.write("\n\n")

    nat = len(atnums)
    for a in range(nat):
        e = GetAtomicSymbol(atnums[a])
        c = crds[a,:] / AU_PER_ANGSTROM()
        ifh.write("%2s %20.13f %20.13f %20.13f\n"%(e,c[0],c[1],c[2]))
    ifh.write("\n")
    if extvals is not None and extpts is not None:
        for a in range(len(extvals)):
            c = extpts[a,:] / AU_PER_ANGSTROM()
            q = extvals[a]
            ifh.write("%20.13f %20.13f %20.13f %20.13f\n"%(c[0],c[1],c[2],q))
        ifh.write("\n")

    for a in range(esppts.shape[0]):
        c = esppts[a,:] / AU_PER_ANGSTROM()
        ifh.write("%20.13f %20.13f %20.13f\n"%(c[0],c[1],c[2]))
    ifh.write("\n")
    ifh.close()
    
    args = program + f" {ifname}"
    print(f"Running: {args}")
    subp.run(args,shell=True,check=True)
    
    if os.path.exists(ioname):
        shutil.move(ioname,ilname)

    espvals = None
    if os.path.exists(iename):
        espvals = []
        efh = open(iename,"r")
        line = next(efh)
        for kk in range(10):
            line = next(efh)
            cs = line.strip().split()
            if cs[-1] == "ESP":
                break
        for a in range(esppts.shape[0]):
            line = next(efh)
            cs = line.strip().split()
            espvals.append( float(cs[-1]) )
        if len(espvals) != esppts.shape[0]:
            raise Exception(f"Expected to read {esppts.shape[0]} espvals, but only read {len(espvals)} from {iename}")
    else:
        raise Exception(f"File not found {iename}")

    fh = open(ilname,"a")
    WriteESPOutput(fh,extpts,extvals,esppts,espvals)
    fh.close()
    if os.path.exists(iename):
        os.remove(iename)

    
def ReadQuickEsp(fname):
    import os
    if not os.path.exists(fname):
        raise Exception("ReadQuickEsp file not found: %s"%(fname))

    from . GaussianEsp import ReadGaussianEsp

    return ReadGaussianEsp(fname)



def ReadQuickOutput(fname):
    #from ffpopt.constants import GetAtomicNumber
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import GetAtomicNumber
    from .. constants import AU_PER_ANGSTROM
    import os
    import re
    if not os.path.exists(fname):
        raise Exception(f"File not found: {fname}")
    fh = open(fname,"r")
    charge = None
    multiplicity = None
    crds = []
    atnums = []
    
    for line in fh:

        if "TOTAL MOLECULAR CHARGE" in line:
            cs = line.strip().split()
            charge = int(cs[4])
            multiplicity = int(cs[-1])
        if "-- INPUT GEOMETRY --" in line:
            for line in fh:
                cs = line.strip().split()
                if len(cs) == 0:
                    break
                atnums.append( GetAtomicNumber(cs[0]) )
                conv = AU_PER_ANGSTROM()
                crd = [conv*float(cs[1]),conv*float(cs[2]),conv*float(cs[3])]
                crds.append(crd)
        
    return atnums,crds,charge,multiplicity

