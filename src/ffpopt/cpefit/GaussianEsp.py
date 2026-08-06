#!/usr/bin/env python3


def WriteAtomicCenterLine(fh,i,pts):
    #import psi4.core as p4c
    #from ffpopt.constants import AU_PER_ANGSTROM
    from .. constants import AU_PER_ANGSTROM
    x = AU_PER_ANGSTROM()
    fh.write(" Atomic Center %4i is at  %14.6f %14.6f %14.6f\n"%(i,pts[0]/x,pts[1]/x,pts[2]/x))

    
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
        fh.write("%5i Fit %17.6f\n"%(i+1,vals[i]))
    fh.write(" -----------------------------------------------------------------\n\n")


# def WriteESPPts(fh,pts):
#     #import psi4.core as p4c
#     fh.write("\n")
#     fh.write(" **********************************************************************\n\n")
#     fh.write("            Electrostatic Properties Using The SCF Density\n\n")
#     fh.write(" **********************************************************************\n\n")
#     for i in range(pts.shape[0]):
#         WriteReadCenterLine(fh,i+1,pts[i,:])
#     fh.write("\n")




def WriteGaussianEsp\
    (fname,crds,
     atnums,potcrds,
     nproc,mem,theory,charge,mult,
     extcrds,extqs):

    import numpy as np
    from .. constants import AU_PER_ANGSTROM

    hasext = (extqs is not None) and (extcrds is not None)
    
    fh = open(fname,"w")
    crds = np.array(crds,copy=True) / AU_PER_ANGSTROM()
    potcrds = np.array(potcrds,copy=True) / AU_PER_ANGSTROM()
    
    fh.write("%%mem=%s\n"%(mem))
    fh.write("%%chk=%s\n"%(fname + ".chk"))
    fh.write("%%nproc=%i\n"%(nproc))

    opstr = f"#p {theory} NoSymm Prop(Potential,Read)"
    if hasext:
        opstr += " Charge"
    
    fh.write(f"{opstr}\n\ntitle\n\n")
    fh.write(f"{charge} {mult}\n")
    nat = crds.shape[0]
    for a in range(nat):
        fh.write("%2i %19.13f %19.13f %19.13f\n"%\
                 (atnums[a],crds[a,0],crds[a,1],crds[a,2]))
    fh.write("\n")
    if hasext:
        extcrds = np.array(extcrds,copy=True) / AU_PER_ANGSTROM()
        for a in range(len(extqs)):
            fh.write("%19.13f %19.13f %19.13f %19.13f\n"%\
                 (extcrds[a,0],extcrds[a,1],extcrds[a,2],extqs[a]))
        fh.write("\n")
    for a in range(potcrds.shape[0]):
        fh.write("%19.13f %19.13f %19.13f\n"%\
                 (potcrds[a,0],potcrds[a,1],potcrds[a,2]))
    fh.write("\n")

    

def WriteGaussianPolar\
    (fname,crds,
     atnums,potcrds,
     nproc,mem,theory,charge,mult,
     extcrds,extqs):

    import numpy as np
    from .. constants import AU_PER_ANGSTROM

    hasext = (extqs is not None) and (extcrds is not None)
    haspot = False
    if potcrds is not None:
        if potcrds.shape[0] > 0:
            haspot = True

    fh = open(fname,"w")
    crds = np.array(crds,copy=True) / AU_PER_ANGSTROM()
    
    fh.write("%%mem=%s\n"%(mem))
    fh.write("%%chk=%s\n"%(fname + ".chk"))
    fh.write("%%nproc=%i\n"%(nproc))
    
    opstr = f"#p {theory} NoSymm Polar"
    if haspot:
        optstr += " Prop(Potential,Read)"
    if hasext:
        opstr += " Charge"
    
    fh.write(f"{opstr}\n\ntitle\n\n")
    fh.write(f"{charge} {mult}\n")
    nat = crds.shape[0]
    for a in range(nat):
        fh.write("%2i %19.13f %19.13f %19.13f\n"%\
                 (atnums[a],crds[a,0],crds[a,1],crds[a,2]))
    fh.write("\n")
    if hasext:
        extcrds = np.array(extcrds,copy=True) / AU_PER_ANGSTROM()
        for a in range(len(extqs)):
            fh.write("%19.13f %19.13f %19.13f %19.13f\n"%\
                 (extcrds[a,0],extcrds[a,1],extcrds[a,2],extqs[a]))
        fh.write("\n")
    if haspot:
        potcrds = np.array(potcrds,copy=True) / AU_PER_ANGSTROM()
        for a in range(potcrds.shape[0]):
            fh.write("%19.13f %19.13f %19.13f\n"%\
                     (potcrds[a,0],potcrds[a,1],potcrds[a,2]))
    fh.write("\n")



    
    

def ReadGaussianEsp(fname):
    import numpy as np
    import re
    import os
    import sys
    from .. constants import AU_PER_ANGSTROM

    if not os.path.exists(fname):
        raise Exception("ReadGauEsp file not found: %s"%(fname))
    
    fh = open(fname,"r")
    
    crds=[]
    pts=[]
    esp=[]
    extpts=[]
    extqs=[]

    atomic_center_line = re.compile(r" +Atomic Center[ 0-9]+ is at +([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6})")
    fit_center_line = re.compile(r" +ESP Fit Center[ 0-9\*]+ is at +([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6})")
    read_center_line = re.compile(r" +Read-in Center[ 0-9\*]+ is at +([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6})")
    
    esp_line = re.compile(r"[ 0-9\*]+ Fit +([\-\.0-9]+)")
    ext_line = re.compile(r" XYZ= *([\-0-9]{1,5}\.[0-9]{4}) *([\-0-9]{1,5}\.[0-9]{4}) *([\-0-9]{1,5}\.[0-9]{4}) Q= *([\-0-9]{1,5}\.[0-9]{4}) A= *([\-0-9]{1,5}\.[0-9]{4}) R= *([\-0-9]{1,5}\.[0-9]{4}) C= *([\-0-9]{1,5}\.[0-9]{4})")


    for line in fh:
        m = ext_line.match(line)
        if m is not None:
            extpts.append( [ float( m.group(1) ), float( m.group(2) ), float( m.group(3) ) ] )
            extqs.append( float( m.group(4) ) )
        m = atomic_center_line.match(line)
        if m is not None:
            crds.append( [ float( m.group(1) ), float( m.group(2) ), float( m.group(3) ) ] )
            continue
        m = fit_center_line.match(line)
        if m is not None:
            pts.append( [ float( m.group(1) ), float( m.group(2) ), float( m.group(3) ) ] )
            continue
        m = read_center_line.match(line)
        if m is not None:
            pts.append( [ float( m.group(1) ), float( m.group(2) ), float( m.group(3) ) ] )
            continue


        if "  Center     Electric  " in line:
            line = next(fh)
            line = next(fh)
            for line in fh:
                #print(line)
                if "----------" in line:
                    break
                cs = line.strip().split()
                if len(cs) == 2 or len(cs) == 3:
                    if cs[1] != "Atom":
                        esp.append( float(cs[-1]) )

    crds = np.array(crds) * AU_PER_ANGSTROM()
    pts = np.array(pts) * AU_PER_ANGSTROM()
    esp = np.array(esp)
    
    if pts.shape[0] != esp.shape[0]:
        raise Exception("# pts (%i) != # esp values (%i) in %s"%(pts.shape[0],len(esp),fname))


    if len(extpts) > 0:
        extpts = np.array(extpts) * AU_PER_ANGSTROM()
        extqs = np.array(extqs)
    else:
        extpts = None
        extqs = None
    return crds,pts,esp,extpts,extqs





def ReadGaussianPolar(fname):
    import numpy as np
    import re
    import os
    import sys
    from .. constants import AU_PER_ANGSTROM

    if not os.path.exists(fname):
        raise Exception("ReadGauEsp file not found: %s"%(fname))
    
    fh = open(fname,"r")
    
    crds=[]
    pts=[]
    esp=[]
    extpts=[]
    extqs=[]
    
    isopolar = 0
    anisopolar = 0
    polar = np.zeros( (3,3) )
    
    atomic_center_line = re.compile(r" +Atomic Center[ 0-9]+ is at +([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6})")
    fit_center_line = re.compile(r" +ESP Fit Center[ 0-9]+ is at +([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6})")
    read_center_line = re.compile(r" +Read-in Center[ 0-9]+ is at +([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6}) *([\-0-9]{1,3}\.[0-9]{6})")
    
    esp_line = re.compile(r"[ 0-9]+ Fit +([\-\.0-9]+)")
    ext_line = re.compile(r" XYZ= *([\-0-9]{1,5}\.[0-9]{4}) *([\-0-9]{1,5}\.[0-9]{4}) *([\-0-9]{1,5}\.[0-9]{4}) Q= *([\-0-9]{1,5}\.[0-9]{4}) A= *([\-0-9]{1,5}\.[0-9]{4}) R= *([\-0-9]{1,5}\.[0-9]{4}) C= *([\-0-9]{1,5}\.[0-9]{4})")
    pline = re.compile(r"   ([xyz]{2}) *([\-0-9]{1,2}\.[0-9]{6}D[\+\-0-9]{3}) *([\-0-9]{1,2}\.[0-9]{6}D[\+\-0-9]{3}) *([\-0-9]{1,2}\.[0-9]{6}D[\+\-0-9]{3})")
    isoline = re.compile(r"   iso *([\-0-9]{1,2}\.[0-9]{6}D[\+\-0-9]{3}) *([\-0-9]{1,2}\.[0-9]{6}D[\+\-0-9]{3}) *([\-0-9]{1,2}\.[0-9]{6}D[\+\-0-9]{3})")
    anisoline = re.compile(r"   aniso *([\-0-9]{1,2}\.[0-9]{6}D[\+\-0-9]{3}) *([\-0-9]{1,2}\.[0-9]{6}D[\+\-0-9]{3}) *([\-0-9]{1,2}\.[0-9]{6}D[\+\-0-9]{3})")

    

    for line in fh:
        m = ext_line.match(line)
        if m is not None:
            extpts.append( [ float( m.group(1) ), float( m.group(2) ), float( m.group(3) ) ] )
            extqs.append( float( m.group(4) ) )
        m = atomic_center_line.match(line)
        if m is not None:
            crds.append( [ float( m.group(1) ), float( m.group(2) ), float( m.group(3) ) ] )
            continue
        m = fit_center_line.match(line)
        if m is not None:
            pts.append( [ float( m.group(1) ), float( m.group(2) ), float( m.group(3) ) ] )
            continue
        m = read_center_line.match(line)
        if m is not None:
            pts.append( [ float( m.group(1) ), float( m.group(2) ), float( m.group(3) ) ] )
            continue
        
        if "Alpha(0;0):" in line:
            line = next(fh)
            
            line = next(fh)
            m = isoline.match(line)
            if m is None:
                raise Exception(f"Expected to read \"iso\" line, but found:\n{line}")
            isopolar = float(m.group(1).replace("D","e"))
            
            line = next(fh)
            m = anisoline.match(line)
            if m is None:
                raise Exception(f"Expected to read \"aniso\" line, but found:\n{line}")
            anisopolar = float(m.group(1).replace("D","e"))
            
            line = next(fh)
            m = pline.match(line)
            if m is None:
                raise Exception(f"Expected to read \"xx\" line, but found:\n{line}")
            x = float(m.group(2).replace("D","e"))
            polar[0,0] = x

            line = next(fh)
            m = pline.match(line)
            if m is None:
                raise Exception(f"Expected to read \"yx\" line, but found:\n{line}")
            x = float(m.group(2).replace("D","e"))
            polar[1,0] = polar[0,1] = x

            line = next(fh)
            m = pline.match(line)
            if m is None:
                raise Exception(f"Expected to read \"yy\" line, but found:\n{line}")
            x = float(m.group(2).replace("D","e"))
            polar[1,1] = x

            line = next(fh)
            m = pline.match(line)
            if m is None:
                raise Exception(f"Expected to read \"zx\" line, but found:\n{line}")
            x = float(m.group(2).replace("D","e"))
            polar[2,0] = polar[0,2] = x

            line = next(fh)
            m = pline.match(line)
            if m is None:
                raise Exception(f"Expected to read \"zy\" line, but found:\n{line}")
            x = float(m.group(2).replace("D","e"))
            polar[1,0] = polar[1,2] = x

            line = next(fh)
            m = pline.match(line)
            if m is None:
                raise Exception(f"Expected to read \"zz\" line, but found:\n{line}")
            x = float(m.group(2).replace("D","e"))
            polar[2,2] = x


        if "  Center     Electric  " in line:
            line = next(fh)
            line = next(fh)
            for line in fh:
                #print(line)
                if "----------" in line:
                    break
                cs = line.strip().split()
                if len(cs) == 2 or len(cs) == 3:
                    if cs[1] != "Atom":
                        esp.append( float(cs[-1]) )

    crds = np.array(crds) * AU_PER_ANGSTROM()
    pts = np.array(pts) * AU_PER_ANGSTROM()
    esp = np.array(esp)
    
    if pts.shape[0] != esp.shape[0]:
        raise Exception("# pts (%i) != # esp values (%i) in %s"%(pts.shape[0],len(esp),fname))


    if len(extpts) > 0:
        extpts = np.array(extpts) * AU_PER_ANGSTROM()
        extqs = np.array(extqs)
    else:
        extpts = None
        extqs = None
    return crds,pts,esp,extpts,extqs,isopolar,anisopolar,polar




def ConvertGOUT2GESP(gout,gesp):
    from . GaussianOutput import GaussianOutput
    crds,pts,esp,extpts,extqs = ReadGaussianEsp(gout)
    g = GaussianOutput(gout)
    atns = g.steps[-1].GetElements()
    
    WriteGESP(atns,crds,pts,esp,extpts,extqs,gesp)
    
    

def ConvertGOUT2GOUT(gout,gesp):
    from . GaussianOutput import GaussianOutput
    crds,pts,esp,extpts,extqs = ReadGaussianEsp(gout)
    g = GaussianOutput(gout)
    s = g.steps[-1]
    atns = s.GetElements()
    
    from .. constants import AU_PER_ANGSTROM
    from .. constants import GetAtomicNumber

    x = 1. / AU_PER_ANGSTROM()
    
    try:
        fh = open(gesp,"w")
    except:
        import sys
        fh = sys.stdout

    fh.write(" Gaussian 09\n")

    fh.write(" Charge = %2i Multiplicity = %i\n"%(s.GetCharge(),s.GetMultiplicity()))
    fh.write("Input orientation:\n\n\n\n\n")
    # "      2          6           0       -6.286100    2.307500   -1.177900"
    #  1234567123456789011234567890121234567890123456123456789012123456789012
    # "      2          1           0       -1.000000    0.000000   -1.000000"
    # "      1          6           0       -6.006100    1.260200   -0.313000"
    #  0123456789012345678901234567890123456789012345678901234567890123456789
    #            1         2         3         4         5         6
    fh.write(" "+"-"*80+"\n")
    for a in range(crds.shape[0]):
        z = GetAtomicNumber(atns[a])
        c = crds[a,:] * x
        fh.write("%7i%11i%12i%16.6f%12.6f%12.6f\n"%(a+1,z,0,c[0],c[1],c[2]))
    fh.write(" "+"-"*80+"\n")

    
    #fh.write("-- Stationary point found.\n")
    for a in range(crds.shape[0]):
        WriteAtomicCenterLine(fh,a+1,crds[a,:])
        #fh.write("Atomic Center                   %13.8f %13.8f %13.8f\n"%(crds[a,0]*x,crds[a,1]*x,crds[a,2]*x))
    #fh.write("ESP Fit Center\n")
    for a in range(len(esp)):
        WriteFitCenterLine(fh,a+1,pts[a,:])
        #fh.write("ESP Fit Center                  %13.8f %13.8f %13.8f\n"%(pts[a,0],pts[a,1],pts[a,2]))
    fh.write("Electrostatic Properties (Atomic Units)\n")
    #for a in range(len(esp)):
        #fh.write("%5i Fit %13.8f\n"%(min(a+1,99999),esp[a]))
    #WriteESPPts(fh,pts)
    WriteESPValues(fh,esp)
    
    

def WriteGESP(atnums,crds,pts,esp,extpts,extqs,gesp):
    from .. constants import AU_PER_ANGSTROM
    from .. constants import GetAtomicNumber
    x = 1. / AU_PER_ANGSTROM()
    
    try:
        fh = open(gesp,"w")
    except:
        import sys
        fh = sys.stdout
    fh.write(" ESP FILE -\n")
    fh.write(" ATOMIC COORDINATES\n")
    for a in range(crds.shape[0]):
        z = GetAtomicNumber(atnums[a])
        fh.write("%2s %13.8f %13.8f %13.8f %13.8f\n"%(atnums[a],crds[a,0]*x,crds[a,1]*x,crds[a,2]*x,0.0))
    fh.write("DIPOLE\n")
    fh.write("0. 0. 0. 0. 0. 0. 0. 0.\n")
    fh.write("TRACELESS QUADRUPOLE\n")
    fh.write("0. 0. 0. 0. 0. 0. 0. 0.\n")
    fh.write("0. 0. 0. 0. 0. 0. 0. 0.\n")
    fh.write("ESP VALUES\n")
    fh.write("0. 0. 0. 0. 0. 0. 0. 0. %i\n"%(len(esp)))
    for a in range(len(esp)):
        fh.write("%13.8f %13.8f %13.8f %13.8f\n"%(esp[a],pts[a,0]*x,pts[a,1]*x,pts[a,2]*x))
    fh.close()
    
