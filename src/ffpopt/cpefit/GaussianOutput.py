#!/usr/bin/env python3

def IsAnInt(s):
    flag = True
    try:
        i = int(s)
    except ValueError:
        flag = False
    return flag


def GetSymNum(pg):
    num = None
    if pg is None:
        return num
    
    pg = pg.strip().upper()
    
    if len(pg) < 2:
        if pg == "T":
            num = 24
    else:
        if pg[:2] == "CI":
            num = 1
        elif pg[:2] == "CS":
            num = 1
        elif pg[:2] == "C*":
            num = 1
        elif pg[0] == "C" and IsAnInt(pg[1:]):
            num = int(pg[1:])
        elif pg[:2] == "D*":
            num = 2
        elif pg[0] == "D" and IsAnInt(pg[1:]):
            num = 2*int(pg[1:])
        elif pg[0] == "T":
            num = 12
        elif pg[0] == "S" and IsAnInt(pg[1:]):
            num = pg[1:] // 2
        elif pg[:2] == "OH":
            num = 24
        elif pg[:2] == "IH":
            num = 60
    return num


class GaussianArchive(object):
    
    def __init__(self,arcstr):
        self.arcstr = arcstr
        sarc = arcstr.split('\\')
        self.arc = []
        isec = 0
        ssec = 0
        for s in sarc:
            if len(s) == 0:
                isec += 1
                ssec = 0
                continue
            if len(self.arc) == isec:
                self.arc.append( [] )
            elif isec > len(self.arc):
                break
            if len(s) > 2 and isec > 1:
                s = s.replace(","," ").replace("="," ")
            self.arc[isec].append(s)

            
    def GetTheory(self):
        x = None
        if len(self.arc) > 0:
            if len(self.arc[0]) > 4:
                x = self.arc[0][4]
        return x

    
    def GetBasis(self):
        x = None
        if len(self.arc) > 0:
            if len(self.arc[0]) > 5:
                x = self.arc[0][5]
        return x

    
    def GetRoute(self,as_input=False):
        x = None
        if len(self.arc) > 1:
            if len(self.arc[1]) > 0:
                x = self.arc[1][0]
        if as_input:
            cs = x.split("#")[1:]
            x = "#" + "\n#".join(cs)
        return x

    
    def GetInput(self,singlepoint=False,crd=None):
        if singlepoint:
            s = "#P %s/%s "%(self.GetTheory(),self.GetBasis())
            s += "SP SCF(Tight) INTEGRAL(GRID=ULTRAFINE)\n\n"
        else:
            s = self.GetRoute(as_input=True) + "\n\n"
        s += "%s\n\n"%( self.GetTitle() )
        s += "%i %i\n"%(self.GetCharge(),self.GetMultiplicity())
        ele = self.GetElements()
        if crd is None:
            crd = self.GetCrd()    
        for e,c in zip(ele,crd):
            s += " %2s   %16.10f %16.10f %16.10f\n"%(e,c[0],c[1],c[2])
        s += "\n\n"
        return s


    def GetTitle(self):
        x = None
        if len(self.arc) > 2:
            if len(self.arc[2]) > 0:
                x = self.arc[2][0]
        return x

    
    def GetCharge(self):
        q = None
        if len(self.arc) > 3:
            if len(self.arc[3]) > 0:
                q,m = [int(u) for u in self.arc[3][0].split()]
        return q

    
    def GetMultiplicity(self):
        m = None
        if len(self.arc) > 3:
            if len(self.arc[3]) > 0:
                q,m = [int(u) for u in self.arc[3][0].split()]
        return m


    def GetElements(self):
        a = None
        if len(self.arc) > 3:
            if len(self.arc[3]) > 1:
                nat = len(self.arc[3])-1
                a = [ line.split()[0] for line in self.arc[3][1:1+nat] ]
        return a

    
    def GetCrd(self):
        import numpy as np
        crd = None
        if len(self.arc) > 3:
            if len(self.arc[3]) > 1:
                nat = len(self.arc[3])-1
                crd = np.zeros( (nat,3) )
                for iat in range(nat):
                    cs = self.arc[3][1+iat].split()
                    if len(cs) == 4:
                        for k in range(3):
                            crd[iat,k] = float(cs[1+k])
                    elif len(cs) == 5:
                        for k in range(3):
                            crd[iat,k] = float(cs[2+k])
                    else:
                        raise Exception(("Error processing "
                                         f"{self.arc[3][1+iat]} "
                                         "wrong number of columns"))
        return crd


    def GetHF(self):
        a = None
        if len(self.arc) > 4:
            for e in self.arc[4]:
                if e[:2] == "HF":
                    cs = e.split()
                    a = float(cs[1])
        return a


    def GetPolar(self):
        import numpy as np
        a = None
        if len(self.arc) > 4:
            for e in self.arc[4]:
                if e[:5].upper() == "POLAR":
                    s = [ float(x) for x in e.split()[1:] ]
                    a = np.zeros( (3,3) )
                    k = 0
                    for i in range(3):
                        for j in range(i+1):
                            a[i,j] = s[k]
                            a[j,i] = s[k]
                            k += 1
        return a

    
    def GetDipole(self):
        import numpy as np
        a = None
        if len(self.arc) > 4:
            for e in self.arc[4]:
                if e[:6].upper() == "DIPOLE":
                    a = np.array( [ float(x) for x in e.split()[1:] ] )
        return a


    def GetPointGroup(self):
        import numpy as np
        a = None
        if len(self.arc) > 4:
            for e in self.arc[4]:
                if e[:2] == "PG":
                    a = e.split()[1]
        return a

    
    def GetHessian(self):
        import numpy as np
        a = None
        ele = self.GetElements()
        if ele is None:
            return a
        nat = len(ele)
        n3 = 3*nat
        for iP in range(len(self.arc)):
            for p in self.arc[iP]:
                if p[:5].upper() == "NIMAG" and iP+1 < len(self.arc):
                    if len(self.arc[iP+1]) > 0:
                        h = [ float(e) for e in self.arc[iP+1][0].split() ]
                        a = np.zeros( (n3,n3) )
                        k=0
                        for i in range(n3):
                            for j in range(i+1):
                                a[i,j]=h[k]
                                a[j,i]=h[k]
                                k += 1
                        return a
        return a


    def GetSymNum(self):
        return GetSymNum( self.GetPointGroup() )


    
    
    def __str__(self):
        o = ""
        s = self
        o += f"title:  {s.GetTitle()}\n"
        o += f"route:  {s.GetRoute()}\n"
        o += f"charge: {s.GetCharge()}\n"
        o += f"ptgrp:  {s.GetPointGroup()}\n"
        o += f"symnum: {s.GetSymNum()}\n"
        o += f"multiplicity: {s.GetMultiplicity()}\n"
        o += f"theory: {s.GetTheory()}\n"
        o += f"basis:  {s.GetBasis()}\n"
        o += f"HF:     {s.GetHF()}\n"
        ele = s.GetElements()
        crd = s.GetCrd()
        o += f"xyz:\n"
        for e,c in zip(ele,crd):
            o += f"  {e:2} {c[0]:12.8f} {c[1]:12.8f} {c[2]:12.8f}\n"
        x = s.GetDipole()
        if x is not None:
            o += f"dipole: {x[0]:12.8f} {x[1]:12.8f} {x[2]:12.8f}\n"
        else:
            o += f"dipole: {x}\n"
        x = s.GetPolar()
        if x is not None:
            o += f"polar:\n"
            for i in range(3):
                o += f"   {x[i][0]:12.4f} {x[i][1]:12.4f} {x[i][2]:12.4f}\n"
        else:
            o += f"polar: {x}\n"
        x = s.GetHessian()
        if x is not None:
            o += f"hessian:\n"
            n3 = x.shape[0]
            for i in range(n3):
                for j in range(n3):
                    o += f" {x[i][j]:11.2e}"
                o += "\n"
        else:
            o += f"hessian: {x}\n"
        return o



    
class GaussianOutput(object):
    def __init__(self,fname):
        from pathlib import Path
        self.filename = Path(fname)
        self.steps = []
        fh = open(self.filename,"r")
        curstep = ""
        for line in fh:
            line = line.strip()
            if line[:4] == "1\\1\\":
                curstep = line
                for line in fh:
                    line = line.strip()
                    curstep += line
                    if "\\\\@" in curstep:
                        self.steps.append( GaussianArchive(curstep) )
                        curstep = ""
                        break

    def FirstStepWithHessian(self):
        s = None
        for step in self.steps:
            if step.GetHessian() is not None:
                s = step
                break
        return s
    
                    
    def __str__(self):
        o = ""
        o += f"filename: {self.filename}\n"
        for istep,s in enumerate(self.steps):
            o += f"istep:  {istep}\n"
            o += str(s)

        return o


    
if __name__ == "__main__":
    import sys
    
    g = GaussianOutput(sys.argv[1])
    s = g.FirstStepWithHessian()
    
    print(s)
    
