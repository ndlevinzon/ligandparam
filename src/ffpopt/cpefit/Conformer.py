#!/usr/bin/env python3



class SurfaceParameters(object):
    
    def __init__(self,sclfs,ptdensity):
        import numpy as np
        self.sclfs = np.array( sclfs, dtype=np.float64 )
        self.ptdensity = ptdensity

        
    def GetSurface(self,crds,atnums,thickness=None):
        if len(self.sclfs) == 1:
            return self.GetCosmoSurface(crds,atnums,thickness)
        else:
            return self.GetRespSurface(crds,atnums,thickness)

        
    def GetCosmoSurface(self,crds,atnums,thickness=None):
        import numpy as np
        from .. scosmo import CosmoSurface
        from .. constants import GetUffRadius, AU_PER_ANGSTROM
        from .. scosmo import GetLebedevDegreeMatchingDensity

        surfs = []
        crds = np.array(crds)
        nat = crds.shape[0]
        charges = [0]*nat
        origradii = np.array([GetUffRadius(z) for z in atnums])
        for sclf in self.sclfs:
            radii = sclf * origradii
            dens = self.ptdensity
            nsurfs = [ GetLebedevDegreeMatchingDensity(r/AU_PER_ANGSTROM(),dens)
                       for r in radii ]
            #print(radii,nsurfs)
            if thickness is None:
                switch_thickness = np.sqrt(14./np.mean(nsurfs))
            else:
                switch_thickness = thickness
            surfs.append( CosmoSurface(crds,charges,radii,nsurfs,switch_thickness) )
        return surfs

    
    def GetRespSurface(self,crds,atnums,thickness=None):
        import numpy as np
        from .. scosmo import CosmoSurface
        from .. constants import GetRespRadius, AU_PER_ANGSTROM
        from .. scosmo import GetLebedevDegreeMatchingDensity

        surfs = []
        crds = np.array(crds)
        nat = crds.shape[0]
        charges = [0]*nat
        origradii = np.array([GetRespRadius(z) for z in atnums])
        for sclf in self.sclfs:
            radii = sclf * origradii
            dens = self.ptdensity
            nsurfs = [ GetLebedevDegreeMatchingDensity(r/AU_PER_ANGSTROM(),dens)
                       for r in radii ]
            #print(radii,nsurfs)
            #if thickness is None:
            #    switch_thickness = np.sqrt(14./np.mean(nsurfs))
            #else:
            switch_thickness = 0
            surfs.append( CosmoSurface(crds,charges,radii,nsurfs,switch_thickness) )
        return surfs

    


class MultiSurface(object):
    def __init__(self,crds,atnums,surfpars,thickness=None):
        self.surfs = surfpars.GetSurface(crds,atnums,thickness=thickness)
        self.elems = []
        for s in self.surfs:
            self.elems.extend( s.elems )
        self.nelem = len(self.elems)
        self.atoms = self.surfs[0].atoms

        
    def CptPointInteractionMatrix(self):
        import numpy as np
        from scipy.special import erf
        
        nelem = self.nelem
        nat = len(self.atoms)
        A = np.zeros( (nat,nelem) )
        for b in range(nelem):
            pb = self.elems[b]
            #zb = pb.zeta
            for a in range(nat):
                pa = self.atoms[a]
                #zab = zb
                #sqz = np.sqrt(zab)
                #r = np.linalg.norm( pa.crd-pb.crd )
                #x = erf(sqz*r)/r
                A[a,b] = 1./np.linalg.norm( pa.crd-pb.crd )
        return A

    
    def CptGaussianInteractionMatrix(self,atomzetas):
        import numpy as np
        from scipy.special import erf
        
        nelem = self.nelem
        nat = len(self.atoms)

        nzeta = len(atomzetas)
        if nzeta != nat:
            raise Exception(f"The number of exponents {len(atomzetas)} "
                            +f"must match the number of atoms {nat}")
        
        A = np.zeros( (nat,nelem) )
        for b in range(nelem):
            pb = self.elems[b]
            #zb = pb.zeta
            for a in range(nat):
                pa = self.atoms[a]
                za = atomzetas[a]
                #zab = za*zb/(za+zb)
                sqz = np.sqrt(za)
                r = np.linalg.norm( pa.crd-pb.crd )
                x = erf(sqz*r)/r
                A[a,b] = x
        return A

    
    def CptGaussianInteractionMatrixAndGrd(self,atomzetas):
        import numpy as np
        from scipy.special import erf

        SQRTPI = np.sqrt(np.pi)
        
        nelem = self.nelem
        nat = len(self.atoms)

        nzeta = len(atomzetas)
        if nzeta != nat:
            raise Exception(f"The number of exponents {len(atomzetas)} "
                            +f"must match the number of atoms {nat}")
        
        A    = np.zeros( (nat,nelem) )
        dAdz = np.zeros( (nat,nelem) )
        for b in range(nelem):
            pb = self.elems[b]
            #zb = pb.zeta
            for a in range(nat):
                pa = self.atoms[a]
                za = atomzetas[a]
                #zab = za*zb/(za+zb)
                sqz = np.sqrt(za)
                r = np.linalg.norm( pa.crd-pb.crd )
                x = erf(sqz*r)/r
                A[a,b] = x
                #num = np.exp(-(sqz*r)**2) * zb * sqz
                #den = SQRTPI * za * (za+zb)
                dAdz[a,b] = np.exp(-(sqz*r)**2) / (SQRTPI*sqz)
        return A,dAdz

    
    def CptESPMatricesAndGrd(self,atomzetas):
        import numpy as np
        from scipy.special import erf

        SQRTPI = np.sqrt(np.pi)
        
        nelem = self.nelem
        nat = len(self.atoms)

        nzeta = len(atomzetas)
        if nzeta != nat:
            raise Exception(f"The number of exponents {len(atomzetas)} "
                            +f"must match the number of atoms {nat}")

        B    = np.zeros( (nat,nelem) )
        A    = np.zeros( (nat,nelem) )
        dAdz = np.zeros( (nat,nelem) )
        for b in range(nelem):
            pb = self.elems[b]
            #zb = pb.zeta
            for a in range(nat):
                pa = self.atoms[a]
                za = atomzetas[a]
                #zab = za*zb/(za+zb)
                sqz = np.sqrt(za)
                r = np.linalg.norm( pa.crd-pb.crd )
                x = erf(sqz*r)/r
                A[a,b] = x
                B[a,b] = 1./r
                #num = np.exp(-(sqz*r)**2) * zb * sqz
                #den = SQRTPI * za * (za+zb)
                dAdz[a,b] = np.exp(-(sqz*r)**2) / (SQRTPI*sqz)
        return B,A,dAdz



class SurfaceResponseDelta(object):
    def __init__(self,conf,pertid,extvals,aiopts,onlypos=False):
        
        import numpy as np
        import copy
        
        if conf.extpts is None:
            conf.extpts = np.array( [ e.crd for e in conf.extsurf.elems ] )

        self.pertid = str(pertid)
        self.parent = copy.deepcopy(conf)
        self.parent.desps = None
        self.extvals_pos = np.array(extvals,copy=True)
        self.extvals_neg = -self.extvals_pos

        
        self.parent.pertid = pertid + "pos"
        self.parent.extvals = self.extvals_pos
        
        self.parent.espvals = None
        self.parent.RunAbInitioEspIfNeeded(aiopts)
        self.espvals_pos = np.array(self.parent.espvals,copy=True)

        self.espvals_neg = None
        if not onlypos:
            self.parent.pertid = pertid + "neg"
            self.parent.extvals = self.extvals_neg
            self.parent.espvals = None
            self.parent.RunAbInitioEspIfNeeded(aiopts)
            self.espvals_neg = np.array(self.parent.espvals,copy=True)
        
        self.parent.espvals = None

        self.despvals = None
        if not onlypos:
            self.despvals = self.espvals_pos - self.espvals_neg


    def WriteExtPertJMOL(self):
        import numpy as np
        from .. constants import AU_PER_ANGSTROM
        
        self.parent.pertid = self.pertid + "pos"
        base = self.parent.GetBasename()
        
        s = self.parent.extsurf.surfs[0]
        nelem = len(s.elems)

        if nelem != len(self.extvals_pos):
            raise Exception(f"nelem={nelem} but nvals={len(self.extvals_pos)}")
        
        fh = open( "%s.xyz"%(base), "w" )
        fh.write("%i\ntitle\n"%(nelem))
        for i in range(nelem):
            c = np.array(s.elems[i].crd,copy=True) / AU_PER_ANGSTROM()
            q = self.extvals_pos[i]
            fh.write("H  %12.7f %12.7f %12.7f %20.13f\n"%(c[0],c[1],c[2],q))
        fh.close()
        
        # fh = open( "%s.prop"%(base), "w" )
        # for i in range(nelem):
        #     fh.write("%20.13f\n"%(self.extvals_pos[i]))
        # fh.close()

        maxval = np.amax(abs(self.extvals_pos)) * 0.1
        
        fh = open( "%s.jmol"%(base), "w" )
        fh.write("""
load "%s.xyz"
color structure
select all
wireframe off
color atoms PROPERTY partialCharge ABSOLUTE -%.3e %.3e
        """%(base,maxval,maxval))
        fh.close()
        
        
    
    
class Conformer(object):
    def __init__(self,
                 name,pertid,crds,atnums,charge,
                 extpts,extvals,esppts,espvals,
                 esppar,extpar):
        
        import numpy as np


        if extpar is not None:
            nlayer = extpar.sclfs.shape[0]
            if nlayer != 1:
                raise Exception("External perturbation surface "
                                +"parameters (extpar) is expected "
                                +f"to contain 1 layer, but {nlayer} "
                                +"layers were provided")
        
        self.name = name
        
        self.pertid = pertid
        
        self.crds = np.array(crds,copy=True)
        
        self.atnums = [x for x in atnums]
        
        self.charge = charge
        
        if extpts is not None:
            self.extpts = np.array(extpts,copy=True)
        else:
            self.extpts = None
            
        if extvals is not None:
            self.extvals = np.array(extvals,copy=True)
        else:
            self.extvals = None
            
        if esppts is not None:
            self.esppts = np.array(esppts,copy=True)
        else:
            self.esppts = None
            
        if espvals is not None:
            self.espvals = np.array(espvals,copy=True)
        else:
            self.espvals = None

        if extpar is None:
            extpar = esppar

        #self.esppar = esppar
        #self.extpar = extpar
            
        self.SetEspSurface(esppar)
        self.SetExtSurface(extpar)

        self.desps = None


        
    def GetBasename(self):
        return "%s_%s"%(self.name,self.pertid)



    def MakeCosmoAndSurfaceHarmonics(self,lmax,qatoms,aiopts,onlypos=False):
        #theory="hf/6-31g*",nproc=4,
        #mem="2000mb",mult=1):
        self.desps = None
        
        avgqperarea = self.MakeCosmoHarmonic\
            (qatoms,aiopts,onlypos=onlypos) #theory=theory,nproc=nproc,mem=mem,mult=mult)
        
        self.MakeSurfaceHarmonics\
            (lmax,aiopts,avgqperarea)

        
        
    def MakeCosmoHarmonic(self,qatoms,aiopts,onlypos=False):
        #theory="hf/6-31g*",nproc=4,
        #mem="2000mb",mult=1):
        import numpy as np
        
        qext = self.GetExtSurfaceCosmoResponse(qatoms)
        if self.desps is None:
            self.desps = []

        sa = self.GetExtSurfaceArea()
        avgqperarea = np.sum(np.abs(qext)) / sa

        # print("sa=",sa)
        # print("qext=",qext)
        # print("sumabs=",np.sum(np.abs(qext)))
        # print("avgq=",avgqperarea)
        # exit(0)
        
        pertid = "%sH%03i"%(self.pertid,0)
        desp = SurfaceResponseDelta\
            (self,pertid,qext,
             aiopts,onlypos=onlypos)
        #theory=theory,nproc=nproc,
        #     mem=mem,mult=mult)
        self.desps.append( desp )
        return avgqperarea
    

    
    def MakeSurfaceHarmonics(self,lmax,aiopts,avgqperarea=1.):
        #theory="hf/6-31g*",nproc=4,
        #mem="2000mb",mult=1):
        
        if len(self.extsurf.surfs) > 1:
            raise Exception("Harmonics can only be made on "
                            +"a single-layered surface")
        
        U = self.extsurf.surfs[0].CptSurfaceHarmonics(lmax,avgqperarea=avgqperarea)
        nmax = (lmax+1)**2
        if self.desps is None:
            self.desps = []
        for i in range(1,nmax):
            pertid = "%sH%03i"%(self.pertid,i)
            desp = SurfaceResponseDelta\
                (self,pertid,U[:,i],
                 aiopts)
            #theory=theory,nproc=nproc,
            #mem=mem,mult=mult)
            self.desps.append( desp )


    
    def SetEspSurface(self,surface):
        import numpy as np
        if surface is None:
            self.espsurf = None
            self.espwts = None
            self.Bmat = None
            return
        self.espsurf = MultiSurface(self.crds,self.atnums,surface)
        if self.espsurf is not None and self.esppts is not None:
            TOL = np.sqrt( 3 * 0.00001**2 )
            nelem = len(self.espsurf.elems)
            npts = self.esppts.shape[0]
            npot = self.espvals.shape[0]
            if nelem != npts:
                raise Exception(f"EspSurface has {nelem} elements but read {npts} points")
            if npot != npts:
                raise Exception(f"Have {npts} ESP points, but {npot} ESP values")
            for i in range(nelem):
                dr = np.linalg.norm(self.espsurf.elems[i].crd - self.esppts[i,:])
                if dr > TOL:
                    c = np.array(self.extsurf.elems[i].crd,copy=True)
                    pt = np.array(self.esppts[i,:],copy=True)
                    c /= AU_PER_ANGSTROM()
                    pt /= AU_PER_ANGSTROM()
                    print("%18.10f %18.10f %18.10f   %18.10f %18.10f %18.10f"%\
                          (c[0],c[1],c[2],pt[0],pt[1],pt[2]))
                    raise Exception(f"EspSurface position {i} mismatch {dr}")
            self.esppts = numpy.array( [ e.crd for e in self.espsurf.elems ] )
                                    
                 
        self.espwts = np.array([e.switchwt * e.quadwt * e.radwt for e in self.espsurf.elems])
        self.Bmat   = self.espsurf.CptPointInteractionMatrix()

                
    def SetExtSurface(self,surface):
        import numpy as np
        from .. constants import AU_PER_ANGSTROM
        if surface is None:
            self.extsurf = None
            return
        self.extsurf = MultiSurface(self.crds,self.atnums,surface,thickness=0)
        if self.extsurf is not None and self.extpts is not None:
            TOL = np.sqrt( 3 * 0.00005**2 ) * AU_PER_ANGSTROM()
            nelem = len(self.extsurf.elems)
            npts = self.extpts.shape[0]
            npot = self.extvals.shape[0]
            if nelem != npts:
                raise Exception(f"ExtSurface has {nelem} elements but read {npts} points")
            if npot != npts:
                raise Exception(f"Have {npts} EXT points, but {npot} EXT values")
            for i in range(nelem):
                dr = np.linalg.norm(self.extsurf.elems[i].crd - self.extpts[i,:])
                if dr > TOL:
                    c = np.array(self.extsurf.elems[i].crd,copy=True)
                    pt = np.array(self.extpts[i,:],copy=True)
                    c /= AU_PER_ANGSTROM()
                    pt /= AU_PER_ANGSTROM()
                    print("%18.10f %18.10f %18.10f   %18.10f %18.10f %18.10f"%\
                          (c[0],c[1],c[2],pt[0],pt[1],pt[2]))
                    raise Exception(f"ExtSurface position {i} mismatch {dr}")
            self.extpts = numpy.array( [ e.crd for e in self.extsurf.elems ] )


                
    def SetExtPts(self,pts,vals):
        import numpy as np
        from .. constants import AU_PER_ANGSTROM
        if self.extsurf is None:
            self.extpts  = np.array(pts)
            self.extvals = np.array(vals)
        else:
            TOL = np.sqrt( 3 * 0.00005**2 ) * AU_PER_ANGSTROM()
            pts = np.array(pts)
            vals = np.array(vals)
            npts = pts.shape[0]
            nvals = vals.shape[0]
            nelem = len(self.extsurf.elems)
            if npts != nelem:
                raise Exception(f"ExtSurface has {nelem} elements received {npts}")
            if npts != nvals:
                raise Exception(f"Have {npts} EXT pts but passed {nvals} EXT values")
            for i in range(nelem):
                c = np.array(self.extsurf.elems[i].crd,copy=True)
                pt = np.array(pts[i,:],copy=True)
                dr = np.linalg.norm(c - pt)
                if dr > TOL:
                    c /= AU_PER_ANGSTROM()
                    pt /= AU_PER_ANGSTROM()
                    print("%18.10f %18.10f %18.10f   %18.10f %18.10f %18.10f"%\
                          (c[0],c[1],c[2],pt[0],pt[1],pt[2]))
                    raise Exception(f"ExtSurface position {i} mismatch {dr}")
            #self.extpts = pts
            for i in range(nelem):
                self.extsurf.elems[i].q = vals[i]
            self.extvals = vals

            
            
    def SetEspPts(self,pts,vals):
        import numpy as np
        if self.espsurf is None:
            self.esppts  = np.array(pts)
            self.espvals = np.array(vals)
            nelem = len(vals)
            nat = self.crds.shape[0]
            self.espwts = np.array([1]*nelem)
            self.Bmat = np.zeros( (nat,nelem) )
            for b in range(nelem):
                for a in range(nat):
                    ca = self.crds[a,:]
                    cb = self.esppts[b,:]
                    r = np.linalg.norm( ca-cb )
                    self.Bmat[a,b] = 1/r
        else:
            TOL = np.sqrt( 3 * 0.00001**2 )
            pts = np.array(pts)
            vals = np.array(vals)
            npts = pts.shape[0]
            nvals = vals.shape[0]
            nelem = len(self.espsurf.elems)
            if npts != nelem:
                raise Exception(f"EspSurface has {nelem} elements received {npts}")
            if npts != nvals:
                raise Exception(f"Have {npts} ESP pts but passed {nvals} ESP values")
            for i in range(nelem):
                dr = np.linalg.norm(self.espsurf.elems[i].crd - pts[i,:])
                if dr > TOL:
                    c = np.array(self.espsurf.elems[i].crd,copy=True)
                    pt = np.array(pts[i,:],copy=True)
                    c /= AU_PER_ANGSTROM()
                    pt /= AU_PER_ANGSTROM()
                    print("%18.10f %18.10f %18.10f   %18.10f %18.10f %18.10f"%\
                          (c[0],c[1],c[2],pt[0],pt[1],pt[2]))
                    raise Exception(f"EspSurface position {i} mismatch {dr}")
            #self.esppts  = pts
            self.espvals = vals


    def RunAbInitioEspIfNeeded(self,aiopts):
        from pathlib import Path
        pname = Path(aiopts.program.split()[-1]).name
        
        if "psi4" in pname:
            self.RunPsi4EspIfNeeded\
                (program=aiopts.program,
                 theory=aiopts.theory,
                 nproc=aiopts.nproc,
                 mem=aiopts.mem,
                 mult=aiopts.mult)
        elif "quick" in pname:
            self.RunQuickEspIfNeeded\
                (program=aiopts.program,
                 theory=aiopts.theory,
                 nproc=aiopts.nproc,
                 mem=aiopts.mem,
                 mult=aiopts.mult)
        else:
            self.RunGaussianEspIfNeeded\
                (program=aiopts.program,
                 theory=aiopts.theory,
                 nproc=aiopts.nproc,
                 mem=aiopts.mem,
                 mult=aiopts.mult)
            
        
    def WriteGaussianEsp\
        (self,
         theory="hf/6-31g*",
         nproc=4,
         mem="2000mb",
         mult=1):
        
        from . GaussianEsp import WriteGaussianEsp
        import numpy as np
        from pathlib import Path
        
        fname = self.GetBasename() + ".inp"
        pname = Path(fname)
        if self.esppts is None:
            if self.espsurf is not None:
                self.esppts = np.array([ e.crd for e in self.espsurf.elems ])
            else:
                raise Exception(f"Cannot write {fname} because there are no ESP points")

        print("Writing %s"%(fname))

        #print("extpts=",self.extpts)
        #print("extvals=",self.extvals)
        
        WriteGaussianEsp\
            ( fname,
              self.crds,
              self.atnums,
              self.esppts,
              nproc, mem, theory, self.charge, mult,
              self.extpts,self.extvals )

        
        
    def WriteGaussianPolar\
        (self,
         theory="hf/6-31g*",
         nproc=4,
         mem="2000mb",
         mult=1):
        
        from . GaussianEsp import WriteGaussianPolar
        import numpy as np
        from pathlib import Path
        
        fname = self.GetBasename() + ".polar.inp"
        pname = Path(fname)
        print("Writing %s"%(fname))

        #print("extpts=",self.extpts)
        #print("extvals=",self.extvals)
        
        WriteGaussianPolar\
            ( fname,
              self.crds,
              self.atnums,
              self.esppts,
              nproc, mem, theory, self.charge, mult,
              self.extpts,self.extvals )

    

        
    def RunPsi4EspIfNeeded\
        (self,
         program="psi4",
         theory="hf/6-31g*",
         nproc=4,
         mem="2000mb",
         mult=1):

        from . Psi4Esp import CalcPsi4Esp
        import numpy as np
        from pathlib import Path
        import subprocess as subp
        
        oname = self.GetBasename() + ".log"
        
        needs_output = True
        if Path(oname).is_file():
            needs_output = False

        if needs_output:
            print(f"File not found: {oname}")

            if self.esppts is None:
                if self.espsurf is not None:
                    self.esppts = np.array([ e.crd for e in self.espsurf.elems ])
                else:
                    raise Exception(f"Cannot write {fname} because there are no ESP points")
                        
            CalcPsi4Esp\
                ( str(oname),
                  self.crds,
                  self.atnums,
                  self.esppts,
                  program,nproc, mem, theory, self.charge, mult,
                  self.extpts,self.extvals )

        self.SetSurfaceValuesFromGaussianEsp(oname)
        #print("self.espvals=",self.espvals)


    def RunQuickEspIfNeeded\
        (self,
         program="quick",
         theory="hf/6-31g*",
         nproc=4,
         mem="2000mb",
         mult=1):

        from . QuickEsp import CalcQuickEsp
        import numpy as np
        from pathlib import Path
        import subprocess as subp
        
        oname = self.GetBasename() + ".log"
        
        needs_output = True
        if Path(oname).is_file():
            needs_output = False

        if needs_output:
            print(f"File not found: {oname}")

            if self.esppts is None:
                if self.espsurf is not None:
                    self.esppts = np.array([ e.crd for e in self.espsurf.elems ])
                else:
                    raise Exception(f"Cannot write {fname} because there are no ESP points")
                        
            CalcQuickEsp\
                ( str(oname),
                  self.crds,
                  self.atnums,
                  self.esppts,
                  program,nproc, mem, theory, self.charge, mult,
                  self.extpts,self.extvals )

        self.SetSurfaceValuesFromGaussianEsp(oname)
        #print("self.espvals=",self.espvals)



        
    def RunGaussianEspIfNeeded\
        (self,
         program="g16",
         theory="hf/6-31g*",
         nproc=4,
         mem="2000mb",
         mult=1):

        
        from . GaussianEsp import ReadGaussianEsp
        from . GaussianEsp import WriteGaussianEsp
        import numpy as np
        from pathlib import Path
        import subprocess as subp
        
        fname = self.GetBasename() + ".inp"
        oname = self.GetBasename() + ".log"
        pname = Path(fname)
        
        needs_output = True
        if pname.is_file():
            if Path(oname).is_file():
                needs_output = False

        if needs_output:
            print(f"File not found: {oname}")
            self.WriteGaussianEsp(theory,nproc,mem,mult)
            
            print(f"Running: {program} < {fname} > {oname}")
            subp.run(f"{program} < {fname} > {oname}",shell=True,check=True)
            
        self.SetSurfaceValuesFromGaussianEsp(oname)



    def RunGaussianPolarIfNeeded\
        (self,
         program="g16",
         theory="hf/6-31g*",
         nproc=4,
         mem="2000mb",
         mult=1):

        
        from . GaussianEsp import ReadGaussianPolar
        import numpy as np
        from pathlib import Path
        import subprocess as subp
        
        fname = self.GetBasename() + ".polar.inp"
        oname = self.GetBasename() + ".polar.log"
        pname = Path(fname)
        
        needs_output = True
        if pname.is_file():
            if Path(oname).is_file():
                needs_output = False

        if needs_output:
            print(f"File not found: {oname}")
            self.WriteGaussianPolar(theory,nproc,mem,mult)
            
            print(f"Running: {program} < {fname} > {oname}")
            subp.run(f"{program} < {fname} > {oname}",shell=True,check=True)

        crds,pts,esp,extpts,extqs,isopolar,anisopolar,polar = \
            ReadGaussianPolar(oname)
        return crds,pts,esp,extpts,extqs,isopolar,anisopolar,polar
        

        

    def SetSurfaceValuesFromGaussianEsp(self,oname):
        from . GaussianEsp import ReadGaussianEsp
        from pathlib import Path
        if not Path(oname).is_file():
            raise Exception(f"File not found {oname}")
        
        print(f"Reading {oname}")
        crds,esppts,espvals,extpts,extvals = ReadGaussianEsp(oname)

        #print("esppts=",esppts)
        #print("espvals=",espvals)
        if esppts is not None and espvals is not None:
            self.SetEspPts(esppts,espvals)

        #print("extpts=",extpts)
        #print("extvals=",extvals)
        if extpts is not None and extvals is not None:
            self.SetExtPts(extpts,extvals)
        

    def SolveInterCPE(self,bvec,hardness,zetascl,field=None):
        import numpy as np
        from scipy.special import erf

        SQRTPI = np.sqrt(np.pi)
        SQRT2 = np.sqrt(2)
        
        nat = self.crds.shape[0]
        
        zs     = zetascl * hardness**2 * np.pi * 0.5
        dzdscl =           hardness**2 * np.pi * 0.5
        dzdh   = zetascl * hardness    * np.pi

        
        gs = np.zeros( (2*nat+1,) )
        dvec = np.array( [1.]*nat )

        A = np.zeros( (nat,nat) )
        dAdi = np.zeros( (nat,nat) )
        for i in range(nat):
            zi = zs[i]
            for j in range(i):
                zj = zs[j]
                r = np.linalg.norm( self.crds[i,:]-self.crds[j,:] )

                zij = zi*zj/(zi+zj)
                sqz = np.sqrt(zij)
                x = erf(sqz*r)/r
                A[i,j] = x
                A[j,i] = x
                
                dsqz = (0.5/sqz) * (1/(zi+zj))**2
                dxdz = dsqz * 2 * np.exp(-(sqz*r)**2) / SQRTPI
                dAdi[i,j] = zj**2 * dxdz
                dAdi[j,i] = zi**2 * dxdz
                #dAdi[i,j] = (np.exp(-(sqz*r)**2)/(SQRTPI*sqz)) * (zj/(zi+zj))**2
                #dAdi[j,i] = (np.exp(-(sqz*r)**2)/(SQRTPI*sqz)) * (zi/(zi+zj))**2
            A[i,i] = hardness[i]
            

        mybvec = np.array(bvec)
        if field is not None:
            for i in range(nat):
                mybvec[i] -= np.dot( self.crds[i,:], field )

        
                
        Ainv = np.linalg.inv(A)
        AinvB = Ainv @ mybvec
        AinvD = Ainv @ dvec
        mu = np.dot(dvec,AinvB) / np.dot(dvec,AinvD)
        dq = mu*AinvD - AinvB


        dqdh = np.zeros( (nat,nat) )
        dA = np.zeros( (nat,nat) )
        for i in range(nat):
            dA[:,:] = 0.
            dA[i,:] = dAdi[i,:] * dzdh[i]
            dA[:,i] = dAdi[i,:] * dzdh[i]
            dA[i,i] = 1.
            dAinvdh = -np.dot(Ainv,np.dot(dA,Ainv))
            dAinvBdh = dAinvdh @ mybvec
            dAinvDdh = dAinvdh @ dvec
            dmudh = np.dot(dvec,dAinvBdh) / np.dot(dvec,AinvD) \
                - (np.dot(dvec,AinvB) / np.dot(dvec,AinvD)**2) * np.dot(dvec,dAinvDdh)
            dqdh[:,i] = dmudh*AinvD + mu*dAinvDdh - dAinvBdh

            
        dqdscl = np.zeros( (nat,) )
        for i in range(nat):
            dA[:,:] = 0.
            dA[i,:] = dAdi[i,:] * dzdscl[i]
            dA[:,i] = dAdi[i,:] * dzdscl[i]
            dAinvdh = -np.dot(Ainv,np.dot(dA,Ainv))
            dAinvBdh = dAinvdh @ mybvec
            dAinvDdh = dAinvdh @ dvec
            dmudh = np.dot(dvec,dAinvBdh) / np.dot(dvec,AinvD) \
                - (np.dot(dvec,AinvB) / np.dot(dvec,AinvD)**2) * np.dot(dvec,dAinvDdh)
            dqdscl[:] += dmudh*AinvD + mu*dAinvDdh - dAinvBdh

            
        dmudB = np.dot(dvec,Ainv) / np.dot(dvec,AinvD)
        dqdB = np.zeros( (nat,nat) )
        for i in range(nat):
            for j in range(nat):
                dqdB[i,j]  = dmudB[j] * AinvD[i] - Ainv[i,j]
        
        E = np.dot(dq,mybvec) + 0.5 * np.dot(dq,np.dot(A,dq))
        return E,dq,dqdB,dqdh,dqdscl

    

    def CptResponseDipole(self,dq):
        import numpy as np

        nat = self.crds.shape[0]
        dip = np.zeros( (3,) )
        for a in range(nat):
            dip[:] += dq[a] * self.crds[a,:]
        return dip

    
    
    def CptPolarizability(self,hardness,zetascl,delta=0.001,sym=True):
        import numpy as np

        nat = self.crds.shape[0]
        bvec = np.zeros( (nat,) )
        polar = np.zeros( (3,3) )
        for i in range(3):
            field = np.zeros( (3,) )
            
            field[i] = delta
            E,dq,dqdB,dqdh,dqdscl = self.SolveInterCPE(bvec,hardness,zetascl,field=field)
            dhi = self.CptResponseDipole(dq)

            field[i] = -delta
            E,dq,dqdB,dqdh,dqdscl = self.SolveInterCPE(bvec,hardness,zetascl,field=field)
            dlo = self.CptResponseDipole(dq)

            polar[i,:] = (dhi-dlo)/(2*delta)

        if sym:
            for i in range(3):
                for j in range(i):
                    a = polar[i,j]
                    b = polar[j,i]
                    c = (a+b)/2
                    polar[i,j] = c
                    polar[j,i] = c
                    
        return polar

    
            
    @classmethod
    def FromGaussian(cls,fname,esppar=None,extpar=None):
        from pathlib import Path
        #from . GaussianEsp import ReadGaussianEsp
        from . GaussianOutput import GaussianOutput
        from .. constants import GetAtomicNumber
        from .. constants import AU_PER_ANGSTROM
        pname = Path(fname)
        basename = pname.with_suffix("").name
        parts    = basename.split("_")
        name = parts[0]
        pertid = None
        if len(parts) > 1:
            pertid = "_".join( parts[1:] )
            
        if not pname.is_file():
            raise Exception(f"File not found: {fname}")

        #crds,esppts,espvals,extpts,extvals = ReadGaussianEsp(fname)
        o = GaussianOutput(fname)
        s = o.steps[-1]
        crds    = s.GetCrd() * AU_PER_ANGSTROM()
        atnums  = [ GetAtomicNumber(ele) for ele in s.GetElements() ]
        charge  = int(round(s.GetCharge()))
        esppts  = None
        espvals = None
        extpts  = None
        extvals = None

        self = cls(name,pertid,crds,atnums,charge,
                   esppts,espvals,extpts,extvals,
                   esppar,extpar)
        try:
            self.SetSurfaceValuesFromGaussianEsp(fname)
        except:
            pass
            
        return self
     
    @classmethod
    def FromPsi4(cls,fname,esppar=None,extpar=None):
        from pathlib import Path
        #from . GaussianEsp import ReadGaussianEsp
        #from . GaussianOutput import GaussianOutput
        from .. constants import GetAtomicNumber
        from .. constants import AU_PER_ANGSTROM
        from . Psi4Esp import ReadPsi4Output
        
        pname = Path(fname)
        basename = pname.with_suffix("").name
        parts    = basename.split("_")
        name = parts[0]
        pertid = None
        if len(parts) > 1:
            pertid = "_".join( parts[1:] )
            
        if not pname.is_file():
            raise Exception(f"File not found: {fname}")

        #o = GaussianOutput(fname)
        #s = o.steps[-1]
        #crds    = s.GetCrd() * AU_PER_ANGSTROM()
        #atnums  = [ GetAtomicNumber(ele) for ele in s.GetElements() ]
        #charge  = int(round(s.GetCharge()))

        atnums,crds,charge,mult = ReadPsi4Output(str(pname))
        
        esppts  = None
        espvals = None
        extpts  = None
        extvals = None

        self = cls(name,pertid,crds,atnums,charge,
                   esppts,espvals,extpts,extvals,
                   esppar,extpar)
        try:
            self.SetSurfaceValuesFromGaussianEsp(fname)
        except:
            pass
            
        return self


    @classmethod
    def FromQuick(cls,fname,esppar=None,extpar=None):
        from pathlib import Path
        #from . GaussianEsp import ReadGaussianEsp
        #from . GaussianOutput import GaussianOutput
        from .. constants import GetAtomicNumber
        from .. constants import AU_PER_ANGSTROM
        from . QuickEsp import ReadQuickOutput
        
        pname = Path(fname)
        basename = pname.with_suffix("").name
        parts    = basename.split("_")
        name = parts[0]
        pertid = None
        if len(parts) > 1:
            pertid = "_".join( parts[1:] )
            
        if not pname.is_file():
            raise Exception(f"File not found: {fname}")

        #o = GaussianOutput(fname)
        #s = o.steps[-1]
        #crds    = s.GetCrd() * AU_PER_ANGSTROM()
        #atnums  = [ GetAtomicNumber(ele) for ele in s.GetElements() ]
        #charge  = int(round(s.GetCharge()))

        atnums,crds,charge,mult = ReadQuickOutput(str(pname))
        
        esppts  = None
        espvals = None
        extpts  = None
        extvals = None

        self = cls(name,pertid,crds,atnums,charge,
                   esppts,espvals,extpts,extvals,
                   esppar,extpar)
        try:
            self.SetSurfaceValuesFromGaussianEsp(fname)
        except:
            pass
            
        return self



    @classmethod
    def FromJson(cls,fname,esppar=None,extpar=None):
        from pathlib import Path
        import numpy as np
        import parmed
        from .. constants import AU_PER_ANGSTROM
        from .. Reader import FixParmedAtomicNumbers
        from .. Struct import ListOfStruct
        
        pname    = Path(fname)
        basename = pname.with_suffix("").name
        parts    = basename.split("_")
        name     = parts[0]
        pertid   = None
        if len(parts) > 1:
            pertid = "_".join( parts[1:] )
            
        if not pname.is_file():
            raise Exception(f"File not found: {fname}")

        #p       = parmed.load_file(fname)
        #FixParmedAtomicNumbers(p)
        p = ListOfStruct.from_file(fname)
        s = p.structs[0]
        crds    = s.GetCrds() * AU_PER_ANGSTROM()
        atnums  = s.GetAtomicNumbers()
        charge  = s.GetCharge()
        extpts  = None
        extvals = None
        esppts  = None
        espvals = None
        
        return cls(name,pertid,crds,atnums,charge,
                   esppts,espvals,extpts,extvals,
                   esppar,extpar)


    @classmethod
    def FromStruct(cls,s,esppar=None,extpar=None):
        #from pathlib import Path
        #import numpy as np
        #import parmed
        from .. constants import AU_PER_ANGSTROM
        #from .. Reader import FixParmedAtomicNumbers
        #from .. Struct import ListOfStruct
        
        #pname    = Path(fname)
        #basename = pname.with_suffix("").name
        #parts    = basename.split("_")
        #name     = parts[0]
        pertid   = None
        #if len(parts) > 1:
        #    pertid = "_".join( parts[1:] )
            
        #p = ListOfStruct.from_file(fname)
        #s = p.structs[0]
        name = s.data["name"]
        crds    = s.GetCrds() * AU_PER_ANGSTROM()
        atnums  = s.GetAtomicNumbers()
        charge  = s.GetCharge()
        extpts  = None
        extvals = None
        esppts  = None
        espvals = None
        
        return cls(name,pertid,crds,atnums,charge,
                   esppts,espvals,extpts,extvals,
                   esppar,extpar)

    
              
    @classmethod
    def FromMol2(cls,fname,esppar=None,extpar=None):
        from pathlib import Path
        import numpy as np
        import parmed
        from .. constants import AU_PER_ANGSTROM
        from .. Reader import FixParmedAtomicNumbers
        
        pname    = Path(fname)
        basename = pname.with_suffix("").name
        parts    = basename.split("_")
        name     = parts[0]
        pertid   = None
        if len(parts) > 1:
            pertid = "_".join( parts[1:] )
            
        if not pname.is_file():
            raise Exception(f"File not found: {fname}")

        p       = parmed.load_file(fname,structure=True)
        FixParmedAtomicNumbers(p)
        crds    = np.array([ [a.xx,a.xy,a.xz] for a in p.atoms ]) * AU_PER_ANGSTROM()
        atnums  = [a.atomic_number for a in p.atoms]
        charge  = int(round(sum([a.charge for a in p.atoms])))
        extpts  = None
        extvals = None
        esppts  = None
        espvals = None
        
        return cls(name,pertid,crds,atnums,charge,
                   esppts,espvals,extpts,extvals,
                   esppar,extpar)

        
    def GetExtSurfaceCosmoResponse(self,qatoms):
        import copy
        import numpy as np
        if self.extsurf is None:
            raise Exception("Cannot run cosmo when extsurf is None")
        if len(self.extsurf.surfs) != 1:
            raise Exception("Can only run cosmo when extsurf as a single layer")
        s = copy.deepcopy(self.extsurf.surfs[0])
        if len(qatoms) != len(s.charges):
            raise Exception(f"Atom count mismatch {len(qatoms)} vs {len(s.charges)}")
        for i,a in enumerate(s.atoms):
            a.q = qatoms[i]
        s.CptSurfaceResponse()
        qext = np.array([ e.q for e in s.elems ])
        #print("qext=",qext)
        return qext
    

    def GetExtSurfaceArea(self):
        if self.extsurf is None:
            raise Exception("Cannot get surface area when extsurf is None")
        if len(self.extsurf.surfs) != 1:
            raise Exception("Can only get surface area when extsurf as a single layer")
        return self.extsurf.surfs[0].CptSurfaceArea()
