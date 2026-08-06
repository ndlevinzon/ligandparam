#!/usr/bin/env python3


            
class CosmoSurface(object):
    
    def __init__(self,
                 crds,charges,radii,nsurfs,
                 switch_thickness=0.15,
                 eps=78.4,
                 surface_penalty=4.6245957465e-05): # 72 dyne/cm

        #
        # All inputs are atomic units
        # 
        
        from copy import deepcopy
        import numpy as np

        from . CosmoAtom import CosmoAtom
        from . CosmoElement import CosmoElement
        
        self.crds = np.array(crds,copy=True)
        self.charges = np.array(charges,copy=True)
        self.radii = np.array(radii,copy=True)
        self.nsurfs = [n for n in nsurfs]
        self.switch_thickness = switch_thickness
        self.eps = eps
        self.surface_penalty = surface_penalty

        self.dielectric_factor = self.eps/(self.eps-1)
        self.switch_offset = 0.5
        if self.switch_thickness > 1.e-4:
             oos = 1 / self.switch_thickness
             self.switch_offset = 0.5 + oos - np.sqrt(oos*oos - 1./28.)

        nat = self.crds.shape[0]
        self.atoms = []
        for a in range(nat):
            atom = CosmoAtom(self.crds[a,:],self.charges[a],self.nsurfs[a],self.radii[a])
            atom.aidx = a
            if nsurfs[a] > 0:
                thickness = self.switch_thickness * radii[a]
                atom.outer_radius = radii[a] + (1-self.switch_offset)*thickness
                atom.inner_radius = radii[a] - self.switch_offset*thickness
            self.atoms.append( atom )

        self.elems = []
        self.BuildSurface()


        
    def StoreSurfacePotentialFromAtoms(self):
        #
        # Override this method in a derived class, if desired
        #
        import numpy as np
        B = self.CptPointInteractionMatrix()
        qs = np.array([ atom.q for atom in self.atoms ])
        ps = np.dot( B.T, qs )
        for a,p in zip(self.elems,ps):
            a.p = p


            
    def CptEnergyAndStoreAtomPotentialFromSurface(self):
        #
        # Override this method in a derived class, if desired
        #
        import numpy as np
        B = self.CptPointInteractionMatrix()
        cs = np.array([ elem.q for elem in self.elems ])
        qs = np.array([ atom.q for atom in self.atoms ])
        ps = np.dot( B, cs )
        for a,p in zip(self.atoms,ps):
            a.p = p
        return np.dot( ps, qs )


    
    def GetSoluteCharge(self):
        #
        # Override this method in a derived class, if desired
        #
        import numpy as np
        return sum( [atom.q for atom in self.atoms] )
    

             
    def CptFirstOrderGradients(self):
        #
        # Override this method in a derived class, if desired
        #
        import numpy as np
        from scipy.special import erf

        SQRTPI = np.sqrt(np.pi)
        
        for pb in self.elems:
            zb = pb.zeta
            qb = pb.q
            sqz = np.sqrt(zb)
            batom = self.atoms[ pb.atidx ]

            for pa in self.atoms:
                qa = pa.q
                qq = qa*qb

                rab = pa.crd-pb.crd
                r = np.linalg.norm(rab)
                r2 = r*r
                oor2 = 1./r2

                op   = erf(sqz*r)/r
                dedr = qq * ((2*sqz/SQRTPI)*np.exp(-zb*r2)-op)*oor2
                g = dedr * rab
                pa.grd += g
                batom.grd -= g


                
    def BuildSurface(self):
        import numpy as np
        from . Lebedev import GetLebedevRule
        from . Lebedev import LebedevGaussianZetaScaleFactor
        from . CosmoElement import CosmoElement
        from . SwitchFcn import SwitchOn
        
        nat = len(self.atoms)
        self.nlist = self._build_nlist()
        self.elems = []
        angWt = []
        angCrd = []
        nang = 0
        for a in range(nat):
            pa = self.atoms[a]
            pa.elem_begin = len(self.elems)
            pa.elem_end  = len(self.elems)
            nquad = pa.nquad
            rada  = pa.radius
            if nquad != nang:
                angCrd,angWt = GetLebedevRule(nquad)
            for ipt in range(nquad):
                Rpt = rada * angCrd[ipt,:] + pa.crd
                Spt = 1.
                nneighbor = len(self.nlist[a])
                for bb in range(nneighbor):
                    pb = self.atoms[ self.nlist[a][bb] ]
                    innerb = pb.inner_radius
                    outerb = pb.outer_radius
                    innRad2 = innerb*innerb
                    outRad2 = outerb*outerb
                    RadRad2 = (rada + outerb)**2
                    Rab2 = np.linalg.norm( pa.crd-pb.crd )**2

                    #print("%3i %3i %5i Rab2=%13.4e RadRad2=%13.4e,innRad2=%13.4e,outRad2=%13.4e"%\
			#(a,self.nlist[a][bb],ipt,Rab2,RadRad2,innRad2,outRad2))
                    
                    if Rab2 < RadRad2:
                        sep2 = np.linalg.norm(pb.crd-Rpt)**2
                        if sep2 < innRad2:
                            Spt = 0.
                            break
                        elif sep2 < outRad2:
                            Spt *= SwitchOn( sep2, innRad2, outRad2 )
                if Spt > 0.:
                    pa.elem_end += 1
                    factor = LebedevGaussianZetaScaleFactor(nquad) / rada
                    zeta = factor * factor / angWt[ipt]
                    #print("a=%3i ele=%5i Rpt=%12.5f %12.5f %12.5f Spt=%20.10f"%\
                    #      (a,ipt,Rpt[0],Rpt[1],Rpt[2],Spt))
                    ele = CosmoElement( Rpt, a, zeta, Spt, angWt[ipt], rada*rada )
                    self.elems.append(ele)
                    self.elems[-1].aidx = len(self.elems)-1
                    
        
    def _build_nlist(self):
        import numpy as np
        nat = len(self.atoms)
        nlist = []
        for a in range(nat):
            nlist.append([])
        maxRadius = np.max( [ a.outer_radius for a in self.atoms ] )
        sqCutoff = (2*maxRadius)*(2*maxRadius)
        for a in range(1,nat):
            for b in range(a):
                r2 = np.linalg.norm( self.atoms[a].crd - self.atoms[b].crd )**2
                if r2 < sqCutoff:
                    nlist[a].append(b)
                    nlist[b].append(a)
        for a in range(nat):
            nlist[a].sort()
        return nlist

    
    
    def CptSurfaceArea(self):
        sa = 0
        for pa in self.atoms:
            f = pa.radius * pa.radius
            begin = pa.elem_begin
            end = pa.elem_end
            for i in range(begin,end):
                sa += f * self.elems[i].quadwt * self.elems[i].switchwt
        return sa


    
    def CptSurfacePenalty(self):
        return self.surface_penalty * self.CptSurfaceArea()


    
    def CptSurfaceSelfMatrix(self,with_switchwt=True):
        import numpy as np
        from scipy.special import erf
        
        nelem = len(self.elems)
        A = np.zeros( (nelem,nelem) )
        for b in range(nelem):
            pb = self.elems[b]
            zb = pb.zeta
            for a in range(b):
                pa = self.elems[a]
                za = pa.zeta
                zab = za*zb/(za+zb)
                sqz = np.sqrt(zab)
                r = np.linalg.norm( pa.crd-pb.crd )
                x = erf(sqz*r)/r
                A[a,b] = x
                A[b,a] = x
            if with_switchwt:
                swt = pb.switchwt
            else:
                swt = 1
            A[b,b] = np.sqrt( zb * 2. / np.pi ) / swt
        return A


    def CptSurfaceHarmonics(self,Lmax,avgqperarea=None):
        import numpy as np
        A = self.CptSurfaceSelfMatrix(with_switchwt=False)

        nmax = (Lmax+1)**2
        nmax = min(nmax,A.shape[0])
        E,U = np.linalg.eigh(A)
        idx = E.argsort()[::-1]
        #E = E[idx]
        U = U[:,idx]
        #E = E[:nmax]
        U = U[:,:nmax]
        # # e = u @ A @ u
        # # 1 = (u @ A @ u) / e
        # # 2*enenorm = u @ A @ u * (2*enenorm/e)
        # # 2*enenorm = v @ A @ v
        # # v = u * sqrt( 2*enenorm / e )
        # for i in range(nmax):
        #     U[:,i] = U[:,i] * np.sqrt( 2*enenorm / E[i] )
        # UAU = U.T @ A @ U

        sa = self.CptSurfaceArea()

        avgq = np.array([ np.sum(np.abs(U[:,i]))/sa for i in range(U.shape[1]) ])
        for i in range(U.shape[1]):
            scl = 1.
            if avgqperarea is not None:
                if avgqperarea > 0 and avgq[i] > 0:
                    scl = avgqperarea/avgq[i]
            U[:,i] *= scl

        return U
    
    
    def CptESPMatrix(self):
        import numpy as np
        from scipy.special import erf
        
        nelem = len(self.elems)
        nat = len(self.atoms)
        A = np.zeros( (nat,nelem) )
        for b in range(nelem):
            pb = self.elems[b]
            for a in range(nat):
                pa = self.atoms[a]
                A[a,b] = 1. / np.linalg.norm( pa.crd-pb.crd )
        return A

    
    def CptPointInteractionMatrix(self):
        import numpy as np
        from scipy.special import erf
        
        nelem = len(self.elems)
        nat = len(self.atoms)
        A = np.zeros( (nat,nelem) )
        for b in range(nelem):
            pb = self.elems[b]
            zb = pb.zeta
            for a in range(nat):
                pa = self.atoms[a]
                zab = zb
                sqz = np.sqrt(zab)
                r = np.linalg.norm( pa.crd-pb.crd )
                x = erf(sqz*r)/r
                A[a,b] = x
        return A

    
    
    def CptSurfaceResponse(self):
        import numpy as np

        for a in self.atoms:
            a.grd[:] = 0
            a.p = 0
        for a in self.elems:
            a.grd[:] = 0
            a.q = 0
            a.p = 0
        
        self.StoreSurfacePotentialFromAtoms()
        Qtot    = self.GetSoluteCharge()
        A       = self.CptSurfaceSelfMatrix()
        Btq     = np.array( [e.p for e in self.elems] )
        nelem   = len(self.elems)
        nat     = len(self.atoms)
        conVec  = np.array([1]*nelem)
        Ainv    = np.linalg.inv(A)
        AinvBtq = Ainv @ Btq
        
        num     = np.dot(conVec,AinvBtq) - Qtot
        den     = np.sum( Ainv )
        mu      = num/den
        mud_minus_Btq = mu - Btq
        
        c  = (Ainv @ mud_minus_Btq) / self.dielectric_factor
        for e,q in zip(self.elems,c):
            e.q = q
        
        Ac    = A @ c
        E2    = 0.5 * self.dielectric_factor * np.dot(c,Ac)
        Esurf = self.CptSurfacePenalty()
        E1    = self.CptEnergyAndStoreAtomPotentialFromSurface()
        Etot  = Esurf + E1 + E2
        
        #print("surf=%20.10e e1=%20.10e e2=%20.10e mu=%20.10e num=%20.10e den=%20.10e\n"%\
	#      (Esurf,E1,E2,mu,num,den))

        return Etot
    


    def CptGradients(self):
        import numpy as np
        
        for a in self.atoms:
            a.grd[:] = 0.
        for b in self.elems:
            b.grd[:] = 0.
        self.CptSecondOrderGradients()
        self.CptSwitchGradients()
        self.CptFirstOrderGradients()
        
        return np.array([ a.grd for a in self.atoms ])
    

        
    def CptSecondOrderGradients(self):
        import numpy as np
        from scipy.special import erf
        
        SQRTPI = np.sqrt(np.pi)
        nelem = len(self.elems)
        for b in range(nelem):
            pb    = self.elems[b]
            zb    = pb.zeta
            atomb = self.atoms[pb.atidx]
            for a in range(b):
                pa   = self.elems[a]
                za   = pa.zeta
                atoma= self.atoms[pa.atidx]
                qq   = pa.q * pb.q
                zab  = za*zb/(za+zb)
                sqz  = np.sqrt(zab)
                rvec = pa.crd-pb.crd
                r    = np.linalg.norm(rvec)
                r2   = r*r
                oor2 = 1. / r2
                op   = erf(sqz*r)/r
                dedr = self.dielectric_factor * qq * ((2*sqz/SQRTPI)*np.exp(-zab*r2)-op)*oor2
                g = dedr * rvec
                atoma.grd += g
                atomb.grd -= g

                
                
    def CptSwitchGradients(self):
        import numpy as np
        from . SwitchFcn import SwitchOnGrd

        for pa in self.atoms:
            rada = pa.radius
            begin = pa.elem_begin
            end = pa.elem_end
            surfpref = self.surface_penalty * rada * rada

            for pe in self.elems[begin:end]:
                qe = pe.q
                sw = pe.switchwt

                Eself = 0.5 * self.dielectric_factor \
                    * qe * qe * np.sqrt(pe.zeta*2/np.pi)/sw

                dEdsw = surfpref * pe.quadwt - Eself/sw

                nn = len(self.nlist[pa.aidx])
                for bb in range(nn):
                    pb = self.atoms[self.nlist[pa.aidx][bb]]

                    innerb = pb.inner_radius
                    outerb = pb.outer_radius

                    innRad2 = innerb**2
                    outRad2 = outerb**2

                    RadRad2 = (rada + outerb)**2
                    Rab2 = np.linalg.norm(pa.crd-pb.crd)**2

                    if Rab2 < RadRad2:
                        Reb = pe.crd-pb.crd
                        Reb2 = np.linalg.norm(Reb)**2
                        if Reb2 < outRad2:
                            seb,dseb = SwitchOnGrd(Reb2,innRad2,outRad2)
                            if seb > 1.e-10:
                                pref = dEdsw * dseb * 2. * (sw/seb)
                                g = pref * Reb
                                pa.grd += g
                                pb.grd -= g

                   
