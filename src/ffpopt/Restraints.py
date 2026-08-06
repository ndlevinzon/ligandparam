#!/usr/bin/env python3


# def CptRmsTransform(crd,rcrd,wts):
#     """Returns the rotation matrix and center of mass
#     translation that would overlay crd onto rcrd

#     Parameters
#     ----------
#     crd : numpy.array, shape=(nat,3)
#         The coordinates to modify

#     rcrd : numpy.array, shape=(nat,3)
#         The reference coordinates

#     wts : numpy.array, shape=(nat,)
#         The weights (usually masses)

#     Returns
#     -------
#     rmsPD : float
#         The coordinate difference root mean square

#     rot : numpy.array, shape=(3,3)
#         The rotation matrix, such that: 
#         outcrd = np.dot(origcrd-origcom,rot) + refcom

#     origcom : numpy.array, shape=(3,)
#         The center of mass of the atoms being rotated, such that:
#         outcrd = np.dot(origcrd-origcom,rot) + refcom

#     refcom : numpy.array, shape=(3,)
#         The center of mass of the reference atoms, such that:
#         outcrd = np.dot(origcrd-origcom,rot) + refcom

#     """
    
#     import numpy as np
#     from scipy.linalg import svd
#     wts = wts/np.sum(wts)

    
#     com = np.dot(wts,crd)
#     cs = crd-com
    
#     rcom = np.dot(wts,rcrd)
#     rcs = rcrd-rcom
    
#     A = np.zeros( (3,3) )
#     sumsq = 0.
#     for iat in range(crd.shape[0]):
#         for i in range(3):
#             sumsq += wts[iat] * ( cs[iat,i]**2 + rcs[iat,i]**2 )
#             for j in range(3):
#                 A[i,j] += wts[iat] * cs[iat,j] * rcs[iat,i]

#     U,s,VT = svd(A,lapack_driver='gesvd')
#     detU = np.linalg.det(U)
#     detV = np.linalg.det(VT.T)
#     dot = s[0]*s[1]*s[2]
#     if detU*detV < 0:
#         jmin=0
#         if s[1] < s[jmin]:
#             jmin=1
#         if s[2] < s[jmin]:
#             jmin=2
#             dot -= 2*s[jmin]
#         U[:,jmin] = -U[:,jmin]

#     rmsPD = np.sqrt( max(sumsq-2*dot,0) )

#     A = np.dot( VT.T, U.T )
#     #ocrd = np.dot(cs,A) + rcom
#     return rmsPD,A,com,rcom



# def PerformRmsOverlay(crd,rcrd,wts):
#     """Rotates and translates crds to overlay with rcrds

#     Parameters
#     ----------
#     crd : numpy.array, shape=(nat,3)
#         The coordinates to modify

#     rcrd : numpy.array, shape=(nat,3)
#         The reference coordinates

#     wts : numpy.array, shape=(nat,)
#         The weights (usually masses)

#     Returns
#     -------
#     rmsPD : float
#         The coordinate difference root mean square

#     ocrd : numpy.array, shape=(nat,3)
#         The output coordinates
#     """
    
#     import numpy as np
#     rms,rot,com,refcom = CptRmsTransform(crd,rcrd,wts)
#     return rms,np.dot(crd-com,rot)+refcom


class RestraintList(object):
    def __init__(self,rests):
        self.rests = [ res for res in rests ]

    def __iter__(self):
        return iter(self.rests)
        
    def __len__(self):
        return len(self.rests)

    def __getitem__(self, key):
        """Defines reading via bracket syntax: value = obj[key]"""
        return self.rests[key]

    def __setitem__(self, key, value):
        """Defines writing/modifying via bracket syntax: obj[key] = value"""
        # You can add custom validation logic here before storing data
        self.rests[key] = value

    def __delitem__(self, key):
        """Defines deletion via bracket syntax: del obj[key]"""
        del self.rests[key]

        
    def GetValueAndGradient(self,crds):
        import numpy as np
        crds = np.array(crds)
        g = np.zeros( crds.shape )
        e = 0.0
        for res in self.rests:
            mye,myg = res.GetValueAndGradient(crds)
            e += mye
            g += myg
        return e,g

    def extend(self,x):
        self.rests.extend(x)
    
    def append(self,x):
        self.rests.append(x)
    
    @classmethod
    def from_list_of_str(cls,los):
        rests = []
        for s in los:
            cs = s.split(",")
            name = cs[0]
            k = float(cs[1])
            
            val = None
            if name in ["bond","angle","dihed","r12","puckerx","puckery"]:
                if "=" in cs[-1]:
                    last,val = cs[-1].split("=")
                    if "," in val:
                        val = [ float(x) for x in val.split(",") ]
                    else:
                        val = float(val)
                    cs[-1] = last
                    
            if name == "bond":
                idxs = [ int(cs[2]), int(cs[3]) ]
                rests.append( BondRestraint(k,idxs,val) )
            elif name == "angle":
                idxs = [ int(cs[2]), int(cs[3]), int(cs[4]) ]
                rests.append( AngleRestraint(k,idxs,val) )
            elif name == "dihed":
                idxs = [ int(cs[2]), int(cs[3]), int(cs[4]), int(cs[5]) ]
                #print(k,idxs,val)
                rests.append( DihedRestraint(k,idxs,val) )
            elif name == "r12":
                idxs = [ int(cs[2]), int(cs[3]), int(cs[4]), int(cs[5]) ]
                rests.append( R12Restraint(k,idxs,val) )
            elif name == "puckerx":
                idxs = [ int(cs[2]), int(cs[3]), int(cs[4]), int(cs[5]), int(cs[6]) ]
                rests.append( PuckerXRestraint(k,idxs,val) )
            elif name == "puckery":
                idxs = [ int(cs[2]), int(cs[3]), int(cs[4]), int(cs[5]), int(cs[6]) ]
                rests.append( PuckerYRestraint(k,idxs,val) )
                
            elif name == "rms":
                cs = ",".join( cs[2:] )
                cs = cs.replace("="," ")
                crds = None
                icrds = None
                for i in range(len(cs)-1):
                    if cs[i] == "crds":
                        icrds = i
                        break
                if icrds is None:
                    raise Exception("rms requires crds=")
                if cs[icrds] == "crds":
                    crds = np.array( [ float(x) for x in cs[icrds+1].split(",") ] )
                    n3 = len(crds)
                    n = n3//3
                    crds = crds.reshape( (n,3) )
                wts = np.array( [1]*n )
                iwts = None
                for i in range(len(cs)-1):
                    if iwts[i] == "wts":
                        iwts = i
                        break
                if iwts is not None:
                    hs = np.array( [ float(x) for x in cs[iwts+1].split(",") ] )
                    if hs.shape[0] != cs.shape[0]:
                        raise Exception(f"rms wts size {hs.shape} inconsistent with xyz size {crds.shape}")
                    wts = hs
                rests.append( RmsRestraint(k,crds,wts) )
        return cls(rests)

    
    def to_list_of_dict(self):
        return [ r.to_dict() for r in self.rests ]

    @classmethod
    def from_list_of_dict(cls,data):
        rests = []
        for x in data:
            if x["name"] == "bond":
                rests.append( BondRestraint.from_dict(x) )
            elif x["name"] == "angle":
                rests.append( AngleRestraint.from_dict(x) )
            elif x["name"] == "dihed":
                rests.append( DihedRestraint.from_dict(x) )
            elif x["name"] == "r12":
                rests.append( R12Restraint.from_dict(x) )
            elif x["name"] == "puckerx":
                rests.append( PuckerXRestraint.from_dict(x) )
            elif x["name"] == "puckery":
                rests.append( PuckerYRestraint.from_dict(x) )
            elif x["name"] == "rms":
                rests.append( RmsRestraint.from_dict(x) )
            elif x["name"] == "twist":
                rests.append( TwistRestraint.from_dict(x) )
            else:
                name = x["name"]
                raise Exception(f"Unknown restraint name: {name}")
        return cls(rests)

    

class Restraint(object):
    def __init__(self,name):
        self.name = name
        self.idxs = []

    def GetCrdValue(self,crds):
        return 0
    
    def GetValueAndGradients(self,crds):
        import numpy as np
        crds = np.array(crds)
        g = np.zeros( crds.shape )
        return 0,g

    def is_same(self,other):
        return self.name == other.name and self.idxs == other.idxs

    def isper(self):
        return False
    

class BondRestraint(Restraint):
    def __init__(self,k,idxs,value):
        super().__init__("bond")
        self.k = k
        self.idxs = [i for i in idxs]
        self.value = value

    def GetCrdValue(self,crds):
        import numpy as np
        from . Geometry import CptDist
        crds = np.array(crds)
        return CptDist(crds[self.idxs[0],:], crds[self.idxs[1],:])
    
    def GetValueAndGradients(self,crds):
        import numpy as np
        from . Geometry import CptDistAndGrd
        crds = np.array(crds)
        e = 0
        g = np.zeros( crds.shape )
        if self.value is not None:
            z,dzdra,dzdrb = CptDistAndGrd( crds[self.idxs[0],:], crds[self.idxs[1],:] )
            e = self.k * (z-self.value)**2
            tmp = 2 * self.k * (z-self.value)
            g[self.idxs[0],:] = tmp*dzdra
            g[self.idxs[1],:] = tmp*dzdrb
        else:
            raise Exception("BondRestraint value is None")
        return e,g
    
    def to_dict(self):
        ddata = {}
        ddata["name"]  = self.name
        ddata["k"]     = self.k
        ddata["idxs"]  = self.idxs
        ddata["value"] = self.value
        return ddata

    @classmethod
    def from_dict(cls,data):
        return cls(data["k"],data["idxs"],data["value"])
    


class AngleRestraint(Restraint):
    def __init__(self,k,idxs,value):
        super().__init__("angle")
        self.k = k
        self.idxs = [i for i in idxs]
        self.value = value

    def GetCrdValue(self,crds):
        import numpy as np
        from . Geometry import CptAngle
        crds = np.array(crds)
        return CptAngle(crds[self.idxs[0],:], crds[self.idxs[1],:], crds[self.idxs[2],:])
    
    def GetValueAndGradients(self,crds):
        import numpy as np
        from . Geometry import CptAngleAndGrd
        crds = np.array(crds)
        e = 0
        g = np.zeros( crds.shape )
        if self.value is not None:
            z,dzdra,dzdrb,dzdrc = CptAngleAndGrd\
                ( crds[self.idxs[0],:], crds[self.idxs[1],:], crds[self.idxs[2],:] )
            e = self.k * (z-self.value)**2
            tmp = 2 * self.k * (z-self.value)
            g[self.idxs[0],:] = tmp*dzdra
            g[self.idxs[1],:] = tmp*dzdrb
            g[self.idxs[2],:] = tmp*dzdrc
        else:
            raise Exception("AngleRestraint value is None")
        return e,g
    
    def to_dict(self):
        ddata = {}
        ddata["name"]  = self.name
        ddata["k"]     = self.k
        ddata["idxs"]  = self.idxs
        ddata["value"] = self.value
        return ddata

    @classmethod
    def from_dict(cls,data):
        return cls(data["k"],data["idxs"],data["value"])
    
    
class DihedRestraint(Restraint):
    def __init__(self,k,idxs,value):
        super().__init__("dihed")
        self.k = k
        self.idxs = [i for i in idxs]
        self.value = value

    def GetCrdValue(self,crds):
        import numpy as np
        from . Geometry import CptDihed
        crds = np.array(crds)
        return CptDihed(crds[self.idxs[0],:], crds[self.idxs[1],:],
                        crds[self.idxs[2],:], crds[self.idxs[3],:])
    
    def GetValueAndGradients(self,crds):
        import numpy as np
        from . Geometry import CptDihedAndGrd
        crds = np.array(crds)
        e = 0
        g = np.zeros( crds.shape )
        if self.value is not None:
            z,dzdra,dzdrb,dzdrc,dzdrd = CptDihedAndGrd\
                ( crds[self.idxs[0],:], crds[self.idxs[1],:],
                  crds[self.idxs[2],:], crds[self.idxs[3],:] )
            diff = (z - self.value + 180 + 180) % 360 - 180
            e = (self.k/2)*(1+np.cos(np.deg2rad(diff)))
            tmp = self.k * (-np.sin(np.deg2rad(diff)))*(np.pi/180)
            #print(z,self.value,diff,e)
            g[self.idxs[0],:] = tmp*dzdra
            g[self.idxs[1],:] = tmp*dzdrb
            g[self.idxs[2],:] = tmp*dzdrc
            g[self.idxs[3],:] = tmp*dzdrd
        else:
            raise Exception("DihedRestraint value is None")
        return e,g
    
    def to_dict(self):
        ddata = {}
        ddata["name"]  = self.name
        ddata["k"]     = self.k
        ddata["idxs"]  = self.idxs
        ddata["value"] = self.value
        return ddata

    @classmethod
    def from_dict(cls,data):
        return cls(data["k"],data["idxs"],data["value"])
    
    
    def isper(self):
        return True
    

class R12Restraint(Restraint):
    def __init__(self,k,idxs,value):
        super().__init__("r12")
        self.k = k
        self.idxs = [i for i in idxs]
        self.value = value

    def GetCrdValue(self,crds):
        import numpy as np
        from . Geometry import CptR12
        crds = np.array(crds)
        return CptR12(crds[self.idxs[0],:], crds[self.idxs[1],:],
                      crds[self.idxs[2],:], crds[self.idxs[3],:],[1,-1])
    
    def GetValueAndGradients(self,crds):
        import numpy as np
        from . Geometry import CptR12AndGrd
        crds = np.array(crds)
        e = 0
        g = np.zeros( crds.shape )
        if self.value is not None:
            z,dzdra,dzdrb,dzdrc,dzdrd = CptR12AndGrd\
                ( crds[self.idxs[0],:], crds[self.idxs[1],:],
                  crds[self.idxs[2],:], crds[self.idxs[3],:] )
            e   = self.k * (z-self.value)**2
            tmp = 2 * self.k * (z-self.value)
            g[self.idxs[0],:] = tmp*dzdra
            g[self.idxs[1],:] = tmp*dzdrb
            g[self.idxs[2],:] = tmp*dzdrc
            g[self.idxs[3],:] = tmp*dzdrd
        else:
            raise Exception("R12Restraint value is None")
        return e,g
    
    def to_dict(self):
        ddata = {}
        ddata["name"]  = self.name
        ddata["k"]     = self.k
        ddata["idxs"]  = self.idxs
        ddata["value"] = self.value
        return ddata

    @classmethod
    def from_dict(cls,data):
        return cls(data["k"],data["idxs"],data["value"])
    
    



class RmsRestraint(Restraint):
    
    def __init__(self,k,crds,wts):
        super().__init__("rms")
        import numpy as np
        self.k = k
        self.crds = np.array(crds,copy=True)
        self.wts = np.array(wts,copy=True)
        self.value = 0

    def GetCrdValue(self,crds):
        import numpy as np
        from . Geometry import PerformRmsOverlay
        refcrds = crds
        inpcrds = np.array(self.crds,copy=True)
        rms,rotcrds = PerformRmsOverlay(inpcrds,refcrds,self.wts)
        return rms
    
    def GetValueAndGradients(self,crds):
        # The input crds are the "reference"
        import numpy as np
        from . Geometry import PerformRmsOverlay
        refcrds = crds
        inpcrds = np.array(self.crds,copy=True)
        rms,rotcrds = PerformRmsOverlay(inpcrds,refcrds,self.wts)

        grds = np.zeros( refcrds.shape )
        
        pen = 0
        for a in range(refcrds.shape[0]):
            for k in range(3):
                dx = crds[a,k]-rotcrds[a,k]
                dx2 = dx*dx
                pen += self.k * self.wts[a] * dx2
                grds[a,k] += self.k * self.wts[a] * 2 * dx
        return pen,grds


    def to_dict(self):
        ddata = {}
        ddata["name"] = self.name
        ddata["k"]    = self.k
        ddata["crds"] = self.crds.tolist()
        ddata["wts"]  = self.wts.tolist()
        return ddata

    
    @classmethod
    def from_dict(cls,data):
        return cls(data["k"],data["crds"],data["wts"])
    
    

    
    
class TwistRestraint(Restraint):

    def __init__(self,k,crds,wts1,wts2):
        super().__init__("twist")
        import numpy as np
        #import ase.io
        #from . constants import GetAtomicMass
        #from . AmberParm import RotateMask
        
        #geoms = ase.io.read(filename,index=":")
        #atoms = geoms[0]
        #crds = atoms.get_positions()

        #self.filename = filename
        self.crds = np.array(crds,copy=True)
        self.value = 0
        self.k = k
        self.wts1 = np.array([w for w in wts1])
        self.wts2 = np.array([w for w in wts2])
        self.rms1 = RmsRestraint(self.k,crds,self.wts1)
        self.rms2 = RmsRestraint(self.k,crds,self.wts2)

    def GetCrdValue(self,crds):
        return 0
    
    def GetValueAndGradients(self,crds):
        p1,g1 = self.rms1.GetValueAndGradients(crds)
        p2,g2 = self.rms2.GetValueAndGradients(crds)
        return p1+p2,g1+g2

    
    def to_dict(self):
        ddata = {}
        ddata["name"] = self.name
        ddata["k"]    = self.k
        ddata["crds"] = self.crds.tolist()
        ddata["wts1"] = self.wts1.tolist()
        ddata["wts2"] = self.wts2.tolist()
        return ddata

    
    @classmethod
    def from_dict(cls,data):
        return cls(data["k"],data["crds"],data["wts1"],data["wts2"])
    
    
    # @classmethod
    # def from_str(cls,s,graph):
    #     import ase.io
    #     from . constants import GetAtomicMass
    #     from . AmberParm import RotateBondMask
    #     import numpy as np
        
    #     cs = s.split(",")
    #     k = float(cs[0])
    #     iat = int(cs[1])
    #     jat = int(cs[2])
    #     fname = cs[3]

    #     geoms = ase.io.read(fname,index=":")
    #     atoms = geoms[0]
    #     crds = atoms.get_positions()
    #     eles = atoms.get_atomic_numbers()
    #     masses = np.array([ GetAtomicMass(z) for z in eles ])
    #     mask = RotateBondMask(graph,[iat,jat])

    #     if k < 0:
    #         masses *= abs(k)
    #     else:
    #         masses[:] = k
        
    #     wts1 = np.array( masses, copy=True )
    #     wts2 = np.array( masses, copy=True )
    #     for i in range( len(mask) ):
    #         if i == iat:
    #             continue
    #         elif i == jat:
    #             continue
    #         elif mask[i] > 0:
    #             wts2[i] = 0
    #         else:
    #             wts1[i] = 0

    #     return cls(fname,wts1,wts2)

    



class PuckerXRestraint(Restraint):
    def __init__(self,k,idxs,value):
        super().__init__("puckerx")
        self.k = k
        self.idxs = [i for i in idxs]
        if len(self.idxs) != 5:
            raise Exception(f"PuckerXRestraint idxs list must be len=5, received: {idxs}")
        self.value = value

    def GetCrdValue(self,crds):
        import numpy as np
        from . Geometry import CptPuckerX
        crds = np.array(crds)
        return CptPuckerX(crds[self.idxs[0],:], crds[self.idxs[1],:],
                          crds[self.idxs[2],:], crds[self.idxs[3],:],
                          crds[self.idxs[4],:])
    
    def GetValueAndGradients(self,crds):
        import numpy as np
        from . Geometry import CptPuckerXAndGrd
        crds = np.array(crds)
        e = 0
        g = np.zeros( crds.shape )
        if self.value is not None:
            z,dzdra,dzdrb,dzdrc,dzdrd,dzdre = CptPuckerXAndGrd\
                ( crds[self.idxs[0],:], crds[self.idxs[1],:],
                  crds[self.idxs[2],:], crds[self.idxs[3],:],
                  crds[self.idxs[4],:])
            
            # d0 = ( z[0] - self.value[0] )
            # d1 = ( z[1] - self.value[1] )
            # e = self.k * ( d0*d0 + d1*d1 )
            # tmp = 2 * self.k
            # g[self.idxs[0],:] = tmp*(dzdra[0,:]*d0+dzdra[1,:]*d1)
            # g[self.idxs[1],:] = tmp*(dzdrb[0,:]*d0+dzdrb[1,:]*d1)
            # g[self.idxs[2],:] = tmp*(dzdrc[0,:]*d0+dzdrc[1,:]*d1)
            # g[self.idxs[3],:] = tmp*(dzdrd[0,:]*d0+dzdrd[1,:]*d1)
            # g[self.idxs[4],:] = tmp*(dzdre[0,:]*d0+dzdre[1,:]*d1)

            d0 = ( z - self.value )
            e = self.k * ( d0*d0 )
            tmp = 2 * self.k
            g[self.idxs[0],:] = tmp*(dzdra[:]*d0)
            g[self.idxs[1],:] = tmp*(dzdrb[:]*d0)
            g[self.idxs[2],:] = tmp*(dzdrc[:]*d0)
            g[self.idxs[3],:] = tmp*(dzdrd[:]*d0)
            g[self.idxs[4],:] = tmp*(dzdre[:]*d0)
            
        else:
            raise Exception("PuckerRestraint value is None")
        return e,g
    
    def to_dict(self):
        ddata = {}
        ddata["name"]  = self.name
        ddata["k"]     = self.k
        ddata["idxs"]  = self.idxs
        ddata["value"] = self.value
        return ddata

    @classmethod
    def from_dict(cls,data):
        return cls(data["k"],data["idxs"],data["value"])
    
    






class PuckerYRestraint(Restraint):
    def __init__(self,k,idxs,value):
        super().__init__("puckery")
        self.k = k
        self.idxs = [i for i in idxs]
        if len(self.idxs) != 5:
            raise Exception(f"PuckerYRestraint idxs list must be len=5, received: {idxs}")
        self.value = value

    def GetCrdValue(self,crds):
        import numpy as np
        from . Geometry import CptPuckerY
        crds = np.array(crds)
        return CptPuckerY(crds[self.idxs[0],:], crds[self.idxs[1],:],
                          crds[self.idxs[2],:], crds[self.idxs[3],:],
                          crds[self.idxs[4],:])
    
    def GetValueAndGradients(self,crds):
        import numpy as np
        from . Geometry import CptPuckerYAndGrd
        crds = np.array(crds)
        e = 0
        g = np.zeros( crds.shape )
        if self.value is not None:
            z,dzdra,dzdrb,dzdrc,dzdrd,dzdre = CptPuckerYAndGrd\
                ( crds[self.idxs[0],:], crds[self.idxs[1],:],
                  crds[self.idxs[2],:], crds[self.idxs[3],:],
                  crds[self.idxs[4],:])
            
            # d0 = ( z[0] - self.value[0] )
            # d1 = ( z[1] - self.value[1] )
            # e = self.k * ( d0*d0 + d1*d1 )
            # tmp = 2 * self.k
            # g[self.idxs[0],:] = tmp*(dzdra[0,:]*d0+dzdra[1,:]*d1)
            # g[self.idxs[1],:] = tmp*(dzdrb[0,:]*d0+dzdrb[1,:]*d1)
            # g[self.idxs[2],:] = tmp*(dzdrc[0,:]*d0+dzdrc[1,:]*d1)
            # g[self.idxs[3],:] = tmp*(dzdrd[0,:]*d0+dzdrd[1,:]*d1)
            # g[self.idxs[4],:] = tmp*(dzdre[0,:]*d0+dzdre[1,:]*d1)

            d0 = ( z - self.value )
            e = self.k * ( d0*d0 )
            tmp = 2 * self.k
            g[self.idxs[0],:] = tmp*(dzdra[:]*d0)
            g[self.idxs[1],:] = tmp*(dzdrb[:]*d0)
            g[self.idxs[2],:] = tmp*(dzdrc[:]*d0)
            g[self.idxs[3],:] = tmp*(dzdrd[:]*d0)
            g[self.idxs[4],:] = tmp*(dzdre[:]*d0)
            
        else:
            raise Exception("PuckerRestraint value is None")
        return e,g
    
    def to_dict(self):
        ddata = {}
        ddata["name"]  = self.name
        ddata["k"]     = self.k
        ddata["idxs"]  = self.idxs
        ddata["value"] = self.value
        return ddata

    @classmethod
    def from_dict(cls,data):
        return cls(data["k"],data["idxs"],data["value"])
    
    


    
    

def restraints2info(atoms,cons):
    """ Convert a list of constraints to an info dictionary for the atoms object. 
    
    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to which the constraints will be added.
    cons : list of Constraint
        A list of Constraint objects to convert to an info dictionary.
    
    """
    from collections import defaultdict as ddict
    if cons is not None:
        rsts = ddict(list)
        for c in cons:
            rsts[c.rsttype].append( str(c) )
        for rsttype in rsts:
            atoms.info[rsttype] = str(rsts[rsttype])


def info2restraints(atoms,graph=None):
    """ Convert the constraints stored in the atoms.info dictionary to a list of Constraint objects.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object from which to get the constraints.
    graph : object, optional
        A graph object that can be used to set the mask for the constraints.
    
    Returns
    -------
    list of Restraint
        A list of Restraint objects created from the constraints stored in the atoms.info dictionary.
        If no constraints are found, it returns None.
    
    """
    import json
    cons = None
    if "twistrst" in atoms.info:
        cs = json.loads( atoms.info["twistrst"] )
        cons = [ TwistRestraint.from_full_str( c ) for c in cs ]
    return cons


