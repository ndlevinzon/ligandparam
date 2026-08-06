#!/usr/bin/env python3

import numpy as np


def _cross(v1, v2):
    """Computes the cross-product

    Parameters
    ----------
    v1 : numpy.array, shape=(3,)
        The left array

    v2 : numpy.array, shape=(3,)
        The right array

    Returns
    -------
    q : numpy.array, shape=(3,)
        The cross product v1 x v2
    """
    
    # Can't use np.cross for pypy, since it's not yet implemented
    return np.array([v1[1]*v2[2] - v1[2]*v2[1],
                     v1[2]*v2[0] - v1[0]*v2[2],
                     v1[0]*v2[1] - v1[1]*v2[0]])


def CptDist(a,b):
    """The distance between two points

    Parameters
    ----------
    a : numpy.array, shape=(3,)
        A point in space

    b : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    q : float
        The distance abs(a-b)
    """

    return np.linalg.norm(a-b)



def CptR12(a,b,c,d,rstwt):
    """A linear combination of 2 distances

    Parameters
    ----------
    a : numpy.array, shape=(3,)
        A point in space

    b : numpy.array, shape=(3,)
        A point in space

    c : numpy.array, shape=(3,)
        A point in space

    d : numpy.array, shape=(3,)
        A point in space

    rstwt : numpy.array, shape=(2,)
        The weights

    Returns
    -------
    q : float
        The linear combination of distances rstwt[0]*abs(a-b) + rstwt[1]*abs(c-d)
    """

    return rstwt[0] * CptDist(a,b) + rstwt[1] * CptDist(c,d)



def CptAngle(a,b,c):
    """Returns the angle of 3 points in degrees

    Parameters
    ----------
    a : numpy.array, shape=(3,)
        A point in space

    b : numpy.array, shape=(3,)
        A point in space

    c : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    q : float
        The angle in degrees
    """

    v1 = b-a
    v2 = b-c
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    n = np.dot(v1, v2)
    d = (l1*l2)
    return np.degrees( np.arccos(n/d) )



def CptDihed(a,b,c,d):
    """Returns the dihedral angle of 4 points in degrees

    Parameters
    ----------
    a : numpy.array, shape=(3,)
        A point in space

    b : numpy.array, shape=(3,)
        A point in space

    c : numpy.array, shape=(3,)
        A point in space

    d : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    q : float
        The angle in degrees
    """
    
    v1 = b-a
    v2 = b-c
    v3 = d-c
    # Take the cross product between v1-v2 and v2-v3
    v1xv2 = _cross(v1, v2)
    v2xv3 = _cross(v2, v3)
    # Now find the angle between these cross-products
    l1 = np.linalg.norm(v1xv2)
    l2 = np.linalg.norm(v2xv3)
    cosa = np.dot(v1xv2, v2xv3) / (l1 * l2)
    if np.dot(v3, v1xv2) <= 0.0 :
        return np.degrees(np.arccos(cosa))
    else :
        return -np.degrees(np.arccos(cosa))

    



def CptDistAndGrd(ca,cb):
    """The distance between two points and return the gradient

    Parameters
    ----------
    a : numpy.array, shape=(3,)
        A point in space

    b : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    q : float
        The distance abs(a-b)

    dqda : numpy.array, shape=(3,)
        Gradient with respect to a

    dqdb : numpy.array, shape=(3,)
        Gradient with respect to b
    """
    
    import numpy as np
    #rabv = crds[iat,:] - crds[jat,:]
    rabv = ca-cb
    rab2 = np.dot(rabv,rabv)
    rab = np.sqrt(rab2)
    z = rab
    dzdra = (0.5/rab) * (2*rabv[:])
    dzdrb = (0.5/rab) * (2*rabv[:]) * (-1.)
    return z,dzdra,dzdrb


def CptAngleAndGrd(ca,cb,cc):
    """Returns the angle of 3 points in degrees and gradient

    Parameters
    ----------
    a : numpy.array, shape=(3,)
        A point in space

    b : numpy.array, shape=(3,)
        A point in space

    c : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    q : float
        The angle in degrees

    dqda : numpy.array, shape=(3,)
        Gradient with respect to a

    dqdb : numpy.array, shape=(3,)
        Gradient with respect to b

    dqdc : numpy.array, shape=(3,)
        Gradient with respect to c
    """

    #(crds,iat,jat,kat):
    import numpy as np
    f = 180./np.pi
    #x1, y1, z1 = crds[iat,0],crds[iat,1],crds[iat,2]
    #x2, y2, z2 = crds[jat,0],crds[jat,1],crds[jat,2]
    #x3, y3, z3 = crds[kat,0],crds[kat,1],crds[kat,2]
    x1, y1, z1 = ca[0],ca[1],ca[2]
    x2, y2, z2 = cb[0],cb[1],cb[2]
    x3, y3, z3 = cc[0],cc[1],cc[2]
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x2 - x3, y2 - y3, z2 - z3])
    l1 = np.sqrt(np.dot(v1, v1))
    l2 = np.sqrt(np.dot(v2, v2))
    n = np.dot(v1, v2)
    d = (l1*l2)
    cosa = n / d
    z = np.arccos(cosa) * f
    dzdca = -f/np.sqrt(1.-cosa**2)
    dddv1 = v1/l1
    dcadv1 = v2/d + (-n/(d*d)) * l2 * (v1/l1)
    dcadv2 = v1/d + (-n/(d*d)) * l1 * (v2/l2)

    dv1da = np.zeros( (3,3) )
    dv1db = np.zeros( (3,3) )

    dv1da[0,0] = -1.
    dv1da[1,1] = -1.
    dv1da[2,2] = -1.
    dv1db[0,0] =  1.
    dv1db[1,1] =  1.
    dv1db[2,2] =  1.
    dv2db = dv1db
    dv2dc = dv1da
    
    dzda = dzdca * np.dot( dcadv1, dv1da )
    dzdb = dzdca * ( np.dot( dcadv1, dv1db ) + np.dot( dcadv2, dv2db ) )
    dzdc = dzdca * np.dot( dcadv2, dv2dc )

    #dzda = -v1/l1
    #dzdb =  v1/l1 + v2/l2
    #dzdc = -v2/l2
    return z,dzda,dzdb,dzdc




def CptDihedAndGrd(a,b,c,d):
    """Returns the dihedral angle and gradients of 4 points in degrees

    Parameters
    ----------
    a : numpy.array, shape=(3,)
        A point in space

    b : numpy.array, shape=(3,)
        A point in space

    c : numpy.array, shape=(3,)
        A point in space

    d : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    q : float
        The angle in degrees

    dqda : numpy.array, shape=(3,)
        Derivative with respect to a (degrees/distance)

    dqdb : numpy.array, shape=(3,)
        Derivative with respect to b (degrees/distance)

    dqdc : numpy.array, shape=(3,)
        Derivative with respect to c (degrees/distance)

    dqdd : numpy.array, shape=(3,)
        Derivative with respect to d (degrees/distance)
    """
    import numpy as np
    
    T = 0
    
    v1 = b-a
    v2 = b-c
    v3 = d-c
    # Take the cross product between v1-v2 and v2-v3
    v1xv2 = _cross(v1, v2)
    v2xv3 = _cross(v2, v3)
    # Now find the angle between these cross-products
    l1 = np.linalg.norm(v1xv2)
    l2 = np.linalg.norm(v2xv3)
    cosa = min(1,max(-1,np.dot(v1xv2, v2xv3) / (l1 * l2)))
    if np.dot(v3, v1xv2) <= 0.0 :
        T =  np.degrees(np.arccos(cosa))
    else :
        T = -np.degrees(np.arccos(cosa))

    BxA = _cross(v1xv2,v2xv3)
    ncb = np.linalg.norm(v2)
    
    sclA = 1 / ( l1*l1*ncb )
    if abs(sclA) > 1.e+8:
        sclA = 0.
    AxRcb = _cross(v1xv2*sclA,v2)

    sclB = 1 / ( l2*l2*ncb )
    if abs(sclB) > 1.e+8:
        sclB = 0.
    RcbxB = _cross(v2,v2xv3*sclB)

    dTda = _cross(AxRcb,c-b)
    dTdb = _cross(c-a,AxRcb) + _cross(RcbxB,d-c)
    dTdc = _cross(AxRcb,b-a) + _cross(d-b,RcbxB)
    #dTdd = -(dTda+dTdb+dTdc)
    dTdd = _cross(RcbxB,c-b)
    return T,np.degrees(dTda),np.degrees(dTdb),np.degrees(dTdc),np.degrees(dTdd)


    

def CptR12AndGrd(rstwt1,rstwt2,ca,cb,cc,cd):
    """A linear combination of 2 distances and gradient

    Parameters
    ----------
    a : numpy.array, shape=(3,)
        A point in space

    b : numpy.array, shape=(3,)
        A point in space

    c : numpy.array, shape=(3,)
        A point in space

    d : numpy.array, shape=(3,)
        A point in space

    rstwt1 : float
        The weight 1

    rstwt2 : float
        The weight 2

    Returns
    -------
    q : float
        The linear combination of distances rstwt1*abs(a-b) + rstwt2*abs(c-d)

    dqda : numpy.array, shape=(3,)
        Gradient with respect to a

    dqdb : numpy.array, shape=(3,)
        Gradient with respect to b

    dqdc : numpy.array, shape=(3,)
        Gradient with respect to c

    dqdd : numpy.array, shape=(3,)
        Gradient with respect to d
    """

    # (crds, rstwt1, rstwt2, iat,jat,kat,lat):
    import numpy as np
    #rabv = crds[iat,:] - crds[jat,:]
    #rcdv = crds[kat,:] - crds[lat,:]
    rabv = ca-cb
    rcdv = cc-cd
    rab2 = np.dot(rabv,rabv)
    rcd2 = np.dot(rcdv,rcdv)
    #  (dx^2+dy^2+dz^2)^(1/2)
    # (1/2) (dx^2+dy^2+dz^2)^(-1/2) * 2*dx
    rab = np.sqrt(rab2)
    rcd = np.sqrt(rcd2)
    z = rstwt1 * rab + rstwt2 * rcd
    dzdra = rstwt1 * ( (0.5/rab) * (2*rabv[:]) )
    dzdrb = rstwt1 * ( (0.5/rab) * (2*rabv[:]) * (-1.) )
    dzdrc = rstwt2 * ( (0.5/rcd) * (2*rcdv[:]) )
    dzdrd = rstwt2 * ( (0.5/rcd) * (2*rcdv[:]) * (-1.) )
    return z,dzdra,dzdrb,dzdrc,dzdrd


def CptCoM(crd,wts):
    """Return the vector pointing at the center of mass
    
    Parameters
    ----------
    crd : numpy.array, shape=(nat,3)
        The coordinates to modify

    wts : numpy.array, shape=(nat,)
        The weights (usually masses)

    Returns
    -------

    com : numpy.array, shape=(3,)
        Center of mass coordinates, np.dot(wts/np.sum(wts),crd)
    """
    import numpy as np
    return np.dot(wts/np.sum(wts),crd)


def RemoveCoM(crd,wts):
    """Translates the coordinates to remove the center of mass
    
    Parameters
    ----------
    crd : numpy.array, shape=(nat,3)
        The coordinates to modify

    wts : numpy.array, shape=(nat,)
        The weights (usually masses)

    Returns
    -------

    ocrd : numpy.array, shape=(nat,3)
        The output coordinates, crd-CptCoM(crd,wts)
    """
    import numpy as np
    return crd - CptCoM(crd,wts)
    

def CptRmsTransform(crd,rcrd,wts):
    """Returns the rotation matrix and center of mass
    translation that would overlay crd onto rcrd

    Parameters
    ----------
    crd : numpy.array, shape=(nat,3)
        The coordinates to modify

    rcrd : numpy.array, shape=(nat,3)
        The reference coordinates

    wts : numpy.array, shape=(nat,)
        The weights (usually masses)

    Returns
    -------
    rmsPD : float
        The coordinate difference root mean square

    rot : numpy.array, shape=(3,3)
        The rotation matrix, such that: 
        outcrd = np.dot(origcrd-origcom,rot) + refcom

    origcom : numpy.array, shape=(3,)
        The center of mass of the atoms being rotated, such that:
        outcrd = np.dot(origcrd-origcom,rot) + refcom

    refcom : numpy.array, shape=(3,)
        The center of mass of the reference atoms, such that:
        outcrd = np.dot(origcrd-origcom,rot) + refcom

    """
    
    import numpy as np
    from scipy.linalg import svd
    wts = wts/np.sum(wts)

    
    com = np.dot(wts,crd)
    cs = crd-com
    
    rcom = np.dot(wts,rcrd)
    rcs = rcrd-rcom
    
    A = np.zeros( (3,3) )
    sumsq = 0.
    for iat in range(crd.shape[0]):
        for i in range(3):
            sumsq += wts[iat] * ( cs[iat,i]**2 + rcs[iat,i]**2 )
            for j in range(3):
                A[i,j] += wts[iat] * cs[iat,j] * rcs[iat,i]

    U,s,VT = svd(A,lapack_driver='gesvd')
    detU = np.linalg.det(U)
    detV = np.linalg.det(VT.T)
    dot = s[0]*s[1]*s[2]
    if detU*detV < 0:
        jmin=0
        if s[1] < s[jmin]:
            jmin=1
        if s[2] < s[jmin]:
            jmin=2
            dot -= 2*s[jmin]
        U[:,jmin] = -U[:,jmin]

    rmsPD = np.sqrt( max(sumsq-2*dot,0) )

    A = np.dot( VT.T, U.T )
    #ocrd = np.dot(cs,A) + rcom
    return rmsPD,A,com,rcom



def PerformRmsOverlay(crd,rcrd,wts):
    """Rotates and translates crds to overlay with rcrds

    Parameters
    ----------
    crd : numpy.array, shape=(nat,3)
        The coordinates to modify

    rcrd : numpy.array, shape=(nat,3)
        The reference coordinates

    wts : numpy.array, shape=(nat,)
        The weights (usually masses)

    Returns
    -------
    rmsPD : float
        The coordinate difference root mean square

    ocrd : numpy.array, shape=(nat,3)
        The output coordinates
    """
    
    import numpy as np
    rms,rot,com,refcom = CptRmsTransform(crd,rcrd,wts)
    return rms,np.dot(crd-com,rot)+refcom





def _CptPucker(c1,c2,c3,c4,o4):
    """Returns the pucker coordinates of a 5-membered ring

    Parameters
    ----------
    c1 : numpy.array, shape=(3,)
        A point in space

    c2 : numpy.array, shape=(3,)
        A point in space

    c3 : numpy.array, shape=(3,)
        A point in space

    c4 : numpy.array, shape=(3,)
        A point in space

    o4 : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    tuple : (float,float)
        The Zx and Zy components of the pucker (eqs. 1-2 from 10.1021/ct401013s)
    """
    import numpy as np

    den1 = 2 * np.cos( 4*np.pi / 5 )
    den2 = 2 * np.sin( 4*np.pi / 5 )
    v1 = CptDihed(o4,c1,c2,c3)
    v3 = CptDihed(c2,c3,c4,o4)
    Zx = (v1+v3)/den1
    Zy = (v1-v3)/den2
    return (Zx,Zy)



def _CptPuckerAndGrd(c1,c2,c3,c4,o4):
    """Returns the pucker coordinates and gradients of a 5-membered ring


    Parameters
    ----------
    c1 : numpy.array, shape=(3,)
        A point in space

    c2 : numpy.array, shape=(3,)
        A point in space

    c3 : numpy.array, shape=(3,)
        A point in space

    c4 : numpy.array, shape=(3,)
        A point in space

    o4 : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    p : (float,float)
        The Zx and Zy components of the pucker (eqs. 1-2 from 10.1021/ct401013s)
    
    dpdc1 : np.ndarray, shape=(2,3)
        The derivative of p with respect to c1

    dpdc2 : np.ndarray, shape=(2,3)
        The derivative of p with respect to c2

    dpdc3 : np.ndarray, shape=(2,3)
        The derivative of p with respect to c3

    dpdc4 : np.ndarray, shape=(2,3)
        The derivative of p with respect to c4

    dpdo4 : np.ndarray, shape=(2,3)
        The derivative of p with respect to o4
    """
    import numpy as np

    den1 = 2 * np.cos( 4*np.pi / 5 )
    den2 = 2 * np.sin( 4*np.pi / 5 )
    v1,dv1do4,dv1dc1,dv1dc2,dv1dc3 = CptDihedAndGrd(o4,c1,c2,c3)
    v3,dv3dc2,dv3dc3,dv3dc4,dv3do4 = CptDihedAndGrd(c2,c3,c4,o4)
    Zx = (v1+v3)/den1
    Zy = (v1-v3)/den2
    
    dpdc1 = np.zeros( (2,3) )
    dpdc2 = np.zeros( (2,3) )
    dpdc3 = np.zeros( (2,3) )
    dpdc4 = np.zeros( (2,3) )
    dpdo4 = np.zeros( (2,3) )

    dpdc1[0,:] = (dv1dc1[:])/den1
    dpdc1[1,:] = (dv1dc1[:])/den2

    dpdc2[0,:] = (dv1dc2[:]+dv3dc2[:])/den1
    dpdc2[1,:] = (dv1dc2[:]-dv3dc2[:])/den2

    dpdc3[0,:] = (dv1dc3[:]+dv3dc3[:])/den1
    dpdc3[1,:] = (dv1dc3[:]-dv3dc3[:])/den2

    dpdc4[0,:] = ( dv3dc4[:])/den1
    dpdc4[1,:] = (-dv3dc4[:])/den2

    dpdo4[0,:] = (dv1do4[:]+dv3do4[:])/den1
    dpdo4[1,:] = (dv1do4[:]-dv3do4[:])/den2

    return (Zx,Zy),dpdc1,dpdc2,dpdc3,dpdc4,dpdo4







def CptPuckerX(c1,c2,c3,c4,o4):
    """Returns the Zx pucker coordinate of a 5-membered ring

    Parameters
    ----------
    c1 : numpy.array, shape=(3,)
        A point in space

    c2 : numpy.array, shape=(3,)
        A point in space

    c3 : numpy.array, shape=(3,)
        A point in space

    c4 : numpy.array, shape=(3,)
        A point in space

    o4 : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    tuple : float
        The Zx components of the pucker (eqs. 1 from 10.1021/ct401013s)
    """
    import numpy as np
    zz=_CptPucker(c1,c2,c3,c4,o4)
    return zz[0]



def CptPuckerY(c1,c2,c3,c4,o4):
    """Returns the Zx pucker coordinate of a 5-membered ring

    Parameters
    ----------
    c1 : numpy.array, shape=(3,)
        A point in space

    c2 : numpy.array, shape=(3,)
        A point in space

    c3 : numpy.array, shape=(3,)
        A point in space

    c4 : numpy.array, shape=(3,)
        A point in space

    o4 : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    tuple : float
        The Zy components of the pucker (eqs. 2 from 10.1021/ct401013s)
    """
    import numpy as np
    zz=_CptPucker(c1,c2,c3,c4,o4)
    return zz[1]




def CptPuckerXAndGrd(c1,c2,c3,c4,o4):
    """Returns the Zx pucker coordinate and gradients of a 5-membered ring

    Parameters
    ----------
    c1 : numpy.array, shape=(3,)
        A point in space

    c2 : numpy.array, shape=(3,)
        A point in space

    c3 : numpy.array, shape=(3,)
        A point in space

    c4 : numpy.array, shape=(3,)
        A point in space

    o4 : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    p : float
        The Zx component of the pucker (eqs. 1 from 10.1021/ct401013s)
    
    dpdc1 : np.ndarray, shape=(3,)
        The derivative of p with respect to c1

    dpdc2 : np.ndarray, shape=(3,)
        The derivative of p with respect to c2

    dpdc3 : np.ndarray, shape=(3,)
        The derivative of p with respect to c3

    dpdc4 : np.ndarray, shape=(3,)
        The derivative of p with respect to c4

    dpdo4 : np.ndarray, shape=(3,)
        The derivative of p with respect to o4
    """

    zz,dpdc1,dpdc2,dpdc3,dpdc4,dpdo4 = _CptPuckerAndGrd(c1,c2,c3,c4,o4)
    return zz[0],dpdc1[0,:],dpdc2[0,:],dpdc3[0,:],dpdc4[0,:],dpdo4[0,:]







def CptPuckerYAndGrd(c1,c2,c3,c4,o4):
    """Returns the Zy pucker coordinate and gradients of a 5-membered ring

    Parameters
    ----------
    c1 : numpy.array, shape=(3,)
        A point in space

    c2 : numpy.array, shape=(3,)
        A point in space

    c3 : numpy.array, shape=(3,)
        A point in space

    c4 : numpy.array, shape=(3,)
        A point in space

    o4 : numpy.array, shape=(3,)
        A point in space

    Returns
    -------
    p : float
        The Zy component of the pucker (eqs. 2 from 10.1021/ct401013s)
    
    dpdc1 : np.ndarray, shape=(3,)
        The derivative of p with respect to c1

    dpdc2 : np.ndarray, shape=(3,)
        The derivative of p with respect to c2

    dpdc3 : np.ndarray, shape=(3,)
        The derivative of p with respect to c3

    dpdc4 : np.ndarray, shape=(3,)
        The derivative of p with respect to c4

    dpdo4 : np.ndarray, shape=(3,)
        The derivative of p with respect to o4
    """

    zz,dpdc1,dpdc2,dpdc3,dpdc4,dpdo4 = _CptPuckerAndGrd(c1,c2,c3,c4,o4)
    return zz[1],dpdc1[1,:],dpdc2[1,:],dpdc3[1,:],dpdc4[1,:],dpdo4[1,:]


def Pucker2Dihedrals(zx,zy):
    import numpy as np
    c1 = 0.5/np.cos(4*np.pi/5)
    c2 = 0.5/np.sin(4*np.pi/5)
    sq5 = np.sqrt(5)
    c3 = -0.25*(1+sq5)
    c4 = np.sqrt( (5-sq5)/8 )
    A    = [ [ c1, c1 ], [ c2, -c2 ] ]
    Ainv = [ [ c3, c4 ], [ c3, -c4 ] ]
    return Ainv[0][0]*zx + Ainv[0][1]*zy, \
        Ainv[1][0]*zx + Ainv[1][1]*zy

def Dihedrals2Pucker(v1,v3):
    import numpy as np
    c1 = 0.5/np.cos(4*np.pi/5)
    c2 = 0.5/np.sin(4*np.pi/5)
    sq5 = np.sqrt(5)
    c3 = -0.25*(1+sq5)
    c4 = np.sqrt( (5-sq5)/8 )
    A    = [ [ c1, c1 ], [ c2, -c2 ] ]
    Ainv = [ [ c3, c4 ], [ c3, -c4 ] ]
    return A[0][0]*v1 + A[0][1]*v3, \
        A[1][0]*v1 + A[1][1]*v3





