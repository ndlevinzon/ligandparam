#!/usr/bin/env python3


class CosmoElement(object):
    def __init__(self,crd,atidx,zeta,swt,qwt,rwt):
        import numpy as np
        self.crd = np.array(crd,copy=True)
        self.grd = np.zeros( (3,) )
        self.q = 0
        self.p = 0
        self.atidx = atidx
        self.zeta = zeta
        self.switchwt = swt
        self.quadwt = qwt
        self.radwt = rwt
        self.aidx = 0
