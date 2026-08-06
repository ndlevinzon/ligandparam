#!/usr/bin/env python3


class CosmoAtom(object):
    def __init__(self,crd,q,quad_rule,rad):
        import numpy as np
        self.crd = np.array(crd,copy=True)
        self.grd = np.zeros( (3,) )
        self.q = q
        self.p = 0
        self.nquad = quad_rule
        self.radius = rad
        self.inner_radius = 0
        self.outter_radius = 0
        self.elem_begin = 0
        self.elem_end = 0
        self.aidx = 0
        
        
