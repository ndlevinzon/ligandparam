#!/usr/bin/env python3

class AbInitioOptions(object):
    def __init__(self,program="psi4",theory="hf/6-31g*",charge=0,mult=1,mem="1gb",nproc=4):
        self.program = program
        self.theory  = theory
        self.charge  = charge
        self.mult    = mult
        self.mem     = mem
        self.nproc   = nproc

        
