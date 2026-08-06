#1/usr/bin/env python3



def SwitchOff(r,rlo,rhi):
    s=1.
    if r >= rhi:
        s=0.
    elif r <= rlo:
        s=1.
    else:
        u=(rhi-r)/(rhi-rlo)
        u3 = u*u*u
        u4 = u3*u
        u5 = u4*u
        s = 10.*u3 - 15.*u4 + 6.*u5
    return s



def SwitchOffGrd(r,rlo,rhi):
    s = 1.
    dsdr = 0.
    if r >= rhi:
        s = 0.
        dsdr = 0.
    elif r <= rlo:
        s = 1.
        dsdr = 0.
    else:
        u = (rhi-r)/(rhi-rlo)
        u2 = u*u
        u3 = u2*u
        u4 = u3*u
        u5 = u4*u
        dudr = -1. / (rhi-rlo)
        s  =  10.*u3 - 15.*u4 +  6.*u5
        dsdu = (30.*u2 - 60.*u3 + 30.*u4)
        dsdr = dsdu * dudr
    return s,dsdr



def SwitchOffAllGrd(r,rlo,rhi):
    s = 1.
    dsdr  = 0.
    dsdlo = 0.
    dsdhi = 0.
    if r >= rhi:
        s = 0.
    elif r <= rlo:
        s = 1.
    else:
        u = (rhi-r)/(rhi-rlo)
        u2 = u*u
        u3 = u2*u
        u4 = u3*u
        u5 = u4*u
        dudr = -1. / (rhi-rlo)
        dudlo = - u * dudr
        dudhi = -dudr-dudlo
        
        s  =  10.*u3 - 15.*u4 +  6.*u5
        dsdu = (30.*u2 - 60.*u3 + 30.*u4)
        dsdr  = dsdu * dudr
        dsdlo = dsdu * dudlo
        dsdhi = dsdu * dudhi
    return s,dsdr,dsdlo,dsdhi




def SwitchOn(r,rlo,rhi):
    s = SwitchOff(r,rlo,rhi)
    return 1-s



def SwitchOnGrd(r,rlo,rhi):
    s,dsdr = SwitchOffGrd(r,rlo,rhi)
    return 1-s,-dsdr



def SwitchOnAllGrd(r,rlo,rhi):
    s,dsdr,dsdlo,dsdhi = SwitchOffAllGrd(r,rlo,rhi)
    return 1-s,-dsdr,-dsdlo,-dsdhi


