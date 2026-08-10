#!/usr/bin/env python3
"""
Implements the York & Karplus Smooth Conductor-like Solvation Model (SCOSMO)

Brief summary of functions
--------------------------
GetUffRadius(z) -> r
    Given atomic number (int), return UFF Radius (Bohr)

GetLebedevRule(npts) -> pts,wts
    Given number of pts, return the Lebedev positions and weights

LebedevGaussianZetaScaleFactor(npts) -> sclf
    Given number of pts, return surface Gaussian exponent scale factor

SwitchOff(r,rlo,rhi) -> s
SwitchOffGrd(r,rlo,rhi) -> s,dsdr
SwitchOffAllGrd(r,rlo,rhi) -> s,dsdr,dsdrlo,dsdrhi
    Get a value that is s=1 if r <= rlo and r=0 if r >= rhi

SwitchOn(r,rlo,rhi) -> s
SwitchOnGrd(r,rlo,rhi) -> s,dsdr
SwitchOnAllGrd(r,rlo,rhi) -> s,dsdr,dsdrlo,dsdrhi
    Get a value that is s=0 if r <= rlo and r=1 if r >= rhi


Brief summary of classes
------------------------
CosmoElement
    Stores information about a COSMO surface Gaussian

CosmoAtom
    Stores information about an atom which causes a response

CosmoSurface
    Stores collections of CosmoElement's and CosmoAtom's

"""

from . CosmoAtom import CosmoAtom
from . CosmoElement import CosmoElement
from . CosmoSurface import CosmoSurface
from . Lebedev import GetLebedevRule
from . Lebedev import GetLebedevDegreeMatchingDensity
from . Lebedev import GetLebedevValidDegrees
from . Lebedev import LebedevGaussianZetaScaleFactor
from ffpopt.constants.PeriodicTable import GetUffRadius
from . SwitchFcn import SwitchOff
from . SwitchFcn import SwitchOffGrd
from . SwitchFcn import SwitchOffAllGrd
from . SwitchFcn import SwitchOn
from . SwitchFcn import SwitchOnGrd
from . SwitchFcn import SwitchOnAllGrd

__all__ = [ 'CosmoAtom',
            'CosmoElement',
            'CosmoSurface',
            'GetLebedevRule',
            'GetLebedevDegreeMatchingDensity',
            'GetLebedevValidDegrees',
            'LebedevGaussianZetaScaleFactor',
            'GetUffRadius',
            'SwitchOff',
            'SwitchOffGrd',
            'SwitchOffAllGrd',
            'SwitchOn',
            'SwitchOnGrd',
            'SwitchOnAllGrd']

