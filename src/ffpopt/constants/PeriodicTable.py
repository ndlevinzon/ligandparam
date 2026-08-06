#! /usr/bin/env python3

def GetAtomicNumber(atomicSymbol):
    """Return the atomic number given an atomic symbol.

    Parameters
    ----------
    atomicSymbol : string
        The 1- or 2-character string of the element symbol with leading capitalization

    Returns
    -------
    z : int
        Atomic number (returns 0 if the symbol was not found)
    """
    PeriodicTable = {"H"  :   1,                                                                                                                                                                                                 "He" :   2, 
                     "Li" :   3, "Be" :   4,                                                                                                                         "B"  :   5, "C"  :   6, "N"  :   7, "O"  :   8, "F"  :   9, "Ne" :  10, 
                     "Na" :  11, "Mg" :  12,                                                                                                                         "Al" :  13, "Si" :  14, "P"  :  15, "S"  :  16, "Cl" :  17, "Ar" :  18, 
                     "K"  :  19, "Ca" :  20, "Sc" :  21, "Ti" :  22, "V"  :  23, "Cr" :  24, "Mn" :  25, "Fe" :  26, "Co" :  27, "Ni" :  28, "Cu" :  29, "Zn" :  30, "Ga" :  31, "Ge" :  32, "As" :  33, "Se" :  34, "Br" :  35, "Kr" :  36,
                     "Rb" :  37, "Sr" :  38, "Y"  :  39, "Zr" :  40, "Nb" :  41, "Mo" :  42, "Tc" :  43, "Ru" :  44, "Rh" :  45, "Pd" :  46, "Ag" :  47, "Cd" :  48, "In" :  49, "Sn" :  50, "Sb" :  51, "Te" :  52, "I"  :  53, "Xe" :  54,
                     "Cs" :  55, "Ba" :  56, "La" :  57
                     }

    res = 0
    if atomicSymbol in PeriodicTable:
        res = PeriodicTable[atomicSymbol]
    return res

    
    return res

def GetAtomicSymbol(atomicNumber):
    """Return the atomic symbol given an atomic number.  Symbols not on the 
    periodic table are given the dummy value "XX"

    Parameters
    ----------
    atomicNumber : int
        Atomic number

    Returns
    -------
    atomicSymbol : string
        The 1- or 2-character string of the element symbol with leading capitalization

    """
    ReversePeriodicTable = { 0: "XX",
                             1   : "H" ,                                                                                                                                                                                                 2   : "He",
                             3   : "Li", 4   : "Be",                                                                                                                         5   : "B" , 6   : "C" , 7   : "N" , 8   : "O" ,  9  : "F" , 10  : "Ne",
                             11  : "Na", 12  : "Mg",                                                                                                                         13  : "Al", 14  : "Si", 15  : "P" , 16  : "S" , 17  : "Cl", 18  : "Ar",
                             19  : "K" , 20  : "Ca", 21  : "Sc", 22  : "Ti", 23  : "V" , 24  : "Cr", 25  : "Mn", 26  : "Fe", 27  : "Co", 28  : "Ni", 29  : "Cu", 30  : "Zn", 31  : "Ga", 32  : "Ge", 33  : "As", 34  : "Se", 35  : "Br", 36  : "Kr",
                             37  : "Rb", 38  : "Sr", 39  : "Y" , 40  : "Zr", 41  : "Nb", 42  : "Mo", 43  : "Tc", 44  : "Ru", 45  : "Rh", 46  : "Pd", 47  : "Ag", 48  : "Cd", 49  : "In", 50  : "Sn", 51  : "Sb", 52  : "Te", 53  : "I" , 54  : "Xe",
                             55  : "Cs", 56  : "Ba", 57  : "La"
                             }
    
    atomicSymbol = "XX"
    if atomicNumber in ReversePeriodicTable:
        atomicSymbol = ReversePeriodicTable[atomicNumber]

    return atomicSymbol

def GetAtomicMass(atomicNumber):
    """Return the naturally occuring atomic mass. Unrecognized values will give a mass of 0.
    
    Parameters
    ----------
    atomicNumber : int
        Atomic number

    Returns
    -------
    mass : float
        Mass in atomic mass units (g/mol)
    """
 
    AtomicMasses = { 1   :   1.008,                                                                                                                                                                                                                                                 2   :   4.003,
                     3   :   6.941, 4   :   9.012,                                                                                                                                                       5   :  10.811, 6   :  12.011, 7   :  14.007, 8   :  15.999,  9  :  18.998, 10  :  20.180,
                     11  :  22.990, 12  :  24.305,                                                                                                                                                       13  :  26.982, 14  :  28.086, 15  :  30.974, 16  :  32.066, 17  :  35.453, 18  :  39.948,
                     19  :  39.098, 20  :  40.078, 21  :  44.956, 22  :  47.880, 23  :  50.942, 24  :  51.996, 25  :  54.938, 26  :  55.847, 27  :  58.933, 28  :  58.693, 29  :  63.546, 30  :  35.390, 31  :  69.723, 32  :  72.610, 33  :  74.921, 34  :  78.960, 35  :  79.904, 36  :  83.800,
                     37  :  86.468, 38  :  87.620, 39  :  88.906, 40  :  91.224, 41  :  92.906, 42  :  95.940, 43  :  97.907, 44  : 101.070, 45  : 102.906, 46  : 106.420, 47  : 107.868, 48  : 112.411, 49  : 114.818, 50  : 118.710, 51  : 121.757, 52  : 127.600, 53  : 126.904, 54  : 131.290,
                     55  : 132.905, 56  : 137.327, 57  : 138.906
                     }
    
    mass = 0.
    if atomicNumber in AtomicMasses:
        mass = AtomicMasses[atomicNumber]

    return mass

    

def GetStdIsotopeMass(atomicNumber):
    """Return the standard isotope mass
    
    Parameters
    ----------
    atomicNumber : int
        Atomic number

    Returns
    -------
    mass : float
        Mass in atomic mass units (g/mol)
    """


    StdIsotopeMass = [   0.0,  1.0078250,   4.0026000,   7.0160000,   9.0121800,  11.0093100,  12.0000000,  14.0030700,  15.9949100,  18.9984000,  19.9924400,  22.9898000,  23.9850400,  26.9815300,  27.9769300,  30.9937600,  31.9720700,  34.9688500,  39.9627200,  38.9637100,  39.9625900,  44.9559200,  47.9000000,  50.9440000,  51.9405000,  54.9381000,  55.9349000,  58.9332000,  57.9353000,  62.9298000,  63.9291000,  68.9257000,  73.9219000,  74.9216000,  79.9165000,  78.9183000,  83.8000000,  84.9117000,  87.9056000,  88.9054000,  89.9043000,  92.9060000,  97.9055000,  98.9062000, 101.9037000, 102.9048000, 105.9032000, 106.9041000, 113.9036000, 114.9041000, 117.9018000, 120.9038000, 129.9067000, 126.9004000, 131.9042000 ]

    mass = 0
    if atomicNumber < len(StdIsotopeMass):
        mass = StdIsotopeMass[atomicNumber]
    
    return mass


def GetSecondIsotopeMass(atomicNumber):
    """Return the second isotope mass
    
    Parameters
    ----------
    atomicNumber : int
        Atomic number

    Returns
    -------
    mass : float
        Mass in atomic mass units (g/mol)
    """


    SecondIsotopeMass = [ 0.0,  2.0141020,   3.0160290,   6.0151220,   0.0000000,  10.0129370,  13.0033550,  15.0001090,  17.9991600,   0.0000000,  21.9913860,   0.0000000,  25.9825930,   0.0000000,  28.9764950,   0.0000000,  33.9678670,  36.9659030,  35.9675460,  40.9618260,  43.9554810,   0.0000000,  45.9526300,  49.9471630,  52.9406540,   0.0000000,  53.9396150,   0.0000000,  59.9307910,  64.9277940,  65.9260370,  70.9247050,  71.9220760,   0.0000000,  77.9173100,  80.9162910,  85.9106100,  86.9091840,  85.9092620,   0.0000000,  93.9063160,   0.0000000,  95.9046790,  97.9072160, 103.9054300,   0.0000000, 107.9038940, 108.9047560, 111.9027570, 112.9040610, 117.9016060, 122.9042160, 127.9044610,   0.0000000, 128.9047800 ]

    mass = 0
    if atomicNumber < len(SecondIsotopeMass):
        mass = SecondIsotopeMass[atomicNumber]
        
    return mass

def GetCovalentRadius(atomicNumber):
    from . Conversions import AU_PER_ANGSTROM
    CovalentRadiusData = [   0.0,  0.3700000,   0.3200000,   1.3400000,   0.9000000,   0.8200000,   0.7700000,   0.7500000,   0.7300000,   0.7100000,   0.6900000,   1.5400000,   1.3000000,   1.1800000,   1.1100000,   1.0600000,   1.0200000,   0.9900000,   0.9700000,   1.9600000,   1.7400000,   1.4400000,   1.3600000,   1.2500000,   1.2700000,   1.3900000,   1.2500000,   1.2600000,   1.2100000,   1.3800000,   1.3100000,   1.2600000,   1.2200000,   1.1900000,   1.1600000,   1.1400000,   1.1000000,   2.1100000,   1.9200000,   1.6200000,   1.4800000,   1.3700000,   1.4500000,   1.5600000,   1.2600000,   1.3500000,   1.3100000,   1.5300000,   1.4800000,   1.4400000,   1.4100000,   1.3800000,   1.3500000,   1.3300000,   1.3000000 ]
     
    x = 0
    if atomicNumber < len(CovalentRadiusData):
        x = CovalentRadiusData[atomicNumber] * AU_PER_ANGSTROM()
    return x


def GetvdWRadius(atomicNumber):
    from . Conversions import AU_PER_ANGSTROM

    vdWRadiusData = [   0.0,  1.2000000,   1.4000000,   1.8200000,   0.0000000,   0.0000000,   1.7000000,   1.5500000,   1.5200000,   1.4700000,   1.5400000,   2.2700000,   1.7300000,   0.0000000,   2.1000000,   1.8000000,   1.8000000,   1.7500000,   1.8800000,   2.7500000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   1.6300000,   1.4000000,   1.3900000,   1.8700000,   0.0000000,   1.8500000,   1.9000000,   1.8500000,   2.0200000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   0.0000000,   1.6300000,   1.7200000,   1.5800000,   1.9300000,   2.1700000,   0.0000000,   2.0600000,   1.9800000,   2.1600000 ]

    x = 0
    if atomicNumber < len(vdWRadiusData):
        x = vdWRadiusData[atomicNumber] * AU_PER_ANGSTROM()
    return x



def GetElectronegativity(atomicNumber):
    #
    # * \note Result is in atomic units (Hartree)
    # * \note Noble gases: Klaus S. Lackner and George Zweig, Phys. Rev. D, 28:1671-1691, 1983.
    # * \note All others: Pearson, Inorg. Chem. 27:734-740, 1988.
    # * \note Data for Tc are currently missing.
    #
    
    elecneg = [ 0.0, 0.2638740,   0.4479860,   0.1106210,   0.1800810,   0.1576630,   0.2304300,   0.2682840,   0.2771040,   0.3825800,   0.3905280,   0.1047410,   0.1378170,   0.1187060,   0.1753030,   0.2065420,   0.2285920,   0.3050350,   0.2819480,   0.0889380,   0.0808530,   0.1227490,   0.1267920,   0.1323040,   0.1367140,   0.1367140,   0.1492100,   0.1580300,   0.1617050,   0.1646450,   0.1635430,   0.1176040,   0.1690550,   0.1947810,   0.2164650,   0.2789420,   0.2496070,   0.0859980,   0.0735020,   0.1172360,   0.1337740,   0.1470050,   0.1433300,   0.0000000,   0.1653800,   0.1580300,   0.1635430,   0.1631750,   0.1591330,   0.1139290,   0.1580300,   0.1782430,   0.2017640,   0.2484380,   0.2152490 ]
    
    x=0
    if atomicNumber < len(elecneg):
        x = elecneg[atomicNumber]
    return x



def GetHardness(atomicNumber):
    #
    # * \note Result is in atomic units (Hartree)
    # * \note Noble gases: Klaus S. Lackner and George Zweig, Phys. Rev. D, 28:1671-1691, 1983.
    # * \note All others: Pearson, Inorg. Chem. 27:734-740, 1988.
    # * \note Data for Tc are currently missing.
    #
    
    hardness = [ 0.0, 0.2363100,   0.4556300,   0.0878350,   0.1653800,   0.1473720,   0.1837560,   0.2657110,   0.2234470,   0.2576260,   0.4019940,   0.0845280,   0.1433300,   0.1018010,   0.1242190,   0.1793460,   0.1521500,   0.1719960,   0.2972370,   0.0705620,   0.1470050,   0.1176040,   0.1238520,   0.1139290,   0.1124590,   0.1367140,   0.1400220,   0.1323040,   0.1194410,   0.1194410,   0.1815510,   0.1065780,   0.1249540,   0.1653800,   0.1422270,   0.1550900,   0.2648950,   0.0679900,   0.1359790,   0.1172360,   0.1179710,   0.1102540,   0.1139290,   0.0000000,   0.1102540,   0.1161330,   0.1429620,   0.1153990,   0.1712610,   0.1029030,   0.1120910,   0.1396550,   0.1293640,   0.1356120,   0.2305380 ]
    
    x=0
    if atomicNumber < len(hardness):
        x = hardness[atomicNumber]
    return x



def GetRespRadius(atomicNumber):

    from . Conversions import AU_PER_ANGSTROM
    
    MSK_RADII = {  # from FMOPRP in GAMESS fmoio.src
        'H': 1.20,                                                                      'He': 1.20,
        'Li': 1.37, 'Be': 1.45,  'B': 1.45,  'C': 1.50, 'N': 1.50, 'O': 1.40, 'F': 1.35, 'Ne': 1.30,
        'Na': 1.57, 'Mg': 1.36, 'Al': 1.24, 'Si': 1.17, 'P': 1.80, 'S': 1.75, 'Cl': 1.70
    }
    
    #: Bondi, A. van der Waals Volumes and Radii. J. Phys. Chem. 68, 441–451 (1964).
    BONDI_RADII = {
        'H': 1.20,                                                                         'He': 1.40,
        'Li': 1.82,                          'C': 1.70,  'N': 1.55,  'O': 1.52,  'F': 1.47, 'Ne': 1.54,
        'Na': 2.27, 'Mg': 1.73,             'Si': 2.10,  'P': 1.80,  'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
        'K': 1.75,             'Ga': 1.87,             'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
        'In': 1.93, 'Sn': 2.17,             'Te': 2.06,  'I': 1.98, 'Xe': 2.16,
        'Tl': 1.96, 'Pb': 2.02,
        # other metals
        'Ni': 1.63, 'Cu': 1.4, 'Zn': 1.39,
        'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58,
        'Pt': 1.72, 'Au': 1.66, 'Hg': 1.55,
        'U': 1.86,
    }
    
    #: Mantina, M., Chamberlin, A. C., Valero, R., Cramer, C. J. & Truhlar, D. G. Consistent van der Waals Radii for the Whole Main Group. J Phys Chem A 113, 5806–5812 (2009).
    MANTINA_RADII = {
        'H': 1.10,                                                                         'He': 1.40,
        'Li': 1.82, 'Be': 1.53,  'B': 1.92,  'C': 1.70,  'N': 1.55,  'O': 1.52,  'F': 1.47, 'Ne': 1.54,
        'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10,  'P': 1.80,  'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
        'K': 1.75, 'Ca': 2.31, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
        'Rb': 3.03, 'Sr': 2.49, 'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06,  'I': 1.98, 'Xe': 2.16,
        'Cs': 3.43, 'Ba': 2.68, 'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.20,
        'Fr': 3.48, 'Ra': 2.83,
        # other metals
        'Ni': 1.63, 'Cu': 1.4, 'Zn': 1.39,
        'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58,
        'Pt': 1.72, 'Au': 1.66, 'Hg': 1.55,
        'U': 1.86,
    }
    
    HMANTINA_RADII = {  # H  is still from BONDI radii; follows GAMESS
        'H': 1.20,                                                                         'He': 1.40,
        'Li': 1.82, 'Be': 1.53,  'B': 1.92,  'C': 1.70,  'N': 1.55,  'O': 1.52,  'F': 1.47, 'Ne': 1.54,
        'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10,  'P': 1.80,  'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
        'K': 1.75, 'Ca': 2.31, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
        'Rb': 3.03, 'Sr': 2.49, 'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06,  'I': 1.98, 'Xe': 2.16,
        'Cs': 3.43, 'Ba': 2.68, 'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.20,
        'Fr': 3.48, 'Ra': 2.83,
        # other metals
        'Ni': 1.63, 'Cu': 1.4, 'Zn': 1.39,
        'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58,
        'Pt': 1.72, 'Au': 1.66, 'Hg': 1.55,
        'U': 1.86,
    }
    
    #: Alvarez, S. A cartography of the van der Waals territories. Dalton Transactions 42, 8617–8636 (2013).
    ALVAREZ_RADII = {
        'H': 1.20,                                                                         'He': 1.43,
        'Li': 2.12, 'Be': 1.98,  'B': 1.91,  'C': 1.77,  'N': 1.66,  'O': 1.50,  'F': 1.46, 'Ne': 1.58,
        'Na': 2.50, 'Mg': 2.51, 'Al': 2.25, 'Si': 2.19,  'P': 1.90,  'S': 1.89, 'Cl': 1.82, 'Ar': 1.83,
        'K': 2.73, 'Ca': 2.62, 'Ga': 2.32, 'Ge': 2.29, 'As': 1.88, 'Se': 1.82, 'Br': 1.86, 'Kr': 2.25,
        'Rb': 3.21, 'Sr': 2.84, 'In': 2.43, 'Sn': 2.42, 'Sb': 2.47, 'Te': 1.99,  'I': 2.04, 'Xe': 2.06,
        'Cs': 3.48, 'Ba': 3.03, 'Tl': 2.47, 'Pb': 2.60, 'Bi': 2.54,
        # other metals
        'Sc': 2.58, 'Ti': 2.46,  'V': 2.42, 'Cr': 2.45, 'Mn': 2.45, 'Fe': 2.44, 'Co': 2.40, 'Ni': 2.40, 'Cu': 2.38, 'Zn': 2.39,
        'Y': 2.75, 'Zr': 2.52, 'Nb': 2.56, 'Mo': 2.45, 'Tc': 2.44, 'Ru': 2.46, 'Rh': 2.44, 'Pd': 2.15, 'Ag': 2.53, 'Cd': 2.49,
        'La': 2.98, 'Ce': 2.88, 'Pr': 2.92, 'Nd': 2.95, 'Sm': 2.90, 'Eu': 2.87, 'Gd': 2.83,
        'Tb': 2.79, 'Dy': 2.87, 'Ho': 2.81, 'Er': 2.83, 'Tm': 2.79, 'Yb': 2.80, 'Lu': 2.74,
        'Hf': 2.63, 'Ta': 2.53,  'W': 2.57, 'Re': 2.49, 'Os': 2.48, 'Ir': 2.41, 'Pt': 2.29, 'Au': 2.32, 'Hg': 2.45,
        'Ac': 2.8, 'Th': 2.93, 'Pa': 2.88,  'U': 2.71, 'Np': 2.82, 'Pu': 2.81, 'Am': 2.83,
        'Cm': 3.05, 'Bk': 3.4, 'Cf': 3.05, 'Es': 2.7,
    }

    ele = GetAtomicSymbol(atomicNumber)
    rad = 2.0
    if ele in MSK_RADII:
        rad = MSK_RADII[ele]
    elif ele in ALVAREZ_RADII:
        rad = ALVAREZ_RADII[ele]
    elif ele in HMANTINA_RADII:
        rad = HMANTINA_RADII[ele]
    elif ele in MANTINA_RADII:
        rad = MANTINA_RADII[ele]
    elif ele in BONDI_RADII:
        rad = BONDI_RADII[ele]
    return rad * AU_PER_ANGSTROM()



def GetUffRadius(z):
    #
    # Returns radius in Angstrom
    #
    from ndfes.constants import AU_PER_ANGSTROM
    diam = [ 4.0, #// dummy
             2.886, 2.362, #// 1s (2)
             2.451, 2.745, #// 2s (4)
             4.083, 3.851, 3.660, 3.500, 3.364, 3.243, #// 2p (10)
             2.983, 3.021, #// 3s (12)
             4.499, 4.295, 4.147, 4.035, 3.947, 3.868, #// 3p (18)
             3.812, 3.399, #// 4s (20)
             3.295, 3.175, 3.144, 3.023, 2.961, #// 3d
             2.912, 2.872, 2.834, 3.495, 2.763, #// 3d (30)
             4.383, 4.28, 4.23, 4.205, 4.189, 4.141, #// 4p  (36)
             4.114, 3.641, #// 5s (38)
             3.345, 3.124, 3.165, 3.052, 2.998, 
             2.963, 2.929, 2.899, 3.148, 2.848, #// 4d (48)
             4.463, 4.392, 4.42, 4.47, 4.5, 4.404 ] #// 5p (54)

    x = None
    if z >= 0 and z < len(diam):
        x = AU_PER_ANGSTROM() * 0.5 * diam[z]
    else:
        raise Exception(f"Invalid atomic number {z}")
    return x

