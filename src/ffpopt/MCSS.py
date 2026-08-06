#!/usr/bin/env python3

def mcss(mol2str_1, mol2str_2, maxtime=60, isotope_map=None, selec=''):
    """
    Maximum common substructure search via RDKit/fmcs.

    :param mol2str_1: first MOL2 string
    :type mol2str_1: string
    :param mol2str_2: second MOL2 string
    :type mol2str_2: string
    :param maxtime: timeout for fmcs in seconds
    :type maxtime: float
    :param isotope_map: explicit user atom mapping
    :type isotope_map: dict
    :param selec: selection method for multiple MCS
    :type selec: string
    :raises: SetupError
    :returns: index map
    :rtype: dict
    """

    from collections import OrderedDict, defaultdict
    import sys
    from rdkit import rdBase
    import rdkit.Chem


    _fmcs_imp = 'c++'                       # 'python' or 'c++'
    if _fmcs_imp == 'c++':
        from rdkit.Chem.rdFMCS import FindMCS, AtomCompare, BondCompare

        # RDKit 2015.03.1 FMCS C++ implentation, seems not to be exactly the
        # same implementation e.g. SMARTS string more specific
        # NOTE: some different parameters and order!
        #       matchChiralTag = False not implemented before 2015.03.1
        _params = dict(maximizeBonds = False, threshold = 1.0,
                       verbose = False, matchValences = False,
                       ringMatchesRingOnly = True, completeRingsOnly = True,
                       bondCompare = BondCompare.CompareAny)
    else:
        from rdkit.Chem.MCS import FindMCS

        # defaults are minNumAtoms = 2, maximize = 'bonds',
        # atomCompare = 'elements', bondCompare = 'bondtypes',
        # matchValences = False, ringMatchesRingOnly = False,
        # completeRingsOnly = False, timeout = None, threshold = None
        #
        # when completeRingsOnly = True also ringMatchesRingOnly = True
        #
        # if number of atoms in match < minNumAtoms then smarts will be None
        # completeRingsOnly = True disallows partial rings
        # MCS algorithm is exhaustive so use timeout to limit time
        _params = dict(minNumAtoms = 2, maximize = 'atoms', atomCompare = 'elements',
                       bondCompare = 'any', matchValences = False,
                       ringMatchesRingOnly = True, completeRingsOnly = True,
                       threshold = None)



    logger = sys.stderr
    # disable warning about no explicit hydrogens
    rdBase.DisableLog('rdApp.warning')
    mol1 = rdkit.Chem.MolFromMol2Block(mol2str_1, sanitize = False,
                                       removeHs = False)
    mol2 = rdkit.Chem.MolFromMol2Block(mol2str_2, sanitize = False,
                                       removeHs = False)
    rdBase.EnableLog('rdApp.warning')

    _params.update(timeout = int(maxtime) )

    # FIXME: test c++ implementation
    if isotope_map:
        if _fmcs_imp == 'c++':
            _params.update(atomCompare = AtomCompare.CompareIsotopes)
        else:
            _params.update(atomCompare = 'isotopes')

        max_idx1 = mol1.GetNumAtoms()
        max_idx2 = mol2.GetNumAtoms()

        icnt = 0

        # NOTE: would it make sense to have multiple atoms of a molecule tagged
        #       as the same isotope?
        for idx1, idx2 in isotope_map.items():
            icnt += 1

            logger.write('Mapping atom index %i to %i' % (idx1, idx2) )

            if idx2 < 1:
                atom1 = mol1.GetAtomWithIdx(idx1-1)
                atom1.SetIsotope(-1 * idx2 + icnt)
            else:
                if idx1 > max_idx1 or idx2 > max_idx2 or idx1 < 0 or idx2 < 0:
                    logger.write('Error: indices out of bounds (%i, %i)' %
                                 (max_idx1, max_idx2) )
                    raise Exception('Mapping indices out of bounds '
                                    '(%i, %i)' % (max_idx1, max_idx2) )

                # FIXME: guard against non-existing indices
                atom1 = mol1.GetAtomWithIdx(idx1-1)
                atom1.SetIsotope(icnt)

                atom2 = mol2.GetAtomWithIdx(idx2-1)
                atom2.SetIsotope(icnt)
    else:
        if _fmcs_imp == 'c++':
            _params.update(atomCompare = AtomCompare.CompareAny)
        else:
            _params.update(atomCompare = 'any')

        n_chiral1 = len(rdkit.Chem.FindMolChiralCenters(mol1) )
        n_chiral2 = len(rdkit.Chem.FindMolChiralCenters(mol2) )

        if n_chiral1 > 0:
            logger.write('Warning: state 0 has %i chiral center%s. Check if '
                         'configurations are inverted!'
                         % (n_chiral1, 's' if n_chiral1 > 1 else '') )
        if n_chiral2 > 0:
            logger.write('Warning: state 1 has %i chiral center%s. Check if '
                         'configurations are inverted!'
                         % (n_chiral2, 's' if n_chiral2 > 1 else '') )


    mcs = FindMCS( (mol1, mol2), **_params)

    if _fmcs_imp == 'c++':
        smarts = mcs.smartsString
        completed = not mcs.canceled
    else:
        smarts = mcs.smarts
        completed = mcs.completed

    logger.write('Running RDKit/fmcs (%s implementation) with arguments:\n%s\n' %
                 (_fmcs_imp,
                  ', '.join(['%s=%s' % (k,v) for k,v in _params.items()] ) ) )

    if not smarts:
        raise Exception('No MCSS match could be found')

    if not completed:
        logger.write('Warning: MCSS timed out after %.2fs' % maxtime)

    p = rdkit.Chem.MolFromSmarts(smarts)

    #conv = ob.OBConversion()
    #conv.SetInAndOutFormats('mol2', 'mol2')
    ## NOTE: this relies on a modified Openbabel MOL2 writer
    #conv.AddOption('r', ob.OBConversion.OUTOPTIONS)  # do not append resnum
    #obmol1 = ob.OBMol()
    #errlev = ob.obErrorLog.GetOutputLevel()
    #ob.obErrorLog.SetOutputLevel(0)
    #conv.ReadString(obmol1, mol2str_1)
    #ob.obErrorLog.SetOutputLevel(errlev)

    # NOTE: experimental!
    if selec == 'spatially-closest':
        m1 = mol1.GetSubstructMatches(p, uniquify=False, maxMatches=100, useChirality=False)
        m2 = mol2.GetSubstructMatches(p, uniquify=False, maxMatches=100, useChirality=False)

        logger.write('Applying spatially-closest algorithm (%s, %s matches)\n' %
                     (len(m1), len(m2) ) )


        # FIXME: is it possible that the smaller one has more then one matches
        #        when uniquify=True?
        if len(m1) < len(m2):
            m1, m2 = m2, m1
            conf1 = mol2.GetConformer()
            conf2 = mol1.GetConformer()
            swapped = True
        else:
            conf1 = mol1.GetConformer()
            conf2 = mol2.GetConformer()
            swapped = False

        # for x in range(m1)...
        #     match1 = m1[x]
        #     for y in range(m2)...
        #         match2 = m2[y]
        #         find sumd
        #         save x, y with smallest sumd
        mind = 999999.0
        minxy = [-1,-1]
        #DEVTHRESHOLD = 0.2**2# That's 0.2 Angstrom
        for x in range(len(m1)):
            match1 = m1[x]
            for y in range(len(m2)):
                match2 = m2[y]
                sumd = 0.0
                for i, idx1 in enumerate(match1):
                    pos1 = conf1.GetAtomPosition(idx1)
                    idx2 = match2[i]
                    pos2 = conf2.GetAtomPosition(idx2)
                    d2 = (pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 +\
                         (pos1.z - pos2.z)**2
                    sumd += d2
                #print (x,y,sumd,mind)
                if sumd < mind:
                    mind = sumd
                    minxy = [x,y]
        if swapped:
            mapping = dict(zip(m2[minxy[1]], m1[minxy[0]]))
        else:
            # JM bug 11/17 ? 
            #mapping = dict(zip(m1[minxy[1]], m2[minxy[0]]))
            mapping = dict(zip(m1[minxy[0]], m2[minxy[1]]))
    else:
        m1 = mol1.GetSubstructMatch(p)
        m2 = mol2.GetSubstructMatch(p)

        mapping = dict(zip(m1, m2) )

    # FIXME: we may have to reconsider this and understand when rings have
    #        to be assumed "broken"
    #
    # delete atoms from mapping that are also part of an map-external ring
    if False:
        ring_info1 = mol1.GetRingInfo()
        ring_info2 = mol2.GetRingInfo()

        rings1 = ring_info1.AtomRings()
        rings2 = ring_info2.AtomRings()

        map1 = set(mapping.keys())
        map2 = set(mapping.values())

        for ring in rings1:
            if not set(ring).issubset(map1):
                for idx in map1:
                    if idx in ring and ring_info1.NumAtomRings(idx) == 1:
                        del(mapping[idx])

        delete_values = []
        for ring in rings2:
            if not set(ring).issubset(map2):
                for idx in map2:
                    if idx in ring and ring_info2.NumAtomRings(idx) == 1:
                        delete_values.append(idx)

        mapping = {k: v for k, v in mapping.items() if v not in delete_values}

    return mapping




def FixParmedAtomicNumbers(mol):
    """Replace parmed atomic_numbers attribute for atom types matching gaff or gaff2

    Parameters
    ----------
    mol : parmed Structure or ResidueTemplate
        This object should contain an array mol.atoms with the .atomic_number attribute
    """
    
    # Example mapping for common GAFF2 atom types to atomic numbers
    # This is a simplified example; a more robust mapping might be needed for all GAFF2 types.
    gaff2_type_to_atomic_number = {
        'c': 6, 'c1': 6, 'c2': 6, 'c3': 6, 'ca': 6, 'cc': 6, 'cd': 6, 'ce': 6, 'cf': 6, 'cg': 6,
        'ch': 6, 'ci': 6, 'cj': 6, 'ck': 6, 'cl': 6, 'cm': 6, 'cn': 6, 'cp': 6, 'cq': 6, 'cr': 6,
        'cx': 6, 'cy': 6, 'cz': 6, 'n': 7, 'n1': 7, 'n2': 7, 'n3': 7, 'na': 7, 'nb': 7, 'nc': 7,
        'nd': 7, 'ne': 7, 'nf': 7, 'nh': 7, 'no': 7, 'nt': 7, 'o': 8, 'oh': 8, 'os': 8, 'ow': 8,
        'oy': 8, 'p': 15, 'pb': 15, 'pc': 15, 'pd': 15, 'pe': 15, 'pf': 15, 's': 16, 'sh': 16,
        'ss': 16, 'sx': 16, 'sy': 16, 'f': 9, 'cl': 17, 'br': 35, 'i': 53, 'h': 1, 'hc': 1, 'hp': 1,
        'hs': 1, 'hw': 1
    }

    gaff_type_to_atomic_number = {
        "c":6,"c1":6,"c2":6,"c3":6,"ca":6,"cp":6,"cq":6,"cc":6,"cd":6,"ce":6,"cf":6,"cg":6,"ch":6,"cx":6,"cy":6,"cu":6,"cv":6,"cz":6,"h1":1,"h2":1,"h3":1,"h4":1,"h5":1,"ha":1,"hc":1,"hn":1,"ho":1,"hp":1,"hs":1,"hw":1,"hx":1,"f":9, "cl":17,"br":35,"i":53, "n":7,  "n1":7, "n2":7, "n3":7, "n4":7, "na":7, "nb":7, "nc":7, "nd":7, "ne":7, "nf":7, "nh":7, "no":7, "ni":7, "nj":7, "nk":7, "nl":7, "nm":7, "nn":7, "np":7, "nq":7, "o":8,  "oh":8, "os":8, "op":8, "oq":8, "ow":8, "p2":15,"p3":15,"p4":15,"p5":15,"pb":15,"pc":15,"pd":15,"pe":15,"pf":15,"px":15,"py":15,"s":16, "s2":16,"s4":16,"s6":16,"sh":16,"ss":16,"sp":16,"sq":16,"sx":16,"sy":16
        }

    for atom in mol.atoms:
        if atom.type in gaff2_type_to_atomic_number:
            atom.atomic_number = gaff2_type_to_atomic_number[atom.type]
        elif atom.type in gaff_type_to_atomic_number:
            atom.atomic_number = gaff_type_to_atomic_number[atom.type]



def parmed2mol2str(parmed_atoms):
    import parmed
    import copy
    from io import StringIO
    
    mol1 = copy.deepcopy(parmed_atoms)
    for a in mol1.atoms:
        for elem,num in parmed.periodic_table.AtomicNum.items():
            if num == a.atomic_number:
                a.type = elem
                break
    mol2str_1 = StringIO()
    mol1.save(mol2str_1,format="MOL2")
    mol2str_1 = mol2str_1.getvalue()
    return mol2str_1
            

def MutateMap(parmed_atoms1,parmed_atoms2):
    import parmed
    import copy

    mol2str_1 = parmed2mol2str(parmed_atoms1)
    mol2str_2 = parmed2mol2str(parmed_atoms2)
    map1to2 = mcss(mol2str_1, mol2str_2, maxtime=60, isotope_map=None, selec='')

    map2to1 = dict()
    for k in map1to2:
        map2to1[ map1to2[k] ] = k

    return map1to2,map2to1



def AtomsBeyondBondDriver(p,keepatom,delatom,s=[]):
    if delatom in s:
        return s
    else:
        s.append(delatom)
    for a in p.atoms[delatom].bond_partners:
        if a.idx == keepatom:
            pass
        elif a.idx in s:
            pass
        else:
            ss = AtomsBeyondBondDriver(p,delatom,a.idx,s=s)
            for b in ss:
                if b not in s:
                    s.append(b)
    return s

def AtomsBeyondBond(p,keepatom,delatom):
    return AtomsBeyondBondDriver(p,keepatom,delatom,s=[])


def FindIdxByName(p,name):
    idx=None
    for a in p.atoms:
        if a.name == name:
            idx = a.idx
            break
    return idx

def NameMapToIdxMap(p1,p2,nmap):
    from collections import defaultdict as ddict
    i1toi2 = ddict(int)
    for m1 in nmap:
        m2 = nmap[m1]
        i1toi2[ FindIdxByName(p1,m1) ] = FindIdxByName(p2,m2)
    return i1toi2

def PartitionAcrossAtom(p,idx):
    from collections import defaultdict as ddict
    parts=ddict(list)
    for b in p.atoms[idx].bond_partners:
        parts[b.idx] = AtomsBeyondBond(p,idx,b.idx)
    return parts

def ChooseSCAtoms(p,m1to2,idx):
    from collections import defaultdict as ddict
    parts = PartitionAcrossAtom(p,idx)
    cnts = ddict(int)
    maxcnt = -1
    for b in parts:
        commoncnt = 0
        for a in parts[b]:
            if a in m1to2:
                commoncnt += 1
        cnts[b] = commoncnt
        #print(cnts[b],parts[b])
        if commoncnt > maxcnt:
            maxcnt = commoncnt
    scparts = []
    for b in parts:
        if cnts[b] < maxcnt:
            scparts.extend(parts[b])
    return scparts












def ModifiedMCSSModel1(parmed_atoms1,parmed_atoms2,mapfile):
    import parmed
    import sys
    from collections import defaultdict as ddict

    mol1 = parmed_atoms1
    mol2 = parmed_atoms2
    i1toi2,i2toi1 = MutateMap(parmed_atoms1,parmed_atoms2)
    
    m1tom2,m2tom1 = MutateMap(mol1file,mol2file)
    i1toi2 = m1tom2 #NameMapToIdxMap(mol1,mol2,m1tom2)
    i2toi1 = m2tom1 #NameMapToIdxMap(mol2,mol1,m2tom1)

    sc1 = []
    for a in mol1.atoms:
        if a.idx not in i1toi2:
            sc1.append( a.idx )
    sc2 = []
    for a in mol2.atoms:
        if a.idx not in i2toi1:
            sc2.append( a.idx )

            
    for i1 in i1toi2:
        i2 = i1toi2[i1]
        a1 = mol1.atoms[i1]
        a2 = mol2.atoms[i2]
        hybrid1 = len( a1.bond_partners )
        hybrid2 = len( a2.bond_partners )
        if hybrid1 != hybrid2:
            sc1.extend( ChooseSCAtoms(mol1,m1tom2,i1) )
            sc2.extend( ChooseSCAtoms(mol2,m2tom1,i2) )

    for i1 in i1toi2:
        i2 = i1toi2[i1]
        z1 = mol1.atoms[i1].atomic_number
        z2 = mol2.atoms[i2].atomic_number
        if (z1 == 1 or z2 == 1) and z1 + z2 > 2:
            if z1 == 1:
                base1 = mol1.atoms[i1].bond_partners[0].idx
                base2 = i1toi2[base1]
            else:
                base2 = mol2.atoms[i2].bond_partners[0].idx
                base1 = i2toi1[base2]
            sc1.extend( AtomsBeyondBond(mol1,base1,i1) )
            sc2.extend( AtomsBeyondBond(mol2,base2,i2) )

            
    sc1 = list(set(sc1))
    sc2 = list(set(sc2))
    #print([mol1.atoms[a].name for a in sc1])
    #print([mol2.atoms[a].name for a in sc2])
        
    #ats = AtomsBeyondBond(mol1,FindIdxByName(mol1,"C1"),FindIdxByName(mol1,"C3"))

    for sc in sc1:
        if sc in i1toi2:
            del i1toi2[sc]
    for sc in sc2:
        if sc in i2toi1:
            del i2toi1[sc]

    if mapfile is not None: 
        fh=open(mapfile,"w")
        for i1 in i1toi2:
            i2 = i1toi2[i1]
            n1 = mol1.atoms[i1].name
            n2 = mol2.atoms[i2].name
            fh.write("%4s => %4s\n"%(n1,n2))
        fh.close()
        
    return i1toi2,i2toi1




def ModifiedMCSSModel2(parmed_atoms1,parmed_atoms2,mapfile):
    import parmed
    import sys
    from collections import defaultdict as ddict


    mol1 = parmed_atoms1
    mol2 = parmed_atoms2
    i1toi2,i2toi1 = MutateMap(parmed_atoms1,parmed_atoms2)
    
    i1toi2 = m1tom2 #NameMapToIdxMap(mol1,mol2,m1tom2)
    i2toi1 = m2tom1 #NameMapToIdxMap(mol2,mol1,m2tom1)

    sc1 = []
    for a in mol1.atoms:
        if a.idx not in i1toi2:
            sc1.append( a.idx )
    sc2 = []
    for a in mol2.atoms:
        if a.idx not in i2toi1:
            sc2.append( a.idx )

            
    for i1 in i1toi2:
        i2 = i1toi2[i1]
        a1 = mol1.atoms[i1]
        a2 = mol2.atoms[i2]
        hybrid1 = len( a1.bond_partners )
        hybrid2 = len( a2.bond_partners )
        z1 = a1.atomic_number
        z2 = a2.atomic_number
        if z1 != z2:
            sc1.extend( ChooseSCAtoms(mol1,m1tom2,i1) )
            sc2.extend( ChooseSCAtoms(mol2,m2tom1,i2) )
            sc1.append( i1 )
            sc2.append( i2 )
        elif (hybrid1 != hybrid2):
            sc1.extend( ChooseSCAtoms(mol1,m1tom2,i1) )
            sc2.extend( ChooseSCAtoms(mol2,m2tom1,i2) )
        
            
    for i1 in i1toi2:
        i2 = i1toi2[i1]
        z1 = mol1.atoms[i1].atomic_number
        z2 = mol2.atoms[i2].atomic_number
        if (z1 == 1 or z2 == 1) and z1 + z2 > 2:
            if z1 == 1:
                base1 = mol1.atoms[i1].bond_partners[0].idx
                base2 = i1toi2[base1]
            else:
                base2 = mol2.atoms[i2].bond_partners[0].idx
                base1 = i2toi1[base2]
            sc1.extend( AtomsBeyondBond(mol1,base1,i1) )
            sc2.extend( AtomsBeyondBond(mol2,base2,i2) )

            
    sc1 = list(set(sc1))
    sc2 = list(set(sc2))

    for sc in sc1:
        if sc in i1toi2:
            del i1toi2[sc]
    for sc in sc2:
        if sc in i2toi1:
            del i2toi1[sc]

    if mapfile is not None: 
        fh=open(mapfile,"w")
        for i1 in i1toi2:
            i2 = i1toi2[i1]
            n1 = mol1.atoms[i1].name
            n2 = mol2.atoms[i2].name
            fh.write("%4s => %4s\n"%(n1,n2))
        fh.close()
        
    return i1toi2,i2toi1



def OriginalMCSSModel(parmed_atoms1,parmed_atoms2,mapfile):
    import parmed
    import sys
    from collections import defaultdict as ddict

    mol1 = parmed_atoms1
    mol2 = parmed_atoms2
    i1toi2,i2toi1 = MutateMap(parmed_atoms1,parmed_atoms2)

    if mapfile is not None:
        fh=open(mapfile,"w")
        for i1 in i1toi2:
            i2 = i1toi2[i1]
            n1 = mol1.atoms[i1].name
            n2 = mol2.atoms[i2].name
            fh.write("%4s => %4s\n"%(n1,n2))
        fh.close()

    return i1toi2,i2toi1



def OriginalMCSSModelNoMismatchingElements(parmed_atoms1,parmed_atoms2,mapfile):
    import parmed
    import sys
    from collections import defaultdict as ddict

    
    mol1 = parmed_atoms1
    mol2 = parmed_atoms2
    i1toi2,i2toi1 = MutateMap(parmed_atoms1,parmed_atoms2)
    
    #i1toi2 = m1tom2 #NameMapToIdxMap(mol1,mol2,m1tom2)
    #i2toi1 = m2tom1 #NameMapToIdxMap(mol2,mol1,m2tom1)

    sc1 = []
    for a in mol1.atoms:
        if a.idx not in i1toi2:
            sc1.append( a.idx )
    sc2 = []
    for a in mol2.atoms:
        if a.idx not in i2toi1:
            sc2.append( a.idx )

    for i1 in i1toi2:
        i2 = i1toi2[i1]
        a1 = mol1.atoms[i1]
        a2 = mol2.atoms[i2]
        hybrid1 = len( a1.bond_partners )
        hybrid2 = len( a2.bond_partners )
        z1 = a1.atomic_number
        z2 = a2.atomic_number
        if z1 != z2:
            #sc1.extend( ChooseSCAtoms(mol1,m1tom2,i1) )
            #sc2.extend( ChooseSCAtoms(mol2,m2tom1,i2) )
            sc1.append( i1 )
            sc2.append( i2 )

    sc1 = list(set(sc1))
    sc2 = list(set(sc2))

    for sc in sc1:
        if sc in i1toi2:
            del i1toi2[sc]
    for sc in sc2:
        if sc in i2toi1:
            del i2toi1[sc]
            
    return i1toi2,i2toi1
