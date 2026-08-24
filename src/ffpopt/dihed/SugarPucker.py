"""Sugar-pucker atom guesses used by specialty dihedral CLIs."""

from __future__ import annotations

def FindPuckers(s):
    """ Guess the C1,C2,C3,C4,O4 sugar atoms for each 5-membered ring in
    the structure

    Parameters
    ----------
    s : ffpopt.Struct.Struct
        The input structure to examine

    Returns
    -------
    rings : list of tuple
        The length of rings is the number of 5-membered rings
        Each element of the list is a tuple.
        The first element of the tuple is a list: the 5 indexes of the ring
        The second element is the same list sorted in order [C1,C2,C3,C4,O4]
        If the order could not be guessed, then the second element is None
    """
    
    g = s.GetGraph()
    # Closed paths from cycle basis; length 6 => 5 unique ring atoms + close.
    mincycs = g.FindMinCycles()
    keepcycs = []
    for c in mincycs:
        if len(c) == 6:
            h = [x for x in c[:-1]]
            keepcycs.append(h)

    rings = []
    for c in keepcycs:
        ringseq = PuckerGuessByName(c,g,s)
        if ringseq is None:
            ringseq = PuckerGuessByElement(c,g,s)
        rings.append( (c,ringseq) )
        
    return rings


def PuckerGuessByName(cinp,g,s):
    """ Guess the C1,C2,C3,C4,O4 sugar atoms by element

    1. It assumes cinp is a list of 5 integers.
    2. The atom names are checked.
       2a. One atom name must contain "1"
       2b. One atom name must contain "2"
       2c. One atom name must contain "3"
       2d. Two atoms must contain "4"
    3. The O4 position is chosen from 2d by checking for
       a covalent bond to 2a.
    """
    from collections import defaultdict as ddict
    c = [ int(x) for x in cinp ]
    onames = [x for x in s.data["names"]]

    bad = False
    
    ignores = [ str(x) for x in range(10,100) ]
    for i in range(len(onames)):
        for ig in ignores:
            if ig in onames[i]:
                if i in c:
                    # One of the atoms in the ring has a name like
                    # C21 rather than C1 or C2 so we really can't
                    # use the names to figure out what is going on.
                    bad = True
                    
    dpos = ddict(list)
    unknown = []
    if not bad:
        #n = len(onames)
        #cmask = [False]*n
        #for i in c:
        #    cmask[i] = True
        for i in c:
            found=False
            for ipos in [1,2,3,4]:
                pos=str(ipos)
                if pos in onames[i]:
                    dpos[ ipos ].append( i )
                    found=True
                    break
            if not found:
                unknown.append(i)
            
    bad = len(unknown) > 0
    if not bad:
        if len(dpos[4]) != 2:
            bad=True
        elif len(dpos[3]) != 1:
            bad=True
        elif len(dpos[2]) != 1:
            bad=True
        elif len(dpos[1]) != 1:
            bad=True
            
    #hpos = ddict(list)
    #for x in dpos:
    #    for u in dpos[x]:
    #        hpos[x].append( onames[u] )

    if not bad:
        if str(dpos[1][0]) in g.edges[ str(dpos[4][0]) ]:
            O4 = dpos[4][0]
            C4 = dpos[4][1]
        elif str(dpos[1][0]) in g.edges[ str(dpos[4][1]) ]:
            O4 = dpos[4][1]
            C4 = dpos[4][0]
        else:
            bad = True

    ringseq = None
    if not bad:
        C1 = dpos[1][0]
        C2 = dpos[2][0]
        C3 = dpos[3][0]
        ringseq = [ C1,C2,C3,C4,O4 ]
    return ringseq




def PuckerGuessByElement(cinp,g,s):
    """
    Guess the C1,C2,C3,C4,O4 sugar atoms by element

    1. It assumes cinp is a list of 5 integers.
    2. 4 of the positions must correspond to "C", the non-carbon
       takes the O4' position.
    3. The bonding pattern is used to define a "clock". The
       clock either corresponds to O4,C1,C2,C3,C4 or
       O4,C4,C3,C2,C1 -- the rest of the algorithm is to figure
       out which of these is correct
    4. The C2/C3 positions are checked for bonded oxygens.
       a. Algorithm fails if either has more than 1 oxygen.
       b. It one has a O and the other doesn't, then it is assumed to be DNA, and the C3 position is chosen to be the one with the oxygen
       c. If they each have 1 oxygen, then the O4-C1-C2-O2 and O4-C4-C3-O3 dihedrals are calculated, and the decision is made based on these dihedrals.
    
    """

    
    from collections import defaultdict as ddict
    c = [ int(x) for x in cinp ]
    onames = [x for x in s.data["elements"]]

    nonc = [x for x in c if onames[x] != "C"]
    bad = len(nonc) != 1

    bonds = ddict(list)
    for a in g.edges:
        for b in g.edges[a]:
            bonds[ int(a) ].append( int(b) )

    clock = []
    if not bad:
        O4 = nonc[0]
        n1or4 = []
        for x in c:
            if x in bonds[O4]:
                n1or4.append(x)
        bad = len(n1or4) != 2
        
    if not bad:
        excl = n1or4 + [O4]
        clock = [O4,n1or4[0],None,None,n1or4[1]]
        for x in c:
            if x in excl:
                continue
            elif x in bonds[n1or4[0]]:
                clock[2] = x
            elif x in bonds[n1or4[1]]:
                clock[3] = x
        bad = None in clock
        
    if not bad:
        nbors2 = [ x for x in bonds[clock[2]] if x not in clock ]
        nnams2 = [ onames[x] for x in nbors2 ]
        nisO2 = [ 1 if name == "O" else 0 for name in nnams2 ]
        
        nbors3 = [ x for x in bonds[clock[3]] if x not in clock ]
        nnams3 = [ onames[x] for x in nbors3 ]
        nisO3 = [ 1 if name == "O" else 0 for name in nnams3 ]
        
        if sum(nisO2) > 1:
            bad = True
        elif sum(nisO3) > 1:
            bad = True

        if bad:
            clock=None
            return clock
        
        Oin2 = "O" in nnams2
        Oin3 = "O" in nnams3
        if Oin2 and not Oin3:
            # DNA
            clock = clock[::-1]
        elif Oin3 and not Oin2:
            # DNA
            pass
        elif not Oin2 and not Oin3:
            # Unidentifiable from this algorithm
            clock = None
            bad = True
        else:
            # RNA
            O2 = nbors2[ nnams2.index("O") ]
            #print(clock[0],clock[1],clock[2], O2)
            a2 = s.get_dihedral( clock[0],clock[1],clock[2], O2)
            #print(a2)
            O3 = nbors3[ nnams3.index("O") ]
            #print(clock[0],clock[4],clock[3], O3)
            a3 = s.get_dihedral( clock[0],clock[4],clock[3], O3)
            #print(a3)
            if a2 < 180 and a3 > 180:
                O4 = clock[0]
                C1 = clock[1]
                C2 = clock[2]
                C3 = clock[3]
                C4 = clock[4]
                clock = [C1,C2,C3,C4,O4]
                #pass
            elif a2 > 180 and a3 < 180:
                O4 = clock[0]
                C1 = clock[4]
                C2 = clock[3]
                C3 = clock[2]
                C4 = clock[1]
                clock = [C1,C2,C3,C4,O4]
                #clock = clock[::-1]
                pass
            else:
                bad = True
                clock = None
          
    return clock



