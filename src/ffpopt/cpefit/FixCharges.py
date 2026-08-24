


def SymAndFixCharges(refqs,qs,digits):
    return FixCharges(SymCharges(refqs,qs),digits)
    

def SymCharges(refqs,qs):
    import numpy as np
    from collections import defaultdict as ddict
    
    qmap = ddict(list)
    for i in range(len(refqs)):
        qmap[refqs[i]].append(i)

    oqs = np.array(qs,copy=True)
        
    for refq in qmap:
        idxs = qmap[refq]
        m = np.mean( [ qs[i] for i in idxs ] )
        for i in idxs:
            oqs[i] = m
    return oqs




# def FixMaskedCharges(inpcharges,digits,mask):
#     from collections import defaultdict as ddict

#     #print("digits = ",digits)
#     #print(" ".join( ["%.15f"%(x) for x in inpcharges] ))

#     if mask is None:
#         mask = [ True ] * len(inpcharges)

#     fidxs = [ i for i in range(len(mask)) if not mask[i] ]
#     tidxs = [ i for i in range(len(mask)) if mask[i] ]
    
#     charges = [ q for q in inpcharges ]
#     intq = int(round(sum(charges)))
#     tcharges = [ charges[i] for i in tidxs ]
#     tq = sum(tcharges)

#     newq = FixCharges(tcharges,digits,target=tq)
#     for ii,i in enumerate(tidxs):
#         charges[i] = newq[ii]
#     return charges



def FixCharges(inpcharges,digits,target=None):
    from collections import defaultdict as ddict

    #print("digits = ",digits)
    #print(" ".join( ["%.15f"%(x) for x in inpcharges] ))

    charges = [ q for q in inpcharges ]
    q = sum(charges)
    #print(f"Initial net charge %10.{digits}f"%(q))
    nat = len(charges)
    if target is None:
        intq = int(round(q))
    else:
        intq = target
    dq = (intq - q)/nat
    for i in range(nat):
        charges[i] += dq
        charges[i] = float( f"%.{digits}f"%(charges[i]) )

        
    qmap = ddict(list)
    for i in range(nat):
        qmap[charges[i]].append(i)

    uniqueqs = [ q for q in qmap ]
   degens = [ len(qmap[q]) for q in qmap ]

    q = sum([ u*g for u,g in zip(uniqueqs,degens)])
    
    sorted_degens = sorted(degens,reverse=True)

    for g in sorted_degens:
        tmpqs = [charge for charge in charges]
        for i in range(len(uniqueqs)):
            if degens[i] == g:
                for j in qmap[uniqueqs[i]]:
                    dq = float( f"%.{digits}f"%((q-intq)/g) )
                    tmpqs[j] -= dq
                break
        if sum(tmpqs) == 0:
            break
        
    for i in range(nat):
        charges[i] = tmpqs[i]
        
    #q = sum(charges)
    #print(f"Final   net charge %.{digits}f"%(q))
    return charges

