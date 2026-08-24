#!/usr/bin/env python3

# def FixCharges(inpcharges,digits):
#     from collections import defaultdict as ddict

#     charges = [ q for q in inpcharges ]
#     q = sum(charges)
#     #print(f"Initial net charge %10.{digits}f"%(q))
#     nat = len(charges)
#     intq = int(round(q))
#     dq = (intq - q)/nat
#     for i in range(nat):
#         charges[i] += dq
#         charges[i] = float( f"%.{digits}f"%(charges[i]) )

        
#     qmap = ddict(list)
#     for i in range(nat):
#         qmap[charges[i]].append(i)

#     uniqueqs = [ q for q in qmap ]
#    degens = [ len(qmap[q]) for q in qmap ]

#     q = sum([ u*g for u,g in zip(uniqueqs,degens)])
    
#     sorted_degens = sorted(degens,reverse=True)

#     for g in sorted_degens:
#         tmpqs = [charge for charge in charges]
#         for i in range(len(uniqueqs)):
#             if degens[i] == g:
#                 for j in qmap[uniqueqs[i]]:
#                     dq = float( f"%.{digits}f"%(q/g) )
#                     tmpqs[j] -= dq
#                 break
#         if sum(tmpqs) == 0:
#             break
        
#     for i in range(nat):
#         charges[i] = tmpqs[i]
        
#     q = sum(charges)
#     #print(f"Final   net charge %.{digits}f"%(q))
#     return charges




if __name__ == "__main__":

    import argparse
    from ffpopt.Options import AddModelOptions
    from ffpopt.RespFit import RunRespFit

    parser = argparse.ArgumentParser \
        ( formatter_class=argparse.RawDescriptionHelpFormatter,
          description="""Read or create a structure and search for conformations""")
    

    parser.add_argument \
        ("--out",
         type=str,
         required=True,
         help="Output json or mol2 file.")

    parser.add_argument \
        ("--inp",
         type=str,
         required=True,
         help="Input json or mol2 file. The output will be the same as this file but with different charges.  If this is a json file, then only the first structure is examined.")
    
    
    parser.add_argument \
        ("confs",
         type=str,
         nargs='+',
         help="1-or-more conformers. Either json, xyz, or mol2 files")

    parser.add_argument \
        ("--respf",
         action='store_true',
         help="If present, perform resp fit using the 'resp' program (expected to be in path)")
    
    
    parser.add_argument \
        ("--espaloma",
         action='store_true',
         help="If present, then use conformer-independent espaloma charges from https://doi.org/10.1021/acs.jpca.4c01287")
    
    
    parser.add_argument \
        ("--hilfiker",
         action='store_true',
         help="If present, then use conformer-dependent hilfiker charges from https://doi.org/10.48550/arXiv.2512.13579")
    
    parser.add_argument \
        ("--nofit",
         action='store_true',
         help="If present, then charge fitting is not performed; instead the input charges are rounded to the desired digits and the forced to sum to the nearest integer")
    
    parser.add_argument \
        ("--program",
         type=str,
         required=False,
         default="psi4",
         help="Ab initio executable. Default: psi4. This could also be gaussian; e.g., --program=g16")

    
    # parser.add_argument \
    #     ("--model",
    #      type=str,
    #      required=False,
    #      default="hf/6-31g*",
    #      help="Ab initio method used to calculate the electrostatic potential. Default='hf/6-31g*'")

    # parser.add_argument \
    #     ("--psi4-num-threads",
    #      type=int,
    #      required=False,
    #      default=4,
    #      help="Number of CPU cores. Default: 4. Despite the name, this is also used for Gaussian calculations.")

    # parser.add_argument \
    #     ("--psi4-memory",
    #      type=str,
    #      required=False,
    #      default='1gb',
    #      help="Amount of RAM. Default: '1gb'. Despite the name, this is also used for Gaussian calculations.")

    
    parser.add_argument \
        ("--resp-a",
         type=float,
         required=False,
         default=0.001,
         help="Hyperbolic penalty prefactor. Default: 0.001. The penalty is pen= a * sum_i ( sqrt( qi**2 + b**2 ) - b ), where i loops over all heavy atoms.")

    
    parser.add_argument \
        ("--resp-b",
         type=float,
         required=False,
         default=0.1,
         help="Hyperbolic penalty width. Default: 0.1. The penalty is pen= a * sum_i ( sqrt( qi**2 + b**2 ) - b ), where i loops over all heavy atoms.")
    
    
    parser.add_argument \
        ("--density",
         type=float,
         required=False,
         default=6,
         help="The density of surface points. Default: 6 pts/Ang**2.")

    
    parser.add_argument \
        ("--digits",
         type=int,
         default=4,
         help="Round charges to this number of digits. Default: 4")

    
    parser.add_argument \
        ("--scosmo",
         type=float,
         default=0.0,
         help="Calculate fits in gas and cosmo environments and take a weighted average of the 2 charge vectors. The SCOSMO perturbation is a response to the MM charges fit to the gas phase ESP. Default: 0.0, which returns only the gas phase charges.")
    

    parser.add_argument \
        ("--ext-scale",
         type=float,
         required=False,
         default=1.1,
         help="UFF radius scale factor used to generate the external potential surface. Default: 1.1")

    
    parser.add_argument \
        ("--ext-density",
         type=float,
         required=False,
         default=2,
         help="Density of external potential points. Default: 2 pts/Ang**2")


    parser.add_argument \
        ("--group",
         type=str,
         default=[],
         action='append',
         help="Amber-style atom selection string indicating a group of atoms whose charge-sum should not change. This can be used multiple times. This could potentially fail if an atom exists in multiple groups.")
    
    parser.add_argument \
        ("--freeze",
         type=str,
         required=False,
         help="Amber-style atom selection string indicating a group of atoms whose charge should not change. This is the same as creating a --group for each atom in the selection")
     
    parser.add_argument \
        ("--update-only-grouped-atoms",
         action='store_true',
         help="If present, then the output file only update the atomic charges for those atoms present in any group (or freeze). This can only be used if atoms are not part of multiple groups.")
    
    parser.add_argument \
        ("--update-only-ungrouped-atoms",
         action='store_true',
         help="If present, then the output file only update the atomic charges for those atoms not present in any group (or freeze). This can only be used if atoms are not part of multiple groups.")
    
    
    AddModelOptions(parser)
    args = parser.parse_args()

    RunRespFit(**vars(args))
    
        
    
    
