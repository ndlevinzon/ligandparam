#!/usr/bin/env python3

# https://gaussian.com/charge/
#
# %mem=1000mb
# %chk=chkfile
# %nproc=4
# #p HF/6-31G* NoSymm Prop(Potential,Read) Charge
# 
# title Geom(AllCheck) Density(Check) ChkBasis
# 
# 0 1
# O 0.  0.   0.
# H 0. -0.5 -0.5
# H 0.  0.5 -0.5
# 
# 0.0 0.0 1.0 1.0
# 
# 0.0 0.0 1.0
# 0.0 0.0 2.0

 
def WriteFitSh(base):
    import subprocess
    sh = open("%s.resp.sh"%(base),"w")
    sh.write("""#!/bin/bash
out=%s.resp
cat <<EOF > ${out}.qwt
1
1.

EOF

ffpopt-respf -O -i ${out}.inp -o ${out}.out -p ${out}.punch -t ${out}.qout -w ${out}.qwt -e ${out}.esp

rm -f ${out}.punch ${out}.qout ${out}.qwt ${out}.esp

"""%(base))
    sh.close()
    subprocess.call(["bash","%s.resp.sh"%(base)])

    
def WriteArray8(fh,arr):
    for istart in range(0,len(arr),16):
        for ioff in range(16):
            i = istart + ioff
            if i >= len(arr):
                break
            if ioff == 0:
                pass
            fh.write("%5i"%(arr[i]))
        fh.write("\n")


def ReadNextRespCharges(fh):
    import re
    import sys
    #prog = re.compile(r"^ {1}([ \d]{4})([ \d]{4}) {5}([ \d\.\-]{10}) {5}([ \d\.\-]{10})([ \d]{7})([ \d\.\-]{15})([ \d\.\-]{12})")
    prog = re.compile(r"^ {1}([ \d]{4})([ \d]{4}) {5}([ \d\.\-]{10}) {5}([ \d\.\-]{10})([ \d]{7})([ \d\.\-]{15})")

    qs = []
    for line in fh:
        #sys.stderr.write("%s\n"%(line))
        result = prog.match(line)
        if result is not None:
            #print "match ",result.group(4)
            qs.append( float( result.group(4) ) )
            for line in fh:
                #print line
                result = prog.match(line)
                if result is not None:
                    #print "match ",result.group(4)
                    qs.append( float( result.group(4) ) )
                else:
                    break
            break
    return qs


class Molecule(object):

    def __init__(self,filename,conformers,groups=[],freeze=None,verbose=True):
        import parmed
        from parmed.amber.mask import AmberMask
        from copy import deepcopy
        from .. constants import GetHardness
        from .. Reader import FixParmedAtomicNumbers
        from .. Struct import ListOfStruct
        import numpy as np

        self.filename = filename
        if conformers is None:
            self.conformers = []
        else:
            self.conformers = deepcopy(conformers)
            
        try:
            self.parm = parmed.load_file(self.filename,structure=True)
        except:            
            los = ListOfStruct.from_file(self.filename)
            self.parm = los.structs[0].GetParmedAtoms()

        self.groups = [g for g in groups]

        if freeze is not None:
            if len(freeze.strip()) > 1:
                idxs = [ self.parm.atoms[i].idx for i in AmberMask( self.parm, freeze ).Selected() ]
                allidxs = [i for i in range(len(self.parm.atoms))]
                #print(freeze)
                notsele = [ i for i in allidxs if i not in idxs ]
                if verbose:
                    print("    freezing: ",[self.parm.atoms[i].name for i in idxs])
                    print("not freezing: ",[self.parm.atoms[i].name for i in notsele])
                for idx in idxs:
                    self.groups.append( "@%i"%(idx+1) )
                    
        
        self.groupidxs = []
        self.groupcharges = []
        for ig,g in enumerate(self.groups):
            newmaskstr = g.replace("@0","!@*")
            sele = AmberMask( self.parm, newmaskstr ).Selected()
            sele = [ self.parm.atoms[i].idx for i in sele ]
            self.groupidxs.append(sele)
            q = 0
            for i in sele:
                q += self.parm.atoms[i].charge
            self.groupcharges.append(q)
            if verbose:
                print("Constraint: %8.4f %s"%\
                      (q," ".join(["%-4s"%(self.parm.atoms[i].name) for i in sele ])))
            
        self.atnames = [ "%s:%s"%(a.residue.name,a.name)
                         for a in self.parm.atoms ]

        self.atnums = [ a.atomic_number for a in self.parm.atoms ]
        self.parnames = deepcopy(self.atnames)
        self.paridxs = [ i for i in range(len(self.parnames)) ]
        self.netcharge = int(round(sum([a.charge for a in self.parm.atoms])))
        self.loc_constraints = []
        self.append_loc_constraint(self.atnames,self.netcharge)
        for ig in range(len(self.groups)):
            q = self.groupcharges[ig]
            sele = self.groupidxs[ig]
            atnames = [ self.atnames[i] for i in sele ]
            self.append_loc_constraint(atnames,q)

        self.refhardness = np.array([ GetHardness(z) for z in self.atnums ])


    def symmetrize_loc_charges(self,qs):
        from collections import defaultdict as ddict
        import numpy as np
        glb = ddict(list)
        for idx,q in zip(self.paridxs,qs):
            glb[idx].append(q)
        for idx in glb:
            glb[idx] = np.mean(glb[idx])
        out = np.array(qs,copy=True)
        for lidx,gidx in enumerate(self.paridxs):
            out[lidx] = glb[gidx]
        return out

    
    def enforce_loc_constraints(self,qs):
        import numpy as np
        
        newq = np.array(qs,copy=True)
        
        nat = len(newq)
        ncon = len(self.loc_constraints)
        
        A = np.zeros( (ncon,nat) )
        btgt = np.zeros( (ncon,) )

        for icon,con in enumerate(self.loc_constraints):
            names,tgtvalue = con
            #tgtvalue = float( f"%.{args.digits}f"%(tgtvalue) )
            btgt[icon] = tgtvalue
            for name in names:
                locidx = self.atnames.index(name)
                A[icon,locidx] += 1
        bobs = np.dot(A,newq)

        # A.c + bobs = btgt
        # A.c = (btgt-bobs)
        c = np.linalg.lstsq( A, btgt-bobs, rcond=None )
        newq += c[0]
        
        return newq


    def clean_loc_charges(self,qs):
        qs = self.symmetrize_loc_charges(qs)
        return self.enforce_loc_constraints(qs)
    
    
    def get_group_shifted_ids(self):
        import parmed

        ngroups = len(self.groupidxs)
        ids = [] * len(self.atnums)
        for i in range(len(self.atnums)):
            ids.append( [ self.atnums[i] ] + [0]*ngroups )
        for ig,sele in enumerate(self.groupidxs):
            for i in sele:
                ids[i][ig+1] = 1
            
        
        # atnums = [ z for z in self.atnums ]
        # for ig,sele in enumerate(self.groupidxs):
        #     offset = (ig+1) * 100
        #     for i in sele:
        #         atnums[i] += offset
        # return atnums
        return ids


    def get_mask_atom_is_grouped(self):
        g = [ False ] * len(self.atnums)
        for ig,sele in enumerate(self.groupidxs):
            for i in sele:
                g[i] = True
        return g
    

    def get_mask_atom_is_ungrouped(self):
        g = [ True ] * len(self.atnums)
        for ig,sele in enumerate(self.groupidxs):
            for i in sele:
                g[i] = False
        return g

    
    def count_groups_foreach_atom(self):
        g = [ 0 ] * len(self.atnums)
        for ig,sele in enumerate(self.groupidxs):
            for i in sele:
                g[i] += 1
        return g

    
    
    
    def append_loc_constraint(self,atnames,value):
        self.loc_constraints.append( (atnames,value) )

        
    def get_glb_constraints(self,glbparams):
        import numpy as np
        npar = len(glbparams)
        ncon = len(self.loc_constraints)
        conmat = np.zeros( (ncon,npar) )
        convals= np.zeros( (ncon,) )
        for icon,con in enumerate(self.loc_constraints):
            names,value = con
            convals[icon] = value
            for name in names:
                locidx = self.atnames.index(name)
                idx = self.paridxs[locidx]
                conmat[icon,idx] += 1.
        return conmat,convals


    def check_loc_constraints(self,locqs):
        for icon,con in enumerate(self.loc_constraints):
            names,tgtvalue = con
            obsvalue = 0
            for name in names:
                locidx = self.atnames.index(name)
                obsvalue += locqs[locidx]
            print("check constraint tgt=%11.6f obs=%11.6f diff=%11.6f from %s"%(tgtvalue,obsvalue,abs(tgtvalue-obsvalue)," ".join(names)))
    

    def get_resp_header(self):
        nmol = len(self.conformers)
        return "title\n &cntrl inopt=0 ioutopt=0 iqopt=1 ihfree=1 irstrnt=1 iunits=1 qwt=0.0005 nmol=%i &end\n"%(nmol)

    
    def get_resp_body(self):
        import copy
        from collections import defaultdict as ddict
        from io import StringIO

        nat = len(self.atnums)
        parcnt = ddict(list)
        for i,idx in enumerate(self.paridxs):
            parcnt[idx].append(i)
        #for idx in parcnt:
        #    print(idx,parcnt[idx])
            
        respidx = [0]*nat
        for i in range(nat):
            idx = self.paridxs[i]
            if len(parcnt[idx]) > 1:
                if i != parcnt[idx][0]:
                    respidx[i] = parcnt[idx][0]+1

        fh = StringIO()
        fh.write("%10.5f\n"%(1.0))
        fh.write(" molecule\n")
        fh.write("%5i%5i\n"%(self.netcharge,nat))
        for z,eq in zip( self.atnums, respidx ):
            fh.write("%5i%5i\n"%(z,eq))
        if len(self.loc_constraints) > 1:
            for con in self.loc_constraints[1:]:
                cnames,cval = con
                iats = [ self.atnames.index(name) for name in cnames ]
                #pidxs = [ self.paridxs[i] for i in iats ]
                fh.write("%5i%10.5f\n"%(len(iats),cval))
                arr=[]
                for idx in iats:
                    arr.append(1)
                    arr.append(idx+1)
                WriteArray8(fh,arr)
        fh.write("\n")
        return fh.getvalue()


    
    def RunRespF(self,prefix="tmp"):
        import numpy as np
        import os
        #esppts,espvals
        
        nconf = len(self.conformers)
        nat = len(self.atnums)
        header = self.get_resp_header()
        body = self.get_resp_body()
        
        inp = open("%s.resp.inp"%(prefix),"w")
        esp = open("%s.resp.esp"%(prefix),"w")
        inp.write( header )
        for iconf,c in enumerate(self.conformers):
            inp.write(body)
            #equiv.append( self, mdout )
            #self.append_esp( esp, mdout )
            crds = c.crds
            #pts = c.esppts
            pts = np.array([ elem.crd for elem in c.espsurf.elems ])
            vals = c.espvals
            #print(iconf,len(vals),pts)
            npt=len(vals)
            esp.write("%5i%5i\n"%(nat,npt))
            for i in range(nat):
                esp.write("%17s%16.7f%16.7f%16.7f\n"%("",crds[i,0],crds[i,1],crds[i,2]))
            for i in range(npt):
                esp.write(" %16.7f%16.7f%16.7f%16.7f\n"%(vals[i],pts[i,0],pts[i,1],pts[i,2]))
        esp.close()

        if nconf > 0:
            inp.write("\n")
            for a in range(nat):
                idxs = []
                for iconf in range(nconf):
                    idxs.append(iconf+1)
                    idxs.append(a+1)
                if len(idxs) > 0:
                    inp.write("%5i\n"%( len(idxs)//2 ) )
                    WriteArray8(inp,idxs)
        inp.write("\n")
        inp.write("\n\n")
        inp.close()

        print("Writing multifit %s.resp.inp"%(prefix))
        WriteFitSh( prefix )
        
        if not os.path.exists("%s.resp.out"%(prefix)):
            raise Exception(f"File not found: {prefix}.resp.out")
        
        out = open("%s.resp.out"%(prefix),"r")
        qs = ReadNextRespCharges(out)
        return np.array(qs)
    

#m = Molecule("DMG.mol2",None)
#for a in m.parm.atoms:
#    print("%2i %12.8f %12.8f %12.8f %12.8f"%\
#          (a.atomic_number,
#           a.xx,a.xy,a.xz,a.charge))
    
          

#uparams = AssignUniqueParams([m])
#for i in range(len(m.atnames)):
#    print("%9s %9s"%(m.atnames[i],m.parnames[i]))
#print(len(uparams),uparams)

#GetLebedevRule(0)
