#!/bin/bash
set -e
set -u

res=DMG
qmexe=psi4
#qmexe="mpirun -n 6 quick.MPI"
#qmexe="g16"
#qmexe="quick.cuda"

#
# Search for conformers. This will produce up-to 5 structures.
# In practice, this DMG example produces 2 conformations.
#


if [ ! -e ${res}.json ]; then
    python3 ../../src/python/bin/ffpopt-PrepareInput.py \
	    --crd ${res}.mol2 --out ${res}.json
else
    echo "Skipping ffpopt-PrepareInput.py because ${res}.json already exists"
fi

if [ ! -e confs.json ]; then
    python3 ../../src/python/bin/ffpopt-ConfSearch.py \
	    --out confs.json ${res}.json
else
    echo "Skipping ffpopt-ConfSearch.py because confs.json already exists"
fi


#
# We could optimize each structure with a ML or ab initio
# method; however, be aware that the optimization may
# transfer protons between heavy atoms. In fact, the DMG
# example produces 2 conformers, and 1 of them results in
# an internal proton transfer. So, instead, we shall skip
# the optimization.
#

#python3 ../../src/python/bin/ffpopt-Optimize.py \
#        --inp ${res}.json --model=qdpi2 \
#        --out opt.json --geometric-opt


#
# Perform as RESP-like fit.
# It does not use a 2-stage fitting procedure, so the charges
# will be different from other fitting programs. A 2-stage
# fit will:
#   1. Optimize all charges without enforcing charge-equivalence.
#   2. Fix all charges for those atoms that are not involved in
#      charge-equivalence constraints.
#   3. Re-optimize the subset of charges involved in
#      charge-equivalency constraints while imposing those
#      constraints while keeping the remaining charges fixed
#      from the first stage.
#
# In contrast, ffpopt-respfit.py will simultaneously fit
# all charges while imposing charge-equivalency constraints.
# It can also include multiple conformations within the fit.
#
# It's not entirely clear how one would perform the 2-stage
# fitting procedure when there's more than 1 molecule or
# conformation because there could be charge-equivalencies
# between the molecules (and certainly among the conformations).
# Furthermore, the approach used in ffpopt-respfit.py should
# technically fit the electrostatic potential better than
# a 2-stage procedure.
#

#if [ ! -e resp.json ]; then
    
    python3 ../../src/python/bin/ffpopt-RespFit.py \
	--respf \
	--inp ${res}.json \
	--out resp.json \
	--program="${qmexe}" \
	--model="hf/6-31G*" \
	--scosmo=0.5 \
	confs.json

#else

#    echo "Skipping resp fit because resp.json already exists"
    
#fi

echo "Writing resp.mol2 from resp.json"
python3 ../../src/python/bin/ffpopt-Json2Crds.py \
	--inp resp.json --out resp.mol2


#
# The positional arguments of ffpopt-RespFit.py
# are either mol2, gaussian output, psi4 output, quick output,
# or other format that parmed can read
#
# The --respf option causes the fit to be used with Kollmann's original
# resp.f program, which has been installed as ffpopt-respf
# The --program option defaults to psi4, but it could also be
# --program="g16"
# --program="quick.cuda"
# --program="quick"
# --program="mpirun -n 12 quick.MPI"
#

