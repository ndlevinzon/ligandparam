#!/bin/bash


ffpopt-ConfSearch.py -o ribose.json --nkeep=1 "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H]1O"

ffpopt-Json2Crds.py --inp=ribose.json --out=mol.pdb

rm ribose.json


antechamber -i mol.pdb -fi pdb -o MOL.mol2 -fo mol2 -c bcc -nc 0 -rn MOL -at gaff2 -du y -an y -pf y -seq n




#rm mol.pdb
#ffpopt-PrepareInput.py --crd am1bcc.mol2 --out am1bcc.json

if [ -e gaff2.dat ]; then
    rm gaff2.dat
fi

parmchk2 -i MOL.mol2 -o MOL.frcmod -f mol2 -s gaff2


cat <<'EOF' > tleap.inp
source leaprc.protein.ff14SB
source leaprc.gaff2

loadamberparams MOL.frcmod
m = loadmol2 MOL.mol2

saveamberparm m MOL.parm7 MOL.rst7
quit
EOF

tleap -s -f tleap.inp


for f in *~ leap.log sqm.in sqm.out sqm.pdb tleap.inp mol.pdb MOL.frcmod; do
    if [ -e "${f}" ]; then
	rm "${f}"
    fi
done

ffpopt-PrepareInput.py --parm MOL.parm7 --crd MOL.rst7 --out MOL.json


if [ -e ../../src/python/bin/ffpopt-NDimWavefront.py ]; then

    if [ ! -e gaff2.json ]; then
	python3 ../../src/python/bin/ffpopt-NDimWavefront.py \
		-i MOL.json \
		-o gaff2.json \
		--restrain-puckerx='10.,4,6,8,2,3' \
		--restrain-puckery='10.,4,6,8,2,3' \
		--resdim="-60,60,15" \
		--resdim="-60,60,15" \
		--wf-max-levels=100 \
		--nproc=12 2> /dev/null
    fi

fi
    
if [ -e ~/devel/gitlab/fe-toolkit/ndfes/examples/pmf2d/Example2d.py ]; then

    if [ -e wf_workflow_gaff2.xml ]; then
    
	python3 ~/devel/gitlab/fe-toolkit/ndfes/examples/pmf2d/Example2d.py \
		--rbf --xlabel="Zx" --ylabel="Zy" \
		--title="GAFF2" --zerobyhist \
		wf_workflow_gaff2.xml

    fi

fi
