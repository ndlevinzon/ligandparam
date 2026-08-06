#!/bin/bash
set -e
set -u

if [ ! -e start.json ]; then
    python3 ../../src/python/bin/ffpopt-PrepareInput.py \
	    --parm minimal_orig.parm7 --crd minimal_orig.rst7 \
	    --out start.json
fi


../../src/python/bin/ffpopt-DihedTwistWorkflow.py \
    --geometric-opt --nproc=4 \
    --inp start.json \
    --bond=9,8 --bond=8,6 --bond=6,2 \
    --model=qdpi2 --seqscan > scan.sh


#    --bond=9,8 --bond=8,6 --bond=6,2 \

#ffpopt-DihedTwistWorkflow.py --geometric-opt --nproc=4 -p minimal_orig.parm7 -c minimal_orig.rst7 --bond=9,8 --bond=8,6 --bond=6,2 --model=qdpi2 > scan.sh
#bash ./scan.sh


