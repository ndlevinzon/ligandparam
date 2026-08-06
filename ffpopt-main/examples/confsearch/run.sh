#!/bin/bash

python3 ../../src/python/bin/ffpopt-ConfSearch.py \
	--out confs.json DMG.mol2

python3 ../../src/python/bin/ffpopt-Optimize.py \
	--inp confs.json --model="hf/6-31g*" \
	--out opt.json --geometric-opt

python3 ../../src/python/bin/ffpopt-Json2Crds.py \
	--inp confs.json --out confs.mol2

python3 ../../src/python/bin/ffpopt-Json2Crds.py \
	--inp opt.json --out opt.mol2


