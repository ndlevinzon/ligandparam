Gitlab Pages is Here: https://ffpopt-b083ab.gitlab.io/

# INSTALLATION

Many of the software dependencies are installed via conda-forge.
If you don't already have conda environments setup on your computer,
then you can install miniforge.


 1. Download the miniforge installer
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```
 
 2. Install miniforge
```
bash ./Miniforge3-Linux-x86_64.sh -b -f -p ${PWD}/miniforge3
```

 3. Enter the conda environment and create ffpopt environments for pytorch and tensorflow.  The `eval` command is to setup your shell to use conda environments if you have not setup conda on your computer before. If you've already used conda environments on your computer, then it is unnecessary. Simply activate your base environment and create ffpopt environment(s) using the environment.yml file.
```
source ${PWD}/miniforge3/bin/activate
mamba env create --yes -n ffpopt-pytorch python=3.12 -f environment.yml
mamba env create --yes -n ffpopt-tensorflow python=3.12 -f environment.yml
eval "$(mamba shell hook --shell bash)"
```

 4. Install ffpopt source and python dependencies via pip.

    The dependencies are separated into "groups".
    
    - --group=fairchem : installs https://github.com/facebookresearch/fairchem
    - --group=tensorflow : installs https://github.com/deepmodeling/deepmd-kit and relevant dependencies
    - --group=pytorch : installs pytorch libraries and pytorch models
    
    The 3 groups install incompatible software stacks.
    The fairchem models aren't very good, so one doesn't normally install
    the fairchem group.
    The tensorflow group will install deepmd-kit used to evaluate the
    qdpi2 model, which is distributed along with ffpopt.
    The pytorch group is a very popular framework to develop machine learning
    models.
    You will need to create separate environments for each stack.
    For example, --group tensorflow will install deepmd-kit[tf],
    which is incompatible with the --group pytorch which installs
    deepmd-kit[torch].

    If ACADEMIC=TRUE is specified, then machine learning
    models will be downloaded from the internet for academic
    use only. The default behavior is to assume the user is
    in industry, in which case you'll need to contact the
    appropriate authors for permission to use their trained
    models.

```    
mamba activate ffpopt-tensorflow
ACADEMIC=TRUE python3 -m pip install --group tensorflow --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir .
mamba deactivate

mamba activate ffpopt-pytorch
CFLAGS="-Wno-array-bounds" CXXFLAGS="-Wno-array-bounds -include cstdint" ACADEMIC=TRUE python3 -m pip install --group pytorch --extra-index-url https://download.pytorch.org/whl/cu121 --find-links https://data.dgl.ai/wheels/cu121/repo.html --no-cache-dir .
mamba deactivate
```

    
# NOTES

1. On modern OSs, you may need to remove the exec-flag on the 
shared libraries installed by conda/mamba.
For example,
     
```
execstack -c ${PWD}/miniforge3/envs/ffpopt-tensorflow/lib/*.so
execstack -c ${PWD}/miniforge3/envs/ffpopt-pytorch/lib/*.so
```

2. If you have a modern OS, it's easiest to install the dependencies
using conda; however, note that the psi4 conda package will install
numpy-2.x, whereas parmed requires numpy-1.x.  You will need to
create a separate conda installation specifically for psi4. If
you install psi4 in miniforge3_psi4 and all other dependencies in
miniforge3_allother, then you can *prepend* the bin and site-packages
from miniforge3_allother to your PATH and PYTHONPATH variables,
and then *append* the bin and site-packages from miniforge3_psi4.
The psi4 pacakge will still run with numpy-1.x even though it
installs numpy-2.x; and the procedure that I've described will
cause the numpy-1.x installation to be loaded rather than 2.x.


3. If you are on a cluster that has an "old" glibc version, then
the  dacase::ambertools-dac=25 package may not work correctly.
One cannot use the unofficial "ambertools" conda package because
that version of the code has a bug in the charmm module that
produces memory leaks and segmentation faults when the calculator
is reloaded more than 1 time.  For older machines, you should
install ambertools from source and install the other dependencies
using pip. You will need to create separate installations for
tensorflow and pytorch models. The tensorflow and pytorch packages
require different versions of the nvidia libraries that are
incompatible with each other.


4.  To use QUICK installed with ambertools from conda, you'll need to define
```
export QUICK_BASIS="${PWD}/miniforge3/AmberTools/src/quick/basis"
```

5. If you encounter an error like:

   Could not import the compiled Python-sander interface. [...]

   Then you may need to use the execstack utility to clear the executable
stack flag from the libsander.so file. For example:
```
sudo dnf install execstack
execstack -c /path/to/libsander.so
```

6. Make sure the PYTHONPATH  within ${PWD}/psi4 does not
prepend those in ${PWD}/tensorflow nor ${PWD}/pytorch because
parmed/ambertools require numpy < 2, whereas conda installation of psi4
provides numpy==2 But psi4 can still run with numpy < 2.


7. If you want to install pyscf_neo, install
```
     git clone https://github.com/theorychemyang/pyscf.git
     cd pyscf
     cd pyscf/lib
     mkdir build
     cd build
     cmake ..
     make
     export PYTHONPATH=/path/to/pyscf:$PYTHONPATH

```


# DOCUMENTATION


## COMMON OPTIONS

 All scripts support a collection of common options, and each script
 includes a few script-specific options.

 The common options are:
-   --model : str
       Select the model chemistry. The available options are
       listed in the "Models" section (see below). One can also
       specify theory/basis to run ab initio with psi4.
       The default is --model=sander, which uses pysander to
       evalute the energy and forces from the input --parm
       and --crd files.  Most of the available models are
       either ab initio, semiempirical, or machine learning
       potentials; these models do not require parm7/rst7
       files. Instead, one can specify --parm=lig.mol2 --crd=lig.mol2.
       The atomic charges within the mol2 are not important,
       except: the sum of atomic charges should equal the net
       charge of the molecule.  We provide a script called
       `ffpopt-xyz2mol2.py -i lig.xyz -o lig.mol2 --charge=0`
       which will write a mol2 file from an input net charge
       and xyz file.  You cannot run sander using mol2 files.
       You cannot optimize amber torsion parameters with
       mol2 files. In those specific applications, one has to
       have parm7/rst7 files.

-  --no-opt
       Calculate a single point energy rather than a geometry
       optimization

-  --geometric-opt
       Use the geomeTRIC rather than the ASE BFGS optimizer

-  --geometric-maxiter: int
       Number of geometry optimization steps when using geometric.
       Default: 500

-  --geometric-coordsys: str
       Coordinate system. Default: tric

-  --geometric-converge: str
       Optimization tolerance. Default: 'set GAU_TIGHT'

-  --geometric-enforce: float
       Constraint enforcement tolerance. Default 0.1

-  --psi4-memory: str
       Amount of RAM used by psi4. Default: '1gb'

-  --psi4-num-threads: int
       The number of psi4 threads. Default: 4

-  --parm: str
       The amber parm7 file. This is always required.

-  --crd: str
       The amber formatted rst7 file. This is always required.

-  --cpu (no argument)
       If present, then evaluate machine learning potential
       on the cpu even if gpu(s) are available. For many
       models, it can be faster to run it on cpu when
       optimizing a small molecule because the gpu initialization
       can be more expensive than the calculation.
       

The input structure is typically set using the
--parm=lig.parm7 --crd=lig.rst7 options, where --parm is an
amber parm7 file and --crd is a formatted amber restart file.
If you need to evaluate the potential energy using the sander
program, then these file format are required.
Similarly, if you want to optimize amber torsion force
field parameters, then you must read parm7/rst7 files.

You can optimize geometries and perform scans
with ab initio, semiempirical, and machine learning potentials
using only a mol2 file (without needing to generate
parm7/rst7 files).  The atomic partial charges within the
mol2 file are not important except: the sum of partial
charges should equal the net charge of the molecule.
For your convenience we provide the ffpopt-xyz2mol2.py script.
The following command transforms lig.xyz to lig.mol2
while uniformly distributing the net charge to the atoms.
```
ffpopt-xyz2mol2.py --charge=0 -i lig.xyz -o lig.mol2
```
Not only does the mol2 file format provide the net charge
(which is needed for QM calculations), but also allows us
to read the structure with the parmed python package
so we can identify covalent bonds, angles, torsions, etc.




## MODELS

The -m option can take the following values:

- sander
     - Description: MM calculation
     - Link: https://ambermd.org/
     - Note: This interface requires parm7/rst7 files

- dftb2
     - Description: Second order self-consistent charge
       density functional tight binding. MIO-1-1 parameters.
     - Link: https://github.com/dftbparams/mio
     - Elements: H,C,N,O,P,S,Si,Ag,Ga
     - Note: This is run through pysander; therefore it
       requires parm7/rst7 files

- dftb3
     - Description: Full 3rd-order SCC-DFTB with O3B parameters.
     - Link: https://github.com/dftbparams/3ob
     - Elements: H,C,N,O,F,K,P,S,F,Na,Mg,Cl,Ca,I,Br,Zn
     - Note: This is run through pysander; therefore it
       requires parm7/rst7 files

- qdpi2
     - Description: xtb+delta MLP model based on DeepPot-SE
     - Link: https://www.doi.org/10.1021/acs.jpcb.4c01466
     - Elements: H,C,N,O,F,Na,P,S,Cl,K,Br,I
     
- xtb
     - Description: The GFN2-xTB semiempirical model
     - Link: https://github.com/grimme-lab/xtb
     - Link: https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176
     - Elements: 1-84 (H-to-Po)
     
- mace (or mace-off23_medium) 
     - Description: MACE-OFF23_medium.model GNN
     - Link: https://github.com/ACEsuit/mace-off
     - Elements: H, C, N, O, F, P, S, Cl, Br, I

- mace-off23b_medium
     - Description: MACE-OFF23b_medium.model GNN
     - This is like mace-off23_medium; however, it increases the
     graph cutoff radius from 5A to 6A. This was found to be
     crucial for recovering certain condensed phase
     properties.
     - Link: https://github.com/ACEsuit/mace-off
     - Elements: H, C, N, O, F, P, S, Cl, Br, I

- mace-off23_small
     - Description: MACE-OFF23_small.model GNN
     - Link: https://github.com/ACEsuit/mace-off
     - Elements: H, C, N, O, F, P, S, Cl, Br, I

- mace-off23_large
     - Description: MACE-OFF23_large.model GNN
     - Link: https://github.com/ACEsuit/mace-off
     - Elements: H, C, N, O, F, P, S, Cl, Br, I

- mace-off24_medium
     - Description: MACE-OFF23_large.model GNN
     - Link: https://github.com/ACEsuit/mace-off
     - Elements: H, C, N, O, F, P, S, Cl, Br, I

- aimnet2
     - Description: Trained against wB97M-D3
     - Link: https://github.com/isayevlab/aimnetcentral
     - Elements: H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I

- aimnet2_b973c
     - Description: Trained against B97-3c
     - Link: https://github.com/isayevlab/aimnetcentral
     - Elements: H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I

- aimnet2_2025
     - Description: Trained against B97-3c + improved intermolecular interactions
     - Link: https://github.com/isayevlab/aimnetcentral
     - Elements: H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I

- aimnet2nse
     - Description: Open-shell chemistry
     - Link: https://github.com/isayevlab/aimnetcentral
     - Elements: H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I

- aimnet2pd
     - Description: Palladium-containing systems
     - Link: https://github.com/isayevlab/aimnetcentral
     - Elements: H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I



- ani1ccx
     - Description: The ANI-1ccx model is an ensemble of 8 networks that was
     trained on the ANI-1ccx dataset, using transfer learning. The target
     accuracy is CCSD(T)*/CBS (CCSD(T) using the DPLNO-CCSD(T) method). It
     predicts energies on HCNO elements exclusively, it shouldn't be used
     with other atom types.
     - Link: https://github.com/aiqm/torchani
     - Link: https://doi.org/10.1038/s41597-020-0473-z
     - Elements: H, C, N, O
     
- ani1x
     - Description: The ANI-1x model is an ensemble of 8 networks that was
     trained using active learning on the ANI-1x dataset, the target level
     of theory is wB97X/6-31G(d). It predicts energies on HCNO elements
     exclusively, it shouldn't be used with other atom types.
     - Link: https://github.com/aiqm/torchani
     - Link: https://doi.org/10.1038/s41597-020-0473-z
     - Elements: H, C, N, O
     
- ani2x
     - Description: The ANI-2x model is an ensemble of 8 networks that was
     trained on the ANI-2x dataset. The target level of theory is
     wB97X/6-31G(d). It predicts energies on HCNOFSCl elements exclusively
     it shouldn't be used with other atom types.
     - Link: https://github.com/aiqm/torchani
     - Elements: H, C, N, O, F, S, Cl

- ani1xbb
     - Description: An ANI-Based Reactive Potential for Small Organic Molecules
     - Link: https://doi.org/10.1021/acs.jctc.5c00347
     - Elements: H, C, N, O

- pm6ml
     - Description: PM6-ML: The Synergy of Semiempirical Quantum Chemistry and
       Machine Learning Transformed into a Practical Computational Method
     - Link: https://doi.org/10.1021/acs.jctc.4c01330
     - Elements: H, C, N, O, P, S, F, Cl, Br, I, Li, Na, K, Mg, Ca

- fennix-bio1m
     - Description: A Foundation Model for Accurate Atomistic Simulations in Drug Design.
     - Description: Medium sized fit to DFT functional: wB97M-D3BJ / aug-ccpVTZ / ccECP
     - Link: https://doi.org/10.26434/chemrxiv-2025-f1hgn-v4
     - Link: https://doi.org/10.48550/arXiv.2405.01491
     - Link: https://github.com/FeNNol-tools/FeNNol-PMC
     - Elements: B, Br, C, Ca, Cl, F, H, I, K, Li, Mg, N, Na, O, P, S, Si, Zn

- fennix-bio1s
     - Description: A Foundation Model for Accurate Atomistic Simulations in Drug Design.
     - Description: Small sized fit to DFT functional: wB97M-D3BJ / aug-ccpVTZ / ccECP
     - Link: https://doi.org/10.26434/chemrxiv-2025-f1hgn-v4
     - Link: https://doi.org/10.48550/arXiv.2405.01491
     - Link: https://github.com/FeNNol-tools/FeNNol-PMC
     - Elements: B, Br, C, Ca, Cl, F, H, I, K, Li, Mg, N, Na, O, P, S, Si, Zn

- orb-v3-direct-inf-omat
     - Description: A "direct" model doesn't appear to calculate forces from backpropagation and may not conserve energy, but it is much faster than a conservative model.
     - Description: Trained to the OMol25 dataset (wB97M-V/def2-TZVPD)
     - Link: https://github.com/orbital-materials/orb-models
     - Link: https://arxiv.org/abs/2504.06231
     - Elements: Authors did not specify, but OMol25 contains 83 elements

- orb-v3-conservative-inf-omat
     - Description: Conservative models compute forces and stress via backpropagation, which is a physically motivated choice that appears necessary for certain types of simulation such as NVE Molecular dynamics. Conservative models are significantly slower and use more memory than their direct counterparts.
     - Description: Trained to the OMol25 dataset (wB97M-V/def2-TZVPD)
     - Link: https://github.com/orbital-materials/orb-models
     - Link: https://arxiv.org/abs/2504.06231
     - Elements: Authors did not specify, but OMol25 contains 83 elements

- am1
    - Description: AM1 method run through mopac
    - Link: 
    - Elements: H, Li, Be, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, Cr, Zn, Ge, Br, Sn, I, Hg
    
- mndo
    - Description: AM1 method run through mopac
    - Link: 
    - Elements: H, Li, Be, B, C, N, O, F, Al, Si, P, S, Cl, Cr, Ge, Br, Sn, Hg, Pb, I
    
- mndod
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, Be, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, Zn, Ge, Br, Cd, Sn, I, Hg, Pb
    
- pm3
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, Li, Be, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Cs, Ba, Hg, Tl, Pb, Bi
    
- pm6
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi
    
- pm6-d3
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U, Np, Pu
    
- pm6-dh+
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, C, N, O, F, P, S, Cl, Br
    
- pm6-dh2
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, C, N, O, P, S, F, Cl, Br, I, Li, Na, K, Mg, Ca
    
- pm6-dh2x
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Lu
    
- pm6-d3h4
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, C, N, O, F, P, S, Cl, Br, I, Li, Na, K, Mg, Ca
    
- pm6-d3h4x
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, C, N, O, F, P, S, Cl, Br, I
    
- pmep
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, C, N, O, F, P, S, Cl, Br
    
- pm7
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi
    
- pm7-ts
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U
    
- rm1
    - Description: semiempirical method run through mopac
    - Link: 
    - Elements: H, C, N, O, P, S, F, Cl, Br, and I
    

The following methods are available only if you:

1. Install a separate conda environment and install
   ffpopt with:
   ACADEMIC=TRUE python3 -m pip install --group fairchem .
2. Create an account on https://huggingface.co/
3. Visit https://huggingface.co/facebook/OMol25 and
   request permission from the maintainers to access
   the models via their online form. Then wait for
   access to be granted.
4. Visit https://huggingface.co/settings/tokens
   and create an access token. Write down a copy
   of the access token.
5. Use the "hf" command (installed within group fairchem)
   to authenticate by typing: "hf auth login"
   At the prompt, write your access token and press enter.
   Answer "n" when asked "Add token as git credential?"
6. You should see "Login successful".
7. You can then use the models listed below.

- OMOL25-ESEN-SM-DIRECT
     Description: 
     Link: https://huggingface.co/facebook/OMol25
     Link: https://arxiv.org/abs/2505.08762
     Elements: 

- OMOL25-ESEN-SM-CONSERVING
     Description: 
     Link: https://huggingface.co/facebook/OMol25
     Link: https://arxiv.org/abs/2505.08762
     Elements:
     
- OMOL25-ESEN-MD-DIRECT
     Description: 
     Link: https://huggingface.co/facebook/OMol25
     Link: https://arxiv.org/abs/2505.08762
     Elements:
     
- OMOL25-ESEN-LG-DIRECT
     Description: 
     Link: https://huggingface.co/facebook/OMol25
     Link: https://arxiv.org/abs/2505.08762
     Elements: 
    

The following methods are not available in the list of models,
but they are used by ffpopt-RespFit and ffpopt-DeltaRespFit
to preduct partial charges.

- espaloma
     Description: Conformation-independent ML network trained to reproduce AM1-BCC charges. Only available in pytorch version of ffpopt.
     Link: https://doi.org/10.1021/acs.jpca.4c01287
     Link: https://github.com/choderalab/espaloma-charge
     Elements: H,C,N,O,P,S,B,F,Cl,Br,I

- hilfiker
     Description: conformation-dependent ML network trained to reproduce PBE0-D4(BJ)/def2-TZVP. Only available in pytorch version of ffpopt.
     Link: https://doi.org/10.48550/arXiv.2512.13579
     Link: https://github.com/mathilfiker/ml_for_charges
     Elements: H,C,N,O,F,P,S,Cl


## ffpopt-Optimze.py

   Reads parm7 & rst7 (or mol2 files),
   performs a geometry optimization, and writes
   the structure and energy to --oscan=oscan.xyz
   If --no-opt is provided, then it calculates a single point energy.
   If --iscan=iscan.xyz is provided, then it performs a geometry
   optimization of each structure within iscan.xyz.
   If --iscan and --no-opt are provided, then it performs a single
   point calculation for each structure.
   If --geometric-opt is given, then the geomeTRIC optimizer is used
   (the default is to optimize with ASE internal BFGS optimizer).

-   --oscan: str
       The output xyz filename

-   --iscan: str
       Optional. The input xyz filename. If not provided, the
       coordinates within --crd are used.

-   --ignore
       If present, then ignore the constraint definitions stored
       within the xyz file. The default behavior is to enforce
       the constraints specified within --iscan.

-   --constrain: str
       A comma-separated list of 2, 3, or 4 zero-based integers
       (the first atom is index 0).  The list can be appended with
       =value to specify a constraint value, otherwise the constraint
       value is the initial value calculated from the input
       coordinates. This option can be used multiple times to enforce
       multiple constraints.
       For example,
       --constrain='0,1=2.0' will constrain the bond between the
       first 2 atoms to be 2.0 Angstroms.
       --constrain='0,1,2=30.' will constrain the angle between the
       first 3 atoms to be 30. degrees.
       A list of 4 atoms constrains a dihedral.

##  ffpopt-DihedScan.py

   Reads parm7 & rst7 and a list of 4 atoms defining a dihedral angle.
   It then scans the dihedral with a series of relaxed optimizations.
   The procedure is to: 1. optimize for a minimum. 2. Create a schedule
   of angles that uniformly span [0,360). 3. Find the position in the
   schedule that best matches the optimized dihedral. 4. Sequentially
   optimize the dihedrals in the schedule in the foward direction until
   all 360 degrees are considered. 5. Repeat the scan in the reverse
   direction. 6. Sort the two scan and choose the geometry & energy
   that produced the lowest energy.  The output structures are written
   to --oscan.

-   --dihed: str
        A comma-separated list of 4 zero-based integers defining a
        dihedral angle.

-   --oscan: str
        The output XYZ file

-   --delta: int
        The scan spacing. Default: 10 degrees

-   --constrain: str
        Additional constraints applied to each structure.
        These can be bonds, angles, or dihedrals. These coordinates
        are not scanned. See ffpopt-Optimize.py for an extended
        description. This option can be used more than once.



## ffpopt-GenDihedFit.py

   Reads a json input file and optimize torsion parameters.

-   inp: str, positional argument
        The name of the json file.

-   --stride: int
        The stride used when reading the input scans. Default: 1

-   --nlmaxiter: int
        Maximum number of nonlinear optimization steps. Default: 200

-   --nlrhobeg: float
        Initial parameter displacements. Default: 0.25 kcal/mol.

-   --nltol: float
        Parameter optimization termination tolerance. Default: 0.01.



## ffpopt-DihedTwistWorkflow.py

   Reads a json input file and writes a bash script that uses
   ffpopt-DihedScan.py and ffpopt-GenDihedFit.py to iteratively
   parametrize torsion potentials.
   
-   --bond: str
        Two 0-based integers separated by a comma. This option
        can be used more than once.

-   --delta: int
        The scan spacing. Default: 10 degrees

-   --nprim: int
        The number of primitive torsion functions applied to each
        dihedral. Default: 3

-   --maxiter: int
        The number of training iterations. Each iteration repeats
        the sander scans with the current set of parameters.
        Default: 2

-   --bytype
        If present, then all parameters are based on atom types,
        and applied globally (even if the atom quartet isn't being
        scanned).  The global parameters are also written to a
        frcmod file.


## ffpopt-DihedTwistAnimate.py

   Builds animated convergence plots for twist workflow iterations.
   Default behavior uses scan outputs (`<prefix>_<i-j-k-l>.dat`)
   and compares `orig`/`itXX` scans against the high-level reference.
   You can also use `--source mfit` to animate from `mfit.*.dat`.

-   --input-dir: str
        Directory containing scan and/or mfit `.dat` files.
        Default: current directory.

-   --dihedral: str
        Optional target quartet (`i-j-k-l`). Can be used multiple
        times. If omitted, all discovered quartets are animated.

-   --source: str
        Data source: `scan` (default) or `mfit`.

-   --reference-prefix: str
        For `--source scan`, explicitly choose the high-level
        reference prefix if auto-detection is ambiguous.

-   --output: str
        Output animation path (`.gif`, `.mp4`, or `.m4v`).
        Default: `twist_convergence.gif`.
        If multiple quartets are animated, the quartet is appended
        to the output stem.

-   --fps: int
        Animation frame rate. Default: 2

-   --dpi: int
        Output resolution in DPI. Default: 150



## JSON input file format for ffpopt-GenDihedFit.py

 The json file contains 3 main keys: params, output, and systems.

 The params dictionary provides a petite list of unique dihedral
 parameters.  Its subkeys are "nprim" and "masks".

 The nprim value is the integer number of primitive dihedrals.
 For example, nprim=3 would model the dihedral with 3 torsion
 potentials with periodicities 1, 2, and 3. The corresponding
 3 force constants are to be determined.

 The masks value is either "null" or a list containing
 sublists of 4 atom type amber masks. If the value is null,
 then the parameter is
 "bespoke"; that is, the potential must be manually mapped
 to atom name quartets. In constrast, if a list of atom type
 masks are provided, then the potential is applied to all
 proper torsions with a matching set of atom types -- even
 if those torsions are not being scanned.  These are "global"
 parameters that can be written into a frcmod file and applied
 via tleap.
 An example, to define a global parameter applied to all
 cd-nf-ce-o torsions, one would set "masks": [ ["@%cd","@%nf","@%ce","@%o"] ].
 The value of masks is a list of lists because one could choose
 to parametrize a single potential for multiple atom type
 quartets.

 The "output" value is the name of an output frcmod file.
 All global parameters will be written to the frcmod file.
 If all parameters are bespoke, then the frcmod file is empty.

 The "systems" value is a list of dictionaries. Each element
 of the list describes a "system"; a system is characterized
 by a parm7 file. If you have multiple conformations and/or
 scans that all use the same parm7 file, then you only have
 1 system.  In contrast, you may be parametrizing a potential
 that exists in multiple molecules, in which each system
 refers to the same set of training parameters.

 A system's dictionary contains several keys.
 parm: the name of an amber parm7 input file.
 crd: the name of a formatted rst7 input file.
 output: the name of a python output file.
 params: a dictionary that describes how to map
 the parameters to the atoms.
 profiles: a list that collects high-level and
 low-level structures used to train the parameters.
 Each profile is a series of structures that
 share an arbitrary, yet common, zero-of-energy.

 The each key of the params dictionary is one of
 unique parameters. The value is a list-of-lists.
 Each sublist contains 4 elements: the amber masks
 that select each of the 4 atoms in the torsion
 BY ATOM NAME; e.g., [ "@C1", "@C2", "C3", "@H1" ]
 You do not need to map global parameters; therefore,
 if all parameters were global parameters, then the
 "params" dictionary would be empty, {}. If the same
 unique parameter should be applied to more than one
 quartet, then there would be a sublist for each
 quartet.

 The profiles value is a list of dictionaries.
 Each dictionary describes a "scan"; it contains the
 keys: hl, ll, name, and plots.
 The hl value is the name of a scan performed with a
 target model chemistry. These are the energies we
 would like to reproduce.
 The ll value is the scan performed with the force field
 before changing the parameters.
 The name value is a prefix applied to the plots generated
 during the nonlinear optimization procedure.
 The "plots" value is a list of strings used to further
 define filenames.

 The following is an example that parametrizes a single
 bespoke potential

```
 {
    "params": {
        "param_name_1": {
           "nprim": 3,
           "masks": null
         }
    },
    "output": "global.frcmod",
    "systems": [
       {
           "parm": "system1.parm7",
           "crd":  "system1.rst7",
           "output": "system1.py",
           "params": {
                  "param_name_1": [
                     [ "@AtomName1", "@AtomName2", "@AtomName3", "@AtomName4" ]
                  ]
           },
           "profiles": [
               {
                   "hl": "highlevel_scan.xyz",
                   "ll": "lowlevel_scan.xyz",
                   "name": "output_plot_prefix",
                   "plots": [
                        "param_name_1"
                   ]
               }
           ]
       }
 }
```

 The following is an example that optimizes a global parameter.

```
 {
    "params": {
        "param_name_1": {
           "nprim": 3,
           "masks": [ ["@%nf","@%ce","@%ca","@%ca"] ]
         }
    },
    "output": "global.frcmod",
    "systems": [
       {
           "parm": "system1.parm7",
           "crd":  "system1.rst7",
           "output": "system1.py",
           "params": {},
           "profiles": [
               {
                   "hl": "highlevel_scan.xyz",
                   "ll": "lowlevel_scan.xyz",
                   "name": "output_plot_prefix",
                   "plots": [
                        "param_name_1"
                   ]
               }
           ]
       }
 }
```

<!--
 A simple installation, where AmberTools (including pysander) and cmake have already been installed:
 cd ffpopt
 python3 -m pip install -r requirements.txt --prefix=${PWD}/local
 cd build
 bash ./run_cmake.sh
 cd ../
 export PATH="${PWD}/bin:${PATH}"
 export PYTHONPATH="${PWD}/lib/python3.12/site-packages:${PATH}"
-->
