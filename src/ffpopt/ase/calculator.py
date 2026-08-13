#!/usr/bin/env python3

#
# The xtb-python package is no longer maintained and the developers
# recommend using tblite instead.
#
# To use the old XTB implementation, install the following packages:
#   conda install -y xtb xtb-python
#
# Then uncomment the following lines:
#
#from xtb.ase.calculator import XTB as OldXTB
#class XTBCalculator(OldXTB):
#     """ASE calculator for XTB with net charge."""
#     def __init__(self, *args, charge=0, **kwargs):
#         self.charge = charge
#         super().__init__(*args, **kwargs)
#     def _create_api_calculator(self):
#         import numpy as np
#         initial_charges = np.zeros(len(self.atoms))
#         initial_charges[0] = self.charge
#         self.atoms.set_initial_charges(initial_charges)
#         return super()._create_api_calculator()
#
#
# To use the new tblite implementation, install the following packages:
#   python3 -m pip install tblite
#
# Then load the calculator:
#  from tblite.ase import TBLite
#  calc = TBLite(method="GFN2-xTB",charge=self.charge,verbosity=-1)
#

import os
for key in ["OMP_NUM_THREADS","DP_INTRA_OP_PARALLELISM_THREADS","DP_INTER_OP_PARALLELISM_THREADS"]:
    if key not in os.environ:
        os.environ[key] = "1"

from ase.calculators.calculator import Calculator, all_changes
from collections import defaultdict as ddict

from ffpopt.AmberParm import CopyParm


def _sander_energy_forces_direct(positions):
    """Call the active pysander session without ASE Atoms rebuilds.

    Returns ``(energy_eV, forces_eV_per_Ang)`` or ``None`` if sander is not
    set up / the call fails (caller should fall back to ASE).
    """
    try:
        import numpy as np
        import sander
        from ase import units
    except Exception:
        return None
    try:
        crd = np.reshape(np.asarray(positions, dtype=float), (1, -1, 3))
        sander.set_positions(crd)
        e, f = sander.energy_forces()
        energy = float(e.tot) * units.kcal / units.mol
        forces = (
            np.reshape(np.asarray(f, dtype=float), (-1, 3)) * units.kcal / units.mol
        )
        return energy, forces
    except Exception:
        return None


def _scratch_atoms_energy_forces(calc_wrapper, underlying_calc):
    """Reuse one ASE Atoms shell for SANDER-style calculators (avoid rebuilds).

    Prefer a direct ``sander.set_positions`` / ``energy_forces`` path when the
    ASE Amber SANDER calculator already owns the global pysander session.
    """
    import ase

    atoms = calc_wrapper.atoms
    positions = atoms.get_positions()
    # Tight path: skip ASE Calculator protocol for MM steps.
    if getattr(underlying_calc, "name", "").lower() == "sander" or type(
        underlying_calc
    ).__name__ == "SANDER":
        direct = _sander_energy_forces_direct(positions)
        if direct is not None:
            return direct

    charges = atoms.get_initial_charges()
    scratch = getattr(calc_wrapper, "_scratch_atoms", None)
    if scratch is None or len(scratch) != len(atoms):
        eles = atoms.get_chemical_symbols()
        atlist = "".join(["%s1" % (ele,) for ele in eles])
        scratch = ase.Atoms(atlist, positions=positions, charges=charges)
        scratch.calc = underlying_calc
        calc_wrapper._scratch_atoms = scratch
    else:
        scratch.set_positions(positions)
        if charges is not None and len(charges) == len(scratch):
            scratch.set_initial_charges(charges)
        if scratch.calc is not underlying_calc:
            scratch.calc = underlying_calc
    return scratch.get_potential_energy(), scratch.get_forces()


class GenCalculator(Calculator):

    implemented_properties = ['energy','forces','free_energy']
    nolabel=True

    def __init__(self,mode,charge,spin,parm,crd,**kwargs):
        from parmed import load_file

        # print("input mode  =",mode)
        # print("input charge=",charge)
        # print("input spin  =",spin)
        # print("input parm  =",parm)
        # print("input crd   =",crd)
        # print("input kwargs=",kwargs)
        # print("input num_threads=",kwargs.get("num_threads",None))
        
        self.mode=mode.upper()
        self.charge=charge
        self.spin=spin
        self.parm=parm
        self.crd=crd
        
        # if mol is not None:
        #     self.mol = mol
        # else:
        #     try:
        #         self.mol = load_file(parm,xyz=crd)
        #     except:
        #         if ".mol2" in parm:
        #             from .. Reader import ReadMol2
        #             self.mol = ReadMol2(parm)
        #         else:
        #             self.mol = load_file(parm)
        #self.charge = int(round(sum([a.charge for a in self.mol.atoms])))
        #self.mode = mode.upper()
        
        if self.mode == "SANDER":
            import subprocess
            #print(self.parm,self.crd)
            #subprocess.run(["ls", "-l", self.parm])
            #subprocess.run(["ls", "-l", self.crd])
            self.calc = SanderCalculator(parm=self.parm,crd=self.crd,**kwargs)
        elif self.mode in ["DFTB3","DFTB2","AM1D"]:
            self.calc = SanderSQMCalculator\
                (parm=self.parm,crd=self.crd,
                 charge=self.charge,
                 theory=self.mode,
                 **kwargs)
        elif self.mode == "QDPI2":
            import importlib
            import importlib.resources
            from pathlib import Path
            data_file_name = "pkgdata/qdpi/qdpi-2.0.pb"
            data_path = importlib.resources.files("ffpopt") / data_file_name
            model = kwargs["mfile"] if kwargs.get("mfile") is not None else data_path
            mlp = DPModel(model)
            self.calc = QDpi2Calculator(mlp,self.charge,**kwargs)
        elif self.mode == "XTB":
            from .tblite_scf import make_tblite_calculator

            self.calc = make_tblite_calculator(charge=self.charge)
        elif "MACE" in self.mode:
            from mace.calculators import MACECalculator
            import importlib
            import importlib.resources
            from pathlib import Path

            lmode = mode.lower()
            if lmode == "mace-off23_medium":
                data_file_name = "pkgdata/mace-off/mace_off23/MACE-OFF23_medium.model"
            elif lmode == "mace-off23_large":
                data_file_name = "pkgdata/mace-off/mace_off23/MACE-OFF23_large.model"
            elif lmode == "mace-off23_small":
                data_file_name = "pkgdata/mace-off/mace_off23/MACE-OFF23_small.model"
            elif lmode == "mace-off23b_medium":
                data_file_name = "pkgdata/mace-off/mace_off23/MACE-OFF23b_medium.model"
            elif lmode == "mace-off24_medium":
                data_file_name = "pkgdata/mace-off/mace_off24/MACE-OFF24_medium.model"
            elif lmode == "mace":
                data_file_name = "pkgdata/mace-off/mace_off23/MACE-OFF23_medium.model"
                
            data_path = importlib.resources.files("ffpopt") / data_file_name
            model = kwargs["mfile"] if kwargs.get("mfile") is not None else data_path

            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
            
            self.calc = MACECalculator(model_paths=model,device=device)
            
        elif "PYSCFNEO" in self.mode:
            
            if len(self.mode.split("/")) != 5:
                raise ValueError("Invalid mode format for PYSCFNEO")
            model, xc, basis, quantum_nuc, nuc_basis = self.mode.split("/")
            quantum_nuc = quantum_nuc.split(",")
            
            if quantum_nuc == ['']:
                print("Using PySCF model without NEO")
                self.calc = PySCF_DFT_Calculator(basis=basis, xc=xc, charge=self.charge, spin=self.spin-1)
            else:
                from pyscf.neo import Pyscf_NEO
                self.calc = Pyscf_NEO(basis=basis, xc=xc, charge=self.charge, spin=self.spin-1, quantum_nuc=quantum_nuc, nuc_basis=nuc_basis)
        elif "AIMNET" in self.mode:
            from aimnet.calculators import AIMNet2ASE
            self.calc = AIMNet2ASE(base_calc=self.mode.lower(),charge=self.charge)
        elif "OMOL25" in self.mode:
            print("Using OMOL25 model")
            from fairchem.core import pretrained_mlip, FAIRChemCalculator
            if self.mode == "OMOL25-ESEN-SM-DIRECT":
                predictor = pretrained_mlip.get_predict_unit("esen-sm-direct-all-omol", device="cpu")
                print("Using OMOL25 eSEN-sm-direct model")
            elif self.mode == "OMOL25-ESEN-SM-CONSERVING":
                predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-omol", device="cpu")
                print("Using OMOL25 eSEN-sm-conserving model")
            elif self.mode == "OMOL25-ESEN-MD-DIRECT":
                predictor = pretrained_mlip.get_predict_unit("esen-md-direct-all-omol", device="cpu")
                print("Using OMOL25 eSEN-md-direct model")
            elif self.mode == "OMOL25-ESEN-LG-DIRECT":
                raise NotImplementedError("OMOL25 eSEN-lg-direct is coming soon from FAIRCHEM.")
            else:
                raise Exception(f"Expected OMOL25-ESEN-SM-DIRECT, OMOL25-ESEN-MD-DIRECT, or OMOL25-ESEN-SM-CONSERVING, but received {self.mode}")
            self.calc = FAIRChemCalculator(predictor, task_name="omol")
        elif "ANI" in self.mode:
            import torchani.models

            try:
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            except ImportError:
                device = 'cpu'
            
            if self.mode == "ANI1CCX":
                try:
                    model = torchani.models.ANI1ccx().to(device)
                    self.calc = torchani.ase.Calculator(model)
                except Exception as exc:
                    import sys
                    sys.stderr.write(
                        f"[ffpopt] ANI1CCX device path failed "
                        f"({type(exc).__name__}: {exc}); using .ase() fallback\n"
                    )
                    self.calc = torchani.models.ANI1ccx().ase()
            elif self.mode == "ANI1X":
                try:
                    model = torchani.models.ANI1x().to(device)
                    self.calc = torchani.ase.Calculator(model)
                except Exception as exc:
                    import sys
                    sys.stderr.write(
                        f"[ffpopt] ANI1X device path failed "
                        f"({type(exc).__name__}: {exc}); using .ase() fallback\n"
                    )
                    self.calc = torchani.models.ANI1x().ase()
            elif self.mode == "ANI2X":
                try:
                    model = torchani.models.ANI2x().to(device)
                    self.calc = torchani.ase.Calculator(model)
                except Exception as exc:
                    import sys
                    sys.stderr.write(
                        f"[ffpopt] ANI2X device path failed "
                        f"({type(exc).__name__}: {exc}); using .ase() fallback\n"
                    )
                    self.calc = torchani.models.ANI2x().ase()
            else:
                raise Exception(f"Expected ani1x, ani2x, ani1ccx, or ani1xbb but received {self.mode}")

        elif "FENNIX" in self.mode:
            from . fennolase import FENNIXCalculator
            import importlib
            import importlib.resources
            lmode = self.mode.lower()
            data_path = importlib.resources.files("ffpopt")
            if "fennix-bio1m" == lmode:
                data_file_name = "pkgdata/fennix/fennix-bio1M.fnx"
                model = data_path / data_file_name
            elif "fennix-bio1s" == lmode:
                data_file_name = "pkgdata/fennix/fennix-bio1S.fnx"
                model = data_path / data_file_name
            elif kwargs.get("mfile") is not None:
                model = kwargs["mfile"]
            else:
                raise Exception(f"Unknown mode: {self.mode}")

            import os
            platform = os.environ.get("JAX_PLATFORMS","cpu").lower()
            if platform == "cpu":
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            
            self.calc = FENNIXCalculator(model,charge=self.charge)
            
        elif self.mode in ['AM1', 'MNDO', 'MNDOD', 'PM3', 'PM6', 'PM6-D3', 'PM6-DH+',
                           'PM6-DH2', 'PM6-DH2X', 'PM6-D3H4', 'PM6-D3H4X', 'PMEP', 'PM7',
                           'PM7-TS', 'RM1']:
            
            from . mopac import MOPAC            
            self.calc = MOPAC(method=self.mode, charge=self.charge)
            
        elif self.mode == "PM6ML":
            import importlib
            import importlib.resources
            from pathlib import Path
            #from . mopac import MOPAC
            
            data_file_name = "pkgdata/pm6ml/PM6-ML_correction_seed8_best.ckpt"
            data_path = importlib.resources.files("ffpopt") / data_file_name
            model = kwargs["mfile"] if kwargs.get("mfile") is not None else data_path
            
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except:
                device = 'cpu'
                
            self.calc = PM6MLCalculator(str(model),charge=self.charge,device=device)

            #self.calc = MOPAC(command="mopac-ml mopac.mop", method='PM6ML')

        elif "/" in self.mode:
            from ase.calculators.psi4 import Psi4
            import os
            import sys
            cwd = os.getcwd()
            if "PSI_SCRATCH" not in os.environ:
                sys.stderr.write(f"PSI_SCRATCH is unset. Setting to {cwd}\n")
                os.environ["PSI_SCRATCH"] = cwd
            else:
                sdir = os.environ["PSI_SCRATCH"]
                if len(sdir) == 0:
                    sys.stderr.write(f"PSI_SCRATCH='{sdir}' does not refer to a directory. Setting to {cwd}\n")
                    os.environ["PSI_SCRATCH"] = cwd
                else:
                    if not os.path.isdir(sdir):
                        sys.stderr.write(f"PSI_SCRATCH='{sdir}' does not exist. Setting to {cwd}\n")
                        os.environ["PSI_SCRATCH"] = cwd

            
            theory,basis = self.mode.split("/")
            memory = kwargs.get("memory","1gb")
            num_threads = int(kwargs.get("num_threads",4))
            print(f"Creating psi4 calculator with num_threads={num_threads}, method={theory}, basis={basis}, memory={memory}")
            self.calc = Psi4(method=theory,
                             memory=memory,
                             basis=basis,
                             num_threads=num_threads,
                             charge=self.charge,
                             multiplicity=self.spin)

        elif "ORB-" in self.mode:
            # precision='float32-highest'
            #           'float32-high'
            #           'float64'
            # orb-v3-direct-inf-omat
            # orb-v3-conservative-inf-omat
            # orb-v3-conservative-120-omat

            self.calc = WrappedORBCalculator(self.mode,"float32-high","cuda",self.charge,1)
        elif "DUMMY" in self.mode or "NULL" in self.mode:
            print("Using dummy calculator")
            self.calc = DummyCalculator()
        else:
            raise Exception(f"Unknown mode: {self.mode}")


        # if "twistrst" in kwargs:
        #     import json
        #     from .. Restraints import TwistRestraint
        #     #print("Processing:",str(kwargs["twistrst"]))
        #     cons = kwargs["twistrst"]
        #     self.calc = RestrainedCalculator(self.calc,cons,**kwargs)

        # if "restraintfile" in kwargs:
        #     import json
        #     from .. Restraints import TwistRestraint
        #     with open(kwargs["restraintfile"], 'r') as file:
        #         data = json.load(file)
        #     if "twistrst" in data:
        #         cons = []
        #         myrst = data["twistrst"]
        #         for rst in myrst:
        #             cons.append( TwistRestraint(rst[0],rst[1],rst[2]) )
        #             #print(rst)
        #         self.calc = RestrainedCalculator(self.calc,cons,**kwargs)
        #     #exit(0)
        
        Calculator.__init__(self,**kwargs)

    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=all_changes):
        import numpy as np
        import ase
        if self.charge is not None:
            atoms.info["charge"] = self.charge
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        if self.mode == "XTB":
            from .tblite_scf import run_tblite_with_scf_retries

            energy, forces, self.calc = run_tblite_with_scf_retries(
                atoms, self.calc
            )
        else:
            atoms.calc = self.calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        #self.calc.calculate(atoms=atoms,properties=properties,system_changes=system_changes)

class PySCF_DFT_Calculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(self, xc, basis, charge=0, spin=0, **kwargs):
        self.xc = xc
        self.basis = basis
        self.charge = charge
        self.spin = spin
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        import numpy as np
        from pyscf import gto, dft

        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        nat = len(self.atoms)
        # Build PySCF molecule
        atom_list = []
        coords = self.atoms.get_positions()
        symbols = self.atoms.get_chemical_symbols()
        for s, p in zip(symbols, coords):
            atom_list.append(f"{s} {p[0]} {p[1]} {p[2]}")

        mol = gto.M(atom=atom_list, basis=self.basis, charge=self.charge, spin=self.spin, unit='Angstrom')
        # choose RKS or UKS depending on spin
        if self.spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = self.xc
        mf.kernel()

        # Energy: Hartree -> eV
        HARTREE_TO_EV = 27.211386245988
        energy_ev = mf.e_tot * HARTREE_TO_EV

        # Forces: PySCF returns nuclear gradients (dE/dR) in Hartree/Bohr
        # forces = -grad ; convert to eV/Angstrom
        BOHR_TO_ANG = 0.529177210903
        FORCE_CONV = HARTREE_TO_EV / BOHR_TO_ANG
        grad = mf.nuc_grad_method().grad()  # shape (nat,3) in Hartree/Bohr
        forces_evA = -grad * FORCE_CONV

        self.results['energy'] = energy_ev
        self.results['free_energy'] = energy_ev
        self.results['forces'] = np.array(forces_evA)



class DPModel(object):
    
    def __init__(self,fname):
        
        from deepmd.infer import DeepPot
        self.dp = DeepPot(fname)

        # max_retries = 1  # Set a limit to prevent infinite loops
        # retry_delay = 1  # Delay in seconds

        # for attempt in range(max_retries):
        #     try:
        #         self.dp = DeepPot(fname)
        #         break
        #     except Exception as e:
        #         print(f"An exception occurred when trying to load DeepPot within ffpopt/ase/calculator.py class DPModel: {e}")
        #         if attempt < max_retries - 1:
        #             import time
        #             print(f"Waiting {retry_delay} second(s) before retrying...")
        #             time.sleep(retry_delay)
        #         else:
        #             raise Exception(e)
                    
                


        self.cell   = None
        self.rcut   = self.dp.get_rcut()
        self.ntypes = self.dp.get_ntypes()
        self.tmap   = self.dp.get_type_map()

    def GetTypeIdxFromSymbol(self,ele):
        idx = None
        if ele in self.tmap:
            idx = self.tmap.index(ele)
        return idx

    def GetTypeIdxs(self,eles):
        return [ self.GetTypeIdxFromSymbol(ele) for ele in eles ]

    def CalcEne(self,eles,crds):
        import numpy as np
        
        #from dpdata.unit import EnergyConversion
        #from dpdata.unit import ForceConversion
        #from dpdata.unit import LengthConversion

        # Bohr/angstrom
        #length_convert = LengthConversion("angstrom", "bohr").value()
        # hartree/eV
        #energy_convert = EnergyConversion("eV", "hartree").value()
        # hartree/bohr / eV/angstrom
        #force_convert = ForceConversion("eV/angstrom", "hartree/bohr").value()

        energy_convert = 1
        force_convert = 1
        
        coord = np.array(crds).reshape([1, -1])
        atype = self.GetTypeIdxs(eles)
        e, f, v = self.dp.eval(coord, self.cell, atype)
        f = f[0] * force_convert
        e = e[0][0] * energy_convert

        # print("$coord")
        # for i in range(len(eles)):
        #     print("%20.14f %20.14f %20.14f %s"%(
        #         crds[i,0] * length_convert,
        #         crds[i,1] * length_convert,
        #         crds[i,2] * length_convert,
        #         eles[i]))
        # print("$end")
        
        return e,f


    

    


class QDpi2Calculator(Calculator):

    implemented_properties = ['energy','forces','free_energy']
    nolabel=True
    
    def __init__(self,dpmodel,charge,force_components="both",**kwargs):
        from .tblite_scf import make_tblite_calculator

        self.dpmodel = dpmodel
        self.charge = charge
        # both | xtb | dp — xtb-only / dp-only used for cheap opts under --fast.
        self.force_components = str(force_components or "both").strip().lower()
        self.xtbcalc = make_tblite_calculator(charge=self.charge)
        Calculator.__init__(self,**kwargs)
        
    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=all_changes):
        import numpy as np
        import ase
        from .tblite_scf import run_tblite_with_scf_retries

        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        natoms = len(self.atoms)
        forces = np.zeros((natoms, 3))
        energy = 0.0

        eles = self.atoms.get_chemical_symbols()
        crds = self.atoms.get_positions()
        qs = self.atoms.get_initial_charges()
        mode = (self.force_components or "both").lower()
        if mode in {"full", "all"}:
            mode = "both"

        if mode in {"both", "dp", "deepmd", "deeppot", "ml"}:
            e, f = self.dpmodel.CalcEne(eles, crds)
            energy += e
            forces = forces + f

        if mode in {"both", "xtb", "gfn2", "tblite"}:
            atlist = "".join(["%s1" % (ele,) for ele in eles])
            xtb_atoms = ase.Atoms(atlist, positions=crds, charges=qs)
            e2, f2, self.xtbcalc = run_tblite_with_scf_retries(
                xtb_atoms, self.xtbcalc
            )
            energy += e2
            forces = forces + f2

        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces


class SanderCalculator(Calculator):

    implemented_properties = ['energy','forces','free_energy']
    nolabel=True

    def __init__(self,parm="tmp.parm7",crd="tmp.rst7",mol=None,**kwargs):
        from ase.calculators.amber import SANDER
        import sander
        from parmed import load_file
        self.parm = parm
        if mol is not None:
            self.mol = CopyParm(mol)
        else:
            try:
                self.mol = load_file(parm,xyz=crd)
            except Exception as exc:
                import sys
                sys.stderr.write(
                    f"[ffpopt] SANDER load_file({parm!r}, xyz={crd!r}) failed "
                    f"({type(exc).__name__}: {exc}); trying ReadMol2\n"
                )
                from .. Reader import ReadMol2
                self.mol = ReadMol2(parm)

                
        self.mm_options = sander.gas_input()
        self.mm_options.cut=99.
        self.mm_options.ntc=1
        self.mm_options.ntf=1
        self.calc = SANDER(top=self.parm,
                           crd=self.mol,
                           mm_options=self.mm_options)
        Calculator.__init__(self,**kwargs)

    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        # Prefer direct pysander; fall back to ASE SANDER wrapper.
        direct = _sander_energy_forces_direct(self.atoms.get_positions())
        if direct is not None:
            energy, forces = direct
        else:
            energy, forces = _scratch_atoms_energy_forces(self, self.calc)
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces




class SanderSQMCalculator(Calculator):

    implemented_properties = ['energy','forces','free_energy']
    nolabel=True

    def __init__(self,
                 parm="tmp.parm7",
                 crd="tmp.rst7",
                 charge=0,theory="DFTB3",
                 mol=None,**kwargs):
        
        from ase.calculators.amber import SANDER
        import sander
        from parmed import load_file
        
        self.parm = parm
        if mol is not None:
            self.mol = CopyParm(mol)
        else:
            try:
                self.mol = load_file(parm,xyz=crd)
            except Exception as exc:
                import sys
                sys.stderr.write(
                    f"[ffpopt] SANDER load_file({parm!r}, xyz={crd!r}) failed "
                    f"({type(exc).__name__}: {exc}); trying ReadMol2\n"
                )
                from .. Reader import ReadMol2
                self.mol = ReadMol2(parm)

        self.mm_options = sander.gas_input()
        self.mm_options.cut = 99.
        self.mm_options.ntc = 1
        self.mm_options.ntf = 1
        self.mm_options.ifqnt = 1
        self.mm_options.ntb = 0
        
        self.qm_options = sander.qm_input()
        self.qm_options.qmmask = ":1"
        self.qm_options.qm_theory = theory
        self.qm_options.qmcharge = charge
        self.qm_options.spin = 1
        self.qm_options.qmshake = 0
        self.qm_options.qmmm_switch = 1
        self.qm_options.scfconv = 1.e-10
        self.qm_options.tight_p_conv = 1
        self.qm_options.diag_routine = 0
        self.qm_options.pseudo_diag = 1
        self.qm_options.dftb_maxiter = 100

        
        self.calc = SANDER(top=self.parm,
                           crd=self.mol,
                           mm_options=self.mm_options,
                           qm_options=self.qm_options)
        
        
        Calculator.__init__(self,**kwargs)

    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        energy, forces = _scratch_atoms_energy_forces(self, self.calc)
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces


        

class RestrainedCalculator(Calculator):

    implemented_properties = ['energy','forces','free_energy']
    nolabel=True

    def __init__(self,calc,restraints,**kwargs):
        from ase.calculators.amber import SANDER
        self.calc = calc
        self.restraints = restraints
        Calculator.__init__(self,**kwargs)


    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=all_changes):
        
        import numpy as np
        
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        crds = self.atoms.get_positions()
        # Unwrap GenCalculator → SanderCalculator when possible for the
        # direct pysander path; otherwise evaluate through the base calc.
        inner = getattr(self.calc, "calc", self.calc)
        if isinstance(inner, SanderCalculator):
            energy, forces = _scratch_atoms_energy_forces(self, inner.calc)
        else:
            energy, forces = _scratch_atoms_energy_forces(self, self.calc)

        for rst in self.restraints:
            e2, f2 = rst.GetValueAndGradients(crds)
            energy += e2
            forces -= f2
        
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        

class PM6MLCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    

    def __init__(self, ckpt_path, charge=0, device='cpu', **kwargs):
        super().__init__(**kwargs)
        
        import torch
        from . mopac import MOPAC
        from torchmdnet.models.model import load_model
        
        #self.parameters['charge'] = charge
        self.charge = charge
        
        # 1. Load ML Model
        #print(f"device={device}")
        #self.device = torch.device(device)
        self.device = device
        self.model = load_model(ckpt_path, derivative=True).to(self.device)
        self.model.eval()
        
        # 2. Setup MOPAC with the charge keyword
        # MOPAC uses the 'CHARGE' keyword in its input file
        self.base_calc = MOPAC(method='PM6', charge=charge) #keywords=f'CHARGE={charge}')

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        import torch
        import ase
        from ase.units import Debye, kcal, mol
        from dftd3.interface import RationalDampingParam, DispersionModel
        from ase.data import atomic_numbers
        from ase.symbols import symbols2numbers
        import numpy as np
        
        super().calculate(atoms, properties, system_changes)
        
        HARTREE2KJMOL = 627.5094740631 * 4.184
        BOHR2ANGSTROM = 0.529177210903
        z_to_atype = {35: 1, 6: 3, 20: 5, 17: 7, 9: 9, 1: 10, 53: 12, 19: 13, 3: 14, 12: 15, 7: 17, 11: 19, 8: 21, 15: 23, 16: 26}
        
        eles = self.atoms.get_chemical_symbols()
        crds = self.atoms.get_positions()
        qs = self.atoms.get_initial_charges()
        atnums = symbols2numbers(eles)
        atlist = "".join( ["%s1"%(ele) for ele in eles ] )
        atoms = ase.Atoms(atlist,positions=crds,charges=qs)
        atoms.calc =  self.base_calc
        
        
        # Update MOPAC charge if it changed
        #current_charge = self.parameters['charge']
        current_charge = 0
        if self.charge is not None:
            atoms.info["charge"] = self.charge
            current_charge = self.charge
        atoms.set_initial_charges( [ self.charge / len(eles) ] * len(eles) )
        #self.base_calc.set(keywords=f'CHARGE={current_charge}')

        # --- Step A: Base MOPAC ---
        self.base_calc.calculate(atoms, properties, system_changes)
        base_energy = self.base_calc.results['energy']
        base_forces = self.base_calc.results['forces']

        # --- Step B: ML Correction ---
        #pos = torch.tensor(atoms.get_positions(), dtype=torch.float).to(self.device)
        #z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).to(self.device)
        #pos.requires_grad_(True)
        #energy_corr, forces_corr = self.model(z, pos)
        #energy_corr = energy_corr.detach().cpu().item()
        #forces_corr = forces_corr.detach().cpu().numpy().squeeze()
        #energy_corr *= kcal / mol
        #forces_corr *= kcal / mol


        a_types = [ z_to_atype[atom] for atom in atnums ]
        disp = DispersionModel(
            numbers=np.array(atnums), positions=np.array(crds) / BOHR2ANGSTROM
        )
        res = disp.get_dispersion(
            RationalDampingParam(s6=1.0, s8=0.3908, a1=0.566, a2=3.128), grad=True
        )
        dftd3_corr = res.get("energy") * HARTREE2KJMOL
        # Energy calculation
        ene, forces = pm6ml_energy_forces(a_types, crds, self.model, self.device)
        # ... and gradient
        # dftd4_grad should be kj/mol
        dftd3_grad = res.get("gradient") * HARTREE2KJMOL /BOHR2ANGSTROM
        # Multiply with -1 to get gradient from forces
        # grad should be kcal/mol
        grad = (forces * -1.0 + dftd3_grad) / 4.184

        energy_corr = (ene + dftd3_corr) * kcal / mol
        forces_corr = -grad * kcal / mol
        
        # --- Step C: Combine ---
        self.results['energy'] = base_energy + energy_corr
        self.results['forces'] = base_forces + forces_corr
        #self.results['energy'] = base_energy
        #self.results['forces'] = base_forces
        
        #print(pos)
        #print(base_energy)

def pm6ml_energy_forces(elem, geom, model, device):
    import torch
    types = torch.tensor(elem, dtype=torch.long)
    types = types.to(device)
    pos = torch.tensor(geom, dtype=torch.float32)
    pos = pos.to(device)
    energy, forces = model.forward(types, pos)  # ,batch)
    forces = forces.detach()
    if device != "cpu":
        forces = forces.cpu()
    forces = forces.numpy()
    #print("energy=",energy.item())
    #print("forces=",forces)
    return (energy.item(), forces)
    





class WrappedORBCalculator(Calculator):

    implemented_properties = ['energy','forces','free_energy']
    nolabel=True

    def __init__(self,model,ftype,device,charge,spin,mol=None,**kwargs):

        import ase
        from ase.build import bulk
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
        from orb_models.forcefield.inference.calculator import ORBCalculator
        
        self.model = model.lower()
        self.device = device
        self.ftype = ftype # "float32-high", "float32-highest"
        self.charge = charge
        self.spin = spin
        mydevice = device
        if self.model in ORB_PRETRAINED_MODELS:
            try:
                orbff,atoms_adapter = ORB_PRETRAINED_MODELS[self.model](
                    device=mydevice,
                    precision=self.ftype
                )
            except Exception as exc:
                import sys
                sys.stderr.write(
                    f"[ffpopt] ORB model init on {mydevice!r} failed "
                    f"({type(exc).__name__}: {exc}); retrying on cpu\n"
                )
                mydevice="cpu"
                orbff,atoms_adapter = ORB_PRETRAINED_MODELS[self.model](
                    device=mydevice,
                    precision=self.ftype
                )
        else:
            raise Exception(f"Model {self.model} not in {ORB_PRETRAINED_MODELS.keys()}")
                
        self.calc = ORBCalculator(
            orbff,
            atoms_adapter=atoms_adapter,
            device=mydevice
        )
        
        Calculator.__init__(self,**kwargs)

    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=all_changes):
        import numpy as np
        import ase
        import sys
                
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        
        eles = self.atoms.get_chemical_symbols()
        crds = self.atoms.get_positions()
        qs   = self.atoms.get_initial_charges()
        atlist = "".join( ["%s1"%(ele) for ele in eles ] )
        atoms = ase.Atoms(atlist,positions=crds,charges=qs)
        atoms.info["charge"] = self.charge
        atoms.info["spin"] = self.spin
        atoms.calc =  self.calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces




class DummyCalculator(Calculator):

    implemented_properties = ['energy','forces','free_energy']
    nolabel=True

    def __init__(self,**kwargs):
        Calculator.__init__(self,**kwargs)


    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=all_changes):
        
        import numpy as np
        import ase
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        crds = self.atoms.get_positions()
        self.results['energy'] = 0
        self.results['free_energy'] = 0
        self.results['forces'] = np.zeros( crds.shape )
        

class CartCalculator(Calculator):

    implemented_properties = ['energy','forces','free_energy']
    nolabel=True

    def __init__(self,crds,wts,**kwargs):
        import numpy as np
        self.crds = np.array(crds)
        self.wts = np.array(wts)
        if self.crds.shape[0] != self.wts.shape[0]:
            raise Exception(f"Size mismatch {self.crds.shape[0]} vs {self.wts.shape[0]}")
        Calculator.__init__(self,**kwargs)


    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=all_changes):
        
        import numpy as np
        import ase
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        crds = self.atoms.get_positions()
        e = 0
        grd = np.zeros( crds.shape )
        for a in range( crds.shape[0] ):
            d = crds[a,:]-self.crds[a,:]
            e += self.wts[a] * np.dot(d,d)
            grd[a,:] = 2 * self.wts[a] * d[:]
        #print(e,grd)
        
        self.results['energy'] = e
        self.results['free_energy'] = e
        self.results['forces'] = -grd[:,:]
        
