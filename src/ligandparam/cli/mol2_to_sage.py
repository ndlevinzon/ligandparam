from openff.toolkit import Molecule, ForceField, Topology, Quantity
from openff.interchange import Interchange
from openff.units import unit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
 
from ligandparam.stages import StageSageCreate

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate AMBER parameters from a mol2 file.")
    parser.add_argument("input_mol2", type=str, help="Path to the input mol2 file.")
    parser.add_argument("output_tag", type=str, help="Tag for the output files.")
    args = parser.parse_args()
    stage = StageSageCreate("sage_creation", args.input_mol2, os.getcwd(), out_parm=f"{args.output_tag}.parm7")
    stage.execute()
 
if __name__ == "__main__":
    main()