#!/usr/bin/env python3


def ReadGeomsFromXYZ(stdargs,fname):
    """ Read geometries and constraints from an XYZ file.
     
    Parameters
    ----------
    stdargs
        Standard arguments containing the graph for constraints.
    fname : str
        The path to the XYZ file.
    
    Returns
    -------
    list of ase.Atoms
        A list of ASE Atoms objects read from the XYZ file.
    list of list of Constraint
        A list of lists of Constraint objects corresponding to each Atoms object.
        If no constraints are found, the list will contain None for that geometry.
    
    """

    import ase.io
    from . Constraints import info2constraints, FillConstraints
    
    geoms = ase.io.read(fname,index=":")
    cons = []
    rsts = []
    for atoms in geoms:
        try:
            atoms.info["energy"] = atoms.get_potential_energy()
        except:
            pass
        mycons=None
        if "constraints" in atoms.info:
            mycons = info2constraints(atoms,stdargs.graph)
            mycons = FillConstraints(atoms,mycons)
        cons.append(mycons)
        
    return geoms,cons



def FixParmedAtomicNumbers(mol):
    """Replace parmed atomic_numbers attribute for atom types matching gaff or gaff2

    Parameters
    ----------
    mol : parmed Structure or ResidueTemplate
        This object should contain an array mol.atoms with the .atomic_number attribute
    """
    
    # Example mapping for common GAFF2 atom types to atomic numbers
    # This is a simplified example; a more robust mapping might be needed for all GAFF2 types.
    gaff2_type_to_atomic_number = {
        'c': 6, 'c1': 6, 'c2': 6, 'c3': 6, 'ca': 6, 'cc': 6, 'cd': 6, 'ce': 6, 'cf': 6, 'cg': 6,
        'ch': 6, 'ci': 6, 'cj': 6, 'ck': 6, 'cl': 6, 'cm': 6, 'cn': 6, 'cp': 6, 'cq': 6, 'cr': 6,
        'cx': 6, 'cy': 6, 'cz': 6, 'n': 7, 'n1': 7, 'n2': 7, 'n3': 7, 'na': 7, 'nb': 7, 'nc': 7,
        'nd': 7, 'ne': 7, 'nf': 7, 'nh': 7, 'no': 7, 'nt': 7, 'o': 8, 'oh': 8, 'os': 8, 'ow': 8,
        'oy': 8, 'p': 15, 'pb': 15, 'pc': 15, 'pd': 15, 'pe': 15, 'pf': 15, 's': 16, 'sh': 16,
        'ss': 16, 'sx': 16, 'sy': 16, 'f': 9, 'cl': 17, 'br': 35, 'i': 53, 'h': 1, 'hc': 1, 'hp': 1,
        'hs': 1, 'hw': 1
    }

    gaff_type_to_atomic_number = {
        "c":6,"c1":6,"c2":6,"c3":6,"ca":6,"cp":6,"cq":6,"cc":6,"cd":6,"ce":6,"cf":6,"cg":6,"ch":6,"cx":6,"cy":6,"cu":6,"cv":6,"cz":6,"h1":1,"h2":1,"h3":1,"h4":1,"h5":1,"ha":1,"hc":1,"hn":1,"ho":1,"hp":1,"hs":1,"hw":1,"hx":1,"f":9, "cl":17,"br":35,"i":53, "n":7,  "n1":7, "n2":7, "n3":7, "n4":7, "na":7, "nb":7, "nc":7, "nd":7, "ne":7, "nf":7, "nh":7, "no":7, "ni":7, "nj":7, "nk":7, "nl":7, "nm":7, "nn":7, "np":7, "nq":7, "o":8,  "oh":8, "os":8, "op":8, "oq":8, "ow":8, "p2":15,"p3":15,"p4":15,"p5":15,"pb":15,"pc":15,"pd":15,"pe":15,"pf":15,"px":15,"py":15,"s":16, "s2":16,"s4":16,"s6":16,"sh":16,"ss":16,"sp":16,"sq":16,"sx":16,"sy":16
        }

    for atom in mol.atoms:
        if atom.type in gaff2_type_to_atomic_number:
            atom.atomic_number = gaff2_type_to_atomic_number[atom.type]
        elif atom.type in gaff_type_to_atomic_number:
            atom.atomic_number = gaff_type_to_atomic_number[atom.type]



def ConvertXyz2Pdb(xyz_filename,pdb_filename):
    """
    Converts an XYZ file to a basic PDB file without external libraries.
    Default values are used for PDB-specific fields.
    """
    try:
        with open(xyz_filename, 'r') as f_xyz:
            lines = f_xyz.readlines()
            if len(lines) < 2:
                print("Error: XYZ file is empty or invalid.")
                return

            num_atoms = int(lines[0].strip())
            # Second line is typically a comment/header, which we skip.
            atom_lines = lines[2:2 + num_atoms]
            
        with open(pdb_filename, 'w') as f_pdb:
            atom_serial = 1
            for line in atom_lines:
                # Parse XYZ line: format is [element] [x] [y] [z]
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                element = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                # Format for a PDB ATOM record (fixed-width columns):
                # ATOM  | serial | name | resName | chain | resSeq | x | y | z | occupancy | tempFactor | element | charge
                #
                # Format specification (columns):
                # 1-4: "ATOM"
                # 7-11: atom_serial (right-justified)
                # 13-16: atom_name (e.g., " C  ", " CA ") - use element for simplicity
                # 17: alternative location indicator (blank)
                # 18-20: resName (e.g., "MOL") - use "MOL" for simplicity
                # 22: chainID (e.g., "A")
                # 23-26: resSeq (e.g., 1)
                # 27: insertion code (blank)
                # 31-38: x (8.3f format)
                # 39-46: y (8.3f format)
                # 47-54: z (8.3f format)
                # 55-60: occupancy (6.2f format, default 1.00)
                # 61-66: tempFactor (6.2f format, default 0.00)
                # 77-78: element symbol (right-justified)

                pdb_line = (
                    f"ATOM  {atom_serial:5d} {element:>4s} MOL A{1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          "
                    f"{element:>2s}\n"
                )
                f_pdb.write(pdb_line)
                atom_serial += 1
            f_pdb.write("END\n")
        
        #print(f"Successfully converted {xyz_filename} to {pdb_filename}")

    except FileNotFoundError:
        print(f"Error: The file {xyz_filename} was not found.")
    except ValueError:
        print(f"Error: Problem parsing numerical data in the XYZ file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def ConvertPdb2Mol2(pdb_filename,charge,mol2_filename):
    #import subprocess as subp
    import parmed
    from openbabel import pybel
    # try:
    #     cmd = f"antechamber -fi pdb -i {pdb_filename} -fo mol2 -o {mol2_filename} -c gas -nc 0 -at gaff -an y -pf y -seq n" 
    #     result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
    #     print("STDOUT:", result.stdout)
    #     print("STDERR:", result.stderr)
    # except subprocess.CalledProcessError as e:
    #     print(f"Command failed with exit code {e.returncode}")
    #     print("Error output:", e.stderr)

    # mol = parmed.load_file(pdb_filename)
    # dq = charge / len(mol.atoms)
    # for a in mol.atoms:
    #     a.charge = dq

    # from collections import defaultdict as ddict
    # seen = ddict(int)
    # for a in mol.atoms:
    #     a.type = a.name
    #     seen[a.name] += 1
    #     i = seen[a.name]
    #     if i > 1:
    #         a.name = f"{a.name}{i}"
    # mol.save(mol2_filename, format='mol2', overwrite=True)
    
    # 1. Read the PDB file using OpenBabel
    # OpenBabel will automatically perceive bonds if CONECT records are missing
    mol = next(pybel.readfile("pdb", pdb_filename))
    
    # 2. Write to a temporary file or string to preserve connectivity
    # This step assigns atom types and bond orders needed for MOL2
    mol.write("mol2", mol2_filename, overwrite=True)
    
    # 3. Load the newly created MOL2 into ParmEd
    # Setting structure=True ensures it builds the dihedral list
    mol = parmed.load_file(mol2_filename, structure=True)
    
    #mol = parmed.load_file(pdb_filename)
    dq = charge / len(mol.atoms)
    for a in mol.atoms:
        a.charge = dq

    from collections import defaultdict as ddict
    seen = ddict(int)
    for a in mol.atoms:
        a.type = a.name
        seen[a.name] += 1
        i = seen[a.name]
        if i > 1:
            a.name = f"{a.name}{i}"
    mol.save(mol2_filename, format='mol2', overwrite=True)
    
        

def ConvertXyz2Mol2(xyz_filename,charge,mol2_filename=None):
    # pdb_filename = xyz_filename + ".pdb"
    # ConvertXyz2Pdb(xyz_filename,pdb_filename)
    # if mol2_filename is None:
    #     mol2_filename = xyz_filename + ".mol2"
    # ConvertPdb2Mol2(pdb_filename,charge,mol2_filename)
    # from pathlib import Path
    # file_path = Path(pdb_filename)
    # try:
    #     file_path.unlink(missing_ok=True)
    # except OSError as e:
    #     print(f"Error deleting file {file_path}: {e}")
    import parmed as pmd
    from openbabel import pybel

    if mol2_filename is None:
        mol2_filename = xyz_filename + ".mol2"
    
    # 1. Read XYZ using OpenBabel to perceive bonds/connectivity
    mol = next(pybel.readfile("xyz", xyz_filename))
    
    # 2. Export to a temporary mol2 string (OpenBabel handles bond perception)
    mol.write("mol2", mol2_filename, overwrite=True)

    # 3. Load into ParmEd
    # We load from the string to use ParmEd's powerful Structure class
    mol = pmd.load_file(mol2_filename,structure=True)

    dq = charge / len(mol.atoms)
    for a in mol.atoms:
        a.charge = dq

    from collections import defaultdict as ddict
    seen = ddict(int)
    for a in mol.atoms:
        a.type = a.name
        seen[a.name] += 1
        i = seen[a.name]
        if i > 1:
            a.name = f"{a.name}{i}"
    mol.save(mol2_filename, format='mol2', overwrite=True)
    
    # Alternatively, write to disk then load:
    #mol.write("mol2", mol2_filename, overwrite=True)
    #struct = pmd.load_file(output_mol2, structure=True)
        

def ReadMol2(fname):
    from parmed import load_file
    from parmed.topologyobjects import Dihedral
    mol = load_file(fname,structure=True)
    FixParmedAtomicNumbers(mol)
    # Explicitly crawl the bond graph to find dihedrals (A-B-C-D)
    # We iterate through atoms to find all unique 4-atom sequences
    discovered_dihedrals = []
    for a1 in mol.atoms:
        for a2 in a1.bond_partners:
            for a3 in a2.bond_partners:
                if a3 == a1: continue
                for a4 in a3.bond_partners:
                    if a4 == a2 or a4 == a1: continue
                    # Prevent double-counting (1-2-3-4 is the same as 4-3-2-1)
                    if a1.idx < a4.idx:
                        discovered_dihedrals.append(Dihedral(a1, a2, a3, a4))
    # Populate the structure's dihedral list
    mol.dihedrals.extend(discovered_dihedrals)
    return mol


def ReadXyz(xyz,charge):
    import parmed
    
    mol2 = xyz + ".mol2"
    ConvertXyz2Mol2(xyz,charge,mol2_filename=mol2)

    #mol = parmed.load_file(mol2)
    mol = ReadMol2(mol2)
    
    from pathlib import Path

    file_path = Path(mol2)

    try:
        file_path.unlink(missing_ok=True)
        #print(f"File {file_path} has been deleted (if it existed).")
    except OSError as e:
        # Handle other potential errors like permission issues
        print(f"Error deleting file {file_path}: {e}")

    return mol

