#!/usr/bin/env python3

# def FindFuncGrps_from_ChemFG_Tool(mol):
#     from . ChemFG_Tool.ChemFG_Tool import GetFunctionalGroups
#     return GetFunctionalGroups(mol)


# # --- Verification ---
# mol = Chem.MolFromSmiles("CC(=O)O")  # Carboxylic acid (has an OH group)
# # Add an independent alcohol to test both
# mol_with_alcohol = Chem.MolFromSmiles("OCC(=O)O") 
# mol_with_hs = Chem.AddHs(mol_with_alcohol)

# final_dict = identify_unique_groups_with_true_hs(mol_with_hs)
# print(final_dict)

    


def FindFuncGrps_from_rdkit(mol, custom_smarts_dict=None):
    """
    Finds native RDKit functional groups plus optional custom user rules.
    Automatically captures explicit hydrogens and removes subset redundancies.


    # --- Example Usage ---
    # Suppose you want to define a custom rule for a "Trifluoromethyl" group
    my_custom_rules = {
        "Trifluoromethyl": "C(F)(F)F",
        "Custom_Alkyl_Chloride": "[CX4][Cl]"
    }

    mol = Chem.MolFromSmiles("FC(F)(F)CC(O)=O") # Contains Trifluoromethyl & Carboxylic Acid
    mol_with_hs = Chem.AddHs(mol)

    # Run with your custom definitions injected
    final_dict = FindFuncGrps_from_rdkit(mol_with_hs, custom_smarts_dict=my_custom_rules)
    print(final_dict)

    """

    from rdkit import Chem
    from rdkit.Chem import rdfiltercatalog
    
    if custom_smarts_dict is None:
        custom_smarts_dict = {}
        
    all_patterns = []    

    hierarchy = rdfiltercatalog.GetFlattenedFunctionalGroupHierarchy(normalized=True)
    for node in hierarchy:
        if node not in ["amine.aliphatic"]:
            all_patterns.append((node, hierarchy[node]))

    # 2. Append your custom rules to the search list
    for name, smarts_str in custom_smarts_dict.items():
        custom_pattern = Chem.MolFromSmarts(smarts_str)
        if custom_pattern is not None:
            all_patterns.append((name, custom_pattern))
            
    raw_matches = []
    
    # 3. Match patterns and dynamically attach explicit Hydrogen neighbors
    for fg_name, pattern in all_patterns:
        matches = mol.GetSubstructMatches(pattern)
        #print(fg_name,matches)
        for match in matches:
            full_atom_indices = list(match)
            extra_hs = []
            
            for atom_idx in full_atom_indices:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor.GetSymbol() == 'H' and neighbor_idx not in full_atom_indices:
                        extra_hs.append(neighbor_idx)
                        
            full_atom_indices.extend(extra_hs)
            
            raw_matches.append({
                "name": fg_name,
                "atom_set": set(full_atom_indices),
                "atom_list": full_atom_indices
            })
            
    # 4. Filter out subset redundancies
    filtered_matches = []
    for i, match_a in enumerate(raw_matches):
        is_redundant = False
        for j, match_b in enumerate(raw_matches):
            if i == j:
                continue
            if match_a["atom_set"].issubset(match_b["atom_set"]):
                if match_a["atom_set"] == match_b["atom_set"] and len(match_b["name"]) > len(match_a["name"]):
                    is_redundant = True
                    break
                elif match_a["atom_set"] != match_b["atom_set"]:
                    is_redundant = True
                    break
        if not is_redundant:
            filtered_matches.append(match_a)
            
    # 5. Format to dictionary
    results = {}
    for match in filtered_matches:
        fg_name = match["name"]
        if fg_name not in results:
            results[fg_name] = []
        results[fg_name].append(match["atom_list"])
        
    return results





    

def __IdentifySugarAtoms_backend(s,ring):
    from ndfes.constants import GetAtomicNumber
    
    if ring is None:
        return None
    if len(ring[1]) != 5:
        return None

    graph = s.GetGraph()
    
    seen = ring[1]
    atoms = {}
    atoms["C1'"] = seen[0]
    atoms["C2'"] = seen[1]
    atoms["C3'"] = seen[2]
    atoms["C4'"] = seen[3]
    atoms["O4'"] = seen[4]

    
    
    seen = [ str(x) for x in seen ]
    
    # Find the C5' position
    atoms["C5'"] = None   
    for edge in graph.edges[str(atoms["C4'"])]:
        name = s.data["elements"][int(edge)]
        if name == "C5'":
            atoms["C5'"] = int(edge)
            seen.append(edge)
            break
    if atoms["C5'"] is None:
        for edge in graph.edges[str(atoms["C4'"])]:
            if edge in seen:
                continue
            else:
                ele = s.data["elements"][int(edge)]
                z = GetAtomicNumber(ele)
                if z > 1:
                    atoms["C5'"] = int(edge)
                    seen.append(edge)
                    break

    # Find the O5' position
    atoms["O5'"] = None
    if atoms["C5'"] is not None:
        for edge in graph.edges[str(atoms["C5'"])]:
            name = s.data["elements"][int(edge)]
            if name == "O5'":
                atoms["O5'"] = int(edge)
                seen.append(edge)
                break
        if atoms["O5'"] is None:
            for edge in graph.edges[str(atoms["C5'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    if z > 1:
                        atoms["O5'"] = int(edge)
                        seen.append(edge)
                        break
                   
    # Find the HO5' position
    atoms["HO5'"] = None
    if atoms["O5'"] is not None:
        for edge in graph.edges[str(atoms["O5'"])]:
            name = s.data["elements"][int(edge)]
            if name == "HO5'" or name == "P":
                atoms["HO5'"] = int(edge)
                seen.append(edge)
                break
        if atoms["HO5'"] is None:
            for edge in graph.edges[str(atoms["O5'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    if z == 1 or z == 15:
                        atoms["HO5'"] = int(edge)
                        seen.append(edge)
                        break
              
    # Find the O3' position
    atoms["O3'"] = None
    for edge in graph.edges[str(atoms["C3'"])]:
        name = s.data["elements"][int(edge)]
        if name == "O3'":
            atoms["O3'"] = int(edge)
            seen.append(edge)
            break
    if atoms["O3'"] is None:
        for edge in graph.edges[str(atoms["C3'"])]:
            if edge in seen:
                continue
            else:
                ele = s.data["elements"][int(edge)]
                z = GetAtomicNumber(ele)
                if z > 1:
                    atoms["O3'"] = int(edge)
                    seen.append(edge)
                    break

    # Find the HO3' position
    atoms["HO3'"] = None
    if atoms["O3'"] is not None:
        for edge in graph.edges[str(atoms["O3'"])]:
            name = s.data["elements"][int(edge)]
            if name == "HO3'":
                atoms["HO3'"] = int(edge)
                seen.append(edge)
                break
        if atoms["HO3'"] is None:
            for edge in graph.edges[str(atoms["O3'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    if z == 1:
                        atoms["HO3'"] = int(edge)
                        seen.append(edge)
                        break

    # Find the O3' position
    atoms["O2'"] = None
    for edge in graph.edges[str(atoms["C2'"])]:
        name = s.data["elements"][int(edge)]
        if name == "O2'":
            atoms["O2'"] = int(edge)
            seen.append(edge)
            break
    if atoms["O2'"] is None:
        for edge in graph.edges[str(atoms["C2'"])]:
            if edge in seen:
                continue
            else:
                ele = s.data["elements"][int(edge)]
                z = GetAtomicNumber(ele)
                if z > 1:
                    atoms["O2'"] = int(edge)
                    seen.append(edge)
                    break

    # Find the HO2' position
    atoms["HO2'"] = None
    if atoms["O2'"] is not None:
        for edge in graph.edges[str(atoms["O2'"])]:
            name = s.data["elements"][int(edge)]
            if name == "HO2'":
                atoms["HO2'"] = int(edge)
                seen.append(edge)
                break
        if atoms["HO2'"] is None:
            for edge in graph.edges[str(atoms["O2'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    if z == 1:
                        atoms["HO2'"] = int(edge)
                        seen.append(edge)
                        break

      
    # Find the N9/N1 position
    atoms["N9"] = None
    if atoms["C1'"] is not None:
        for edge in graph.edges[str(atoms["C1'"])]:
            name = s.data["elements"][int(edge)]
            if name == "N9" or name == "N1":
                atoms["N9"] = int(edge)
                seen.append(edge)
                break
        if atoms["N9"] is None:
            for edge in graph.edges[str(atoms["C1'"])]:
                if edge in seen:
                    continue
                else:
                    ele = s.data["elements"][int(edge)]
                    z = GetAtomicNumber(ele)
                    # Hmm, why was this outside z>1 check?
                    #if atoms["N9"] is None:
                    #    atoms["N9"] = int(edge)
                    #    seen.append(edge)
                    if z > 1:
                        atoms["N9"] = int(edge)
                        seen.append(edge)
                        break

    # Find the C4/C2 position
    atoms["C4"] = None
    if atoms["N9"] is not None:
        for edge in graph.edges[str(atoms["N9"])]:
            name = s.data["elements"][int(edge)]
            if name == "C4" or name == "C2":
                atoms["C4"] = int(edge)
                seen.append(edge)
                break
        if atoms["C4"] is None:
            for edge in graph.edges[str(atoms["N9"])]:
                if edge in seen:
                    continue
                else:
                    # Count the number of heavy atoms connected to "edge"
                    nheavy = 0
                    for nbor in graph.edges[edge]:
                        ele = s.data["elements"][int(nbor)]
                        z = GetAtomicNumber(ele)
                        if z > 1:
                            nheavy += 1
                    if nheavy > 2:
                        atoms["C4"] = int(edge)
                        seen.append(edge)
                        break
    isdna = False
    if atoms["O2'"] is None and atoms["O3'"] is not None:
        isdna = True

    return atoms,isdna




def IdentifySugarAtoms(s,verbose=False):
    from ndfes.constants import GetAtomicNumber
    from ffpopt.dihed.Dihedrals import FindPuckers
    import sys
    
    rings = FindPuckers(s)
    if verbose:
        sys.stdout.write(f"\nFound {len(rings)} 5-membered rings\n\n")

    results=None
        
    for iring,ring in enumerate(rings):
        a = ring[0]
        b = ring[1]

        bok = True
        if b is None:
            bok = False
        else:
            if len(b) == 0:
                bok = False

        if verbose:
            if not bok:
                names = [ s.data["names"][int(i)] for i in a ]
                idxs = ",".join( [x for x in a] )
                mask = "@" + ",".join(names)
                sys.stdout.write("Ring %2i: FAILED TO IDENTIFY ORDER\n"%(iring+1))
                sys.stdout.write(f" 0-idxs: {idxs}\n")
                sys.stdout.write(f"   mask: {mask}\n")
                sys.stderr.write("\n")
            else:
                names = [ s.data["names"][i] for i in b ]
                idxs = ",".join( [str(x) for x in b] )
                mask = "@" + ",".join(names)
                sys.stdout.write("Ring %2i: CONSISTENT WITH BEING A SUGAR\n"%(iring+1))
                sys.stdout.write(f" 0-idxs: {idxs}\n")
                sys.stdout.write(f"   mask: {mask}\n")
        if bok:
            #PrintConstraints(s,ring,args.rna,args.dna)
            
            if ring[1] is not None:
                atoms,isdna = __IdentifySugarAtoms_backend(s,ring)
                num_not_none = sum([ 1 for a in atoms if atoms[a] is not None ])
                if results is None:
                    results=(atoms,isdna)
                else:
                    prev_not_none = sum([ 1 for a in results[0] if results[0][a] is not None ])
                    if num_not_none > prev_not_none:
                        results=(atoms,isdna)


    if verbose:
        sys.stdout.write("\n Atom position map\n")
        sys.stdout.write("-------------------\n")
        sys.stdout.write("%8s %5s %4s\n"%("Position","0-idx","Name"))
        for pos in atoms:
            if atoms[pos] is None:
                sys.stdout.write("%8s %5s %4s\n"%(pos,"None","None"))
            else:
                sys.stdout.write("%8s %5s %4s\n"%(pos,atoms[pos],s.data["names"][atoms[pos]]))

    return results

