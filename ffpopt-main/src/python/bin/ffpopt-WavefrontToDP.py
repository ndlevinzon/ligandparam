#!/usr/bin/env python3

import numpy as np

try:
    import dpdata
except ImportError:
    _HAS_DPDATA = False
else:
    _HAS_DPDATA = True

from pathlib import Path

from ffpopt.WaveFront import wavefront_loader

def convert_wavefront_to_data(input_file, default_cell=[[30,0,0],[0,30,0],[0,0,30]]):
    wf_data = wavefront_loader(input_file)

    coord, energy, force = [], [], []
    nopbc = True

    nodes = wf_data.min_nodes
    sorted_nodes = [nodes[key] for key in sorted(nodes.keys())]

    atom_names, atom_numbs, atom_types = [], [], []
    for atom in sorted_nodes[0].opt_geom.get_chemical_symbols():
        if atom in atom_names:
            atom_types.append(atom_names.index(atom))
            atom_numbs[atom_names.index(atom)] += 1
        else:
            atom_numbs.append(1)
            atom_names.append(atom)
            atom_types.append(len(atom_names) - 1)

    for node in sorted_nodes:
        coord.append(node.opt_geom.positions)
        energy.append(node.energy)
        force.append(node.forces)

    cell = default_cell
    cells = [cell] * len(energy)

    return {
        "coords": np.array(coord),
        "energies": np.array(energy),
        "force": np.array(force),
        "cells": np.array(cells),
        "atom_names": atom_names,
        "atom_numbs": atom_numbs,
        "atom_types": np.array(atom_types),
        "orig": np.array([0,0,0]),
        "nopbc": nopbc
    }

def data_to_labeled_system(data, output_file):
    # Convert the data to a labeled system format
    try:
        labeled_data = {
            "coords": data["coords"],
            "energies": data["energies"],
            "force": data["force"],
            "cells": data["cells"],
            "atom_names": data["atom_names"],
            "atom_numbs": data["atom_numbs"],
            "atom_types": data["atom_types"],
            "orig": data["orig"],
            "nopbc": data["nopbc"]
        }
    except KeyError as e:
        print(f"Missing key in data: {e}")
        return

    if not _HAS_DPDATA:
        raise RuntimeError("DPData is not available. No HDF5 written.")

    dpdata_system = dpdata.LabeledSystem(data=labeled_data)
    print(dpdata_system)
    output = Path(output_file)
    if output.suffix == ".hdf5":
        dpdata_system.to_deepmd_hdf5(str(output))
    else:
        raise ValueError("Unsupported output file format.")
    
def merge_data(all_data):
    atom_names_list = [data["atom_names"] for data in all_data]
    first_atom_names = atom_names_list[0]
    for idx, atom_names in enumerate(atom_names_list[1:], 1):
        if atom_names != first_atom_names:
            raise ValueError(f"atom_names mismatch between dataset 0 and dataset {idx}: {first_atom_names} vs {atom_names}")
        
    merged = {
        "coords": np.concatenate([data["coords"] for data in all_data]),
        "energies": np.concatenate([data["energies"] for data in all_data]),
        "force": np.concatenate([data["force"] for data in all_data]),
        "cells": np.concatenate([data["cells"] for data in all_data]),
        "atom_names": all_data[0]["atom_names"],
        "atom_numbs": all_data[0]["atom_numbs"],
        "atom_types": all_data[0]["atom_types"],
        "orig": all_data[0]["orig"],
        "nopbc": all_data[0]["nopbc"]
    }
    return merged

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert WaveFront data to DPData format")
    parser.add_argument("-i", "--input", nargs='+', help="Path to the input WaveFront file")
    parser.add_argument("-o", "--output", help="Path to the output DPData file")
    args = parser.parse_args()

    all_data = []
    for file in args.input:
        data = convert_wavefront_to_data(file)
        all_data.append(data)
    merged = merge_data(all_data)
    data_to_labeled_system(merged, args.output)