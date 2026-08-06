#!/usr/bin/env python3
"""Example: fragment a parent ligand with scission, run the dihed-twist
workflow on each fragment, and merge the fitted torsion terms back into
a unified parent frcmod.

Requires the ``scission`` package (FragmentMol) importable in the env, plus
AmberTools on PATH (scission uses tleap to write fragment parm7/rst7).

Run from this directory:

    python3 run.py
"""

from ffpopt.Workflows import run_fragmented_dihed_twist_workflow


def main():
    result = run_fragmented_dihed_twist_workflow(
        mol2="ejm_45_0.mol2",
        lib="ejm_45_0.lib",
        frcmod="ejm_45_0.frcmod",
        out_dir="fragments",
        merged_frcmod="ejm_45_0.merged.frcmod",
        model="qdpi2",
        geometric_opt=True,
        nproc=10,
        maxiter=2,
    )
    print(f"merged frcmod: {result['merged_frcmod']}")
    print(f"fragments fit: {[f['fragment_id'] for f in result['fragments']]}")


if __name__ == "__main__":
    main()
