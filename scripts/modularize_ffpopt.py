#!/usr/bin/env python3
"""Mechanical modularization for ffpopt (behavior-preserving).

Moves mega-modules into packages with:
  package/_impl.py          - full legacy implementation (imports rewritten)
  package/<concern>.py      - named re-export modules (public surface)
  package/__init__.py       - aggregate exports
  ffpopt/<Legacy>.py        - compatibility facade

Also relocates runtime helpers under ffpopt/runtime/ with root facades.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "src" / "ffpopt"


def rewrite_relimports(text: str, *, self_mod: str | None = None, self_pkg: str | None = None) -> str:
    """Rewrite ``from . Foo`` to ``from ffpopt.Foo`` for code moved into a subpackage."""

    def repl(match: re.Match) -> str:
        ws_and_name = match.group(1)
        name = ws_and_name.lstrip()
        leading = ws_and_name[: len(ws_and_name) - len(name)]
        if self_mod and self_pkg:
            if name == self_mod or name.startswith(self_mod + " ") or name.startswith(self_mod + "."):
                # from . GeomOpt import X  -> from ffpopt.geomopt import X
                suffix = name[len(self_mod) :]
                return f"from {self_pkg}{suffix}"
        return f"from ffpopt.{name}"

    text = re.sub(r"from \.(\s*[A-Za-z_][\w.]*)", repl, text)
    if self_mod and self_pkg:
        text = re.sub(
            rf"from ffpopt\.{re.escape(self_mod)} import",
            f"from {self_pkg} import",
            text,
        )
        text = re.sub(
            rf"from ffpopt\.{re.escape(self_mod)}\.",
            f"from {self_pkg}.",
            text,
        )
    return text


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def make_package(
    legacy_name: str,
    pkg_name: str,
    export_modules: dict[str, list[str]],
    *,
    extra_init: str = "",
    postprocess_impl=None,
) -> None:
    """Move ``legacy_name.py`` into ``pkg_name/_impl.py`` + named re-export modules."""
    legacy = SRC / f"{legacy_name}.py"
    text = legacy.read_text(encoding="utf-8")
    if text.lstrip().startswith('"""Compatibility facade'):
        print(f"  skip {legacy_name} (already facade)")
        return

    pkg = SRC / pkg_name
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir()

    impl = rewrite_relimports(text, self_mod=legacy_name, self_pkg=f"ffpopt.{pkg_name}")
    if postprocess_impl:
        impl = postprocess_impl(impl)
    write(pkg / "_impl.py", impl)

    all_names: list[str] = []
    init_imports: list[str] = []
    for mod, names in export_modules.items():
        all_names.extend(names)
        joined = ",\n    ".join(names)
        write(
            pkg / f"{mod}.py",
            f'"""{legacy_name} - {mod}."""\n'
            f"from ._impl import (\n    {joined},\n)\n\n"
            f"__all__ = {names!r}\n",
        )
        init_imports.append(f"from .{mod} import *  # noqa: F403")

    write(
        pkg / "__init__.py",
        f'"""{legacy_name} package (modularized; behavior unchanged)."""\n'
        + "\n".join(init_imports)
        + "\n"
        + extra_init
        + f"\n__all__ = {all_names!r}\n",
    )
    write(
        legacy,
        f'"""Compatibility facade - implementation lives in ``ffpopt.{pkg_name}``."""\n'
        f"from ffpopt.{pkg_name} import *  # noqa: F403\n"
        f"from ffpopt.{pkg_name} import __all__ as __all__  # noqa: F401\n",
    )


def phase1_runtime() -> None:
    print("=== Phase 1: runtime ===")
    runtime = SRC / "runtime"
    runtime.mkdir(exist_ok=True)
    mapping = [
        "cpu_budget",
        "fast_wavefront",
        "console",
        "progress_board",
        "fragment_progress",
    ]
    for name in mapping:
        src = SRC / f"{name}.py"
        text = src.read_text(encoding="utf-8")
        if text.lstrip().startswith('"""Compatibility facade'):
            continue
        text = rewrite_relimports(text)
        # Prefer package-internal imports among runtime helpers
        for sib in mapping:
            if sib == name:
                continue
            text = text.replace(f"from ffpopt.{sib} import", f"from ffpopt.runtime.{sib} import")
            text = text.replace(f"from ffpopt.{sib} ", f"from ffpopt.runtime.{sib} ")
        write(runtime / f"{name}.py", text)
        write(
            src,
            f'"""Compatibility facade - implementation lives in ``ffpopt.runtime.{name}``."""\n'
            f"from ffpopt.runtime.{name} import *  # noqa: F403\n"
            f"try:\n"
            f"    from ffpopt.runtime.{name} import __all__ as __all__  # noqa: F401\n"
            f"except ImportError:\n"
            f"    pass\n",
        )
    write(
        runtime / "__init__.py",
        '"""Runtime helpers shared by wavefront / workflows / ligandparam boards."""\n'
        + "\n".join(f"from .{n} import *  # noqa: F403" for n in mapping)
        + "\n",
    )


def phase2_geomopt() -> None:
    print("=== Phase 2: geomopt ===")
    make_package(
        "GeomOpt",
        "geomopt",
        {
            "recovery": ["opt_recovery_label", "is_soft_opt_recovery"],
            "ase_opt": ["GeomOpt_ASE", "_ase_fmax", "_ase_loose_fmax"],
            "watchdog": [
                "_linux_process_tree_cputime",
                "_path_tree_mtime",
                "_geometric_stall_timeout_sec",
                "_run_geometric_with_watchdog",
            ],
            "geometric_opt": ["GeomOpt_GEOMETRIC"],
            "core": [
                "bare_potential_energy",
                "GeomOpt_SinglePoint",
                "GeomOpt",
                "CheckForces",
                "_geomopt_fallback_note",
            ],
            "scan": [
                "ApplyDihedConstraint",
                "DihedScan",
                "FwdRevDihedScan",
                "FwdRevDihedScan_worker",
            ],
            "parallel": [
                "ParallelGeomOpt",
                "ParallelGeomOpt_threads",
                "ParallelGeomOpt_mpi",
                "CalcNode",
                "is_mpi",
                "is_mpi_worker",
            ],
        },
    )


def phase3_wavefront() -> None:
    print("=== Phase 3: wavefront ===")
    make_package(
        "WaveFront",
        "wavefront",
        {
            "ipc": [
                "_WORKER",
                "_clear_los_calc",
                "_init_worker",
                "_clone_struct_geometry",
                "_struct_from_coords",
                "_run_node_job",
                "_run_node",
            ],
            "node": ["WavefrontNode"],
            "level": ["WavefrontLevel"],
            "engine": ["Wavefront"],
            "plot": ["plot_wavefront"],
            "utils": ["find_adjacent_dihedrals", "wavefront_loader"],
            "api": ["run_dihed_wavefront"],
        },
    )


def phase4_wavefront_nd() -> None:
    print("=== Phase 4: wavefront_nd ===")
    # Shared mixin helpers live in wavefront/mixins.py for ND to import
    mixins = SRC / "wavefront" / "mixins.py"
    write(
        mixins,
        '"""Shared wavefront node helpers (1-D and N-D)."""\n'
        "from __future__ import annotations\n\n"
        "import copy\n\n"
        "import numpy as np\n\n\n"
        "def clone_struct_geometry(struct, coords, ene=0.0, frcs=None):\n"
        '    """Prefer ``Struct.clone_geometry``; fall back to deepcopy for test doubles."""\n'
        '    clone = getattr(struct, "clone_geometry", None)\n'
        "    if callable(clone):\n"
        "        return clone(coords=coords, ene=ene, frcs=frcs)\n"
        "    out = copy.deepcopy(struct)\n"
        "    out.Update(ene, np.asarray(coords, dtype=float), frcs)\n"
        "    return out\n\n\n"
        "def clear_los_calc(los) -> None:\n"
        '    """Drop live calculators so workers rebuild (and cache) in-process."""\n'
        '    clearer = getattr(los, "clear_runtime_caches", None)\n'
        "    if callable(clearer):\n"
        "        clearer()\n"
        "        return\n"
        '    calc = getattr(los, "calc", None)\n'
        "    if calc is not None:\n"
        "        try:\n"
        "            calc.Reset()\n"
        "        except Exception:\n"
        "            pass\n"
        "        try:\n"
        "            los.calc = None\n"
        "        except Exception:\n"
        "            pass\n",
    )
    make_package(
        "WaveFrontND",
        "wavefront_nd",
        {
            "grid": ["GetGridNeighbors", "is_mpi_worker"],
            "ipc": [
                "_WORKER",
                "_clear_los_calc",
                "_init_worker",
                "_clone_struct_geometry",
                "_struct_from_coords",
                "_run_node_job",
                "_run_node",
            ],
            "node": ["WavefrontNode"],
            "level": ["WavefrontLevel"],
            "engine": ["Wavefront"],
            "utils": ["wavefront_loader"],
            "api": ["run_dihed_wavefront"],
        },
    )


def phase5_workflows() -> None:
    print("=== Phase 5: workflows ===")
    make_package(
        "Workflows",
        "workflows",
        {
            "paths": [
                "_as_path",
                "_in_workdir",
                "_subprocess_cwd",
                "_resolve_logger",
                "normalize_bond_pairs0",
                "bonds0_from_scission_fit_torsions",
                "_parent_paths_from_args",
            ],
            "bin_runners": [
                "_ffpopt_bin_script",
                "_run_current_python",
                "_run_fit_script_inprocess",
                "_run_ffpopt_bin",
            ],
            "fit_io": [
                "_TwistParam",
                "_resolve_scans_and_params",
                "_write_fit_json",
                "_require_files",
                "_run_gendihedfit",
                "_compare_per_bond",
                "_apply_fit_and_prepare",
            ],
            "bond_scan_pool": [
                "_run_one_scan",
                "_slim_scan_result",
                "_run_bond_scan_job",
                "_run_scans_for_bonds",
            ],
            "parallel": [
                "_split_fragment_nproc",
                "_NonDaemonSpawnProcess",
                "_NonDaemonSpawnContext",
                "_make_nondaemon_spawn_pool",
            ],
            "twist": ["run_dihed_twist_workflow"],
            "fragmented": [
                "_load_existing_fragments",
                "_build_structure_image_map",
                "_prepare_fragment_input",
                "_slim_twist_result",
                "_run_fragment_twist_job",
                "run_fragmented_dihed_twist_workflow",
            ],
        },
    )


def phase6_dihedrals() -> None:
    print("=== Phase 6: dihedrals ===")
    make_package(
        "Dihedrals",
        "dihedrals",
        {
            "edit": [
                "DeleteDihedrals",
                "ChangeDihedrals",
                "FindDihedrals",
                "GetMultiDihedFcnFromIdxs",
                "ChangeParmFromMultiDihedFcn",
                "GetDihedClasses",
            ],
            "primitives": ["PrimDihedFcn", "MultiDihedFcn", "CptDihedralEne"],
            "solvers": [
                "EnergyScansWithoutDihedrals",
                "IsolatedLinearSolve",
                "AngularStdDev",
                "DihedFitObjFcn",
                "_DihedFitObjFcn_reopt",
                "NonlinearSolve",
                "use_dihed_fit_reopt",
                "build_fixed_geometry_ll_cache",
                "ll_energies_kcal_from_cache",
                "_fitted_dihed_idxs",
                "_analytical_fitted_torsion_kcal",
            ],
            "types": [
                "ParamType",
                "ParamInstance",
                "ProfileType",
                "SystemType",
                "FitInputType",
            ],
            "align": [
                "_normalize_scan_angle",
                "struct_scan_angle",
                "_angle_map_from_los",
                "align_scan_profiles",
            ],
            "parmed_script": ["WriteParmedScript"],
            "pucker": ["FindPuckers", "PuckerGuessByName", "PuckerGuessByElement"],
        },
    )


def main() -> None:
    phase1_runtime()
    phase2_geomopt()
    phase3_wavefront()
    phase4_wavefront_nd()
    phase5_workflows()
    phase6_dihedrals()
    print("ffpopt modularization complete")


if __name__ == "__main__":
    main()
