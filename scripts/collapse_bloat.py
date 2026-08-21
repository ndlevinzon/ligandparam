#!/usr/bin/env python3
"""Collapse thin re-export shims; keep one meaningful implementation per package."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path.cwd()
FF = ROOT / "src" / "ffpopt"
LP = ROOT / "src" / "ligandparam"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def collapse_package(
    pkg_dir: Path,
    *,
    impl_name: str,
    keep: set[str] | None = None,
    facade_path: Path | None = None,
    facade_import: str | None = None,
) -> None:
    """Keep ``impl_name.py`` (+ optional keep set), thin ``__init__.py``, delete stubs."""
    keep = set(keep or ())
    keep |= {"__init__.py", impl_name}
    impl_src = pkg_dir / "_impl.py"
    impl_dst = pkg_dir / impl_name
    if not impl_src.is_file() and not impl_dst.is_file():
        print(f"  skip missing impl in {pkg_dir}")
        return
    if impl_src.is_file() and impl_src.name != impl_name:
        if impl_dst.exists():
            impl_dst.unlink()
        impl_src.rename(impl_dst)
        print(f"  renamed {impl_src.name} -> {impl_name}")

    for f in list(pkg_dir.glob("*.py")):
        if f.name not in keep and f.name != impl_name:
            f.unlink()
            print(f"  deleted {f.relative_to(ROOT)}")

    # Also drop unused _header etc.
    for f in list(pkg_dir.glob("*")):
        if f.is_file() and f.suffix != ".py" and f.name not in keep:
            pass

    mod = impl_name[:-3] if impl_name.endswith(".py") else impl_name
    write(
        pkg_dir / "__init__.py",
        f'"""{pkg_dir.name} — see ``{mod}`` for implementation."""\n'
        "from __future__ import annotations\n\n"
        f"from .{mod} import *  # noqa: F403\n"
        f"from .{mod} import __all__ as __all__  # noqa: F401\n"
        "\n"
        "# Re-export private helpers so tests/patches on this package keep working.\n"
        f"from . import {mod} as _{mod}\n"
        f"for _n in dir(_{mod}):\n"
        '    if _n.startswith("__") and _n.endswith("__"):\n'
        "        continue\n"
        f"    if _n not in globals():\n"
        f"        globals()[_n] = getattr(_{mod}, _n)\n"
        "del _n, _{mod}\n".replace(f"_{mod}", f"_{mod}"),
    )
    # Fix botched replace - write cleanly
    write(
        pkg_dir / "__init__.py",
        f'"""{pkg_dir.name} — implementation in ``{mod}.py``."""\n'
        "from __future__ import annotations\n\n"
        f"from . import {mod} as _impl\n"
        "from ._impl import *  # type: ignore  # noqa: F403\n".replace(
            "from ._impl import", f"from .{mod} import"
        )
        + "\n"
        "for _name in dir(_impl):\n"
        '    if _name.startswith("__") and _name.endswith("__"):\n'
        "        continue\n"
        "    globals()[_name] = getattr(_impl, _name)\n"
        "del _name, _impl\n",
    )

    if facade_path is not None and facade_import is not None:
        write(
            facade_path,
            f'"""Compatibility facade — implementation lives in ``{facade_import}``."""\n'
            "from __future__ import annotations\n\n"
            "import importlib\n"
            "import sys\n\n"
            f"_pkg = importlib.import_module('{facade_import}')\n"
            "sys.modules[__name__] = _pkg\n",
        )


def fix_init(pkg_dir: Path, mod: str) -> None:
    write(
        pkg_dir / "__init__.py",
        f'"""{pkg_dir.name} — implementation in ``{mod}.py``."""\n'
        "from __future__ import annotations\n\n"
        f"from . import {mod} as _impl\n\n"
        "for _name in dir(_impl):\n"
        '    if _name.startswith("__") and _name.endswith("__"):\n'
        "        continue\n"
        "    globals()[_name] = getattr(_impl, _name)\n"
        "del _name, _impl\n",
    )


def facade(path: Path, import_path: str) -> None:
    write(
        path,
        f'"""Compatibility facade — implementation lives in ``{import_path}``."""\n'
        "from __future__ import annotations\n\n"
        "import importlib\n"
        "import sys\n\n"
        f"_pkg = importlib.import_module('{import_path}')\n"
        "sys.modules[__name__] = _pkg\n",
    )


def collapse_to_single_module(pkg_dir: Path, dest_module: Path) -> None:
    """Flatten a package back to one module file (delete package dir)."""
    impl = pkg_dir / "_impl.py"
    if not impl.is_file():
        # already renamed?
        cands = [p for p in pkg_dir.glob("*.py") if p.name not in ("__init__.py",)]
        # prefer largest
        cands.sort(key=lambda p: p.stat().st_size, reverse=True)
        if not cands:
            print(f"  nothing to flatten in {pkg_dir}")
            return
        impl = cands[0]
    text = impl.read_text(encoding="utf-8")
    shutil.rmtree(pkg_dir)
    write(dest_module, text)
    print(f"  flattened {pkg_dir.name} -> {dest_module.relative_to(ROOT)}")


def main() -> None:
    print("=== ffpopt packages: drop shim files ===")
    # geomopt
    pkg = FF / "geomopt"
    if (pkg / "_impl.py").exists() or (pkg / "optimize.py").exists():
        if (pkg / "_impl.py").exists():
            (pkg / "_impl.py").replace(pkg / "optimize.py")
        for f in list(pkg.glob("*.py")):
            if f.name not in ("__init__.py", "optimize.py"):
                f.unlink()
                print("  deleted", f.relative_to(ROOT))
        fix_init(pkg, "optimize")
        facade(FF / "GeomOpt.py", "ffpopt.geomopt.optimize")

    # wavefront — keep mixins
    pkg = FF / "wavefront"
    if (pkg / "_impl.py").exists():
        (pkg / "_impl.py").replace(pkg / "scan.py")
    for f in list(pkg.glob("*.py")):
        if f.name not in ("__init__.py", "scan.py", "mixins.py"):
            f.unlink()
            print("  deleted", f.relative_to(ROOT))
    fix_init(pkg, "scan")
    facade(FF / "WaveFront.py", "ffpopt.wavefront.scan")

    # wavefront_nd
    pkg = FF / "wavefront_nd"
    if (pkg / "_impl.py").exists():
        (pkg / "_impl.py").replace(pkg / "scan.py")
    for f in list(pkg.glob("*.py")):
        if f.name not in ("__init__.py", "scan.py"):
            f.unlink()
            print("  deleted", f.relative_to(ROOT))
    fix_init(pkg, "scan")
    facade(FF / "WaveFrontND.py", "ffpopt.wavefront_nd.scan")

    # workflows
    pkg = FF / "workflows"
    if (pkg / "_impl.py").exists():
        (pkg / "_impl.py").replace(pkg / "twist.py")
    for f in list(pkg.glob("*.py")):
        if f.name not in ("__init__.py", "twist.py"):
            f.unlink()
            print("  deleted", f.relative_to(ROOT))
    fix_init(pkg, "twist")
    facade(FF / "Workflows.py", "ffpopt.workflows.DihedTwist")

    # dihedrals
    pkg = FF / "dihedrals"
    if (pkg / "_impl.py").exists():
        (pkg / "_impl.py").replace(pkg / "fit.py")
    for f in list(pkg.glob("*.py")):
        if f.name not in ("__init__.py", "fit.py"):
            f.unlink()
            print("  deleted", f.relative_to(ROOT))
    fix_init(pkg, "fit")
    facade(FF / "Dihedrals.py", "ffpopt.dihedrals.fit")

    # runtime — already meaningful; trim empty __init__ noise only
    write(
        FF / "runtime" / "__init__.py",
        '"""Runtime helpers: CPU leases, progress boards, fast presets, console tee."""\n'
        "from .cpu_budget import *  # noqa: F403\n"
        "from .fast_wavefront import *  # noqa: F403\n"
        "from .console import *  # noqa: F403\n"
        "from .progress_board import *  # noqa: F403\n"
        "from .fragment_progress import *  # noqa: F403\n",
    )

    print("=== ligandparam: flatten gaussian + parm packages ===")
    gpkg = LP / "stages" / "gaussian"
    if gpkg.is_dir():
        collapse_to_single_module(gpkg, LP / "stages" / "gaussian.py")

    ppkg = LP / "multiresp" / "parm"
    if ppkg.is_dir():
        collapse_to_single_module(ppkg, LP / "multiresp" / "parmhelper.py")

    # Remove deprecated tree entirely
    dep = LP / "deprecated"
    if dep.is_dir():
        shutil.rmtree(dep)
        print("  removed ligandparam/deprecated/")

    # Remove empty/useless teststage if tiny stub
    ts = LP / "stages" / "teststage.py"
    if ts.is_file() and ts.stat().st_size < 500:
        # only delete if unused
        print("  keeping teststage.py (review manually)")

    print("done")


if __name__ == "__main__":
    main()
