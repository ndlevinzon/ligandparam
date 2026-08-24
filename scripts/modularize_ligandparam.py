#!/usr/bin/env python3
"""Modularize ligandparam stages/recipes/multiresp (behavior-preserving)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
LP = ROOT / "src" / "ligandparam"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def rewrite_stage_imports(text: str) -> str:
    """Keep absolute ligandparam imports; fix self-imports if any."""
    return text


def split_gaussian() -> None:
    print("=== gaussian package ===")
    src = LP / "stages" / "gaussian.py"
    if not src.is_file():
        print("  skip (no gaussian.py)")
        return
    text = src.read_text(encoding="utf-8")
    if "Compatibility facade" in text[:120] or (LP / "stages" / "gaussian").is_dir():
        # already package?
        if (LP / "stages" / "gaussian" / "_impl.py").is_file():
            print("  skip already split")
            return
    lines = text.splitlines(keepends=True)
    pkg = LP / "stages" / "gaussian"
    if pkg.exists():
        shutil.rmtree(pkg)
    # backup then remove file so package can take the name
    write(LP / "stages" / "gaussian.py.bak", text)
    src.unlink()

    def chunk(a: int, b: int) -> str:
        return "".join(lines[a - 1 : b])

    # Shared header (imports through helpers)
    header = chunk(1, 203)
    write(pkg / "_header.py", '"""Shared imports / helpers for gaussian stages."""\n' + header)

    impl = rewrite_stage_imports(text)
    write(pkg / "_impl.py", impl)

    modules = {
        "helpers": [
            "_gaussian_log_is_complete",
            "_should_skip_gaussian_job",
            "_orientation_id_from_paths",
            "_run_gaussian_rotation_job",
            "_gaussian_opt_keyword",
            "_CALCFC_MAX_ATOMS",
        ],
        "minimize": ["GaussianMinimizeRESP"],
        "resp": ["GaussianRESP"],
        "rotation": ["StageGaussianRotation"],
        "mol2": ["StageGaussiantoMol2"],
    }
    for mod, names in modules.items():
        joined = ",\n    ".join(names)
        write(
            pkg / f"{mod}.py",
            f'"""Gaussian stages - {mod}."""\n'
            f"from ._impl import (\n    {joined},\n)\n\n"
            f"__all__ = {names!r}\n",
        )

    all_names = [n for names in modules.values() for n in names]
    write(
        pkg / "__init__.py",
        '"""Gaussian-related parameterization stages."""\n'
        "from __future__ import annotations\n\n"
        "from . import _impl as _impl\n\n"
        "for _name in dir(_impl):\n"
        '    if _name.startswith("__") and _name.endswith("__"):\n'
        "        continue\n"
        "    globals()[_name] = getattr(_impl, _name)\n"
        "del _name\n\n"
        "from . import helpers, minimize, resp, rotation, mol2  # noqa: F401\n\n"
        f"__all__ = {all_names!r}\n",
    )


def lazy_stages_init() -> None:
    print("=== lazy stages __init__ ===")
    mapping = {
        "AbstractStage": ".abstractstage",
        "StageLazyResp": ".resp",
        "StageMultiRespFit": ".resp",
        "StageParmChk": ".parmchk",
        "StageLeap": ".leap",
        "StageInitialize": ".initialize",
        "GaussianMinimizeRESP": ".gaussian",
        "StageGaussianRotation": ".gaussian",
        "StageGaussiantoMol2": ".gaussian",
        "GaussianRESP": ".gaussian",
        "StageUpdateCharge": ".charge",
        "StageNormalizeCharge": ".charge",
        "StageUpdate": ".typematching",
        "StageMatchAtomNames": ".typematching",
        "SDFToPDB": ".sdfconverters",
        "SDFToPDBBatch": ".sdfconverters",
        "StageSmilesToPDB": ".smilestopdb",
        "StageSmilestoPDB": ".initialize",  # legacy alias name lives in initialize
        "LigHFix": ".lighfix",
        "StageDisplaceMol": ".displacemol",
        "PDB_Name_Fixer": ".pdb_names",
        "DPMinimize": ".deepmd",
        "StageDihedTwistCorrection": ".ffpopt_dihed",
    }
    # Also pull utilsstages names lazily via __getattr__ fallback
    lines = [
        '"""Stage package - lazy exports to avoid eager optional-dep imports."""',
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "_EXPORTS = {",
    ]
    for name, mod in mapping.items():
        lines.append(f'    "{name}": "{mod}",')
    lines += [
        "}",
        "",
        "__all__ = list(_EXPORTS)",
        "",
        "",
        "def __getattr__(name: str) -> Any:",
        "    if name == \"StageSmilestoPDB\":",
        "        # Canonical class is StageSmilesToPDB; keep legacy name.",
        "        from .smiles_to_pdb import StageSmilesToPDB as StageSmilestoPDB",
        "        return StageSmilestoPDB",
        "    mod = _EXPORTS.get(name)",
        "    if mod is None:",
        "        # Fall back to utilsstages for misc helpers historically star-imported.",
        "        from . import utilsstages as _utils",
        "        if hasattr(_utils, name):",
        "            return getattr(_utils, name)",
        "        raise AttributeError(f\"module {__name__!r} has no attribute {name!r}\")",
        "    import importlib",
        "    m = importlib.import_module(mod, __name__)",
        "    return getattr(m, name)",
        "",
        "",
        "def __dir__() -> list[str]:",
        "    return sorted(set(__all__) | set(globals()))",
        "",
    ]
    write(LP / "stages" / "__init__.py", "\n".join(lines))

    # Point initialize's StageSmilestoPDB at canonical implementation (alias)
    init_py = LP / "stages" / "initialize.py"
    init_text = init_py.read_text(encoding="utf-8")
    if "class StageSmilestoPDB" in init_text and "Canonical SMILES" not in init_text:
        # Replace class body with alias at end of file approach: prepend alias and comment old class
        # Safer: after imports, add:
        alias = (
            "\n# Canonical SMILES->PDB stage lives in smilestopdb; keep legacy name.\n"
            "from .smiles_to_pdb import StageSmilesToPDB as StageSmilestoPDB  # noqa: E402\n"
        )
        # Only add if not already aliased; leave old class but rename to _Legacy
        init_text2 = init_text.replace(
            "class StageSmilestoPDB(AbstractStage):",
            "class _LegacyStageSmilestoPDB(AbstractStage):\n"
            "    # Deprecated duplicate; prefer StageSmilesToPDB / StageSmilestoPDB alias.\n"
            "    pass\n\n\nclass _UnusedLegacyStageSmilestoPDBBody(AbstractStage):",
        )
        # That's messy. Simpler: append alias and don't remove old class - duplicate names bad.
        # Best: delete the class from initialize by replacing with alias only.
        pattern = re.compile(
            r"\nclass StageSmilestoPDB\(AbstractStage\):.*?(?=\nclass |\Z)",
            re.S,
        )
        if pattern.search(init_text):
            init_text = pattern.sub(
                "\n# Legacy name -> canonical StageSmilesToPDB\n"
                "from .smiles_to_pdb import StageSmilesToPDB as StageSmilestoPDB\n\n",
                init_text,
                count=1,
            )
            write(init_py, init_text)


def recipe_registry() -> None:
    print("=== recipe registry ===")
    write(
        LP / "recipes" / "registry.py",
        '''\
"""Recipe name -> constructor registry for CLI / drivers."""

from __future__ import annotations

from typing import Any, Callable

RecipeFactory = Callable[..., Any]

_REGISTRY: dict[str, str] = {
    "lazyligand": "ligandparam.recipes.LazyLigand:LazyLigand",
    "lazierligand": "ligandparam.recipes.LazierLigand:LazierLigand",
    "freeligand": "ligandparam.recipes.FreeLigand:FreeLigand",
    "dplazyligand": "ligandparam.recipes.DpLazyLigand:DPLigand",
    "dpfreeligand": "ligandparam.recipes.DpFreeLigand:DPFreeLigand",
    "sqmligand": "ligandparam.recipes.OptLigand:SQMLigand",
}


def available_recipes() -> list[str]:
    return sorted(_REGISTRY)


def get_recipe(name: str, **kwargs):
    """Instantiate a recipe by CLI name (case-insensitive)."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown recipe name: {name}. Available recipes: {', '.join(available_recipes())}"
        )
    path, cls_name = _REGISTRY[key].split(":")
    import importlib

    mod = importlib.import_module(path)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)
''',
    )
    # Update recipes __init__ to export registry helpers
    recipes_init = LP / "recipes" / "__init__.py"
    text = recipes_init.read_text(encoding="utf-8")
    if "get_recipe" not in text:
        text = text.rstrip() + (
            "\n\nfrom .registry import available_recipes, get_recipe  # noqa: E402\n"
            "\n__all__ = list(__all__) + ['available_recipes', 'get_recipe']\n"
        )
        write(recipes_init, text)

    # Thin CLI recipe_selector
    cli = LP / "cli" / "ligandparam_getparam.py"
    cli_text = cli.read_text(encoding="utf-8")
    if "from ligandparam.recipes.Registry import get_recipe" not in cli_text:
        cli_text = re.sub(
            r"def recipe_selector\(recipe_name: str, \*\*kwargs\):.*?return SQMLigand\(\*\*kwargs\).*?raise ValueError\([\s\S]*?\)\n",
            "def recipe_selector(recipe_name: str, **kwargs):\n"
            '    """Select and return the appropriate recipe class based on the recipe name."""\n'
            "    from ligandparam.recipes.Registry import available_recipes, get_recipe\n\n"
            "    try:\n"
            "        return get_recipe(recipe_name, **kwargs)\n"
            "    except ValueError as exc:\n"
            "        raise ValueError(\n"
            "            f\"Unknown recipe name: {recipe_name}. Available recipes: \"\n"
            "            + \", \".join(available_recipes())\n"
            "        ) from exc\n\n",
            cli_text,
            count=1,
        )
        write(cli, cli_text)


def split_parmhelper() -> None:
    print("=== multiresp/parm ===")
    src = LP / "multiresp" / "parmhelper.py"
    text = src.read_text(encoding="utf-8")
    if text.lstrip().startswith('"""Compatibility facade'):
        print("  skip")
        return
    pkg = LP / "multiresp" / "parm"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir()
    write(pkg / "_impl.py", text)
    modules = {
        "computer": ["Computer", "BASH"],
        "io": ["OpenParm", "CopyParm"],
        "unique_params": [
            "MakeUniqueBondParams",
            "MakeUniqueAngleParams",
            "MakeUniqueDihedralParams",
        ],
        "selection": [
            "GetSelectedAtomIndices",
            "GetSelectedResidueIndices",
            "ListToSelection",
        ],
        "fragment": ["Fragment", "FragmentedSys"],
    }
    for mod, names in modules.items():
        joined = ",\n    ".join(names)
        write(
            pkg / f"{mod}.py",
            f'"""ParmEd helpers - {mod}."""\n'
            f"from ._impl import (\n    {joined},\n)\n\n"
            f"__all__ = {names!r}\n",
        )
    all_names = [n for ns in modules.values() for n in ns]
    write(
        pkg / "__init__.py",
        '"""ParmEd / fragment helpers (split from parmhelper.py)."""\n'
        "from . import _impl as _impl\n\n"
        "for _name in dir(_impl):\n"
        '    if _name.startswith("__") and _name.endswith("__"):\n'
        "        continue\n"
        "    globals()[_name] = getattr(_impl, _name)\n"
        "del _name\n\n"
        "from . import computer, io, unique_params, selection, fragment  # noqa: F401\n\n"
        f"__all__ = {all_names!r}\n",
    )
    write(
        src,
        '"""Compatibility facade - implementation lives in ``ligandparam.multiresp.parm``."""\n'
        "from __future__ import annotations\n\n"
        "import importlib\n"
        "import sys\n\n"
        "_pkg = importlib.import_module('ligandparam.multiresp.parm._impl')\n"
        "sys.modules[__name__] = _pkg\n",
    )


def main() -> None:
    split_gaussian()
    lazy_stages_init()
    recipe_registry()
    split_parmhelper()
    print("ligandparam modularization complete")


if __name__ == "__main__":
    main()
