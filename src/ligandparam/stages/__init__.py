"""Stage package — lazy exports to avoid eager optional-dep imports."""
from __future__ import annotations

from typing import Any

_EXPORTS = {
    "AbstractStage": ".abstractstage",
    "StageLazyResp": ".resp",
    "StageMultiRespFit": ".resp",
    "StageParmChk": ".parmchk",
    "StageLeap": ".leap",
    "StageInitialize": ".initialize",
    "GaussianMinimizeRESP": ".gaussian",
    "StageGaussianRotation": ".gaussian",
    "StageGaussiantoMol2": ".gaussian",
    "StageGaussianToMol2": ".gaussian",
    "GaussianRESP": ".gaussian",
    "StageUpdateCharge": ".charge",
    "StageNormalizeCharge": ".charge",
    "StageUpdate": ".typematching",
    "StageMatchAtomNames": ".typematching",
    "SDFToPDB": ".sdfconverters",
    "SDFToPDBBatch": ".sdfconverters",
    "StageSmilesToPDB": ".smiles_to_pdb",
    "LigHFix": ".lighfix",
    "StageDisplaceMol": ".displacemol",
    "PDB_Name_Fixer": ".pdb_names",
    "StagePdbNameFixer": ".pdb_names",
    "DPMinimize": ".deepmd",
    "StageDihedTwistCorrection": ".ffpopt_dihed",
    "StageSageCreate": ".generate_sage_params",
    "StageSageToAmber": ".generate_sage_params",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name == "StageSmilestoPDB":
        from .smiles_to_pdb import StageSmilesToPDB as StageSmilestoPDB

        return StageSmilestoPDB
    mod = _EXPORTS.get(name)
    if mod is None:
        # Fall back to utilsstages for misc helpers historically star-imported.
        from . import utilsstages as _utils
        if hasattr(_utils, name):
            return getattr(_utils, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    m = importlib.import_module(mod, __name__)
    return getattr(m, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
