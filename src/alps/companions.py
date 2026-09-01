"""Select bundled or external ``ffpopt`` / ``scission`` before other imports.

ligandparam keeps in-tree copies under ``src/ffpopt`` and ``src/scission``.
Independent checkouts can replace either tree without uninstalling the
bundle. Import names stay ``ffpopt`` and ``scission``.

Environment (read once per process, before any companion import)::

    LIGANDPARAM_FFPOPT=internal|external     # default internal
    LIGANDPARAM_SCISSION=internal|external   # default internal
    LIGANDPARAM_FFPOPT_PATH=<dir>            # required for external ffpopt
    LIGANDPARAM_SCISSION_PATH=<dir>          # required for external scission

A PATH value is either the directory that *contains* the package
(``.../src`` with ``src/ffpopt/``) or the package directory itself
(``.../src/ffpopt``). Two pip distributions cannot both own the top-level
name ``ffpopt``; PATH is how an independent tree coexists with the bundle.

This module is imported from :mod:`alps` ``__init__``, which
installs an import hook so the first ``import ffpopt`` / ``import scission``
binds the chosen trees. Set the env vars before starting the process.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

BUNDLE_MARKER = "__ligandparam_bundle__"
_COMPANION_NAMES = ("scission", "ffpopt")
_INTERNAL_MODES = frozenset({"", "internal", "bundled", "in-tree", "in_tree", "local"})
_EXTERNAL_MODES = frozenset({"external", "ext"})

_BOOTSTRAPPED = False
_HOOK_INSTALLED = False
_IN_BOOTSTRAP = False
_STATE: dict[str, "CompanionInfo"] | None = None


@dataclass(frozen=True)
class CompanionInfo:
    """Where one companion package was loaded from."""

    name: str
    mode: str
    origin: Path
    path_entry: Path
    bundled: bool

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "mode": self.mode,
            "origin": str(self.origin),
            "path_entry": str(self.path_entry),
            "bundled": self.bundled,
        }


def bundled_src_root() -> Path:
    """Directory that contains the in-tree ``ffpopt`` and ``scission`` packages.

    Editable install: ``<repo>/src``. Wheel: ``site-packages``.
    """
    return Path(__file__).resolve().parent.parent


def companion_status() -> dict[str, CompanionInfo]:
    """Return the resolved companion map (bootstraps if needed)."""
    return dict(bootstrap())


def format_status_line() -> str:
    """One ASCII line naming each companion tree."""
    parts = ["companions:"]
    for name in ("ffpopt", "scission"):
        info = companion_status()[name]
        parts.append(f"{name}={info.mode} ({info.origin})")
    return " ".join(parts)


def print_status_line(*, file=None) -> None:
    """Write :func:`format_status_line` (CLIs call this after the banner)."""
    print(format_status_line(), file=file if file is not None else sys.stdout, flush=True)


def reset_for_tests() -> None:
    """Allow a later :func:`bootstrap` to run again (unit tests only)."""
    global _BOOTSTRAPPED, _STATE
    _BOOTSTRAPPED = False
    _STATE = None


def install_import_hook() -> None:
    """Intercept the first ``ffpopt`` / ``scission`` import and bind the chosen tree.

    Installed from :mod:`ligandparam` ``__init__`` so ``import ligandparam``
    stays cheap until a caller actually needs a companion.
    """
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _CompanionFinder())
    _HOOK_INSTALLED = True


class _CompanionFinder:
    """sys.meta_path entry that bootstraps, then defers to the normal finder."""

    def find_spec(self, fullname, path, target=None):  # noqa: ARG002
        top = fullname.split(".", 1)[0]
        if top not in ("ffpopt", "scission"):
            return None
        if _IN_BOOTSTRAP or _BOOTSTRAPPED:
            return None
        bootstrap()
        return None


def bootstrap() -> Mapping[str, CompanionInfo]:
    """Bind ``ffpopt`` and ``scission`` according to the env vars.

    Loads scission first (ffpopt workflows import it). Idempotent.
    """
    global _BOOTSTRAPPED, _STATE, _IN_BOOTSTRAP
    if _BOOTSTRAPPED and _STATE is not None:
        return _STATE

    _IN_BOOTSTRAP = True
    try:
        infos: dict[str, CompanionInfo] = {}
        for name in _COMPANION_NAMES:
            infos[name] = _load(name, _read_mode(name), _path_hint(name))
        _apply_pythonpath(infos)
        _STATE = infos
        _BOOTSTRAPPED = True
        return _STATE
    finally:
        _IN_BOOTSTRAP = False


def _env_key(name: str, suffix: str) -> str:
    return f"LIGANDPARAM_{name.upper()}{suffix}"


def _read_mode(name: str) -> str:
    raw = os.environ.get(_env_key(name, ""), "internal")
    val = (raw or "internal").strip().lower()
    if val in _INTERNAL_MODES:
        return "internal"
    if val in _EXTERNAL_MODES:
        return "external"
    raise RuntimeError(
        f"{_env_key(name, '')}={raw!r} is not a known mode. "
        "Use internal (bundled src/ tree) or external "
        f"(independent checkout via {_env_key(name, '_PATH')})."
    )


def _path_hint(name: str) -> str | None:
    raw = os.environ.get(_env_key(name, "_PATH"))
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def _load(name: str, mode: str, path_hint: str | None) -> CompanionInfo:
    bundled_root = bundled_src_root()
    if mode == "internal":
        entry = bundled_root
        pkg = entry / name
        if not (pkg / "__init__.py").is_file():
            raise RuntimeError(
                f"internal {name} not found at {pkg}. "
                "Reinstall ligandparam so ffpopt and scission sit next to it."
            )
        _bind(name, entry)
        origin = Path(sys.modules[name].__file__).resolve().parent
        return CompanionInfo(
            name=name,
            mode="internal",
            origin=origin,
            path_entry=entry.resolve(),
            bundled=True,
        )

    if path_hint:
        entry = _resolve_path_entry(name, Path(path_hint))
    else:
        found = _find_non_bundled(name, bundled_root)
        if found is None:
            raise RuntimeError(
                f"{_env_key(name, '')}=external requires "
                f"{_env_key(name, '_PATH')} set to the directory that contains "
                f"the `{name}` package (parent of {name}/), or a non-bundled "
                f"{name} already on sys.path. Two pip packages cannot both "
                f"own the top-level name {name!r}; PATH is how they coexist. "
                "See docs: companions."
            )
        entry = found
    _bind(name, entry)
    mod = sys.modules[name]
    if getattr(mod, BUNDLE_MARKER, False):
        raise RuntimeError(
            f"{_env_key(name, '')}=external loaded the ligandparam-bundled "
            f"{name} at {Path(mod.__file__).resolve().parent}. Point "
            f"{_env_key(name, '_PATH')} at an independent source tree."
        )
    origin = Path(mod.__file__).resolve().parent
    return CompanionInfo(
        name=name,
        mode="external",
        origin=origin,
        path_entry=entry.resolve(),
        bundled=False,
    )


def _resolve_path_entry(name: str, hint: Path) -> Path:
    hint = hint.expanduser().resolve()
    if (hint / name / "__init__.py").is_file():
        return hint
    if hint.name == name and (hint / "__init__.py").is_file():
        return hint.parent
    raise RuntimeError(
        f"{_env_key(name, '_PATH')}={hint} is not a {name} tree. "
        f"Pass the parent of {name}/ (e.g. .../src) or the {name}/ "
        "package directory itself."
    )


def _init_is_bundled(init_py: Path) -> bool:
    try:
        text = init_py.read_text(encoding="utf-8")
    except OSError:
        return False
    return f"{BUNDLE_MARKER} = True" in text


def _find_non_bundled(name: str, bundled_root: Path) -> Path | None:
    bundled_pkg = (bundled_root / name).resolve()
    for raw in sys.path:
        if not raw:
            continue
        entry = Path(raw)
        init_py = entry / name / "__init__.py"
        if not init_py.is_file():
            continue
        pkg = init_py.parent.resolve()
        if pkg == bundled_pkg or _init_is_bundled(init_py):
            continue
        return entry.resolve()
    return None


def _already_bound(name: str, path_entry: Path) -> bool:
    mod = sys.modules.get(name)
    file = getattr(mod, "__file__", None) if mod is not None else None
    if not file:
        return False
    origin = Path(file).resolve().parent
    expected = (path_entry / name).resolve()
    return origin == expected


def _purge(name: str) -> None:
    prefix = name + "."
    for key in list(sys.modules):
        if key == name or key.startswith(prefix):
            del sys.modules[key]


def _prepend_sys_path(entry: str) -> None:
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)


def _bind(name: str, path_entry: Path) -> None:
    entry = str(path_entry.resolve())
    if _already_bound(name, path_entry):
        _prepend_sys_path(entry)
        return
    _purge(name)
    _prepend_sys_path(entry)
    importlib.invalidate_caches()
    importlib.import_module(name)


def _apply_pythonpath(infos: Mapping[str, CompanionInfo]) -> None:
    """Put chosen trees first so ffpopt.bin subprocesses import the same copy."""
    entries: list[str] = []
    for name in ("ffpopt", "scission"):
        item = str(infos[name].path_entry)
        if item not in entries:
            entries.append(item)
    old = os.environ.get("PYTHONPATH", "")
    rest = [part for part in old.split(os.pathsep) if part and part not in entries]
    os.environ["PYTHONPATH"] = os.pathsep.join(entries + rest)
