"""Packaged ``FFPOPT_*`` defaults plus ``EXPORT`` overlays.

The JSON at ``ffpopt/pkgdata/files/env_defaults.json`` is the store of values
that call sites read. Comments (``//``, ``/* */``) are allowed. Resolution
order:

1. Packaged JSON (ships with the install)
2. Optional overlay file from ``FFPOPT_DEFAULTS=/path/to.json``
3. Per-key ``export FFPOPT_*=...`` (always wins)

``null`` in JSON means "leave to code policy" (for example ASE-first auto).
Internal process flags (``FFPOPT_IN_SPAWN_WORKER``, ``FFPOPT_IN_WALL_CHILD``) are not in this file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

_FALSEY = {"0", "false", "no", "off", ""}

_PACKAGED: dict[str, Any] | None = None
_OVERLAY_CACHE: tuple[str, float, dict[str, Any]] | None = None

DEFAULTS_RESOURCE = "pkgdata/files/env_defaults.json"
OVERLAY_ENV = "FFPOPT_DEFAULTS"


def defaults_path() -> Path:
    """Filesystem path of the packaged defaults JSON."""
    try:
        from importlib.resources import files

        return Path(files("ffpopt") / "pkgdata" / "files" / "env_defaults.json")
    except Exception:
        return Path(__file__).resolve().parents[1] / "pkgdata" / "files" / "env_defaults.json"


def strip_jsonc(text: str) -> str:
    """Remove ``//`` line and ``/* */`` block comments outside of strings."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    escape = False
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            while i < n and text[i] not in "\n\r":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_jsonc(path: Path | str) -> dict[str, Any]:
    """Parse a JSON / JSONC object; drop keys that start with ``_``."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(strip_jsonc(text))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(k): v for k, v in data.items() if not str(k).startswith("_")}


def clear_defaults_cache() -> None:
    """Drop cached packaged / overlay JSON (tests)."""
    global _PACKAGED, _OVERLAY_CACHE
    _PACKAGED = None
    _OVERLAY_CACHE = None


def packaged_defaults() -> dict[str, Any]:
    """Defaults as shipped (no overlay, no ``EXPORT``)."""
    global _PACKAGED
    if _PACKAGED is None:
        _PACKAGED = load_jsonc(defaults_path())
    return dict(_PACKAGED)


def _overlay_defaults() -> dict[str, Any]:
    global _OVERLAY_CACHE
    raw = os.environ.get(OVERLAY_ENV, "").strip()
    if not raw:
        return {}
    path = Path(raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"{OVERLAY_ENV}={raw!r} is not a readable JSON file"
        )
    mtime = path.stat().st_mtime
    key = str(path.resolve())
    if _OVERLAY_CACHE is not None:
        cached_key, cached_mtime, cached = _OVERLAY_CACHE
        if cached_key == key and cached_mtime == mtime:
            return dict(cached)
    data = load_jsonc(path)
    _OVERLAY_CACHE = (key, mtime, data)
    return dict(data)


def json_defaults() -> dict[str, Any]:
    """Packaged JSON with optional ``FFPOPT_DEFAULTS`` overlay (no ``EXPORT``)."""
    data = packaged_defaults()
    data.update(_overlay_defaults())
    return data


def as_bool(value: Any) -> bool:
    """Coerce JSON / env values to bool (``0`` / ``false`` / ``off`` / ``no`` are false)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in _FALSEY


def _coerce(raw: str, packaged: Any) -> Any:
    text = raw.strip()
    if packaged is None:
        return _coerce_untyped(text)
    if isinstance(packaged, bool):
        return as_bool(text)
    if isinstance(packaged, int) and not isinstance(packaged, bool):
        return int(text)
    if isinstance(packaged, float):
        return float(text)
    if isinstance(packaged, str):
        return text
    return _coerce_untyped(text)


def _coerce_untyped(text: str) -> Any:
    low = text.lower()
    if low in {"1", "true", "yes", "on"}:
        return True
    if low in {"0", "false", "no", "off"}:
        return False
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def env_value(name: str, default: Any = None) -> Any:
    """Resolved value: ``EXPORT`` if set, else JSON (overlay), else ``default``.

    ``null`` in JSON is returned as ``None`` (callers apply auto-policy).
    """
    packaged = json_defaults()
    if name in packaged:
        fallback = packaged[name]
    else:
        fallback = default
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return fallback
    try:
        return _coerce(str(raw), fallback)
    except (TypeError, ValueError):
        return fallback


def env_bool(name: str, default: bool = False) -> bool:
    value = env_value(name, default)
    if value is None:
        return bool(default)
    return as_bool(value)


def env_int(name: str, default: int = 0) -> int:
    value = env_value(name, default)
    if value is None:
        return int(default)
    return int(value)


def env_float(name: str, default: float = 0.0) -> float:
    value = env_value(name, default)
    if value is None:
        return float(default)
    return float(value)


def env_str(name: str, default: str = "") -> str:
    value = env_value(name, default)
    if value is None:
        return str(default)
    return str(value)


def resolved_defaults() -> dict[str, Any]:
    """Full map of JSON keys after overlay + ``EXPORT`` (what the code uses)."""
    data = json_defaults()
    return {key: env_value(key) for key in data}


def dump_resolved(path: Path | str, *, extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Write the resolved defaults (no comments) for a job directory."""
    out = resolved_defaults()
    if extra:
        out.update(dict(extra))
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest
