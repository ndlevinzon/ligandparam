"""Robust GFN2-xTB (tblite) construction and SCF retry helpers.

Constrained dihedral steps often land on strained geometries where the
default tblite SCC (250 cycles, 300 K, SAD guess) fails. This module
provides stabler defaults and an escalating retry ladder used by
:class:`~ffpopt.ase.calculator.GenCalculator` and
:class:`~ffpopt.ase.calculator.QDpi2Calculator`.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, MutableMapping, Optional


# Defaults tuned for large / constrained ligands (vs tblite stock defaults).
_DEFAULT_TBLITE: dict[str, Any] = {
    "method": "GFN2-xTB",
    "max_iterations": 500,
    "electronic_temperature": 500.0,
    "mixer_damping": 0.25,
    "verbosity": -1,
    "cache_api": True,
}

# Prefer eeq when supported; fall back handled in make_tblite_calculator.
_DEFAULT_GUESS = "eeq"

# Env → parameter mapping.
_ENV_MAP = {
    "FFPOPT_XTB_MAX_ITER": ("max_iterations", int),
    "FFPOPT_XTB_ETEMP": ("electronic_temperature", float),
    "FFPOPT_XTB_MIXER_DAMPING": ("mixer_damping", float),
    "FFPOPT_XTB_GUESS": ("initial_guess", str),
}


def _is_scf_failure(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a tblite SCC non-convergence."""
    msg = str(exc).lower()
    if "scf not converged" in msg:
        return True
    # Walk the exception chain (CalculationFailed wrapping TBLiteRuntimeError).
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _is_scf_failure(cause)
    return False


def tblite_kwargs_from_env(
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Merge robust defaults, env overrides, and explicit ``overrides``.

    Environment variables (optional):

    * ``FFPOPT_XTB_MAX_ITER``
    * ``FFPOPT_XTB_ETEMP``
    * ``FFPOPT_XTB_MIXER_DAMPING``
    * ``FFPOPT_XTB_GUESS``
    """
    kwargs = dict(_DEFAULT_TBLITE)
    kwargs["initial_guess"] = _DEFAULT_GUESS
    for env_key, (param, caster) in _ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            kwargs[param] = caster(raw)
        except (TypeError, ValueError):
            continue
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


def make_tblite_calculator(charge: Optional[float] = None, **overrides: Any):
    """Construct a :class:`tblite.ase.TBLite` with robust SCF defaults.

    If ``initial_guess='eeq'`` is rejected by the installed tblite, falls
    back to ``'sad'``. Unknown kwargs that older tblite versions reject are
    dropped one at a time until construction succeeds (keeps core defaults).
    """
    from tblite.ase import TBLite

    kwargs = tblite_kwargs_from_env(overrides)
    if charge is not None:
        kwargs["charge"] = charge

    # Prefer initial_guess; some older docs used ``guess``.
    attempt = dict(kwargs)
    last_err: Optional[BaseException] = None
    for _ in range(len(attempt) + 2):
        try:
            return TBLite(**attempt)
        except TypeError as exc:
            last_err = exc
            msg = str(exc)
            # Drop unsupported keyword mentioned in the error, if any.
            dropped = False
            for key in list(attempt):
                if key in ("method", "charge"):
                    continue
                if f"'{key}'" in msg or f'"{key}"' in msg or key in msg:
                    attempt.pop(key, None)
                    dropped = True
                    break
            if not dropped and "initial_guess" in attempt:
                # Common: rename or fall back to sad.
                attempt.pop("initial_guess", None)
                attempt["guess"] = kwargs.get("initial_guess", "sad")
                dropped = True
            if not dropped and attempt.get("initial_guess") == "eeq":
                attempt["initial_guess"] = "sad"
                dropped = True
            if not dropped and attempt.get("guess") == "eeq":
                attempt["guess"] = "sad"
                dropped = True
            if not dropped:
                # Drop optional knobs from newest to oldest.
                for key in (
                    "annealing",
                    "mixer_memory",
                    "mixer",
                    "mixer_damping",
                    "initial_guess",
                    "guess",
                    "electronic_temperature",
                    "max_iterations",
                    "cache_api",
                    "verbosity",
                ):
                    if key in attempt and key not in ("method", "charge"):
                        attempt.pop(key)
                        dropped = True
                        break
            if not dropped:
                raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("make_tblite_calculator failed without an error")


def _scf_retry_configs(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Escalating SCF settings for retries after non-convergence."""
    charge = base.get("charge")
    common = {
        "method": base.get("method", "GFN2-xTB"),
        "verbosity": base.get("verbosity", -1),
        "cache_api": base.get("cache_api", True),
        "initial_guess": base.get("initial_guess", base.get("guess", "eeq")),
    }
    if charge is not None:
        common["charge"] = charge

    return [
        {
            **common,
            "electronic_temperature": 1000.0,
            "max_iterations": 750,
            "mixer_damping": base.get("mixer_damping", 0.25),
        },
        {
            **common,
            "electronic_temperature": 500.0,
            "max_iterations": 750,
            "mixer_damping": base.get("mixer_damping", 0.25),
            "annealing": (2000.0, 500.0, 5),
        },
        {
            **common,
            "electronic_temperature": 1000.0,
            "max_iterations": 1000,
            "mixer_damping": 0.15,
        },
    ]


def run_tblite_with_scf_retries(atoms, calc):
    """Evaluate energy/forces on ``atoms``, retrying on SCF non-convergence.

    Parameters
    ----------
    atoms : ase.Atoms
        Geometry to evaluate; ``atoms.calc`` is set to ``calc`` (and may be
        replaced by a rebuilt calculator on retry).
    calc
        Initial TBLite (or compatible) calculator.

    Returns
    -------
    tuple
        ``(energy, forces, calc)`` where ``calc`` is the calculator that
        succeeded (may differ from the input after retries).

    Raises
    ------
    Exception
        Re-raises the last failure if all retries are exhausted, or any
        non-SCF error immediately.
    """
    atoms.calc = calc
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        return energy, forces, calc
    except Exception as exc:
        if not _is_scf_failure(exc):
            raise
        last_exc = exc

    # Pull whatever we can from the failed calculator for charge / method.
    base: dict[str, Any] = {}
    params = getattr(calc, "parameters", None)
    if isinstance(params, Mapping):
        base.update(params)
    elif hasattr(calc, "todict"):
        try:
            base.update(calc.todict())
        except Exception:
            pass
    if "charge" not in base and getattr(calc, "charge", None) is not None:
        base["charge"] = calc.charge

    for i, cfg in enumerate(_scf_retry_configs(base), start=1):
        label = (
            f"etemp={cfg.get('electronic_temperature')}, "
            f"max_iter={cfg.get('max_iterations')}"
        )
        if "annealing" in cfg:
            label += f", annealing={cfg['annealing']}"
        sys.stderr.write(
            f"[ffpopt] xTB SCF failed; retry {i}/3 with {label}\n"
        )
        try:
            new_calc = make_tblite_calculator(**cfg)
        except TypeError:
            # Annealing / kwargs unsupported — skip this rung.
            continue
        atoms.calc = new_calc
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            return energy, forces, new_calc
        except Exception as exc:
            if not _is_scf_failure(exc):
                raise
            last_exc = exc

    raise last_exc
