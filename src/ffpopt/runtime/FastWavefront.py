"""Fast wavefront presets and allocation helpers for dihedral scans.

Enable with ``FFPOPT_FAST_WAVEFRONT=1`` or ``--fast`` on ``lig-dihed-correct``.
Presets favor wall-time over ultra-tight converge: looser geomeTRIC criteria
and fewer maxiters. Scan ``delta`` stays at the library default (10 deg) so
HL and LL share one angle grid. Depth vs breadth when splitting
``nproc`` is decided by :func:`prefer_wavefront_depth`,
:func:`prefer_bond_pool_depth`, and :func:`prefer_fragment_pool_depth`
(small fair-share leases prefer concurrent outer jobs).
"""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

from ffpopt.runtime.EnvDefaults import as_bool, env_bool, env_int, env_value

# Documented library defaults for knobs that ``--fast`` may override when
# the caller left them at the stock value.
LIBRARY_DEFAULTS: dict[str, Any] = {
    "delta": 10,
    "geometric_maxiter": 500,
    "geometric_converge": "set GAU",
    "wf_convergence_threshold": 0.01,
    "ase_opt_tol": 0.01,
}

# Applied when fast mode is on and the corresponding knob is still at
# LIBRARY_DEFAULTS (explicit user overrides always win).
FAST_WAVEFRONT_PRESETS: dict[str, Any] = {
    "geometric_maxiter": 200,
    "geometric_converge": "set GAU_LOOSE",
    "wf_convergence_threshold": 0.05,
    "ase_opt_tol": 0.03,
}


def fast_wavefront_enabled(explicit: Optional[bool] = None) -> bool:
    """Return whether fast-wavefront presets are active."""
    if explicit is not None:
        return bool(explicit)
    return env_bool("FFPOPT_FAST_WAVEFRONT")


def prefer_wavefront_depth(*, model: str | None = None, fast: Optional[bool] = None) -> bool:
    """Prefer larger nested wavefront pools over more concurrent outer jobs.

    Default on for:

    * ``FFPOPT_PREF_WF_DEPTH=1``
    * XTB-like HL models when fast mode is on

    Sander / Amber MM used to always prefer depth; that is now decided by
    :func:`prefer_bond_pool_depth` so small fair-share leases can run bonds
    concurrently instead of a single 1–2-wide wavefront.
    """
    if env_bool("FFPOPT_PREF_WF_DEPTH"):
        return True
    if env_bool("FFPOPT_PREF_WF_BREADTH"):
        return False
    if not fast_wavefront_enabled(fast):
        return False
    m = (model or "").strip().lower()
    return m in {"xtb", "gfn2-xtb", "gfn2", "tblite"} or m.startswith("xtb")


def prefer_bond_pool_depth(
    *,
    model: str | None,
    nproc: int,
    n_bonds: int,
    prefer: Optional[bool] = None,
) -> bool:
    """Choose bond-pool depth vs breadth for one scan phase.

    With tiny fair-share leases (common when many fragments share ``nproc``),
    depth collapses to ``1 × nproc`` and serializes independent bonds. Prefer
    breadth whenever depth would not keep at least ``n_bonds`` outer workers.
    Explicit ``prefer`` is a model/fast hint only; small-lease breadth and
    ``FFPOPT_PREF_WF_*`` env overrides still win.
    """
    if env_bool("FFPOPT_PREF_WF_DEPTH"):
        return True
    if env_bool("FFPOPT_PREF_WF_BREADTH"):
        return False
    nproc = max(1, int(nproc))
    n_bonds = max(1, int(n_bonds))
    if n_bonds >= 2:
        min_inner = max(1, env_int("FFPOPT_MIN_WF_NPROC"))
        # Depth would force n_outer < n_bonds → run bonds concurrently instead.
        if (nproc // min_inner) < n_bonds:
            return False
    if prefer is not None:
        return bool(prefer)
    return prefer_wavefront_depth(model=model)


def prefer_fragment_pool_depth(
    *,
    model: str | None,
    nproc: int,
    n_fragments: int,
    fast: Optional[bool] = None,
) -> bool:
    """Fragment-pool depth vs breadth for the parent fragmented workflow.

    When many fragments share a modest core budget, prefer breadth so more
    fragments run at once (each with a small wavefront) instead of parking
    half the fragments behind a deep-but-narrow pool.
    """
    if env_bool("FFPOPT_PREF_WF_DEPTH"):
        return True
    if env_bool("FFPOPT_PREF_WF_BREADTH"):
        return False
    nproc = max(1, int(nproc))
    n_fragments = max(1, int(n_fragments))
    if n_fragments >= 4 and (nproc // n_fragments) <= 2:
        return False
    return prefer_wavefront_depth(model=model, fast=fast)


def apply_fast_wavefront_presets(
    kwargs: MutableMapping[str, Any],
    *,
    enabled: Optional[bool] = None,
) -> dict[str, Any]:
    """Overwrite library-default knobs in ``kwargs`` with fast presets.

    Returns the subset of keys that were changed.
    """
    if not fast_wavefront_enabled(enabled):
        return {}
    applied: dict[str, Any] = {}
    for key, preset in FAST_WAVEFRONT_PRESETS.items():
        if key not in kwargs:
            kwargs[key] = preset
            applied[key] = preset
            continue
        cur = kwargs[key]
        if cur == LIBRARY_DEFAULTS.get(key):
            kwargs[key] = preset
            applied[key] = preset
    return applied


def split_nproc_for_items(
    nproc: int,
    n_items: int,
    *,
    prefer_depth: bool = False,
    min_inner: int | None = None,
    flatten_nested: bool = True,
) -> tuple[int, int]:
    """Split ``nproc`` into ``(n_outer_workers, n_inner_per_worker)``.

    When ``prefer_depth`` is True, keep at least ``min_inner`` cores per outer
    worker (default from ``FFPOPT_MIN_WF_NPROC`` or 2) so HL wavefronts are not
    forced to 1-wide when many fragments share a modest allocation.

    When ``flatten_nested`` is True (default), never return both outer and
    inner greater than 1. Nested ``spawn`` pools (bond workers that each open
    a wavefront pool) dominate bootstrap cost inside an already-spawned
    fragment worker. Bond-scan callers that are *not* nested should pass
    ``flatten_nested=False`` with ``prefer_depth=True`` so a 2-bond batch on
    44 cores is ``2 x 22``, not a serial ``1 x 44`` wavefront that cannot fill.
    """
    nproc = max(1, int(nproc))
    n_items = max(1, int(n_items))
    if n_items == 1:
        return 1, nproc
    if not prefer_depth:
        n_outer = min(nproc, n_items)
        n_inner = max(1, nproc // n_outer)
    else:
        if min_inner is None:
            min_inner = max(1, env_int("FFPOPT_MIN_WF_NPROC"))
        min_inner = max(1, int(min_inner))
        n_outer = min(n_items, max(1, nproc // min_inner))
        n_inner = max(1, nproc // n_outer)
    if flatten_nested and n_outer > 1 and n_inner > 1:
        if prefer_depth:
            return 1, nproc
        return min(nproc, n_items), 1
    return n_outer, n_inner


def split_core_budget(total_cores: int, n_jobs: int) -> tuple[int, int]:
    """Split a core budget across concurrent jobs (breadth-first).

    Equivalent to :func:`split_nproc_for_items` with ``prefer_depth=False``.
    Prefer this name from ESP / Gaussian pool callers.
    """
    return split_nproc_for_items(total_cores, n_jobs, prefer_depth=False)


def is_xtb_like_model(model: str | None) -> bool:
    m = (model or "").strip().lower()
    return m in {"xtb", "gfn2-xtb", "gfn2", "tblite"} or m.startswith("xtb")


def is_qdpi2_model(model: str | None) -> bool:
    m = (model or "").strip().lower()
    return m in {"qdpi2", "qdpi"} or m.startswith("qdpi")


def prefer_ase_first_model(model: str | None, *, fast: Optional[bool] = None) -> bool:
    """True when wavefront should default to ASE-first (skip geomeTRIC ladder).

    Always for sander/MM. Under fast mode also for XTB-like and QDpi2 HL.
    Override with ``FFPOPT_ASE_FIRST=0|1``.
    """
    raw = env_value("FFPOPT_ASE_FIRST")
    if raw is not None:
        return as_bool(raw)
    m = (model or "").strip().lower()
    if m in {"sander", "amber", "mm"} or m.startswith("sander"):
        return True
    if not fast_wavefront_enabled(fast):
        return False
    return is_xtb_like_model(m) or is_qdpi2_model(m)


def qdpi2_opt_components(*, fast: Optional[bool] = None) -> str:
    """Which QDpi2 force terms to use during geometry optimization.

    ``both`` (default): DeepPot + GFN2-xTB every step.
    ``xtb``: XTB-only opt (much cheaper); pair with a full QDpi2 single-point
    on the accepted frame for wavefront ranking.
    ``dp``: DeepPot-only opt.

    Under ``--fast`` / ``FFPOPT_FAST_WAVEFRONT=1`` defaults to ``xtb``.
    Override with ``FFPOPT_QDPI2_OPT=both|xtb|dp``.
    """
    raw = env_value("FFPOPT_QDPI2_OPT")
    if raw is not None:
        mode = str(raw).strip().lower()
        if mode in {"both", "full", "all"}:
            return "both"
        if mode in {"xtb", "gfn2", "tblite"}:
            return "xtb"
        if mode in {"dp", "deepmd", "deeppot", "ml"}:
            return "dp"
        return "both"
    if fast_wavefront_enabled(fast):
        return "xtb"
    return "both"


def qdpi2_refine_energy_after_opt(*, fast: Optional[bool] = None) -> bool:
    """Re-score XTB/DP-only QDpi2 opts with a full QDpi2 single-point."""
    raw = env_value("FFPOPT_QDPI2_REFINE")
    if raw is not None:
        return as_bool(raw)
    return qdpi2_opt_components(fast=fast) != "both"


def wf_checkpoint_every(nproc: int, *, fast: Optional[bool] = None) -> int:
    """How many completed nodes between wavefront checkpoints."""
    raw = env_value("FFPOPT_WF_CHECKPOINT_EVERY")
    if raw is not None:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass
    base = max(1, int(nproc))
    if fast_wavefront_enabled(fast):
        return base * 4
    return base


def write_success_node_pickle(*, fast: Optional[bool] = None) -> bool:
    """Whether to write per-node pickles after successful opts.

    Failures always write. Default: off in fast mode, on otherwise.
    Override with ``FFPOPT_WF_NODE_PICKLE=0|1``.
    """
    raw = env_value("FFPOPT_WF_NODE_PICKLE")
    if raw is not None:
        return as_bool(raw)
    return not fast_wavefront_enabled(fast)


def geomopt_verbose() -> bool:
    """Print per-constraint / restraint summaries from GeomOpt.

    Default on; silence in fast mode. Override with ``FFPOPT_GEOMOPT_VERBOSE``.
    """
    raw = env_value("FFPOPT_GEOMOPT_VERBOSE")
    if raw is not None:
        return as_bool(raw)
    return not fast_wavefront_enabled(None)


def fast_recovery_ladder() -> bool:
    """Use a shorter geomeTRIC recovery ladder (primary → loose → soft)."""
    if env_bool("FFPOPT_GEOMOPT_FAST_RECOVERY"):
        return True
    return fast_wavefront_enabled(None)
