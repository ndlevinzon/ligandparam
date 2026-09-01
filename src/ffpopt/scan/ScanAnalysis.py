from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScanCompareConfig:
    """ Tunable thresholds for :func:`compare_scans`.

    Defaults are reasonable starting points for kcal/mol scans on a ~10 deg
    grid; tighten or loosen per workflow.

    Attributes
    ----------
    angle_tol : float, optional
        Maximum absolute angle difference (degrees, periodic) for two
        extrema to be considered the same extremum. Default is 15.0.
    energy_tol : float, optional
        Maximum absolute energy difference (kcal/mol) between matched-pair
        extrema. Default is 0.5.
    barrier_tol : float, optional
        Maximum absolute barrier-height difference (kcal/mol) between the
        two profiles' overall barriers (max - min). Default is 1.0.
    flat_threshold : float, optional
        If the HL barrier height is below this (kcal/mol), the dihedral is
        treated as "soft": comparison short-circuits with ``is_close=True``
        and ``is_flat=True`` regardless of LL agreement. Default is 1.0.
    require_extremum_identity : bool, optional
        If True, an HL maximum can only match an LL maximum (and likewise
        for minima). If False, match by angle position regardless of type.
        Default is True.
    prominence : float, optional
        Minimum peak prominence (kcal/mol) passed to
        ``scipy.signal.find_peaks`` when detecting maxima/minima.
        Suppresses small numerical wiggles. Default is 0.3.
    require_same_count : bool, optional
        If True, profiles must have the same number of maxima *and* the
        same number of minima for ``is_close=True``. Default is True.
    interpolate_to_finer_grid : bool, optional
        If HL and LL are sampled on different angle deltas, linearly
        interpolate the coarser profile onto the finer grid before extrema
        detection. If False, raise on mismatched grids. Default is True.
    """

    angle_tol: float = 15.0
    energy_tol: float = 0.5
    barrier_tol: float = 1.0
    flat_threshold: float = 1.0
    require_extremum_identity: bool = True
    prominence: float = 0.3
    require_same_count: bool = True
    interpolate_to_finer_grid: bool = True


@dataclass
class ScanComparison:
    """ Structured result of :func:`compare_scans`.

    Attributes
    ----------
    is_close : bool
        Overall verdict. True means the two profiles agree within the
        configured thresholds and the dihedral can be dropped from
        refitting.
    is_flat : bool
        True if comparison was short-circuited because the HL barrier was
        below ``ScanCompareConfig.flat_threshold``.
    hl_extrema : list of tuple
        Detected HL extrema as ``(angle, energy, kind)`` tuples with
        ``kind in {'max', 'min'}``.
    ll_extrema : list of tuple
        Detected LL extrema in the same format.
    matched : list of tuple
        Pairs of indices into ``hl_extrema`` / ``ll_extrema`` as
        ``(hl_idx, ll_idx, dangle, denergy)``.
    unmatched_hl : list of int
        Indices into ``hl_extrema`` that found no LL partner within tol.
    unmatched_ll : list of int
        Indices into ``ll_extrema`` that found no HL partner within tol.
    barrier_hl : float
        Overall HL barrier height (max - min, kcal/mol).
    barrier_ll : float
        Overall LL barrier height (max - min, kcal/mol).
    reasons : list of str
        Human-readable strings explaining each criterion failure. Empty
        when ``is_close=True``.
    """

    is_close: bool
    is_flat: bool
    hl_extrema: list[tuple[float, float, str]] = field(default_factory=list)
    ll_extrema: list[tuple[float, float, str]] = field(default_factory=list)
    matched: list[tuple[int, int, float, float]] = field(default_factory=list)
    unmatched_hl: list[int] = field(default_factory=list)
    unmatched_ll: list[int] = field(default_factory=list)
    barrier_hl: float = 0.0
    barrier_ll: float = 0.0
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _periodic_angle_diff(a: float, b: float, period: float = 360.0) -> float:
    """ Smallest ``|a - b|`` under modular arithmetic. Always non-negative."""
    d = abs(a - b) % period
    return min(d, period - d)


def _to_sorted_arrays(angles, energies) -> tuple[np.ndarray, np.ndarray]:
    """ Sort by angle and modulo into ``[0, 360)``. Validates shape."""
    a = np.asarray(angles, dtype=float).ravel()
    e = np.asarray(energies, dtype=float).ravel()
    if a.shape != e.shape:
        raise ValueError(
            f"angles and energies shapes differ: {a.shape} vs {e.shape}"
        )
    if a.size == 0:
        raise ValueError("empty scan profile")
    a = a % 360.0
    order = np.argsort(a)
    return a[order], e[order]


def _interpolate_to(
    target_angles: np.ndarray, src_angles: np.ndarray, src_energies: np.ndarray
) -> np.ndarray:
    """ Periodic linear interpolation of ``(src_angles, src_energies)`` onto
    ``target_angles`` (angles in degrees, period 360)."""
    # Extend source with a wrap-around point so np.interp handles edges.
    a_ext = np.concatenate([src_angles, [src_angles[0] + 360.0]])
    e_ext = np.concatenate([src_energies, [src_energies[0]]])
    return np.interp(target_angles % 360.0, a_ext, e_ext)


def _find_periodic_extrema(
    angles: np.ndarray, energies: np.ndarray, prominence: float
) -> list[tuple[float, float, str]]:
    """ Detect maxima and minima of a periodic 0-360 deg energy profile.

    Returns ``[(angle, energy, kind), ...]`` with ``kind in {'max', 'min'}``.
    Wrap-around is handled by tiling one extra cycle on each side and
    de-duplicating into the canonical ``[0, 360)`` window.
    """
    from scipy.signal import find_peaks

    n = angles.size
    if n < 3:
        # Not enough points to define an interior extremum; return empty.
        return []

    # Tile three cycles: [-360, 0) ? [0, 360) ? [360, 720). Run find_peaks
    # on the tiled energy array; keep only peaks whose angle lies in [0, 360).
    e_tiled = np.concatenate([energies, energies, energies])
    a_tiled = np.concatenate([angles - 360.0, angles, angles + 360.0])

    extrema: list[tuple[float, float, str]] = []

    for kind, signed in (("max", e_tiled), ("min", -e_tiled)):
        idxs, _ = find_peaks(signed, prominence=prominence)
        for idx in idxs:
            ang = a_tiled[idx]
            if 0.0 <= ang < 360.0:
                extrema.append((float(ang), float(e_tiled[idx]), kind))

    # Sort by angle for downstream determinism.
    extrema.sort(key=lambda t: t[0])
    return extrema


def _match_extrema(
    hl: list[tuple[float, float, str]],
    ll: list[tuple[float, float, str]],
    config: ScanCompareConfig,
) -> tuple[
    list[tuple[int, int, float, float]],
    list[int],
    list[int],
]:
    """ Greedy nearest-angle matching of HL extrema to LL extrema.

    Returns
    -------
    matched : list of tuple
        Pairs as ``(hl_idx, ll_idx, dangle, denergy)``.
    unmatched_hl : list of int
        HL indices with no LL partner within ``config.angle_tol``.
    unmatched_ll : list of int
        LL indices with no HL partner within ``config.angle_tol``.
    """
    candidates = []
    for i, (ai, ei, ki) in enumerate(hl):
        for j, (aj, ej, kj) in enumerate(ll):
            if config.require_extremum_identity and ki != kj:
                continue
            da = _periodic_angle_diff(ai, aj)
            if da > config.angle_tol:
                continue
            de = ei - ej
            candidates.append((da, abs(de), i, j, da, de))

    # Greedy: sort by angle delta, then |energy delta|.
    candidates.sort(key=lambda c: (c[0], c[1]))

    used_hl: set[int] = set()
    used_ll: set[int] = set()
    matched: list[tuple[int, int, float, float]] = []
    for _, _, i, j, da, de in candidates:
        if i in used_hl or j in used_ll:
            continue
        matched.append((i, j, da, de))
        used_hl.add(i)
        used_ll.add(j)

    unmatched_hl = [i for i in range(len(hl)) if i not in used_hl]
    unmatched_ll = [j for j in range(len(ll)) if j not in used_ll]
    return matched, unmatched_hl, unmatched_ll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_scans(
    hl_angles: Sequence[float],
    hl_energies: Sequence[float],
    ll_angles: Sequence[float],
    ll_energies: Sequence[float],
    config: ScanCompareConfig | None = None,
) -> ScanComparison:
    """ Decide whether two dihedral scans are "close enough."

    Both profiles are min-shifted internally so the global minimum sits at 0;
    pass in either shifted or unshifted energies (kcal/mol).

    Parameters
    ----------
    hl_angles : sequence of float
        High-level scan angles in degrees.
    hl_energies : sequence of float
        High-level scan energies (kcal/mol), aligned with ``hl_angles``.
    ll_angles : sequence of float
        Low-level scan angles in degrees.
    ll_energies : sequence of float
        Low-level scan energies (kcal/mol), aligned with ``ll_angles``.
    config : ScanCompareConfig, optional
        Tunable thresholds for the comparison. Default is None, which uses
        the :class:`ScanCompareConfig` defaults.

    Returns
    -------
    ScanComparison
        Structured verdict and diagnostic detail. ``is_close=True`` means
        the dihedral can be dropped from refitting. ``reasons`` lists every
        criterion failure when ``is_close=False`` (empty otherwise).
    """
    if config is None:
        config = ScanCompareConfig()

    a_hl, e_hl = _to_sorted_arrays(hl_angles, hl_energies)
    a_ll, e_ll = _to_sorted_arrays(ll_angles, ll_energies)

    # Min-shift (defensive; the wavefront already does this, but raw JSON
    # energies from struct.data["energy"] are not necessarily min-shifted).
    e_hl = e_hl - e_hl.min()
    e_ll = e_ll - e_ll.min()

    # Align grids: interpolate the coarser onto the finer.
    same_grid = a_hl.shape == a_ll.shape and np.allclose(a_hl, a_ll)
    if not same_grid:
        if not config.interpolate_to_finer_grid:
            raise ValueError(
                "HL and LL scans are on different angle grids and "
                "interpolate_to_finer_grid=False"
            )
        if a_hl.size >= a_ll.size:
            e_ll = _interpolate_to(a_hl, a_ll, e_ll)
            a_ll = a_hl
        else:
            e_hl = _interpolate_to(a_ll, a_hl, e_hl)
            a_hl = a_ll

    barrier_hl = float(e_hl.max() - e_hl.min())
    barrier_ll = float(e_ll.max() - e_ll.min())

    # Early exit: HL profile is too flat to bother fitting.
    if barrier_hl < config.flat_threshold:
        return ScanComparison(
            is_close=True,
            is_flat=True,
            barrier_hl=barrier_hl,
            barrier_ll=barrier_ll,
            reasons=[],
        )

    hl_extrema = _find_periodic_extrema(a_hl, e_hl, config.prominence)
    ll_extrema = _find_periodic_extrema(a_ll, e_ll, config.prominence)

    matched, unmatched_hl, unmatched_ll = _match_extrema(
        hl_extrema, ll_extrema, config
    )

    reasons: list[str] = []

    if abs(barrier_hl - barrier_ll) > config.barrier_tol:
        reasons.append(
            f"barrier delta {abs(barrier_hl - barrier_ll):.3f} kcal/mol "
            f"exceeds tol {config.barrier_tol}"
        )

    if config.require_same_count:
        n_max_hl = sum(1 for x in hl_extrema if x[2] == "max")
        n_max_ll = sum(1 for x in ll_extrema if x[2] == "max")
        n_min_hl = sum(1 for x in hl_extrema if x[2] == "min")
        n_min_ll = sum(1 for x in ll_extrema if x[2] == "min")
        if n_max_hl != n_max_ll:
            reasons.append(
                f"max count differs: HL={n_max_hl} vs LL={n_max_ll}"
            )
        if n_min_hl != n_min_ll:
            reasons.append(
                f"min count differs: HL={n_min_hl} vs LL={n_min_ll}"
            )

    if unmatched_hl:
        ang_str = ", ".join(f"{hl_extrema[i][0]:.1f} deg" for i in unmatched_hl)
        reasons.append(f"unmatched HL extrema at angle(s): {ang_str}")
    if unmatched_ll:
        ang_str = ", ".join(f"{ll_extrema[j][0]:.1f} deg" for j in unmatched_ll)
        reasons.append(f"unmatched LL extrema at angle(s): {ang_str}")

    bad_pairs = [
        (i, j, da, de)
        for (i, j, da, de) in matched
        if abs(de) > config.energy_tol
    ]
    if bad_pairs:
        worst = max(bad_pairs, key=lambda p: abs(p[3]))
        reasons.append(
            f"matched-pair energy delta {abs(worst[3]):.3f} kcal/mol "
            f"exceeds tol {config.energy_tol} "
            f"(at HL {hl_extrema[worst[0]][0]:.1f} deg)"
        )

    return ScanComparison(
        is_close=not reasons,
        is_flat=False,
        hl_extrema=hl_extrema,
        ll_extrema=ll_extrema,
        matched=matched,
        unmatched_hl=unmatched_hl,
        unmatched_ll=unmatched_ll,
        barrier_hl=barrier_hl,
        barrier_ll=barrier_ll,
        reasons=reasons,
    )


def load_scan_dat(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """ Load a wavefront ``.dat`` file and return ``(angles, energies)``.

    The companion file written by :func:`ffpopt.scan.WaveFront.run_dihed_wavefront`
    has three columns per line: ``angle e_shifted e_unshifted``. This
    function reads the first two - the shifted energies are already in
    kcal/mol.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.dat`` scan file.

    Returns
    -------
    tuple of numpy.ndarray
        ``(angles, energies_shifted_kcal_per_mol)``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"scan .dat not found: {p}")
    data = np.loadtxt(str(p))
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1]


def load_scan_json(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """ Load a wavefront ``.json`` file and return ``(angles, energies)``.

    The dihedral angle comes from the first 4-index constraint on each
    structure; the energy comes from ``struct.data['energy']`` (ASE eV)
    and is converted to kcal/mol.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``ListOfStruct`` JSON file with per-conformer dihedral
        constraints and energies.

    Returns
    -------
    tuple of numpy.ndarray
        ``(angles, energies_kcal_per_mol)``.
    """
    from ffpopt.Struct import ListOfStruct
    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT

    kcal_per_ev = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    los = ListOfStruct.from_file(str(path))

    angles: list[float] = []
    energies: list[float] = []
    for s in los.structs:
        ang = None
        for c in s.data.get("constraints", []):
            if len(c.get("idxs", [])) == 4 and c.get("value") is not None:
                ang = float(c["value"])
                break
        if ang is None or s.data.get("energy") is None:
            continue
        angles.append(ang)
        energies.append(float(s.data["energy"]) * kcal_per_ev)

    if not angles:
        raise ValueError(
            f"no usable (angle, energy) pairs found in {path}; expected each "
            "structure to carry a 4-index dihedral constraint with a value "
            "and a non-None data['energy']"
        )
    return np.asarray(angles), np.asarray(energies)


def _load_any(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """ Dispatch a scan-file loader by extension.

    ``.dat`` -> :func:`load_scan_dat`; ``.json`` -> :func:`load_scan_json`.
    Any other extension raises ``ValueError``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a wavefront ``.dat`` or ``.json`` scan file.

    Returns
    -------
    tuple of numpy.ndarray
        ``(angles, energies_kcal_per_mol)``.
    """
    p = Path(path)
    if p.suffix == ".dat":
        if not p.is_file():
            alt = p.with_suffix(".json")
            if alt.is_file():
                return load_scan_json(alt)
            raise FileNotFoundError(
                f"scan file not found: {p} (no companion {alt.name} either)"
            )
        return load_scan_dat(p)
    if p.suffix == ".json":
        return load_scan_json(p)
    raise ValueError(f"unsupported scan file extension: {p.suffix} ({p})")


def compare_scan_files(
    hl_path: str | Path,
    ll_path: str | Path,
    config: ScanCompareConfig | None = None,
    *,
    plot_path: str | Path | None = None,
    plot_title: str | None = None,
    structure_image_path: str | Path | None = None,
) -> ScanComparison:
    """ Load two scan files and compare via :func:`compare_scans`.

    Each path may be a ``.dat`` (wavefront's companion file) or a ``.json``
    (``ListOfStruct`` with per-conformer dihedral constraints + energies).
    When ``plot_path`` is given, also calls :func:`plot_comparison` to save
    a PNG with extrema highlighted, the matched-pair connections, and the
    list of failed criteria.

    Parameters
    ----------
    hl_path : str or pathlib.Path
        Path to the high-level scan file (``.dat`` or ``.json``).
    ll_path : str or pathlib.Path
        Path to the low-level scan file (``.dat`` or ``.json``).
    config : ScanCompareConfig, optional
        Tunable thresholds for the comparison. Default is None, which uses
        the :class:`ScanCompareConfig` defaults.
    plot_path : str or pathlib.Path, optional
        If given, also save a comparison plot to this path via
        :func:`plot_comparison`. Default is None (no plot).
    plot_title : str, optional
        Title prefix for the plot. Default is None, which uses
        ``"<hl_stem> vs <ll_stem>"``.
    structure_image_path : str or pathlib.Path, optional
        2D fragment-structure image (PNG/JPG or SVG) to render as a top
        panel on the plot. Default is None (no structure panel).

    Returns
    -------
    ScanComparison
        Structured verdict and diagnostic detail (see :func:`compare_scans`).
    """
    a_hl, e_hl = _load_any(hl_path)
    a_ll, e_ll = _load_any(ll_path)
    result = compare_scans(a_hl, e_hl, a_ll, e_ll, config=config)
    if plot_path is not None:
        plot_comparison(
            a_hl,
            e_hl,
            a_ll,
            e_ll,
            result,
            out_path=plot_path,
            title=plot_title,
            hl_label=Path(hl_path).stem,
            ll_label=Path(ll_path).stem,
            config=config,
            structure_image_path=structure_image_path,
        )
    return result


def _load_structure_image_array(path: Path):
    """ Rasterize a 2D structure image into an RGBA array for plotting.

    Handles ``.png`` / ``.jpg`` via :mod:`matplotlib.image` directly. For
    ``.svg`` drawings, prefers :mod:`cairosvg` (no
    subprocess), falls back to the ``rsvg-convert`` system binary, and
    returns ``None`` with a warning print if neither is available.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.png``, ``.jpg``, or ``.svg`` image.

    Returns
    -------
    numpy.ndarray or None
        RGBA array suitable for :meth:`matplotlib.axes.Axes.imshow`, or
        ``None`` when the image is missing or the SVG cannot be rasterized
        with the available tooling.
    """
    suffix = path.suffix.lower()
    if not path.exists():
        print(f"[plot] structure image missing: {path} - skipping panel")
        return None
    if suffix in {".png", ".jpg", ".jpeg"}:
        import matplotlib.image as mpimg
        return mpimg.imread(str(path))
    if suffix == ".svg":
        # Preferred: cairosvg (pure-python, fast)
        try:
            import io
            import cairosvg
            import matplotlib.image as mpimg
            png_bytes = cairosvg.svg2png(url=str(path), output_width=1200)
            return mpimg.imread(io.BytesIO(png_bytes), format="png")
        except ImportError:
            pass
        # Fallback: rsvg-convert subprocess (commonly available on Linux)
        import shutil
        if shutil.which("rsvg-convert"):
            try:
                import io
                import subprocess as _sp
                import matplotlib.image as mpimg
                result = _sp.run(
                    ["rsvg-convert", "-w", "1200", str(path)],
                    check=True, capture_output=True,
                )
                return mpimg.imread(io.BytesIO(result.stdout), format="png")
            except Exception as exc:
                print(f"[plot] rsvg-convert failed for {path}: {exc}")
                return None
        print(
            f"[plot] cannot rasterize {path}: install `cairosvg` or put "
            f"`rsvg-convert` on PATH - skipping structure panel"
        )
        return None
    print(f"[plot] unsupported structure-image extension {suffix!r} ({path})")
    return None


def plot_comparison(
    hl_angles: Sequence[float],
    hl_energies: Sequence[float],
    ll_angles: Sequence[float],
    ll_energies: Sequence[float],
    comparison: ScanComparison,
    *,
    out_path: str | Path,
    title: str | None = None,
    hl_label: str = "HL",
    ll_label: str = "LL",
    config: ScanCompareConfig | None = None,
    structure_image_path: str | Path | None = None,
) -> Path:
    """ Save a comparison plot for two dihedral scan profiles.

    Layout: an optional top panel with the 2D fragment structure (when
    ``structure_image_path`` is provided and rasterizable), the scan
    comparison in the middle, and an optional bottom panel listing each
    failed criterion when ``comparison.reasons`` is non-empty. Empty panels
    are omitted so the figure shrinks to fit.

    The scan panel shows both profiles (min-shifted to match
    :func:`compare_scans`' internal view) with extrema highlighted: filled
    triangles for matched extrema (up=max, down=min), open red triangles
    for unmatched extrema. Matched-pair connections are coloured by the
    ratio of the absolute matched-pair energy delta to ``config.energy_tol``
    (green when at or below tol, red when above). The figure title reports
    the overall verdict (close / flat / fail) and barrier heights.

    Parameters
    ----------
    hl_angles : sequence of float
        High-level scan angles in degrees.
    hl_energies : sequence of float
        High-level scan energies (kcal/mol), aligned with ``hl_angles``.
    ll_angles : sequence of float
        Low-level scan angles in degrees.
    ll_energies : sequence of float
        Low-level scan energies (kcal/mol), aligned with ``ll_angles``.
    comparison : ScanComparison
        Verdict and diagnostic detail from :func:`compare_scans`. Drives
        the extrema markers, matched-pair connectors, and reasons panel.
    out_path : str or pathlib.Path
        Output PNG path. Parent directories are created if missing.
    title : str, optional
        Title prefix. Default is None, which uses ``"<hl_label> vs <ll_label>"``.
    hl_label : str, optional
        Legend label for the high-level profile. Default is "HL".
    ll_label : str, optional
        Legend label for the low-level profile. Default is "LL".
    config : ScanCompareConfig, optional
        Used only to colour the matched-pair connectors against
        ``energy_tol``. Default is None (uses :class:`ScanCompareConfig`
        defaults).
    structure_image_path : str or pathlib.Path, optional
        2D fragment-structure image (PNG/JPG or SVG) to render as a top
        panel. Default is None (no structure panel).

    Returns
    -------
    pathlib.Path
        The written output path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = config if config is not None else ScanCompareConfig()

    a_hl, e_hl = _to_sorted_arrays(hl_angles, hl_energies)
    a_ll, e_ll = _to_sorted_arrays(ll_angles, ll_energies)
    e_hl = e_hl - e_hl.min()
    e_ll = e_ll - e_ll.min()

    image_arr = None
    if structure_image_path is not None:
        image_arr = _load_structure_image_array(Path(structure_image_path))
    has_image_panel = image_arr is not None
    has_reasons_panel = bool(comparison.reasons)

    height_ratios: list[float] = []
    if has_image_panel:
        height_ratios.append(3.0)
    height_ratios.append(5.0)
    if has_reasons_panel:
        height_ratios.append(1.0 + 0.25 * len(comparison.reasons))

    fig = plt.figure(figsize=(9, sum(height_ratios) * 0.9), constrained_layout=True)
    gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios)

    row = 0
    if has_image_panel:
        ax_img = fig.add_subplot(gs[row]); row += 1
        ax_img.imshow(image_arr)
        ax_img.axis("off")
    ax = fig.add_subplot(gs[row]); row += 1
    ax_text = None
    if has_reasons_panel:
        ax_text = fig.add_subplot(gs[row])
        ax_text.axis("off")

    ax.plot(a_hl, e_hl, "-o", color="tab:blue", label=hl_label, markersize=4, linewidth=1.4)
    ax.plot(a_ll, e_ll, "-s", color="tab:orange", label=ll_label, markersize=4, linewidth=1.4)

    def _draw_extrema(extrema, unmatched, base_color):
        for i, (ang, en, kind) in enumerate(extrema):
            marker = "^" if kind == "max" else "v"
            if i in unmatched:
                ax.plot(
                    ang, en,
                    marker=marker, markersize=13,
                    markerfacecolor="none", markeredgecolor="red",
                    markeredgewidth=2.0, linestyle="none",
                )
            else:
                ax.plot(
                    ang, en,
                    marker=marker, markersize=10,
                    markerfacecolor=base_color, markeredgecolor=base_color,
                    linestyle="none",
                )

    _draw_extrema(comparison.hl_extrema, set(comparison.unmatched_hl), "tab:blue")
    _draw_extrema(comparison.ll_extrema, set(comparison.unmatched_ll), "tab:orange")

    for (i, j, _da, de) in comparison.matched:
        ang_hl, en_hl, _ = comparison.hl_extrema[i]
        ang_ll, en_ll, _ = comparison.ll_extrema[j]
        bad = abs(de) > cfg.energy_tol
        ax.plot(
            [ang_hl, ang_ll], [en_hl, en_ll],
            linestyle=":",
            color="tab:red" if bad else "tab:green",
            linewidth=1.6 if bad else 1.0,
            alpha=0.9 if bad else 0.6,
        )
        if bad:
            ax.annotate(
                f"|DeltaE|={abs(de):.2f}",
                xy=((ang_hl + ang_ll) / 2.0, (en_hl + en_ll) / 2.0),
                fontsize=7, color="tab:red",
                xytext=(3, 3), textcoords="offset points",
            )

    if comparison.is_flat:
        verdict = "FLAT (HL barrier below threshold)"
    elif comparison.is_close:
        verdict = "OK (close)"
    else:
        verdict = "FAIL"
    header = title if title else f"{hl_label} vs {ll_label}"
    fig.suptitle(
        f"{header}  -  {verdict}  "
        f"(barrier HL={comparison.barrier_hl:.2f}, LL={comparison.barrier_ll:.2f} kcal/mol)",
        y=0.995,
    )
    ax.set_xlabel("Dihedral angle (deg)")
    ax.set_ylabel("Energy (kcal/mol, min-shifted)")
    ax.set_xlim(0.0, 360.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    if ax_text is not None:
        reasons_text = "Failed criteria:\n" + "\n".join(
            f"* {r}" for r in comparison.reasons
        )
        ax_text.text(
            0.01, 0.97, reasons_text,
            transform=ax_text.transAxes,
            verticalalignment="top", horizontalalignment="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose",
                      edgecolor="tab:red", alpha=0.95),
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
