#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


MFIT_RE = re.compile(r"^mfit\.(it\d+)\.([0-9]+(?:-[0-9]+){3})\.(\d+)\.dat$")
SCAN_RE = re.compile(r"^(.+)_([0-9]+(?:-[0-9]+){3})$")


def _read_profile(path):
    import numpy as np

    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 3:
        raise ValueError(f"Expected 3 columns in {path}, got {data.shape[1]}")
    order = np.argsort(data[:, 0])
    ang = data[order, 0]
    hl = data[order, 1]
    ll = data[order, 2]
    return ang, hl, ll


def _read_scan_profile(path):
    import numpy as np

    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got {data.shape[1]}")
    order = np.argsort(data[:, 0])
    ang = data[order, 0]
    ene = data[order, 1]
    return ang, ene


def _iteration_number(name):
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits == "":
        return 0
    return int(digits)


def discover_profiles(input_dir, recursive=False):
    search = Path(input_dir).rglob if recursive else Path(input_dir).glob
    picked = {}

    for path in search("mfit.it*.*.*.dat"):
        m = MFIT_RE.match(path.name)
        if not m:
            continue
        it_name, dihedral, inner_it = m.group(1), m.group(2), int(m.group(3))
        key = (dihedral, it_name)
        prev = picked.get(key)
        if prev is None or inner_it > prev[0]:
            picked[key] = (inner_it, path)

    by_dihedral = {}
    for (dihedral, it_name), (_, path) in picked.items():
        by_dihedral.setdefault(dihedral, []).append((it_name, path))

    for dihedral in by_dihedral:
        by_dihedral[dihedral].sort(key=lambda x: _iteration_number(x[0]))
    return by_dihedral


def discover_scan_profiles(input_dir, recursive=False, reference_prefix=None, include_orig=True, prefer_normed=True):
    search = Path(input_dir).rglob if recursive else Path(input_dir).glob
    data = {}

    for path in search("*.dat"):
        name = path.name
        if name.startswith("mfit."):
            continue

        if name.endswith("_normed.dat"):
            stem = name[: -len("_normed.dat")]
            normed = True
        elif name.endswith(".dat"):
            stem = name[: -len(".dat")]
            normed = False
        else:
            continue

        m = SCAN_RE.match(stem)
        if not m:
            continue
        prefix, dihedral = m.group(1), m.group(2)
        if prefix == "":
            continue
        data.setdefault(dihedral, {}).setdefault(prefix, {})[normed] = path

    if not data:
        return {}

    def _pick_file(paths_by_normed):
        if prefer_normed and True in paths_by_normed:
            return paths_by_normed[True]
        if False in paths_by_normed:
            return paths_by_normed[False]
        return paths_by_normed[True]

    out = {}
    for dihedral, prefixes in data.items():
        iter_names = sorted([p for p in prefixes if re.fullmatch(r"it\d+", p)], key=_iteration_number)
        if not iter_names:
            continue

        if reference_prefix is not None:
            if reference_prefix not in prefixes:
                raise KeyError(f"Reference prefix '{reference_prefix}' not found for dihedral {dihedral}")
            ref_name = reference_prefix
        else:
            ref_candidates = [p for p in prefixes if p != "orig" and not re.fullmatch(r"it\d+", p)]
            if len(ref_candidates) == 0:
                raise KeyError(
                    f"No high-level reference scan found for dihedral {dihedral}. "
                    "Use --reference-prefix to select one explicitly."
                )
            if len(ref_candidates) > 1:
                raise KeyError(
                    f"Multiple reference prefixes found for dihedral {dihedral}: {', '.join(sorted(ref_candidates))}. "
                    "Use --reference-prefix."
                )
            ref_name = ref_candidates[0]

        ref_path = _pick_file(prefixes[ref_name])
        profiles = []
        if include_orig and "orig" in prefixes:
            profiles.append(("orig", _pick_file(prefixes["orig"])))
        for itn in iter_names:
            profiles.append((itn, _pick_file(prefixes[itn])))

        out[dihedral] = {"reference": (ref_name, ref_path), "profiles": profiles}
    return out


def _stats(hl, ll):
    import numpy as np

    diff = ll - hl
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae


def _align_to_reference(ref_angles, angles, vals, tol=1e-2):
    import numpy as np

    out = np.full(len(ref_angles), np.nan, dtype=float)
    for i, a in enumerate(ref_angles):
        j = int(np.argmin(np.abs(angles - a)))
        if abs(float(angles[j] - a)) <= tol:
            out[i] = vals[j]
    return out


def _make_output_path(output, dihedral, multiple):
    out = Path(output)
    if not multiple:
        return out
    return out.with_name(f"{out.stem}_{dihedral}{out.suffix}")


def _write_animation(dihedral, iterations, angles, hl_ref, ll_profiles, rmse_series, mae_series, output_file, fps=2, dpi=150, title_prefix="Twist Workflow Convergence"):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation

    ll_profiles = np.array(ll_profiles, dtype=float)
    rmse_series = np.array(rmse_series, dtype=float)
    mae_series = np.array(mae_series, dtype=float)
    finite_ll = ll_profiles[np.isfinite(ll_profiles)]
    if finite_ll.size == 0:
        raise ValueError(f"No valid LL values to plot for dihedral {dihedral}")
    ymin = float(min(np.min(hl_ref), np.min(finite_ll)))
    ymax = float(max(np.max(hl_ref), np.max(finite_ll)))
    pad = 0.08 * max(1.0, ymax - ymin)
    ymin -= pad
    ymax += pad

    fig, (ax_prof, ax_err) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    def _draw(frame):
        ax_prof.clear()
        ax_err.clear()

        ll = ll_profiles[frame, :]
        rmse = rmse_series[frame]
        mae = mae_series[frame]
        it = iterations[frame]
        known = np.isfinite(ll)

        ax_prof.plot(angles, hl_ref, color="#1d3557", linewidth=2.2, label="Reference (HL)")
        ax_prof.plot(angles[known], ll[known], color="#e63946", linewidth=2.2, marker="o", markersize=4, label="Current fit (LL)")
        ax_prof.set_xlim(float(np.min(angles)), float(np.max(angles)))
        ax_prof.set_ylim(ymin, ymax)
        ax_prof.set_xlabel("Dihedral angle (deg)")
        ax_prof.set_ylabel("Relative energy (kcal/mol)")
        ax_prof.grid(alpha=0.25)
        ax_prof.legend(loc="upper right")
        ax_prof.set_title(
            f"{title_prefix} | dihedral {dihedral} | {it} ({frame + 1}/{len(iterations)})"
        )
        ax_prof.text(
            0.02,
            0.96,
            f"RMSE: {rmse:.3f} kcal/mol\nMAE: {mae:.3f} kcal/mol",
            transform=ax_prof.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#999999"},
        )

        x = np.arange(len(iterations))
        ax_err.plot(x[: frame + 1], rmse_series[: frame + 1], color="#457b9d", marker="o", label="RMSE")
        ax_err.plot(x[: frame + 1], mae_series[: frame + 1], color="#2a9d8f", marker="s", label="MAE")
        ax_err.set_xlim(-0.3, len(iterations) - 0.7 if len(iterations) > 1 else 0.7)
        yerr_max = max(np.max(rmse_series), np.max(mae_series), 0.1)
        ax_err.set_ylim(0.0, yerr_max * 1.15)
        ax_err.set_xticks(x)
        ax_err.set_xticklabels(iterations, rotation=0)
        ax_err.set_xlabel("Twist workflow iteration")
        ax_err.set_ylabel("Error (kcal/mol)")
        ax_err.grid(alpha=0.25)
        ax_err.legend(loc="upper right")

    ani = animation.FuncAnimation(
        fig,
        _draw,
        frames=len(iterations),
        interval=max(100, int(1000 / max(1, fps))),
        repeat=False,
    )

    out = Path(output_file)
    suffix = out.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix in (".mp4", ".m4v"):
        writer = animation.FFMpegWriter(fps=fps, codec="h264")
    else:
        raise ValueError("Output extension must be .gif, .mp4, or .m4v")

    ani.save(str(out), writer=writer, dpi=dpi)
    plt.close(fig)
    return out


def animate_dihedral_from_mfit(dihedral, it_paths, output_file, fps=2, dpi=150, title_prefix="Twist Workflow Convergence"):
    import numpy as np

    iterations = []
    angles = None
    hl_ref = None
    ll_profiles = []
    rmse_series = []
    mae_series = []

    for it_name, path in it_paths:
        ang, hl, ll = _read_profile(path)
        if angles is None:
            angles = ang
            hl_ref = hl
            ll_aligned = ll
        else:
            ll_aligned = _align_to_reference(angles, ang, ll, tol=1e-2)
        mask = ~np.isnan(ll_aligned)
        if not mask.any():
            continue
        rmse, mae = _stats(hl_ref[mask], ll_aligned[mask])
        iterations.append(it_name)
        ll_profiles.append(ll_aligned)
        rmse_series.append(rmse)
        mae_series.append(mae)

    if len(iterations) == 0:
        raise ValueError(f"No iteration profiles found for dihedral {dihedral}")

    return _write_animation(
        dihedral=dihedral,
        iterations=iterations,
        angles=angles,
        hl_ref=hl_ref,
        ll_profiles=ll_profiles,
        rmse_series=rmse_series,
        mae_series=mae_series,
        output_file=output_file,
        fps=fps,
        dpi=dpi,
        title_prefix=title_prefix,
    )


def animate_dihedral_from_scan(dihedral, ref_path, profile_paths, output_file, fps=2, dpi=150, title_prefix="Twist Workflow Convergence"):
    import numpy as np

    ref_angles, ref_ene = _read_scan_profile(ref_path)

    iterations = []
    ll_profiles = []
    rmse_series = []
    mae_series = []
    for label, path in profile_paths:
        ang, ene = _read_scan_profile(path)
        ll = _align_to_reference(ref_angles, ang, ene, tol=1e-2)
        mask = np.isfinite(ll)
        if not mask.any():
            continue
        rmse, mae = _stats(ref_ene[mask], ll[mask])
        iterations.append(label)
        ll_profiles.append(ll)
        rmse_series.append(rmse)
        mae_series.append(mae)

    if len(iterations) == 0:
        raise ValueError(f"No scan iterations were usable for dihedral {dihedral}")

    return _write_animation(
        dihedral=dihedral,
        iterations=iterations,
        angles=ref_angles,
        hl_ref=ref_ene,
        ll_profiles=ll_profiles,
        rmse_series=rmse_series,
        mae_series=mae_series,
        output_file=output_file,
        fps=fps,
        dpi=dpi,
        title_prefix=title_prefix,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create animated convergence plots from ffpopt-DihedTwistWorkflow outputs. "
            "By default, the tool reads scan .dat outputs (<prefix>_<i-j-k-l>.dat) from "
            "the twist workflow and animates orig/itXX progression relative to the high-level "
            "reference scan. You can also use --source mfit."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default=".",
        help="Directory containing twist workflow .dat files. Default: current directory.",
    )
    parser.add_argument(
        "-d",
        "--dihedral",
        action="append",
        default=None,
        help=(
            "Specific dihedral index quartet to animate (format: i-j-k-l). "
            "May be provided multiple times. Default: animate all discovered quartets."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="twist_convergence.gif",
        help=(
            "Output animation path. If multiple dihedrals are requested or discovered, "
            "the dihedral label is appended to the stem."
        ),
    )
    parser.add_argument("--fps", type=int, default=2, help="Animation frame rate. Default: 2")
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution in DPI. Default: 150")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories under --input-dir.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Twist Workflow Convergence",
        help="Title prefix to use in each animation frame.",
    )
    parser.add_argument(
        "--source",
        choices=["scan", "mfit"],
        default="scan",
        help="Data source: 'scan' uses workflow scan .dat files; 'mfit' uses mfit.*.dat files. Default: scan",
    )
    parser.add_argument(
        "--reference-prefix",
        default=None,
        help="For --source scan, explicit high-level reference prefix (for example 'mace-off23_medium').",
    )
    parser.add_argument(
        "--no-orig",
        action="store_true",
        help="For --source scan, omit the 'orig' baseline frame.",
    )
    parser.add_argument(
        "--no-normed",
        action="store_true",
        help="For --source scan, ignore *_normed.dat even if available.",
    )
    args = parser.parse_args()

    if args.source == "mfit":
        by_dihedral = discover_profiles(args.input_dir, recursive=args.recursive)
    else:
        by_dihedral = discover_scan_profiles(
            args.input_dir,
            recursive=args.recursive,
            reference_prefix=args.reference_prefix,
            include_orig=not args.no_orig,
            prefer_normed=not args.no_normed,
        )
    if not by_dihedral:
        if args.source == "mfit":
            raise FileNotFoundError(
                f"No mfit.itXX.<i-j-k-l>.<inner>.dat files were found in {Path(args.input_dir).resolve()}"
            )
        raise FileNotFoundError(
            f"No scan files matching <prefix>_<i-j-k-l>.dat were found in {Path(args.input_dir).resolve()}"
        )

    if args.dihedral is None or len(args.dihedral) == 0:
        selected = sorted(by_dihedral.keys())
    else:
        selected = []
        for d in args.dihedral:
            if d not in by_dihedral:
                raise KeyError(f"Requested dihedral {d} not found in discovered files.")
            selected.append(d)

    multiple = len(selected) > 1
    for d in selected:
        outfile = _make_output_path(args.output, d, multiple)
        if args.source == "mfit":
            out = animate_dihedral_from_mfit(
                d,
                by_dihedral[d],
                output_file=outfile,
                fps=args.fps,
                dpi=args.dpi,
                title_prefix=args.title_prefix,
            )
        else:
            ref_name, ref_path = by_dihedral[d]["reference"]
            out = animate_dihedral_from_scan(
                d,
                ref_path=ref_path,
                profile_paths=by_dihedral[d]["profiles"],
                output_file=outfile,
                fps=args.fps,
                dpi=args.dpi,
                title_prefix=f"{args.title_prefix} | ref={ref_name}",
            )
        print(f"Wrote animation to {out}")


if __name__ == "__main__":
    main()
