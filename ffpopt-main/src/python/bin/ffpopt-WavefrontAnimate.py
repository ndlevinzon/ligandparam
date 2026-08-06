#!/usr/bin/env python3

import argparse
from pathlib import Path

def _build_profile_line(profile_map, angles, reference_min):
    import numpy as np

    vals = np.full(len(angles), np.nan, dtype=float)
    for i, ang in enumerate(angles):
        if ang in profile_map:
            vals[i] = profile_map[ang]
    if np.all(np.isnan(vals)):
        return vals
    return vals - reference_min


def _build_level_grid(levels, delta, upto_level, total_levels):
    import numpy as np

    n_levels = max(total_levels, 1)
    n_angles = int(360 // delta)
    grid = np.zeros((n_levels, n_angles), dtype=int)
    counts = np.zeros((n_levels, n_angles), dtype=int)

    for i in range(min(upto_level, len(levels), n_levels)):
        level = levels[i]
        for node in level.nodes:
            idx = int(round(node.angle / delta)) % n_angles
            counts[i, idx] += 1
            grid[i, idx] = 1 if node.active else 2
    return grid, counts


def animate_wavefront(
    wf,
    output_file,
    fps=2,
    dpi=150,
):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import ListedColormap

    if not wf.level_energies:
        raise ValueError("No level convergence history found in pickle (wavefront.level_energies is empty).")

    delta = int(wf.delta)
    if delta <= 0 or (360 % delta) != 0:
        raise ValueError(f"Unsupported wavefront delta: {delta}. Expected a positive divisor of 360.")

    angles = np.arange(0, 360, delta, dtype=int)
    frames = len(wf.level_energies)
    total_levels = len(wf.levels)
    from ffpopt.constants import AU_PER_KCAL_PER_MOL, AU_PER_ELECTRON_VOLT
    kcal_per_au = AU_PER_ELECTRON_VOLT() / AU_PER_KCAL_PER_MOL()
    cmap = ListedColormap(["white", "#ff9f1c", "#3a86ff"])

    final_profile_map = wf.level_energies[-1]
    final_values = np.array(
        [final_profile_map[ang] for ang in angles if ang in final_profile_map],
        dtype=float,
    )
    if final_values.size == 0:
        raise ValueError("Final wavefront level has no energies; cannot determine reference minimum.")
    final_min = float(np.nanmin(final_values))

    all_profiles = np.full((frames, len(angles)), np.nan, dtype=float)
    for i, profile_map in enumerate(wf.level_energies):
        all_profiles[i, :] = _build_profile_line(profile_map, angles, final_min) * kcal_per_au
    finite = np.isfinite(all_profiles)
    if np.any(finite):
        global_ymin = float(np.nanmin(all_profiles))
        global_ymax = float(np.nanmax(all_profiles))
    else:
        global_ymin, global_ymax = 0.0, 1.0
    ymin = min(-0.1, global_ymin - 0.05 * max(1.0, abs(global_ymax - global_ymin)))
    ymax = max(1.0, global_ymax * 1.1 if global_ymax > 0 else global_ymax + 1.0)

    fig, (ax_profile, ax_grid) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    def _draw(frame_idx):
        ax_profile.clear()
        ax_grid.clear()

        profile = all_profiles[frame_idx, :]
        known = ~np.isnan(profile)
        ax_profile.plot(angles, profile, color="#1d3557", linewidth=2)
        ax_profile.scatter(angles[known], profile[known], color="#e63946", s=35, zorder=3)
        ax_profile.set_xlim(0, 360 - delta)
        ax_profile.set_ylim(ymin, ymax)
        ax_profile.set_xticks(np.arange(0, 361, max(delta, 30)))
        ax_profile.set_xlabel("Dihedral angle (deg)")
        ax_profile.set_ylabel("Relative energy (kcal/mol)")
        ax_profile.grid(alpha=0.25)
        ax_profile.set_title(
            f"Wavefront Convergence: level {frame_idx + 1}/{frames} | known angles {int(np.sum(known))}/{len(angles)}"
        )

        grid, counts = _build_level_grid(wf.levels, delta=delta, upto_level=frame_idx + 1, total_levels=total_levels)
        ax_grid.imshow(grid, aspect="auto", origin="upper", cmap=cmap, vmin=0, vmax=2)
        ax_grid.set_yticks(np.arange(grid.shape[0]))
        ax_grid.set_yticklabels(np.arange(1, grid.shape[0] + 1))
        ax_grid.set_ylabel("Wavefront level")
        ax_grid.set_xticks(np.arange(len(angles)))
        ax_grid.set_xticklabels(angles, rotation=90)
        ax_grid.set_xlabel("Angle bin (deg)")

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if counts[i, j] > 0:
                    ax_grid.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=7)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor="#ff9f1c", edgecolor="black", label="Active node"),
            Patch(facecolor="#3a86ff", edgecolor="black", label="Inactive node"),
        ]
        ax_grid.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ani = animation.FuncAnimation(fig, _draw, frames=frames, interval=max(100, int(1000 / max(1, fps))), repeat=False)

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


def main():
    parser = argparse.ArgumentParser(
        description="Create an animation showing wavefront dihedral scan convergence from a wavefront pickle.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to Wavefront pickle file (for example, scan.pkl).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output animation path (.gif, .mp4, or .m4v). Default: <input>_convergence.gif",
    )
    parser.add_argument("--fps", type=int, default=2, help="Animation frame rate. Default: 2")
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution in DPI. Default: 150")
    args = parser.parse_args()

    infile = Path(args.input)
    if not infile.exists():
        raise FileNotFoundError(f"Input file does not exist: {infile}")

    if args.output is None:
        outfile = infile.with_name(f"{infile.stem}_convergence.gif")
    else:
        outfile = Path(args.output)

    from ffpopt.WaveFront import wavefront_loader
    wf = wavefront_loader(str(infile))
    animate_wavefront(wf, output_file=str(outfile), fps=args.fps, dpi=args.dpi)
    print(f"Wrote animation to {outfile}")


if __name__ == "__main__":
    main()
