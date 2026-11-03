"""QC frame for the constrained re-run: five along-stream transects overlaid,
median profile, relief window, offset limits, bank picks, class and flags,
hillshade inset."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .constrained import MultiTransect  # noqa: E402
from .qc import hillshade  # noqa: E402
from .tiles import Tile  # noqa: E402

C_PROFILE = "#2a78d6"
C_BANK = "#eb6834"
C_BEYOND = "#9a9995"
C_TEXT = "#52514e"
C_BAND = "#1baf7a"


def plot_point_constrained(tile: Tile, point: dict, mt: MultiTransect, per: list, agg: dict,
                           out_png: str, cfg: dict, ccfg: dict) -> None:
    fig = plt.figure(figsize=(11.5, 4.8), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])
    grid = np.arange(-cfg["transect_half_length_m"], cfg["transect_half_length_m"] + 1e-9, cfg["transect_step_m"])
    stack = []
    for off, tr, m in zip(mt.offsets, mt.transects, per):
        if tr is None or m is None:
            continue
        s_rel = tr.s - m["thalweg_s"]
        z_rel = tr.z - m["z_thalweg"]
        ax.plot(s_rel, z_rel, color=C_PROFILE, lw=0.7, alpha=0.45)
        stack.append(np.interp(grid, s_rel, z_rel, left=np.nan, right=np.nan))
        for side in ("left", "right"):
            sk, rel, cl = m[f"s_bank_{side}"], m[f"relief_{side}_m"], m[f"{side}_class"]
            if np.isfinite(sk):
                if cl == "bank":
                    ax.plot(sk - m["thalweg_s"], rel, marker="o", mfc="white", mec=C_BANK, mew=1.6, ms=6, ls="none")
                elif cl == "wall":
                    ax.plot(sk - m["thalweg_s"], min(rel, 9.5), marker="|", color=C_BEYOND, ms=10, mew=1.5, ls="none")
                else:
                    ax.plot(sk - m["thalweg_s"], rel, marker="s", mfc="none", mec=C_BEYOND, mew=1.2, ms=6, ls="none")
    if stack:
        med = np.nanmedian(np.vstack(stack), axis=0)
        ax.plot(grid, med, color=C_PROFILE, lw=2.4, label=f"median of {len(stack)} transects (thin: each transect)")
    ax.axhspan(ccfg["bank_min_relief_m"], ccfg["bank_max_relief_m"], color=C_BAND, alpha=0.12, lw=0,
               label=f"bank relief window {ccfg['bank_min_relief_m']:g}-{ccfg['bank_max_relief_m']:g} m")
    for sgn in (-1, 1):
        ax.axvline(sgn * ccfg["bank_max_offset_m"], color=C_BAND, lw=1.0, ls="--")
    ax.plot([], [], marker="o", mfc="white", mec=C_BANK, mew=1.6, ms=6, ls="none", label="accepted bank pick (per transect)")
    ax.plot([], [], marker="s", mfc="none", mec=C_BEYOND, ms=6, ls="none", label="break in relief window but beyond the offset limit")
    ax.plot([], [], marker="|", color=C_BEYOND, ms=10, mew=1.5, ls="none", label="wall: first break above the relief ceiling (clipped)")
    ax.axhline(0, color="#d9d8d4", lw=0.8, zorder=0)
    ax.set_xlim(-cfg["transect_half_length_m"], cfg["transect_half_length_m"])
    ymax = np.nanmax(np.vstack(stack)) if stack else 3
    ax.set_ylim(-0.4, max(2.5, min(float(ymax), 10.0)) * 1.05)
    ax.set_xlabel("distance from thalweg (m), + is left bank looking downstream")
    ax.set_ylabel("height above thalweg (m)")
    ax.grid(color="#efeeea", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
    wb = f"W_trench = {agg['W_trench']:.2f} m (n={agg['W_trench_n_valid']}, spread {agg['W_trench_spread']:.2f}), depth {agg['trench_depth_m']:.2f} m" if np.isfinite(agg.get("W_trench", np.nan)) else "W_trench: none"
    ws = f"W_bed = {agg['W_bed']:.2f} m" + (f", ratio trench/stage {agg['ratio_trench_over_stage']:.2f}" if np.isfinite(agg.get("ratio_trench_over_stage", np.nan)) else "")
    ax.set_title(f"{point['site']} tile {point['tile']}  id={point['point_id']}   A = {point['area_m2']:.3g} m²   S = {point['local_slope']:.4f}\n"
                 f"class = {agg['point_class']}  [{agg.get('class_counts', '')}]   {wb}   {ws}\nflags: {agg.get('qc_flags') or 'none'}",
                 fontsize=8.3, loc="left", color="#0b0b0b")

    half = cfg["inset_half_size_m"]
    r0 = int(max(0, point["row"] - half / tile.dy)); r1 = int(min(tile.shape[0], point["row"] + half / tile.dy + 1))
    c0 = int(max(0, point["col"] - half / tile.dx)); c1 = int(min(tile.shape[1], point["col"] + half / tile.dx + 1))
    hs = hillshade(tile.z[r0:r1, c0:c1], tile.dx, tile.dy, cfg["hillshade_azimuth_deg"], cfg["hillshade_altitude_deg"])
    xl, xr = tile.x0 + c0 * tile.dx, tile.x0 + c1 * tile.dx
    yt, yb = tile.y0 - r0 * tile.dy, tile.y0 - r1 * tile.dy
    axm.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=(xl, xr, yb, yt), interpolation="nearest")
    chan = tile.area[r0:r1, c0:c1] >= ccfg["a_min_sample_m2"]
    rr, cc = np.nonzero(chan)
    cx, cy = tile.rc_to_xy(rr + r0, cc + c0)
    axm.plot(cx, cy, ".", color=C_PROFILE, ms=2, alpha=0.35, label=f"A ≥ {ccfg['a_min_sample_m2']:g} m²")
    for tr in mt.transects:
        if tr is not None:
            axm.plot(tr.x, tr.y, "-", color=C_BANK, lw=1.0, alpha=0.8)
    axm.plot(point["x"], point["y"], marker="o", mfc="white", mec="#0b0b0b", ms=6, ls="none")
    axm.set_xlim(xl, xr); axm.set_ylim(yb, yt); axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])
    axm.set_title(f"hillshade, {2 * half:.0f} m window; transects at {ccfg['transect_offsets_m']} m", fontsize=7.5, color=C_TEXT)
    axm.legend(loc="lower left", fontsize=6.5, frameon=True, framealpha=0.85)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
