"""QC figure for one sample point: cross-section with width picks plus a hillshade inset."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .tiles import Tile  # noqa: E402
from .transect import Transect  # noqa: E402
from .widths import stage_label  # noqa: E402

# categorical slots (light surface) from the dataviz reference palette
C_PROFILE = "#2a78d6"   # blue: the cross-section
C_BREAK = "#eb6834"     # orange: inner-channel (bank-break) picks
C_VALLEY = "#4a3aa7"    # violet: valley-floor span
C_TEXT = "#52514e"
C_FIXED = "#9a9995"


def hillshade(z: np.ndarray, dx: float, dy: float, azimuth_deg: float, altitude_deg: float) -> np.ndarray:
    gy, gx = np.gradient(z, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def plot_point(tile: Tile, point: dict, tr: Transect, m: dict, out_png: str, cfg: dict) -> None:
    s, z = tr.s, tr.z
    zrel = z - m["z_thalweg"]
    fig = plt.figure(figsize=(11, 4.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])

    ax.plot(s, zrel, color=C_PROFILE, lw=1.6, label="cross-section (bilinear, filled DEM)")
    ax.plot(m["s_thalweg"], 0.0, marker="v", color=C_PROFILE, ms=8, ls="none", label="thalweg")

    for hf in cfg["fixed_stages_m"]:
        k = f"W_stage_{stage_label(hf)}"
        sl, sr, w = m[k + "_sL"], m[k + "_sR"], m[k]
        ax.hlines(hf, sl, sr, color=C_FIXED, lw=1.0, ls=(0, (3, 2)))
        ax.text(sr, hf, f"  stage {hf:g} m: {w:.2f} m", va="center", ha="left", fontsize=7.5, color=C_TEXT)

    if np.isfinite(m["s_bank_left"]) and np.isfinite(m["s_bank_right"]):
        sb = np.array([m["s_bank_left"], m["s_bank_right"]])
        zb = np.interp(sb, s, zrel)
        ax.plot(sb, zb, marker="o", mfc="white", mec=C_BREAK, mew=2, ms=8, ls="none",
                label=f"bank breaks: inner-channel W={m['W_inner_bankbreak']:.2f} m, relief {m['inner_relief_m']:.2f} m")
        ax.hlines(min(zb), sb[0], sb[1], color=C_BREAK, lw=2.2)
    yv = -0.12 * max(0.5, np.nanmax(zrel[np.abs(s) <= 10]))
    ax.hlines(yv, m["s_valley_left"], m["s_valley_right"], color=C_VALLEY, lw=2.0,
              label=f"valley floor (slope<{cfg['valley_slope_threshold']:g}): W={m['W_valley_floor']:.2f} m")

    ax.axhline(0, color="#d9d8d4", lw=0.8, zorder=0)
    ax.set_xlabel("distance along transect (m), + is left bank looking downstream")
    ax.set_ylabel("height above thalweg (m)")
    ax.set_xlim(s[0], s[-1])
    ymax = np.nanmax(zrel[np.abs(s) <= 25])
    ax.set_ylim(min(yv * 1.6, -0.3), max(2.5, min(ymax, 12.0)) * 1.05)
    ax.grid(color="#efeeea", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", fontsize=7.5, frameon=False, ncol=2)

    prim = (f"primary W = {m['W_primary']:.2f} m ({m['W_primary_source']})" if np.isfinite(m["W_primary"])
            else "primary W unresolved")
    cens = f"CENSORED (bound {m['censor_bound']:.2f} m = 2 pixels)" if m["censored"] else "not censored"
    ax.set_title(
        f"{point['site']} tile {point['tile']}  id={point['point_id']}   A = {point['area_m2']:.3g} m²   "
        f"S = {point['local_slope']:.4f}   channelized = {m['channelized']}   shape = {m['shape_class']}\n"
        f"{prim}; {cens}   flags: {m['qc_flags'] or 'none'}",
        fontsize=8.5, loc="left", color="#0b0b0b")

    half = cfg["inset_half_size_m"]
    r0 = int(max(0, point["row"] - half / tile.dy))
    r1 = int(min(tile.shape[0], point["row"] + half / tile.dy + 1))
    c0 = int(max(0, point["col"] - half / tile.dx))
    c1 = int(min(tile.shape[1], point["col"] + half / tile.dx + 1))
    zw = tile.z[r0:r1, c0:c1]
    hs = hillshade(zw, tile.dx, tile.dy, cfg["hillshade_azimuth_deg"], cfg["hillshade_altitude_deg"])
    xl = tile.x0 + c0 * tile.dx
    xr = tile.x0 + c1 * tile.dx
    yt = tile.y0 - r0 * tile.dy
    yb = tile.y0 - r1 * tile.dy
    axm.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=(xl, xr, yb, yt), interpolation="nearest")
    chan = tile.area[r0:r1, c0:c1] >= cfg["a_min_sample_m2"]
    rr, cc = np.nonzero(chan)
    cx, cy = tile.rc_to_xy(rr + r0, cc + c0)
    axm.plot(cx, cy, ".", color=C_PROFILE, ms=2, alpha=0.35, label=f"A ≥ {cfg['a_min_sample_m2']:g} m²")
    axm.plot(tr.trace_xy[:, 0], tr.trace_xy[:, 1], "-", color=C_PROFILE, lw=1.5, label="trace (±10 m)")
    axm.plot(tr.x, tr.y, "-", color=C_BREAK, lw=1.4, label="transect (±25 m)")
    axm.plot(point["x"], point["y"], marker="o", mfc="white", mec="#0b0b0b", ms=6, ls="none")
    axm.set_xlim(xl, xr)
    axm.set_ylim(yb, yt)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    axm.set_title(f"hillshade, {2 * half:.0f} m window; azimuth {tr.azimuth_deg:.0f}°", fontsize=8, color=C_TEXT)
    axm.legend(loc="lower left", fontsize=6.5, frameon=True, framealpha=0.85)

    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
