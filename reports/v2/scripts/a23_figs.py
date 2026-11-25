#!/usr/bin/env python3
"""Addendum 23 figures (directive section 5): little text on the plot, numbers in captions and tables, full plain names in legends,
1 m hillshades (never a 12 m block mean), roads and trails as thin grey lines, the M&D study area and the domain outlined, scale bar
and north arrow, no undefined shorthand. Figures 1-8 of section 5.5 in reports/v2/figures/a23/; captions to <run>/post/captions.json.
usage: a23_figs.py [--only N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a23_common import (  # noqa: E402
    BLOCK_M,
    CANDIDATE_PLAIN,
    CANDIDATES,
    COLOUR,
    D_PHYS,
    FIG_DIR,
    HA_SHARE_FOUND_MIN,
    HA_SHARE_REAL_MIN,
    MAP_ORDER,
    MARKER,
    MODELS,
    PLAIN,
    RATIO_BAND,
    READING_BAND_M,
    T_PRIMARY_M,
    TOLERANCES_M,
    VALLEY_SHARE_RANGE,
    dump,
    hillshade,
    load_domain,
    load_json,
    out_path,
)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.titlesize": 12})
CAPTIONS: dict[str, dict[str, str]] = {}
CAND_COLOUR = {"mfd": "#1b9e77", "d8s3": "#d95f02", "d8s6": "#7570b3", "ref12": "#000000"}
CAND_MARKER = {"mfd": "o", "d8s3": "s", "d8s6": "D", "ref12": "*"}
ROAD_GREY = "0.35"
MAPPED_COLOUR = "#e6550d"
TP = f"{T_PRIMARY_M:g}"


def save(fig, name: str, how: str, what: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.text(0.01, -0.01, f"How to read: {how}\nWhat it shows: {what}", ha="left", va="top", fontsize=8.5, wrap=True, transform=fig.transFigure)
    path = os.path.join(FIG_DIR, name)
    for sc in (1.0, 0.8, 0.6, 0.45):
        fig.savefig(path, dpi=int(160 * sc), bbox_inches="tight", facecolor="white")
        if os.path.getsize(path) <= 5_000_000:
            break
    plt.close(fig)
    CAPTIONS[name] = {"how": how, "what": what}
    return path


def handle(n: str, **kw) -> Line2D:
    return Line2D([], [], color=COLOUR[n], marker=MARKER[n], linestyle=kw.get("ls", "none"), markersize=kw.get("ms", 7), label=PLAIN[n])


def map_decor(ax, dom: dict[str, Any], roads: bool = True, outline_domain: bool = True, outline_poly: bool = True, scale_m: float = 500.0) -> None:
    ny, nx = dom["shape"]
    if roads:
        rc = dom["road_cells"] if "road_cells" in dom else (dom["road_dist"] <= 0.5)
        rr, cc = np.nonzero(rc)
        ax.plot(cc, rr, ",", color=ROAD_GREY, ms=1, zorder=4)
    if outline_poly:
        ax.contour(dom["poly"].astype(float), levels=[0.5], colors="k", linewidths=1.1, linestyles="--", zorder=5)
    if outline_domain:
        ax.contour(dom["domain"].astype(float), levels=[0.5], colors="#1f78b4", linewidths=1.1, zorder=5)
    x0, y0 = nx * 0.04, ny * 0.96
    ax.plot([x0, x0 + scale_m], [y0, y0], color="k", lw=3, solid_capstyle="butt", zorder=6)
    ax.plot([x0, x0 + scale_m], [y0, y0], color="w", lw=1, solid_capstyle="butt", zorder=7)
    ax.text(x0 + scale_m / 2, y0 - ny * 0.012, f"{scale_m:.0f} m", ha="center", va="bottom", fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1}, zorder=8)
    xa, ya = nx * 0.95, ny * 0.13
    ax.annotate("", xy=(xa, ya - ny * 0.08), xytext=(xa, ya), arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 1.5}, zorder=8)
    ax.text(xa, ya - ny * 0.09, "N", ha="center", va="bottom", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1}, zorder=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)


def hs_image(Z: np.ndarray) -> np.ndarray:
    return np.nan_to_num(hillshade(Z, 1.0), nan=1.0)


MAP_LEGEND = [Line2D([], [], color=ROAD_GREY, lw=1, label="roads and trails (TIGER)"), Line2D([], [], color="k", ls="--", label="M&D study area"),
              Line2D([], [], color="#1f78b4", label="domain (the basins above the outlets)")]


# ----------------------------------------------------------------------------- figure 1: the domain

def fig1(dom: dict[str, Any]) -> None:
    rec = dom["rec"]
    fig, ax = plt.subplots(figsize=(12, 8.2))
    ax.imshow(hs_image(dom["Z_lidar_box"]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.where(dom["domain"], 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#1f78b4"]), alpha=0.12, interpolation="nearest")
    orr, occ = np.nonzero(dom["outlets"])
    ax.plot(occ, orr, marker="o", color="#e31a1c", ms=7, markeredgecolor="k", ls="none", zorder=6)
    pk = dom["picks"]
    ax.plot(pk["col"], pk["row"], marker="^", color=MAPPED_COLOUR, ms=5, markeredgecolor="k", markeredgewidth=0.4, ls="none", zorder=6)
    map_decor(ax, dom)
    ax.set_title("The domain: the drainage basins above the outlets of the M&D study area, on the 1 m lidar hillshade (lit from the north-west)")
    leg = MAP_LEGEND + [Line2D([], [], marker="o", color="#e31a1c", markeredgecolor="k", ls="none", ms=7, label="outlet (held at its observed elevation)"),
                        Line2D([], [], marker="^", color=MAPPED_COLOUR, markeredgecolor="k", ls="none", ms=6, label="M&D channel heads (Steve's picks)")]
    ax.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.9)
    d, o = rec["domain"], rec["outlets"]
    save(fig, "fig1_domain.png",
         "Hillshade of the full-site 1 m lidar around the M&D study area (dashed). The blue outline and tint are the domain: every cell that drains to one of the red outlets, "
         "which are the cells where flow leaves the study area with drainage area of at least 10,000 m². Orange triangles are the mapped channel heads. Scale bar 500 m; north up.",
         f"Domain {d['area_km2']:.2f} km² ({d['n_cells']:,} cells), {o['n_fixed']} outlets held at their observed elevation ({o['n_nested']} nested exits inside the domain); the domain covers "
         f"{d['share_of_polygon_covered']:.2f} of the study area, {d['share_of_domain_outside_polygon']:.2f} of the domain lies outside it, and the hillslope fringe that drains out below "
         f"10,000 m² is {d['hillslope_fringe_share_of_polygon']:.2f} of the study area. {rec['picks']['n_in_domain_and_polygon']} of {rec['picks']['n']} picks lie in the domain and the study area.")


# ----------------------------------------------------------------------------- figure 2: the road check

def fig2(RD: dict[str, Any]) -> None:
    Rz = np.load(out_path("roads", "road_check.npz"))
    Z1 = Rz["Z1m"].astype(float)
    ny, nx = Z1.shape
    poly = np.unpackbits(Rz["poly"])[: ny * nx].reshape(ny, nx).astype(bool)
    trunk = np.unpackbits(Rz["trunk"])[: ny * nx].reshape(ny, nx).astype(bool)
    roads = np.unpackbits(Rz["road_cells"])[: ny * nx].reshape(ny, nx).astype(bool)
    wall, extra = Rz["wall_rc"], Rz["extra_rc"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4))
    for ax, Zs, title in ((axes[0], Z1, "Real landscape (lidar), 1 m hillshade"), (axes[1], None, "AD model surface (12 m block mean, the Addendum 22 fit)")):
        if Zs is None:
            Z12 = Rz["AD_Z12"].astype(float)
            ax.imshow(np.nan_to_num(hillshade(Z12, float(BLOCK_M)), nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=(-0.5, Z12.shape[1] * BLOCK_M - 0.5, Z12.shape[0] * BLOCK_M - 0.5, -0.5))
        else:
            ax.imshow(hs_image(Zs), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        rr, cc = np.nonzero(roads)
        ax.plot(cc, rr, ",", color=ROAD_GREY, ms=1, zorder=4)
        tr, tc = np.nonzero(trunk)
        ax.plot(tc, tr, ",", color="#1f78b4", ms=1, zorder=4)
        ax.contour(poly.astype(float), levels=[0.5], colors="k", linewidths=1.0, linestyles="--", zorder=5)
        other = np.array([q for q in extra.tolist() if q not in wall.tolist()]) if len(extra) else np.zeros((0, 2))
        if len(other):
            ax.plot(other[:, 1] * BLOCK_M + BLOCK_M // 2, other[:, 0] * BLOCK_M + BLOCK_M // 2, marker="o", markerfacecolor="none", markeredgecolor=COLOUR["AD"], ms=7, markeredgewidth=1.2, ls="none", zorder=6)
        if len(wall):
            ax.plot(wall[:, 1] * BLOCK_M + BLOCK_M // 2, wall[:, 0] * BLOCK_M + BLOCK_M // 2, marker="o", color=COLOUR["AD"], markeredgecolor="k", ms=7, ls="none", zorder=6)
        pr, pc = Rz["pick_rc"]
        ax.plot(pc, pr, marker="^", color=MAPPED_COLOUR, ms=6, markeredgecolor="k", markeredgewidth=0.4, ls="none", zorder=6)
        x0, y0 = nx * 0.04, ny * 0.96
        ax.plot([x0, x0 + 200], [y0, y0], color="k", lw=3, solid_capstyle="butt")
        ax.plot([x0, x0 + 200], [y0, y0], color="w", lw=1, solid_capstyle="butt")
        ax.text(x0 + 100, y0 - 14, "200 m", ha="center", va="bottom", fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1})
        ax.annotate("", xy=(nx * 0.95, ny * 0.05), xytext=(nx * 0.95, ny * 0.13), arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 1.5})
        ax.text(nx * 0.95, ny * 0.04, "N", ha="center", va="bottom", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1})
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(ny - 0.5, -0.5)
        ax.set_title(title)
    leg = [Line2D([], [], color=ROAD_GREY, lw=1, label="roads and trails (TIGER)"), Line2D([], [], color="#1f78b4", lw=1, label="observed trunk flow path (drainage area ≥ 100,000 m²)"),
           Line2D([], [], color="k", ls="--", label="M&D study area"), Line2D([], [], marker="o", color=COLOUR["AD"], markeredgecolor="k", ls="none", ms=7, label="extra AD heads on too-steep walls (heads found on a 12 m block mean of the AD surface)"),
           Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=COLOUR["AD"], ls="none", ms=7, label="other extra AD heads"), Line2D([], [], marker="^", color=MAPPED_COLOUR, markeredgecolor="k", ls="none", ms=6, label="M&D channel heads")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Is the AD model's line of heads on tile TTV2 a road? The Addendum 22 figure 7 redrawn on the 1 m lidar hillshade with the TIGER roads", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    w15, t5 = RD["wall_heads_near_road"]["15"], RD["trunk"]["near_road"]["5"]
    save(fig, "fig2_road_check.png",
         "Left: the 1 m lidar of tile TTV2 with the TIGER roads (grey), the observed trunk flow path (blue), the M&D study area (dashed) and the extra heads of the AD model from Addendum 22 "
         "(filled: on too-steep gorge walls; open: elsewhere). Right: the same symbols on the AD model's own surface. Heads were found on a 12 m block mean of the AD surface and are drawn at their cell centres.",
         f"{w15['n_within']} of {RD['n_wall_heads']} wall heads lie within 15 m of a TIGER road (share {w15['share']:.2f}); {t5['share']:.2f} of the trunk path lies within 5 m of a road. "
         f"TIGER holds {RD['roads']['n_features']} features on the tile and no trail-class feature. Reading 'road-influenced': {'yes' if RD['reading']['road_influenced'] else 'no'}.")


# ----------------------------------------------------------------------------- figure 3: the head-finder test

def fig3(H1: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for j, (cal, test) in enumerate((("TTV2", "TTV13"), ("TTV13", "TTV2"))):
        for i, key in enumerate(("share_found", "share_real")):
            ax = axes[i, j]
            for c in CANDIDATES:
                q = H1["candidates"].get(c)
                if q is None:
                    continue
                h = q["halves"][cal]
                ys = [h["by_t"][f"{t:g}"][key] if h["by_t"][f"{t:g}"][key] is not None else np.nan for t in TOLERANCES_M]
                ax.plot(TOLERANCES_M, ys, marker=CAND_MARKER[c], color=CAND_COLOUR[c], lw=1.4, ms=6)
            ax.axhline(HA_SHARE_FOUND_MIN if key == "share_found" else HA_SHARE_REAL_MIN, color="0.4", ls=":", lw=1)
            ax.axvline(T_PRIMARY_M, color="0.7", ls="--", lw=0.8)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("matching tolerance (m)")
            if j == 0:
                ax.set_ylabel("share of mapped heads found" if key == "share_found" else "share of found heads that are real")
            if i == 0:
                ax.set_title(f"calibrated on {'TTV2 ground' if cal == 'TTV2' else 'TTV1 + TTV3 ground'}, tested on {'TTV1 + TTV3 ground' if test == 'TTV13' else 'TTV2 ground'}")
    leg = [Line2D([], [], color=CAND_COLOUR[c], marker=CAND_MARKER[c], ls="-", label=CANDIDATE_PLAIN[c]) for c in CANDIDATES] + [Line2D([], [], color="0.4", ls=":", label="the reading's bar (0.6 found; 0.5 real)"),
                                                                                                                                   Line2D([], [], color="0.7", ls="--", label="25 m, the primary tolerance")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Can a head finder on the real 1 m lidar find the mapped heads? Each candidate's threshold is set on one half of the picks and tested on the other", y=0.995)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    pooled = "; ".join(f"{CANDIDATE_PLAIN[c]}: {H1['candidates'][c]['pooled'][TP]['share_found']:.2f} found, {H1['candidates'][c]['pooled'][TP]['share_real']:.2f} real" for c in CANDIDATES if c in H1["candidates"])
    save(fig, "fig3_head_finder_test.png",
         "Top row: the share of the test half's mapped heads that a candidate pairs one-to-one within the tolerance on the x axis. Bottom row: the share of the candidate's heads on that ground that are real "
         "(paired). Left column: threshold set on TTV2 ground (34 picks), tested on TTV1 + TTV3 ground (27); right column: the other way. A candidate is accepted if both shares clear the dotted bars at 25 m in both columns.",
         f"Pooled over both test halves at 25 m: {pooled}. Accepted: {H1['accepted']['candidate'] if H1.get('accepted') else 'none'}.")


# ----------------------------------------------------------------------------- figure 4: misfit; figure 5: K and D

def fig4(F: dict[str, Any], R: dict[str, Any]) -> None:
    names = [n for n in ("AD", "PF_C", "PF_A", "SJ_C", "F2_relief") if n in F["models"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [F["models"][n]["mse_poly"] for n in names]
    ax.bar(range(len(names)), vals, color=[COLOUR[n] for n in names], edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PLAIN[n].replace(", ", ",\n") for n in names], fontsize=8)
    ax.set_ylabel("misfit inside the M&D study area (m², mean squared elevation error)")
    ax.set_title("How well each model fits the real landscape on the domain (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    ratios = "; ".join(f"{PLAIN[n]} {F['models'][n]['mse_poly']:.1f} m²" + (f" ({R['misfit'][n]['ratio_to_AD']:.2f} × AD)" if R["misfit"][n].get("ratio_to_AD") is not None else "") for n in names if n in R["misfit"])
    save(fig, "fig4_misfit.png", "One bar per model: the mean squared difference between the model surface and the lidar over the domain cells inside the M&D study area. Lower is better.", ratios + ".")


def fig5(F: dict[str, Any]) -> None:
    names = [n for n in ("AD", "PF_C", "PF_A", "SJ_C") if n in F["models"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for i, n in enumerate(names):
        ax.plot(i, F["models"][n]["K_over_K0"], marker=MARKER[n], color=COLOUR[n], ms=10, ls="none", markeredgecolor="k")
    ax.axhspan(0.5, 2.0, color="0.85", zorder=0)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.set_ylabel("fitted K ÷ measured K₀ (log axis: 2 means twice)")
    ax.set_title("K against the measured value (grey band: within a factor of 2)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PLAIN[n].replace(", ", ",\n") for n in names], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    Dr = F["D_relief"]
    for i, n in enumerate(names):
        ax.plot(i, F["models"][n]["D"], marker=MARKER[n], color=COLOUR[n], ms=10, ls="none", markeredgecolor="k")
    ax.axhspan(Dr * 0.67, Dr * 1.5, color="0.85", zorder=0)
    ax.axhline(Dr, color="k", lw=0.8)
    ax.axhline(D_PHYS, color="#e31a1c", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("fitted D (m²/yr, log axis)")
    ax.set_title("D against the relief D (black line, grey band 0.67–1.5 ×) and Heimsath's D (red dashed)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PLAIN[n].replace(", ", ",\n") for n in names], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    txt = "; ".join(f"{PLAIN[n]}: K {F['models'][n]['K_over_K0']:.2f} × K₀, D {F['models'][n]['D']:.4f} m²/yr ({F['models'][n]['D'] / Dr:.2f} × relief D, {F['models'][n]['D'] / D_PHYS:.1f} × Heimsath's)" for n in names)
    save(fig, "fig5_K_D.png", "Left: each fitted model's K divided by the K measured from the channel profiles (K₀ = U/k_s on the domain); the grey band is a factor of 2 either way. Right: each model's D against the relief D "
         "measured on the domain (black line, grey band 0.67–1.5 ×) and Heimsath et al.'s 0.0025 m²/yr (red dashed).", f"K₀ {F['K0']:.3e}, relief D {Dr:.4f} m²/yr. {txt}.")


# ----------------------------------------------------------------------------- figure 6: spectrum ratio; figure 7: valley share

def fig6(MEAS: dict[str, Any]) -> None:
    sp = MEAS["spectrum"]
    edges = np.array(sp["bin_edges_m"])
    lam = np.sqrt(edges[:-1] * edges[1:])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhspan(RATIO_BAND[0], RATIO_BAND[1], color="0.88", zorder=0)
    ax.axvspan(READING_BAND_M[0], READING_BAND_M[1], color="#deebf7", zorder=0, alpha=0.6)
    for n in MODELS:
        q = sp["per_surface"].get(n)
        if q is None or n == "obs":
            continue
        ax.plot(lam, q["ratio_to_lidar"], marker=MARKER[n], color=COLOUR[n], lw=1.4, ms=6)
    ax.axhline(1.0, color="k", lw=0.8)
    ro = sp["per_surface"]["obs"]["rolloff"].get("characteristic_wavelength_m")
    if ro:
        ax.axvline(ro, color="0.3", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wavelength (m, log axis)")
    ax.set_ylabel("power spectrum: model ÷ lidar (log axis: 2 means twice the lidar's power)")
    ax.set_title("Does each model carry the lidar's topographic power at each wavelength?")
    ax.grid(alpha=0.3, which="both")
    leg = [handle(n, ls="-") for n in MODELS if n != "obs" and n in sp["per_surface"]] + [Patch(facecolor="0.88", label="within a factor of 2 of the lidar"), Patch(facecolor="#deebf7", label="20–200 m, the reading's band")]
    if ro:
        leg.append(Line2D([], [], color="0.3", ls=":", label="the lidar's characteristic wavelength (peak above its background power law)"))
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)  # below the axes: a legend inside the axes covered the AD points at 27-38 m
    fig.tight_layout()
    txt = "; ".join(f"{PLAIN[n]}: {q['ratio_in_reading_band']['min']:.2f}–{q['ratio_in_reading_band']['max']:.2f} over 20–200 m" for n, q in sp["per_surface"].items() if n != "obs")
    save(fig, "fig6_power_spectrum_ratio.png", "For each model, the binned 2-D power spectrum of its surface divided by the lidar's, on the largest square inside the domain, against wavelength. Above 1 the model is rougher "
         "than the real landscape at that scale; below 1 smoother. The grey band is a factor of 2 either way; the blue band is the reading's range of wavelengths.",
         f"Square side {sp['square']['side_m']} m; the lidar's characteristic wavelength {ro:.0f} m. {txt}." if ro else f"Square side {sp['square']['side_m']} m. {txt}.")


def fig7(MEAS: dict[str, Any]) -> None:
    v10, v5 = MEAS["valley"]["per_sigma"]["10"]["per_surface"], MEAS["valley"]["per_sigma"]["5"]["per_surface"]
    names = [n for n in MODELS if n != "obs" and n in v10]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [v10[n]["ratio_to_lidar"] for n in names], width=0.4, color=[COLOUR[n] for n in names], edgecolor="k", linewidth=0.5)
    ax.bar(x + 0.2, [v5[n]["ratio_to_lidar"] for n in names], width=0.4, color=[COLOUR[n] for n in names], edgecolor="k", linewidth=0.5, hatch="//", alpha=0.7)
    ax.axhspan(VALLEY_SHARE_RANGE[0], VALLEY_SHARE_RANGE[1], color="0.88", zorder=0)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PLAIN[n].replace(", ", ",\n") for n in names], fontsize=8)
    ax.set_ylabel("share of the domain that is valley: model ÷ lidar")
    ax.set_title("How much of each model surface is valley, relative to the real landscape")
    ax.legend(handles=[Patch(facecolor="0.6", edgecolor="k", label="smoothing 10 m (primary)"), Patch(facecolor="0.6", edgecolor="k", hatch="//", alpha=0.7, label="smoothing 5 m (sensitivity)"),
                       Patch(facecolor="0.88", label="within 0.67–1.5 of the lidar")], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    txt = "; ".join(f"{PLAIN[n]} {v10[n]['ratio_to_lidar']:.2f} (5 m: {v5[n]['ratio_to_lidar']:.2f})" for n in names)
    save(fig, "fig7_valley_share.png", "A cell is valley where the contour curvature of the smoothed surface exceeds the lidar's mean plus two standard deviations (the same threshold for every surface). "
         "Bars: each model's valley share of the domain divided by the lidar's, at 10 m smoothing (solid) and 5 m (hatched). The grey band is the reading's range.",
         f"Lidar valley share {v10['obs']['share_domain']:.3f} at 10 m, {v5['obs']['share_domain']:.3f} at 5 m. {txt}.")


# ----------------------------------------------------------------------------- figure 8: maps

def fig8(dom: dict[str, Any], F: dict[str, Any], M12: dict[str, Any]) -> None:
    S: dict[str, np.ndarray] = {"obs": dom["Z"]}
    for f in ("fixed", "AD", "SJ_C"):
        p = out_path("fits", f"{f}_surfaces.npz")
        if os.path.exists(p):
            z = np.load(p)
            for k in z.files:
                S[k] = z[k].astype(float)
    names = [n for n in MAP_ORDER if n in S]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.4))
    for ax in axes.ravel():
        ax.axis("off")
    pk = dom["picks"]
    use = pk["in_domain"] & pk["in_poly"]
    for ax, n in zip(axes.ravel(), names, strict=False):
        ax.axis("on")
        ax.imshow(hs_image(S[n]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        q = M12["models"].get(n)
        if q is not None:
            b = q["rules"]["primary"]["by_t"][TP]
            if b["extra_rc"]:
                e = np.array(b["extra_rc"])
                ax.plot(e[:, 1], e[:, 0], marker="o", markerfacecolor="none", markeredgecolor=COLOUR[n] if n != "obs" else "#444444", ms=5, markeredgewidth=1.0, ls="none", zorder=6)
            if b["pairs"]:
                m = np.array([p["model_rc"] for p in b["pairs"]])
                ax.plot(m[:, 1], m[:, 0], marker="o", color=COLOUR[n] if n != "obs" else "#444444", markeredgecolor="w", ms=5, ls="none", zorder=7)
        ax.plot(pk["col"][use], pk["row"][use], marker="^", color=MAPPED_COLOUR, ms=5, markeredgecolor="k", markeredgewidth=0.4, ls="none", zorder=6)
        map_decor(ax, dom)
        title = PLAIN[n]
        if n != "obs" and n in F["models"]:
            mm = F["models"][n]
            title += f"\nmisfit {mm['mse_poly']:.0f} m²" + (f", K {mm['K_over_K0']:.2f} × K₀, D {mm['D']:.4f} m²/yr" if mm.get("K_over_K0") else "")
        ax.set_title(title, fontsize=9.5)
    leg = MAP_LEGEND + [Line2D([], [], marker="^", color=MAPPED_COLOUR, markeredgecolor="k", ls="none", ms=6, label="M&D channel heads"),
                        Line2D([], [], marker="o", color="0.4", markeredgecolor="w", ls="none", ms=5, label="model head matched to a mapped head (heads found on a 12 m block mean of this 1 m surface)"),
                        Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="0.4", ls="none", ms=5, label="extra model head")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("The real landscape and each model on the domain: 1 m hillshades (lit from the north-west) with the TIGER roads, the M&D study area and the domain outline", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), h_pad=3.0)
    save(fig, "fig8_maps.png", "Six panels: the lidar and the five models, each shown as the hillshade of its own 1 m surface. Grey lines are TIGER roads and trails, the dashed line the M&D study area, the blue line the domain. "
         "Orange triangles are the mapped heads; circles are the heads the 12 m method finds on a block mean of each surface (filled where matched one-to-one within 25 m, open where extra).",
         "; ".join(f"{PLAIN[n]}: {M12['models'][n]['rules']['primary']['by_t'][TP]['n_matched']} matched of {M12['n_mapped']}, {M12['models'][n]['rules']['primary']['by_t'][TP]['n_extra']} extra" for n in names if n in M12["models"]) + ".")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    dom = load_domain()
    Dz = np.load(out_path("domain", "domain.npz"))
    dom["road_cells"] = np.unpackbits(Dz["road_cells_bits"])[: dom["shape"][0] * dom["shape"][1]].reshape(dom["shape"]).astype(bool)
    from a23_common import TTV_FULL, full_grid_meta, read_window
    b = dom["rec"]["box"]
    Zb = read_window(TTV_FULL["filled"], b["full_rows"][0], b["full_rows"][1], b["full_cols"][0], b["full_cols"][1]).astype(float)
    nod = full_grid_meta()["nodata"]
    if nod is not None:
        Zb[Zb == nod] = np.nan
    dom["Z_lidar_box"] = Zb                      # the lidar of the whole box (context outside the domain), for figure 1 only
    want = {a.only} if a.only else set(range(1, 9))
    if 1 in want:
        fig1(dom)
    if 2 in want and os.path.exists(out_path("roads", "road_check.json")):
        fig2(load_json(out_path("roads", "road_check.json")))
    if 3 in want and os.path.exists(out_path("heads", "heads1m.json")):
        fig3(load_json(out_path("heads", "heads1m.json")))
    post = out_path("post", "metrics.json")
    if os.path.exists(post):
        PM, R = load_json(post), load_json(out_path("post", "rules.json"))["readings"]
        if 4 in want:
            fig4(PM["fits"], R)
        if 5 in want:
            fig5(PM["fits"])
        if 8 in want:
            fig8(dom, PM["fits"], PM["heads12"])
    meas = out_path("measures", "measures.json")
    if os.path.exists(meas):
        MEAS = load_json(meas)
        if 6 in want:
            fig6(MEAS)
        if 7 in want:
            fig7(MEAS)
    dump(CAPTIONS, out_path("post", "captions.json"))
    print(f"figures: {sorted(CAPTIONS)} [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
