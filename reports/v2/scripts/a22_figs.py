#!/usr/bin/env python3
"""Addendum 22 figures (directive section 5: plain-language titles, no undefined abbreviation, linear axes unless stated, the good direction
marked, reference values labelled in words, values on bars, one colour and marker per model, at most six models per panel, two-line
captions). Figures 1-8 of section 5.4 in reports/v2/figures/a22/; the captions are also written to <run>/post/captions.json.
usage: a22_figs.py
"""
from __future__ import annotations

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
from a22_common import (  # noqa: E402
    BLOCK_M,
    C_F,
    COLOUR,
    D_PHYS,
    FIG_DIR,
    FITTED_MODELS,
    FIXED_MODELS,
    MAP_PANELS,
    MAPPED_COLOUR,
    MARKER,
    MODELS,
    N_MAPPED_HEADS_POOLED,
    PLAIN,
    T_PRIMARY_M,
    TILES,
    TOLERANCES_M,
    criterion_sets,
    dump,
    hillshade,
    load_json,
    out_path,
)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.titlesize": 12})
TP = f"{T_PRIMARY_M:g}"
TILE_WORDS = {"TTV1": "tile TTV1", "TTV2": "tile TTV2", "TTV3": "tile TTV3"}
AREA_NOTE = "misfit computed only inside the M&D study area on tiles TTV1, TTV2 and TTV3"
GROUP_A = ["obs", "AD", "PF_A", "PF_C", "SJ_C", "P4"]          # six models per panel (rule 11)
GROUP_B = ["obs", "F2_relief", "F2_phys"]
CAPTIONS: dict[str, dict[str, str]] = {}


def save(fig, name: str, how: str, what: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.text(0.01, -0.01, f"How to read: {how}\nWhat it shows: {what}", ha="left", va="top", fontsize=8.5, wrap=True, transform=fig.transFigure)
    path = os.path.join(FIG_DIR, name)
    for sc in (1.0, 0.8, 0.6):
        fig.savefig(path, dpi=int(170 * sc), bbox_inches="tight", facecolor="white")
        if os.path.getsize(path) <= 5_000_000:
            break
    plt.close(fig)
    CAPTIONS[name] = {"how": how, "what": what}
    return path


def handle(n: str, **kw) -> Line2D:
    return Line2D([], [], color=COLOUR[n], marker=MARKER[n], linestyle=kw.get("ls", "none"), markersize=kw.get("ms", 7), markeredgecolor="k" if n == "F2_phys" else COLOUR[n], label=PLAIN[n])


def better_arrow(ax, axis: str, direction: str, text: str, x: float = 1.02, y: float = 0.5) -> None:
    """A labelled arrow at the edge of the axes saying which way is good."""
    if axis == "y":
        dy = 0.18 if direction == "up" else -0.18
        ax.annotate("", xy=(x, y + dy), xytext=(x, y - dy), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 1.2}, annotation_clip=False)
        ax.text(x + 0.015, y, text, rotation=90, va="center", ha="left", fontsize=8, color="0.3", transform=ax.transAxes)
    else:
        dx = 0.12 if direction == "right" else -0.12
        ax.annotate("", xy=(x + dx, y), xytext=(x - dx, y), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 1.2}, annotation_clip=False)
        ax.text(x, y - 0.05, text, ha="center", va="top", fontsize=8, color="0.3", transform=ax.transAxes)


def bar_values(ax, bars, fmt: str = "{:.0f}", fontsize: float = 7.5, rotation: int = 0) -> None:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize, rotation=rotation)


def g(a: dict[str, Any], n: str, *keys, default=float("nan")):
    """a[n][keys...] or the default when the model is absent (partial runs)."""
    try:
        v = a[n]
        for k in keys:
            v = v[k]
        return v
    except (KeyError, TypeError):
        return default


def load_all() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return load_json(out_path("post", "metrics.json")), load_json(out_path("post", "rules.json")), load_json(out_path("post", "ad_check.json"))


# ----------------------------------------------------------------------------- figure 1

def avail(group: list[str], have) -> list[str]:
    return [n for n in group if n in have]


def fig1(F: dict[str, Any]) -> None:
    six = avail(["AD", "PF_A", "PF_C", "SJ_C", "P4", "F2_relief"], F[TILES[0]]["models"])
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    for ax, tile in zip(axes[:3], TILES, strict=True):
        vals = [F[tile]["models"][n]["mse_poly"] for n in six]
        bars = ax.bar(range(len(six)), vals, color=[COLOUR[n] for n in six], edgecolor="k", linewidth=0.5)
        bar_values(ax, bars, "{:.0f}")
        ad = g(F[tile]["models"], "AD", "mse_poly")
        if np.isfinite(ad):
            ax.axhline(ad, color=COLOUR["AD"], ls="--", lw=1)
            ax.text(len(six) - 0.5, ad, f"AD model: {ad:.0f} m²", ha="right", va="bottom", fontsize=8, color=COLOUR["AD"])
        ax.set_xticks(range(len(six)))
        ax.set_xticklabels([PLAIN[n].replace("Split model, ", "Split:\n").replace(" rule", "\nrule").replace("area × slope²", "area ×\nslope²") for n in six], rotation=0, fontsize=6.5)
        ax.set_ylabel("misfit inside the M&D area (mean squared elevation error, m²)")
        ax.set_title(f"{TILE_WORDS[tile]}: misfit of each model inside the M&D area")
        ax.set_ylim(0, max(vals) * 1.25)
        better_arrow(ax, "y", "down", "lower is better", x=1.01, y=0.5)
    ax = axes[3]
    x = np.arange(3)
    v1 = [g(F[t]["models"], "F2_phys", "mse_poly") for t in TILES]
    v2 = [g(F[t]["models"], "AD", "mse_poly") for t in TILES]
    b1 = ax.bar(x - 0.2, v1, 0.4, color=COLOUR["F2_phys"], edgecolor="k", linewidth=0.5, label=PLAIN["F2_phys"])
    b2 = ax.bar(x + 0.2, v2, 0.4, color=COLOUR["AD"], edgecolor="k", linewidth=0.5, label=PLAIN["AD"])
    bar_values(ax, b1, "{:,.0f}")
    bar_values(ax, b2, "{:,.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels(TILES)
    ax.set_ylabel("misfit inside the M&D area (m²)")
    ax.set_title("For scale: the nothing-fitted model with\nHeimsath's physical D, beside the AD model")
    ax.set_ylim(0, max(v1) * 1.45)
    ax.legend(loc="upper center", fontsize=7)
    better_arrow(ax, "y", "down", "lower is better", x=1.01, y=0.5)
    fig.suptitle("How well does each model fit the M&D area?", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_misfit_md_area.png",
         f"Each bar is a model's misfit (mean squared elevation error, m²) computed only over the cells inside the M&D study area, one panel per tile (TTV1, TTV2, TTV3); the dashed line marks the AD model's value; lower is better. The fourth panel puts the physical-D model on its own scale because its misfit is tens of times larger. {AREA_NOTE}.",
         "; ".join(f"{t}: AD {g(F[t]['models'], 'AD', 'mse_poly'):.0f}, split M&D area×slope² rule {g(F[t]['models'], 'PF_C', 'mse_poly'):.0f}, split area×slope² fitted {g(F[t]['models'], 'SJ_C', 'mse_poly'):.0f}, split area rule fitted {g(F[t]['models'], 'P4', 'mse_poly'):.0f}, nothing fitted {g(F[t]['models'], 'F2_relief', 'mse_poly'):.0f} m²" for t in TILES) + ".")


# ----------------------------------------------------------------------------- figure 2

def fig2(PM: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1.1]})
    for ax, group, title in ((axes[0], avail(GROUP_A, PM["models"]), "the real landscape and the five fitted models"), (axes[1], avail(GROUP_B, PM["models"]), "the real landscape and the two nothing-fitted models")):
        x = np.arange(len(group))
        w = 0.26
        mt = [PM["models"][n]["by_t"][TP]["n_matched"] for n in group]
        ms = [PM["models"][n]["by_t"][TP]["n_missed"] for n in group]
        ex = [PM["models"][n]["by_t"][TP]["n_extra"] for n in group]
        b1 = ax.bar(x - w, mt, w, color=[COLOUR[n] for n in group], edgecolor="k", linewidth=0.6, label="matched: a mapped head with a model head within 25 m (one-to-one)")
        b2 = ax.bar(x, ms, w, color="white", edgecolor=MAPPED_COLOUR, hatch="xx", linewidth=1.0, label="missed: mapped heads with no model head within 25 m")
        b3 = ax.bar(x + w, ex, w, color=[COLOUR[n] for n in group], alpha=0.35, edgecolor="k", linewidth=0.6, label="extra: model heads with no mapped head within 25 m")
        for bb in (b1, b2, b3):
            bar_values(ax, bb, "{:.0f}", 8)
        ax.axhline(N_MAPPED_HEADS_POOLED, color=MAPPED_COLOUR, ls="--", lw=1.2)
        ax.text(len(group) - 0.5, N_MAPPED_HEADS_POOLED, f"number of M&D heads ({N_MAPPED_HEADS_POOLED})", ha="right", va="bottom", fontsize=8, color=MAPPED_COLOUR)
        ax.set_xticks(x)
        ax.set_xticklabels([PLAIN[n].replace("Split model, ", "Split:\n").replace(" rule", "\nrule").replace(", physical D", ",\nphysical D") for n in group], fontsize=7.5)
        ax.set_ylabel("number of channel heads inside the M&D area (TTV1 + TTV2 + TTV3)" if "AD" in group or len(group) > 3 else "number of heads")
        ax.set_title(f"At a 25 m match tolerance:\n{title}")
        ax.set_ylim(0, max(max(mt), max(ms), max(ex), N_MAPPED_HEADS_POOLED) * 1.2)
        if "AD" in group:
            ax.legend(loc="upper left", fontsize=7.5)
    axes[0].text(1.0, -0.22, "good: matched high (close to 61), missed low, extra low", transform=axes[0].transAxes, ha="right", fontsize=8, color="0.3")
    fig.suptitle("How many of M&D's mapped heads does each model find, and how many extra heads does it make?", y=1.0)
    fig.tight_layout()
    a = PM["models"]
    save(fig, "fig2_matched_missed_extra.png",
         "For each model, three bars pooled over tiles TTV1–TTV3 inside the M&D study area: mapped heads matched one-to-one to a model head within 25 m (solid), mapped heads missed (hatched orange), and extra model heads with no mapped head within 25 m (pale). The dashed line is the number of M&D heads (61). Good is matched near 61, missed and extra near 0. The right panel holds the two nothing-fitted models; the real landscape (lidar) is repeated as the reference.",
         f"Real landscape {g(a, 'obs', 'by_t', TP, 'n_matched')} matched / {g(a, 'obs', 'by_t', TP, 'n_extra')} extra; AD {g(a, 'AD', 'by_t', TP, 'n_matched')} / {g(a, 'AD', 'by_t', TP, 'n_extra')}; split M&D area×slope² rule {g(a, 'PF_C', 'by_t', TP, 'n_matched')} / {g(a, 'PF_C', 'by_t', TP, 'n_extra')}; "
         f"split area×slope² fitted {g(a, 'SJ_C', 'by_t', TP, 'n_matched')} / {g(a, 'SJ_C', 'by_t', TP, 'n_extra')}; split area rule fitted {g(a, 'P4', 'by_t', TP, 'n_matched')} / {g(a, 'P4', 'by_t', TP, 'n_extra')}; nothing fitted {g(a, 'F2_relief', 'by_t', TP, 'n_matched')} / {g(a, 'F2_relief', 'by_t', TP, 'n_extra')}; physical D {g(a, 'F2_phys', 'by_t', TP, 'n_matched')} / {g(a, 'F2_phys', 'by_t', TP, 'n_extra')}.")


# ----------------------------------------------------------------------------- figure 3

def fig3(PM: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharex=True)
    for row, group in enumerate((avail(GROUP_A, PM["models"]), avail(GROUP_B, PM["models"]))):
        for col, (key, ylab, ttl) in enumerate((("share_found", "share of M&D's mapped heads found (matched ÷ mapped)", "Share of mapped heads found"),
                                                ("share_real", "share of the model's heads that are real (matched ÷ model heads)", "Share of model heads that are real"))):
            ax = axes[row, col]
            for n in group:
                ys = [PM["models"][n]["by_t"][f"{t:g}"][key] for t in TOLERANCES_M]
                lab = PLAIN[n] + (" — the best achievable with this head-finding method" if n == "obs" else "")
                ax.plot(TOLERANCES_M, ys, color=COLOUR[n], marker=MARKER[n], ms=7 if n != "obs" else 10, lw=1.6 if n != "obs" else 2.2, ls="-" if n != "obs" else "--",
                        markeredgecolor="k" if n == "F2_phys" else COLOUR[n], label=lab)
            ax.axvline(T_PRIMARY_M, color="0.6", ls=":", lw=1)
            ax.text(T_PRIMARY_M + 0.5, 0.02, "primary tolerance 25 m", rotation=90, fontsize=7.5, color="0.4", va="bottom")
            ax.axhline(0.5, color="0.6", ls="--", lw=0.8)
            ax.text(TOLERANCES_M[0], 0.51, "reading threshold 0.5", fontsize=7.5, color="0.4", va="bottom")
            ax.set_ylim(0, 1.02)
            ax.set_xticks(TOLERANCES_M)
            ax.set_ylabel(ylab)
            ax.set_title(f"{ttl}: {'real landscape and fitted models' if row == 0 else 'real landscape and nothing-fitted models'}")
            if row == 1:
                ax.set_xlabel("match tolerance: how close a model head must be to a mapped head to count (m)")
            better_arrow(ax, "y", "up", "higher is better", x=1.01)
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("Does the answer depend on how close counts as a match?", y=1.0)
    fig.tight_layout()
    a = PM["models"]
    save(fig, "fig3_tolerance.png",
         "Left: the share of M&D's mapped heads that have a model head paired to them (one-to-one), against the match tolerance from 10 to 50 m. Right: the share of the model's heads that are paired to a mapped head. Higher is better on both. The black dashed line is the real landscape (lidar) through the same head-finding method, the best achievable; the vertical dotted line is the primary tolerance (25 m). Pooled over tiles TTV1–TTV3 inside the M&D study area.",
         f"At 25 m the real landscape reaches {g(a, 'obs', 'by_t', TP, 'share_found'):.2f} found / {g(a, 'obs', 'by_t', TP, 'share_real'):.2f} real; AD {g(a, 'AD', 'by_t', TP, 'share_found'):.2f} / {g(a, 'AD', 'by_t', TP, 'share_real'):.2f}; split M&D area×slope² rule {g(a, 'PF_C', 'by_t', TP, 'share_found'):.2f} / {g(a, 'PF_C', 'by_t', TP, 'share_real'):.2f}; "
         f"split area×slope² fitted {g(a, 'SJ_C', 'by_t', TP, 'share_found'):.2f} / {g(a, 'SJ_C', 'by_t', TP, 'share_real'):.2f}; nothing fitted {g(a, 'F2_relief', 'by_t', TP, 'share_found'):.2f} / {g(a, 'F2_relief', 'by_t', TP, 'share_real'):.2f}.")


# ----------------------------------------------------------------------------- figure 4

def fig4(PM: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [2, 1.1]}, sharey=True)
    rng = np.random.default_rng(20261123)
    allv = [v for n in avail(MODELS, PM["models"]) for v in PM["models"][n]["by_t"][TP]["offset_values_m"]]
    ylim = max(60.0, float(np.percentile(np.abs(allv), 98)) * 1.1) if allv else 60.0
    for ax, group in ((axes[0], avail(GROUP_A, PM["models"])), (axes[1], avail(GROUP_B, PM["models"]))):
        for i, n in enumerate(group):
            b = PM["models"][n]["by_t"][TP]
            v = np.array(b["offset_values_m"], float)
            x = i + rng.uniform(-0.22, 0.22, v.size)
            ax.scatter(x, np.clip(v, -ylim, ylim), s=22, color=COLOUR[n], marker=MARKER[n], alpha=0.75, edgecolors="k" if n == "F2_phys" else "none", linewidths=0.4)
            if b["median_offset_m"] is not None:
                ax.hlines(b["median_offset_m"], i - 0.35, i + 0.35, color="k", lw=2.2)
                ax.text(i, b["median_offset_m"] + ylim * 0.05, f"median {b['median_offset_m']:+.0f} m", fontsize=7.5, va="bottom", ha="center")
            ax.text(i, -ylim * 0.9, f"{b['n_matched']} matched\n{v.size} with an offset\n{b['offset_kinds']['neighbouring hollow']} in another hollow", ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="0.3", lw=1)
        ax.set_yticks([-60, -40, -20, 0, 20, 40, 60])
        ax.set_yticklabels(["-60", "-40", "-20", "0\nstarts where\nM&D mapped", "20", "40", "60"])
        ax.axhspan(-25, 25, color="0.92", zorder=0)
        ax.text(-0.45, 25, "within 25 m: counts as matching the map", fontsize=7.5, color="0.4", va="bottom")
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels([PLAIN[n].replace("Split model, ", "Split:\n").replace(" rule", "\nrule").replace(", physical D", ",\nphysical D") for n in group], fontsize=7.5)
        ax.set_ylim(-ylim, ylim)
        ax.set_title("Real landscape and fitted models" if "AD" in group or len(group) > 3 else "Real landscape and nothing-fitted models")
    axes[0].set_ylabel("along-valley offset of each matched pair (m)")
    for ax in axes:
        ax.text(0.01, 0.98, "↑ positive: the model channel starts downslope of the mapped head (channel too short)", transform=ax.transAxes, fontsize=8, color="0.3", va="top", ha="left")
        ax.text(0.01, 0.02, "↓ negative: the model channel starts upslope of the mapped head (channel too long)", transform=ax.transAxes, fontsize=8, color="0.3", va="bottom", ha="left")
    fig.suptitle("Does each model start its channels too high or too low? (matched pairs at 25 m, offset traced along the real landscape's flow paths)", y=1.0)
    fig.tight_layout()
    a = PM["models"]
    save(fig, "fig4_offsets.png",
         "Each point is one matched pair (a mapped head and its model head within 25 m): the distance along the real landscape's 12 m flow path between the two. Positive means the model channel starts downslope of the mapped head (too short), negative upslope (too long); the zero line is 'starts where M&D mapped'; the black bar is the median; the grey band is ± 25 m. Pairs whose two heads are in different hollows have no offset and are counted in the text under each strip. Pooled over tiles TTV1–TTV3 inside the M&D study area.",
         "Medians: " + "; ".join(f"{PLAIN[n]} {a[n]['by_t'][TP]['median_offset_m']:+.0f} m" if a[n]["by_t"][TP]["median_offset_m"] is not None else f"{PLAIN[n]} no pairs" for n in avail(MODELS, a)) + ".")


# ----------------------------------------------------------------------------- figure 5

def fig5(PM: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2, 1.1]}, sharey=False)
    md = PM["md_length_m"]
    for ax, group in ((axes[0], avail(GROUP_A, PM["models"])), (axes[1], avail(GROUP_B, PM["models"]))):
        vals = [PM["models"][n]["channel_length_m"] / 1000 for n in group]
        bars = ax.bar(range(len(group)), vals, color=[COLOUR[n] for n in group], edgecolor="k", linewidth=0.6)
        bar_values(ax, bars, "{:.1f} km")
        ax.axhline(md / 1000, color=MAPPED_COLOUR, ls="--", lw=1.4)
        ax.text(len(group) - 0.5, md / 1000, f"M&D's traced channel length inside the area: {md / 1000:.1f} km", ha="right", va="bottom", fontsize=8, color=MAPPED_COLOUR)
        if "obs" in PM["models"]:
            ol = PM["models"]["obs"]["channel_length_m"] / 1000
            ax.axhline(ol, color="k", ls=":", lw=1.2)
            ax.legend(handles=[Line2D([], [], color=MAPPED_COLOUR, ls="--", lw=1.4, label=f"M&D's traced channel length inside the area ({md / 1000:.1f} km)"),
                               Line2D([], [], color="k", ls=":", lw=1.2, label=f"real landscape through the same method ({ol:.1f} km)")], loc="upper left", fontsize=7.5)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels([PLAIN[n].replace("Split model, ", "Split:\n").replace(" rule", "\nrule").replace(", physical D", ",\nphysical D") for n in group], fontsize=7.5)
        ax.set_ylabel("total 12 m channel length inside the M&D area (km)")
        ax.set_ylim(0, np.nanmax([max(vals), md / 1000]) * 1.25)
        ax.set_title(("Real landscape and fitted models" if "AD" in group or len(group) > 3 else "Real landscape and nothing-fitted models") + "\n(closer to the dashed line is better; the dotted line is what this method gives on the real landscape)", fontsize=9)
    fig.suptitle("Total channel length inside the M&D area (TTV1 + TTV2 + TTV3)", y=1.0)
    fig.tight_layout()
    a = PM["models"]
    save(fig, "fig5_channel_length.png",
         "Each bar is the total length of a model's 12 m channel network inside the M&D study area on tiles TTV1–TTV3 (sum of cell-to-cell flow distances; 876 m² area × slope² rule applied to the model's own surface); the dashed line is the length of the channels M&D traced inside the same area. A bar at the line matches the map.",
         "Model ÷ mapped: " + "; ".join(f"{PLAIN[n]} {a[n]['length_ratio']:.2f}" for n in avail(MODELS, a)) + ".")


# ----------------------------------------------------------------------------- figure 6 (maps)

def scale_bar(ax, nb: int, length_m: float = 200.0) -> None:
    cells = length_m / BLOCK_M
    x0, y0 = nb * 0.05, nb * 0.95
    ax.plot([x0, x0 + cells], [y0, y0], color="k", lw=3, solid_capstyle="butt")
    ax.plot([x0, x0 + cells], [y0, y0], color="w", lw=1, solid_capstyle="butt")
    ax.text(x0 + cells / 2, y0 - 1.2, f"{length_m:.0f} m", ha="center", va="bottom", fontsize=7.5, color="k", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})


def north_arrow(ax, nb: int) -> None:
    x, y = nb * 0.94, nb * 0.14
    ax.annotate("", xy=(x, y - nb * 0.09), xytext=(x, y), arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 1.5})
    ax.text(x, y - nb * 0.10, "N", ha="center", va="bottom", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})


def fig6(M: dict[str, Any]) -> None:
    for tile in TILES:
        P = dict(np.load(out_path("tiles", f"{tile}_products.npz")))
        for name in ("AD", "SJ_C", "P4"):
            p = out_path("profiles", f"{name}_{tile}_products.npz")
            if os.path.exists(p):
                P.update(np.load(p))
        inside = P["poly12_centre"]
        nb = inside.shape[0]
        pr, pc = P["pick_rc"]
        ids = P["pick_ids"]
        pos = {int(i): (c / BLOCK_M - 0.5, r / BLOCK_M - 0.5) for i, r, c in zip(ids, pr, pc, strict=True)}
        TM = M["per_tile"]["primary"][tile]
        mapped_ids = TM["mapped_ids"]
        panels = [n for n in MAP_PANELS if n in TM["models"]]
        fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.5))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, name in zip(axes.ravel(), panels, strict=False):
            ax.axis("on")
            Z12 = P[f"{name}_Z12"].astype(float)
            ch, hd = criterion_sets(P[f"{name}_M_a2"], P[f"{name}_Mdon_a2"], C_F)
            ax.imshow(np.nan_to_num(hillshade(Z12, float(BLOCK_M)), nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.imshow(np.where(ch, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap([COLOUR[name]]), alpha=0.45, interpolation="nearest")
            ax.contour(inside.astype(float), levels=[0.5], colors="k", linewidths=1.0, linestyles="--")
            b = TM["models"][name]["by_t"][TP]
            matched_map = {p["mapped_id"]: p for p in b["pairs"]}
            for pid in mapped_ids:
                x, y = pos[pid]
                if pid in matched_map:
                    mc = matched_map[pid]["model_cell"]
                    ax.plot([x, mc[1]], [y, mc[0]], color="k", lw=0.7, zorder=3)
                    ax.plot(x, y, marker="^", color=MAPPED_COLOUR, ms=8, markeredgecolor="k", markeredgewidth=0.5, zorder=4, ls="none")
                    ax.plot(mc[1], mc[0], marker="o", color=COLOUR[name], ms=8, markeredgecolor="w", markeredgewidth=1.0, zorder=5, ls="none")
                else:
                    ax.plot(x, y, marker="^", color=MAPPED_COLOUR, ms=8, markeredgecolor="k", markeredgewidth=0.5, zorder=4, ls="none")
                    ax.plot(x, y, marker="x", color="k", ms=7, markeredgewidth=1.4, zorder=6, ls="none")
            for ec in b["extra_cells"]:
                ax.plot(ec["c"], ec["r"], marker="o", markerfacecolor="none", markeredgecolor=COLOUR[name], ms=7, markeredgewidth=1.3, zorder=5, ls="none")
            rr, cc = np.nonzero(hd & ~inside)
            ax.plot(cc, rr, marker=".", color="0.5", ms=3, ls="none", zorder=2)
            scale_bar(ax, nb)
            north_arrow(ax, nb)
            ax.text(nb * 0.02, nb * 0.03, "M&D study area (dashed)", fontsize=7.5, va="top", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
            ax.set_title(f"{PLAIN[name]}\n{TILE_WORDS[tile]}: {b['n_matched']} matched, {b['n_missed']} missed, {b['n_extra']} extra (25 m)", fontsize=9.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, nb - 0.5)
            ax.set_ylim(nb - 0.5, -0.5)
        legend = [Line2D([], [], marker="^", color=MAPPED_COLOUR, markeredgecolor="k", ls="none", ms=8, label="M&D heads (mapped), matched"),
                  Line2D([], [], marker="^", color=MAPPED_COLOUR, markeredgecolor="k", ls="none", ms=8, label="M&D heads (mapped), missed: black cross"),
                  Line2D([], [], marker="o", color="0.4", markeredgecolor="k", ls="none", ms=7, label="model head matched to a mapped head (filled; thin line joins the pair)"),
                  Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="0.4", ls="none", ms=7, label="extra model head (open)"),
                  Patch(facecolor="0.4", alpha=0.45, label="the model's 12 m channel cells (area × slope² ≥ 876 m²)"),
                  Line2D([], [], marker=".", color="0.5", ls="none", label="model heads outside the M&D area (not counted)"),
                  Line2D([], [], color="k", ls="--", label="M&D study area")]
        fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.005))
        fig.suptitle(f"Where are the channel heads on {TILE_WORDS[tile]}? Mapped heads against each model's heads inside the M&D study area (hillshade: the model's own 12 m surface, lit from the north-west)", y=0.995)
        fig.tight_layout(rect=(0, 0.04, 1, 0.98), h_pad=3.0)
        save(fig, f"fig6_map_{tile}.png",
             f"Six panels for {TILE_WORDS[tile]}: the real landscape and five models, each shown as the hillshade of its own 12 m surface with its channel cells shaded. Orange triangles are M&D's mapped heads (a black cross marks one no model head came within 25 m of); filled circles are model heads matched one-to-one to a mapped head (a thin line joins the pair); open circles are extra model heads. Only heads inside the dashed M&D study area are counted. Scale bar 200 m; north up.",
             "; ".join(f"{PLAIN[n]}: {TM['models'][n]['by_t'][TP]['n_matched']} matched of {TM['n_mapped']}, {TM['models'][n]['by_t'][TP]['n_extra']} extra" for n in panels) + ".")


# ----------------------------------------------------------------------------- figure 7 (the AD check)

def fig7(ADC: dict[str, Any], M: dict[str, Any]) -> None:
    tile = ADC["tile"]
    P = dict(np.load(out_path("tiles", f"{tile}_products.npz")))
    P.update(np.load(out_path("profiles", f"AD_{tile}_products.npz")))
    inside = P["poly12_centre"]
    nb = inside.shape[0]
    s_obs, s_ad = P["obs_slope12"].astype(float), P["AD_slope12"].astype(float)
    vmax = float(np.nanpercentile(np.concatenate([s_obs[inside], s_ad[inside]]), 99))
    fig = plt.figure(figsize=(17, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    b = M["per_tile"]["primary"][tile]["models"]["AD"]["by_t"][TP]
    for ax, S, name in ((ax1, s_ad, "AD"), (ax2, s_obs, "obs")):
        im = ax.imshow(S, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.contour(inside.astype(float), levels=[0.5], colors="w", linewidths=1.0, linestyles="--")
        for ec in b["extra_cells"]:
            ax.plot(ec["c"], ec["r"], marker="o", markerfacecolor="none", markeredgecolor="#00e5ff" if ec["class"] == "too-steep wall" else "w", ms=7, markeredgewidth=1.3, ls="none")
        for p in b["pairs"]:
            ax.plot(p["model_cell"][1], p["model_cell"][0], marker="o", color="#00e5ff", ms=6, markeredgecolor="k", markeredgewidth=0.4, ls="none")
        if "transect" in ADC:
            cells = np.array(ADC["transect"]["cells"])
            ax.plot(cells[:, 1], cells[:, 0], color="lime", lw=1.8)
            e = np.array(ADC["line_end_cells"])
            ax.plot(e[:, 1], e[:, 0], color="w", lw=1.0, ls=":")
        scale_bar(ax, nb)
        north_arrow(ax, nb)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{'AD model' if name == 'AD' else 'Real landscape (lidar)'} on {TILE_WORDS[tile]}:\nlocal slope of the 12 m surface" + (", with the AD model's heads" if name == "AD" else " (same colour scale)"), fontsize=9.5)
    cb = fig.colorbar(im, ax=[ax1, ax2], location="bottom", fraction=0.04, pad=0.03, shrink=0.6)
    cb.set_label("local slope of the 12 m surface (rise ÷ run)")
    if "transect" in ADC:
        tr = ADC["transect"]
        d = np.array(tr["distance_m"])
        ax3.plot(d, tr["z_obs"], color="k", lw=2, label="real landscape (lidar), 12 m surface")
        ax3.plot(d, tr["z_AD"], color=COLOUR["AD"], lw=2, label="AD model surface")
        ax3.axvline(0, color="lime", lw=1.2, ls="--")
        ax3.text(2, min(min(tr["z_obs"]), min(tr["z_AD"])), "line of AD's extra heads", rotation=90, va="bottom", fontsize=8, color="green")
        ax3.set_xlabel("distance across the line of extra heads (m; the green transect on the maps)")
        ax3.set_ylabel("elevation (m)")
        ax3.set_title(f"Profile across the line: AD and the real landscape on {TILE_WORDS[tile]}", fontsize=9.5)
        ax3.legend(loc="best", fontsize=8)
        ax3.grid(alpha=0.3)
    leg = [Line2D([], [], marker="o", color="#00e5ff", markeredgecolor="k", ls="none", ms=6, label="AD head matched to a mapped head (filled)"),
           Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#00e5ff", ls="none", ms=7, label="extra AD head on a slope steeper than the steepest 5 % of the real landscape (open, cyan)"),
           Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="w", ls="none", ms=7, label="other extra AD head (open, white)"),
           Line2D([], [], color="lime", lw=1.8, label="profile transect (right panel)"), Line2D([], [], color="w", ls=":", label="principal axis of the steep-wall extra heads"),
           Line2D([], [], color="w", ls="--", label="M&D study area")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02), facecolor="0.3", labelcolor="w")
    fig.suptitle(f"What is the line of extra heads on the AD surface? ({TILE_WORDS[tile]})", y=1.0)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.2, wspace=0.12)
    what = (f"AD makes {ADC['n_extra']} extra heads on {tile} at 25 m, {ADC['n_wall']} of them on slopes above the real landscape's 95th percentile ({ADC.get('slope_at_wall_heads', {}).get('obs_95th_in_polygon', float('nan')):.2f}); "
            f"at those cells the AD slope is {ADC.get('slope_at_wall_heads', {}).get('AD_median', float('nan')):.2f} against {ADC.get('slope_at_wall_heads', {}).get('obs_median', float('nan')):.2f} in the real landscape, and the AD surface sits "
            f"{ADC.get('elevation_difference_at_wall_heads_m', {}).get('AD_minus_obs_median', float('nan')):+.1f} m from it.") if "slope_at_wall_heads" in ADC else f"AD makes {ADC['n_extra']} extra heads on {tile}; fewer than three are on steep walls, so no line was fitted."
    save(fig, "fig7_ad_check.png",
         "Left and middle: the local slope of the AD model's 12 m surface and of the real landscape's, on one colour scale (brighter = steeper), with the AD model's heads inside the M&D study area overlaid (filled = matched to a mapped head, open = extra; cyan open = on a too-steep slope). The green line is the transect of the right panel, drawn through the extra heads' centroid across their principal axis (white dotted). Right: the elevation of both surfaces along the transect.",
         what)


# ----------------------------------------------------------------------------- figure 8

def fig8(F: dict[str, Any]) -> None:
    fm = avail(FITTED_MODELS, F[TILES[0]]["models"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    offs = np.linspace(-0.3, 0.3, len(fm))
    ax = axes[0]
    ax.axhspan(0.5, 2.0, color="#dfe9f3", zorder=0)
    ax.text(2.45, 2.0, "within a factor of 2 of measured", ha="right", va="bottom", fontsize=8, color="0.3")
    ax.axhline(1.0, color="0.4", lw=1, ls="--")
    ax.text(-0.45, 0.52, "dashed line: the measured erosion-rate-based K (K₀ = U ÷ kₛ); shaded: within a factor of 2 of it", fontsize=7.5, va="bottom", color="0.3")
    for i, t in enumerate(TILES):
        for j, n in enumerate(fm):
            v = F[t]["models"][n]["K_over_K0"]
            ax.plot(i + offs[j], v, marker=MARKER[n], color=COLOUR[n], ms=9, ls="none", markeredgecolor="k", markeredgewidth=0.4)
            ax.annotate(f"{v:.2f}", (i + offs[j], v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    ax.set_yscale("log")
    yt = [0.25, 0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y:g}" for y in yt])
    ax.set_ylim(0.2, 25)
    ax.set_xticks(range(3))
    ax.set_xticklabels(TILES)
    ax.set_ylabel("fitted K ÷ measured erosion-rate-based K (log scale: 2 means twice the measured value)")
    ax.set_title("Is the fitted K physical? (misfit inside the M&D area)")
    ax.legend(handles=[handle(n) for n in fm], loc="lower right", fontsize=7.5)
    ax = axes[1]
    ax.axhline(D_PHYS, color="k", lw=1.2, ls="--")
    ax.text(-0.45, D_PHYS, "Heimsath's measured D (0.0025 m²/yr)", fontsize=7.5, va="bottom")
    for i, t in enumerate(TILES):
        Dr = F[t]["D_relief"]
        ax.hlines(Dr, i - 0.4, i + 0.4, color="0.35", lw=1.4, ls="-.")
        ax.text(i + 0.42, Dr, f"relief D, {t}\n({Dr:.4f})", fontsize=7, va="center", color="0.35")
        for j, n in enumerate(fm):
            v = F[t]["models"][n]["D"]
            ax.plot(i + offs[j], v, marker=MARKER[n], color=COLOUR[n], ms=9, ls="none", markeredgecolor="k", markeredgewidth=0.4)
            ax.annotate(f"{v:.4f}" if v >= 1e-3 else f"{v:.1e}", (i + offs[j], v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.5)
    ax.set_yscale("log")
    yt = [1e-6, 1e-5, 1e-4, 1e-3, 0.0025, 0.01, 0.02, 0.05]  # the lower limit below leaves room for the legend under the points
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y:g}" for y in yt])
    ax.set_xticks(range(3))
    ax.set_xticklabels(TILES)
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("fitted D (m²/yr; log scale: each tick is a stated value)")
    ax.set_title("Is the fitted D physical? Fitted D against Heimsath's D and the relief D of each tile")
    ax.legend(handles=[handle(n) for n in fm] + [Line2D([], [], color="k", ls="--", label="Heimsath's measured D"), Line2D([], [], color="0.35", ls="-.", label="relief D of the tile (D that linear diffusion needs for the measured hillslope relief at rate U)")],
              loc="lower center", ncol=2, fontsize=6.5)
    ax.set_ylim(1e-7, 0.08)
    fig.suptitle("Are the fitted K and D physical?", y=1.0)
    fig.tight_layout()
    save(fig, "fig8_K_D_physical.png",
         "Left: each model's fitted K divided by the measured erosion-rate-based K, one point per tile; the shaded band is within a factor of two of the measured value (log scale: 2 means twice as large). Right: the fitted D on a log axis (tick marks carry the values), with Heimsath's measured D as the dashed line and each tile's relief D as a short dash-dot line. Fits use the misfit inside the M&D study area on TTV1–TTV3.",
         "K ÷ measured: " + "; ".join(PLAIN[n] + " " + " / ".join(format(F[t]["models"][n]["K_over_K0"], ".2f") for t in TILES) for n in fm) + ". D ÷ relief D: " +
         "; ".join(PLAIN[n] + " " + " / ".join(format(F[t]["models"][n]["D"] / F[t]["D_relief"], ".2f") for t in TILES) for n in fm) + ".")


def main() -> None:
    t0 = time.time()
    M, R, ADC = load_all()
    F = M["fits"]
    PM = M["metrics"]["pooled"]["primary"]
    fig1(F)
    fig2(PM)
    fig3(PM)
    fig4(PM)
    fig5(PM)
    fig6(M["metrics"])
    if os.path.exists(out_path("profiles", f"AD_{ADC.get('tile', 'TTV2')}_products.npz")) and "n_extra" in ADC:
        fig7(ADC, M["metrics"])
    fig8(F)
    dump(CAPTIONS, out_path("post", "captions.json"))
    print(f"figs: {len(CAPTIONS)} written to {FIG_DIR} [{time.time() - t0:.0f} s]", flush=True)
    _ = (R, FIXED_MODELS)


if __name__ == "__main__":
    main()
