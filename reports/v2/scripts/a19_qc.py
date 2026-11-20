#!/usr/bin/env python3
"""Addendum 19 figures: section 2.3 (hilltop QC in the addendum 18 section 3.2 style, classes relative to the 10^3 network plus the two new
filter classes; pair figures; galleries; the six-tile rollover-against-10^3 comparison per site; TTV5 with the road layer), 2.5.3 (|C_ht|
against L with Perron's line), 2.6 (sigma_C(L) with the two-segment fit; the D0_N(L) galleries), 2.7.4 (the D_C / D_R interval figure with the
addendum 18 fitted D of AD, P4 and P3), 3.2 (the area-slope figure). PNG at 200 dpi, each under 5 MB, under reports/v2/figures/a19/.
usage: a19_qc.py --part tiles|pairs|gallery|compare|ttv5|curves|intervals|heads|readme [--sites GM,AP,TTV]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_hilltop_qc import best_box, box_to_other  # noqa: E402
from a19_common import (  # noqa: E402
    A18_SUMMARY,
    ALL_HALF_WIDTHS,
    E_SITE,
    FIG_DIR,
    L_F_M,
    MD_EQ10,
    MEASURED_TILES,
    NO_D0,
    PERRON_2012,
    PRIMARY,
    QC_DIR,
    SCAN_HALF_WIDTHS,
    SURVEY_PAIRS,
    TIER2,
    W_F,
    dump,
    hillshade,
    load_json,
    md_table1,
    metre_tile,
    out_path,
    site_of,
    tile_registry,
)

from lemcvt.config import data_root  # noqa: E402
from lemcvt.hydro.d8 import d8_divides  # noqa: E402
from lemcvt.io.grids import read_geotiff  # noqa: E402

DPI = 200
MAX_BYTES = 5_000_000
ZOOM_M = 200
ALPHA = 0.6
# Okabe-Ito; the a18 colours for the four shared classes, two new ones
CLASSES = {1: ("channel: A ≥ 10³ m² (the fixed network)", "#0072B2"), 2: ("divide dropped: within L_f/2 of the network", "#E69F00"),
           3: ("counted hilltop (D₀,N)", "#D55E00"), 4: ("divide dropped: not between two basins, edge, |∇z| > 0.4, or non-convex", "#999999"),
           5: ("dropped by the segment filter (< 50 px)", "#CC79A7"), 6: ("dropped by the road/trail filter (10 m)", "#009E73")}
ROMAN = {1: "i", 2: "ii", 3: "iii", 4: "iv", 5: "vi", 6: "vii"}


def survey_of(tile):
    reg = tile_registry()
    return reg[tile]["survey"] if tile in reg else ("PAMAP 2006-08" if tile.startswith("AP") else "NCALM 2003")


def load_tile_item(tile: str):
    td = metre_tile(tile)
    z = np.load(out_path("hilltops", f"{tile}_masks.npz"))
    rec = load_json(out_path("hilltops", f"{tile}.json"))
    Z, FD = td.Z, td.FD
    channel = z["channel_1e3"]
    mask0 = z["divide_between_basins"]
    geom, roadfree, segfilt, counted = z["geom"], z["roadfree"], z["segfilt"], z["counted"]
    divides = d8_divides(FD).astype(bool)
    cls = np.zeros(Z.shape, dtype=np.uint8)
    cls[channel] = 1
    cls[mask0 & ~geom & ~channel] = 2                      # dropped by the distance rule (and the slope rule inside mask0's own filter: see hilltop_mask)
    cls[divides & ~channel & ~mask0] = 4
    cls[segfilt & ~counted] = 4                            # non-convex at L_f
    cls[geom & ~roadfree] = 6
    cls[roadfree & ~segfilt] = 5
    cls[counted] = 3
    from lemcvt.measure.hilltop_curvature import total_curvature
    curv = -total_curvature(Z, W_F, td.dx, td.filled.dy)
    fill = np.zeros(Z.shape, dtype=bool)
    reg = tile_registry().get(tile)
    if reg:
        for k, p in reg["masks"].items():
            if k in ("nnfill_mask", "gapfill_mask"):
                g = read_geotiff(data_root() / p).data
                fill |= np.nan_to_num(g, nan=0) >= 1
    return {"tile": tile, "td": td, "Z": Z, "hs": hillshade(Z, td.dx), "cls": cls, "curv": curv.astype(np.float32), "fill": fill, "mask": counted, "rollover": z["rollover_set"],
            "road": z["road_mask"], "rec": rec, "stats": {"n_class": {k: int((cls == k).sum()) for k in CLASSES}, "survey": survey_of(tile)}}


def draw_classes(ax, item, r0=0, c0=0, size=None, show_fill=True, title=None):
    hs, cls, fill = item["hs"], item["cls"], item["fill"]
    size_r, size_c = hs.shape if size is None else (size, size)
    sl = (slice(r0, r0 + size_r), slice(c0, c0 + size_c))
    ax.imshow(np.nan_to_num(hs[sl], nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cmap = ListedColormap([CLASSES[k][1] for k in sorted(CLASSES)])
    c = cls[sl].astype(float)
    c[c == 0] = np.nan
    ax.imshow(c - 1, cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5, alpha=ALPHA, interpolation="nearest")
    if show_fill and fill[sl].any():
        ax.contourf(fill[sl].astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"], linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def legend_handles():
    hs = [Patch(facecolor=CLASSES[k][1], alpha=ALPHA, label=f"({ROMAN[k]}) {CLASSES[k][0]}") for k in sorted(CLASSES)]
    hs.append(Patch(facecolor="none", edgecolor="k", hatch="////", label="nnfill_mask or gapfill_mask (filled cells)"))
    hs.append(Patch(facecolor="white", edgecolor="grey", label="(v) not part of the measurement"))
    return hs


def g(d, k, fmt):
    v = d.get(k) if isinstance(d, dict) else None
    return format(v, fmt) if isinstance(v, (int, float)) and v is not None else "-"


def title_of(item) -> str:
    r = item["rec"]
    d0 = r["D0_N"]
    A, B = d0.get("A", {}), d0.get("B", {})
    wt = d0.get("window_test_B", {})
    nod0 = "" if d0["usable"] else "; no D₀,N (< 500 px)"
    t2 = " [Tier 2]" if item["tile"] in TIER2 else ""
    return (f"{item['tile']}{t2} ({item['stats']['survey']}): 10³ m² network; counted n = {d0['n_counted']:,} (geom {r['set']['n_geom']:,}, road −{r['set']['n_removed_road']}, "
            f"segment −{r['set']['n_removed_segment']}, non-convex −{r['set']['n_removed_nonconvex']}); crest width to 10³ {g(B, 'crest_width_m', '.0f')} m (rollover "
            f"{g(d0.get('B_prime_rollover_width', {}), 'crest_width_m', '.0f')} m); C_ht(L_f) = {g(A, 'C_ht', '.4g')} m⁻¹, D₀,N,A = {g(A, 'D0', '.4g')}, "
            f"D₀,N,B = {g(B, 'D0', '.4g')} m²/yr (L_w {B.get('L_w_m')} m); window test {'pass' if wt.get('pass') else 'fail'}{' (trivial)' if wt.get('only_L_f_in_range') else ''}; "
            f"D₀,R,A = {g(r['part_b'], 'D0_A', '.4g')}{nod0}")


def save(fig, path, rec, scale_steps=(1.0, 0.85, 0.72, 0.6, 0.5)):
    for s in scale_steps:
        fig.savefig(path, dpi=int(DPI * s), bbox_inches="tight", facecolor="white")
        if os.path.getsize(path) <= MAX_BYTES:
            rec[os.path.basename(path)] = {"bytes": os.path.getsize(path), "dpi": int(DPI * s)}
            return
    rec[os.path.basename(path)] = {"bytes": os.path.getsize(path), "dpi": int(DPI * scale_steps[-1]), "warning": "over 5 MB"}


def tile_figure(item, site_scale, out, rec):
    tile, mask = item["tile"], item["mask"]
    r0, c0 = best_box(mask, ZOOM_M) if mask.any() else (400, 400)
    fig, axes = plt.subplots(2, 2, figsize=(13, 12.5))
    draw_classes(axes[0, 0], item, title="(a) the full tile (1 km) with the pixel classes; box = the 200 m zoom of (b)-(c)")
    axes[0, 0].add_patch(Rectangle((c0, r0), ZOOM_M, ZOOM_M, fill=False, edgecolor="k", lw=1.2))
    draw_classes(axes[0, 1], item, r0, c0, ZOOM_M, title="(b) 200 × 200 m zoom with the pixel classes (the densest counted 200 m box)")
    ax = axes[1, 0]
    sl = (slice(r0, r0 + ZOOM_M), slice(c0, c0 + ZOOM_M))
    ax.imshow(np.nan_to_num(item["hs"][sl], nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cz = np.where(mask[sl], item["curv"][sl], np.nan)
    im = ax.imshow(cz, cmap="viridis", vmin=site_scale[0], vmax=site_scale[1], interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(c) the same zoom: −∇²z at L_f = 11 m on the counted pixels (site scale {site_scale[0]:.4g}-{site_scale[1]:.4g} m⁻¹)", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("−∇²z (m⁻¹)", fontsize=7)
    ax = axes[1, 1]
    vals = item["curv"][mask]
    vals = vals[np.isfinite(vals)]
    cht = item["rec"]["D0_N"].get("A", {}).get("C_ht")
    if vals.size:
        lo, hi = np.percentile(vals, [0.5, 99.5])
        ax.hist(vals, bins=80, range=(min(lo, 0), hi), color=CLASSES[3][1], alpha=0.8, label="counted (10³ network)")
    rv = item["curv"][item["rollover"]]
    rv = rv[np.isfinite(rv)]
    if rv.size:
        lo2, hi2 = np.percentile(rv, [0.5, 99.5])
        ax.hist(rv, bins=80, range=(min(lo2, 0), hi2), histtype="step", color="k", lw=0.8, label=f"rollover set (Part B), n = {rv.size:,}")
    if cht:
        ax.axvline(cht, color=CLASSES[3][1], lw=1.4, label=f"C_ht(L_f) counted = {cht:.4g}")
    pb = item["rec"]["part_b"]["C_ht_Lf"]
    if pb:
        ax.axvline(pb, color="k", lw=1.0, ls="--", label=f"C_ht(L_f) rollover = {pb:.4g}")
    ax.axvline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("−∇²z at L_f (m⁻¹)", fontsize=8)
    ax.set_ylabel("pixels", fontsize=8)
    ax.set_title(f"(d) histogram of the counted curvature, n = {vals.size:,}", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.suptitle(title_of(item), fontsize=7)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, os.path.join(out, f"{tile}_hilltop_qc.png"), rec)
    plt.close(fig)
    return {"zoom_box_rc": [int(r0), int(c0)], "zoom_m": ZOOM_M}


def pair_figure(site, old, new, items, sp, out, rec):
    io, inw = items[old], items[new]
    r0, c0 = best_box(io["mask"], ZOOM_M) if io["mask"].any() else (400, 400)
    r1, c1 = box_to_other(io, r0, c0, inw)
    fig, axes = plt.subplots(3, 2, figsize=(13, 18))
    for j, (it, rr, cc) in enumerate(((io, r0, c0), (inw, r1, c1))):
        draw_classes(axes[0, j], it, title=f"(a) {it['tile']} ({it['stats']['survey']}): full tile; counted n = {it['rec']['D0_N']['n_counted']:,}")
        axes[0, j].add_patch(Rectangle((cc, rr), ZOOM_M, ZOOM_M, fill=False, edgecolor="k", lw=1.2))
        draw_classes(axes[1, j], it, rr, cc, ZOOM_M, title=f"(b) {it['tile']}: the same 200 m box on both surveys")
    from a18_prep import _map_set
    r_on, c_on, _ = _map_set(io["mask"], io["td"], inw["td"], L_F_M)
    old_on_new = np.zeros(inw["Z"].shape, dtype=bool)
    old_on_new[r_on, c_on] = True
    both = old_on_new & inw["mask"]
    ax = axes[2, 0]
    sl = (slice(r1, r1 + ZOOM_M), slice(c1, c1 + ZOOM_M))
    ax.imshow(np.nan_to_num(inw["hs"][sl], nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ov = np.full(inw["Z"].shape, np.nan)
    ov[old_on_new & ~inw["mask"]] = 0
    ov[inw["mask"] & ~old_on_new] = 1
    ov[both] = 2
    ax.imshow(ov[sl], cmap=ListedColormap(["#56B4E9", "#D55E00", "#000000"]), vmin=-0.5, vmax=2.5, alpha=0.85, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(c) counted sets on the {inw['stats']['survey']} hillshade: {old} only (blue), {new} only (orange), both (black); "
                 f"{int((old_on_new & ~inw['mask']).sum()):,} / {int((inw['mask'] & ~old_on_new).sum()):,} / {int(both.sum()):,}", fontsize=7)
    ax = axes[2, 1]
    for it, col in ((io, "#56B4E9"), (inw, "#D55E00")):
        v = it["curv"][it["mask"]]
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(v, bins=80, range=(min(np.percentile(v, 0.5), 0), np.percentile(v, 99.5)), histtype="step", color=col, lw=1.2, label=f"{it['tile']} ({it['stats']['survey']}), n = {v.size:,}")
            ch = it["rec"]["D0_N"].get("A", {}).get("C_ht")
            if ch:
                ax.axvline(ch, color=col, lw=1, ls="--")
    ax.legend(fontsize=7)
    ax.set_xlabel("−∇²z at L_f (m⁻¹)", fontsize=8)
    ax.tick_params(labelsize=7)
    cap = ""
    if sp:
        p = sp["pairs"].get(f"{old}/{new}")
        if p and p.get("a"):
            cap = f"  same-pixel (a) {p['a']:.4g} (b) {p['b']:.4g} (c) {p['c']:.4g} (d) {p['d']:.4g}; surface {p['surface_ratios'][0]:.2f}, {p['surface_ratios'][1]:.2f}; selection {p['selection_ratios'][0]:.2f}, {p['selection_ratios'][1]:.2f}"
    ax.set_title("(d) counted curvature at L_f on both surveys" + cap, fontsize=6)
    fig.suptitle(f"{site} survey pair {old} ({io['stats']['survey']}) / {new} ({inw['stats']['survey']}): D₀,N counted sets on the 10³ m² network", fontsize=8)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.003))
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    save(fig, os.path.join(out, f"{old}_{new}_hilltop_pair.png"), rec)
    plt.close(fig)


def gallery(site, items, out, rec, ncols=6, per_page=18):
    tiles = sorted([t for t in PRIMARY[site] if t in items], key=lambda t: items[t]["rec"]["D0_N"]["n_counted"])
    pages = [tiles[i:i + per_page] for i in range(0, len(tiles), per_page)]
    for k, page in enumerate(pages, 1):
        nrows = int(np.ceil(len(page) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows + 0.8))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, t in zip(axes.ravel(), page, strict=False):
            it = items[t]
            ax.axis("on")
            d0 = it["rec"]["D0_N"]
            draw_classes(ax, it, show_fill=False, title=f"{t}{' [T2]' if t in TIER2 else ''}: n {d0['n_counted']:,}{'' if d0['usable'] else ' (no D₀)'}; C_ht {d0.get('A', {}).get('C_ht', float('nan')):.3g}; D₀,N,A {d0.get('A', {}).get('D0', float('nan')):.3g}")
        fig.suptitle(f"{site} primary tiles ({len(tiles)}), page {k}/{len(pages)}, ordered by counted pixel count: the pixel classes on the 10³ m² network", fontsize=9)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        save(fig, os.path.join(out, f"{site}_hilltop_gallery_p{k}.png"), rec)
        plt.close(fig)


def compare_page(site, items, partA, out, rec):
    """Six tiles by rank of D0_N/D0_R (lowest two, middle two, highest two): the rollover set against the 10^3 counted set."""
    tl = [t for t in PRIMARY[site] if t in items and partA["tiles"][t].get("N_over_R_A")]
    tl = sorted(tl, key=lambda t: partA["tiles"][t]["N_over_R_A"])
    if len(tl) >= 6:
        mid = len(tl) // 2
        pick = tl[:2] + tl[mid - 1:mid + 1] + tl[-2:]
    else:
        pick = tl
    fig, axes = plt.subplots(2, len(pick), figsize=(3.4 * len(pick), 7.6))
    axes = np.atleast_2d(axes)
    for j, t in enumerate(pick):
        it = items[t]
        rr, cc = best_box(it["mask"] | it["rollover"], 300) if (it["mask"] | it["rollover"]).any() else (350, 350)
        for i, zoom in enumerate((None, 300)):
            ax = axes[i, j]
            r0, c0, sz = (0, 0, it["hs"].shape[0]) if zoom is None else (rr, cc, zoom)
            sl = (slice(r0, r0 + sz), slice(c0, c0 + sz))
            ax.imshow(np.nan_to_num(it["hs"][sl], nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ov = np.full(it["Z"].shape, np.nan)
            ov[it["rollover"] & ~it["mask"]] = 0
            ov[it["mask"] & ~it["rollover"]] = 1
            ov[it["mask"] & it["rollover"]] = 2
            ax.imshow(ov[sl], cmap=ListedColormap(["#56B4E9", "#D55E00", "#000000"]), vmin=-0.5, vmax=2.5, alpha=0.9, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if zoom is None:
                ax.add_patch(Rectangle((cc, rr), 300, 300, fill=False, edgecolor="k", lw=1))
                x = partA["tiles"][t]
                ax.set_title(f"{t}: D₀,N/D₀,R = {x['N_over_R_A']:.2f}\nrollover n {int(it['rollover'].sum()):,} (A_c {x['A_c_rollover']:.0f} m²), 10³ counted n {x['n_counted']:,}", fontsize=7)
            else:
                ax.set_title("300 m zoom", fontsize=7)
    fig.legend(handles=[Patch(facecolor="#56B4E9", label="rollover set only (Part B, D₀,R)"), Patch(facecolor="#D55E00", label="10³ counted set only (D₀,N)"), Patch(facecolor="k", label="both")],
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{site}: the rollover (Part B) hilltop set against the 10³ m² counted set on six tiles by rank of D₀,N,A/D₀,R,A (lowest two, middle two, highest two)", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(fig, os.path.join(out, f"{site}_rollover_vs_1e3_compare.png"), rec)
    plt.close(fig)
    return pick


def ttv5_roads(items, out, rec):
    it = items["TTV5"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2))
    draw_classes(axes[0], it, title="(a) TTV5 with the pixel classes on the 10³ network")
    ax = axes[1]
    ax.imshow(np.nan_to_num(it["hs"], nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    road = it["road"].astype(float)
    road[road == 0] = np.nan
    ax.imshow(road, cmap=ListedColormap(["#009E73"]), alpha=0.5, interpolation="nearest")
    cnt = np.where(it["mask"], 1.0, np.nan)
    ax.imshow(cnt, cmap=ListedColormap(["#D55E00"]), alpha=0.9, interpolation="nearest")
    rl = np.where(it["rollover"] & ~it["mask"], 1.0, np.nan)
    ax.imshow(rl, cmap=ListedColormap(["#56B4E9"]), alpha=0.7, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    n_road_roll = int((it["rollover"] & it["road"]).sum())
    ax.set_title(f"(b) TIGER roads/trails 10 m buffer (green), the counted set (orange), the rollover set outside it (blue); rollover pixels inside the buffer: {n_road_roll:,} of {int(it['rollover'].sum()):,}", fontsize=7)
    fig.suptitle(title_of(it), fontsize=7)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, os.path.join(out, "TTV5_roads.png"), rec)
    plt.close(fig)


def curves_figures(partA, out, rec):
    # 2.5.3 |C_ht| against L with Perron's line; 2.6.1 sigma_C(L); 2.6.3 D0_N(L) galleries
    for site in ("GM", "AP"):
        p = partA["perron"][site]
        L = [p["by_window"][str(w)]["L_nominal_m"] for w in ALL_HALF_WIDTHS]
        m = [p["by_window"][str(w)]["median"] for w in ALL_HALF_WIDTHS]
        lo = [p["by_window"][str(w)]["ci68"][0] if p["by_window"][str(w)].get("ci68") else np.nan for w in ALL_HALF_WIDTHS]
        hi = [p["by_window"][str(w)]["ci68"][1] if p["by_window"][str(w)].get("ci68") else np.nan for w in ALL_HALF_WIDTHS]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for t in p["tiles"]:
            r = load_json(out_path("hilltops", f"{t}.json"))
            if r["D0_N"]["usable"]:
                ax.plot([r["curve_counted"][str(w)]["L_nominal_m"] for w in ALL_HALF_WIDTHS], [r["curve_counted"][str(w)]["median"] for w in ALL_HALF_WIDTHS], color="grey", lw=0.5, alpha=0.5)
        ax.fill_between(L, lo, hi, color="#D55E00", alpha=0.25, label="pooled median, 68 % segment bootstrap")
        ax.plot(L, m, color="#D55E00", lw=2, label="pooled |C_ht|(L), primary tiles")
        ax.axhline(PERRON_2012[site]["curvature_per_m"], color="k", ls="--", label=f"Perron et al. (2012): {PERRON_2012[site]['curvature_per_m']:.4g} m⁻¹")
        ax.axhspan(0.67 * PERRON_2012[site]["curvature_per_m"], 1.5 * PERRON_2012[site]["curvature_per_m"], color="k", alpha=0.07, label="R11 range 0.67-1.5 ×")
        ax.axvline(L_F_M, color="#0072B2", ls=":", label="L_f = 11 m")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("window L (m)")
        ax.set_ylabel("|C_ht| (m⁻¹)")
        ax.set_title(f"{site}: pooled hilltop curvature of the D₀,N set against the window (grey: tiles)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, os.path.join(out, f"{site}_Cht_vs_L_perron.png"), rec)
        plt.close(fig)
    for site in ("GM", "AP", "TTV"):
        sb = partA["scaling_break"][site]
        L = np.array(sb["L_nominal_m"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for c in sb["sigma_curves_normalised"].values():
            ax.plot(L, c, color="grey", lw=0.5, alpha=0.6)
        ax.plot(L, sb["pooled_median_curve"], color="#D55E00", lw=2, label="median of tiles")
        f = sb["fit"]
        xs = np.log10(np.linspace(L.min(), L.max(), 200))
        ys = f["a"] + np.where(xs <= f["xb"], f["b1"] * (xs - f["xb"]), f["b2"] * (xs - f["xb"]))
        ax.plot(10 ** xs, 10 ** ys, color="k", lw=1.5, label=f"two-segment fit: break {sb['L_break_m']:.1f} m [{sb['L_break_ci68'][0]:.1f}, {sb['L_break_ci68'][1]:.1f}], slopes {f['b1']:.2f} / {f['b2']:.2f}")
        ax.axvline(L_F_M, color="#0072B2", ls=":", label="L_f = 11 m")
        ax.axvline(sb["L_S_m"], color="#009E73", ls="--", label=f"L_S = {sb['L_S_m']} m (nearest odd window)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("window L (m)")
        ax.set_ylabel("σ_C(L) / σ_C(11 m)")
        ax.set_title(f"{site}: curvature standard deviation against the window, primary tiles (decision 11)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, os.path.join(out, f"{site}_sigmaC_scaling_break.png"), rec)
        plt.close(fig)
        # D0_N(L) gallery
        tl = [t for t in PRIMARY[site] if t in partA["tiles"]]
        ncols = 6
        nrows = int(np.ceil(len(tl) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.3 * nrows + 0.6), sharex=True)
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, t in zip(axes.ravel(), tl, strict=False):
            ax.axis("on")
            r = load_json(out_path("hilltops", f"{t}.json"))
            x = partA["tiles"][t]
            cv = r["curve_counted"]
            Ls = [cv[str(w)]["L_nominal_m"] for w in ALL_HALF_WIDTHS]
            E = E_SITE[site]
            d = [E / cv[str(w)]["median"] if cv[str(w)]["median"] and cv[str(w)]["median"] > 0 else np.nan for w in ALL_HALF_WIDTHS]
            lo = [E / cv[str(w)]["ci68"][1] if cv[str(w)].get("ci68") and cv[str(w)]["ci68"][0] > 0 else np.nan for w in ALL_HALF_WIDTHS]
            hi = [E / cv[str(w)]["ci68"][0] if cv[str(w)].get("ci68") and cv[str(w)]["ci68"][0] > 0 else np.nan for w in ALL_HALF_WIDTHS]
            ax.fill_between(Ls, lo, hi, color="#D55E00", alpha=0.25)
            ax.plot(Ls, d, color="#D55E00", lw=1.2)
            ax.axvline(L_F_M, color="#0072B2", ls=":", lw=0.8)
            if x.get("L_w_m"):
                ax.axvline(x["L_w_m"], color="k", ls="--", lw=0.8)
            if x.get("L_w_prime_m"):
                ax.axvline(x["L_w_prime_m"], color="grey", ls="--", lw=0.8)
            if x.get("D0_R_A"):
                ax.axhline(x["D0_R_A"], color="#56B4E9", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{t}: n {x['n_counted']:,}{'' if x['usable'] else ' (no D₀)'}; L_w {x.get('L_w_m')} / L_w′ {x.get('L_w_prime_m')} m", fontsize=6)
            ax.tick_params(labelsize=6)
        fig.suptitle(f"{site}: D₀,N(L) = E/C_ht(L) per primary tile (band: 68 %); dotted L_f, dashed black L_w (10³ width), dashed grey L_w′ (rollover width), blue D₀,R,A", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, os.path.join(out, f"{site}_D0N_vs_L_gallery.png"), rec)
        plt.close(fig)


def interval_figure(partA, out, rec):
    fits = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    for site in ("GM", "AP", "TTV"):
        tl = [t for t in PRIMARY[site] if t in partA["tiles"]]
        fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(tl) + 2), 5))
        xs = np.arange(len(tl))
        for i, t in enumerate(tl):
            x = partA["tiles"][t]
            if x.get("D0_N_A_ci68"):
                ax.plot([i - 0.15, i - 0.15], x["D0_N_A_ci68"], color="#0072B2", lw=3, solid_capstyle="butt")
                ax.plot(i - 0.15, x["D0_N_A"], "o", color="#0072B2", ms=3)
            if x.get("D_R_ci68"):
                ax.plot([i + 0.15, i + 0.15], x["D_R_ci68"], color="#D55E00", lw=3, solid_capstyle="butt")
                ax.plot(i + 0.15, x["D_R"], "o", color="#D55E00", ms=3)
            if x.get("bracket"):
                ax.add_patch(Rectangle((i - 0.3, x["bracket"][0]), 0.6, x["bracket"][1] - x["bracket"][0], facecolor="grey", alpha=0.15, edgecolor="none"))
            if x.get("D0_R_A"):
                ax.plot(i, x["D0_R_A"], "_", color="#56B4E9", ms=10, mew=1.5)
            f = fits.get(f"{t}:primary") or fits.get(f"{t}:offmesa")
            if f:
                for name, mk, col in (("AD", "s", "k"), ("P4", "D", "#009E73"), ("P3", "^", "#CC79A7")):
                    if f.get(name):
                        ax.plot(i, f[name]["D"], mk, color=col, ms=5, mfc="none" if f[name].get("where") in ("y_lo", "y_hi") or f[name].get("D_at_bound") else col)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(tl, rotation=90, fontsize=7)
        ax.set_ylabel("D (m²/yr)")
        ax.set_title(f"{site}: the D bracket per primary tile: D_C = D₀,N,A (blue, 68 %) and D_R (orange, 68 %); grey box = [D_C,lo, D_R,hi]; D₀,R,A (light blue tick); "
                     f"addendum 18 fitted D: AD (square), P4 (diamond), P3 (triangle); open = at a bound", fontsize=7)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color="#0072B2", lw=3, label="D_C = D₀,N,A"), Line2D([], [], color="#D55E00", lw=3, label="D_R (10³ network)"), Line2D([], [], color="#56B4E9", marker="_", ls="", label="D₀,R,A (Part B)"),
                           Line2D([], [], color="k", marker="s", ls="", label="AD"), Line2D([], [], color="#009E73", marker="D", ls="", label="P4"), Line2D([], [], color="#CC79A7", marker="^", ls="", label="P3")], fontsize=7, ncol=3)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        fig.tight_layout()
        save(fig, os.path.join(out, f"{site}_D_bracket_intervals.png"), rec)
        plt.close(fig)


def heads_figure(out, rec):
    h = load_json(out_path("heads", "heads.json"))
    t1 = md_table1()
    A1 = np.array([x["A"] for x in t1])
    S1 = np.array([x["S"] for x in t1])
    pk = h["picks"]
    A3 = np.array([x["A_3_m2"] for x in pk])
    A10 = np.array([x["A_10_m2"] for x in pk])
    Sup = np.array([x["S_up"] for x in pk])
    Sdn = np.array([x["S_down"] for x in pk])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(S1, A1, marker="o", s=22, facecolors="none", edgecolors="k", label="M&D (1989) Table 1 (63)")
    ax.scatter(Sup, A3, marker="^", s=22, color="#D55E00", label="picks: A_3, S_up (64)")
    ax.scatter(Sdn, A3, marker="v", s=22, color="#0072B2", label="picks: A_3, S_down (64)")
    ax.scatter(Sup, A10, marker="+", s=18, color="grey", label="picks: A_10, S_up (sensitivity)")
    i49 = [k for k, x in enumerate(pk) if x["pick_id"] == 49][0]
    ax.annotate("49", (Sup[i49], A3[i49]), fontsize=7)
    s = np.logspace(np.log10(0.15), np.log10(1.0), 50)
    ax.plot(s, MD_EQ10["a_m2"] * s ** (-MD_EQ10["b"]), "k-", lw=1.5, label="M&D eq. 10: A = 1978 tanθ^−1.65")
    for key, col, lab in (("i_table1", "k", "(i) Table 1"), ("ii_picks_A3_Sup", "#D55E00", "(ii) picks A_3-S_up"), ("iii_picks_A3_Sdown", "#0072B2", "(iii) picks A_3-S_down")):
        f = h["fits"][key]
        ax.plot(s, f["a"] * s ** (-f["b"]), ls="--", color=col, lw=1, label=f"{lab}: a = {f['a']:.0f} m², b = {f['b']:.2f}, R² = {f['R2']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("slope S (m/m)")
    ax.set_ylabel("source area A (m²)")
    ax.set_title("Channel-head area against slope: M&D (1989) Table 1 and Steve's 64 picks on the Marin 2019 1 m D8 routing", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout()
    save(fig, os.path.join(out, "ttv_heads_area_slope.png"), rec)
    plt.close(fig)


def write_readme(out, rec_all):
    lines = ["# Addendum 19 figures (reports/v2/figures/a19)", "", "Hilltop QC (section 2.3) under `hilltop_qc/`; classes relative to the 10³ m² network:", ""]
    for k in sorted(CLASSES):
        lines.append(f"- ({ROMAN[k]}) {CLASSES[k][0]}")
    lines += ["- (v) everything else, uncoloured; hatch = nnfill_mask or gapfill_mask", "", "Files:", ""]
    for name, r in sorted(rec_all.items()):
        lines.append(f"- `{name}` ({r['bytes'] / 1e6:.2f} MB, {r['dpi']} dpi){' **' + r['warning'] + '**' if r.get('warning') else ''}")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True)
    p.add_argument("--sites", default="GM,AP,TTV")
    a = p.parse_args()
    t0 = time.time()
    os.makedirs(QC_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    rec_path = out_path("figures", "figures.json")
    rec_all = load_json(rec_path) if os.path.exists(rec_path) else {}
    rec: dict = {}
    sites = a.sites.split(",")
    if a.part in ("tiles", "pairs", "gallery", "compare", "ttv5"):
        partA = load_json(out_path("site", "partA.json")) if os.path.exists(out_path("site", "partA.json")) else None
        sp = load_json(out_path("samepixel", "samepixel.json")) if os.path.exists(out_path("samepixel", "samepixel.json")) else None
        if a.part == "tiles":
            zoom = {}
            for site in sites:
                tl = [t for t in MEASURED_TILES if site_of(t) == site]
                items = {t: load_tile_item(t) for t in tl}
                vals = np.concatenate([it["curv"][it["mask"]] for it in items.values() if it["mask"].any()])
                vals = vals[np.isfinite(vals)]
                scale = (float(np.percentile(vals, 2)), float(np.percentile(vals, 98)))
                for t, it in items.items():
                    zoom[t] = tile_figure(it, scale, QC_DIR, rec) | {"n_class": it["stats"]["n_class"]}
                    print(t, f"[{time.time() - t0:.0f} s]", flush=True)
            dump(zoom, out_path("figures", "tiles_qc.json"))
        elif a.part == "pairs":
            for site in sites:
                for old, new in SURVEY_PAIRS.get(site, []):
                    items = {old: load_tile_item(old), new: load_tile_item(new)}
                    pair_figure(site, old, new, items, sp, QC_DIR, rec)
                    print(old, new, f"[{time.time() - t0:.0f} s]", flush=True)
        elif a.part == "gallery":
            for site in sites:
                items = {t: load_tile_item(t) for t in PRIMARY[site]}
                gallery(site, items, QC_DIR, rec)
        elif a.part == "compare":
            picks = {}
            for site in sites:
                items = {t: load_tile_item(t) for t in PRIMARY[site]}
                picks[site] = compare_page(site, items, partA, QC_DIR, rec)
            dump(picks, out_path("figures", "compare_tiles.json"))
        elif a.part == "ttv5":
            ttv5_roads({"TTV5": load_tile_item("TTV5")}, QC_DIR, rec)
    elif a.part == "curves":
        curves_figures(load_json(out_path("site", "partA.json")), FIG_DIR, rec)
    elif a.part == "intervals":
        interval_figure(load_json(out_path("site", "partA.json")), FIG_DIR, rec)
    elif a.part == "heads":
        heads_figure(FIG_DIR, rec)
    elif a.part == "readme":
        write_readme(QC_DIR, {k: v for k, v in rec_all.items() if k.endswith(".png")})
    rec_all.update(rec)
    dump(rec_all, rec_path)
    print(a.part, "done:", len(rec), "figures", f"[{time.time() - t0:.0f} s]", json.dumps({k: v for k, v in rec.items() if v.get('warning')}))
    _ = (glob, SCAN_HALF_WIDTHS, NO_D0)


if __name__ == "__main__":
    main()
