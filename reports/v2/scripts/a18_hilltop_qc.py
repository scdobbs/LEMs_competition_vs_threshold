#!/usr/bin/env python3
"""Addendum 18, section 3.2: hilltop QC figures for Steve's visual check (and, after the fits, the model-partition figures of item 5).

Pixel classes (item 1), every cell in exactly one: (i) channel = the rollover network (A >= the tile's rollover A_c); (ii) excluded near
channel = divide cells of the measure hilltop set dropped because they lie within L_f/2 of a channel cell; (iii) counted hilltop = the
option (iii) set at L_f = 11 m (the pixels that give C_ht and D0); (iv) divide dropped for any other reason (not between two basins of
the rollover network, within two cells of the tile edge, or |grad z| > 0.4); (v) everything else, uncoloured. Cells of nnfill_mask or
gapfill_mask carry a thin hatch. Hillshade: azimuth 315, altitude 45, no vertical exaggeration, from the tile's own 1 m filled grid
(a17_common.hillshade). Colours (Okabe-Ito, colour-blind safe) at 60 % opacity, the same legend on every figure.
Figures: <tile>_hilltop_qc.png (item 2, four panels), <pair>_hilltop_pair.png (item 3), <site>_hilltop_gallery_p<k>.png (item 4),
<tile>_partition.png (item 5, with --partition after the fits), README.md (item 6). PNG at 200 dpi, each under 5 MB.
usage: a18_hilltop_qc.py --part tiles|pairs|gallery|readme|partition [--sites GM,AP,TTV]
"""
from __future__ import annotations

import argparse
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
from scipy.signal import fftconvolve  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_common import (  # noqa: E402
    AP1_3DEP,
    AP_PAMAP,
    D0_MIN_PIXELS,
    FITTED_AT_SITE_M,
    GM_NCALM,
    GM_OFFMESA,
    HILLSHADE,
    L_F_M,
    NO_D0,
    PRIMARY,
    QC_DIR,
    SURVEY,
    SURVEY_PAIRS,
    TIER2,
    W_F,
    dump,
    fnum,
    hillshade,
    partb_measurement,
    partb_per_tile,
    run_dir,
    site_of,
    slurm_record,
    tile_registry,
)
from a18_prep import _map_set  # noqa: E402

from lemcvt.config import data_root  # noqa: E402
from lemcvt.hydro.d8 import d8_divides  # noqa: E402
from lemcvt.io.grids import read_geotiff  # noqa: E402
from lemcvt.measure.hilltop_curvature import total_curvature  # noqa: E402
from lemcvt.measure.phase6_inputs import option_iii_mask  # noqa: E402
from lemcvt.measure.pipeline import _metre_tile  # noqa: E402

CLASSES = {1: ("channel (rollover network, A ≥ A_c)", "#0072B2"), 2: ("divide, excluded: within L_f/2 of a channel cell", "#E69F00"),
           3: ("counted hilltop (option (iii) at L_f = 11 m): gives C_ht and D₀", "#D55E00"),
           4: ("divide, dropped: not between two basins, within 2 cells of the edge, or |∇z| > 0.4", "#009E73")}
ALPHA = 0.6
ZOOM_M = 200
DPI = 200
MAX_BYTES = 5_000_000
OVERLAY = {"old": "#CC79A7", "new": "#56B4E9", "both": "#000000"}
PAIR_LABEL = {"GM": ("NCALM 2003", "FEMA 2018"), "AP": ("PAMAP", "3DEP 2019-20")}


def survey_of(tile: str) -> str:
    reg = tile_registry()
    if tile in reg:
        return reg[tile]["survey"]
    return "PAMAP 2006-08" if tile.startswith("AP") else "NCALM 2003"


def classify(tile: str, meas: dict):
    """Classes grid (uint8), the -curvature grid at L_f, the fill-mask grid, the stats of the tile."""
    td = _metre_tile(tile)
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    A_c = float(meas["tiles"][tile]["measure_run"]["A_c_rollover"])
    mask, mask0, dist, diag = option_iii_mask(Z, A, FD, dx, dy, A_c, W_F)
    channel = np.isfinite(A) & (A >= A_c)
    divides = d8_divides(FD).astype(bool)
    cls = np.zeros(Z.shape, dtype=np.uint8)
    cls[channel] = 1
    cls[mask0 & ~mask & ~channel] = 2
    cls[mask] = 3
    other = divides & ~channel & ~mask0
    cls[other] = 4
    edge = np.zeros(Z.shape, dtype=bool)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    curv = -total_curvature(Z, W_F, dx, dy)
    fill = np.zeros(Z.shape, dtype=bool)
    reg = tile_registry().get(tile)
    masks_used = []
    if reg:
        for k, p in reg["masks"].items():
            if k in ("nnfill_mask", "gapfill_mask"):
                g = read_geotiff(data_root() / p).data
                fill |= np.nan_to_num(g, nan=0) >= 1
                masks_used.append(k)
    vals = curv[mask]
    vals = vals[np.isfinite(vals)]
    stats = {"A_c_rollover": A_c, "n_class": {k: int((cls == k).sum()) for k in (1, 2, 3, 4)}, "n_other_divides_edge": int((other & edge).sum()),
             "n_removed_steep": diag.get("n_removed_steep"), "n_divide": diag.get("n_divide"), "n_divide_between_basins": diag.get("n_divide_between_basins"),
             "C_ht_Lf_recomputed": float(np.median(vals)) if vals.size else None, "fill_masks": masks_used, "n_fill": int(fill.sum()),
             "survey": survey_of(tile), "dx": dx, "dy": dy, "transform": list(td.filled.transform)[:6], "crs": str(td.filled.crs)}
    hs = hillshade(Z, dx)
    return {"cls": cls, "curv": curv.astype(np.float32), "fill": fill, "hs": hs, "Z": Z, "stats": stats, "td": td, "mask": mask}


def best_box(mask: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left (row, col) of the size x size box holding the most True cells."""
    k = np.ones((size, size))
    c = fftconvolve(mask.astype(float), k, mode="valid")
    j = int(np.argmax(c))
    r0, c0 = np.unravel_index(j, c.shape)
    return int(r0), int(c0)


def _draw_classes(ax, item, r0=0, c0=0, size=None, show_fill=True, title=None):
    hs, cls, fill = item["hs"], item["cls"], item["fill"]
    if size is None:
        size_r, size_c = hs.shape
    else:
        size_r = size_c = size
    sl = (slice(r0, r0 + size_r), slice(c0, c0 + size_c))
    ax.imshow(np.nan_to_num(hs[sl], nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cmap = ListedColormap([CLASSES[k][1] for k in (1, 2, 3, 4)])
    c = cls[sl].astype(float)
    c[c == 0] = np.nan
    ax.imshow(c - 1, cmap=cmap, vmin=-0.5, vmax=3.5, alpha=ALPHA, interpolation="nearest")
    if show_fill and fill[sl].any():
        ax.contourf(fill[sl].astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"], linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def legend_handles():
    hs = [Patch(facecolor=CLASSES[k][1], alpha=ALPHA, label=f"({'i' * 1 if k == 1 else 'ii' if k == 2 else 'iii' if k == 3 else 'iv'}) {CLASSES[k][0]}") for k in (1, 2, 3, 4)]
    hs.append(Patch(facecolor="none", edgecolor="k", hatch="////", label="nnfill_mask or gapfill_mask (filled cells)"))
    hs.append(Patch(facecolor="white", edgecolor="grey", label="(v) not part of the measurement"))
    return hs


def tile_title(tile: str, per: dict, st: dict) -> str:
    p = per[tile]
    n3 = st["n_class"][3]
    d0a, d0b = fnum(p["D0_A"]), fnum(p["D0_B"])
    wt = "window test: " + ("pass" if p["window_pass"] == "True" else "fail") + (" (trivial)" if p["only_Lf_in_range"] == "True" else "")
    nod0 = f"; no D₀ (< {D0_MIN_PIXELS} px)" if tile in NO_D0 else ""
    t2 = " [Tier 2]" if tile in TIER2 else ""
    return (f"{tile}{t2} ({st['survey']}): rollover A_c = {st['A_c_rollover']:.0f} m², class (iii) n = {n3:,}, median crest width {fnum(p['median_crest_width_m']) or float('nan'):.1f} m, "
            f"C_ht(L_f) = {fnum(p['C_ht_Lf']) or float('nan'):.4g} m⁻¹, D₀,A = {d0a if d0a else float('nan'):.4g}, D₀,B = {d0b if d0b else float('nan'):.4g} m²/yr; {wt}{nod0}")


def save(fig, path: str, rec: dict, scale_steps=(1.0, 0.85, 0.72, 0.6, 0.5)):
    for s in scale_steps:
        fig.savefig(path, dpi=int(DPI * s), bbox_inches="tight", facecolor="white")
        if os.path.getsize(path) <= MAX_BYTES:
            rec[os.path.basename(path)] = {"bytes": os.path.getsize(path), "dpi": int(DPI * s)}
            return
    rec[os.path.basename(path)] = {"bytes": os.path.getsize(path), "dpi": int(DPI * scale_steps[-1]), "warning": "over 5 MB"}


def tile_figure(tile: str, item, per: dict, site_scale: tuple[float, float], out: str, rec: dict):
    st = item["stats"]
    mask = item["mask"]
    r0, c0 = best_box(mask, ZOOM_M)
    fig, axes = plt.subplots(2, 2, figsize=(13, 12.5))
    _draw_classes(axes[0, 0], item, title="(a) the full tile (1 km) with the pixel classes; box = the 200 m zoom of (b)-(c)")
    axes[0, 0].add_patch(Rectangle((c0, r0), ZOOM_M, ZOOM_M, fill=False, edgecolor="k", lw=1.2))
    _draw_classes(axes[0, 1], item, r0, c0, ZOOM_M, title="(b) 200 × 200 m zoom with the pixel classes (box centred on the densest class (iii) 200 m box)")
    ax = axes[1, 0]
    sl = (slice(r0, r0 + ZOOM_M), slice(c0, c0 + ZOOM_M))
    ax.imshow(np.nan_to_num(item["hs"][sl], nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cz = np.where(mask[sl], item["curv"][sl], np.nan)
    im = ax.imshow(cz, cmap="viridis", vmin=site_scale[0], vmax=site_scale[1], interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(c) the same zoom: −∇²z at L_f = 11 m on the class (iii) pixels (site scale: 2nd-98th percentiles {site_scale[0]:.4g}-{site_scale[1]:.4g} m⁻¹)", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("−∇²z (m⁻¹)", fontsize=7)
    ax = axes[1, 1]
    vals = item["curv"][mask]
    vals = vals[np.isfinite(vals)]
    cht = fnum(per[tile]["C_ht_Lf"])
    if vals.size:
        lo, hi = np.percentile(vals, [0.5, 99.5])
        ax.hist(vals, bins=80, range=(min(lo, 0), hi), color=CLASSES[3][1], alpha=0.8)
    if cht:
        ax.axvline(cht, color="k", lw=1.2, label=f"C_ht(L_f) = {cht:.4g} m⁻¹ (median)")
    ax.axvline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("−∇²z at L_f on the class (iii) pixels (m⁻¹)", fontsize=8)
    ax.set_ylabel("pixels", fontsize=8)
    ax.set_title(f"(d) histogram of the class (iii) curvature, n = {vals.size:,}", fontsize=7)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.suptitle(tile_title(tile, per, st), fontsize=8)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, os.path.join(out, f"{tile}_hilltop_qc.png"), rec)
    plt.close(fig)
    return {"zoom_box_rc": [r0, c0], "zoom_m": ZOOM_M}


def box_to_other(item_src, r0, c0, item_dst):
    """The map-coordinate box of (r0, c0, ZOOM_M) on item_src's lattice, as (row, col) on item_dst's lattice (nearest cell of the corner)."""
    from rasterio.warp import transform as rio_transform

    tr_s, tr_d = item_src["td"].filled.transform, item_dst["td"].filled.transform
    x = tr_s.c + c0 * tr_s.a
    y = tr_s.f + r0 * tr_s.e
    crs_s, crs_d = item_src["td"].filled.crs, item_dst["td"].filled.crs
    if crs_s is not None and crs_d is not None and crs_s != crs_d:
        xd, yd = rio_transform(crs_s, crs_d, [x], [y])
        x, y = xd[0], yd[0]
    c = int(round((x - tr_d.c) / tr_d.a))
    r = int(round((y - tr_d.f) / tr_d.e))
    ny, nx = item_dst["Z"].shape
    return min(max(r, 0), ny - ZOOM_M), min(max(c, 0), nx - ZOOM_M)


def pair_figure(site: str, old: str, new: str, items: dict, per: dict, sp: dict | None, out: str, rec: dict):
    io, inw = items[old], items[new]
    r0, c0 = best_box(io["mask"], ZOOM_M)
    r1, c1 = box_to_other(io, r0, c0, inw)
    fig, axes = plt.subplots(4, 2, figsize=(13, 25))
    lab_o, lab_n = PAIR_LABEL[site]
    _draw_classes(axes[0, 0], io, title=f"(a) {old} ({lab_o}): full tile with the pixel classes")
    axes[0, 0].add_patch(Rectangle((c0, r0), ZOOM_M, ZOOM_M, fill=False, edgecolor="k", lw=1.2))
    _draw_classes(axes[0, 1], inw, title=f"(a) {new} ({lab_n}): full tile with the pixel classes")
    axes[0, 1].add_patch(Rectangle((c1, r1), ZOOM_M, ZOOM_M, fill=False, edgecolor="k", lw=1.2))
    _draw_classes(axes[1, 0], io, r0, c0, ZOOM_M, title=f"(b) {old}: the 200 m zoom (box taken from the old survey)")
    _draw_classes(axes[1, 1], inw, r1, c1, ZOOM_M, title=f"(b) {new}: the same zoom on the new survey's lattice")
    # third row: the two class (iii) sets on the new survey's hillshade (old set mapped to the new lattice)
    ro, co, info = _map_set(io["mask"], io["td"], inw["td"], L_F_M)
    old_on_new = np.zeros(inw["Z"].shape, dtype=bool)
    old_on_new[ro, co] = True
    both = old_on_new & inw["mask"]
    ov = np.full(inw["Z"].shape, np.nan)
    ov[old_on_new & ~inw["mask"]] = 0
    ov[inw["mask"] & ~old_on_new] = 1
    ov[both] = 2
    cmap = ListedColormap([OVERLAY["old"], OVERLAY["new"], OVERLAY["both"]])
    for ax, (rr, cc, sz, ttl) in zip(axes[2], [(0, 0, None, "(c) full tile: class (iii) of both surveys on the new survey's hillshade"),
                                              (r1, c1, ZOOM_M, "(c) the zoom")], strict=True):
        sz_r = inw["Z"].shape[0] if sz is None else sz
        sl = (slice(rr, rr + sz_r), slice(cc, cc + sz_r))
        ax.imshow(np.nan_to_num(inw["hs"][sl], nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(ov[sl], cmap=cmap, vmin=-0.5, vmax=2.5, alpha=0.85, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl, fontsize=7)
    axes[2, 0].legend(handles=[Patch(facecolor=OVERLAY["old"], label=f"{lab_o} only (n = {int((old_on_new & ~inw['mask']).sum()):,})"),
                               Patch(facecolor=OVERLAY["new"], label=f"{lab_n} only (n = {int((inw['mask'] & ~old_on_new).sum()):,})"),
                               Patch(facecolor=OVERLAY["both"], label=f"both (n = {int(both.sum()):,})")], fontsize=7, loc="lower left")
    ax = axes[3, 0]
    vo = io["curv"][io["mask"]]
    vn = inw["curv"][inw["mask"]]
    vo, vn = vo[np.isfinite(vo)], vn[np.isfinite(vn)]
    allv = np.concatenate([vo, vn]) if vo.size and vn.size else (vo if vo.size else vn)
    lo, hi = np.percentile(allv, [0.5, 99.5]) if allv.size else (0, 0.05)
    ax.hist(vo, bins=80, range=(min(lo, 0), hi), color=OVERLAY["old"], alpha=0.6, label=f"{old} ({lab_o}), n = {vo.size:,}, median {np.median(vo):.4g}" if vo.size else old, density=True)
    ax.hist(vn, bins=80, range=(min(lo, 0), hi), color=OVERLAY["new"], alpha=0.6, label=f"{new} ({lab_n}), n = {vn.size:,}, median {np.median(vn):.4g}" if vn.size else new, density=True)
    ax.set_xlabel("−∇²z at L_f = 11 m on the class (iii) pixels (m⁻¹)", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("(d) class (iii) curvature of both surveys", fontsize=7)
    ax.tick_params(labelsize=7)
    axes[3, 1].axis("off")
    cap = ""
    if sp:
        cap = (f"Same-pixel diagnostic (3.1), C_ht(L_f) medians: (a) old surface on old pixels {sp['a']:.4g}; (b) new surface on old pixels {sp['b']:.4g}; "
               f"(c) old surface on new pixels {sp['c']:.4g}; (d) new surface on new pixels {sp['d']:.4g} m⁻¹.\nSurface ratios (b)/(a) = {sp['b'] / sp['a']:.3f}, (d)/(c) = {sp['d'] / sp['c']:.3f}; "
               f"selection ratios (c)/(a) = {sp['c'] / sp['a']:.3f}, (d)/(b) = {sp['d'] / sp['b']:.3f}; total (d)/(a) = {sp['d'] / sp['a']:.3f}.\n"
               f"Old pixels mapped onto the new lattice: {info['n_mapped']:,} of {info['n_source']:,} (median offset {info['offset_median_m']:.2f} m, max {info['offset_max_m']:.2f} m; "
               f"{info['n_dropped_edge']} dropped within L_f/2 + 1 m of the edge, {info['n_dropped_nodata']} on nodata).")
    axes[3, 1].text(0, 0.95, cap, fontsize=7.5, va="top", wrap=True, transform=axes[3, 1].transAxes)
    fig.suptitle(f"{old} / {new}: {lab_o} against {lab_n} on the same footprint; hillshade az {HILLSHADE['azimuth_deg']:.0f}°, alt {HILLSHADE['altitude_deg']:.0f}°, no exaggeration", fontsize=9)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.003))
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    save(fig, os.path.join(out, f"{old}_{new}_hilltop_pair.png"), rec)
    plt.close(fig)
    return {"zoom_old_rc": [r0, c0], "zoom_new_rc": [r1, c1], "mapping": info, "n_old_only": int((old_on_new & ~inw["mask"]).sum()), "n_new_only": int((inw["mask"] & ~old_on_new).sum()), "n_both": int(both.sum())}


def gallery(site: str, items: dict, per: dict, out: str, rec: dict, ncols=6, per_page=18):
    tiles = sorted(PRIMARY[site], key=lambda t: items[t]["stats"]["n_class"][3])
    pages = [tiles[i:i + per_page] for i in range(0, len(tiles), per_page)]
    names = []
    for k, page in enumerate(pages, 1):
        nrows = int(np.ceil(len(page) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.9 + 1.0))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, t in zip(axes.ravel(), page, strict=False):
            ax.axis("on")
            st = items[t]["stats"]
            p = per[t]
            _draw_classes(ax, items[t])
            d0 = fnum(p["D0_A"])
            ax.set_title(f"{t}{' [T2]' if t in TIER2 else ''}: A_c {st['A_c_rollover']:.0f} m², n(iii) {st['n_class'][3]:,}, W {fnum(p['median_crest_width_m']) or float('nan'):.0f} m\n"
                         f"C_ht {fnum(p['C_ht_Lf']) or float('nan'):.3g}, D₀,A {d0 if d0 else float('nan'):.3g}, D₀,B {fnum(p['D0_B']) or float('nan'):.3g}; "
                         f"{'pass' if p['window_pass'] == 'True' else 'fail'}{'(t)' if p['only_Lf_in_range'] == 'True' else ''}{'; no D₀' if t in NO_D0 else ''}", fontsize=5.5)
        fig.suptitle(f"{site} primary tiles, page {k} of {len(pages)}: panel (a) of every tile, ordered by the class (iii) pixel count (smallest first); every panel 1 × 1 km", fontsize=8)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=6.5, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        name = f"{site}_hilltop_gallery_p{k}.png"
        save(fig, os.path.join(out, name), rec)
        plt.close(fig)
        names.append(name)
    return {"pages": names, "order": tiles}


def partition_figure(tile: str, item, per: dict, A_crit: float, mse_rec: dict, out: str, rec: dict):
    st = item["stats"]
    r0, c0 = best_box(item["mask"], ZOOM_M)
    sl = (slice(r0, r0 + ZOOM_M), slice(c0, c0 + ZOOM_M))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    ax = axes[0]
    A = item["td"].A
    ax.imshow(np.nan_to_num(item["hs"][sl], nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ch = np.where(np.isfinite(A[sl]), (A[sl] >= A_crit).astype(float), np.nan)
    ax.imshow(ch, cmap=ListedColormap(["#F0E442", "#0072B2"]), vmin=-0.5, vmax=1.5, alpha=0.55, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(a) the P4 partition at the fitted A_crit = {A_crit:.0f} m² (zoom of item 2): channel A ≥ A_crit (blue) against hillslope (yellow)", fontsize=7)
    ax.legend(handles=[Patch(facecolor="#0072B2", alpha=0.55, label=f"channel, A ≥ A_crit (n in zoom = {int(np.nansum(ch == 1)):,})"),
                       Patch(facecolor="#F0E442", alpha=0.55, label=f"hillslope, A < A_crit (n = {int(np.nansum(ch == 0)):,})")], fontsize=7, loc="lower left")
    _draw_classes(axes[1], item, r0, c0, ZOOM_M, title="(b) the item 1 pixel classes of the hilltop measurement (same zoom)")
    fig.suptitle(f"{tile} ({st['survey']}): P4 A_crit {A_crit:.0f} m² (MSE {mse_rec.get('mse', float('nan')):.2f} m², K/K₀ {mse_rec.get('K_over_K0', float('nan')):.2f}); "
                 f"rollover A_c {st['A_c_rollover']:.0f} m², class (iii) n = {st['n_class'][3]:,}", fontsize=8)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, os.path.join(out, f"{tile}_partition.png"), rec)
    plt.close(fig)


def write_readme(out: str, rec_all: dict):
    lines = ["# Addendum 18, section 3.2: hilltop QC figures", "",
             f"Made by `reports/v2/scripts/a18_hilltop_qc.py` from the addendum 17 Part B `measure` runs (the option (iii) hilltop sets at L_f = {L_F_M:.0f} m, "
             "recomputed from each tile's rollover A_c with `lemcvt.measure.phase6_inputs.option_iii_mask`; no new measurement). PNG at 200 dpi (downscaled where a "
             f"file would exceed 5 MB); hillshade azimuth {HILLSHADE['azimuth_deg']:.0f}°, altitude {HILLSHADE['altitude_deg']:.0f}°, no vertical exaggeration, from the tile's "
             "own 1 m filled grid.", "", "## Colour legend (the same on every figure; Okabe-Ito colours at 60 % opacity over the hillshade)", ""]
    for k in (1, 2, 3, 4):
        lines.append(f"- ({'i' if k == 1 else 'ii' if k == 2 else 'iii' if k == 3 else 'iv'}) `{CLASSES[k][1]}`: {CLASSES[k][0]}")
    lines += ["- (v) uncoloured: not part of the measurement", "- thin `////` hatch: cells of `nnfill_mask` or `gapfill_mask` (NCALM and PAMAP tiles have no mask)",
              f"- survey-pair overlays: `{OVERLAY['old']}` old survey only, `{OVERLAY['new']}` new survey only, black both", "",
              "Zoom boxes: the 200 × 200 m box holding the most class (iii) pixels of the tile (on pair figures, the box of the old survey). Panel (c) of the per-tile "
              "figures uses one colour scale per site (the 2nd-98th percentiles of the site's class (iii) curvature values).", "", "## Figures", ""]
    for section, key in (("Per-tile QC (item 2)", "tiles"), ("Survey pairs (item 3)", "pairs"), ("Galleries (item 4)", "gallery"), ("Model partition (item 5, after the fits)", "partition")):
        files = sorted(f for f in rec_all.get(key, {}).get("files", {}))
        if not files:
            continue
        lines.append(f"### {section}")
        lines.append("")
        for f in files:
            lines.append(f"- `{f}`")
        lines.append("")
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True, choices=["tiles", "pairs", "gallery", "readme", "partition"])
    p.add_argument("--sites", default="GM,AP,TTV")
    a = p.parse_args()
    rd = run_dir()
    qd = os.path.join(rd, "hilltop_qc")
    os.makedirs(qd, exist_ok=True)
    os.makedirs(QC_DIR, exist_ok=True)
    slurm_record(qd)
    meas = partb_measurement()
    per = partb_per_tile()
    rec_path = os.path.join(qd, f"{a.part}.json")
    rec: dict = {"files": {}, "tiles": {}}
    t_all = time.time()
    sites = a.sites.split(",")
    if a.part == "tiles":
        for site in sites:
            tiles = [t for t in FITTED_AT_SITE_M if site_of(t) == site and t in meas["tiles"]]
            items = {}
            for t in tiles:
                t0 = time.time()
                items[t] = classify(t, meas)
                rec["tiles"][t] = items[t]["stats"] | {"classify_s": time.time() - t0}
                print(t, items[t]["stats"]["n_class"], f"{time.time() - t0:.0f} s", flush=True)
            allv = np.concatenate([items[t]["curv"][items[t]["mask"]] for t in tiles])
            allv = allv[np.isfinite(allv)]
            scale = (float(np.percentile(allv, 2)), float(np.percentile(allv, 98)))
            rec.setdefault("site_scale", {})[site] = scale
            for t in tiles:
                rec["tiles"][t].update(tile_figure(t, items[t], per, scale, QC_DIR, rec["files"]))
                items[t]["curv"] = None
            print(site, "done", flush=True)
    elif a.part == "pairs":
        sp_all = json.load(open(os.path.join(rd, "prep", "samepixel.json")))["pairs"] if os.path.exists(os.path.join(rd, "prep", "samepixel.json")) else {}
        for site in [s for s in sites if s in SURVEY_PAIRS]:
            for old, new in SURVEY_PAIRS[site]:
                items = {old: classify(old, meas), new: classify(new, meas)}
                rec["tiles"][f"{old}/{new}"] = pair_figure(site, old, new, items, per, sp_all.get(f"{old}/{new}"), QC_DIR, rec["files"])
                print(old, new, "done", flush=True)
    elif a.part == "gallery":
        for site in sites:
            items = {t: classify(t, meas) for t in PRIMARY[site]}
            for t in items:
                items[t]["curv"] = None
            rec["tiles"][site] = gallery(site, items, per, QC_DIR, rec["files"])
            print(site, "gallery done", flush=True)
    elif a.part == "partition":
        summ = json.load(open(os.path.join(rd, "summary", "per_tile_fits.json")))
        for site in sites:
            for t in PRIMARY[site]:
                r = summ.get(f"{t}:primary")
                if not r or r.get("P4") is None:
                    print(t, "no P4 fit", flush=True)
                    continue
                item = classify(t, meas)
                partition_figure(t, item, per, float(r["P4"]["A_crit"]), {"mse": r["P4"]["mse"], "K_over_K0": r["P4"]["K_over_K0"]}, QC_DIR, rec["files"])
                rec["tiles"][t] = {"A_crit": r["P4"]["A_crit"]}
                print(t, "partition done", flush=True)
    rec["wall_s"] = time.time() - t_all
    dump(rec, rec_path)
    # the README from every part's record present so far
    rec_all = {}
    for part in ("tiles", "pairs", "gallery", "partition"):
        pth = os.path.join(qd, f"{part}.json")
        if os.path.exists(pth):
            rec_all[part] = json.load(open(pth))
    write_readme(QC_DIR, rec_all)
    big = {k: v for k, v in rec["files"].items() if v["bytes"] > MAX_BYTES}
    print(f"{a.part}: {len(rec['files'])} files, over 5 MB: {list(big)}, {rec['wall_s']:.0f} s", flush=True)
    _ = (GM_OFFMESA, GM_NCALM, AP_PAMAP, AP1_3DEP, SURVEY)


if __name__ == "__main__":
    main()
