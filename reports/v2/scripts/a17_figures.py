#!/usr/bin/env python3
"""Addendum 17, section 3: the figures for visual review (PNG at 200 dpi and PDF).

Conventions (section 3): the same hillshade everywhere, azimuth 315, altitude 45, no vertical exaggeration, computed from the
1 m filled grid (Horn's gradient on the 3 x 3 neighbourhood; a17_common.hillshade); overview maps from the full 1 m grid's
hillshade block-averaged to 5 m (GM, TTV) or 10 m (AP) on a world-aligned lattice; tile panels at 1 x 1 km, the channel network
A >= 1e3 m2 from the tile's registered area grid (area_model for GM1-GM5, area otherwise), line width scaled by log10 A.
Figures: <site>_overview, <site>_gallery (pages), GM_control_pairs, AP_pamap_3dep_pairs; README.md index; figures.json record.
usage: a17_figures.py --out DIR --fig-dir DIR --grid-gmc DIR --tiles-gmc DIR --tiles-ttv DIR --tiles-gmf DIR [--data $LEMCVT_DATA] [--only overview,gallery,pairs]
"""
import argparse
import json
import os
import shutil
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17_common import (  # noqa: E402
    A16,
    A_NET_FIG_M2,
    HILLSHADE,
    KIND_COLORS,
    KIND_LABELS,
    NODATA,
    OVERVIEW_BLOCK_M,
    STAGE_B,
    SURVEY,
    dump,
    hillshade,
    slurm_record,
)

from lemcvt.hydro.d8 import d8_receiver_arrays  # noqa: E402

DPI = 200
PNG_MAX_BYTES = 5 * 1024 * 1024
NET_COLOR = "#0a3fd6"
HS_LEVELS = 64


def quant(hs):
    return np.round(hs * (HS_LEVELS - 1)) / (HS_LEVELS - 1)


# ------------------------------------------------------------------------------------------------ coarse hillshade

def coarse_lattice(paths, block):
    L = [rasterio.open(p).bounds for p in paths]
    x0 = np.floor(min(b.left for b in L) / block) * block
    x1 = np.ceil(max(b.right for b in L) / block) * block
    y0 = np.floor(min(b.bottom for b in L) / block) * block
    y1 = np.ceil(max(b.top for b in L) / block) * block
    return float(x0), float(y0), float(x1), float(y1), int((x1 - x0) / block), int((y1 - y0) / block)


def coarse_hillshade(paths, block, with_hs=True, chunk_rows=1024):
    """Block-mean hillshade (and the valid-cell fraction) of 1 m grids on a common world-aligned `block` m lattice."""
    x0, y0, x1, y1, nx, ny = coarse_lattice(paths, block)
    S = np.zeros(ny * nx)
    C = np.zeros(ny * nx)
    T = np.zeros(ny * nx)
    t0 = time.time()
    for p in paths:
        with rasterio.open(p) as ds:
            tr = ds.transform
            assert abs(tr.a - 1.0) < 1e-9 and abs(tr.e + 1.0) < 1e-9, p
            xs = tr.c + (np.arange(ds.width) + 0.5)
            bc = np.floor((xs - x0) / block).astype(np.int64)
            for r0 in range(0, ds.height, chunk_rows):
                r1 = min(ds.height, r0 + chunk_rows)
                ra, rb = max(0, r0 - 1), min(ds.height, r1 + 1)
                z = ds.read(1, window=Window(0, ra, ds.width, rb - ra)).astype(np.float64)
                z[z == NODATA] = np.nan
                zz = z[r0 - ra:r1 - ra]
                ys = tr.f - (np.arange(r0, r1) + 0.5)
                br = np.floor((y1 - ys) / block).astype(np.int64)
                idx = br[:, None] * nx + bc[None, :]
                vz = np.isfinite(zz)
                T += np.bincount(idx[vz], minlength=ny * nx)
                if with_hs:
                    hs = hillshade(z, 1.0)[r0 - ra:r1 - ra]
                    v = np.isfinite(hs)
                    S += np.bincount(idx[v], weights=hs[v], minlength=ny * nx)
                    C += np.bincount(idx[v], minlength=ny * nx)
                del z, zz
        print("  coarse", os.path.basename(p), round(time.time() - t0), "s", flush=True)
    hs = np.where(C > 0, S / np.maximum(C, 1), np.nan).reshape(ny, nx).astype(np.float32)
    frac = (T / (block * block)).reshape(ny, nx).astype(np.float32)
    return hs, frac, (x0, y0, x1, y1)


# ------------------------------------------------------------------------------------------------ tiles

def read_tile(data, version, tile, role_area):
    d = os.path.join(data, version, tile)
    with rasterio.open(os.path.join(d, f"{tile}_1m_filled.tif")) as ds:
        z = ds.read(1).astype(np.float64)
        tr, epsg = ds.transform, ds.crs.to_epsg()
        z[z == (ds.nodata if ds.nodata is not None else NODATA)] = np.nan
    with rasterio.open(os.path.join(d, f"{tile}_1m_{role_area}.tif")) as ds:
        area = ds.read(1).astype(np.float64)
    with rasterio.open(os.path.join(d, f"{tile}_1m_fd.tif")) as ds:
        fd = ds.read(1)
    nn = None
    pnn = os.path.join(d, f"{tile}_1m_nnfill_mask.tif")
    if os.path.exists(pnn):
        with rasterio.open(pnn) as ds:
            nn = ds.read(1)
    return {"z": z, "area": area, "fd": fd, "nn": nn, "dx": float(tr.a), "transform": tr, "epsg": epsg}


def read_pamap_tile(data, n):
    d = os.path.join(data, "AP", f"tile{n}")
    with rasterio.open(os.path.join(d, "AP_1m_best_tile_filled_m.tif")) as ds:
        z = ds.read(1).astype(np.float64)
        tr, epsg = ds.transform, ds.crs.to_epsg()
        if ds.nodata is not None:
            z[z == ds.nodata] = np.nan
    with rasterio.open(os.path.join(d, "AP_1m_best_tile_area.tif")) as ds:
        area = ds.read(1).astype(np.float64)
    with rasterio.open(os.path.join(d, "AP_1m_best_tile_fd.tif")) as ds:
        fd = ds.read(1)
    return {"z": z, "area": area, "fd": fd, "nn": None, "dx": float(tr.a), "transform": tr, "epsg": epsg}


def panel(ax, T, title, a_net=A_NET_FIG_M2, scalebar=False):
    z, area, fd, dx = T["z"], T["area"], T["fd"], T["dx"]
    ny, nx = z.shape
    hs = quant(np.nan_to_num(hillshade(z, dx), nan=1.0))
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=[0, nx * dx, 0, ny * dx], interpolation="nearest")
    r_to, c_to, good = d8_receiver_arrays(fd)
    sel = (area >= a_net) & good
    rr, cc = np.nonzero(sel)
    if rr.size:
        x = (cc + 0.5) * dx
        y = (ny - rr - 0.5) * dx
        x2 = (c_to[rr, cc] + 0.5) * dx
        y2 = (ny - r_to[rr, cc] - 0.5) * dx
        la = np.log10(area[rr, cc])
        lw = 0.15 + 0.45 * np.clip(la - 3.0, 0, 4)
        order = np.argsort(la)
        segs = np.stack([np.stack([x, y], 1), np.stack([x2, y2], 1)], 1)[order]
        ax.add_collection(LineCollection(segs, linewidths=lw[order], colors=NET_COLOR, alpha=0.9, capstyle="round"))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    ax.set_title(title, fontsize=5.2, pad=2, linespacing=1.1)
    if scalebar:
        ax.plot([40, 240], [40, 40], color="k", lw=1.5, solid_capstyle="butt")
        ax.text(140, 60, "200 m", ha="center", va="bottom", fontsize=5, color="k")
    v = np.isfinite(z)
    return {"z_min": float(np.nanmin(z)), "z_max": float(np.nanmax(z)), "A_max_m2": float(area.max()), "n_net_cells": int(rr.size),
            "frac_nn_filled": float((T["nn"] == 1).mean()) if T["nn"] is not None else None, "frac_valid": float(v.mean())}


def ttl(name, kind, survey, st):
    nn = "n/a" if st["frac_nn_filled"] is None else f"{100 * st['frac_nn_filled']:.1f} %"
    return f"{name} · {KIND_LABELS.get(kind, kind)} · {survey}\nz {st['z_min']:.0f}-{st['z_max']:.0f} m · A {st['A_max_m2'] / 1e6:.2f} km² · NN {nn}"


def save(fig, base, rec, scale_steps=(1.0, 0.85, 0.72, 0.6)):
    """PNG at 200 dpi under 5 MB (the figure is shrunk, not the dpi, if needed) and PDF."""
    w, h = fig.get_size_inches()
    for s in scale_steps:
        fig.set_size_inches(w * s, h * s)
        fig.savefig(base + ".png", dpi=DPI)
        nb = os.path.getsize(base + ".png")
        if nb <= PNG_MAX_BYTES:
            break
        print("  png", os.path.basename(base), nb, "bytes > 5 MB at scale", s, "- shrinking", flush=True)
    fig.savefig(base + ".pdf", dpi=DPI)
    rec[os.path.basename(base)] = {"png_bytes": os.path.getsize(base + ".png"), "pdf_bytes": os.path.getsize(base + ".pdf"), "size_in": [round(w * s, 2), round(h * s, 2)], "scale": s, "dpi": DPI}
    plt.close(fig)
    print("  saved", os.path.basename(base), rec[os.path.basename(base)], flush=True)


def add_scalebar_north(ax, length_m, epsg):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx = x0 + 0.04 * (x1 - x0)
    by = y0 + 0.04 * (y1 - y0)
    ax.plot([bx, bx + length_m], [by, by], color="k", lw=2.5, solid_capstyle="butt", zorder=20)
    ax.plot([bx, bx + length_m], [by, by], color="w", lw=1.0, solid_capstyle="butt", zorder=21)
    ax.text(bx + length_m / 2, by + 0.012 * (y1 - y0), f"{length_m / 1000:g} km", ha="center", va="bottom", fontsize=7, zorder=22, bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 1})
    nx_ = x1 - 0.05 * (x1 - x0)
    ny_ = y1 - 0.10 * (y1 - y0)
    ax.annotate("N", xy=(nx_, ny_ + 0.05 * (y1 - y0)), xytext=(nx_, ny_), ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "k"}, zorder=22, bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 1})
    ax.set_xlabel(f"Easting (km, EPSG:{epsg})", fontsize=8)
    ax.set_ylabel(f"Northing (km, EPSG:{epsg})", fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    ax.tick_params(labelsize=7)


def read_eligible(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    if "eligible" in df.columns:
        e = df["eligible"]
        e = e.astype(str).str.lower().eq("true") if e.dtype == object else e.astype(bool)
        df = df[e]
    return df[["fp_x0", "fp_y0"]].astype(float).values


def gpkg_layer(path, layer):
    from osgeo import ogr
    ogr.UseExceptions()
    ds = ogr.Open(path)
    lyr = ds.GetLayer(layer)
    out = []
    for f in lyr:
        g = f.GetGeometryRef()
        if g.GetGeometryName() in ("POINT",):
            out.append(("point", [(g.GetX(), g.GetY())]))
        else:
            gg = g if g.GetGeometryName() != "MULTIPOLYGON" else max((g.GetGeometryRef(i) for i in range(g.GetGeometryCount())), key=lambda q: q.GetArea())
            ring = gg.GetGeometryRef(0)
            out.append(("poly", [(ring.GetX(i), ring.GetY(i)) for i in range(ring.GetPointCount())]))
    return out


# ------------------------------------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="run directory (figures/ and figures.json are written here)")
    p.add_argument("--fig-dir", required=True, help="reports/v2/figures/a17 (PNG and PDF copies, README.md)")
    p.add_argument("--grid-gmc", required=True)
    p.add_argument("--tiles-gmc", required=True)
    p.add_argument("--tiles-ttv", required=True)
    p.add_argument("--tiles-gmf", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    p.add_argument("--only", default="overview,gallery,pairs")
    a = p.parse_args()
    fdir = os.path.join(a.out, "figures")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    only = a.only.split(",")
    rec = {"figures": {}, "panels": {}, "hillshade": HILLSHADE, "block_m": OVERVIEW_BLOCK_M, "a_net_m2": A_NET_FIG_M2}
    # ---- the tile set: records of the cut runs (site, tile, kind, version, footprint)
    recs = []
    for path in (os.path.join(STAGE_B, "tiles", "tiles.json"), os.path.join(a.tiles_ttv, "tiles.json"), os.path.join(a.tiles_gmf, "tiles.json"), os.path.join(a.tiles_gmc, "tiles.json")):
        for t in json.load(open(path))["tiles"]:
            recs.append({"site": "GM" if t["site"] in ("GMF", "GMC") else t["site"], "tile": t["tile"], "kind": t["kind"], "version": t["version"], "fp_x0": t["fp_x0"], "fp_y0": t["fp_y0"], "epsg": t["epsg"]})
    tiles = pd.DataFrame(recs)
    rec["tiles"] = recs
    print("tiles:", tiles.groupby(["site", "version", "kind"]).size().to_dict(), flush=True)

    def role_for(version, tile):
        return "area_model" if (version == "GM-x" and tile in ("GM1", "GM2", "GM3", "GM4", "GM5")) else "area"

    # ---------------------------------------------------------------- overview maps
    if "overview" in only:
        sites = {"GM": {"grids": [os.path.join(A16, "grid_GMF", "GMF_1m_filled.tif"), os.path.join(a.grid_gmc, "GMC_1m_filled.tif")], "epsg": 6339,
                        "eligible": [(os.path.join(STAGE_B, "draw", "GM_eligibility.csv"), "NCALM 2003 pool (stage B, ±50 %)"), (os.path.join(A16, "gm_ext", "GMF_eligible_windows.csv"), "FEMA 2018 pool (addendum 16 gate, ±25 %)")],
                        "ncalm": os.path.join(STAGE_B, "grid_GM", "GM_1m_filled.tif"), "bar": 5000},
                 "AP": {"grids": [os.path.join(STAGE_B, "grid_AP", "AP_1m_filled.tif")], "epsg": 6346, "eligible": [(os.path.join(STAGE_B, "draw", "AP_eligibility.csv"), "3DEP pool (stage B, ±25 %)")], "bar": 5000},
                 "TTV": {"grids": [os.path.join(STAGE_B, "grid_TTV", "TTV_1m_filled.tif")], "epsg": 6339, "eligible": [(os.path.join(A16, "draw", "TTV_eligibility.csv"), "Marin pool (addendum 16, ±25 %)")], "bar": 2000}}
        for site, S in sites.items():
            print("overview", site, flush=True)
            block = OVERVIEW_BLOCK_M[site]
            hs, frac, (x0, y0, x1, y1) = coarse_hillshade(S["grids"], block)
            ext = [x0, x1, y0, y1]
            fig, ax = plt.subplots(figsize=(11, 11 * (y1 - y0) / (x1 - x0) + 1.2), dpi=DPI)
            ax.imshow(quant(np.nan_to_num(hs, nan=1.0)), cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest", zorder=1)
            X = x0 + (np.arange(hs.shape[1]) + 0.5) * block
            Y = y1 - (np.arange(hs.shape[0]) + 0.5) * block
            ax.contour(X, Y, (frac > 0.5).astype(float), levels=[0.5], colors="k", linewidths=0.6, zorder=5)
            handles = [Line2D([], [], color="k", lw=0.6, label="lidar footprint (valid 1 m cells)")]
            if site == "GM":
                _, fr2, (nx0, ny0, nx1, ny1) = coarse_hillshade([S["ncalm"]], block, with_hs=False)
                X2 = nx0 + (np.arange(fr2.shape[1]) + 0.5) * block
                Y2 = ny1 - (np.arange(fr2.shape[0]) + 0.5) * block
                ax.contour(X2, Y2, (fr2 > 0.5).astype(float), levels=[0.5], colors="#7b2cbf", linewidths=1.4, linestyles="--", zorder=13)
                handles.append(Line2D([], [], color="#7b2cbf", lw=1.4, ls="--", label="NCALM 2003 footprint (the grid as delivered; open where it meets its raster edge)"))
            n_el = {}
            greys = ["#9a9a9a", "#5a5a5a"]
            for k, (csv_path, lab) in enumerate(S["eligible"]):
                W = read_eligible(csv_path)
                n_el[lab] = int(len(W))
                for wx, wy in W:
                    ax.add_patch(Rectangle((wx, wy), 1000, 1000, fill=False, ec=greys[k % 2], lw=0.3, zorder=7))
                handles.append(Line2D([], [], color=greys[k % 2], lw=0.5, label=f"eligible window, {lab} (n = {len(W)})"))
            if site == "TTV":
                gp = os.path.join(A16, "md_georef", "md1989_fig3_a16.gpkg")
                for _kind, pts in gpkg_layer(gp, "study_area_clipped"):
                    ax.add_patch(MplPolygon(pts, closed=True, fill=False, ec="#2ca02c", lw=1.2, zorder=9))
                heads = [pts[0] for _kind, pts in gpkg_layer(gp, "channel_heads")]
                ax.plot([h[0] for h in heads], [h[1] for h in heads], ".", color="#2ca02c", ms=2.5, zorder=10)
                handles += [Line2D([], [], color="#2ca02c", lw=1.2, label="M&D (1989) study area, clipped at 61 m (1.943 km²)"), Line2D([], [], color="#2ca02c", marker=".", ls="", ms=4, label=f"M&D digitised channel heads (n = {len(heads)})")]
            sub = tiles[tiles.site == site]
            for _, t in sub.iterrows():
                col = KIND_COLORS[t.kind]
                if t.kind == "control":
                    ax.add_patch(Rectangle((t.fp_x0, t.fp_y0), 1000, 1000, fill=False, ec=col, lw=1.2, ls=(0, (3, 2)), zorder=12))
                    ax.text(t.fp_x0 + 500, t.fp_y0 + 120, t.tile, ha="center", va="bottom", fontsize=3.6, color=col, zorder=14)
                else:
                    ax.add_patch(Rectangle((t.fp_x0, t.fp_y0), 1000, 1000, fc=col, ec=col, alpha=0.35, lw=0.8, zorder=11))
                    ax.text(t.fp_x0 + 500, t.fp_y0 + 560, t.tile, ha="center", va="center", fontsize=4.8, color="k", fontweight="bold", zorder=14, bbox={"fc": "w", "ec": "none", "alpha": 0.6, "pad": 0.6})
            kinds = [k for k in ("original", "drawn", "M&D", "control") if (sub.kind == k).any()]
            handles += [Patch(fc=KIND_COLORS[k], ec=KIND_COLORS[k], alpha=0.5, label=f"{KIND_LABELS[k]} (n = {int((sub.kind == k).sum())})") if k != "control" else Line2D([], [], color=KIND_COLORS[k], lw=1.2, ls="--", label=f"{KIND_LABELS[k]} (n = {int((sub.kind == k).sum())})") for k in kinds]
            # limits: the valid data's bounding box (plus the NCALM lattice at GM) with a 500 m margin, not the whole lattice
            rr_, cc_ = np.nonzero(frac > 0)
            bx0, bx1 = x0 + cc_.min() * block, x0 + (cc_.max() + 1) * block
            by1, by0 = y1 - rr_.min() * block, y1 - (rr_.max() + 1) * block
            if site == "GM":
                bx0, bx1, by0, by1 = min(bx0, nx0), max(bx1, nx1), min(by0, ny0), max(by1, ny1)
            m = 500.0
            ax.set_xlim(bx0 - m, bx1 + m)
            ax.set_ylim(by0 - m, by1 + m)
            ax.set_aspect("equal")
            fig.set_size_inches(11, 11 * (by1 - by0 + 2 * m) / (bx1 - bx0 + 2 * m) + 1.2)
            add_scalebar_north(ax, S["bar"], S["epsg"])
            ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.9)
            grids_txt = " + ".join(os.path.basename(g) for g in S["grids"])
            cap = (f"{site} overview: hillshade (azimuth {HILLSHADE['azimuth_deg']:.0f}°, altitude {HILLSHADE['altitude_deg']:.0f}°, no vertical exaggeration) of the full 1 m filled grid "
                   f"({grids_txt}), block-averaged to {block} m; CRS EPSG:{S['epsg']}; tiles 1 × 1 km coloured by kind; eligible windows thin grey.")
            ax.set_title(cap, fontsize=7, wrap=True)
            save(fig, os.path.join(fdir, f"{site}_overview"), rec["figures"])
            rec["figures"][f"{site}_overview"].update({"block_m": block, "lattice": [x0, y0, x1, y1], "shown": [bx0 - m, by0 - m, bx1 + m, by1 + m], "eligible_counts": n_el, "n_tiles": int(len(sub)), "epsg": S["epsg"]})

    # ---------------------------------------------------------------- galleries
    def gallery(name, items, ncols=6, pw=2.55):
        n = len(items)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * pw, nrows * (pw + 0.32)), dpi=DPI, squeeze=False)
        for k, ax in enumerate(axes.ravel()):
            if k >= n:
                ax.axis("off")
                continue
            it = items[k]
            T = it["read"]()
            st = panel(ax, T, "", scalebar=(k == 0))
            ax.set_title(ttl(it["label"], it["kind"], it["survey"], st), fontsize=5.2, pad=2, linespacing=1.1)
            rec["panels"][it["label"]] = {**st, **{k2: v for k2, v in it.items() if k2 != "read"}}
            del T
        fig.suptitle(f"{name}: every panel 1 × 1 km; hillshade az {HILLSHADE['azimuth_deg']:.0f}°, alt {HILLSHADE['altitude_deg']:.0f}°, no exaggeration; channel network A ≥ 10³ m² (line width ∝ log₁₀ A); "
                     "title: tile · kind · survey · z range · largest A inside · nearest-neighbour-filled fraction", fontsize=6.5, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=0.4, w_pad=0.3)
        save(fig, os.path.join(fdir, name), rec["figures"])

    PER_PAGE = 15   # 6 columns x 3 rows: 15 panels of 2.55 in stay under the 5 MB PNG limit (30 panels gave 7.5 MB, 15 TTV panels 4.7 MB)

    def paged(name, items):
        pages = [items[k:k + PER_PAGE] for k in range(0, len(items), PER_PAGE)]
        for k, page in enumerate(pages, start=1):
            gallery(f"{name}_p{k}" if len(pages) > 1 else name, page)

    if "gallery" in only:
        for site in ("GM", "AP", "TTV"):
            sub = tiles[tiles.site == site]
            items = []
            for _, t in sub.iterrows():
                v, tl = t.version, t.tile
                items.append({"label": tl, "kind": t.kind, "survey": SURVEY[v], "version": v, "site": site, "area_role": role_for(v, tl),
                              "read": (lambda v=v, tl=tl: read_tile(a.data, v, tl, role_for(v, tl)))})
            if site == "GM":
                main_items = [it for it in items if it["kind"] != "control"]
                ctrl_items = [it for it in items if it["kind"] == "control"]
                paged("GM_gallery", main_items)                 # GM1-GM7 (NCALM) and GM8-GM30 (FEMA): two pages
                gallery("GM_gallery_control", ctrl_items)       # GM1_fema-GM7_fema
            else:
                paged(f"{site}_gallery", items)

    # ---------------------------------------------------------------- pairs
    if "pairs" in only:
        items = []
        for n in range(1, 8):
            items.append({"label": f"GM{n}", "kind": "original" if n <= 5 else "drawn", "survey": "NCALM 2003", "version": "GM-x", "site": "GM", "area_role": role_for("GM-x", f"GM{n}"), "read": (lambda n=n: read_tile(a.data, "GM-x", f"GM{n}", role_for("GM-x", f"GM{n}")))})
            items.append({"label": f"GM{n}_fema", "kind": "control", "survey": "FEMA 2018", "version": "GM-FEMA2018-ctrl", "site": "GM", "area_role": "area", "read": (lambda n=n: read_tile(a.data, "GM-FEMA2018-ctrl", f"GM{n}_fema", "area"))})
        gallery("GM_control_pairs", items, ncols=4)
        items = []
        for n in range(1, 6):
            items.append({"label": f"AP{n} (PAMAP)", "kind": "original", "survey": "PAMAP 2006-08 (v1/v2 tile, EPSG:26917)", "version": "AP/tile", "site": "AP", "area_role": "area", "read": (lambda n=n: read_pamap_tile(a.data, n))})
            items.append({"label": f"AP{n}_3dep", "kind": "original", "survey": "3DEP 2019-20", "version": "AP-3DEP-x", "site": "AP", "area_role": "area", "read": (lambda n=n: read_tile(a.data, "AP-3DEP-x", f"AP{n}_3dep", "area"))})
        gallery("AP_pamap_3dep_pairs", items, ncols=4)

    # ---------------------------------------------------------------- copies and the index
    for f in sorted(os.listdir(fdir)):
        if f.endswith((".png", ".pdf")):
            shutil.copy2(os.path.join(fdir, f), os.path.join(a.fig_dir, f))
    rec["seconds"] = time.time() - t0
    dump(rec, os.path.join(a.out, "figures.json"))
    print("done", round(rec["seconds"]), "s", flush=True)


if __name__ == "__main__":
    main()
