#!/usr/bin/env python3
"""Addendum 20, supplement (Steve, 2026-11-20): calibration of E, D and K in the Montgomery & Dietrich (1989) Tennessee Valley study area
against the published field values, reproducing the published methods on our lidar before using them.

Published values used (all read from the papers in `papers/`, page and table cited in the report):
  Heimsath et al. (1999, Geomorphology 27): soil production 77 +- 9 exp(-0.023 h) m/Myr (eq. 9); Table 1 cosmogenic rates (bedrock under
    soil 21-60 m/Myr, exposed greywacke 39-107, stream sediment 64, 66 and 102 m/Myr); K = 50 cm2/yr (Reneau 1988) and rho_s/rho_r = 0.5
    (eq. 4; Table 1: soil 1.4, greywacke 2.2 g/cm3); curvature from a 9-point stencil (eq. 6) on a kriged grid, 5 m the finest grid at
    which curvature is scale-independent (their Fig. 4); nose curvatures in their Fig. 6.
  Montgomery & Dietrich (1989, WRR): source areas planimetered on the USGS Point Bonita 7.5' map (40 ft contours), Table 1 (63 heads).
  Perron et al. (2009, Nature 460): U/D = -lap(z)_h; S* = |grad z| / (lap z - lap z_h) = (D/K) A^-m, D/K and m by regression of log S*
    on log A (their eqs. 4-5); Gabilan Mesa D/K = 124 +- 3 m^(2m+1), m = 0.35 +- 0.003, lap_h = -11.8e-3 (Table 1).

Tests:
  T1  Heimsath's curvature method on our lidar: block means at b = 1, 2, 3, 5, 7, 9 m, eq. 6; nose cells (convex at 5 m, slope 8-30 deg,
      off the D8 divides and off cells with A >= 1e3 m2) in the Heimsath/M&D area; the scale test of their Fig. 4; the distribution against
      their Fig. 6 range; the erosion rate those curvatures imply with their D (0.5 x 50 cm2/yr) against their Table 1.
  T2  Map-scale source areas at Steve's 64 heads: the 1 m lidar block-averaged to b = 2, 3, 5, 10, 12 m, pit-filled, D8 (and D-infinity
      at 5 and 10 m); the largest area in the 3 x 3 coarse cells around each pick; against M&D Table 1.
  T3  Perron's S* regression in the M&D polygon (1 m, 7 m disk), lap_h from the crest pixels; D/K and m; and the same on the NCALM GM
      tiles (Perron's survey) as a check of our implementation against his published D/K and m.
  T4  Synthesis: E, D, K for the M&D area; the addendum 18 fits rescaled to each E.
usage: a20_ttv_cal.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a20_common import (  # noqa: E402
    A18_SUMMARY,
    GM_NCALM,
    HEADS_CSV,
    MD_GPKG,
    RESULTS_ROOT,
    SEED,
    TTV_FULL,
    a18_specs,
    dump,
    load_json,
    md_table1,
    metre_tile,
    out_path,
    pct,
    slurm_record,
)
from a20_perron import disk_quadratic  # noqa: E402

WIN = (538800.0, 541600.0, 4188800.0, 4192400.0)          # x0, x1, y0, y1 (EPSG:6339): the M&D area, Heimsath's inset, the heads' basins
HEIMSATH = {"P0_m_per_Myr": 77.0, "P0_se": 9.0, "alpha_per_cm": 0.023, "K_soil_m2_per_yr": 50e-4, "rho_ratio_used": 0.5, "rho_soil": 1.4, "rho_rock": 2.2,
            "table1_soil_covered_m_per_Myr": {"TV-3": 47, "TV-6": 26, "TV-7": 21, "TV-10": 25, "TV-12": 60, "TV-13": 26, "TV-15": 48, "TV-16": 33, "TV-17": 27},
            "table1_exposed_greywacke_m_per_Myr": {"TV-2": 39, "TV-11": 107, "TV-23": 91}, "table1_exposed_other_m_per_Myr": {"TV-4 greenstone": 20, "TV-5 chert": 15},
            "table1_stream_sediment_m_per_Myr": {"creek1a": 64, "creek1b": 66, "creek2": 102},
            "fig6_range_neg_lap_per_m": [-0.015, 0.045], "fig6_bulk_per_m": [0.005, 0.035], "grid_scale_m": 5.0}
PERRON_GM = {"D_over_K": 124.0, "D_over_K_se": 3.0, "m": 0.35, "m_se": 0.003, "lap_h": -11.8e-3}
GRID_B = (1, 2, 3, 5, 7, 9)
AREA_B = (1, 2, 3, 5, 10, 12)
SITE_M = 0.33


def read_window():
    import rasterio
    from rasterio.windows import from_bounds
    x0, x1, y0, y1 = WIN
    out = {}
    with rasterio.open(TTV_FULL["filled"]) as ds:
        w = from_bounds(x0, y0, x1, y1, ds.transform)
        Z = ds.read(1, window=w).astype(float)
        Z[Z == ds.nodata] = np.nan
        tr = ds.window_transform(w)
    with rasterio.open(TTV_FULL["area_cells"]) as ds:
        Ac = ds.read(1, window=from_bounds(x0, y0, x1, y1, ds.transform)).astype(float)
    with rasterio.open(TTV_FULL["fd"]) as ds:
        FD = ds.read(1, window=from_bounds(x0, y0, x1, y1, ds.transform))
    out.update(Z=Z, A=Ac * abs(tr.a * tr.e), FD=FD, tr=tr, dx=abs(tr.a))
    return out


def block_mean(Z, b):
    if b == 1:
        return Z.copy()
    ny, nx = (Z.shape[0] // b) * b, (Z.shape[1] // b) * b
    return np.nanmean(Z[:ny, :nx].reshape(ny // b, b, nx // b, b), axis=(1, 3))


def lap_eq6(Zb, b):
    """Heimsath et al. (1999) eq. 6: [2(E2+E4+E6+E8) + (E1+E3+E7+E9) - 12 E5] / (4 b^2)."""
    L = np.full(Zb.shape, np.nan)
    c = Zb[1:-1, 1:-1]
    edge = Zb[:-2, 1:-1] + Zb[2:, 1:-1] + Zb[1:-1, :-2] + Zb[1:-1, 2:]
    corner = Zb[:-2, :-2] + Zb[:-2, 2:] + Zb[2:, :-2] + Zb[2:, 2:]
    L[1:-1, 1:-1] = (2 * edge + corner - 12 * c) / (4 * b * b)
    return L


def slope_deg(Zb, b):
    gy, gx = np.gradient(Zb, b)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def md_polygon_on(shape, tr):
    from osgeo import gdal, ogr
    gdal.UseExceptions()
    ds = gdal.GetDriverByName("MEM").Create("", shape[1], shape[0], 1, gdal.GDT_Byte)
    ds.SetGeoTransform((tr.c, tr.a, tr.b, tr.f, tr.d, tr.e))
    from osgeo import osr
    s = osr.SpatialReference()
    s.ImportFromEPSG(6339)
    ds.SetProjection(s.ExportToWkt())
    src = ogr.Open(MD_GPKG)              # keep the data source alive while its layer is used
    gdal.RasterizeLayer(ds, [1], src.GetLayerByName("study_area"), burn_values=[1])
    return ds.GetRasterBand(1).ReadAsArray().astype(bool)


# ----------------------------------------------------------------------------- T1
def t1_heimsath(W, poly) -> dict:
    from lemcvt.hydro.d8 import d8_divides
    Z, A, FD = W["Z"], W["A"], W["FD"]
    div = d8_divides(FD).astype(bool)
    rec: dict = {"method": "block-mean grids, Heimsath eq. 6 (9-point stencil); nose cells defined on the 5 m grid", "grids_m": list(GRID_B)}
    laps = {b: lap_eq6(block_mean(Z, b), b) for b in GRID_B}
    b5 = 5
    Z5 = block_mean(Z, b5)
    s5 = slope_deg(Z5, b5)
    A5 = block_mean(np.where(np.isfinite(A), A, np.nan), b5)          # mean 1 m area in the 5 m cell (a cell with a channel has a large mean)
    div5 = block_mean(div.astype(float), b5) > 0
    poly5 = block_mean(poly.astype(float), b5) > 0.5
    nose = poly5 & (laps[5] < 0) & (s5 >= 8) & (s5 <= 30) & ~div5 & (A5 < 1e3) & np.isfinite(laps[5])
    rec["n_nose_cells_5m"] = int(nose.sum())
    rec["nose_definition"] = "inside the M&D study polygon; convex at the 5 m grid; slope 8-30 deg (Heimsath Table 1 samples 8-25 deg); no D8 divide cell in the 5 m cell; mean 1 m D8 area < 1e3 m2"
    # map the nose mask to each grid by cell centre
    ny5, nx5 = nose.shape
    rr, cc = np.nonzero(nose)
    xc = (cc + 0.5) * b5
    yc = (rr + 0.5) * b5
    scale = {}
    for b in GRID_B:
        Lb = laps[b]
        ri = np.clip((yc // b).astype(int), 0, Lb.shape[0] - 1)
        ci = np.clip((xc // b).astype(int), 0, Lb.shape[1] - 1)
        v = -Lb[ri, ci]
        v = v[np.isfinite(v)]
        scale[str(b)] = {"median_neg_lap": float(np.median(v)), "q16": float(np.percentile(v, 16)), "q84": float(np.percentile(v, 84)), "sd": float(np.std(v)),
                         "frac_convex": float(np.mean(v > 0)), "n": int(v.size)}
    rec["scale_test"] = scale
    v5 = -laps[5][nose]
    rec["at_5m"] = pct(v5) | {"frac_in_fig6_bulk": float(np.mean((v5 >= HEIMSATH["fig6_bulk_per_m"][0]) & (v5 <= HEIMSATH["fig6_bulk_per_m"][1])))}
    # implied soil production with Heimsath's own D (rock equivalent), against Table 1
    for tag, ratio in (("ratio_0.5", HEIMSATH["rho_ratio_used"]), ("ratio_measured_0.64", HEIMSATH["rho_soil"] / HEIMSATH["rho_rock"])):
        D = ratio * HEIMSATH["K_soil_m2_per_yr"]
        rec[f"implied_E_m_per_Myr_{tag}"] = pct(D * v5 * 1e6) | {"D_rock_m2_per_yr": D}
    t = list(HEIMSATH["table1_soil_covered_m_per_Myr"].values())
    rec["table1_soil_covered"] = {"median": float(np.median(t)), "range": [min(t), max(t)], "n": len(t)}
    # implied soil depth from the soil production function at each curvature (production = 0.5 K |lap|), report only
    P = HEIMSATH["rho_ratio_used"] * HEIMSATH["K_soil_m2_per_yr"] * v5 * 1e6
    h = np.where(P > 0, -np.log(np.clip(P / HEIMSATH["P0_m_per_Myr"], 1e-6, None)) / HEIMSATH["alpha_per_cm"], np.nan)
    rec["implied_soil_depth_cm"] = pct(np.clip(h, 0, None))
    return rec, laps


# ----------------------------------------------------------------------------- T2
def t2_areas(W) -> dict:
    from lemcvt.hydro.d8 import d8_area, d8_flow_direction
    from lemcvt.hydro.fill import priority_flood_fill
    Z, tr = W["Z"], W["tr"]
    with open(HEADS_CSV) as f:
        P = [(int(r["pick_id"]), float(r["x_epsg6339"]), float(r["y_epsg6339"])) for r in csv.DictReader(f)]
    t1 = md_table1()
    At1 = np.array([x["A"] for x in t1])
    rec: dict = {"table1": pct(At1), "scales": {}}
    x0, y1 = tr.c, tr.f
    for b in AREA_B:
        Zb = block_mean(Z, b)
        valid = np.isfinite(Zb)
        Zf = priority_flood_fill(np.where(valid, Zb, np.nan), float(b))
        fd = d8_flow_direction(Zf, border="outlets")
        Ab = d8_area(fd, Zf, float(b))
        Ab[~valid] = np.nan
        rows = []
        for pid, x, y in P:
            c = int((x - x0) // b)
            r = int((y1 - y) // b)
            sub = Ab[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            rows.append({"pick_id": pid, "A_cell": float(Ab[r, c]), "A_3x3": float(np.nanmax(sub)), "edge_dist_cells": int(min(r, c, Ab.shape[0] - 1 - r, Ab.shape[1] - 1 - c))})
        a33 = np.array([q["A_3x3"] for q in rows])
        ac = np.array([q["A_cell"] for q in rows])
        rec["scales"][str(b)] = {"A_3x3": pct(a33), "A_cell": pct(ac), "median_ratio_to_table1_median": float(np.median(a33) / np.median(At1)),
                                 "frac_within_table1_5_95": float(np.mean((a33 >= np.percentile(At1, 5)) & (a33 <= np.percentile(At1, 95)))), "picks": rows}
        if b in (5, 10):
            from lemcvt.hydro.dinf import compute_dinf_area, compute_dinf_flow_direction
            ang, rf, fi = compute_dinf_flow_direction(Zf, float(b), FD_d8=fd)
            Ad = compute_dinf_area(Zf, float(b), ang, rf, fi)
            ad = []
            for _pid, x, y in P:
                c = int((x - x0) // b)
                r = int((y1 - y) // b)
                ad.append(float(np.nanmax(Ad[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2])))
            rec["scales"][str(b)]["Dinf_A_3x3"] = pct(ad)
        print(f"T2 b {b} m: A_3x3 median {np.median(a33):.0f} m2 (Table 1 {np.median(At1):.0f}); within Table 1 5-95 % {rec['scales'][str(b)]['frac_within_table1_5_95']:.2f}", flush=True)
    return rec


# ----------------------------------------------------------------------------- T3
def profile_cells(A, FD, a_net=1e3, a_min=1e2):
    """'Stream profiles' (Perron et al. 2009, Fig. 3b): the D8 network A >= a_net, extended upstream from each of its heads along the
    largest-area donor chain until A < a_min."""
    from lemcvt.hydro.d8 import d8_receiver_arrays
    net = np.isfinite(A) & (A >= a_net)
    r_to, c_to, good = d8_receiver_arrays(FD)
    ny, nx = A.shape
    # best donor of every cell: the donor with the largest area
    best = np.full((ny, nx), -1, dtype=np.int64)
    best_a = np.full((ny, nx), -1.0)
    rr, cc = np.nonzero(good & np.isfinite(A))
    tr_, tc_ = r_to[rr, cc], c_to[rr, cc]
    order = np.argsort(A[rr, cc])
    for k in order:                      # ascending area, so the last write per receiver is the largest donor
        best[tr_[k], tc_[k]] = rr[k] * nx + cc[k]
        best_a[tr_[k], tc_[k]] = A[rr[k], cc[k]]
    has_net_donor = np.zeros((ny, nx), dtype=bool)
    nr, nc = np.nonzero(net & good)
    has_net_donor[r_to[nr, nc], c_to[nr, nc]] = True
    heads = net & ~has_net_donor
    out = net.copy()
    for r, c in zip(*np.nonzero(heads), strict=True):
        cur = best[r, c]
        while cur >= 0:
            r2, c2 = divmod(int(cur), nx)
            if not A[r2, c2] >= a_min:
                break
            out[r2, c2] = True
            cur = best[r2, c2]
    return out


def sstar_fit(A, grad, lap, lap_h, blocks, a_rng=(1e2, 1e5), m_fixed=None, n_boot=500, seed=SEED, stat="mean"):
    d = lap - lap_h
    ok = np.isfinite(A) & np.isfinite(grad) & np.isfinite(d) & (d > 0) & (grad > 0) & (A >= a_rng[0]) & (A <= a_rng[1])
    la, ls, bl = np.log10(A[ok]), np.log10(grad[ok] / d[ok]), blocks[ok]
    edges = np.arange(math.floor(math.log10(a_rng[0]) * 10) / 10, math.log10(a_rng[1]) + 1e-9, 0.1)

    def fit(la_, ls_):
        idx = np.digitize(la_, edges) - 1
        xs, ys = [], []
        for k in range(edges.size - 1):
            s = ls_[idx == k]
            if s.size >= 30:
                xs.append(0.5 * (edges[k] + edges[k + 1]))
                ys.append(np.mean(s) if stat == "mean" else np.median(s))
        xs, ys = np.array(xs), np.array(ys)
        if xs.size < 3:
            return None
        if m_fixed is None:
            sl, ic = np.polyfit(xs, ys, 1)
            return {"D_over_K": float(10 ** ic), "m": float(-sl), "n_bins": int(xs.size), "bins_x": xs.tolist(), "bins_y": ys.tolist()}
        ic = float(np.mean(ys + m_fixed * xs))
        return {"D_over_K": float(10 ** ic), "m": m_fixed, "n_bins": int(xs.size)}

    pt = fit(la, ls)
    if pt is None:
        return {"n_cells": int(ok.sum())}
    uniq, inv = np.unique(bl, return_inverse=True)
    rng = np.random.default_rng(seed)
    bs_dk, bs_m = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size)[inv]
        sel = np.repeat(np.arange(la.size), w)
        f = fit(la[sel], ls[sel])
        if f:
            bs_dk.append(f["D_over_K"])
            bs_m.append(f["m"])
    pt["n_cells"] = int(ok.sum())
    pt["D_over_K_ci68"] = [float(np.percentile(bs_dk, 16)), float(np.percentile(bs_dk, 84))]
    if m_fixed is None:
        pt["m_ci68"] = [float(np.percentile(bs_m, 16)), float(np.percentile(bs_m, 84))]
    pt["A_range_m2"] = list(a_rng)
    pt["lap_h"] = float(lap_h)
    pt["stat"] = stat
    return pt


def t3_sstar(W, poly) -> dict:
    from a20_perron import block_index
    rec: dict = {"method": "Perron et al. 2009 eqs. 4-5: S* = |grad z|/(lap z - lap_h); bins of 0.1 in log10 A (>= 30 cells), mean log10 S* per bin (their Fig. 3b: means of log-transformed data), OLS of the bin values (median as a variant); "
                           "|grad z| and lap z from the 7 m disk quadratic (their SI 2.1); A the 1 m D8 area; cells with lap z > lap_h; block bootstrap 100 m, 500 draws"}
    q = disk_quadratic(W["Z"], 7.0, W["dx"], W["dx"])
    blocks, _ = block_index(W["Z"].shape, W["dx"], W["dx"])
    # lap_h: the crest pixels (addendum 20 D0_N set) inside the M&D polygon on TTV1-TTV3, with the 7 m disk and the 11 m square
    frames = []
    for t in ("TTV1", "TTV2", "TTV3"):
        d = pd.read_parquet(out_path("tiles", f"{t}_divide_pixels.parquet"), columns=["lap7", "C_Lf", "in_md_polygon"])
        frames.append(d[d["in_md_polygon"]])
    cp = pd.concat(frames)
    lap_h = {"disk7_median": float(np.nanmedian(cp["lap7"])), "square11_median": float(-np.nanmedian(cp["C_Lf"])), "n_crest_px": int(cp.shape[0]),
             "n_crest_px_without_disk7": int(cp["lap7"].isna().sum())}
    rec["lap_h_crest"] = lap_h
    A = np.where(poly, W["A"], np.nan)
    prof = profile_cells(W["A"], W["FD"])
    Ap = np.where(poly & prof, W["A"], np.nan)
    rec["n_profile_cells_in_polygon"] = int((poly & prof).sum())
    rec["fits"] = {}
    for tag, lh in (("lap_h_disk7", lap_h["disk7_median"]), ("lap_h_square11", lap_h["square11_median"])):
        for rng_ in ((1e2, 1e5), (1e3, 1e5), (10.0, 1e5)):
            k = f"{tag}:A{rng_[0]:g}-{rng_[1]:g}"
            rec["fits"][k] = sstar_fit(A, q["grad"], q["lap"], lh, blocks, rng_)
            rec["fits"][k + ":m0.33"] = sstar_fit(A, q["grad"], q["lap"], lh, blocks, rng_, m_fixed=SITE_M)
            rec["fits"][k + ":median"] = sstar_fit(A, q["grad"], q["lap"], lh, blocks, rng_, stat="median")
            rec["fits"][k + ":profiles"] = sstar_fit(Ap, q["grad"], q["lap"], lh, blocks, rng_)
            rec["fits"][k + ":profiles:m0.33"] = sstar_fit(Ap, q["grad"], q["lap"], lh, blocks, rng_, m_fixed=SITE_M)
            print(f"T3 TTV {k}:profiles: D/K {rec['fits'][k + ':profiles'].get('D_over_K')} m {rec['fits'][k + ':profiles'].get('m')}", flush=True)
            print(f"T3 TTV {k}: D/K {rec['fits'][k].get('D_over_K')} m {rec['fits'][k].get('m')}; at m 0.33 D/K {rec['fits'][k + ':m0.33'].get('D_over_K')}", flush=True)
    # implementation check on Perron's own survey (NCALM GM tiles) against his published D/K and m
    gm = {}
    for t in GM_NCALM:
        td = metre_tile(t)
        qq = disk_quadratic(td.Z, 7.0, td.dx, td.filled.dy)
        bl, _ = block_index(td.Z.shape, td.dx, td.filled.dy)
        pc = profile_cells(td.A, td.FD)
        gm[t] = (td.A, qq["grad"], qq["lap"], bl, np.where(pc, td.A, np.nan))
    off = 0
    Acat, Gcat, Lcat, Bcat, Pcat = [], [], [], [], []
    for _t, (a_, g_, l_, b_, p_) in gm.items():
        Pcat.append(p_.ravel())
        Acat.append(a_.ravel())
        Gcat.append(g_.ravel())
        Lcat.append(l_.ravel())
        Bcat.append(b_.ravel() + off)
        off += int(b_.max()) + 1
    Acat, Gcat, Lcat, Bcat, Pcat = (np.concatenate(x) for x in (Acat, Gcat, Lcat, Bcat, Pcat))
    rec["gm_check"] = {"published": PERRON_GM, "tiles": GM_NCALM}
    a20site = load_json(out_path("site", "partA.json"))
    gm_crest = float(np.median([-(a20site["tiles"][t]["four_way"]["divide_disk7"]["median_neg"]) for t in GM_NCALM]))
    for tag, lh in (("perron_lap_h", PERRON_GM["lap_h"]), ("our_crest_disk7", gm_crest)):
        for rng_ in ((1e2, 1e5), (1e3, 1e5), (10.0, 1e5)):
            k = f"{tag}:A{rng_[0]:g}-{rng_[1]:g}"
            rec["gm_check"][k] = sstar_fit(Acat, Gcat, Lcat, lh, Bcat, rng_)
            rec["gm_check"][k + ":median"] = sstar_fit(Acat, Gcat, Lcat, lh, Bcat, rng_, stat="median")
            rec["gm_check"][k + ":profiles"] = sstar_fit(Pcat, Gcat, Lcat, lh, Bcat, rng_)
            print(f"T3 GM check {k}:profiles: D/K {rec['gm_check'][k + ':profiles'].get('D_over_K')} m {rec['gm_check'][k + ':profiles'].get('m')}", flush=True)
            print(f"T3 GM check {k}: D/K {rec['gm_check'][k].get('D_over_K')} m {rec['gm_check'][k].get('m')} (Perron 124, 0.35)", flush=True)
    rec["gm_check"]["our_crest_disk7_lap_h"] = gm_crest
    return rec


# ----------------------------------------------------------------------------- T4
def t4_synthesis(T1, T3) -> dict:
    fits = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    specs = a18_specs()
    E_cfg = 77e-6
    E = {"ridge_soil_covered_median": 27e-6, "ridge_soil_covered_mean": float(np.mean(list(HEIMSATH["table1_soil_covered_m_per_Myr"].values()))) * 1e-6,
         "basin_creek1": 65e-6, "basin_creek2": 102e-6, "config_P0": E_cfg}
    D_pub = {"rho0.5": 0.5 * HEIMSATH["K_soil_m2_per_yr"], "rho_measured": HEIMSATH["rho_soil"] / HEIMSATH["rho_rock"] * HEIMSATH["K_soil_m2_per_yr"]}
    out: dict = {"E_m_per_yr": E, "D_published_rock_equiv_m2_per_yr": D_pub}
    # D0 from our crest curvature at each E (TTV1-3 crest pixels in the M&D polygon)
    lh = T3["lap_h_crest"]
    out["D0_crest"] = {k: {"disk7": e / abs(lh["disk7_median"]), "square11": e / abs(lh["square11_median"])} for k, e in E.items()}
    out["D0_heimsath5m_nose_median"] = {k: e / T1["at_5m"]["p50"] for k, e in E.items()}
    # K from D/K at m = 0.33 (comparable with the fits) and free m
    key = "lap_h_disk7:A100-100000"
    dk033 = T3["fits"][key + ":m0.33"]["D_over_K"]
    dkf = T3["fits"][key]
    out["K_from_SStar"] = {"m0.33": {k: d / dk033 for k, d in D_pub.items()} | {"D_over_K": dk033},
                           "m_free": {k: d / dkf["D_over_K"] for k, d in D_pub.items()} | {"D_over_K": dkf["D_over_K"], "m": dkf["m"]}}
    # the addendum 18 fits on TTV1-TTV3 rescaled to each E (the fitted K and D scale with U; D/D0 and K/K0 do not)
    rows = []
    for t in ("TTV1", "TTV2", "TTV3"):
        f = fits[f"{t}:primary"]
        s = specs[f"{t}:primary"]
        r = {"tile": t, "U_fit": float(s["E"]), "K0_cfg": float(s["K0"])}
        for name in ("P4", "AD"):
            r[f"D_{name}_at_U_cfg"] = f[name]["D"]
            r[f"K_{name}_at_U_cfg"] = f[name]["K"]
            for k, e in E.items():
                r[f"D_{name}@{k}"] = f[name]["D"] * e / float(s["E"])
                r[f"K_{name}@{k}"] = f[name]["K"] * e / float(s["E"])
        for k, e in E.items():
            r[f"K0@{k}"] = float(s["K0"]) * e / float(s["E"])
        r["A_crit_P4"] = f["P4"]["A_crit"]
        rows.append(r)
    out["a18_rescaled"] = rows
    return out


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("ttv_cal", "ttv_cal.json")))
    W = read_window()
    poly = md_polygon_on(W["Z"].shape, W["tr"])
    print(f"window {W['Z'].shape}, polygon {poly.sum()} cells [{time.time() - t0:.0f} s]", flush=True)
    T1, _ = t1_heimsath(W, poly)
    print("T1:", {k: round(v["median_neg_lap"], 4) for k, v in T1["scale_test"].items()}, "implied E", T1["implied_E_m_per_Myr_ratio_0.5"].get("p50"), f"[{time.time() - t0:.0f} s]", flush=True)
    T2 = t2_areas(W)
    T3 = t3_sstar(W, poly)
    T4 = t4_synthesis(T1, T3)
    out = {"heimsath": HEIMSATH, "perron_gm": PERRON_GM, "window": WIN, "T1": T1, "T2": T2, "T3": T3, "T4": T4, "wall_s": time.time() - t0}
    dump(out, out_path("ttv_cal", "ttv_cal.json"))
    print(json.dumps(T4["D0_crest"], indent=0)[:800], "\nK:", T4["K_from_SStar"], flush=True)
    _ = RESULTS_ROOT


if __name__ == "__main__":
    main()
