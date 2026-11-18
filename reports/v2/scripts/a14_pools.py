#!/usr/bin/env python3
"""Addendum 14, section 3: candidate pools, screens and selection metrics on the coarse grids (GM 10 m, AP 30 m, TTV 10 m).

Per site, on the full coarse grid: priority-flood fill, D8 directions and drainage area (lemcvt.hydro, the TopoAnalysis
ports), the footprint-edge flag propagated downstream (a cell is "edge-touched" if it or any upstream cell borders nodata
or the grid edge), and the terminal-outlet label of every cell.

Windows (3.1): a lattice of non-overlapping 1 km windows (W = round(1000/dx) cells) at four offsets (0 and W/2 cells in
each axis). A window is a candidate when every coarse cell is valid in the coarse grid and "full" in the lidar footprint
mask (every fine lidar cell inside it valid; a14_footprints.py). On AP the footprint of decision 7 (the PAMAP watershed)
is reported separately from the 3DEP tile-index footprint.

Screens (3.2), per candidate:
  1 trunk: the largest drainage area inside the window and the area entering across its edge, from the full-grid routing;
  2 transient: per edge-outlet basin of the window, the through-origin chi-z R^2 at the site reference theta over the
    channel pixels (A >= A_c), and a slope-area break test above 1e3 m^2 (continuous two-segment against one-segment
    log S - log A fit on the basin's pixels, delta BIC = BIC_one - BIC_two); window statistics: the basin-area-weighted mean
    and the worst basin among those holding >= 10 % of the window's draining area;
  3 disturbance: fraction of window cells in NLCD developed (21-24), barren (31) and open-water (11) classes, and on AP the
    fraction under PA DEP longwall panels, mined-out areas and historic surface-mining permits (layers fetched in the job);
  4 independence: the directed window graph "A drains through B" (any cell of A reaches an edge outlet of B on the full
    grid), its transitive closure as the conflict graph.

Selection metrics (3.3), per candidate and for the original ten tiles and the TTV M&D proxy area, by the same code:
  theta by chi: the legacy method (lemcvt.measure.legacy_v1.legacy_chi: tile-edge outlets, A >= A_c, trapezoid chi,
  theta grid 0.10-0.80 then refinement, total per-basin through-origin SSR);
  k_s at the site reference theta: the median of per-basin through-origin chi-z slopes (the statistic the legacy
  reference values are; the pooled slope is reported beside it);
  |C_ht| by the original coarse-grid method: 3x3 (9-point) Laplacian at first-order divides between the basins of the
  A_c network, median of convex pixels, saddles (mixed-sign second derivatives) and flats (|lap| < 1e-4 1/m) excluded.
Tolerances (decision 5): k_s and |C_ht| within +-25 % of the reference, theta within +-0.05.

usage: a14_pools.py --site GM|AP|TTV --footprints DIR --out DIR [--ttv-ned TIF] [--no-network]
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import LEGACY_REF, SITES, dump, slurm_record, write_tif  # noqa: E402

from lemcvt.hydro.d8 import (  # noqa: E402
    d8_area,
    d8_divides,
    d8_flow_direction,
    d8_receiver_arrays,
    d8_upstream_lists,  # noqa: E402
)
from lemcvt.hydro.fill import priority_flood_fill  # noqa: E402
from lemcvt.measure.chi import channel_network, chi_for_thetas, relative_elevation  # noqa: E402
from lemcvt.measure.hilltop_curvature import basin_labels  # noqa: E402
from lemcvt.measure.legacy_v1 import _chi_traverse, _ols_origin, coastline_outlets, legacy_chi  # noqa: E402
from lemcvt.measure.thresholds import d8_slope  # noqa: E402

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None

A_C = {"GM": 1.0e4, "AP": 5.04e4, "TTV": 1.0e4}       # channel threshold of the chi and divide code (legacy A_MIN; TTV as GM, 10 m grid)
THETA_REF = {"GM": LEGACY_REF["GM"]["theta"], "AP": LEGACY_REF["AP"]["theta"], "TTV": None}   # TTV: from the M&D proxy area
A_BREAK_MIN = 1.0e3
TOL_REL, TOL_THETA = 0.25, 0.05
NLCD_WCS = ("https://www.mrlc.gov/geoserver/mrlc_download/wcs?service=WCS&version=2.0.1&request=GetCoverage"
            "&coverageId=mrlc_download__NLCD_2021_Land_Cover_L48&subset=X({x0},{x1})&subset=Y({y0},{y1})&format=image/geotiff")
PASDA_DEP2 = "https://mapservices.pasda.psu.edu/server/rest/services/pasda/DEP2/MapServer/{layer}/query"
DEP_LAYERS = {"longwall_panels": 0, "mined_out_underground": 1, "surface_mining_permits_historic": 22}
DISTURBED_NLCD = {11: "open water", 21: "developed open", 22: "developed low", 23: "developed medium", 24: "developed high", 31: "barren"}
# Montgomery & Dietrich (1989) proxy: trunk cell nearest 61 m near the Tennessee Valley trailhead (UTM 10N box), catchment upstream
MD_OUTLET_BOX = (540300.0, 4189600.0, 541400.0, 4190600.0)
MD_OUTLET_Z = 61.0


# ------------------------------------------------------------------------------------------ numba kernels

if nb is not None:
    @nb.njit(cache=True)
    def _propagate_down(order, target, flag):
        # donors before receivers (descending filled elevation): receiver |= donor
        for k in order:
            t = target[k]
            if t >= 0 and flag[k]:
                flag[t] = True
        return flag

    @nb.njit(cache=True)
    def _terminal(order_asc, target, lab):
        # receivers before donors (ascending elevation): lab[c] = lab[receiver]; roots keep their own index
        for k in order_asc:
            t = target[k]
            if t >= 0:
                lab[k] = lab[t]
        return lab

    @nb.njit(cache=True)
    def _next_window(order_asc, target, wid, nextw):
        for k in order_asc:
            t = target[k]
            if t < 0:
                nextw[k] = -1
            elif wid[t] >= 0 and wid[t] != wid[k]:
                nextw[k] = wid[t]
            else:
                nextw[k] = nextw[t]
        return nextw

    @nb.njit(cache=True)
    def _catchment(root, target_rev_ptr, target_rev_idx, n):
        # upstream BFS over donors (CSR of the receiver relation)
        mask = np.zeros(n, np.bool_)
        stack = [root]
        mask[root] = True
        while len(stack):
            c = stack.pop()
            for p in range(target_rev_ptr[c], target_rev_ptr[c + 1]):
                d = target_rev_idx[p]
                if not mask[d]:
                    mask[d] = True
                    stack.append(d)
        return mask
else:  # pragma: no cover
    def _propagate_down(order, target, flag):
        for k in order:
            t = target[k]
            if t >= 0 and flag[k]:
                flag[t] = True
        return flag

    def _terminal(order_asc, target, lab):
        for k in order_asc:
            t = target[k]
            if t >= 0:
                lab[k] = lab[t]
        return lab

    def _next_window(order_asc, target, wid, nextw):
        for k in order_asc:
            t = target[k]
            if t < 0:
                nextw[k] = -1
            elif wid[t] >= 0 and wid[t] != wid[k]:
                nextw[k] = wid[t]
            else:
                nextw[k] = nextw[t]
        return nextw

    def _catchment(root, ptr, idx, n):
        mask = np.zeros(n, bool)
        stack = [root]
        mask[root] = True
        while stack:
            c = stack.pop()
            for p in range(ptr[c], ptr[c + 1]):
                d = idx[p]
                if not mask[d]:
                    mask[d] = True
                    stack.append(d)
        return mask


# ------------------------------------------------------------------------------------------ full-grid routing

def route_full_grid(z, dx):
    """Fill, D8, area, edge-touched flag, terminal outlet label, receiver target index; NaN = nodata."""
    t0 = time.time()
    zf = priority_flood_fill(z, dx)
    fd = d8_flow_direction(zf)
    area = d8_area(fd, zf, dx)
    area[~np.isfinite(z)] = np.nan
    r_to, c_to, good = d8_receiver_arrays(fd)
    ny, nx = z.shape
    target = np.where(good.ravel(), r_to.ravel() * nx + c_to.ravel(), -1).astype(np.int64)
    valid = np.isfinite(z)
    target[~valid.ravel()] = -1
    # receivers that are invalid: treat as roots (flow leaves the valid grid there)
    tv = target.copy()
    bad_t = (tv >= 0) & ~valid.ravel()[np.maximum(tv, 0)]
    tv[bad_t] = -1
    zff = np.where(valid, zf, -np.inf).ravel()
    order_desc = np.lexsort((np.arange(zff.size), -zff))
    order_asc = order_desc[::-1].copy()
    # edge-adjacent cells: border nodata or the grid edge
    pad = np.pad(valid, 1, constant_values=False)
    adj = np.zeros_like(valid)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            adj |= ~pad[1 + dr:ny + 1 + dr, 1 + dc:nx + 1 + dc]
    adj &= valid
    flag = adj.ravel().copy()
    flag = np.asarray(_propagate_down(order_desc, tv, flag)).reshape(ny, nx)
    lab = np.arange(ny * nx, dtype=np.int64)
    lab = np.asarray(_terminal(order_asc, tv, lab)).reshape(ny, nx)
    lab[~valid] = -1
    return {"zf": zf, "fd": fd, "area": area, "target": tv, "edge_touched": flag, "terminal": lab, "order_asc": order_asc,
            "valid": valid, "seconds": time.time() - t0}


def donors_csr(target, n):
    ok = target >= 0
    src = np.nonzero(ok)[0]
    dst = target[ok]
    order = np.argsort(dst, kind="stable")
    idx = src[order].astype(np.int64)
    counts = np.bincount(dst, minlength=n)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return ptr, idx


# ------------------------------------------------------------------------------------------ windows

def window_lattice(shape, W, off_r, off_c):
    ny, nx = shape
    wid = np.full(shape, -1, np.int64)
    rows = list(range(off_r, ny - W + 1, W))
    cols = list(range(off_c, nx - W + 1, W))
    wins = []
    k = 0
    for r0 in rows:
        for c0 in cols:
            wid[r0:r0 + W, c0:c0 + W] = k
            wins.append((r0, c0))
            k += 1
    return wid, wins


def crop(arr, r0, c0, W):
    return arr[r0:r0 + W, c0:c0 + W]


# ------------------------------------------------------------------------------------------ metrics on a window

def ks_per_basin_at_theta(z, area, fd, dx, theta, a_min):
    """Legacy chi traversal from the tile-edge outlets; per-basin through-origin slopes at a fixed theta."""
    rows, cols = coastline_outlets(z)
    outlets = [o for o in zip(rows.tolist(), cols.tolist(), strict=True) if np.isfinite(area[o]) and area[o] >= a_min]
    ups = d8_upstream_lists(fd)
    ks, xs, ys, areas = [], [], [], []
    for o in outlets:
        x, y = _chi_traverse(theta, o, area, z, ups, dx, a_min)
        if x.size >= 3:
            k, _ssr = _ols_origin(x, y)
            if np.isfinite(k):
                ks.append(k)
                areas.append(float(area[o]))
                xs.append(x)
                ys.append(y)
    if not ks:
        return {"n_basins": 0, "ks_median": np.nan, "ks_pooled": np.nan}
    X, Y = np.concatenate(xs), np.concatenate(ys)
    kp, _ = _ols_origin(X, Y)
    return {"n_basins": len(ks), "ks_median": float(np.median(ks)), "ks_pooled": float(kp), "ks_q16": float(np.percentile(ks, 16)),
            "ks_q84": float(np.percentile(ks, 84)), "basin_areas": areas}


def hilltop_curvature_coarse(z, area, fd, dx, a_c):
    """3x3 Laplacian at first-order divides between the basins of the A >= a_c network; convex median; saddles/flats excluded."""
    channel = np.isfinite(area) & (area >= a_c)
    lab = basin_labels(area, fd, channel)
    div = d8_divides(fd).astype(bool)
    zxx = np.full(z.shape, np.nan)
    zyy = np.full(z.shape, np.nan)
    zxx[1:-1, 1:-1] = (z[1:-1, 2:] - 2 * z[1:-1, 1:-1] + z[1:-1, :-2]) / dx ** 2
    zyy[1:-1, 1:-1] = (z[2:, 1:-1] - 2 * z[1:-1, 1:-1] + z[:-2, 1:-1]) / dx ** 2
    lap = zxx + zyy
    diff = np.zeros_like(div)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            sh = np.roll(np.roll(lab, dr, axis=0), dc, axis=1)
            diff |= (sh != lab) & (sh > 0) & (lab > 0)
    sel = div & diff & ~channel & np.isfinite(lap)
    vals = lap[sel]
    flat = np.abs(vals) < 1e-4
    saddle = (zxx[sel] * zyy[sel]) < 0
    convex = (vals < 0) & ~flat & ~saddle
    out = {"n_divide_pixels": int(sel.sum()), "n_flat": int(flat.sum()), "n_saddle": int((saddle & ~flat).sum()), "n_convex": int(convex.sum()),
           "n_concave": int(((vals > 0) & ~flat & ~saddle).sum())}
    out["C_ht"] = float(np.median(-vals[convex])) if convex.sum() >= 5 else np.nan
    return out


def basin_stats_window(z, area, fd, dx, theta_ref, a_c):
    """Per edge-outlet basin of the window: chi-z R^2 (through origin) at theta_ref, and the slope-area break test."""
    from lemcvt.measure.diagnostics import edge_outlet_labels
    ny, nx = z.shape
    lab = edge_outlet_labels(area, fd)              # flat index of the edge cell each pixel drains to; -1 none
    S = d8_slope(z, fd, dx)
    out = {"basins": []}
    # chi at theta_ref over the channel network of the window (edge pixels are roots)
    try:
        net = channel_network(z, area, fd, a_c, dx, min_basin_pixels=1)
        chi = chi_for_thetas(net, np.array([theta_ref]))[:, 0]
        zrel = relative_elevation(net)
        pix_flat = net.rows * nx + net.cols
        pix_lab = lab.ravel()[pix_flat]
    except Exception as e:  # no channel pixels
        net, chi, zrel, pix_lab = None, None, None, None
        out["chi_error"] = repr(e)
    ids, counts = np.unique(lab[lab >= 0], return_counts=True)
    dx * dx
    tot_cells = float(counts.sum())
    la_all = np.log10(np.where(np.isfinite(area) & (area > 0), area, np.nan))
    ls_all = np.log10(np.where(np.isfinite(S) & (S > 0), S, np.nan))
    for bid, n_px in zip(ids.tolist(), counts.tolist(), strict=True):
        b = {"outlet_flat": int(bid), "n_px": int(n_px), "share_of_draining_cells": n_px / tot_cells, "A_outlet": float(area.ravel()[bid])}
        # chi R^2
        if net is not None:
            m = pix_lab == bid
            m &= chi > 0
            if m.sum() >= 3:
                x, y = chi[m], zrel[m]
                k, ssr = _ols_origin(x, y)
                syy = float(np.dot(y, y))
                b["chi_n"] = int(m.sum())
                b["chi_ks"] = k
                b["chi_r2_origin"] = 1.0 - ssr / syy if syy > 0 else np.nan
        # slope-area break on the basin's pixels above A_BREAK_MIN
        bm = (lab == bid) & np.isfinite(la_all) & np.isfinite(ls_all) & (area >= A_BREAK_MIN)
        la, ls = la_all[bm], ls_all[bm]
        if la.size >= 20:
            n = la.size
            X1 = np.column_stack([np.ones(n), la])
            beta1, *_ = np.linalg.lstsq(X1, ls, rcond=None)
            ssr1 = float(np.sum((ls - X1 @ beta1) ** 2))
            bic1 = n * np.log(max(ssr1, 1e-12) / n) + 2 * np.log(n)
            cands = np.unique(np.quantile(la, np.linspace(0.15, 0.85, 15)))
            best = None
            for bp in cands:
                if (la < bp).sum() < 5 or (la >= bp).sum() < 5:
                    continue
                X2 = np.column_stack([np.ones(n), la, np.maximum(la - bp, 0)])
                beta2, *_ = np.linalg.lstsq(X2, ls, rcond=None)
                ssr2 = float(np.sum((ls - X2 @ beta2) ** 2))
                if best is None or ssr2 < best[0]:
                    best = (ssr2, bp, beta2)
            if best is not None:
                ssr2, bp, beta2 = best
                bic2 = n * np.log(max(ssr2, 1e-12) / n) + 4 * np.log(n)
                b.update({"sa_n": int(n), "sa_slope_one": float(beta1[1]), "sa_break_log10A": float(bp), "sa_slope_below": float(beta2[1]),
                          "sa_slope_above": float(beta2[1] + beta2[2]), "sa_dbic_one_minus_two": float(bic1 - bic2)})
        out["basins"].append(b)
    # window statistics
    bs = out["basins"]
    w = np.array([b["share_of_draining_cells"] for b in bs]) if bs else np.array([])
    r2 = np.array([b.get("chi_r2_origin", np.nan) for b in bs]) if bs else np.array([])
    db = np.array([b.get("sa_dbic_one_minus_two", np.nan) for b in bs]) if bs else np.array([])
    big = w >= 0.10 if bs else np.array([], bool)
    def wmean(v):
        m = np.isfinite(v)
        return float(np.sum(v[m] * w[m]) / np.sum(w[m])) if m.any() and np.sum(w[m]) > 0 else np.nan
    out["n_basins"] = len(bs)
    out["n_basins_ge10pct"] = int(big.sum())
    out["chi_r2_area_weighted"] = wmean(r2)
    out["chi_r2_worst_big_basin"] = float(np.nanmin(r2[big])) if big.any() and np.isfinite(r2[big]).any() else np.nan
    out["chi_r2_min_all"] = float(np.nanmin(r2)) if np.isfinite(r2).any() else np.nan
    out["sa_dbic_area_weighted"] = wmean(db)
    out["sa_dbic_worst_big_basin"] = float(np.nanmax(db[big])) if big.any() and np.isfinite(db[big]).any() else np.nan
    out["sa_dbic_max_all"] = float(np.nanmax(db)) if np.isfinite(db).any() else np.nan
    out["frac_draining_cells_break_preferred"] = float(np.sum(w[np.isfinite(db) & (db > 0)])) if bs else np.nan
    return out


def window_metrics(site, zc, ac, fdc, dx, theta_ref, a_c):
    """Selection metrics on a cropped window (areas from the full grid)."""
    rec = {}
    try:
        lc = legacy_chi(zc, ac, fdc, dx, site if site in ("GM", "AP") else "GM", a_min=a_c)
        rec["theta_legacy"] = lc.theta_star
        rec["theta_legacy_coarse"] = lc.theta_coarse
        rec["ks_at_theta_legacy_pooled"] = lc.ks
        rec["chi_r2_at_theta_legacy"] = lc.r2
        rec["n_basins_chi"] = lc.n_basins
        rec["n_points_chi"] = lc.n_points
    except Exception as e:
        rec["theta_legacy"] = np.nan
        rec["chi_error"] = repr(e)
    if theta_ref is not None:
        k = ks_per_basin_at_theta(zc, ac, fdc, dx, theta_ref, a_c)
        rec["ks_ref_median"] = k["ks_median"]
        rec["ks_ref_pooled"] = k["ks_pooled"]
        rec["ks_ref_n_basins"] = k["n_basins"]
    cv = hilltop_curvature_coarse(zc, ac, fdc, dx, a_c)
    rec["C_ht"] = cv["C_ht"]
    rec["C_ht_n_convex"] = cv["n_convex"]
    rec["C_ht_n_divide"] = cv["n_divide_pixels"]
    rec["C_ht_n_saddle"] = cv["n_saddle"]
    rec["C_ht_n_flat"] = cv["n_flat"]
    return rec


# ------------------------------------------------------------------------------------------ disturbance layers

def fetch_nlcd(out, transform, shape, epsg, tag):
    """NLCD 2021 land cover (MRLC WCS, native EPSG:5070) warped (nearest) onto the coarse lattice. Returns class grid or None."""
    from osgeo import gdal, osr
    gdal.UseExceptions()
    path = os.path.join(out, f"{tag}_nlcd2021.tif")
    raw = os.path.join(out, f"{tag}_nlcd2021_raw5070.tif")
    if not os.path.exists(raw):
        src = osr.SpatialReference()
        src.ImportFromEPSG(epsg)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(5070)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(src, dst)
        ny, nx = shape
        xs = [transform.c, transform.c + nx * transform.a]
        ys = [transform.f + ny * transform.e, transform.f]
        pts = [ct.TransformPoint(x, y) for x in xs for y in ys]
        X = [p[0] for p in pts]
        Y = [p[1] for p in pts]
        url = NLCD_WCS.format(x0=min(X) - 300, x1=max(X) + 300, y0=min(Y) - 300, y1=max(Y) + 300)
        r = subprocess.run(["curl", "-s", "-L", "-m", "600", "-o", raw, "-w", "%{http_code} %{size_download}", url], capture_output=True, text=True)
        with open(os.path.join(out, f"{tag}_nlcd2021_fetch.txt"), "w") as f:
            f.write(url + "\n" + r.stdout + "\n" + r.stderr)
        if not r.stdout.startswith("200") or os.path.getsize(raw) < 1000:
            return None, {"error": "WCS fetch failed", "curl": r.stdout, "url": url}
    try:
        ds = gdal.Open(raw)
        info = {"raw_size": list(ds.RasterXSize for _ in [0]) + [ds.RasterYSize], "raw_srs": ds.GetProjection()[:80], "url_file": f"{tag}_nlcd2021_fetch.txt"}
        ds = None
        tr = (transform.c, transform.a, 0, transform.f, 0, transform.e)
        ny, nx = shape
        gdal.Warp(path, raw, dstSRS=f"EPSG:{epsg}", outputBounds=(tr[0], tr[3] + ny * tr[5], tr[0] + nx * tr[1], tr[3]), width=nx, height=ny,
                  resampleAlg="near", outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=DEFLATE"])
        with rasterio.open(path) as d2:
            cls = d2.read(1)
        info["classes_present"] = {int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True), strict=False)}
        return cls, info
    except Exception as e:
        return None, {"error": repr(e)}


def fetch_pa_dep(out, transform, shape, epsg):
    """PA DEP mining layers (PASDA DEP2 MapServer) as GeoJSON over the AP coarse domain, rasterised onto the lattice."""
    from osgeo import gdal, ogr
    gdal.UseExceptions()
    ny, nx = shape
    env = f"{transform.c},{transform.f + ny * transform.e},{transform.c + nx * transform.a},{transform.f}"
    grids, info = {}, {}
    for name, layer in DEP_LAYERS.items():
        gj = os.path.join(out, f"AP_padep_{name}.geojson")
        if not os.path.exists(gj):
            url = (PASDA_DEP2.format(layer=layer) + f"?geometry={env}&geometryType=esriGeometryEnvelope&inSR={epsg}&spatialRel=esriSpatialRelIntersects"
                   f"&outFields=*&outSR={epsg}&f=geojson")
            r = subprocess.run(["curl", "-s", "-L", "-m", "300", "-o", gj, "-w", "%{http_code} %{size_download}", url], capture_output=True, text=True)
            info[name] = {"layer": layer, "curl": r.stdout, "url": url}
        try:
            ds = ogr.Open(gj)
            lyr = ds.GetLayer()
            n = lyr.GetFeatureCount()
            tif = os.path.join(out, f"AP_padep_{name}.tif")
            drv = gdal.GetDriverByName("GTiff")
            o = drv.Create(tif, nx, ny, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE"])
            o.SetGeoTransform((transform.c, transform.a, 0, transform.f, 0, transform.e))
            o.SetProjection(rasterio.crs.CRS.from_epsg(epsg).to_wkt())
            o.GetRasterBand(1).Fill(0)
            gdal.RasterizeLayer(o, [1], lyr, burn_values=[1], options=["ALL_TOUCHED=FALSE"])
            grids[name] = o.GetRasterBand(1).ReadAsArray().astype(bool)
            o = None
            info.setdefault(name, {})["n_features"] = n
            info[name]["n_cells"] = int(grids[name].sum())
        except Exception as e:
            info.setdefault(name, {})["error"] = repr(e)
    return grids, info


# ------------------------------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True, choices=["GM", "AP", "TTV"])
    p.add_argument("--footprints", required=True, help="a14_footprints.py output dir")
    p.add_argument("--out", required=True)
    p.add_argument("--ttv-ned", default=None)
    p.add_argument("--no-network", action="store_true")
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    site = a.site
    S = SITES[site]
    coarse = a.ttv_ned if site == "TTV" else S["coarse"]
    with rasterio.open(coarse) as ds:
        z = ds.read(1).astype(np.float64)
        tr, crs, nod = ds.transform, ds.crs, ds.nodata
    if nod is not None and np.isfinite(nod):
        z[z == nod] = np.nan
    if site == "TTV":
        z[z <= 0] = np.nan                       # sea and below-sea cells of the NED are not land
    dx = float(tr.a)
    dy = float(-tr.e)
    ny, nx = z.shape
    rec = {"site": site, "coarse": coarse, "shape": [ny, nx], "dx": dx, "dy": dy, "epsg": crs.to_epsg(), "n_valid": int(np.isfinite(z).sum()),
           "A_c": A_C[site], "theta_ref": THETA_REF[site], "tolerances": {"ks_rel": TOL_REL, "C_ht_rel": TOL_REL, "theta_abs": TOL_THETA}}
    # footprint masks on the coarse lattice
    if site == "TTV":
        with rasterio.open(os.path.join(a.footprints, "TTV_marin_index_cover.tif")) as ds:
            full = ds.read(1).astype(bool)
        with rasterio.open(os.path.join(a.footprints, "..", "ttv_inventory", "TTV_ncalm_valid_frac_10m.tif")) as ds:
            ncalm_frac = ds.read(1)
        ncalm_full = ncalm_frac >= 0.999
        rec["footprint"] = "Marin 2019 QL1 (USGS CA_NoCal_Wildfires_B5b_QL1_2018, Lot 12) tile index, cell centres"
    else:
        with rasterio.open(os.path.join(a.footprints, f"{site}_lidar_full.tif")) as ds:
            full = ds.read(1).astype(bool)
        rec["footprint"] = S["lidar"]
        if site == "AP":
            # AP footprint (3.1) = the 3DEP tile index; the PAMAP watershed (decision 7) = coarse cells holding any PAMAP data.
            # PAMAP has scattered nodata holes, so "every fine cell valid" (the `full` mask) leaves no 33 x 33 window; the
            # watershed test uses the any-data mask and the hole fraction is reported per window.
            with rasterio.open(os.path.join(a.footprints, "AP_3dep_index_cover.tif")) as ds:
                index_cover = ds.read(1).astype(bool)
            with rasterio.open(os.path.join(a.footprints, "AP_lidar_valid_frac.tif")) as ds:
                pamap_frac = ds.read(1)
            pamap_any = pamap_frac > 0
            rec["footprint"] = "3DEP tile index (QL2_5K_SPSouth_TileIndex_Clipped); PAMAP watershed = any-data mask"
    print(site, "grid", z.shape, "valid", rec["n_valid"], flush=True)

    # full-grid routing
    R = route_full_grid(z, dx)
    rec["routing_seconds"] = R["seconds"]
    area, fd, _zf = R["area"], R["fd"], R["zf"]
    write_tif(os.path.join(a.out, f"{site}_area_full.tif"), np.where(np.isfinite(area), area, -9999).astype(np.float32), tr, crs, nodata=-9999.0)
    write_tif(os.path.join(a.out, f"{site}_fd_full.tif"), fd.astype(np.uint8), tr, crs)
    write_tif(os.path.join(a.out, f"{site}_edge_touched.tif"), R["edge_touched"].astype(np.uint8), tr, crs)
    print("routing done in", round(R["seconds"]), "s; max area km2", np.nanmax(area) / 1e6, flush=True)

    # disturbance layers
    nlcd, nlcd_info, dep, dep_info = None, {"skipped": True}, {}, {}
    if not a.no_network:
        nlcd, nlcd_info = fetch_nlcd(a.out, tr, z.shape, crs.to_epsg(), site)
        if site == "AP":
            dep, dep_info = fetch_pa_dep(a.out, tr, z.shape, crs.to_epsg())
    rec["nlcd"] = nlcd_info
    rec["pa_dep"] = dep_info
    disturbed = np.isin(nlcd, list(DISTURBED_NLCD)) if nlcd is not None else None
    mined = np.zeros(z.shape, bool)
    for g in dep.values():
        mined |= g

    # TTV reference: the M&D proxy catchment
    theta_ref = THETA_REF[site]
    if site == "TTV":
        x0b, y0b, x1b, y1b = MD_OUTLET_BOX
        cols = np.floor((np.array([x0b, x1b]) - tr.c) / tr.a).astype(int)
        rows = np.floor((np.array([y1b, y0b]) - tr.f) / tr.e).astype(int)
        sub_a = area[rows[0]:rows[1], cols[0]:cols[1]]
        sub_z = z[rows[0]:rows[1], cols[0]:cols[1]]
        cand = np.isfinite(sub_a) & (sub_a >= 1e5)
        if cand.any():
            score = np.where(cand, np.abs(sub_z - MD_OUTLET_Z), np.inf)
            i, j = np.unravel_index(np.argmin(score), score.shape)
            root = (rows[0] + i) * nx + (cols[0] + j)
            ptr, idx = donors_csr(R["target"], ny * nx)
            cm = np.asarray(_catchment(root, ptr, idx, ny * nx)).reshape(ny, nx)
            rr, cc = np.nonzero(cm)
            r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
            zc = np.where(cm, z, np.nan)[r0:r1, c0:c1]
            ac = np.where(cm, area, np.nan)[r0:r1, c0:c1]
            fdc = fd[r0:r1, c0:c1]
            md = window_metrics(site, zc, ac, fdc, dx, None, A_C[site])
            theta_ref = md["theta_legacy"]
            k = ks_per_basin_at_theta(zc, ac, fdc, dx, theta_ref, A_C[site])
            md.update({"ks_ref_median": k["ks_median"], "ks_ref_pooled": k["ks_pooled"], "ks_ref_n_basins": k["n_basins"]})
            md.update({"outlet_row": int(rows[0] + i), "outlet_col": int(cols[0] + j), "outlet_x": tr.c + (cols[0] + j + 0.5) * tr.a,
                       "outlet_y": tr.f + (rows[0] + i + 0.5) * tr.e, "outlet_z": float(z[rows[0] + i, cols[0] + j]), "outlet_area_km2": float(area[rows[0] + i, cols[0] + j] / 1e6),
                       "catchment_cells": int(cm.sum()), "catchment_km2": float(cm.sum() * dx * dy / 1e6), "z_min": float(np.nanmin(zc)), "z_max": float(np.nanmax(zc)),
                       "bbox_rows": [int(r0), int(r1)], "bbox_cols": [int(c0), int(c1)]})
            write_tif(os.path.join(a.out, "TTV_md_proxy_catchment.tif"), cm.astype(np.uint8), tr, crs)
            rec["md_proxy"] = md
            rec["theta_ref"] = theta_ref
            rec["reference"] = {"theta": theta_ref, "ks": md["ks_ref_median"], "C_ht": md["C_ht"], "source": "M&D proxy catchment on the 10 m NED, same code"}
            print("TTV M&D proxy:", {k: v for k, v in md.items() if not isinstance(v, (list, dict))}, flush=True)
        else:
            rec["md_proxy"] = {"error": "no trunk cell >= 1e5 m2 in the outlet box"}
    else:
        rec["reference"] = dict(LEGACY_REF[site], source="legacy TAYLOR_REF (updated_best_fit_tile_finder.ipynb cell 1)")
    ref = rec["reference"]

    # whole-grid metrics by this code (k_s at theta_ref, C_ht; theta by the vectorised chi fit as a cross-check)
    try:
        t0 = time.time()
        kw = ks_per_basin_at_theta(z, area, fd, dx, theta_ref, A_C[site])
        cw = hilltop_curvature_coarse(z, area, fd, dx, A_C[site])
        rec["whole_grid"] = {"ks_ref_median": kw["ks_median"], "ks_ref_pooled": kw["ks_pooled"], "n_basins": kw["n_basins"], "C_ht": cw["C_ht"],
                             "C_ht_n_convex": cw["n_convex"], "C_ht_n_divide": cw["n_divide_pixels"], "seconds": time.time() - t0}
        if site != "TTV" and rec["n_valid"] < 2.0e6:
            t1 = time.time()
            lw = legacy_chi(z, area, fd, dx, site, a_min=A_C[site])
            rec["whole_grid"]["theta_legacy"] = lw.theta_star
            rec["whole_grid"]["theta_legacy_n_basins"] = lw.n_basins
            rec["whole_grid"]["ks_pooled_at_theta_legacy"] = lw.ks
            rec["whole_grid"]["theta_legacy_seconds"] = time.time() - t1
            print("whole-grid legacy theta", lw.theta_star, lw.n_basins, "basins in", round(time.time() - t1), "s", flush=True)
        from lemcvt.measure.concavity import fit_chi
        net = channel_network(z, area, fd, A_C[site], dx, min_basin_pixels=10)
        cf = fit_chi(net, thetas=np.round(np.arange(0.10, 0.801, 0.01), 2), min_pixels_per_basin=10, n_boot=0)
        rec["whole_grid"]["theta_fit_chi_vectorised"] = cf.theta_star
        rec["whole_grid"]["theta_fit_chi_n_basins"] = cf.n_basins_used
        rec["whole_grid"]["ks_pooled_fit_chi_at_theta_star"] = cf.ks_pooled
        print("whole grid:", rec["whole_grid"], flush=True)
    except Exception as e:
        rec["whole_grid"] = {"error": repr(e)}

    # windows at four offsets
    W = int(round(1000.0 / dx))
    H = W // 2
    rec["W_cells"] = W
    rec["W_m"] = W * dx
    rec["offsets_cells"] = [[0, 0], [0, H], [H, 0], [H, H]]
    rec["offset_m"] = H * dx
    rows_out = []
    conflicts = {}
    counts = {}
    for oi, (orr, occ) in enumerate(rec["offsets_cells"]):
        wid, wins = window_lattice(z.shape, W, orr, occ)
        n_win = len(wins)
        cand = np.zeros(n_win, bool)
        in_ws = np.zeros(n_win, bool)
        in_idx = np.zeros(n_win, bool)
        for k, (r0, c0) in enumerate(wins):
            fz = np.isfinite(crop(z, r0, c0, W)).all()
            ff = crop(full, r0, c0, W).all()
            if site == "AP":
                in_idx[k] = crop(index_cover, r0, c0, W).all() and fz
                in_ws[k] = crop(pamap_any, r0, c0, W).all() and fz
                cand[k] = in_idx[k] and in_ws[k]   # pool = inside the 3DEP index and the PAMAP watershed (decision 7)
            elif site == "TTV":
                cand[k] = ff and fz
                in_ws[k] = cand[k] and crop(ncalm_full, r0, c0, W).all()   # also inside the NCALM 2006 raster
            else:
                cand[k] = ff and fz
        counts[f"offset_{oi}"] = {"n_windows": n_win, "n_candidates": int(cand.sum()), "n_coarse_valid_only": int(sum(np.isfinite(crop(z, r0, c0, W)).all() for r0, c0 in wins)),
                                  **({"n_in_3dep_index_and_coarse_valid": int(in_idx.sum()), "n_in_pamap_watershed": int(in_ws.sum())} if site == "AP" else {}),
                                  **({"n_also_in_ncalm_2006_raster": int(in_ws.sum())} if site == "TTV" else {})}
        print(site, "offset", oi, counts[f"offset_{oi}"], flush=True)
        # independence: next window downstream, for candidate windows only
        widc = np.where(np.isin(wid, np.nonzero(cand)[0]), wid, -1).ravel()
        nextw = np.full(ny * nx, -1, np.int64)
        nextw = np.asarray(_next_window(R["order_asc"], R["target"], widc, nextw))
        edges = set()
        edge_w = {}
        af_ = np.where(np.isfinite(area), area, 0.0).ravel()
        for k in np.nonzero(cand)[0]:
            cells = widc == k
            nw = nextw[cells]
            ac_ = af_[cells]
            for w2 in np.unique(nw):
                if w2 >= 0 and w2 != k:
                    edges.add((int(k), int(w2)))
                    sel_ = nw == w2
                    edge_w[(int(k), int(w2))] = {"max_area_m2": float(ac_[sel_].max()), "n_cells": int(sel_.sum())}
        # transitive closure
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
        closure = set()
        for u in list(adj):
            seen, stack = set(), [u]
            while stack:
                c = stack.pop()
                for v in adj.get(c, ()):
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            for v in seen:
                closure.add((u, v))
        und = {tuple(sorted(e)) for e in closure}
        deg = np.zeros(n_win, int)
        for u, v in und:
            deg[u] += 1
            deg[v] += 1
        deg_direct = np.zeros(n_win, int)
        for u, v in {tuple(sorted(e)) for e in edges}:
            deg_direct[u] += 1
            deg_direct[v] += 1
        conflicts[f"offset_{oi}"] = {"n_direct_edges": len(edges), "n_direct_pairs_undirected": len({tuple(sorted(e)) for e in edges}),
                                     "n_conflict_pairs_closure": len(und), "n_windows_with_conflict_closure": int((deg > 0).sum()),
                                     "max_degree_closure": int(deg.max()) if n_win else 0, "max_degree_direct": int(deg_direct.max()) if n_win else 0,
                                     "direct_edges": [{"from": u, "to": v, **edge_w[(u, v)]} for (u, v) in sorted(edges)], "pairs_closure": sorted(und)}
        # per-window screens and metrics
        tgt = R["target"]
        for k in np.nonzero(cand)[0]:
            r0, c0 = wins[k]
            zc, ac, fdc = crop(z, r0, c0, W), crop(area, r0, c0, W), crop(fd, r0, c0, W)
            row = {"site": site, "offset": oi, "window": int(k), "row0": int(r0), "col0": int(c0), "x0": tr.c + c0 * tr.a, "y1": tr.f + r0 * tr.e,
                   "x1": tr.c + (c0 + W) * tr.a, "y0": tr.f + (r0 + W) * tr.e, "candidate": True,
                   "in_pamap_watershed": bool(in_ws[k]) if site == "AP" else None, "in_ncalm_2006": bool(in_ws[k]) if site == "TTV" else None}
            # screen 1: trunk
            row["A_max_inside_m2"] = float(np.nanmax(ac))
            cells = np.zeros(z.shape, bool)
            cells[r0:r0 + W, c0:c0 + W] = True
            cf_ = cells.ravel()
            donors_out = (~cf_) & (tgt >= 0) & cf_[np.maximum(tgt, 0)]
            row["A_inflow_edge_m2"] = float(np.nansum(area.ravel()[donors_out]))
            row["A_inflow_max_single_m2"] = float(np.nanmax(area.ravel()[donors_out])) if donors_out.any() else 0.0
            row["upstream_fully_valid"] = bool(~crop(R["edge_touched"], r0, c0, W).any())
            row["edge_touched_frac"] = float(crop(R["edge_touched"], r0, c0, W).mean())
            row["relief_m"] = float(np.nanmax(zc) - np.nanmin(zc))
            if site == "AP":
                row["pamap_valid_frac_mean"] = float(crop(pamap_frac, r0, c0, W).mean())
                row["pamap_full_cells_frac"] = float(crop(full, r0, c0, W).mean())
            gy, gx = np.gradient(zc, dy, dx)
            row["mean_slope"] = float(np.nanmean(np.hypot(gx, gy)))
            # screen 2: transient
            bsw = basin_stats_window(zc, ac, fdc, dx, theta_ref, A_C[site])
            for key in ("n_basins", "n_basins_ge10pct", "chi_r2_area_weighted", "chi_r2_worst_big_basin", "chi_r2_min_all", "sa_dbic_area_weighted",
                        "sa_dbic_worst_big_basin", "sa_dbic_max_all", "frac_draining_cells_break_preferred"):
                row[key] = bsw.get(key, np.nan)
            # screen 3: disturbance
            row["nlcd_disturbed_frac"] = float(crop(disturbed, r0, c0, W).mean()) if disturbed is not None else np.nan
            if nlcd is not None:
                nc = crop(nlcd, r0, c0, W)
                row["nlcd_developed_frac"] = float(np.isin(nc, [21, 22, 23, 24]).mean())
                row["nlcd_water_frac"] = float((nc == 11).mean())
                row["nlcd_barren_frac"] = float((nc == 31).mean())
            row["mining_frac"] = float(crop(mined, r0, c0, W).mean()) if dep else np.nan
            for name, g in dep.items():
                row[f"{name}_frac"] = float(crop(g, r0, c0, W).mean())
            # screen 4: independence
            row["conflict_degree_closure"] = int(deg[k])
            row["conflict_degree_direct"] = int(deg_direct[k])
            row["downstream_direct_max_inflow_area_m2"] = max([edge_w[(u, v)]["max_area_m2"] for (u, v) in edges if v == k], default=0.0)
            # metrics
            row.update(window_metrics(site, zc, ac, fdc, dx, theta_ref, A_C[site]))
            row["tol_theta"] = bool(np.isfinite(row.get("theta_legacy", np.nan)) and abs(row["theta_legacy"] - ref["theta"]) <= TOL_THETA) if ref.get("theta") is not None else None
            row["tol_ks"] = bool(np.isfinite(row.get("ks_ref_median", np.nan)) and abs(row["ks_ref_median"] / ref["ks"] - 1) <= TOL_REL) if ref.get("ks") else None
            row["tol_C_ht"] = bool(np.isfinite(row.get("C_ht", np.nan)) and abs(row["C_ht"] / ref["C_ht"] - 1) <= TOL_REL) if ref.get("C_ht") else None
            row["tol_all"] = bool(row["tol_theta"] and row["tol_ks"] and row["tol_C_ht"])
            rows_out.append(row)
        print(site, "offset", oi, "windows done", int(cand.sum()), flush=True)
    rec["counts"] = counts
    rec["conflicts"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("pairs_closure", "direct_edges")} for k, v in conflicts.items()}
    dump(conflicts, os.path.join(a.out, f"{site}_conflicts.json"))

    # the original tiles (3.4)
    orig = []
    if a.data and S["tiles"]:
        for t in S["tiles"]:
            n = t[-1]
            pattern = os.path.join(a.data, site, f"tile{n}")
            cand_files = [f for f in os.listdir(pattern) if f.endswith("_filled.tif")] if os.path.isdir(pattern) else []
            if not cand_files:
                orig.append({"tile": t, "error": "filled grid not found"})
                continue
            with rasterio.open(os.path.join(pattern, cand_files[0])) as ds:
                b = ds.bounds
            c0 = int(np.round((b.left - tr.c) / tr.a))
            r0 = int(np.round((b.top - tr.f) / tr.e))
            Wt = int(np.round((b.right - b.left) / tr.a))
            if r0 < 0 or c0 < 0 or r0 + Wt > ny or c0 + Wt > nx:
                orig.append({"tile": t, "error": "outside the coarse grid", "row0": r0, "col0": c0, "W": Wt})
                continue
            zc, ac, fdc = crop(z, r0, c0, Wt), crop(area, r0, c0, Wt), crop(fd, r0, c0, Wt)
            row = {"tile": t, "row0": r0, "col0": c0, "W": Wt, "lattice_offset_cells": [float((b.left - tr.c) / tr.a - c0), float((b.top - tr.f) / tr.e - r0)],
                   "coarse_all_valid": bool(np.isfinite(zc).all()), "footprint_full": bool(crop(full, r0, c0, Wt).all()),
                   "A_max_inside_m2": float(np.nanmax(ac)), "relief_m": float(np.nanmax(zc) - np.nanmin(zc))}
            cells = np.zeros(z.shape, bool)
            cells[r0:r0 + Wt, c0:c0 + Wt] = True
            cf_ = cells.ravel()
            donors_out = (~cf_) & (R["target"] >= 0) & cf_[np.maximum(R["target"], 0)]
            row["A_inflow_edge_m2"] = float(np.nansum(area.ravel()[donors_out]))
            row["A_inflow_max_single_m2"] = float(np.nanmax(area.ravel()[donors_out])) if donors_out.any() else 0.0
            row["upstream_fully_valid"] = bool(~crop(R["edge_touched"], r0, c0, Wt).any())
            row["edge_touched_frac"] = float(crop(R["edge_touched"], r0, c0, Wt).mean())
            bsw = basin_stats_window(zc, ac, fdc, dx, theta_ref, A_C[site])
            for key in ("n_basins", "n_basins_ge10pct", "chi_r2_area_weighted", "chi_r2_worst_big_basin", "chi_r2_min_all", "sa_dbic_area_weighted",
                        "sa_dbic_worst_big_basin", "sa_dbic_max_all", "frac_draining_cells_break_preferred"):
                row[key] = bsw.get(key, np.nan)
            row["nlcd_disturbed_frac"] = float(crop(disturbed, r0, c0, Wt).mean()) if disturbed is not None else np.nan
            row["mining_frac"] = float(crop(mined, r0, c0, Wt).mean()) if dep else np.nan
            row.update(window_metrics(site, zc, ac, fdc, dx, theta_ref, A_C[site]))
            row["tol_theta"] = bool(np.isfinite(row.get("theta_legacy", np.nan)) and abs(row["theta_legacy"] - ref["theta"]) <= TOL_THETA)
            row["tol_ks"] = bool(np.isfinite(row.get("ks_ref_median", np.nan)) and abs(row["ks_ref_median"] / ref["ks"] - 1) <= TOL_REL)
            row["tol_C_ht"] = bool(np.isfinite(row.get("C_ht", np.nan)) and abs(row["C_ht"] / ref["C_ht"] - 1) <= TOL_REL)
            row["tol_all"] = bool(row["tol_theta"] and row["tol_ks"] and row["tol_C_ht"])
            # which lattice window(s) contain/overlap it
            row["overlapping_candidates"] = [(w["offset"], w["window"]) for w in rows_out if not (w["col0"] >= c0 + Wt or w["col0"] + W <= c0 or w["row0"] >= r0 + Wt or w["row0"] + W <= r0)]
            orig.append(row)
            print(t, {k: v for k, v in row.items() if k in ("A_max_inside_m2", "A_inflow_edge_m2", "theta_legacy", "ks_ref_median", "C_ht", "tol_all", "chi_r2_area_weighted", "sa_dbic_area_weighted")}, flush=True)
    rec["original_tiles"] = orig

    import pandas as pd
    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(a.out, f"{site}_windows.csv"), index=False)
    dump(rec, os.path.join(a.out, f"{site}_pools.json"))
    print("wrote", os.path.join(a.out, f"{site}_windows.csv"), len(df), "candidate windows")


if __name__ == "__main__":
    main()
