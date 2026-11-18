#!/usr/bin/env python3
"""Addendum 14, section 2 and 4.3.4: the TTV NCALM bare-earth raster and the 10 m NED that Steve staged.

  * copies `output_be.tif` (NCALM 2006 bare earth, OpenTopography) and `USGS_13_n38w123_20120801.tif` (USGS 1/3 arc-second
    NED) from $SCRATCH/lem_topography to $GROUP_HOME/lemcvt_data/source/TTV/ with sha256 manifest entries, and verifies;
  * records format, CRS (horizontal and vertical as stated in the file), units, resolution, extent, nodata, value range,
    GDAL metadata tags and the COG layout;
  * tests how `output_be.tif` relates to the NCALM LAS tile (class-2 points): nearest-cell and bilinear residuals, and
    whether each cell equals the mean or the minimum of its class-2 points (which identifies the gridding method), plus
    the fraction of valid cells that hold no class-2 point (interpolated cells);
  * coverage of the Montgomery & Dietrich (1989) area (Point Bonita quadrangle, 61-314 m band) and of the eastern strip
    of the quadrangle east of 122.515 W;
  * warps the NED to the TTV lattice (NAD83(2011) / UTM 10N, EPSG:6339, 10 m, integer-metre origin) by bilinear resampling,
    for section 3 (selection on the coarse grid), and runs rule 5.3 (NED against the lidar, block-averaged to 10 m cells)
    for TTV.

usage: a14_ttv_inventory.py --out DIR [--las FILE]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a13_inventory import raster_record  # noqa: E402
from a14_common import (  # noqa: E402
    LEM_TOPO,
    SOURCE,
    TTV_LATTICE,
    bin_to_lattice,
    dump,
    dz_stats,
    sha256_file,
    slurm_record,
    theil_sen_sub,
    write_tif,
)

QUAD = {"lat": (37.75, 37.875), "lon": (-122.625, -122.5)}     # USGS Point Bonita 7.5' quadrangle
EAST_STRIP_LON = -122.515
MD_BAND = (61.0, 314.0)


def copy_with_manifest(src, item_dir, manifest):
    os.makedirs(item_dir, exist_ok=True)
    dst = os.path.join(item_dir, os.path.basename(src))
    sha_src = sha256_file(src)
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copy2(src, dst)
    sha_dst = sha256_file(dst)
    rel = os.path.relpath(dst, os.path.dirname(manifest))
    lines = open(manifest).read().splitlines() if os.path.exists(manifest) else []
    if not any(line.endswith("  " + rel) for line in lines):
        with open(manifest, "a") as f:
            f.write(f"{sha_src}  {rel}\n")
    return {"src": src, "dst": dst, "bytes": os.path.getsize(dst), "sha256_src": sha_src, "sha256_dst": sha_dst, "verified": sha_src == sha_dst,
            "src_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(src)))}


def gdalinfo_json(path):
    from osgeo import gdal
    gdal.UseExceptions()
    return gdal.Info(path, format="json")


def lonlat_bounds(bounds, epsg):
    from pyproj import Transformer
    t = Transformer.from_crs(epsg, 4269, always_xy=True)
    xs = [bounds[0], bounds[2], bounds[2], bounds[0]]
    ys = [bounds[1], bounds[1], bounds[3], bounds[3]]
    lon, lat = t.transform(xs, ys)
    return [min(lon), min(lat), max(lon), max(lat)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--las", default=os.path.join(SOURCE, "TTV", "LTTV06_538471_4188025_LAS", "LTTV06_538471_4188025.las"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    rec = {}

    # 1. copies of record ---------------------------------------------------------------------------------------------
    man = os.path.join(SOURCE, "TTV", "MANIFEST.sha256")
    be_src = os.path.join(LEM_TOPO, "output_be.tif")
    ned_src = os.path.join(LEM_TOPO, "USGS_13_n38w123_20120801.tif")
    rec["copy_be"] = copy_with_manifest(be_src, os.path.join(SOURCE, "TTV", "NCALM_2006_be"), man)
    rec["copy_ned"] = copy_with_manifest(ned_src, os.path.join(SOURCE, "TTV", "NED_13arcsec"), man)
    chk = subprocess.run(["sha256sum", "-c", "MANIFEST.sha256"], cwd=os.path.join(SOURCE, "TTV"), capture_output=True, text=True)
    rec["manifest_check"] = {"exit": chk.returncode, "stdout": chk.stdout}
    be, ned = rec["copy_be"]["dst"], rec["copy_ned"]["dst"]
    print("copies verified:", rec["copy_be"]["verified"], rec["copy_ned"]["verified"], flush=True)

    # 2. records --------------------------------------------------------------------------------------------------------
    rec["be_record"] = raster_record(be)
    rec["be_gdalinfo"] = gdalinfo_json(be)
    rec["ned_record"] = raster_record(ned)
    rec["ned_gdalinfo"] = gdalinfo_json(ned)
    with rasterio.open(be) as ds:
        be_bounds, be_epsg, be_tr, be_nodata = list(ds.bounds), ds.crs.to_epsg(), ds.transform, ds.nodata
        be_z = ds.read(1).astype(np.float64)
    be_valid = np.isfinite(be_z) & (be_z != be_nodata)
    rec["be_lonlat_bounds_nad83"] = lonlat_bounds(be_bounds, be_epsg)
    lon0, lat0, lon1, lat1 = rec["be_lonlat_bounds_nad83"]
    rec["be_coverage"] = {
        "inside_point_bonita_quad": bool(lon0 >= QUAD["lon"][0] and lon1 <= QUAD["lon"][1] and lat0 >= QUAD["lat"][0] and lat1 <= QUAD["lat"][1]),
        "east_edge_lon": lon1, "covers_east_of_122.515W": bool(lon1 > EAST_STRIP_LON + 1e-4),
        "frac_valid_cells_east_of_122.515W": None,
        "frac_valid_in_MD_band_61_314": float(np.mean((be_z[be_valid] >= MD_BAND[0]) & (be_z[be_valid] <= MD_BAND[1]))),
        "max_valid_z": float(be_z[be_valid].max()), "min_valid_z": float(be_z[be_valid].min()),
        "n_valid": int(be_valid.sum()), "n_cells": int(be_z.size)}
    # fraction of valid cells east of the strip longitude (per-column longitude at the raster's mid latitude)
    from pyproj import Transformer
    t = Transformer.from_crs(be_epsg, 4269, always_xy=True)
    xs = be_tr.c + (np.arange(be_z.shape[1]) + 0.5) * be_tr.a
    ymid = be_tr.f + be_z.shape[0] / 2 * be_tr.e
    lon_cols, _ = t.transform(xs, np.full_like(xs, ymid))
    east = lon_cols > EAST_STRIP_LON
    rec["be_coverage"]["frac_valid_cells_east_of_122.515W"] = float(be_valid[:, east].sum() / max(1, be_valid.sum()))
    rec["be_coverage"]["n_columns_east_of_122.515W"] = int(east.sum())
    # sea-level surface (vertical datum evidence, as in addendum 13 for the LAS)
    low = be_z[be_valid & (be_z < 3.0)]
    rec["be_low_surface"] = {"n_cells_below_3m": int(low.size), "percentiles_0.1_1_5_50": [float(v) for v in np.percentile(low, [0.1, 1, 5, 50])] if low.size else None}

    # 3. raster against the NCALM LAS class-2 points -------------------------------------------------------------------
    import laspy
    from scipy.ndimage import map_coordinates
    t0 = time.time()
    with laspy.open(a.las) as f:
        las = f.read()
    c = np.asarray(las.classification)
    g = c == 2
    x, y, z = np.asarray(las.x)[g], np.asarray(las.y)[g], np.asarray(las.z)[g]
    col = np.floor((x - be_tr.c) / be_tr.a).astype(np.int64)
    row = np.floor((y - be_tr.f) / be_tr.e).astype(np.int64)
    inb = (row >= 0) & (row < be_z.shape[0]) & (col >= 0) & (col < be_z.shape[1])
    row, col, x, y, z = row[inb], col[inb], x[inb], y[inb], z[inb]
    cellv = be_z[row, col]
    ok = be_valid[row, col]
    dz_near = cellv[ok] - z[ok]
    fc = (x - be_tr.c) / be_tr.a - 0.5
    fr = (y - be_tr.f) / be_tr.e - 0.5
    zb = np.where(be_valid, be_z, np.nan)
    bil = map_coordinates(zb, [fr, fc], order=1, mode="nearest")
    okb = np.isfinite(bil)
    dz_bil = bil[okb] - z[okb]
    flat = row * be_z.shape[1] + col
    n_pts = np.bincount(flat, minlength=be_z.size)
    s_pts = np.bincount(flat, weights=z, minlength=be_z.size)
    mn = np.full(be_z.size, np.inf)
    np.minimum.at(mn, flat, z)
    mx = np.full(be_z.size, -np.inf)
    np.maximum.at(mx, flat, z)
    has = n_pts > 0
    mean_pts = np.where(has, s_pts / np.maximum(n_pts, 1), np.nan)
    bv = be_valid.ravel()
    cellz = be_z.ravel()
    sel = has & bv
    d_mean = cellz[sel] - mean_pts[sel]
    d_min = cellz[sel] - mn[sel]
    d_max = cellz[sel] - mx[sel]
    rec["be_vs_las_class2"] = {
        "las": a.las, "n_class2": int(g.sum()), "n_class2_in_raster_extent": int(inb.sum()),
        "nearest_cell_dz_raster_minus_point": dz_stats(dz_near), "bilinear_dz_raster_minus_point": dz_stats(dz_bil),
        "cells_with_ge1_class2_point": int(has.sum()), "valid_cells": int(bv.sum()),
        "frac_valid_cells_without_class2_point_(interpolated)": float(np.sum(bv & ~has) / max(1, bv.sum())),
        "frac_cells_with_points_where_raster_equals_point_mean_within_1mm": float(np.mean(np.abs(d_mean) < 1e-3)),
        "frac_within_1cm_of_mean": float(np.mean(np.abs(d_mean) < 1e-2)),
        "frac_within_1mm_of_min": float(np.mean(np.abs(d_min) < 1e-3)), "frac_within_1mm_of_max": float(np.mean(np.abs(d_max) < 1e-3)),
        "frac_raster_between_min_and_max_of_cell_points": float(np.mean((cellz[sel] >= mn[sel] - 1e-3) & (cellz[sel] <= mx[sel] + 1e-3))),
        "dz_raster_minus_cell_mean": dz_stats(d_mean), "points_per_cell_percentiles_5_50_95": [float(v) for v in np.percentile(n_pts[has], [5, 50, 95])],
        "seconds": time.time() - t0}
    print("raster vs LAS done", rec["be_vs_las_class2"]["nearest_cell_dz_raster_minus_point"], flush=True)
    del las

    # 4. NED to the TTV lattice ------------------------------------------------------------------------------------------
    from osgeo import gdal
    gdal.UseExceptions()
    L = TTV_LATTICE
    ned_utm = os.path.join(a.out, "TTV_NED_10m_utm10n.tif")
    gdal.Warp(ned_utm, ned, dstSRS=f"EPSG:{L['epsg']}", outputBounds=(L["x0"], L["y0"], L["x1"], L["y1"]), xRes=L["dx"], yRes=L["dx"],
              resampleAlg="bilinear", dstNodata=-9999.0, outputType=gdal.GDT_Float32, creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
              targetAlignedPixels=False)
    with rasterio.open(ned_utm) as ds:
        ned_z = ds.read(1).astype(np.float64)
        ned_tr, ned_crs = ds.transform, ds.crs
    ned_valid = ned_z != -9999.0
    rec["ned_utm"] = {"path": ned_utm, "transform": list(ned_tr)[:6], "shape": list(ned_z.shape), "epsg": L["epsg"], "resample": "bilinear",
                      "n_valid": int(ned_valid.sum()), "n_land_gt_0": int((ned_z > 0).sum()), "min": float(ned_z[ned_valid].min()), "max": float(ned_z[ned_valid].max()),
                      "datum_note": "source EPSG:4269 (NAD83) to EPSG:6339 (NAD83(2011)); the shift is below 1 m here and GDAL applies none without grids"}
    # datum shift between the raster CRS (26910) and the lattice (6339) at the tile centre, from PROJ (offline: null if no grids)
    tt = Transformer.from_crs(26910, 6339, always_xy=True)
    xc, yc = (be_bounds[0] + be_bounds[2]) / 2, (be_bounds[1] + be_bounds[3]) / 2
    xs2, ys2 = tt.transform(xc, yc)
    rec["crs_shift_26910_to_6339_at_tile_centre_m"] = [float(xs2 - xc), float(ys2 - yc)]

    # 5. rule 5.3 for TTV: NED against the lidar block-averaged to the 10 m lattice -------------------------------------
    b = bin_to_lattice(be, ned_tr, ned_z.shape, 1.0, block_rows=1024)
    full = (b["n_total"] == b["n_valid"]) & (b["n_total"] >= 95)
    lid_mean = np.where(b["n_valid"] > 0, b["sum_z"] / np.maximum(b["n_valid"], 1), np.nan)
    write_tif(os.path.join(a.out, "TTV_ncalm_valid_frac_10m.tif"), (b["n_valid"] / 100.0).astype(np.float32), ned_tr, ned_crs, nodata=None)
    write_tif(os.path.join(a.out, "TTV_ncalm_mean_10m.tif"), np.where(np.isfinite(lid_mean), lid_mean, -9999).astype(np.float32), ned_tr, ned_crs, nodata=-9999.0)
    land = full & ned_valid & (lid_mean > 3.0)
    dz = ned_z[land] - lid_mean[land]
    fit = theil_sen_sub(lid_mean[land], ned_z[land])
    rec["rule_5_3_TTV"] = {"n_cells_fully_covered_land": int(land.sum()), "dz_ned_minus_lidar": dz_stats(dz), "fit_ned_on_lidar": fit,
                           "slope_pass_0.98_1.02": bool(0.98 <= fit["theil_sen_slope"] <= 1.02), "offset_pass_abs_lt_5m": bool(abs(float(np.median(dz))) < 5.0),
                           "note": "lidar = NCALM 2006 output_be.tif (EPSG:26910) block-averaged onto the EPSG:6339 10 m lattice; cells with 100 % lidar coverage and mean > 3 m (land)"}
    rec["rule_5_3_TTV"]["pass"] = rec["rule_5_3_TTV"]["slope_pass_0.98_1.02"] and rec["rule_5_3_TTV"]["offset_pass_abs_lt_5m"]
    print("rule 5.3 TTV", rec["rule_5_3_TTV"]["dz_ned_minus_lidar"].get("median"), fit["theil_sen_slope"], flush=True)

    dump(rec, os.path.join(a.out, "ttv_inventory.json"))
    print("wrote", os.path.join(a.out, "ttv_inventory.json"))


if __name__ == "__main__":
    main()
