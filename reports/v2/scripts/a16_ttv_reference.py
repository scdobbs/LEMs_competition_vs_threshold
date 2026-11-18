#!/usr/bin/env python3
"""Addendum 16, section 2.2 (decision 2): the TTV reference from the clipped M&D polygon on the 10 m NED.

Same code path as a15_ttv_reference.py (stage A functions: legacy-chi theta, per-basin through-origin slopes at that theta,
|C_ht| = 3x3 Laplacian at convex first-order divides of the A >= 1e4 m2 network), with two changes fixed by decision 2:
  * the polygon is the clipped one (layer study_area_clipped of md1989_fig3_a16.gpkg, a16_ttv_clip.py);
  * the k_s reference is the POOLED through-origin chi-z slope over the channel points of the polygon at the polygon's theta
    (ks_per_basin_at_theta(...)["ks_pooled"]); the per-basin median is written beside it for the record.
The window statistic of the addendum 15 rules is unchanged (each window's k_s is still the median of its per-basin slopes,
a15_draw.py reads ks_ref_median); the window's pooled slope is also written (ks_ref_pooled) so the count with that statistic
can be reported. The theta-dependent TTV window statistics are recomputed at the new theta for every candidate window, as
in a15 (deviation 4 there).
Outputs (run dir ttv_reference/): TTV_reference.json, TTV_windows_a16.csv, TTV_md_polygon_clipped_10m.tif.
usage: a16_ttv_reference.py --gpkg md1989_fig3_a16.gpkg --out DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_pools import A_C, basin_stats_window, crop, ks_per_basin_at_theta, route_full_grid, window_metrics  # noqa: E402
from a16_common import STAGE_A, STAGE_B, TOL_REL, dump, slurm_record, write_tif  # noqa: E402

LAYER = "study_area_clipped"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpkg", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    ned = os.path.join(STAGE_A, "ttv_inventory", "TTV_NED_10m_utm10n.tif")
    with rasterio.open(ned) as ds:
        z = ds.read(1).astype(np.float64)
        tr, crs, nod = ds.transform, ds.crs, ds.nodata
    if nod is not None and np.isfinite(nod):
        z[z == nod] = np.nan
    z[z <= 0] = np.nan
    dx = float(tr.a)
    ny, nx = z.shape
    R = route_full_grid(z, dx)
    area, fd = R["area"], R["fd"]
    from osgeo import gdal, ogr
    src = ogr.Open(a.gpkg)
    lyr = src.GetLayerByName(LAYER)
    assert lyr is not None, f"{a.gpkg}: no layer {LAYER}"
    drv = gdal.GetDriverByName("MEM")
    tgt = drv.Create("", nx, ny, 1, gdal.GDT_Byte)
    tgt.SetGeoTransform((tr.c, tr.a, 0.0, tr.f, 0.0, tr.e))
    tgt.SetProjection(crs.to_wkt())
    gdal.RasterizeLayer(tgt, [1], lyr, burn_values=[1], options=["ALL_TOUCHED=FALSE"])
    cm = tgt.ReadAsArray().astype(bool) & np.isfinite(z)
    write_tif(os.path.join(a.out, "TTV_md_polygon_clipped_10m.tif"), cm.astype(np.uint8), tr, crs)
    rr, cc = np.nonzero(cm)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    zc = np.where(cm, z, np.nan)[r0:r1, c0:c1]
    ac = np.where(cm, area, np.nan)[r0:r1, c0:c1]
    fdc = fd[r0:r1, c0:c1]
    md = window_metrics("TTV", zc, ac, fdc, dx, None, A_C["TTV"])
    theta = md["theta_legacy"]
    k = ks_per_basin_at_theta(zc, ac, fdc, dx, theta, A_C["TTV"])
    md.update({"ks_ref_median": k["ks_median"], "ks_ref_pooled": k["ks_pooled"], "ks_ref_n_basins": k["n_basins"], "ks_q16": k.get("ks_q16"), "ks_q84": k.get("ks_q84"),
               "basin_areas_m2": k.get("basin_areas"),
               "polygon_cells": int(cm.sum()), "polygon_km2": float(cm.sum() * dx * dx / 1e6), "z_min": float(np.nanmin(zc)), "z_max": float(np.nanmax(zc)),
               "A_max_inside_m2": float(np.nanmax(ac)), "bbox_rows": [int(r0), int(r1)], "bbox_cols": [int(c0), int(c1)]})
    md["basin_stats_at_theta"] = {kk: vv for kk, vv in basin_stats_window(zc, ac, fdc, dx, theta, A_C["TTV"]).items() if kk != "basins"}
    stage_b = json.load(open(os.path.join(STAGE_B, "ttv_reference", "TTV_reference.json")))
    ref = {"theta": theta, "ks": md["ks_ref_pooled"], "ks_statistic": "pooled through-origin chi-z slope over the channel points of the clipped polygon (decision 2)",
           "ks_median_for_the_record": md["ks_ref_median"], "C_ht": md["C_ht"], "polygon_layer": LAYER,
           "source": "clipped M&D (1989) study-area polygon (z >= 61 m) on the 10 m NED; k_s pooled (addendum 16 decision 2)",
           "stage_b_reference": {kk: stage_b[kk] for kk in ("theta", "ks", "C_ht", "source")}, "stage_b_polygon": {kk: stage_b["polygon"].get(kk) for kk in ("theta_legacy", "ks_ref_median", "ks_ref_pooled", "ks_ref_n_basins", "C_ht", "polygon_km2", "z_min", "z_max")},
           "stage_a_proxy": stage_b.get("stage_a_proxy"), "polygon": md, "seconds": time.time() - t0}
    print("TTV reference (clipped polygon):", {kk: vv for kk, vv in md.items() if not isinstance(vv, (list, dict))}, flush=True)
    kw = ks_per_basin_at_theta(z, area, fd, dx, theta, A_C["TTV"])
    ref["whole_grid_ks_at_theta"] = {"ks_ref_median": kw["ks_median"], "ks_ref_pooled": kw["ks_pooled"], "n_basins": kw["n_basins"]}
    dump(ref, os.path.join(a.out, "TTV_reference.json"))
    # recompute the theta-dependent window statistics at the new theta (addendum 15 deviation 4, repeated)
    df = pd.read_csv(os.path.join(STAGE_A, "pools", "TTV_windows.csv"))
    W = 100
    t1 = time.time()
    for i, row in df.iterrows():
        r0, c0 = int(row["row0"]), int(row["col0"])
        zc, ac, fdc = crop(z, r0, c0, W), crop(area, r0, c0, W), crop(fd, r0, c0, W)
        bsw = basin_stats_window(zc, ac, fdc, dx, theta, A_C["TTV"])
        for key in ("n_basins", "n_basins_ge10pct", "chi_r2_area_weighted", "chi_r2_worst_big_basin", "chi_r2_min_all", "sa_dbic_area_weighted",
                    "sa_dbic_worst_big_basin", "sa_dbic_max_all", "frac_draining_cells_break_preferred"):
            df.loc[i, key] = bsw.get(key, np.nan)
        k = ks_per_basin_at_theta(zc, ac, fdc, dx, theta, A_C["TTV"])
        df.loc[i, "ks_ref_median"], df.loc[i, "ks_ref_pooled"], df.loc[i, "ks_ref_n_basins"] = k["ks_median"], k["ks_pooled"], k["n_basins"]
        df.loc[i, "tol_theta"] = bool(np.isfinite(row["theta_legacy"]) and abs(row["theta_legacy"] - theta) <= 0.05)
        df.loc[i, "tol_ks"] = bool(np.isfinite(k["ks_median"]) and abs(k["ks_median"] / ref["ks"] - 1) <= TOL_REL)
        df.loc[i, "tol_ks_pooled_statistic"] = bool(np.isfinite(k["ks_pooled"]) and abs(k["ks_pooled"] / ref["ks"] - 1) <= TOL_REL)
        df.loc[i, "tol_C_ht"] = bool(np.isfinite(row["C_ht"]) and abs(row["C_ht"] / ref["C_ht"] - 1) <= TOL_REL)
        df.loc[i, "tol_all"] = bool(df.loc[i, "tol_theta"] and df.loc[i, "tol_ks"] and df.loc[i, "tol_C_ht"])
        if i % 50 == 0:
            print("window", i, "of", len(df), round(time.time() - t1), "s", flush=True)
    df["theta_ref_used"] = theta
    df["ks_ref_used"] = ref["ks"]
    df.to_csv(os.path.join(a.out, "TTV_windows_a16.csv"), index=False)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
