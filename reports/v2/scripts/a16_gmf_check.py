#!/usr/bin/env python3
"""Addendum 16, section 3.3.3: units and datum of the GM extension grid (GMF, the 2018 FEMA survey at 1 m).

Rule 5.3 of stage A against the 10 m NED (both NAVD88): the NED (a16 warp, EPSG:32610) is compared with the GMF grid block-averaged
to its 10 m cells (bin_to_lattice on the ground cells only, i.e. no nearest-neighbour fill): |median offset| < 5 m, Theil-Sen slope
0.98-1.02. Then the NCALM 2003 grid (Gabilan_mesa.tif, EPSG:32610, ellipsoidal) on the overlap: the GMF grid sampled at the NCALM cell
centres (bilinear, ground cells), expect an offset of about 30-34 m in magnitude (NCALM, ellipsoidal, about 32 m below the NAVD88 grid) and a slope of 0.98-1.02; the work
stops on a slope near 3.28 or 0.305 (a feet/metres error). The EPSG:32610 -> EPSG:6339 shift is the identity offline (PROJ has no
grid), as in stages A and B. Output: gmf_check.json (run dir grid_GMF/).
usage: a16_gmf_check.py --grid DIR --footprints DIR --out DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import SITES, bin_to_lattice, dz_stats, theil_sen_sub  # noqa: E402
from a16_common import dump, slurm_record  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--footprints", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--site", default="GMF", help="grid prefix: GMF (addendum 16) or GMC (the addendum 17 control grid)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    rec = {}
    mean_path = os.path.join(a.grid, f"{a.site}_1m_mean.tif")
    mask_path = os.path.join(a.grid, f"{a.site}_1m_filled_mask.tif")
    with rasterio.open(mean_path) as ds:
        gtr, gcrs, gnod = ds.transform, ds.crs, ds.nodata
        gshape = (ds.height, ds.width)
    # ground-only copy of the grid (NN-filled cells removed) written to the run dir for the two comparisons
    ground_path = os.path.join(a.out, f"_{a.site}_1m_ground.tif")
    with rasterio.open(mean_path) as dm, rasterio.open(mask_path) as dk:
        prof = dm.profile
        prof.update(compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(ground_path, "w", **prof) as dst:
            for _, w in dm.block_windows(1):
                z = dm.read(1, window=w)
                k = dk.read(1, window=w)
                z[k != 0] = gnod
                dst.write(z, 1, window=w)
    # 1. rule 5.3: NED 10 m against the block-mean of the ground cells (the NED is EPSG:32610; the grid EPSG:6339: identity offline)
    with rasterio.open(os.path.join(a.footprints, "GM_NED_10m_utm10n.tif")) as ds:
        ned = ds.read(1).astype(np.float64)
        ntr = ds.transform
        ned[ned == ds.nodata] = np.nan
    B = bin_to_lattice(ground_path, ntr, ned.shape, valid_fn=lambda arr: arr != gnod)
    full = B["n_total"] >= 100
    ok = full & (B["n_valid"] == B["n_total"]) & np.isfinite(ned)
    bm = np.where(ok, B["sum_z"] / np.maximum(B["n_valid"], 1), np.nan)
    dz = (ned - bm)[ok]
    fit = theil_sen_sub(bm[ok], ned[ok])
    med = float(np.median(dz))
    rec["rule_5_3_ned_minus_gmf"] = {"n_cells_10m": int(ok.sum()), "dz": dz_stats(dz), "fit_ned_on_gmf": fit, "median_pass_abs_lt_5m": bool(abs(med) < 5.0),
                                     "slope_pass_0.98_1.02": bool(0.98 <= fit["theil_sen_slope"] <= 1.02)}
    rec["rule_5_3_ned_minus_gmf"]["pass"] = rec["rule_5_3_ned_minus_gmf"]["median_pass_abs_lt_5m"] and rec["rule_5_3_ned_minus_gmf"]["slope_pass_0.98_1.02"]
    print("rule 5.3", med, fit["theil_sen_slope"], rec["rule_5_3_ned_minus_gmf"]["pass"], flush=True)
    # 2. NCALM 2003 on the overlap: the GMF ground grid sampled at NCALM cell centres (bilinear)
    from scipy.ndimage import map_coordinates
    ncalm = SITES["GM"]["lidar_record"]
    with rasterio.open(ncalm) as ds, rasterio.open(ground_path) as dg:
        # work on the overlap window in NCALM coordinates (EPSG:32610 = EPSG:6339 offline)
        gb = dg.bounds
        nb = ds.bounds
        x0, x1 = max(gb.left, nb.left), min(gb.right, nb.right)
        y0, y1 = max(gb.bottom, nb.bottom), min(gb.top, nb.top)
        if x1 - x0 < 100 or y1 - y0 < 100:
            rec["ncalm_overlap"] = {"n": 0, "note": "no overlap"}
        else:
            from rasterio.windows import from_bounds
            wn = from_bounds(x0, y0, x1, y1, ds.transform)
            zn = ds.read(1, window=wn).astype(np.float64)
            ntr2 = ds.window_transform(wn)
            zn[zn < -1e30] = np.nan
            wg = from_bounds(x0, y0, x1, y1, dg.transform)
            zg = dg.read(1, window=wg).astype(np.float64)
            gtr2 = dg.window_transform(wg)
            zg[zg == gnod] = np.nan
            # subsample every 5th NCALM cell to keep the comparison light (the overlap is up to 39 km2)
            rr, cc = np.mgrid[0:zn.shape[0]:5, 0:zn.shape[1]:5]
            xs = ntr2.c + (cc.ravel() + 0.5) * ntr2.a
            ys = ntr2.f + (rr.ravel() + 0.5) * ntr2.e
            fc = (xs - gtr2.c) / gtr2.a - 0.5
            fr = (ys - gtr2.f) / gtr2.e - 0.5
            v = map_coordinates(zg, [fr, fc], order=1, mode="constant", cval=np.nan)
            zz = zn[rr, cc].ravel()
            ok2 = np.isfinite(v) & np.isfinite(zz)
            d2 = zz[ok2] - v[ok2]
            fit2 = theil_sen_sub(v[ok2], zz[ok2])
            med2 = float(np.median(d2))
            rec["ncalm_overlap"] = {"n": int(ok2.sum()), "sampling": "every 5th NCALM cell, bilinear in the GMF ground grid", "dz_ncalm_minus_gmf": dz_stats(d2), "fit_ncalm_on_gmf": fit2,
                                    "offset_magnitude_in_30_34m": bool(30.0 <= abs(med2) <= 34.0), "slope_pass_0.98_1.02": bool(0.98 <= fit2["theil_sen_slope"] <= 1.02),
                                    "stop_slope_near_3.28_or_0.305": bool(abs(fit2["theil_sen_slope"] - 3.28) < 0.2 or abs(fit2["theil_sen_slope"] - 0.305) < 0.05),
                                    "stage_a_rule_5_3_gm_offset_ned_minus_ncalm_blockmean": 32.27}
            # NCALM (ellipsoidal) lies about 32 m BELOW NAVD88 here (stage A rule 5.3: NED - lidar = +32.3 m), so dz is negative; the first run
            # (job 52856556) tested 30 <= dz <= 34 without the absolute value and wrote offset_in_30_34m = False for a -33.8 m offset (reported)
            print("NCALM overlap", med2, fit2["theil_sen_slope"], flush=True)
    rec["stop"] = bool(rec.get("ncalm_overlap", {}).get("stop_slope_near_3.28_or_0.305", False)) or not rec["rule_5_3_ned_minus_gmf"]["slope_pass_0.98_1.02"]
    rec["grid"] = {"path": mean_path, "shape": list(gshape), "transform": list(gtr)[:6], "epsg": gcrs.to_epsg()}
    rec["seconds"] = time.time() - t0
    os.remove(ground_path)
    dump(rec, os.path.join(a.out, f"{a.site.lower()}_check.json"))
    print(json.dumps({k: v for k, v in rec.items() if k != "grid"}, indent=1), flush=True)
    if rec["stop"]:
        print("STOP: units/datum check failed", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
