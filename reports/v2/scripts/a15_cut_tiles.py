#!/usr/bin/env python3
"""Addendum 15, section 6: cut the drawn and original tiles from the re-routed 1 m full grids and register them.

For every tile of draw/<site>_tiles.csv: a 1,000 x 1,000-cell square on the site's 1 m lattice with its south-west corner at
(fp_x0, fp_y0); products clipped from the full-grid routing (never recomputed): filled elevation (m, float32), D8 flow
direction (uint8, lemcvt codes), drainage area (m^2, float64, = cell count at 1 m, the format of the existing tiles), and,
for AP and TTV, the class-2 count per cell, the stage A nearest-neighbour fill mask (1 = filled within 10 m, 2 = gap left
nodata in stage A) and the stage B filled-gap mask (1 = enclosed gap filled for routing).
Original tiles: AP1-AP5 as AP<n>_3dep at the original footprints (EPSG:26917 bounds reprojected to EPSG:6346, south-west
corner floored to the lattice, 1,000 m); GM1-GM5 re-cut from the full-grid routing at their footprints and compared cell by
cell with the existing area and flow-direction grids (the existing tiles are not replaced).
Unit check (6.4): z median and range inside the site window (GM 100-500, AP 230-490, TTV 0-320 m); any tile outside: stop.
Registration (6.5): data versions GM-x, AP-3DEP-x, TTV-x under $LEMCVT_DATA/<version>/<tile>/, rows appended to
data/manifest.csv, a `versions` block in configs/data.yaml. Data stay outside git.
usage: a15_cut_tiles.py --draw DIR --grids RUN_B --out DIR [--data $LEMCVT_DATA] [--no-register]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15_common import NODATA, STAGE_A, TILE_EPSG, Z_RANGE, dump, sha256_file, slurm_record  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
VERSION = {"GM": "GM-x", "AP": "AP-3DEP-x", "TTV": "TTV-x", "GMF": "GM-FEMA2018-x"}   # GMF: addendum 16 section 3.3.6
N = 1000


def write(path, arr, x0, y1, epsg, nodata=None):
    prof = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1, "dtype": str(arr.dtype), "crs": rasterio.crs.CRS.from_epsg(epsg),
            "transform": from_origin(x0, y1, 1.0, 1.0), "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def read_square(path, x0, y0):
    with rasterio.open(path) as ds:
        col = int(round((x0 - ds.transform.c) / ds.transform.a))
        row = int(round((ds.transform.f - (y0 + N)) / -ds.transform.e))
        if col < 0 or row < 0 or col + N > ds.width or row + N > ds.height:
            raise ValueError(f"footprint ({x0}, {y0}) outside {path}")
        if abs(ds.transform.c + col * ds.transform.a - x0) > 1e-6:
            raise ValueError("footprint not on the lattice")
        return ds.read(1, window=Window(col, row, N, N)), ds.nodata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draw", required=True)
    p.add_argument("--grids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--sites", default="GM,AP,TTV")
    p.add_argument("--run-label", default=None, help="run id recorded in configs/data.yaml (default: basename of --grids)")
    p.add_argument("--draw-note", default=None, help="draw provenance recorded in configs/data.yaml (default: the addendum 15 note)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    records = []
    gm_cmp = {}
    stop = []
    for site in a.sites.replace("+", ",").split(","):
        tiles = pd.read_csv(os.path.join(a.draw, f"{site}_tiles.csv"))
        epsg = TILE_EPSG[site]
        gb = os.path.join(a.grids, f"grid_{site}")
        ga = gb if site == "GMF" else os.path.join(STAGE_A, f"grid_{site}")   # GMF: its class-2 count and fill mask sit beside its routing
        vdir = os.path.join(a.data, VERSION[site])
        for _, t in tiles.iterrows():
            tid = str(t["tile"])
            if site == "AP" and t["kind"] == "original":
                tid = tid + "_3dep"
            x0, y0 = int(t["fp_x0"]), int(t["fp_y0"])
            tdir = os.path.join(vdir, tid)
            os.makedirs(tdir, exist_ok=True)
            z, nod = read_square(os.path.join(gb, f"{site}_1m_filled.tif"), x0, y0)
            fd, _ = read_square(os.path.join(gb, f"{site}_1m_fd.tif"), x0, y0)
            cnt, _ = read_square(os.path.join(gb, f"{site}_1m_area_cells.tif"), x0, y0)
            gap, _ = read_square(os.path.join(gb, f"{site}_1m_gapfill_mask.tif"), x0, y0)
            z = z.astype(np.float32)
            valid = z != NODATA
            rec = {"site": site, "tile": tid, "kind": t["kind"], "version": VERSION[site], "fp_x0": x0, "fp_y0": y0, "epsg": epsg, "offset": t.get("offset"), "window": t.get("window"),
                   "n_valid": int(valid.sum()), "frac_valid": float(valid.mean()), "z_median": float(np.median(z[valid])), "z_min": float(z[valid].min()), "z_max": float(z[valid].max()),
                   "A_max_m2": float(cnt.max()), "n_outlets_fd0": int(((fd == 0) & valid).sum()), "gapfill_cells": int((gap == 1).sum()),
                   "gapfill_channel_cells_A_ge_1e3": int(((gap == 1) & (cnt >= 1000)).sum())}
            lo, hi = Z_RANGE[site]
            rec["unit_check_pass"] = bool(lo <= rec["z_min"] and rec["z_max"] <= hi and lo <= rec["z_median"] <= hi)
            if not rec["unit_check_pass"]:
                stop.append(rec)
            files = {}
            write(os.path.join(tdir, f"{tid}_1m_filled.tif"), z, x0, y0 + N, epsg, nodata=NODATA)
            files["filled"] = f"{tid}_1m_filled.tif"
            write(os.path.join(tdir, f"{tid}_1m_fd.tif"), fd.astype(np.uint8), x0, y0 + N, epsg)
            files["fd"] = f"{tid}_1m_fd.tif"
            write(os.path.join(tdir, f"{tid}_1m_area.tif"), cnt.astype(np.float64), x0, y0 + N, epsg)
            files["area"] = f"{tid}_1m_area.tif"
            write(os.path.join(tdir, f"{tid}_1m_gapfill_mask.tif"), gap.astype(np.uint8), x0, y0 + N, epsg)
            files["gapfill_mask"] = f"{tid}_1m_gapfill_mask.tif"
            if site in ("AP", "TTV", "GMF"):
                c2, _ = read_square(os.path.join(ga, f"{site}_1m_class2_count.tif"), x0, y0)
                nn, _ = read_square(os.path.join(ga, f"{site}_1m_filled_mask.tif"), x0, y0)
                write(os.path.join(tdir, f"{tid}_1m_class2_count.tif"), c2.astype(np.uint32), x0, y0 + N, epsg)
                files["class2_count"] = f"{tid}_1m_class2_count.tif"
                write(os.path.join(tdir, f"{tid}_1m_nnfill_mask.tif"), nn.astype(np.uint8), x0, y0 + N, epsg)
                files["nnfill_mask"] = f"{tid}_1m_nnfill_mask.tif"
                rec.update({"class2_density_pts_m2": float(c2.astype(np.float64).sum() / N / N), "frac_cells_no_class2": float((c2 == 0).mean()),
                            "frac_nn_filled_le10m": float((nn == 1).mean()), "frac_gap_nodata_stage_a": float((nn == 2).mean())})
            rec["files"] = {k: {"path": f"{VERSION[site]}/{tid}/{v}", "sha256": sha256_file(os.path.join(tdir, v)), "bytes": os.path.getsize(os.path.join(tdir, v))} for k, v in files.items()}
            # GM originals: compare with the existing tile grids
            if site == "GM" and t["kind"] == "original":
                n = tid[-1]
                d = os.path.join(a.data, "GM", f"tile{n}")

                def old(role):
                    for name in (f"GM_1m_best_tile_{role}.tif", f"GM_1m_best_tile_{role}"):
                        if os.path.exists(os.path.join(d, name)):
                            return os.path.join(d, name)
                    raise FileNotFoundError(f"{d}: GM_1m_best_tile_{role}")
                with rasterio.open(old("area")) as ds:
                    a_old = ds.read(1)
                    b = ds.bounds
                with rasterio.open(old("fd")) as ds:
                    fd_old = ds.read(1)
                with rasterio.open(old("filled")) as ds:
                    z_old = ds.read(1).astype(np.float32)
                same_fp = (int(b.left), int(b.bottom)) == (x0, y0)
                a_new = cnt.astype(np.float64)
                d_ = a_new - a_old
                rel = np.abs(d_) / np.maximum(a_old, 1)
                inner = np.zeros((N, N), bool)
                inner[1:-1, 1:-1] = True
                gm_cmp[tid] = {"same_footprint": same_fp, "area_frac_equal": float((d_ == 0).mean()), "area_frac_rel_diff_gt_1pct": float((rel > 0.01).mean()),
                               "area_frac_rel_diff_gt_1pct_interior": float((rel[inner] > 0.01).mean()), "area_max_abs_diff_m2": float(np.abs(d_).max()),
                               "area_old_max": float(a_old.max()), "area_new_max": float(a_new.max()), "area_frac_new_gt_old": float((d_ > 0).mean()), "area_frac_new_lt_old": float((d_ < 0).mean()),
                               "fd_frac_equal": float((fd == fd_old).mean()), "fd_frac_equal_interior": float((fd == fd_old)[inner].mean()),
                               "fd_old_zero_cells": int((fd_old == 0).sum()), "fd_new_zero_cells": int((fd == 0).sum()),
                               "z_frac_equal_1mm": float((np.abs(z - z_old) < 1e-3).mean()), "z_max_abs_diff": float(np.abs(z - z_old).max()),
                               "z_median_diff": float(np.median(z - z_old))}
            records.append(rec)
            print(site, tid, {k: rec[k] for k in ("fp_x0", "fp_y0", "z_median", "z_min", "z_max", "A_max_m2", "unit_check_pass", "gapfill_channel_cells_A_ge_1e3")}, flush=True)
    dump({"tiles": records, "gm_comparison": gm_cmp, "unit_check_failures": stop, "seconds": time.time() - t0}, os.path.join(a.out, "tiles.json"))
    pd.DataFrame([{k: v for k, v in r.items() if k != "files"} for r in records]).to_csv(os.path.join(a.out, "tiles.csv"), index=False)
    if stop:
        print("UNIT CHECK FAILED for", [r["tile"] for r in stop], "- stop; nothing registered", flush=True)
        sys.exit(2)
    if a.no_register:
        return
    # ---- manifest rows and data.yaml versions block
    mpath = os.path.join(REPO, "data", "manifest.csv")
    rows = list(csv.DictReader(open(mpath)))
    have = {r["path"] for r in rows}
    added = 0
    for r in records:
        for role, f in r["files"].items():
            if f["path"] in have:
                continue
            rows.append({"path": f["path"], "sha256": f["sha256"], "bytes": f["bytes"], "site": r["version"], "tile": r["tile"], "kind": "input", "role": role})
            added += 1
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "sha256", "bytes", "site", "tile", "kind", "role"])
        w.writeheader()
        w.writerows(rows)
    import yaml
    ypath = os.path.join(REPO, "configs", "data.yaml")
    txt = open(ypath).read()
    cfg = yaml.safe_load(txt) or {}
    versions = cfg.get("versions", {})
    run_id = a.run_label or os.path.basename(a.grids.rstrip("/"))
    for site in a.sites.replace("+", ",").split(","):
        v = VERSION[site]
        tl = [r["tile"] for r in records if r["site"] == site]
        versions[v] = {"dir": v, "site": site, "crs_epsg": TILE_EPSG[site], "cell_m": 1.0, "tiles": tl, "n_tiles": len(tl),
                       "source": {"GM": "Gabilan_mesa.tif routed as one full grid (addendum 15 section 3)", "AP": "USGS 3DEP Entwine copy of PA_WesternPA_1_2019, class-2 mean grid, re-routed with the gap rule",
                                  "TTV": "USGS 3DEP Entwine copy of CA_NoCal_Wildfires_B5b_QL1_2018 (Marin 2019), class-2 mean grid, re-routed with the gap rule",
                                  "GMF": "USGS 3DEP Entwine copy of CA_FEMA_Z4_B2_2018 (2018 FEMA Region 9, InPort 64466), class-2 mean grid at 1 m on EPSG:6339, re-routed with the gap rule (addendum 16 section 3.3)"}[site],
                       "run": run_id, "products": ["filled (m, float32)", "fd (uint8 lemcvt codes)", "area (m^2, float64)", "gapfill_mask"] + (["class2_count", "nnfill_mask"] if site != "GM" else []),
                       "site_label": "GM" if site == "GMF" else site,
                       "draw": a.draw_note or "addendum 15 section 5 (seed 20261118); originals re-cut (GM) or cut from 3DEP (AP<n>_3dep)"}
    cfg["versions"] = versions
    versions_note = ("# `versions` (addendum 15, tile expansion stage B): tile sets cut from the re-routed 1 m full grids; files live under\n"
                     "# $LEMCVT_DATA/<dir>/<tile>/ and are listed in data/manifest.csv (site column = the version name). Data stay outside git.\n")
    header_lines = [line for line in txt.splitlines() if line.startswith("#")]
    header = "".join(line + "\n" for line in header_lines if line + "\n" not in versions_note)   # the note is appended once (a15 wrote it twice)
    with open(ypath, "w") as fh:
        fh.write(header)
        fh.write(versions_note)
        fh.write(yaml.safe_dump({k: v for k, v in cfg.items()}, sort_keys=False, width=120))
    print("registered:", added, "manifest rows;", {v: versions[v]["n_tiles"] for v in versions}, flush=True)


if __name__ == "__main__":
    main()
