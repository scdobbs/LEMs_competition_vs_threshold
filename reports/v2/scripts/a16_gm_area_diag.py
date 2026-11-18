#!/usr/bin/env python3
"""Addendum 16, section 4 diagnostic (not part of the rule): the GM area check against the NED routed over the whole box.

The check of a16_gm_area_check.py takes the coarse inflow from the stage A routing of Perron's 10 m grid, whose valid area
is a 58 km2 strip around the lidar (gm_footprints map); basins that extend beyond that strip are truncated there. This
script routes the a16 NED crop (gm_footprints/GM_NED_10m_utm10n.tif over GM_GATE_CROP_UTM, every valid cell, data-edge
outlets) and recomputes, at the same tile-entry cells, the area the NED routes in from outside the NCALM footprint
(A - A_inside_only), so the truncation hypothesis can be tested. It also draws, for GM2 and GM5, the channels of the old
tile grid, the new full-grid routing (stage B), the Perron-strip routing and the NED-box routing around the tile.
Nothing registered here changes; the area_model grids of section 4 follow the rule as written.
usage: a16_gm_area_diag.py --out DIR --footprints DIR --check gm_area_check.json [--data $LEMCVT_DATA]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_pools import route_full_grid  # noqa: E402
from a16_common import GM_GATE_CROP_UTM, STAGE_A, STAGE_B, dump, slurm_record  # noqa: E402


def read_crop(path, box):
    with rasterio.open(path) as ds:
        b = ds.bounds
        assert b.left <= box[0] and b.bottom <= box[1] and box[2] <= b.right and box[3] <= b.top, (path, box, tuple(b))   # the crop must lie inside the raster
        w = from_bounds(box[0], box[1], box[2], box[3], ds.transform)
        return ds.read(1, window=w), ds.window_transform(w), ds.nodata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--footprints", required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    box = GM_GATE_CROP_UTM
    z, tr, nod = read_crop(os.path.join(a.footprints, "GM_NED_10m_utm10n.tif"), box)
    z = z.astype(np.float64)
    z[z == nod] = np.nan
    frac, _, _ = read_crop(os.path.join(a.footprints, "GM_ncalm_valid_frac_10m.tif"), box)
    inside = frac > 0
    dx = float(tr.a)
    ny, nx = z.shape
    R = route_full_grid(z, dx)
    area, tgt = R["area"], R["target"]
    print("NED box routed", z.shape, round(R["seconds"]), "s; max area km2", float(np.nanmax(area)) / 1e6, flush=True)
    order = np.argsort(np.nan_to_num(area, nan=-1).ravel(), kind="stable")       # ascending area: donors before receivers
    acc = np.where(inside & np.isfinite(z), dx * dx, 0.0).ravel()
    for k in order:
        t = tgt[k]
        if t >= 0:
            acc[t] += acc[k]
    ext = np.where(np.isfinite(area), area - acc.reshape(ny, nx), np.nan)
    ext[~inside] = np.nan
    chk = json.load(open(a.check))
    out = {"crop": box, "seconds_routing": R["seconds"], "tiles": {}}

    def at(E, N, arr):
        cc = int(np.floor((E - tr.c) / tr.a))
        rr = int(np.floor((tr.f - N) / -tr.e))
        win = arr[max(rr - 1, 0):rr + 2, max(cc - 1, 0):cc + 2]
        return float(np.nanmax(win)) if np.isfinite(win).any() else np.nan
    for t, rec in chk["tiles"].items():
        ents = []
        for e in rec["entries"]:
            E, N = e["tile_cell_EN"]
            ext_ned = at(E, N, ext)
            A_ned = at(E, N, area)
            d = e["D_old_minus_new_m2"]
            ents.append({"tile_cell_EN": [E, N], "D_old_minus_new_m2": d, "perron_strip_outside_inflow_m2": e["coarse_outside_inflow_at_tile_entry_m2"], "perron_rel_diff": e["rel_diff"],
                         "ned_box_outside_inflow_m2": ext_ned, "ned_box_area_m2": A_ned, "ned_rel_diff": abs(d - ext_ned) / d if d > 0 and np.isfinite(ext_ned) else np.nan,
                         "ned_pass_10pct": bool(np.isfinite(ext_ned) and abs(d - ext_ned) / d <= 0.10) if d > 0 else None})
        out["tiles"][t] = {"entries": ents, "all_pass_ned": all(x["ned_pass_10pct"] for x in ents) if ents else True}
        print(t, json.dumps(ents), flush=True)
    dump(out, os.path.join(a.out, "gm_area_diag.json"))
    # figures for GM2 and GM5: channels (A >= 1e5 m2) of the four routings within 2.5 km of the tile
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        with rasterio.open(os.path.join(STAGE_A, "pools", "GM_area_full.tif")) as ds:
            Ap_full = ds.read(1).astype(np.float64)
            trp = ds.transform
            Ap_full[Ap_full == ds.nodata] = np.nan
        for t in ("GM2", "GM5", "GM1"):
            fp = chk["tiles"][t]["fp"]
            bx = (fp[0] - 1500, fp[1] - 1500, fp[0] + 2500, fp[1] + 2500)
            with rasterio.open(os.path.join(STAGE_B, "grid_GM", "GM_1m_area_cells.tif")) as ds:
                w = from_bounds(bx[0], bx[1], bx[2], bx[3], ds.transform)
                a_new = ds.read(1, window=w).astype(np.float64)
                trn = ds.window_transform(w)
            with rasterio.open(os.path.join(a.data, "GM", f"tile{t[-1]}", "GM_1m_best_tile_area.tif")) as ds:
                a_old = ds.read(1).astype(np.float64)
                tro = ds.transform
            fig, ax = plt.subplots(figsize=(11, 11), dpi=110)
            ext_n = [trn.c, trn.c + a_new.shape[1] * trn.a, trn.f + a_new.shape[0] * trn.e, trn.f]
            ax.imshow(np.log10(np.maximum(a_new, 1)), cmap="gray", extent=ext_n, vmin=0, vmax=7)
            yy, xx = np.indices(a_new.shape)
            Xn = trn.c + (xx + 0.5) * trn.a
            Yn = trn.f + (yy + 0.5) * trn.e
            ax.contour(Xn, Yn, (a_new >= 1e5).astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
            yy, xx = np.indices(a_old.shape)
            Xo = tro.c + (xx + 0.5) * tro.a
            Yo = tro.f + (yy + 0.5) * tro.e
            ax.contour(Xo, Yo, (a_old >= 1e5).astype(float), levels=[0.5], colors="red", linewidths=0.8)
            for arr, trr, col in ((Ap_full, trp, "yellow"), (area, tr, "lime")):      # Perron strip, NED box
                c0 = int(np.floor((bx[0] - trr.c) / trr.a))
                c1 = int(np.ceil((bx[2] - trr.c) / trr.a))
                r0 = int(np.floor((trr.f - bx[3]) / -trr.e))
                r1 = int(np.ceil((trr.f - bx[1]) / -trr.e))
                sub = arr[max(r0, 0):r1, max(c0, 0):c1]
                yy, xx = np.indices(sub.shape)
                X = trr.c + (max(c0, 0) + xx + 0.5) * trr.a
                Y = trr.f + (max(r0, 0) + yy + 0.5) * trr.e
                ax.contour(X, Y, (np.nan_to_num(sub) >= 1e5).astype(float), levels=[0.5], colors=col, linewidths=0.8)
            ax.add_patch(Rectangle((fp[0], fp[1]), 1000, 1000, fill=False, ec="white", lw=1.2))
            for e in chk["tiles"][t]["entries"]:
                ax.plot(e["tile_cell_EN"][0], e["tile_cell_EN"][1], "o", color="magenta", ms=6)
                ax.text(e["tile_cell_EN"][0], e["tile_cell_EN"][1] + 40, f"D {e['D_old_minus_new_m2'] / 1e6:.2f} km2", color="magenta", fontsize=7)
            ax.set_title(f"{t}: channels A >= 0.1 km2: old tile grid (red), stage B 1 m routing (cyan), Perron 10 m strip (yellow), NED box (lime); entries magenta")
            ax.set_xlim(bx[0], bx[2])
            ax.set_ylim(bx[1], bx[3])
            fig.savefig(os.path.join(a.out, f"{t}_area_channels.png"), bbox_inches="tight")
            plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("figures failed", repr(e), flush=True)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
