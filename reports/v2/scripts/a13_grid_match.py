#!/usr/bin/env python3
"""Addendum 13, section 2.2.1-2.2.2: are the AP and GM model tiles cut from the grids Steve placed in lem_topography?

For each tile (filled grid in the data root, LEMCVT_DATA layout) and its candidate parent grid:
  * locate the tile in the parent (window of the parent covering the tile's bounds plus a margin);
  * test lattice alignment: the tile's origin offset from the parent's origin in parent pixels (fractional part) and the
    ratio of pixel sizes;
  * sample the parent at the tile's cell centres by nearest neighbour and bilinear interpolation (scipy.ndimage), and
    compare with the tile cell by cell, in metres: the tile is converted with the stated z units (AP: ftUS -> m by
    1200/3937; the parent's units are tested from the ratio of medians and reported, never assumed);
  * report max |dz|, the count of cells differing by more than 1e-3, 0.01, 0.1, 1 and 5 m, and where they are (a 10 x 10
    block count grid and the row/column extent of the differing cells);
  * report the parent's margin around the tile on each side and the nodata count within the tile and within a 1 km buffer.

usage: a13_grid_match.py --data $LEMCVT_DATA --ap-parent PAMAP.tif --gm-parent Gabilan.tif --out DIR
"""
import argparse, json, os
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from scipy.ndimage import map_coordinates

FT_US = 1200.0 / 3937.0
AP_TILES = {f"AP{i}": f"AP/tile{i}/AP_1m_best_tile_filled.tif" for i in range(1, 6)}
AP_TILES_M = {f"AP{i}": f"AP/tile{i}/AP_1m_best_tile_filled_m.tif" for i in range(1, 6)}
GM_TILES = {f"GM{i}": f"GM/tile{i}/GM_1m_best_tile_filled.tif" for i in range(1, 6)}


def valid(a, nodata):
    ok = np.isfinite(a)
    if nodata is not None and not np.isnan(nodata):
        ok &= a != nodata
    return ok


def compare(tile_name, tile_path, parent, z_factor_tile, out, tile_m_path=None):
    rec = {"tile": tile_name, "tile_path": tile_path, "parent": parent.name}
    with rasterio.open(tile_path) as t:
        Zt = t.read(1).astype(np.float64); tt = t.transform; tb = t.bounds
        rec.update({"tile_crs_epsg": t.crs.to_epsg(), "tile_shape": [t.height, t.width], "tile_res": [tt.a, -tt.e], "tile_bounds": list(tb), "tile_nodata": t.nodata})
        okt = valid(Zt, t.nodata)
    if tile_m_path and os.path.exists(tile_m_path):
        with rasterio.open(tile_m_path) as tm:
            Zm = tm.read(1).astype(np.float64)
            okm = valid(Zm, tm.nodata) & okt
            rec["tile_m_over_tile_median_ratio"] = float(np.median(Zm[okm] / Zt[okm])) if okm.any() else None
            rec["tile_m_vs_tile_times_1200_3937_max_abs"] = float(np.nanmax(np.abs(Zm[okm] - Zt[okm] * FT_US))) if okm.any() else None
            rec["tile_m_vs_tile_times_0.3048_max_abs"] = float(np.nanmax(np.abs(Zm[okm] - Zt[okm] * 0.3048))) if okm.any() else None
    rec["parent_crs_epsg"] = parent.crs.to_epsg(); rec["parent_res"] = [parent.res[0], parent.res[1]]; rec["parent_bounds"] = list(parent.bounds)
    rec["same_crs"] = parent.crs == rasterio.open(tile_path).crs
    pb = parent.bounds
    rec["margin_m"] = {"west": tb.left - pb.left, "east": pb.right - tb.right, "south": tb.bottom - pb.bottom, "north": pb.top - tb.top}
    rec["tile_inside_parent"] = all(v >= 0 for v in rec["margin_m"].values())
    pt = parent.transform
    off_col = (tt.c - pt.c) / pt.a; off_row = (tt.f - pt.f) / pt.e
    rec["origin_offset_parent_pixels"] = {"col": off_col, "row": off_row, "col_frac": off_col - np.floor(off_col), "row_frac": off_row - np.floor(off_row)}
    rec["res_ratio_tile_over_parent"] = [tt.a / pt.a, tt.e / pt.e]
    if not rec["tile_inside_parent"]:
        return rec
    # read the parent over the tile plus a margin of 1 km (for nodata) and the exact window (for sampling)
    buf = 1000.0
    wb = from_bounds(max(tb.left - buf, pb.left), max(tb.bottom - buf, pb.bottom), min(tb.right + buf, pb.right), min(tb.top + buf, pb.top), pt)
    wb = Window(int(np.floor(wb.col_off)), int(np.floor(wb.row_off)), int(np.ceil(wb.width)) + 1, int(np.ceil(wb.height)) + 1)
    Pb = parent.read(1, window=wb).astype(np.float64)
    okb = valid(Pb, parent.nodata)
    rec["parent_nodata_within_1km_buffer"] = int((~okb).sum()); rec["parent_cells_within_1km_buffer"] = int(Pb.size)
    w = from_bounds(tb.left, tb.bottom, tb.right, tb.top, pt)
    c0, r0 = int(np.floor(w.col_off)) - 2, int(np.floor(w.row_off)) - 2
    w = Window(c0, r0, int(np.ceil(w.width)) + 5, int(np.ceil(w.height)) + 5)
    P = parent.read(1, window=w).astype(np.float64)
    okp = valid(P, parent.nodata)
    rec["parent_nodata_within_tile_window"] = int((~okp).sum())
    P = np.where(okp, P, np.nan)
    # tile cell centres in parent pixel coordinates
    rows, cols = np.mgrid[0:Zt.shape[0], 0:Zt.shape[1]]
    xs = tt.c + (cols + 0.5) * tt.a; ys = tt.f + (rows + 0.5) * tt.e
    pc = (xs - pt.c) / pt.a - 0.5 - c0; pr = (ys - pt.f) / pt.e - 0.5 - r0
    near = map_coordinates(P, [pr, pc], order=0, mode="constant", cval=np.nan)
    bil = map_coordinates(np.nan_to_num(P, nan=0.0), [pr, pc], order=1, mode="constant", cval=np.nan)
    wgt = map_coordinates(okp.astype(np.float64), [pr, pc], order=1, mode="constant", cval=0.0)
    bil = np.where(wgt > 0.999, bil / np.maximum(wgt, 1e-12), np.nan)
    ok = okt & np.isfinite(near)
    Zt_m = Zt * z_factor_tile
    # parent units: ratio of medians against the tile in metres
    ratio = float(np.median(near[ok]) / np.median(Zt_m[ok]))
    rec["parent_over_tile_m_median_ratio"] = ratio
    if abs(ratio - 1) < 0.05:
        pf, punits = 1.0, "metres (ratio to the tile in metres within 5 %)"
    elif abs(ratio - 1 / FT_US) < 0.05 * (1 / FT_US):
        pf, punits = FT_US, "US survey feet (ratio to the tile in metres within 5 % of 3.2808)"
    else:
        pf, punits = 1.0, f"undetermined (ratio {ratio:.4f}); compared without conversion"
    rec["parent_units_inferred"] = punits
    for name, S in (("nearest", near), ("bilinear", bil)):
        okk = okt & np.isfinite(S)
        d = S[okk] * pf - Zt_m[okk]
        r = {"n_compared": int(okk.sum()), "max_abs_dz_m": float(np.max(np.abs(d))) if d.size else None, "median_dz_m": float(np.median(d)) if d.size else None,
             "p05_dz_m": float(np.percentile(d, 5)) if d.size else None, "p95_dz_m": float(np.percentile(d, 95)) if d.size else None,
             "rms_dz_m": float(np.sqrt(np.mean(d * d))) if d.size else None,
             "n_abs_dz_gt": {str(th): int((np.abs(d) > th).sum()) for th in (1e-3, 1e-2, 0.1, 1.0, 5.0)},
             "n_exact_equal": int((d == 0).sum())}
        D = np.full(Zt.shape, np.nan); D[okk] = d
        big = np.abs(D) > 0.1
        if big.any():
            rr, cc = np.nonzero(big)
            r["where_abs_dz_gt_0.1m"] = {"row_min": int(rr.min()), "row_max": int(rr.max()), "col_min": int(cc.min()), "col_max": int(cc.max()),
                                         "block10x10_counts": np.add.reduceat(np.add.reduceat(big.astype(int), np.linspace(0, Zt.shape[0], 11, dtype=int)[:-1], axis=0),
                                                                              np.linspace(0, Zt.shape[1], 11, dtype=int)[:-1], axis=1).tolist(),
                                         "frac_within_5_cells_of_edge": float(np.mean((rr < 5) | (rr >= Zt.shape[0] - 5) | (cc < 5) | (cc >= Zt.shape[1] - 5))),
                                         "sign": {"parent_higher": int((D[big] > 0).sum()), "tile_higher": int((D[big] < 0).sum())}}
        rec[name] = r
        np.save(os.path.join(out, f"{tile_name}_dz_{name}.npy"), D.astype(np.float32))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        D = np.load(os.path.join(out, f"{tile_name}_dz_bilinear.npy"))
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        im = ax[0].imshow(D, cmap="RdBu_r", vmin=-2, vmax=2); fig.colorbar(im, ax=ax[0], label="parent - tile (m), bilinear")
        ax[0].set_title(f"{tile_name}: dz bilinear"); ax[1].imshow(np.abs(D) > 0.1, cmap="gray_r"); ax[1].set_title("|dz| > 0.1 m")
        fig.savefig(os.path.join(out, f"{tile_name}_dz_map.png"), dpi=110); plt.close(fig)
    except Exception as e:
        rec["figure_error"] = repr(e)
    return rec


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--ap-parent"); p.add_argument("--gm-parent"); p.add_argument("--out", required=True)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = {}
    if a.ap_parent:
        with rasterio.open(a.ap_parent) as parent:
            for t, rel in AP_TILES.items():
                res[t] = compare(t, os.path.join(a.data, rel), parent, FT_US, a.out, os.path.join(a.data, AP_TILES_M[t])); print(t, "done", flush=True)
    if a.gm_parent:
        with rasterio.open(a.gm_parent) as parent:
            for t, rel in GM_TILES.items():
                res[t] = compare(t, os.path.join(a.data, rel), parent, 1.0, a.out); print(t, "done", flush=True)
    json.dump(res, open(os.path.join(a.out, "grid_match.json"), "w"), indent=1, default=str)
    for t, r in res.items():
        n, b = r.get("nearest", {}), r.get("bilinear", {})
        print(t, "inside", r["tile_inside_parent"], "offset frac", r["origin_offset_parent_pixels"]["col_frac"], r["origin_offset_parent_pixels"]["row_frac"],
              "res ratio", r["res_ratio_tile_over_parent"], "units", r.get("parent_units_inferred"), "| nearest max", n.get("max_abs_dz_m"), "gt0.01", n.get("n_abs_dz_gt", {}).get("0.01"),
              "| bilinear max", b.get("max_abs_dz_m"), "median", b.get("median_dz_m"), "gt0.1", b.get("n_abs_dz_gt", {}).get("0.1"))


if __name__ == "__main__":
    main()
