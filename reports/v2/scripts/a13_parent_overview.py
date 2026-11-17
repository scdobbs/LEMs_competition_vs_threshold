#!/usr/bin/env python3
"""Addendum 13, section 2.2.1 (drainage-area context): where the reprojected PAMAP grid has data around the AP tiles.

Reads the parent at 1/16 resolution for an overview figure with the tile outlines, and counts nodata cells within 0.5, 1, 2 and
5 km of each tile at full resolution. usage: a13_parent_overview.py --parent TIF --data LEMCVT_DATA --out DIR
"""
import argparse, json, os
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds


def main():
    p = argparse.ArgumentParser(); p.add_argument("--parent", required=True); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    tiles = {}
    for n in range(1, 6):
        with rasterio.open(os.path.join(a.data, f"AP/tile{n}/AP_1m_best_tile_filled_m.tif")) as t:
            tiles[f"AP{n}"] = list(t.bounds)
    res = {}
    with rasterio.open(a.parent) as ds:
        f = 16
        ov = ds.read(1, out_shape=(ds.height // f, ds.width // f), resampling=rasterio.enums.Resampling.nearest).astype(np.float64)
        valid = np.isfinite(ov) & (ov != ds.nodata)
        for t, b in tiles.items():
            r = {}
            for buf in (500, 1000, 2000, 5000):
                w = from_bounds(max(b[0] - buf, ds.bounds.left), max(b[1] - buf, ds.bounds.bottom), min(b[2] + buf, ds.bounds.right), min(b[3] + buf, ds.bounds.top), ds.transform)
                w = Window(int(w.col_off), int(w.row_off), int(w.width), int(w.height))
                A = ds.read(1, window=w)
                nod = ~(np.isfinite(A) & (A != ds.nodata))
                r[f"buffer_{buf}m"] = {"cells": int(A.size), "nodata": int(nod.sum()), "frac_nodata": float(nod.mean()),
                                      "nodata_by_side": {"north_half": float(nod[: nod.shape[0] // 2].mean()), "south_half": float(nod[nod.shape[0] // 2:].mean()),
                                                         "west_half": float(nod[:, : nod.shape[1] // 2].mean()), "east_half": float(nod[:, nod.shape[1] // 2:].mean())}}
            res[t] = r
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 7))
            ext = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
            ax.imshow(np.where(valid, ov, np.nan), extent=ext, cmap="terrain"); ax.imshow(np.where(valid, np.nan, 1.0), extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.6)
            for t, b in tiles.items():
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="red", lw=1.2)); ax.text(b[2] + 300, b[3], t, color="red", fontsize=9)
            ax.set_title("PAMAP_DEM_FIXED_REPROJECT_12-2025.tif at 1/16: data (colour), nodata (grey), AP tiles (red)"); ax.set_xlabel("E (m, EPSG:26917)"); ax.set_ylabel("N (m)")
            fig.savefig(os.path.join(a.out, "pamap_parent_overview.png"), dpi=130); plt.close(fig)
        except Exception as e:
            res["figure_error"] = repr(e)
    json.dump(res, open(os.path.join(a.out, "parent_nodata_around_tiles.json"), "w"), indent=1)
    for t, r in res.items():
        if t.startswith("AP"):
            print(t, {k: round(v["frac_nodata"], 3) for k, v in r.items()}, "1km by side", {k: round(v, 2) for k, v in r["buffer_1000m"]["nodata_by_side"].items()})


if __name__ == "__main__":
    main()
