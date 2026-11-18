#!/usr/bin/env python3
"""Addendum 17, section 2.2.4 (report only): at the GM2 and GM5 trunk entries of addendum 16 section 4, the FEMA full-grid
drainage area (grid_GMC, routed over the tiles' whole catchments) against the old tile grids; a third, independent check of
decision 2. For every entry of the addendum 16 check (all five originals): the FEMA area at the entry cell (the cell itself, the
maximum over 3 x 3 and over 11 x 11 neighbourhoods, since the two surveys' channels can sit a few metres apart), the old grid's
area, the lidar-only (stage B) area and D = old - new. Also, for GM1-GM7, the largest area inside the footprint on the FEMA grid
against the registered area grid (area_model for GM1-GM5, area for GM6-GM7). Output: gm_ctrl_area_check.json.
usage: a17_gm_ctrl_area_check.py --out DIR --grid DIR --check gm_area_check.json [--data $LEMCVT_DATA]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17_common import CTRL_TILES, dump, slurm_record  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    chk = json.load(open(a.check))
    ds = rasterio.open(os.path.join(a.grid, "GMC_1m_area_cells.tif"))
    tr = ds.transform

    def fema_at(E, N, half):
        c = int(np.floor((E - tr.c) / tr.a))
        r = int(np.floor((tr.f - N) / -tr.e))
        w = Window(c - half, r - half, 2 * half + 1, 2 * half + 1)
        arr = ds.read(1, window=w).astype(np.float64)
        k = int(np.argmax(arr))
        rr, cc = divmod(k, arr.shape[1])
        return float(arr.max()), [int(rr - half), int(cc - half)]

    out = {"entries": {}, "tiles": {}}
    for t, rec in chk["tiles"].items():
        n = t[-1]
        with rasterio.open(os.path.join(a.data, "GM", f"tile{n}", "GM_1m_best_tile_area.tif")) as d:
            old = d.read(1).astype(np.float64)
        ents = []
        for e in rec["entries"]:
            E, N = e["tile_cell_EN"]
            r, c = e["tile_cell_rc"]
            a0, _ = fema_at(E, N, 0)
            a1, off1 = fema_at(E, N, 1)
            a5, off5 = fema_at(E, N, 5)
            D = e["D_old_minus_new_m2"]
            o = float(old[r, c])
            ents.append({"tile_cell_EN": [E, N], "D_old_minus_new_m2": D, "old_m2": o, "new_lidar_only_m2": e["new_m2"], "perron_strip_inflow_m2": e["coarse_outside_inflow_at_tile_entry_m2"],
                         "fema_at_cell_m2": a0, "fema_max_3x3_m2": a1, "fema_max_3x3_offset_rc": off1, "fema_max_11x11_m2": a5, "fema_max_11x11_offset_rc": off5,
                         "fema_3x3_vs_old_rel": (a1 - o) / o if o > 0 else None, "fema_11x11_vs_old_rel": (a5 - o) / o if o > 0 else None,
                         "fema_3x3_minus_new_vs_D_rel": (a1 - e["new_m2"] - D) / D if D > 1e4 else None, "trunk_ge_1e5": bool(D >= 1e5)})
        out["entries"][t] = ents
        for e in ents:
            print(t, json.dumps(e), flush=True)
    # per tile maxima inside the footprint
    for t in CTRL_TILES:
        role = "area_model" if t in ("GM1", "GM2", "GM3", "GM4", "GM5") else "area"
        with rasterio.open(os.path.join(a.data, "GM-x", t, f"{t}_1m_{role}.tif")) as d:
            reg = d.read(1).astype(np.float64)
            b = d.bounds
        c = int(np.floor((b.left - tr.c) / tr.a))
        r = int(np.floor((tr.f - b.top) / -tr.e))
        fema = ds.read(1, window=Window(c, r, 1000, 1000)).astype(np.float64)
        k = int(np.argmax(fema))
        kr = int(np.argmax(reg))
        out["tiles"][t] = {"registered_role": role, "registered_max_m2": float(reg.max()), "registered_argmax_rc": [kr // 1000, kr % 1000], "fema_max_m2": float(fema.max()), "fema_argmax_rc": [k // 1000, k % 1000],
                           "rel_diff": float((fema.max() - reg.max()) / reg.max()), "fema_cells_ge_1e5": int((fema >= 1e5).sum()), "registered_cells_ge_1e5": int((reg >= 1e5).sum())}
        print(t, out["tiles"][t], flush=True)
    out["seconds"] = time.time() - t0
    dump(out, os.path.join(a.out, "gm_ctrl_area_check.json"))


if __name__ == "__main__":
    main()
