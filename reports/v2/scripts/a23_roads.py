#!/usr/bin/env python3
"""Addendum 23, Part A item 2 (decision 2.2): is the AD gorge line on TTV2 a road?

On the addendum 22 TTV2 surfaces: the share of the AD model's too-steep-wall extra heads (25 m; addendum 22 section 6) within 15 m of a
TIGER road or trail; the share of the observed 1 m D8 trunk flow path (cells with drainage area >= 1e5 m2 inside the polygon) within
5 m of a road; both at 5, 15 and 25 m as sensitivities; the reading "road-influenced". Also the TIGER feature list on the tile and the
statement that the file carries no trail class there. Outputs <run>/roads/road_check.json and the arrays for figure 2 (roads/road_check.npz).
usage: a23_roads.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_hilltops import md_polygon_mask  # noqa: E402
from a23_common import (  # noqa: E402
    BLOCK_M,
    ROAD_BUFFER_SENS_M,
    ROAD_HALF,
    ROAD_HEAD_BUFFER_M,
    ROAD_TRUNK_BUFFER_M,
    T_PRIMARY_M,
    TRUNK_A_M2,
    a22_path,
    dump,
    json_ready,
    load_json,
    metre_tile,
    out_path,
    slurm_record,
    tiger_roads_in_box,
)


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("roads", "road_check.json")))
    from scipy.ndimage import distance_transform_edt
    td = metre_tile("TTV2")
    Z, A = td.Z, td.A
    ny, nx = Z.shape
    tr = td.filled.transform
    x0, y1 = tr.c, tr.f
    rd = tiger_roads_in_box(x0, y1, nx, ny)
    dist = distance_transform_edt(~rd["cells"]) if rd["cells"].any() else np.full(Z.shape, np.inf)
    poly = md_polygon_mask(td)
    metrics = load_json(a22_path("post", "metrics.json"))["metrics"]["per_tile"]["primary"]["TTV2"]["models"]["AD"]["by_t"][f"{T_PRIMARY_M:g}"]
    wall = [(c["r"], c["c"]) for c in metrics["extra_cells"] if c["class"] == "too-steep wall"]
    extra_all = [(c["r"], c["c"]) for c in metrics["extra_cells"]]
    out: dict[str, Any] = {"tile": "TTV2", "roads": {k: rd[k] for k in ("features", "n_features", "mtfcc", "source")}, "n_road_cells_on_tile": int(rd["cells"].sum()),
                           "trail_classes_present": sorted(k for k in rd["mtfcc"] if k in ("S1500", "S1710", "S1820", "S1830")), "n_wall_heads": len(wall), "n_extra_heads": len(extra_all)}

    def share_near(cells: list[tuple[int, int]], buf: float) -> dict[str, Any]:
        if not cells:
            return {"n": 0, "n_within": 0, "share": None}
        rr = np.clip(np.array([r * BLOCK_M + BLOCK_M // 2 for r, _ in cells]), 0, ny - 1)
        cc = np.clip(np.array([c * BLOCK_M + BLOCK_M // 2 for _, c in cells]), 0, nx - 1)
        d = dist[rr, cc]
        return {"n": int(d.size), "n_within": int(np.sum(d <= buf)), "share": float(np.mean(d <= buf)), "median_distance_m": float(np.median(d))}

    out["wall_heads_near_road"] = {f"{b:g}": share_near(wall, b) for b in ROAD_BUFFER_SENS_M}
    out["extra_heads_near_road"] = {f"{b:g}": share_near(extra_all, b) for b in ROAD_BUFFER_SENS_M}
    trunk = np.isfinite(A) & (A >= TRUNK_A_M2) & poly
    dtr = dist[trunk]
    out["trunk"] = {"A_min_m2": TRUNK_A_M2, "n_cells_in_polygon": int(trunk.sum()), "near_road": {f"{b:g}": {"n_within": int(np.sum(dtr <= b)), "share": float(np.mean(dtr <= b)) if dtr.size else None} for b in ROAD_BUFFER_SENS_M},
                    "median_distance_m": float(np.median(dtr)) if dtr.size else None}
    s_heads = out["wall_heads_near_road"][f"{ROAD_HEAD_BUFFER_M:g}"]["share"]
    s_trunk = out["trunk"]["near_road"][f"{ROAD_TRUNK_BUFFER_M:g}"]["share"]
    out["reading"] = {"wall_heads_within_15m_share": s_heads, "trunk_within_5m_share": s_trunk,
                      "road_influenced": bool((s_heads is not None and s_heads >= ROAD_HALF) or (s_trunk is not None and s_trunk >= ROAD_HALF))}
    # the AD surface's 12 m products for the figure, and the lidar hillshade inputs
    P = np.load(a22_path("profiles", "AD_TTV2_products.npz"))
    T = np.load(a22_path("tiles", "TTV2_products.npz"))
    np.savez_compressed(out_path("roads", "road_check.npz"), road_cells=np.packbits(rd["cells"]), road_dist=dist.astype(np.float32), poly=np.packbits(poly), trunk=np.packbits(trunk),
                        wall_rc=np.array(wall, dtype=np.int32).reshape(-1, 2), extra_rc=np.array(extra_all, dtype=np.int32).reshape(-1, 2), AD_Z12=P["AD_Z12"], obs_Z12=T["obs_Z12"],
                        Z1m=Z.astype(np.float32), pick_rc=T["pick_rc"], pick_ids=T["pick_ids"], transform=np.array([tr.a, tr.b, tr.c, tr.d, tr.e, tr.f]))
    out["wall_s"] = time.time() - t0
    dump(json_ready(out), out_path("roads", "road_check.json"))
    print(f"road check TTV2: {rd['n_features']} TIGER features ({rd['mtfcc']}); wall heads within 15 m {s_heads}; trunk within 5 m {s_trunk}; road-influenced {out['reading']['road_influenced']} [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
