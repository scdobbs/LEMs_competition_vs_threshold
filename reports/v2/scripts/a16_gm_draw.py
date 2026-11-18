#!/usr/bin/env python3
"""Addendum 16, sections 3.3.4-3.3.5: eligibility of the GM extension windows on the 1 m grid and the draw (seed 20261119).

Candidates: the gate's windows of the extension survey (a16_gm_gate.py CSV) that are eligible at the tolerance the widening rule
selected on the combined count (gate json), lie in the landscape, and do not overlap GM1-GM7. On the 1 m grid (grid_GMF/, the survey
re-routed with the addendum 15 gap rule) the decision-8 gap exclusion is applied (a channel cell, A >= 1e3 m2, in a filled enclosed
gap excludes the window). Independence as in a15_draw.py: within a lattice offset the stage A "next window downstream" edge over all
candidate windows of that offset; across offsets and with the pre-placed GM1-GM7 the first-entered rule; computed on the NED crop
masked to the survey (the gate's routing). Draw: seed 20261119 written to the config before the draw; the eligible windows sorted by
(offset, window), permuted once; greedy acceptance (no overlap with a pre-placed or accepted tile, no direct drains-through edge, not
gap-excluded); target = 30 GM tiles in all minus the 7 existing. Tiles are named GM8, GM9, ... in the order of acceptance.
Outputs (run dir draw_gmf/): draw_config.json, GMF_tiles.csv, GMF_eligibility.csv, GMF_draw.json, GMF_draw_map.png.
usage: a16_gm_draw.py --out DIR --gate DIR --footprints DIR --grids RUN_DIR [--survey NAME]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_pools import _next_window, crop, route_full_grid  # noqa: E402
from a15_draw import TILE_SIZE, gap_exclusion, overlaps  # noqa: E402
from a16_common import (  # noqa: E402
    A_NET_1M,
    GM_GATE_CROP_UTM,
    GM_GATE_SURVEYS,
    SEED_GM_EXT,
    THRESHOLDS,
    TOL_REL,
    TOL_REL_WIDE,
    dump,
    slurm_record,
)
from a16_gm_footprints import GM_TILES  # noqa: E402

TARGET_TOTAL_GM = 30


def read_crop(path, box):
    with rasterio.open(path) as ds:
        b = ds.bounds
        assert b.left <= box[0] and b.bottom <= box[1] and box[2] <= b.right and box[3] <= b.top, (path, box, tuple(b))
        w = from_bounds(box[0], box[1], box[2], box[3], ds.transform)
        return ds.read(1, window=w), ds.window_transform(w), ds.crs, ds.nodata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--footprints", required=True)
    p.add_argument("--grids", required=True, help="run dir holding grid_GMF/ (the 1 m survey grid re-routed with the gap rule)")
    p.add_argument("--survey", default=GM_GATE_SURVEYS[0])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    gate = json.load(open(os.path.join(a.gate, "gm_gate.json")))
    g = gate["gate"][a.survey]
    tol = g["tolerance_used"]
    tol_val = TOL_REL if tol == "25" else TOL_REL_WIDE
    target = TARGET_TOTAL_GM - len(GM_TILES)
    config = {"seed": SEED_GM_EXT, "survey": a.survey, "thresholds": THRESHOLDS, "tolerance_used": tol_val, "tolerance_from": "the widening rule on the combined count (gate)",
              "combined_eligible_at_25": g["combined_eligible_at_25"], "target_total_GM": TARGET_TOTAL_GM, "pre_placed": list(GM_TILES), "target_new": target,
              "gap_rule_A_m2": A_NET_1M, "reference": gate["reference"], "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": "written before the draw"}
    dump(config, os.path.join(a.out, "draw_config.json"))
    print("config written:", {k: config[k] for k in ("seed", "tolerance_used", "target_new")}, flush=True)
    df = pd.read_csv(os.path.join(a.gate, f"{a.survey}_windows.csv"))
    box = GM_GATE_CROP_UTM
    z, tr, crs, nod = read_crop(os.path.join(a.footprints, "GM_NED_10m_utm10n.tif"), box)
    z = z.astype(np.float64)
    z[z == nod] = np.nan
    land, _, _, _ = read_crop(os.path.join(a.gate, "GM_landscape_10m.tif"), box)
    m, _, _, _ = read_crop(os.path.join(a.footprints, f"GM_survey_{a.survey}_10m.tif"), box)
    W = 100
    dx = float(tr.a)
    ny, nx = z.shape
    df["in_landscape"] = [crop(land.astype(bool), int(r.row0), int(r.col0), W).all() for r in df.itertuples()]
    df["eligible_gate"] = df[f"eligible{tol}"] & ~df["overlaps_GM1_7"] & df["in_landscape"]
    # 1 m gap exclusion for the gate-eligible windows
    ex, n_in, n_ch, notes = [], [], [], []
    for r in df.itertuples():
        if r.eligible_gate:
            e, n, nch, note = gap_exclusion("GMF", (int(r.fp_x0), int(r.fp_y0)), a.grids)
        else:
            e, n, nch, note = None, None, None, None
        ex.append(e)
        n_in.append(n)
        n_ch.append(nch)
        notes.append(note)
    df["gap_excluded"] = ex
    df["gap_channel_cells_in_filled_gap"] = n_in
    df["gap_channel_cells"] = n_ch
    df["gap_note"] = notes
    df["eligible"] = df["eligible_gate"] & (df["gap_excluded"] == False)  # noqa: E712
    elig = {"n_candidates": int(len(df)), "n_eligible_gate": int(df["eligible_gate"].sum()), "n_gap_excluded": int((df["eligible_gate"] & (df["gap_excluded"] == True)).sum()),  # noqa: E712
            "n_outside_1m_grid": int((df["eligible_gate"] & (df["gap_note"] == "footprint outside the 1 m grid")).sum()), "n_eligible": int(df["eligible"].sum()),
            "per_offset": {int(o): int(df.loc[df["offset"] == o, "eligible"].sum()) for o in sorted(df["offset"].unique())}}
    print("eligibility", json.dumps(elig), flush=True)
    # independence on the survey-masked NED routing
    zs = np.where(m.astype(bool), z, np.nan)
    R = route_full_grid(zs, dx)
    groups = {}
    for oi in sorted(df["offset"].unique()):
        wid = np.full(z.shape, -1, np.int64)
        for r in df[df["offset"] == oi].itertuples():
            wid[int(r.row0):int(r.row0) + W, int(r.col0):int(r.col0) + W] = int(r.window)
        groups[int(oi)] = wid
    pre = [{"id": t, "kind": "original", "fp": fp} for t, fp in GM_TILES.items()]
    wid = np.full(z.shape, -1, np.int64)
    xs = tr.c + (np.arange(nx) + 0.5) * tr.a
    ys = tr.f + (np.arange(ny) + 0.5) * tr.e
    for k, pw in enumerate(pre):
        fp = pw["fp"]
        cx = (xs >= fp[0]) & (xs < fp[0] + TILE_SIZE)
        ry = (ys >= fp[1]) & (ys < fp[1] + TILE_SIZE)
        wid[np.outer(ry, cx)] = k
    groups["pre"] = wid
    nextw = {g_: np.asarray(_next_window(R["order_asc"], R["target"], w_.ravel(), np.full(ny * nx, -1, np.int64))) for g_, w_ in groups.items()}
    win_cells, win_fp = {}, {}
    for r in df.itertuples():
        key = (int(r.offset), int(r.window))
        mm = np.zeros(z.shape, bool)
        mm[int(r.row0):int(r.row0) + W, int(r.col0):int(r.col0) + W] = True
        win_cells[key] = mm
        win_fp[key] = (int(r.fp_x0), int(r.fp_y0))
    pre_cells = {k: groups["pre"] == k for k in range(len(pre))}
    same_offset_edges = set()
    for (o, w), mm in win_cells.items():
        for v in np.unique(nextw[o][mm.ravel()]):
            if v >= 0 and v != w:
                same_offset_edges.add(((o, w), (o, int(v))))
    anywin = np.zeros(z.shape, bool)
    for w_ in groups.values():
        anywin |= w_ >= 0
    all_keys = [("win", o, w) for (o, w) in win_cells] + [("pre", k) for k in range(len(pre))]

    def cells(key):
        return win_cells[(key[1], key[2])] if key[0] == "win" else pre_cells[key[1]]

    def fp_of(key):
        return win_fp[(key[1], key[2])] if key[0] == "win" else pre[key[1]]["fp"]

    cache = {}

    def first_entered(key):
        if key in cache:
            return cache[key]
        fp = fp_of(key)
        ban = np.zeros(z.shape, bool)
        for q in all_keys:
            if overlaps(fp, fp_of(q)):
                ban |= cells(q)
        tgt = anywin & ~ban
        wid_u = np.where(tgt.ravel(), np.arange(ny * nx, dtype=np.int64), -1)
        nw = np.asarray(_next_window(R["order_asc"], R["target"], wid_u, np.full(ny * nx, -1, np.int64)))
        out = set()
        for c in np.unique(nw[cells(key).ravel()]):
            if c < 0:
                continue
            for g_, w_ in groups.items():
                v = int(w_.ravel()[c])
                if v >= 0:
                    out.add(("pre", v) if g_ == "pre" else ("win", g_, v))
        out.discard(key)
        cache[key] = out
        return out

    def direct_edge(u, v):
        if u[0] == "win" and v[0] == "win" and u[1] == v[1]:
            a_, b_ = (u[1], u[2]), (v[1], v[2])
            if (a_, b_) in same_offset_edges or (b_, a_) in same_offset_edges:
                return True
        return v in first_entered(u) or u in first_entered(v)
    # the draw
    E = df[df["eligible"]].sort_values(["offset", "window"]).reset_index()
    rng = np.random.default_rng(SEED_GM_EXT)
    perm = rng.permutation(len(E))
    accepted, ordered = [], []
    pre_keys = [("pre", k) for k in range(len(pre))]
    for rank, j in enumerate(perm):
        r = E.loc[j]
        key = ("win", int(r["offset"]), int(r["window"]))
        fp = (int(r["fp_x0"]), int(r["fp_y0"]))
        rec = {"rank": rank + 1, "offset": key[1], "window": key[2], "fp": fp, "ks_rel_dev": float(r["ks_rel_dev"]), "C_ht_rel_dev": float(r["C_ht_rel_dev"]), "theta": float(r["theta_legacy"]),
               "in_ncalm_full": bool(r["in_ncalm_full"])}
        if len(accepted) >= target:
            rec["result"] = "not reached (target met)"
            ordered.append(rec)
            continue
        reason = None
        for q in pre_keys:
            if overlaps(fp, pre[q[1]]["fp"]):
                reason = f"overlaps pre-placed {pre[q[1]]['id']}"
                break
        if reason is None:
            for acc in accepted:
                if overlaps(fp, acc["fp"]):
                    reason = f"overlaps accepted rank {acc['rank']} (offset {acc['offset']} window {acc['window']})"
                    break
        if reason is None:
            for q in pre_keys:
                if direct_edge(key, q):
                    reason = f"drains-through edge with pre-placed {pre[q[1]]['id']}"
                    break
        if reason is None:
            for acc in accepted:
                if direct_edge(key, ("win", acc["offset"], acc["window"])):
                    reason = f"drains-through edge with accepted rank {acc['rank']} (offset {acc['offset']} window {acc['window']})"
                    break
        if reason is None:
            rec["result"] = "accepted"
            accepted.append(rec)
        else:
            rec["result"] = "skipped: " + reason
        ordered.append(rec)
    tiles = [{"tile": f"GM{len(GM_TILES) + 1 + k}", "kind": "drawn", "fp_x0": acc["fp"][0], "fp_y0": acc["fp"][1], "epsg": 6339, "offset": acc["offset"], "window": acc["window"], "rank": acc["rank"]}
             for k, acc in enumerate(accepted)]
    pd.DataFrame(tiles).to_csv(os.path.join(a.out, "GMF_tiles.csv"), index=False)
    df.to_csv(os.path.join(a.out, "GMF_eligibility.csv"), index=False)
    skip = {}
    for o in ordered:
        if o["result"].startswith("skipped"):
            k = " ".join(o["result"].split(":")[1].strip().split(" (")[0].split()[:3])
            skip[k] = skip.get(k, 0) + 1
    draw = {"survey": a.survey, "seed": SEED_GM_EXT, "tolerance_used": tol_val, "eligibility": elig, "n_eligible": int(len(E)), "target": target, "n_accepted": len(accepted),
            "n_skipped": sum(1 for o in ordered if o["result"].startswith("skipped")), "skip_reasons": skip, "ordered": ordered, "permutation": perm.tolist(),
            "eligible_order_keys": [[int(o), int(w)] for o, w in zip(E["offset"], E["window"], strict=True)], "pre_placed": pre, "tiles": tiles,
            "n_tiles_total_GM": len(GM_TILES) + len(accepted), "footprint_crs_note": "window footprints are on the EPSG:32610 NED lattice; the 1 m grid is EPSG:6339; the shift is the identity offline (PROJ has no grid), as in stages A and B",
            "seconds": time.time() - t0}
    dump(draw, os.path.join(a.out, "GMF_draw.json"))
    print("draw:", {k: draw[k] for k in ("n_eligible", "target", "n_accepted", "n_skipped", "skip_reasons", "n_tiles_total_GM")}, flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)), dx, dx)
        sl = np.arctan(np.hypot(gx, gy))
        asp = np.arctan2(-gx, gy)
        hs = np.sin(np.radians(45)) * np.cos(sl) + np.cos(np.radians(45)) * np.sin(sl) * np.cos(np.radians(315) - asp)
        fig, ax = plt.subplots(figsize=(14, 14 * ny / nx + 1), dpi=110)
        ext = [tr.c, tr.c + nx * tr.a, tr.f + ny * tr.e, tr.f]
        ax.imshow(hs, cmap="gray", extent=ext, vmin=-0.3, vmax=1.3)
        X, Y = np.meshgrid(xs, ys)
        ax.contour(X, Y, land.astype(float), levels=[0.5], colors="yellow", linewidths=1.2)
        for r in df[df["eligible"]].itertuples():
            ax.add_patch(Rectangle((r.fp_x0, r.fp_y0), TILE_SIZE, TILE_SIZE, fill=False, ec="deepskyblue", lw=0.7))
        for t, fp in GM_TILES.items():
            ax.add_patch(Rectangle(fp, TILE_SIZE, TILE_SIZE, fill=False, ec="lime", lw=1.4))
            ax.text(fp[0] + 500, fp[1] + 500, t, color="lime", fontsize=7, ha="center", va="center")
        for t in tiles:
            ax.add_patch(Rectangle((t["fp_x0"], t["fp_y0"]), TILE_SIZE, TILE_SIZE, fill=False, ec="red", lw=1.6))
            ax.text(t["fp_x0"] + 500, t["fp_y0"] + 500, t["tile"], color="red", fontsize=7, ha="center", va="center")
        ax.set_title(f"GM extension draw ({a.survey}): landscape yellow, eligible blue, original lime, drawn red; seed {SEED_GM_EXT}")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        fig.savefig(os.path.join(a.out, "GMF_draw_map.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("map failed", repr(e), flush=True)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
