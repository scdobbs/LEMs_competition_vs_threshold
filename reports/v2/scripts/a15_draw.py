#!/usr/bin/env python3
"""Addendum 15, sections 4 and 5: eligibility under the settled rules, the M&D windows, the random draw.

Eligibility (section 4), per site, from the stage A candidate windows and screen statistics (TTV: the a15 recomputation at
the polygon theta): threshold set B screens (make_a14_tables.screen_masks: trunk <= 5 km^2, inflow <= 2 km^2, chi-z R^2 >= 0.90,
dBIC <= 20, NLCD classes 11/22-24/31 <= 10 %, AP longwall = 0, GM/TTV upstream fully valid); k_s and |C_ht| within +-25 % of
the reference (GM, AP: legacy TAYLOR_REF; TTV: the M&D polygon); theta reported, not used; the decision-8 gap exclusion on
the 1 m grid (any cell with A >= 1e3 m^2 inside the tile footprint that is a filled enclosed gap); widening to +-50 % once
if fewer than 25 (TTV 30) windows are eligible.
Tile footprint of a coarse window: 1,000 x 1,000 m on the site's 1 m lattice, south-west corner = the window's south-west
corner snapped (floor) to the lattice (AP windows are 990 m; the footprint takes 1,000 m from the same corner).
Independence (section 5.1): "u drains directly into v" = a cell of u reaches v along its D8 path on the coarse grid before
entering any other candidate window of v's lattice offset (for u and v of the same offset this is exactly the stage A
direct edge, which is checked against pools/<site>_conflicts.json); the pre-placed tiles form a fifth group. Overlap = any
shared area between tile footprints.
Draw (section 5.1): seed 20261118 (written to the config before the draw); the eligible windows of all four offsets,
sorted by (offset, window), random-permuted once; greedy acceptance in that order; every skip recorded with its reason.
TTV (section 5.2): the fewest non-overlapping candidate windows covering >= 80 % of the M&D polygon are pre-placed.
usage: a15_draw.py --out DIR --ttv-windows TTV_windows_a15.csv --ttv-reference TTV_reference.json --ttv-polygon TTV_md_polygon_10m.tif --grids RUN_B
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import LEGACY_REF, SITES  # noqa: E402
from a14_pools import _next_window, route_full_grid  # noqa: E402
from a15_common import (  # noqa: E402
    A_NET_1M,
    MIN_ELIGIBLE,
    SEED,
    STAGE_A,
    TARGET_NEW,
    THRESHOLDS,
    TILE_EPSG,
    TOL_REL,
    TOL_REL_WIDE,
    dump,
    slurm_record,
)
from make_a14_tables import screen_masks  # noqa: E402

ORIG_TILES = {"GM": ["GM1", "GM2", "GM3", "GM4", "GM5"], "AP": ["AP1", "AP2", "AP3", "AP4", "AP5"], "TTV": []}


class Transformer:
    """Minimal pyproj-like transformer on GDAL/OSR (x, y order), used because the lemcvt environment has no pyproj."""

    def __init__(self, src_epsg, dst_epsg):
        from osgeo import osr
        s, d = osr.SpatialReference(), osr.SpatialReference()
        s.ImportFromEPSG(src_epsg)
        d.ImportFromEPSG(dst_epsg)
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.ct = osr.CoordinateTransformation(s, d)

    @classmethod
    def from_crs(cls, src, dst, always_xy=True):
        return cls(int(str(src).split(":")[1]), int(str(dst).split(":")[1]))

    def transform(self, x, y):
        if np.ndim(x) == 0:
            X, Y, _ = self.ct.TransformPoint(float(x), float(y))
            return X, Y
        pts = self.ct.TransformPoints([(float(a), float(b)) for a, b in zip(x, y, strict=True)])
        return [p[0] for p in pts], [p[1] for p in pts]
TILE_SIZE = 1000


def tile_footprint(site, x0, y0, transformer=None):
    """South-west corner of the 1,000 m tile footprint on the 1 m lattice (floor to integer metres in the tile CRS)."""
    if transformer is not None:
        x0, y0 = transformer.transform(x0, y0)
    return int(np.floor(x0)), int(np.floor(y0))


def overlaps(f1, f2):
    return f1[0] < f2[0] + TILE_SIZE and f2[0] < f1[0] + TILE_SIZE and f1[1] < f2[1] + TILE_SIZE and f2[1] < f1[1] + TILE_SIZE


def gap_exclusion(site, fp, grids):
    """Decision 8: any channel cell (A >= 1e3 m^2) of the footprint that is a filled enclosed gap. Returns (excluded, n_cells, n_channel)."""
    d = os.path.join(grids, f"grid_{site}")
    with rasterio.open(os.path.join(d, f"{site}_1m_area_cells.tif")) as da, rasterio.open(os.path.join(d, f"{site}_1m_gapfill_mask.tif")) as dg:
        col = int(round((fp[0] - da.transform.c) / da.transform.a))
        row = int(round((da.transform.f - (fp[1] + TILE_SIZE)) / -da.transform.e))
        if col < 0 or row < 0 or col + TILE_SIZE > da.width or row + TILE_SIZE > da.height:
            return True, -1, -1, "footprint outside the 1 m grid"
        w = Window(col, row, TILE_SIZE, TILE_SIZE)
        a = da.read(1, window=w)
        g = dg.read(1, window=w)
    ch = a >= A_NET_1M
    n = int((ch & (g == 1)).sum())
    return n > 0, n, int(ch.sum()), None


def coarse_cells_of_footprint(fp, tr, shape):
    """Boolean mask of coarse cells whose centres lie inside the tile footprint (footprint in the coarse grid's CRS)."""
    ny, nx = shape
    cols = np.arange(nx)
    rows = np.arange(ny)
    xs = tr.c + (cols + 0.5) * tr.a
    ys = tr.f + (rows + 0.5) * tr.e
    cx = (xs >= fp[0]) & (xs < fp[0] + TILE_SIZE)
    ry = (ys >= fp[1]) & (ys < fp[1] + TILE_SIZE)
    return np.outer(ry, cx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--grids", required=True, help="stage B run dir with grid_<site>/ (re-routed 1 m grids)")
    p.add_argument("--ttv-windows", required=True)
    p.add_argument("--ttv-reference", required=True)
    p.add_argument("--ttv-polygon", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    p.add_argument("--sites", default="GM,AP,TTV")
    p.add_argument("--ttv-target-total", type=int, default=None,
                   help="addendum 16 decision 3: TTV target counted over all tiles (M&D windows included); default: addendum 15 behaviour (target = drawn tiles)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    T = json.load(open(THRESHOLDS))
    config = {"seed": SEED, "thresholds": THRESHOLDS, "thresholds_values": T, "tolerance": TOL_REL, "tolerance_wide": TOL_REL_WIDE, "min_eligible": MIN_ELIGIBLE,
              "target_new": TARGET_NEW, "ttv_target_total": a.ttv_target_total, "gap_rule_A_m2": A_NET_1M, "references": {}, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": "written before the draw"}
    ttv_ref = json.load(open(a.ttv_reference))
    for s in ("GM", "AP"):
        config["references"][s] = {"theta": LEGACY_REF[s]["theta"], "ks": LEGACY_REF[s]["ks"], "C_ht": LEGACY_REF[s]["C_ht"], "source": "legacy TAYLOR_REF"}
    config["references"]["TTV"] = {"theta": ttv_ref["theta"], "ks": ttv_ref["ks"], "C_ht": ttv_ref["C_ht"], "source": ttv_ref["source"]}
    dump(config, os.path.join(a.out, "draw_config.json"))
    print("config written:", {k: config[k] for k in ("seed", "tolerance", "tolerance_wide", "min_eligible", "target_new")}, flush=True)
    summary = {}
    for site in a.sites.split(","):
        t_site = time.time()
        S = SITES[site]
        pj = json.load(open(os.path.join(STAGE_A, "pools", f"{site}_pools.json")))
        conf = json.load(open(os.path.join(STAGE_A, "pools", f"{site}_conflicts.json")))
        df = pd.read_csv(a.ttv_windows if site == "TTV" else os.path.join(STAGE_A, "pools", f"{site}_windows.csv"))
        df = df[df["candidate"].astype(str).str.lower() == "true"].reset_index(drop=True)
        ref = config["references"][site]
        W = int(pj["W_cells"])
        coarse = os.path.join(STAGE_A, "ttv_inventory", "TTV_NED_10m_utm10n.tif") if site == "TTV" else S["coarse"]
        with rasterio.open(coarse) as ds:
            z = ds.read(1).astype(np.float64)
            tr, ccrs, nod = ds.transform, ds.crs, ds.nodata
        if nod is not None and np.isfinite(nod):
            z[z == nod] = np.nan
        if site == "TTV":
            z[z <= 0] = np.nan
        dx = float(tr.a)
        ny, nx = z.shape
        R = route_full_grid(z, dx)
        print(site, "coarse routing", round(R["seconds"], 1), "s", flush=True)
        c_epsg = ccrs.to_epsg()
        t_epsg = TILE_EPSG[site]
        to_tile = Transformer.from_crs(f"EPSG:{c_epsg}", f"EPSG:{t_epsg}", always_xy=True) if c_epsg != t_epsg else None
        to_coarse = Transformer.from_crs(f"EPSG:{t_epsg}", f"EPSG:{c_epsg}", always_xy=True) if c_epsg != t_epsg else None
        # ---------------- screens and tolerances
        masks = screen_masks(df, T, site, STAGE_A)
        df["screens_pass"] = np.logical_and.reduce([m.values for m in masks.values()])
        for name, m in masks.items():
            df["screen: " + name] = m.values
        ks = pd.to_numeric(df["ks_ref_median"], errors="coerce")
        ch = pd.to_numeric(df["C_ht"], errors="coerce")
        df["ks_rel_dev"] = ks / ref["ks"] - 1
        df["C_ht_rel_dev"] = ch / ref["C_ht"] - 1
        df["theta_dev"] = pd.to_numeric(df["theta_legacy"], errors="coerce") - ref["theta"]
        df["tol25"] = (df["ks_rel_dev"].abs() <= TOL_REL) & (df["C_ht_rel_dev"].abs() <= TOL_REL)
        df["tol50"] = (df["ks_rel_dev"].abs() <= TOL_REL_WIDE) & (df["C_ht_rel_dev"].abs() <= TOL_REL_WIDE)
        # tile footprints
        fps = []
        for _, r in df.iterrows():
            fps.append(tile_footprint(site, float(r["x0"]), float(r["y0"]), to_tile))
        df["fp_x0"] = [f[0] for f in fps]
        df["fp_y0"] = [f[1] for f in fps]
        df["fp_epsg"] = t_epsg
        # ---------------- gap exclusion for every window passing screens and the wide tolerance
        cand = df["screens_pass"] & df["tol50"]
        gap_ex, gap_n, gap_ch, gap_note = [], [], [], []
        for i, r in df.iterrows():
            if cand[i]:
                ex, n, nch, note = gap_exclusion(site, (int(r["fp_x0"]), int(r["fp_y0"])), a.grids)
            else:
                ex, n, nch, note = None, None, None, None
            gap_ex.append(ex)
            gap_n.append(n)
            gap_ch.append(nch)
            gap_note.append(note)
        df["gap_excluded"] = gap_ex
        df["gap_channel_cells_in_filled_gap"] = gap_n
        df["gap_channel_cells"] = gap_ch
        df["gap_note"] = gap_note
        n25 = int((df["screens_pass"] & df["tol25"] & (df["gap_excluded"] == False)).sum())  # noqa: E712
        n50 = int((df["screens_pass"] & df["tol50"] & (df["gap_excluded"] == False)).sum())  # noqa: E712
        tol_used = TOL_REL if n25 >= MIN_ELIGIBLE[site] else TOL_REL_WIDE
        df["eligible"] = df["screens_pass"] & (df["tol25"] if tol_used == TOL_REL else df["tol50"]) & (df["gap_excluded"] == False)  # noqa: E712
        elig = {"n_candidates": int(len(df)), "n_screens_pass": int(df["screens_pass"].sum()), "n_tol25": int(df["tol25"].sum()), "n_tol50": int(df["tol50"].sum()),
                "n_screens_and_tol25": int((df["screens_pass"] & df["tol25"]).sum()), "n_screens_and_tol50": int((df["screens_pass"] & df["tol50"]).sum()),
                "n_gap_excluded_of_screens_tol50": int((cand & (df["gap_excluded"] == True)).sum()),  # noqa: E712
                "n_gap_excluded_of_screens_tol25": int((df["screens_pass"] & df["tol25"] & (df["gap_excluded"] == True)).sum()),  # noqa: E712
                "n_eligible_tol25_after_gap": n25, "n_eligible_tol50_after_gap": n50, "min_eligible": MIN_ELIGIBLE[site], "tolerance_used": tol_used,
                "widened": tol_used == TOL_REL_WIDE, "n_eligible": int(df["eligible"].sum()),
                "per_offset": {int(o): int(df.loc[df["offset"] == o, "eligible"].sum()) for o in sorted(df["offset"].unique())},
                "per_offset_tol25": {int(o): int((df["screens_pass"] & df["tol25"] & (df["gap_excluded"] == False) & (df["offset"] == o)).sum()) for o in sorted(df["offset"].unique())},  # noqa: E712
                "screens_each": {name: int((~m).sum()) for name, m in masks.items()}}
        print(site, "eligibility", json.dumps(elig), flush=True)
        # ---------------- pre-placed windows
        pre = []
        if site in ("GM", "AP") and a.data:
            for t in ORIG_TILES[site]:
                n = t[-1]
                d = os.path.join(a.data, site, f"tile{n}")
                f = [x for x in os.listdir(d) if x.endswith("_filled") or x.endswith("_filled.tif")][0]
                with rasterio.open(os.path.join(d, f)) as ds:
                    b = ds.bounds
                    e = ds.crs.to_epsg()
                tf = Transformer.from_crs(f"EPSG:{e}", f"EPSG:{t_epsg}", always_xy=True) if e != t_epsg else None
                fp = tile_footprint(site, b.left, b.bottom, tf)
                pre.append({"id": t, "kind": "original", "fp": fp, "orig_bounds": [b.left, b.bottom, b.right, b.top], "orig_epsg": e})
        md_rec = None
        if site == "TTV":
            with rasterio.open(a.ttv_polygon) as ds:
                poly = ds.read(1).astype(bool)
            poly_cells = int(poly.sum())
            # candidate windows intersecting the polygon
            inter = []
            for i, r in df.iterrows():
                r0, c0 = int(r["row0"]), int(r["col0"])
                n = int(poly[r0:r0 + W, c0:c0 + W].sum())
                if n > 0:
                    inter.append((i, n))
            best = None
            for k in range(1, 6):
                for combo in itertools.combinations(inter, k):
                    idx = [c[0] for c in combo]
                    f_ = [(int(df.loc[i, "fp_x0"]), int(df.loc[i, "fp_y0"])) for i in idx]
                    if any(overlaps(f_[u], f_[v]) for u in range(k) for v in range(u + 1, k)):
                        continue
                    cov = np.zeros(poly.shape, bool)
                    for i in idx:
                        r0, c0 = int(df.loc[i, "row0"]), int(df.loc[i, "col0"])
                        cov[r0:r0 + W, c0:c0 + W] = True
                    c = int((cov & poly).sum()) / poly_cells
                    if best is None or c > best[1]:
                        best = (idx, c)
                if best is not None and best[1] >= 0.80:
                    break
            idx, cov = best
            md_rec = {"polygon_cells_10m": poly_cells, "polygon_km2": poly_cells * dx * dx / 1e6, "n_candidate_windows_intersecting": len(inter), "n_windows": len(idx),
                      "coverage": cov, "coverage_pass_80": cov >= 0.80,
                      "windows": [{"offset": int(df.loc[i, "offset"]), "window": int(df.loc[i, "window"]), "fp": [int(df.loc[i, "fp_x0"]), int(df.loc[i, "fp_y0"])],
                                   "screens_pass": bool(df.loc[i, "screens_pass"]), "tol25": bool(df.loc[i, "tol25"]), "tol50": bool(df.loc[i, "tol50"]),
                                   "gap_excluded": df.loc[i, "gap_excluded"], "ks_rel_dev": float(df.loc[i, "ks_rel_dev"]), "C_ht_rel_dev": float(df.loc[i, "C_ht_rel_dev"]),
                                   "theta_dev": float(df.loc[i, "theta_dev"]), "screens": {name: bool(df.loc[i, "screen: " + name]) for name in masks}} for i in idx]}
            for j, i in enumerate(idx):
                ex, n, nch, note = gap_exclusion(site, (int(df.loc[i, "fp_x0"]), int(df.loc[i, "fp_y0"])), a.grids)
                md_rec["windows"][j]["gap_excluded"] = ex
                md_rec["windows"][j]["gap_channel_cells_in_filled_gap"] = n
                pre.append({"id": f"TTV{j + 1}", "kind": "M&D", "fp": (int(df.loc[i, "fp_x0"]), int(df.loc[i, "fp_y0"])), "offset": int(df.loc[i, "offset"]), "window": int(df.loc[i, "window"]), "df_index": int(i)})
            print(site, "M&D windows", json.dumps({k: v for k, v in md_rec.items() if k != "windows"}), [w["fp"] for w in md_rec["windows"]], flush=True)
        # ---------------- independence: next-window maps per lattice offset and for the pre-placed group
        offsets = pj["offsets_cells"]
        groups = {}
        for oi in range(len(offsets)):
            wid = np.full(z.shape, -1, np.int64)
            sub = df[df["offset"] == oi]
            for _, r in sub.iterrows():
                r0, c0 = int(r["row0"]), int(r["col0"])
                wid[r0:r0 + W, c0:c0 + W] = int(r["window"])
            groups[oi] = wid
        if pre:
            wid = np.full(z.shape, -1, np.int64)
            for k, pw in enumerate(pre):
                fp = pw["fp"]
                if to_coarse is not None:
                    xs = [fp[0], fp[0] + TILE_SIZE, fp[0], fp[0] + TILE_SIZE]
                    ys = [fp[1], fp[1], fp[1] + TILE_SIZE, fp[1] + TILE_SIZE]
                    cx, cy = to_coarse.transform(xs, ys)
                    fpc = (min(cx), min(cy))
                else:
                    fpc = fp
                m = coarse_cells_of_footprint(fpc, tr, z.shape)
                wid[m] = k
                pw["n_coarse_cells"] = int(m.sum())
            groups["pre"] = wid
        nextw = {}
        for g, wid in groups.items():
            nw = np.full(ny * nx, -1, np.int64)
            nextw[g] = np.asarray(_next_window(R["order_asc"], R["target"], wid.ravel(), nw))
        # direct edges. (a) Within one lattice offset: the stage A rule (the next candidate window of that offset downstream),
        # recomputed here and checked against pools/<site>_conflicts.json. (b) Across groups (other offsets, the pre-placed
        # tiles): u drains directly into v when v contains the first cell of u's D8 path that lies in any candidate window or
        # pre-placed tile not overlapping u's footprint, i.e. no other window lies between them in the drainage sequence
        # (stage A's "next window" idea, with every window of every group as a possible interceptor). Overlapping pairs are
        # excluded by the overlap rule anyway.
        def edges_from(cells_mask, g):
            vals = np.unique(nextw[g][cells_mask.ravel()])
            return set(int(v) for v in vals if v >= 0)
        win_cells = {}
        win_fp = {}
        for _, r in df.iterrows():
            key = (int(r["offset"]), int(r["window"]))
            m = np.zeros(z.shape, bool)
            r0, c0 = int(r["row0"]), int(r["col0"])
            m[r0:r0 + W, c0:c0 + W] = True
            win_cells[key] = m
            win_fp[key] = (int(r["fp_x0"]), int(r["fp_y0"]))
        pre_cells = {}
        for k in range(len(pre)):
            pre_cells[k] = groups["pre"] == k
        # check against stage A within-offset direct edges
        chk = {}
        same_offset_edges = set()
        for oi in range(len(offsets)):
            sa = {(e["from"], e["to"]) for e in conf[f"offset_{oi}"]["direct_edges"]}
            here = set()
            for (o, w), m in win_cells.items():
                if o == oi:
                    for v in edges_from(m, oi):
                        if v != w:
                            here.add((w, v))
                            same_offset_edges.add(((o, w), (o, v)))
            chk[oi] = {"stage_a": len(sa), "here": len(here), "equal": sa == here, "missing_here": sorted(sa - here)[:10], "extra_here": sorted(here - sa)[:10]}
        print(site, "within-offset edge check:", {k: (v["stage_a"], v["here"], v["equal"]) for k, v in chk.items()}, flush=True)
        anywin = np.zeros(z.shape, bool)
        for _g, wid in groups.items():
            anywin |= wid >= 0
        all_keys = [("win", o, w) for (o, w) in win_cells] + [("pre", k) for k in range(len(pre))]

        def cells(key):
            return win_cells[(key[1], key[2])] if key[0] == "win" else pre_cells[key[1]]

        def fp_of(key):
            return win_fp[(key[1], key[2])] if key[0] == "win" else pre[key[1]]["fp"]

        first_cache = {}

        def first_entered(key):
            """Windows (keys) containing the first cell of any path from `key` that lies in a window not overlapping `key`."""
            if key in first_cache:
                return first_cache[key]
            fp = fp_of(key)
            ban = np.zeros(z.shape, bool)
            for q in all_keys:
                if overlaps(fp, fp_of(q)):
                    ban |= cells(q)
            tgt = anywin & ~ban
            wid_u = np.where(tgt.ravel(), np.arange(ny * nx, dtype=np.int64), -1)
            nw = np.asarray(_next_window(R["order_asc"], R["target"], wid_u, np.full(ny * nx, -1, np.int64)))
            firsts = np.unique(nw[cells(key).ravel()])
            out = set()
            for c in firsts:
                if c < 0:
                    continue
                for g, wid in groups.items():
                    v = int(wid.ravel()[c])
                    if v >= 0:
                        out.add(("pre", v) if g == "pre" else ("win", g, v))
            out.discard(key)
            first_cache[key] = out
            return out

        def direct_edge(key_u, key_v):
            """True if u drains directly into v or v into u (stage A same-offset edge, or the first-entered rule across groups)."""
            if key_u[0] == "win" and key_v[0] == "win" and key_u[1] == key_v[1]:
                a_, b_ = (key_u[1], key_u[2]), (key_v[1], key_v[2])
                if (a_, b_) in same_offset_edges or (b_, a_) in same_offset_edges:
                    return True
            return key_v in first_entered(key_u) or key_u in first_entered(key_v)
        # ---------------- the draw
        E = df[df["eligible"]].sort_values(["offset", "window"]).reset_index()
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(E))
        accepted = []
        pre_fps = [pw["fp"] for pw in pre]
        pre_keys = [("pre", k) for k in range(len(pre))]
        ordered = []
        target = TARGET_NEW[site]
        if site == "TTV" and a.ttv_target_total is not None:
            target = max(0, a.ttv_target_total - len(pre))   # decision 3 of addendum 16: 30 in all, M&D windows included
        for rank, j in enumerate(perm):
            r = E.loc[j]
            key = ("win", int(r["offset"]), int(r["window"]))
            fp = (int(r["fp_x0"]), int(r["fp_y0"]))
            rec = {"rank": rank + 1, "offset": key[1], "window": key[2], "fp": fp, "ks_rel_dev": float(r["ks_rel_dev"]), "C_ht_rel_dev": float(r["C_ht_rel_dev"]), "theta": float(r["theta_legacy"])}
            if len(accepted) >= target:
                rec["result"] = "not reached (target met)"
                ordered.append(rec)
                continue
            reason = None
            for q, qfp in zip(pre_keys, pre_fps, strict=True):
                if overlaps(fp, qfp):
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
            if reason is None and bool(r["gap_excluded"]):
                reason = "decision-8 gap exclusion"
            if reason is None:
                rec["result"] = "accepted"
                accepted.append(rec)
            else:
                rec["result"] = "skipped: " + reason
            ordered.append(rec)
        # final tile list
        tiles = []
        for pw in pre:
            tiles.append({"tile": pw["id"] if site == "TTV" else pw["id"], "kind": pw["kind"], "fp_x0": pw["fp"][0], "fp_y0": pw["fp"][1], "epsg": t_epsg,
                          "offset": pw.get("offset"), "window": pw.get("window")})
        start = 6 if site in ("GM", "AP") else len(pre) + 1
        for k, acc in enumerate(accepted):
            tiles.append({"tile": f"{site}{start + k}", "kind": "drawn", "fp_x0": acc["fp"][0], "fp_y0": acc["fp"][1], "epsg": t_epsg, "offset": acc["offset"], "window": acc["window"], "rank": acc["rank"]})
        pd.DataFrame(tiles).to_csv(os.path.join(a.out, f"{site}_tiles.csv"), index=False)
        df.to_csv(os.path.join(a.out, f"{site}_eligibility.csv"), index=False)
        pre_links = {}
        for k in range(len(pre)):
            fe = first_entered(("pre", k))
            pre_links[pre[k]["id"]] = {"n_first_entered": len(fe), "eligible_directly_linked": int(sum(1 for _, r in E.iterrows() if direct_edge(("win", int(r["offset"]), int(r["window"])), ("pre", k))))}
        draw = {"site": site, "seed": SEED, "pre_placed_links": pre_links, "n_eligible": int(len(E)), "permutation": perm.tolist(), "eligible_order_keys": [[int(o), int(w)] for o, w in zip(E["offset"], E["window"], strict=True)],
                "pre_placed": [{k: v for k, v in pw.items()} for pw in pre], "ordered": ordered, "n_accepted": len(accepted), "target": target,
                "n_skipped": sum(1 for o in ordered if o["result"].startswith("skipped")),
                "skip_reasons": {}, "within_offset_edge_check": chk, "eligibility": elig, "md_windows": md_rec, "reference": ref, "seconds": time.time() - t_site}
        for o in ordered:
            if o["result"].startswith("skipped"):
                k = o["result"].split(":")[1].strip().split(" (")[0]
                k = " ".join(k.split()[:3])
                draw["skip_reasons"][k] = draw["skip_reasons"].get(k, 0) + 1
        dump(draw, os.path.join(a.out, f"{site}_draw.json"))
        summary[site] = {"n_eligible": int(len(E)), "tolerance_used": tol_used, "n_accepted": len(accepted), "target": target, "n_tiles_total": len(tiles), "skip_reasons": draw["skip_reasons"]}
        print(site, "draw:", summary[site], flush=True)
        # ---------------- map
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            zz = z.copy()
            gy, gx = np.gradient(np.nan_to_num(zz, nan=np.nanmean(zz)), dx, dx)
            sl = np.arctan(np.hypot(gx, gy))
            asp = np.arctan2(-gx, gy)
            hs = np.sin(np.radians(45)) * np.cos(sl) + np.cos(np.radians(45)) * np.sin(sl) * np.cos(np.radians(315) - asp)
            hs[~np.isfinite(zz)] = np.nan
            fig, ax = plt.subplots(figsize=(16, 16 * ny / nx + 1), dpi=110)
            ext = [tr.c, tr.c + nx * tr.a, tr.f + ny * tr.e, tr.f]
            ax.imshow(hs, cmap="gray", extent=ext, vmin=-0.3, vmax=1.3)
            def rect(fp, **kw):
                if to_coarse is not None:
                    cx, cy = to_coarse.transform([fp[0], fp[0] + TILE_SIZE], [fp[1], fp[1] + TILE_SIZE])
                    ax.add_patch(Rectangle((cx[0], cy[0]), cx[1] - cx[0], cy[1] - cy[0], fill=False, **kw))
                else:
                    ax.add_patch(Rectangle((fp[0], fp[1]), TILE_SIZE, TILE_SIZE, fill=False, **kw))
            for _, r in df.iterrows():
                rect((int(r["fp_x0"]), int(r["fp_y0"])), ec="0.6", lw=0.3)
            for _, r in df[df["eligible"]].iterrows():
                rect((int(r["fp_x0"]), int(r["fp_y0"])), ec="deepskyblue", lw=0.8)
            for t in tiles:
                col = {"original": "lime", "M&D": "orange", "drawn": "red"}[t["kind"]]
                rect((t["fp_x0"], t["fp_y0"]), ec=col, lw=1.6)
                if to_coarse is not None:
                    cx, cy = to_coarse.transform(t["fp_x0"] + 500, t["fp_y0"] + 500)
                else:
                    cx, cy = t["fp_x0"] + 500, t["fp_y0"] + 500
                ax.text(cx, cy, t["tile"], color=col, fontsize=7, ha="center", va="center", fontweight="bold")
            if site == "TTV":
                ax.contour(np.arange(nx) * dx + tr.c + dx / 2, tr.f + (np.arange(ny) + 0.5) * tr.e, poly.astype(float), levels=[0.5], colors="orange", linewidths=1.2)
            ax.set_title(f"{site}: candidates (grey), eligible (blue, tol +-{int(tol_used * 100)} %), original (green), M&D (orange), drawn (red); seed {SEED}")
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            fig.savefig(os.path.join(a.out, f"{site}_draw_map.png"), bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print("map failed", repr(e), flush=True)
    dump(summary, os.path.join(a.out, "draw_summary.json"))
    print("summary", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
