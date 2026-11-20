#!/usr/bin/env python3
"""Addendum 19, section 3 (decisions 2-4): Steve's 64 hand-picked M&D (1989) channel heads on the stage B full-site TTV grids.

Per pick: A_cell, A_3 (primary: the largest D8 area within 3 m), A_10; S_up over 10 m (5, 20 m sensitivity) along the largest-area donor
chain above the A_3 cell; S_down over 10 m of D8 path below it (the `slope_threshold.valley_slope_reach` definition, stepped the same way);
the TTV tile the pick falls in; the union of D8 flow paths downstream of the A_3 cells (the picked-head network of decision 12). The area-slope
fits (i)-(iv) with the 10,000-draw bootstrap, the A S^2 distributions, A_F and C_F for section 5. D-infinity A_3 on a window around the picks
(decision 3, see --dinf): the full 19,010 x 19,010 grid through the package's pure-Python D-infinity loops would exceed the 10 core-hour cap,
so the routing runs on the picks' bounding box plus a margin, with a containment check (the D8 area recomputed inside the window equals the
full-grid D8 area at the A_3 cell). Every area is cells x the cell area read from the transform.
usage: a19_heads.py [--dinf-pilot TILE] [--dinf] [--no-dinf]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_common import (  # noqa: E402
    ALPHAS_FIXED,
    DINF_MAX_CORE_H,
    HEAD_RADII_M,
    HEADS_DST,
    HEADS_SRC,
    MD_EQ10,
    MD_GPKG,
    MD_TABLE1_MEDIAN_M2,
    R4_A_TOL,
    R4_B_TOL,
    R5_RATIO_RANGE,
    S_DOWN_REACH_M,
    S_UP_REACHES_M,
    TTV_FULL,
    TTV_PRIMARY,
    dump,
    md_table1,
    ols_loglog_bootstrap,
    out_path,
    pct,
    sha256_file,
    slurm_record,
    tile_bounds,
)

# lemcvt D8 codes (hydro/d8.py): 1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE  (receiver offsets)
CODES = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
TTV_TILES_ALL = [f"TTV{n}" for n in range(1, 16)]


def read_full():
    import rasterio
    with rasterio.open(TTV_FULL["filled"]) as ds:
        Z = ds.read(1).astype(np.float32)
        nod = ds.nodata
        tr = ds.transform
        crs = ds.crs
        Z[Z == nod] = np.nan
    with rasterio.open(TTV_FULL["fd"]) as ds:
        FD = ds.read(1)
    with rasterio.open(TTV_FULL["area_cells"]) as ds:
        Ac = ds.read(1)
    dx, dy = abs(tr.a), abs(tr.e)
    return Z, FD, Ac, tr, crs, dx, dy


def picks() -> list[dict]:
    with open(HEADS_SRC) as f:
        rows = list(csv.DictReader(f))
    assert [r["pick_id"] for r in rows] == [str(i) for i in range(1, 65)], "pick ids must be 1..64 in order"
    return [{"pick_id": int(r["pick_id"]), "x": float(r["x_epsg6339"]), "y": float(r["y_epsg6339"]), "origin": r["origin"], "uncertain": int(r["uncertain"]), "note": r["note"]} for r in rows]


def receiver_of(FD, r, c):
    code = int(FD[r, c])
    if code not in CODES:
        return None
    dr, dc = CODES[code]
    return r + dr, c + dc


def donors_of(FD, r, c, ny, nx):
    out = []
    for code, (dr, dc) in CODES.items():
        rr, cc = r - dr, c - dc          # a neighbour at (r - dr, c - dc) with this code flows to (r, c)
        if 0 <= rr < ny and 0 <= cc < nx and int(FD[rr, cc]) == code:
            out.append((rr, cc))
    return out


def step_len(dr, dc, dx, dy):
    return math.hypot(dx, dy) if (dr != 0 and dc != 0) else (dy if dr != 0 else dx)


def slope_down(Z, FD, dx, dy, r, c, reach):
    """Exactly `valley_slope_reach`: step along receivers until the path length is >= reach (or the receiver is missing)."""
    ny, nx = Z.shape
    L = 0.0
    cur = (r, c)
    left = False
    n_steps = int(math.ceil(reach / min(dx, dy))) + 2
    for _ in range(n_steps):
        if L >= reach:
            break
        nxt = receiver_of(FD, *cur)
        if nxt is None or not (0 <= nxt[0] < ny and 0 <= nxt[1] < nx) or not np.isfinite(Z[nxt]):
            left = True
            break
        L += step_len(nxt[0] - cur[0], nxt[1] - cur[1], dx, dy)
        cur = nxt
    S = float((Z[r, c] - Z[cur]) / L) if L > 0 else float("nan")
    return {"S": S, "L": L, "left": left, "end": [int(cur[0]), int(cur[1])]}


def slope_up(Z, FD, A, dx, dy, r, c, reach):
    """The largest-area D8 donor chain above (r, c) for `reach` m of path; the gain over the length available."""
    ny, nx = Z.shape
    L = 0.0
    cur = (r, c)
    short = False
    while L < reach:
        d = donors_of(FD, cur[0], cur[1], ny, nx)
        if not d:
            short = True
            break
        nxt = max(d, key=lambda p: (float(A[p]), -p[0] * nx - p[1]))
        L += step_len(nxt[0] - cur[0], nxt[1] - cur[1], dx, dy)
        cur = nxt
    S = float((Z[cur] - Z[r, c]) / L) if L > 0 else float("nan")
    return {"S": S, "L": L, "short": short, "end": [int(cur[0]), int(cur[1])]}


def within(A, r, c, radius_m, dx, dy):
    ny, nx = A.shape
    kr, kc = int(math.ceil(radius_m / dy)), int(math.ceil(radius_m / dx))
    r0, r1, c0, c1 = max(r - kr, 0), min(r + kr + 1, ny), max(c - kc, 0), min(c + kc + 1, nx)
    sub = A[r0:r1, c0:c1].astype(float)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    d = np.hypot((rr - r) * dy, (cc - c) * dx)
    sub = np.where(d <= radius_m + 1e-9, sub, -1.0)
    j = int(np.argmax(sub))
    return float(sub.ravel()[j]), int(rr.ravel()[j]), int(cc.ravel()[j])


def downstream_paths(FD, Z, starts):
    ny, nx = FD.shape
    visited = np.zeros((ny, nx), dtype=bool)
    for r, c in starts:
        cur = (r, c)
        n = 0
        while not visited[cur]:
            visited[cur] = True
            nxt = receiver_of(FD, *cur)
            n += 1
            if nxt is None or not (0 <= nxt[0] < ny and 0 <= nxt[1] < nx) or not np.isfinite(Z[nxt]) or n > ny * nx:
                break
            cur = nxt
    return visited


def tile_of(x, y, bounds):
    for t, b in bounds.items():
        if b["left"] <= x < b["right"] and b["bottom"] < y <= b["top"]:
            return t
    return "none"


def in_polygon(xs, ys):
    from osgeo import ogr
    ogr.UseExceptions()
    ds = ogr.Open(MD_GPKG)
    polys = [f.GetGeometryRef().Clone() for f in ds.GetLayerByName("study_area")]
    lines = [f.GetGeometryRef().Clone() for f in ds.GetLayerByName("channels")]
    inside, dist = [], []
    for x, y in zip(xs, ys, strict=True):
        p = ogr.Geometry(ogr.wkbPoint)
        p.AddPoint(float(x), float(y))
        inside.append(bool(any(g.Contains(p) for g in polys)))
        dist.append(float(min(g.Distance(p) for g in lines)))
    return inside, dist


def dinf_pilot(tile: str) -> dict:
    from lemcvt.hydro.dinf import compute_dinf_area, compute_dinf_flow_direction
    from lemcvt.measure.pipeline import _metre_tile
    td = _metre_tile(tile)
    t0 = time.perf_counter()
    ang, rf, fi = compute_dinf_flow_direction(td.Z, td.dx, FD_d8=td.FD)
    t1 = time.perf_counter()
    compute_dinf_area(td.Z, td.dx, ang, rf, fi)
    t2 = time.perf_counter()
    n = int(np.isfinite(td.Z).sum())
    per_cell = (t2 - t0) / n
    full = 19010 * 19010
    rec = {"tile": tile, "n_cells": n, "direction_s": t1 - t0, "area_s": t2 - t1, "s_per_cell": per_cell, "full_grid_cells": full,
           "full_grid_core_h_estimate": per_cell * full / 3600.0, "cap_core_h": DINF_MAX_CORE_H}
    rec["full_grid_within_cap"] = rec["full_grid_core_h_estimate"] <= DINF_MAX_CORE_H
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dinf-pilot", default=None)
    p.add_argument("--dinf", action="store_true", help="windowed D-infinity A_3 after the attributes")
    p.add_argument("--dinf-full", action="store_true", help="D-infinity on the full 19,010 x 19,010 grid (the pilot: 3.5 core-hours, under the 10 core-hour cap)")
    p.add_argument("--window-margin-m", type=float, default=1000.0)
    a = p.parse_args()
    t0 = time.time()
    out_dir = os.path.dirname(out_path("heads", "heads.json"))
    slurm_record(out_dir)
    if a.dinf_pilot:
        rec = dinf_pilot(a.dinf_pilot)
        dump(rec, out_path("heads", "dinf_pilot.json"))
        print("dinf pilot:", rec, flush=True)
        return
    # decision 2: the CSV of record
    os.makedirs(os.path.dirname(HEADS_DST), exist_ok=True)
    shutil.copyfile(HEADS_SRC, HEADS_DST)
    csv_rec = {"source": HEADS_SRC, "copy": HEADS_DST, "sha256": sha256_file(HEADS_DST), "sha256_source": sha256_file(HEADS_SRC), "n_rows": 64}
    pk = picks()
    inside, d_line = in_polygon([q["x"] for q in pk], [q["y"] for q in pk])
    Z, FD, Ac, tr, crs, dx, dy = read_full()
    cell_area = dx * dy
    ny, nx = Z.shape
    print(f"full grid {ny} x {nx}, dx {dx} dy {dy}, cell area {cell_area}, crs {crs} [{time.time() - t0:.0f} s]", flush=True)
    bounds = {t: tile_bounds(t) for t in TTV_TILES_ALL}
    rows = []
    a3_cells = []
    for q, ins, dl in zip(pk, inside, d_line, strict=True):
        c = int(math.floor((q["x"] - tr.c) / dx))
        r = int(math.floor((tr.f - q["y"]) / dy))
        rec = dict(q) | {"row": r, "col": c, "inside_md_polygon": ins, "dist_to_traced_channel_m": dl, "tile": tile_of(q["x"], q["y"], bounds),
                         "primary_tile": tile_of(q["x"], q["y"], bounds) in TTV_PRIMARY, "A_cell_m2": float(Ac[r, c]) * cell_area}
        for name, rad in HEAD_RADII_M.items():
            cells, rr, cc = within(Ac, r, c, rad, dx, dy)
            rec[f"{name}_m2"] = cells * cell_area
            rec[f"{name}_row"], rec[f"{name}_col"] = rr, cc
            rec[f"{name}_offset_m"] = float(math.hypot((rr - r) * dy, (cc - c) * dx))
        r3, c3 = rec["A_3_row"], rec["A_3_col"]
        a3_cells.append((r3, c3))
        for reach in S_UP_REACHES_M:
            su = slope_up(Z, FD, Ac, dx, dy, r3, c3, reach)
            tag = "" if reach == S_UP_REACHES_M[0] else f"_{reach:g}m"
            rec[f"S_up{tag}"], rec[f"S_up{tag}_L"], rec[f"S_up{tag}_short"] = su["S"], su["L"], su["short"]
        sd = slope_down(Z, FD, dx, dy, r3, c3, S_DOWN_REACH_M)
        rec["S_down"], rec["S_down_L"], rec["S_down_left"] = sd["S"], sd["L"], sd["left"]
        rec["z_m"] = float(Z[r3, c3])
        rows.append(rec)
    print(f"attributes done [{time.time() - t0:.0f} s]", flush=True)
    # the picked-head network (decision 12)
    net = downstream_paths(FD, Z, a3_cells)
    rr, cc = np.nonzero(net)
    np.savez_compressed(out_path("heads", "picked_network.npz"), rows=rr.astype(np.int32), cols=cc.astype(np.int32), transform=np.array(list(tr)[:6]),
                        a3_rows=np.array([x[0] for x in a3_cells]), a3_cols=np.array([x[1] for x in a3_cells]))
    print(f"picked-head network: {rr.size} cells [{time.time() - t0:.0f} s]", flush=True)
    del net
    # table and CSV
    with open(out_path("heads", "head_attributes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # decision 4: fits
    t1 = md_table1()
    A1 = np.array([x["A"] for x in t1])
    S1 = np.array([x["S"] for x in t1])
    A3 = np.array([x["A_3_m2"] for x in rows])
    Sup = np.array([x["S_up"] for x in rows])
    Sdn = np.array([x["S_down"] for x in rows])
    not49 = np.array([x["pick_id"] != 49 for x in rows])

    def fit(A, S, mask=None):
        ok = np.isfinite(A) & np.isfinite(S) & (A > 0) & (S > 0)
        if mask is not None:
            ok &= mask
        f = ols_loglog_bootstrap(A[ok], S[ok])
        f["n_dropped_nonpositive"] = int((~ok).sum() - (0 if mask is None else (~mask).sum()))
        return f

    fits = {"i_table1": fit(A1, S1), "ii_picks_A3_Sup": fit(A3, Sup), "iii_picks_A3_Sdown": fit(A3, Sdn), "iv_picks_A3_Sup_without_49": fit(A3, Sup, not49)}
    fi = fits["i_table1"]
    r4 = {"a": fi["a"], "b": fi["b"], "a_rel_diff": fi["a"] / MD_EQ10["a_m2"] - 1, "b_diff": fi["b"] - MD_EQ10["b"],
          "pass": bool(abs(fi["a"] / MD_EQ10["a_m2"] - 1) <= R4_A_TOL and abs(fi["b"] - MD_EQ10["b"]) <= R4_B_TOL), "tolerances": {"a_rel": R4_A_TOL, "b_abs": R4_B_TOL}}

    def r5(f, A, S, mask=None):
        ok = np.isfinite(A) & np.isfinite(S) & (S > 0)
        if mask is not None:
            ok &= mask
        ratio = A[ok] / (MD_EQ10["a_m2"] * S[ok] ** (-MD_EQ10["b"]))
        med = float(np.median(ratio))
        b_in = bool(f["b_ci95"][0] <= MD_EQ10["b"] <= f["b_ci95"][1])
        med_in = bool(R5_RATIO_RANGE[0] <= med <= R5_RATIO_RANGE[1])
        return {"b_1.65_inside_ci95": b_in, "median_ratio_to_eq10": med, "median_in_range": med_in, "consistent": bool(b_in and med_in), "ratio_pct": pct(ratio)}

    rules = {"R4": r4, "R5": {"ii": r5(fits["ii_picks_A3_Sup"], A3, Sup), "iii_report_only": r5(fits["iii_picks_A3_Sdown"], A3, Sdn),
                              "iv_report_only": r5(fits["iv_picks_A3_Sup_without_49"], A3, Sup, not49)}}
    rules["R5"]["verdict"] = rules["R5"]["ii"]["consistent"]
    as2 = {"picks_Sup": pct(A3 * Sup ** 2), "picks_Sdown": pct(A3 * Sdn ** 2), "table1": pct(A1 * S1 ** 2), "MD1992_range_m2": [500.0, 4000.0],
           "picks_Sdown_in_MD_range_frac": float(np.mean((A3 * Sdn ** 2 >= 500) & (A3 * Sdn ** 2 <= 4000))),
           "picks_Sup_in_MD_range_frac": float(np.mean((A3 * Sup ** 2 >= 500) & (A3 * Sup ** 2 <= 4000)))}
    dists = {"A_cell": pct([x["A_cell_m2"] for x in rows]), "A_3": pct(A3), "A_10": pct([x["A_10_m2"] for x in rows]), "table1_area": pct(A1),
             "A_3_without_49": pct(A3[not49]), "S_up": pct(Sup), "S_down": pct(Sdn), "table1_slope": pct(S1),
             "A_3_offset_m": pct([x["A_3_offset_m"] for x in rows]), "S_up_short_n": int(sum(x["S_up_short"] for x in rows)),
             "S_down_left_n": int(sum(x["S_down_left"] for x in rows)), "dist_to_traced_channel_m": pct([x["dist_to_traced_channel_m"] for x in rows]),
             "n_inside_md_polygon": int(sum(x["inside_md_polygon"] for x in rows)), "tiles": {t: int(sum(x["tile"] == t for x in rows)) for t in sorted({x["tile"] for x in rows})}}
    b_hat = fits["ii_picks_A3_Sup"]["b"]
    alpha_b = round(b_hat / 0.05) * 0.05
    alphas = sorted(set(ALPHAS_FIXED) | {float(f"{alpha_b:.2f}")})
    field = {"A_F_m2": float(np.median(A3)), "A_F_without_49_m2": float(np.median(A3[not49])), "A_F_table1_m2": MD_TABLE1_MEDIAN_M2, "b_hat": b_hat, "alpha_b_hat": float(f"{alpha_b:.2f}"),
             "alphas": alphas, "C_F": {f"{al:g}": float(np.median(A3 * np.where(Sdn > 0, Sdn, np.nan) ** al)) if al > 0 else float(np.median(A3)) for al in alphas},
             "C_F_without_49": {f"{al:g}": float(np.nanmedian((A3 * np.where(Sdn > 0, Sdn, np.nan) ** al)[not49])) if al > 0 else float(np.median(A3[not49])) for al in alphas},
             "C_F_definition": "median over the 64 picks of A_3 * S_down^alpha (S_down the 10 m D8 reach slope below the A_3 cell)"}
    out = {"csv": csv_rec, "grid": {"shape": [ny, nx], "dx": dx, "dy": dy, "cell_area_m2": cell_area, "crs": str(crs), "transform": list(tr)[:6], "files": TTV_FULL},
           "picks": rows, "fits": fits, "rules": rules, "AS2": as2, "distributions": dists, "field_values": field, "md_eq10": MD_EQ10,
           "picked_network_cells": int(rr.size), "wall_s": time.time() - t0}
    dump(out, out_path("heads", "heads.json"))
    print("fits:", {k: (round(v["a"]), round(v["b"], 3), round(v["R2"], 3)) for k, v in fits.items()}, "R4", r4["pass"], "R5", rules["R5"]["verdict"], flush=True)
    print("A_F", field["A_F_m2"], "alpha_b", field["alpha_b_hat"], "C_F", field["C_F"], flush=True)
    if a.dinf:
        dinf_window(Z, FD, Ac, tr, dx, dy, rows, a.window_margin_m, t0)
    if a.dinf_full:
        dinf_full(Z, FD, Ac, dx, dy, rows, t0)


def dinf_full(Z, FD, Ac, dx, dy, rows, t0):
    """Decision 3 as written: A_3 from D-infinity routing on the full TTV grid (the package's routine; pilot 35 us per cell)."""
    from lemcvt.hydro.dinf import compute_dinf_area, compute_dinf_flow_direction
    t1 = time.perf_counter()
    ang, rf, fi = compute_dinf_flow_direction(Z, dx, FD_d8=FD)
    t2 = time.perf_counter()
    print(f"dinf full: directions {t2 - t1:.0f} s", flush=True)
    del FD
    Ad = compute_dinf_area(Z.astype(float), dx, ang, rf, fi)
    del ang, rf, fi
    t3 = time.perf_counter()
    print(f"dinf full: areas {t3 - t2:.0f} s", flush=True)
    per = []
    for x in rows:
        r, c = x["A_3_row"], x["A_3_col"]
        r0, c0 = x["row"], x["col"]
        cells3, rr, cc = within(Ad, r0, c0, HEAD_RADII_M["A_3"], dx, dy)
        cells10, _, _ = within(Ad, r0, c0, HEAD_RADII_M["A_10"], dx, dy)
        per.append({"pick_id": x["pick_id"], "A_3_d8_m2": x["A_3_m2"], "A_3_dinf_m2": float(cells3), "A_10_d8_m2": x["A_10_m2"], "A_10_dinf_m2": float(cells10),
                    "A_dinf_at_pick_cell_m2": float(Ad[r0, c0]), "A_dinf_at_d8_A3_cell_m2": float(Ad[r, c])})
    rec = {"mode": "full grid", "n_cells": int(np.isfinite(Z).sum()), "wall_s": t3 - t1, "core_h": (t3 - t1) / 3600.0, "cap_core_h": DINF_MAX_CORE_H, "picks": per,
           "A_3_dinf_pct": pct([q["A_3_dinf_m2"] for q in per]), "A_10_dinf_pct": pct([q["A_10_dinf_m2"] for q in per]),
           "ratio_A3_dinf_over_d8_median": float(np.median([q["A_3_dinf_m2"] / q["A_3_d8_m2"] for q in per if q["A_3_d8_m2"] > 0])),
           "A_F_dinf_m2": float(np.median([q["A_3_dinf_m2"] for q in per]))}
    dump(rec, out_path("heads", "dinf_full.json"))
    print("dinf full:", rec["n_cells"], "cells", f"{rec['core_h']:.2f} core-h", "A_3 dinf median", rec["A_3_dinf_pct"].get("p50"), flush=True)


def dinf_window(Z, FD, Ac, tr, dx, dy, rows, margin, t0):
    from lemcvt.hydro.d8 import d8_area
    from lemcvt.hydro.dinf import compute_dinf_area, compute_dinf_flow_direction
    r3 = np.array([x["A_3_row"] for x in rows])
    c3 = np.array([x["A_3_col"] for x in rows])
    kr, kc = int(math.ceil(margin / dy)), int(math.ceil(margin / dx))
    r0, r1 = max(r3.min() - kr, 0), min(r3.max() + kr + 1, Z.shape[0])
    c0, c1 = max(c3.min() - kc, 0), min(c3.max() + kc + 1, Z.shape[1])
    Zw = Z[r0:r1, c0:c1].astype(float)
    FDw = FD[r0:r1, c0:c1]
    rec = {"window_rows": [int(r0), int(r1)], "window_cols": [int(c0), int(c1)], "n_cells": int(np.isfinite(Zw).sum()), "margin_m": margin}
    t1 = time.perf_counter()
    A8w = d8_area(FDw, Zw, dx)          # D8 area recomputed inside the window: equals the full-grid area where the catchment lies inside
    ang, rf, fi = compute_dinf_flow_direction(Zw, dx, FD_d8=FDw)
    Ad = compute_dinf_area(Zw, dx, ang, rf, fi)
    rec["wall_s"] = time.perf_counter() - t1
    per = []
    for x in rows:
        r, c = x["A_3_row"] - r0, x["A_3_col"] - c0
        a8_full = float(Ac[x["A_3_row"], x["A_3_col"]]) * dx * dy
        a8_win = float(A8w[r, c])
        cells, rr, cc = within(Ad, r, c, HEAD_RADII_M["A_3"], dx, dy)
        per.append({"pick_id": x["pick_id"], "A_3_d8_m2": x["A_3_m2"], "A_3_dinf_m2": float(cells), "A_dinf_at_A3_cell_m2": float(Ad[r, c]),
                    "d8_area_window_m2": a8_win, "d8_area_full_m2": a8_full, "contained": bool(abs(a8_win - a8_full) <= 0.5 * dx * dy)})
    rec["picks"] = per
    rec["n_contained"] = int(sum(q["contained"] for q in per))
    v = np.array([q["A_3_dinf_m2"] for q in per if q["contained"]])
    rec["A_3_dinf_pct_contained"] = pct(v)
    rec["ratio_dinf_over_d8_median"] = float(np.median([q["A_3_dinf_m2"] / q["A_3_d8_m2"] for q in per if q["contained"] and q["A_3_d8_m2"] > 0]))
    dump(rec, out_path("heads", "dinf_window.json"))
    print("dinf window:", rec["n_cells"], "cells", f"{rec['wall_s']:.0f} s", "contained", rec["n_contained"], "median A_3 dinf", rec["A_3_dinf_pct_contained"].get("p50"), flush=True)


if __name__ == "__main__":
    main()
