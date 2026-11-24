#!/usr/bin/env python3
"""Addendum 22 (Tennessee Valley clean-up: M&D area only, one-to-one head metrics, plain-language figures): shared constants and helpers.

Directive `reports/directives/2026-11-23T1600PST_v2_addendum_22_ttv_head_metrics_cleanup.md`. Decisions of its section 2 and the readings of
its section 6 are constants here, fixed before any result. No package code is changed: the polygon misfit is the masked Gram sum of
`lemcvt.analyses.sensitivity._terms_masked` (the reduction is a per-cell linear form, so the mask is a `keep` array over the profiler's
cells), and the AD polygon misfit sets the observed elevation to NaN outside the polygon before `ADProfiler.best_y` (its cell mask is
`isfinite(obs)`); both are checked against the addendum 21 fits with the mask set to every cell (relative 1e-8).

Choices made in applying the directive (listed in the report's state check):
  - The M&D study polygon is rasterised on the tile's 1 m lattice exactly as in addenda 19-21 (`md_polygon_mask`, ALL_TOUCHED=FALSE);
    a 12 m head or channel cell is "inside the polygon" when its centre (the 1 m cell (6, 6) of the block) is inside it; a mapped head is
    inside when its 1 m cell is.
  - Head finding (directive 2.3) is the addendum 21 extraction on each surface's own 12 m tile lattice (83 x 83 full blocks of the 1 m tile,
    the 4 m remainder dropped; pit-filled; D8 with `border='outlets'`; S_12 to the largest-area donor; T = A S^2; a head where the upstream
    maximum of T first reaches C_F). The 12 m statistics M and Mdon are stored so that any threshold (the 16-placement medians) is a
    re-application, not a rerun.
  - One-to-one matching: Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on the rectangular distance matrix with every pair
    farther than t set to a large constant, so the assignment first maximises the number of pairs within t and then minimises their
    total distance; pairs farther than t are then dropped.
  - Along-valley offsets are traced on the observed surface's 12 m D8 (the obs tile-lattice routing); the mapped head's cell is the 12 m
    block holding the pick, the model head's cell is its 12 m cell; "on or within one cell of the path" is 8-neighbour adjacency; the
    offset is the path distance (cell-to-cell D8 steps, 12 or 12 sqrt 2 m) from the start cell to the path cell at or adjacent to the other
    head. The downstream trace is tried first (model channel too short, offset > 0), then the upstream one (too long, offset < 0).
  - Channel length inside the polygon: every 12 m channel cell whose centre is inside the polygon contributes the distance to its D8 receiver
    (12 or 12 sqrt 2 m; nothing where it has none). The M&D traced length is the gpkg `channels` layer clipped to the tile rectangle
    intersected with the study polygon (OGR geometry intersection).
  - Local slope for the extra-head classification is the gradient magnitude of the pit-filled 12 m surface (`np.gradient`); "steeper than
    the steepest 5 %" is the 95th percentile of the real landscape's 12 m slope inside the polygon on the same tile; "tile edge" is a 12 m
    cell whose centre lies within 24 m of the tile boundary (the two outer rings).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("A21_RUN", "20261123T184712Z_a21_ttv_channel_head_test")
from a21_common import (  # noqa: E402,F401  # noqa: E402
    A18_SUMMARY,
    A_F_TABLE1,
    ALPHA_PRIMARY,
    BLOCK_M,
    D_PHYS,
    DCOL,
    DROW,
    E_RIDGE,
    HOLDOUT_A_C,
    K_CAL,
    M_SITE,
    MD_GPKG,
    MD_TILES,
    RESCALE,
    RESULTS_ROOT,
    SCRATCH,
    SJC_A_RANGE,
    SJC_REFINE_MAX,
    SJC_THIN,
    U_NEW,
    a18_specs,
    a20_path,
    a21_run_dir,
    block_mean,
    channel_heads,
    coarse_grid,
    criterion_sets,
    donor_slope,
    dump,
    head_statistic_coarse,
    hillshade,
    json_ready,
    load_json,
    mask_from_entries,
    md_lines_mask,
    md_polygon_mask,
    metre_tile,
    picks_on_tile,
    receiver_flat,
    slurm_record,
)
from a21_common import out_path as a21_out_path

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DIRECTIVE = "reports/directives/2026-11-23T1600PST_v2_addendum_22_ttv_head_metrics_cleanup.md"
A21_REPORT = "reports/v2/20261123T1245PST_ttv_channel_head_test.md"
FIG_DIR = os.path.join(REPO, "reports", "v2", "figures", "a22")
JOB_DIR = os.path.join(SCRATCH, "lemcvt_jobs", "a22")

# ----------------------------------------------------------------------------- decision 2.1 (scope and parameter values)
TILES = ["TTV1", "TTV2", "TTV3"]
A_F = 6120.0                        # m2, area threshold at M&D's heads (full-site 12 m lattice, addendum 21 section 3)
C_F = 876.2715906699304             # m2, area x slope^2 threshold (addendum 21 grid12.json; the directive rounds it to 876)
A_F_PLACEMENT_MEDIAN = 6228.0       # the 16-placement medians (sensitivity only; addendum 21 appendix F4)
C_F_PLACEMENT_MEDIAN = 817.0
MASK_CHECK_TOL = 1e-8               # decision 2.1: the all-cells mask reproduces the addendum 21 fit
N_MAPPED_HEADS_POOLED = 61          # mapped heads inside the polygon on TTV1-TTV3 (figure 2's dashed line)

# ----------------------------------------------------------------------------- decision 2.2 (the models, plain names, colours, markers)
MODELS = ["obs", "AD", "PF_A", "PF_C", "SJ_C", "P4", "F2_relief", "F2_phys"]
PLAIN = {"obs": "Real landscape (lidar)", "AD": "AD model", "PF_A": "Split model, M&D area rule", "PF_C": "Split model, M&D area × slope² rule",
         "SJ_C": "Split model, area × slope² rule fitted", "P4": "Split model, area rule fitted", "F2_relief": "Split model, nothing fitted",
         "F2_phys": "Split model, nothing fitted, physical D"}
WHAT = {"obs": "The observed surface, run through the same head-finding method as the models",
        "AD": "Co-located advection–diffusion; incision and diffusion act in every cell",
        "PF_A": "Stream power where drainage area ≥ 6,120 m² (measured at M&D's heads), diffusion elsewhere",
        "PF_C": "Stream power where area × slope² ≥ 876 m² (measured at M&D's heads), diffusion elsewhere",
        "SJ_C": "As above, but the threshold is chosen by the fit",
        "P4": "Area threshold chosen by the fit (the original tuned model)",
        "F2_relief": "M&D area × slope² rule, K = measured K₀, D = relief D",
        "F2_phys": "As above, but D = Heimsath's 0.0025 m²/yr"}
FITTED = {"obs": "none", "AD": "K, D", "PF_A": "K, D", "PF_C": "K, D", "SJ_C": "K, D, threshold", "P4": "K, D, threshold", "F2_relief": "none", "F2_phys": "none"}
COLOUR = {"obs": "#000000", "AD": "#d95f02", "PF_A": "#1f78b4", "PF_C": "#08306b", "SJ_C": "#1b9e77", "P4": "#7570b3", "F2_relief": "#666666", "F2_phys": "#bbbbbb"}
MARKER = {"obs": "*", "AD": "o", "PF_A": "^", "PF_C": "s", "SJ_C": "D", "P4": "p", "F2_relief": "P", "F2_phys": "X"}
MAPPED_COLOUR = "#e6550d"
FITTED_MODELS = ["AD", "PF_A", "PF_C", "SJ_C", "P4"]
FIXED_MODELS = ["F2_relief", "F2_phys"]
MAP_PANELS = ["obs", "AD", "PF_C", "PF_A", "SJ_C", "F2_relief"]      # 5.4 item 6

# ----------------------------------------------------------------------------- decision 2.4 (metrics) and section 6 (readings)
TOLERANCES_M = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
T_PRIMARY_M = 25.0
EDGE_M = 24.0
STEEP_PCT = 95.0
SHARE_FOUND_MIN = 0.5
SHARE_REAL_MIN = 0.5
OFFSET_MAX_M = 25.0
K_PHYS_RANGE = (0.5, 2.0)
D_RELIEF_RANGE = (0.67, 1.5)
TOL_VERDICT_RANGE = (15.0, 40.0)
AD_N_GRID = 200                     # addendum 18 AD profile grid (reproduced for the all-cells check)
P4_THIN = 2000                      # addendum 18 P4 grid (thin of the unique areas in ACRIT_RANGE) and staircase refinement
P4_REFINE_MAX = 1000
P4_ACRIT_RANGE = (1e2, 1e5)
BIG = 1e9

# ----------------------------------------------------------------------------- run directory


def a22_run_dir() -> str:
    rid = os.environ.get("A22_RUN")
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a22_ttv_cleanup_head_metrics"
        os.environ["A22_RUN"] = rid
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def out_path(*parts: str) -> str:
    p = os.path.join(a22_run_dir(), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def a21_path(*parts: str) -> str:
    return os.path.join(a21_run_dir(), *parts)


def unpack(bits: np.ndarray, shape) -> np.ndarray:
    return np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool)


# ----------------------------------------------------------------------------- the polygon misfit

def terms_of(red, Z_obs_flat: np.ndarray, keep: np.ndarray | None) -> dict[str, float]:
    from lemcvt.analyses.sensitivity import _terms_masked
    return _terms_masked(red, Z_obs_flat, keep)


def fit_terms(t: dict[str, float], U: float, K0: float, lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    """The closed-form box minimum in (U/K, U/D) of one Gram-term set, with the K = K0 variant."""
    from lemcvt.analyses.sensitivity import _fit, _q
    out: dict[str, Any] = {"n_cells": int(t.get("n", 0))}
    if not t.get("n", 0):
        return out
    x, y, where = _fit(t, "joint", lo, hi, U / K0, None)
    out["free_KD"] = {"mse": _q(t, x, y), "K": U / x, "D": U / y, "where": where, "K_over_K0": (U / x) / K0}
    xk, yk, wk = _fit(t, "fix_x", lo, hi, U / K0, None)
    out["K_fixed"] = {"mse": _q(t, xk, yk), "K": K0, "D": U / yk, "where": wk}
    return out


def mse_at(t: dict[str, float], U: float, K: float, D: float) -> float:
    from lemcvt.analyses.sensitivity import _q
    return _q(t, U / K, U / D)


def reduce_mask(prof, mask: np.ndarray, m: float):
    """One exact reduction of the partitioned model with an arbitrary 1 m channel mask (the addendum 21 mask route)."""
    gd = prof.gd
    saved = prof.stat
    prof.stat = mask[gd["rev_j"], gd["rev_i"]].astype(float)
    try:
        return prof.reduce(0.5, m)
    finally:
        prof.stat = saved


def partitioned_surface(prof, red, K: float, D: float, Z_obs: np.ndarray) -> tuple[np.ndarray, int]:
    flat = red.Z_flat(K, D)
    Z = prof.to_grid(flat)
    hole = np.isfinite(Z_obs) & ~np.isfinite(Z)
    Z[hole] = Z_obs[hole]
    return Z, int(hole.sum())


def tile_setup(tile: str) -> dict[str, Any]:
    """The tile, its profiler inputs, bounds at the new U (the addendum 18 box in (U/K, U/D)), the polygon and the addendum 21 records."""
    td = metre_tile(tile)
    mp = td.cfg.model
    U, m = float(mp.U), float(mp.m)
    assert abs(U - U_NEW) < 1e-15 and abs(m - M_SITE) < 1e-12, (U, m)
    K0 = float(mp.K0_partitioned)
    Kb = (mp.K_bounds[0] * RESCALE, mp.K_bounds[1] * RESCALE)
    Db = (mp.D_bounds[0] * RESCALE, mp.D_bounds[1] * RESCALE)
    lo = np.array([U / Kb[1], U / Db[1]])
    hi = np.array([U / Kb[0], U / Db[0]])
    a21 = load_json(a21_path("tiles", f"{tile}.json"))
    a21s = load_json(a21_path("sjc", f"{tile}.json"))
    poly = md_polygon_mask(td)
    pick_rc, pick_ids = picks_on_tile(td)
    return {"td": td, "Z": td.Z, "A": td.A, "FD": td.FD, "dx": td.dx, "dy": td.filled.dy, "U": U, "m": m, "K0": K0, "Kb": Kb, "Db": Db, "lo": lo, "hi": hi,
            "a21": a21, "a21_sjc": a21s, "poly": poly, "pick_rc": pick_rc, "pick_ids": pick_ids, "D_relief": float(a21["D"]["D_relief"]),
            "D_relief_network": a21["D"]["D_relief_network"]}


def keep_masks(prof, poly: np.ndarray) -> dict[str, np.ndarray]:
    gd = prof.gd
    return {"all": np.ones(prof.n, dtype=bool), "poly": poly[gd["rev_j"], gd["rev_i"]]}


# ----------------------------------------------------------------------------- the 12 m products of a surface (decision 2.3)

def coarse_products(Zs: np.ndarray, dx: float) -> dict[str, np.ndarray]:
    """Block-mean to 12 m, fill, route, S_12 to the largest-area donor, and the head statistics M, Mdon at alpha = 2 and alpha = 0."""
    g = coarse_grid(Zs, BLOCK_M, dx)
    sl = donor_slope(g["Zf"], g["A"], g["FD"], g["dx"], 1)
    h2 = head_statistic_coarse(g["A"], sl["S"], g["FD"], ALPHA_PRIMARY)
    h0 = head_statistic_coarse(g["A"], sl["S"], g["FD"], 0.0)
    zy, zx = np.gradient(np.where(np.isfinite(g["Zf"]), g["Zf"], np.nan), g["dx"])
    return {"Z12": g["Zb"].astype(np.float32), "Zf12": g["Zf"].astype(np.float32), "FD12": g["FD"].astype(np.uint8), "A12": g["A"].astype(np.float32),
            "S12": sl["S"].astype(np.float32), "M_a2": h2["M"].astype(np.float64), "Mdon_a2": h2["Mdon"].astype(np.float64),
            "M_a0": h0["M"].astype(np.float64), "Mdon_a0": h0["Mdon"].astype(np.float64), "slope12": np.hypot(zx, zy).astype(np.float32)}


def poly12_centres(poly: np.ndarray, nb: int) -> np.ndarray:
    """12 m cells whose centre (the 1 m cell (6, 6) of the block) is inside the polygon."""
    return poly[BLOCK_M // 2: nb * BLOCK_M: BLOCK_M, BLOCK_M // 2: nb * BLOCK_M: BLOCK_M][:nb, :nb]


def heads_from_products(P: dict[str, np.ndarray], alpha: float, C: float) -> tuple[np.ndarray, np.ndarray]:
    key = "a2" if alpha == ALPHA_PRIMARY else "a0"
    return criterion_sets(P[f"M_{key}"], P[f"Mdon_{key}"], C)


# ----------------------------------------------------------------------------- decision 2.4 metrics

def match_heads(mapped_xy: np.ndarray, model_xy: np.ndarray, t: float) -> dict[str, Any]:
    """One-to-one pairing within t (Hungarian on the rectangular distance matrix; pairs beyond t cost BIG and are dropped)."""
    from scipy.optimize import linear_sum_assignment
    n_map, n_mod = mapped_xy.shape[0], model_xy.shape[0]
    out: dict[str, Any] = {"t": t, "n_mapped": int(n_map), "n_model": int(n_mod)}
    if n_map == 0 or n_mod == 0:
        out.update({"n_matched": 0, "n_missed": int(n_map), "n_extra": int(n_mod), "pairs": [], "share_found": 0.0 if n_map else None,
                    "share_real": 0.0 if n_mod else None, "matched_dist_m": []})
        return out
    d = np.hypot(mapped_xy[:, None, 0] - model_xy[None, :, 0], mapped_xy[:, None, 1] - model_xy[None, :, 1])
    cost = np.where(d <= t, d, BIG)
    ri, ci = linear_sum_assignment(cost)
    ok = d[ri, ci] <= t
    pairs = [(int(i), int(j), float(d[i, j])) for i, j in zip(ri[ok], ci[ok], strict=True)]
    n_m = len(pairs)
    out.update({"n_matched": n_m, "n_missed": int(n_map - n_m), "n_extra": int(n_mod - n_m), "pairs": pairs, "share_found": n_m / n_map, "share_real": n_m / n_mod,
                "matched_dist_m": [p[2] for p in pairs]})
    return out


def trace_cells(FD12: np.ndarray, r: int, c: int, max_steps: int = 20000) -> tuple[list[tuple[int, int]], list[float]]:
    """The D8 path from (r, c) as a list of cells and the cumulative path distance (m) at each."""
    ny, nx = FD12.shape
    path = [(r, c)]
    dist = [0.0]
    for _ in range(max_steps):
        code = int(FD12[r, c])
        if code == 0:
            break
        nr, nc = r + int(DROW[code]), c + int(DCOL[code])
        if not (0 <= nr < ny and 0 <= nc < nx):
            break
        dist.append(dist[-1] + BLOCK_M * math.hypot(nr - r, nc - c))
        r, c = nr, nc
        path.append((r, c))
    return path, dist


def _on_or_adjacent(path: list[tuple[int, int]], dist: list[float], cell: tuple[int, int]) -> float | None:
    """Path distance to the first path cell equal to or 8-adjacent to ``cell`` (None if the path never comes within one cell)."""
    for (pr, pc), dd in zip(path, dist, strict=True):
        if abs(pr - cell[0]) <= 1 and abs(pc - cell[1]) <= 1:
            return dd
    return None


def valley_offset(FD12_obs: np.ndarray, mapped_cell: tuple[int, int], model_cell: tuple[int, int]) -> dict[str, Any]:
    """Decision 2.4 item 2 on the observed 12 m routing: + path distance when the model head lies on the mapped head's downstream path
    (channel too short), - when the mapped head lies on the model head's path (too long), else 'neighbouring hollow'."""
    if mapped_cell == model_cell:
        return {"kind": "same cell", "offset_m": 0.0}
    path, dist = trace_cells(FD12_obs, *mapped_cell)
    d = _on_or_adjacent(path, dist, model_cell)
    if d is not None:
        return {"kind": "too short" if d > 0 else "same cell", "offset_m": float(d)}
    path, dist = trace_cells(FD12_obs, *model_cell)
    d = _on_or_adjacent(path, dist, mapped_cell)
    if d is not None:
        return {"kind": "too long" if d > 0 else "same cell", "offset_m": float(-d)}
    return {"kind": "neighbouring hollow", "offset_m": None}


def channel_length_m(channel12: np.ndarray, FD12: np.ndarray, inside12: np.ndarray) -> float:
    """Sum over channel cells inside the polygon of the distance to the D8 receiver (12 or 12 sqrt 2 m)."""
    ny, nx = FD12.shape
    recv = receiver_flat(FD12)
    cells = np.nonzero((channel12 & inside12).ravel() & (recv >= 0))[0]
    if cells.size == 0:
        return 0.0
    rr, cc = cells // nx, cells % nx
    tr, tc = recv[cells] // nx, recv[cells] % nx
    return float(np.sum(BLOCK_M * np.hypot(tr - rr, tc - cc)))


def classify_extra(slope12: np.ndarray, steep_threshold: float, cells: list[tuple[int, int]], nb: int) -> dict[str, Any]:
    """Decision 2.4 item 4: every extra head is 'too-steep wall' (slope above the real landscape's 95th percentile), 'tile edge'
    (centre within 24 m of the boundary) or 'other'; a wall head on the edge counts as a wall."""
    edge_cells = int(math.ceil(EDGE_M / BLOCK_M - 0.5 + 1e-9))      # centres at 6 and 18 m are within 24 m; 30 m is not
    out = {"too-steep wall": 0, "tile edge": 0, "other": 0, "cells": []}
    for r, c in cells:
        s = float(slope12[r, c])
        on_edge = r < edge_cells or c < edge_cells or r >= nb - edge_cells or c >= nb - edge_cells
        if np.isfinite(s) and s > steep_threshold:
            k = "too-steep wall"
        elif on_edge:
            k = "tile edge"
        else:
            k = "other"
        out[k] += 1
        out["cells"].append({"r": int(r), "c": int(c), "slope": s, "class": k})
    return out


def md_traced_length_in_tile_polygon(td) -> dict[str, float]:
    """M&D's traced channel length (gpkg `channels`) inside the study polygon intersected with the tile rectangle, in metres."""
    from osgeo import ogr
    ogr.UseExceptions()
    tr = td.filled.transform
    ny, nx = td.Z.shape
    x0, y1 = tr.c, tr.f
    x1, y0 = x0 + nx * tr.a, y1 + ny * tr.e
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
        ring.AddPoint(x, y)
    rect = ogr.Geometry(ogr.wkbPolygon)
    rect.AddGeometry(ring)
    src = ogr.Open(MD_GPKG)
    sa = src.GetLayerByName("study_area")
    poly = None
    for f in sa:
        g = f.GetGeometryRef().Clone()
        g.FlattenTo2D()
        poly = g if poly is None else poly.Union(g)
    clip = poly.Intersection(rect)
    ch = src.GetLayerByName("channels")
    total = 0.0
    total_tile = 0.0
    for f in ch:
        g = f.GetGeometryRef().Clone()
        g.FlattenTo2D()
        gi = g.Intersection(rect)
        if gi is not None and not gi.IsEmpty():
            total_tile += gi.Length()
        gc = g.Intersection(clip)
        if gc is not None and not gc.IsEmpty():
            total += gc.Length()
    return {"length_in_tile_and_polygon_m": float(total), "length_in_tile_m": float(total_tile), "polygon_area_in_tile_m2": float(clip.GetArea())}


def reading_heads(share_found: float | None, share_real: float | None, median_offset: float | None) -> dict[str, Any]:
    a = share_found is not None and share_found >= SHARE_FOUND_MIN
    b = share_real is not None and share_real >= SHARE_REAL_MIN
    c = median_offset is not None and abs(median_offset) <= OFFSET_MAX_M
    return {"share_found_ok": bool(a), "share_real_ok": bool(b), "offset_ok": bool(c), "reproduces_mapped_heads": bool(a and b and c)}


_ = (json, A18_SUMMARY, A_F_TABLE1, D_PHYS, E_RIDGE, HOLDOUT_A_C, K_CAL, MD_TILES, SJC_A_RANGE, SJC_REFINE_MAX, SJC_THIN, a18_specs, a20_path, block_mean,
     channel_heads, dump, hillshade, json_ready, mask_from_entries, md_lines_mask, slurm_record, a21_out_path)
