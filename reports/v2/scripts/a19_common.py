#!/usr/bin/env python3
"""Addendum 19 (hilltop curvature on a fixed network, the D bracket, Steve's TTV heads, the area-slope threshold): shared constants and
helpers for the a19_* scripts.

Directive `reports/directives/2026-11-20T0755PST_v2_addendum_19_hilltops_heads_slope.md` (the 07:55 PST revision). Decisions of its
section 1 as constants, rules of its section 6 as thresholds; all fixed before any result. Package code is not changed: everything new lives here
and imports from `lemcvt`.

Choices made in applying the directive (every one is listed in the report's state check):
  - L_w "measured to the 10^3 network" makes the crest width 2 x the hillslope length, so L_w is the largest tested window (51 m) on nearly
    every tile; D0_N_B is computed as written and a labelled D0_N_B' (crest width to the rollover network, the Part B meaning) beside it.
  - The C_ht >= 0 drop is new against Part B (which kept non-convex pixels); D0_N is also reported without it (labelled).
  - Filter order: divide cells between outlet basins of the 10^3 network -> edge margin, |grad z| <= 0.4, distance >= L_f/2 (as Part B)
    -> road/trail buffer -> 8-connected segments < 50 px dropped -> C_ht(L_f) >= 0 dropped (a value filter, last). Segments for the
    bootstrap are the 8-connected groups of the counted pixels.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import GROUP_HOME, RESULTS_ROOT, SCRATCH, SOURCE, dump, sha256_file, slurm_record  # noqa: E402,F401
from a17_common import E_M_PER_YR, HILLSHADE, SURVEY, hillshade  # noqa: E402,F401
from a17b_common import TIER2, site_of, tile_registry  # noqa: E402,F401
from a18_common import (  # noqa: E402,F401
    A15,
    AP1_3DEP,
    AP_PAMAP,
    AP_PRIMARY,
    CLEAN,
    FIT_STEPS,
    FITTED_AT_SITE_M,
    GM_NCALM,
    GM_OFFMESA,
    GM_PRIMARY,
    GROUPS,
    HOLD_FRAC,
    NO_D0,
    PARTB,
    PARTB_MEASUREMENT,
    PARTB_PER_TILE,
    PRIMARY,
    SITE_M,
    SURVEY_PAIRS,
    TTV_PRIMARY,
    fnum,
    hold_n,
    partb_measurement,
    partb_per_tile,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DIRECTIVE = "reports/directives/2026-11-20T0755PST_v2_addendum_19_hilltops_heads_slope.md"
A18 = os.path.join(RESULTS_ROOT, "20261119T143945Z_a18_expanded_fits")
A18_SPECS = os.path.join(A18, "prep", "fit_specs.json")
A18_SUMMARY = os.path.join(A18, "summary")
A18_REPORT = "reports/v2/20261119T2100PST_expanded_fits.md"
A16 = os.path.join(RESULTS_ROOT, "20261118T185206Z_a16_tile_finish")
MD_TABLE1 = os.path.join(A16, "md_georef", "md1989_table1.csv")
MD_GPKG = os.path.join(A15, "md_georef", "md1989_fig3.gpkg")
TTV_GRID = os.path.join(A15, "grid_TTV")
TTV_FULL = {k: os.path.join(TTV_GRID, f"TTV_1m_{k}.tif") for k in ("filled", "fd", "area_cells")}
HEADS_SRC = os.path.join(REPO, "reports", "directives", "ttv_heads_manual_20261119T1911.csv")   # the file Steve placed (directive 0.3 names it differently)
HEADS_DST = os.path.join(REPO, "data", "ttv", "ttv_heads_manual_steve_20261119.csv")             # decision 2
ROADS_DIR = os.path.join(SOURCE, "TIGER2024_roads")
TIGER_ROADS_URL = "https://www2.census.gov/geo/tiger/TIGER2024/ROADS/tl_2024_{fips}_roads.zip"
TIGER_COUNTY_ZIP = os.path.join(ROADS_DIR, "tl_2024_us_county.zip")
FIG_DIR = os.path.join(REPO, "reports", "v2", "figures", "a19")
QC_DIR = os.path.join(FIG_DIR, "hilltop_qc")
JOB_DIR = os.path.join(SCRATCH, "lemcvt_jobs", "a19")

# ----------------------------------------------------------------------------- decisions (section 1)
A_NET_M2 = 1e3                       # decision 1: the fixed network (the k_s network)
L_F_M = 11.0
W_F = 5
SLOPE_MAX = 0.4
EDGE_MARGIN = 2
SEGMENT_MIN_PX = 50                  # decision 1: segments under 50 px dropped (Hurst et al. 2012: 50 m)
ROAD_BUFFER_M = 10.0                 # decision 1: pixels within 10 m of a mapped road or trail dropped
TEST_HALF_WIDTHS = (3, 4, 5, 7, 10, 12, 15, 20, 25)          # the Part B window test: L = 7 ... 51 m
SCAN_HALF_WIDTHS = tuple(range(1, 16))                       # decision 11: L = 3, 5, ..., 31 m
ALL_HALF_WIDTHS = tuple(sorted(set(TEST_HALF_WIDTHS) | set(SCAN_HALF_WIDTHS)))
WINDOW_TOL = 0.10
N_BOOT = 2000                        # decision 1: the segment bootstrap
SEED = 20261120
D0_MIN_PIXELS = 500                  # addendum 18 decision 4 applies to D0_N
E_SITE = dict(E_M_PER_YR)            # m/yr: GM 1.47e-4, AP 5.5e-5, TTV 7.7e-5
PERRON_2012 = {"GM": {"E_m_per_Myr": 147.0, "curvature_per_m": 11.8e-3, "D_cm2_per_yr": 124.0, "D_pm": 19.0},
               "AP": {"E_m_per_Myr": 55.0, "curvature_per_m": 5.5e-3, "D_cm2_per_yr": 100.0, "D_pm": 8.0}}      # decision 10 (Supplementary Table S2, as the directive states them)
L_BREAK_LITERATURE_M = (8.0, 20.0)   # rule R12: Hurst et al. 2012 (12 m) and Struble & Roering 2021 (15-20 m)
E_STAR_BANDS = (10.0, 30.0)          # decision 13
TTV_ANCHOR = {"soil_flux_convention_m2_per_yr": 5e-3, "bedrock_convention_m2_per_yr": 2.5e-3, "rho_s_over_rho_r": 0.5}   # decision 14
HEAD_RADII_M = {"A_3": 3.0, "A_10": 10.0}                    # decision 3
S_UP_REACHES_M = (10.0, 5.0, 20.0)                            # primary 10 m; 5 and 20 m sensitivity
S_DOWN_REACH_M = 10.0                                         # slope_threshold.valley_slope_reach
MD_EQ10 = {"a_m2": 1978.0, "b": 1.65, "R2": 0.75}             # decision 4
MD_TABLE1_MEDIAN_M2 = 7400.0
N_BOOT_FITS = 10000                                           # decision 4
ALPHAS_FIXED = (0.0, 2.0)                                     # decision 5b: plus b_hat rounded to 0.05
DINF_MAX_CORE_H = 10.0                                        # decision 3
BUDGET_CORE_H = 320.0
MAX_SUBMISSION_LIMIT_CORE_H = 300.0
MAX_CONCURRENT_CPUS = 300

# ----------------------------------------------------------------------------- rules (section 6)
R1_MIN_FRAC_TILES = 0.90
R4_A_TOL, R4_B_TOL = 0.10, 0.05
R5_RATIO_RANGE = (0.5, 2.0)
R6_MIN = 9                            # of 11 TTV tiles
R7_IMPROVEMENT, R7_MIN = 0.02, 9
R8_MEDIAN_DIST_M, R8_RATIO_RANGE = 25.0, (0.67, 1.5)
R10_FRAC = 0.80
R11_RANGE = (0.67, 1.5)
F4_LOG10 = 1.0

MEASURED_TILES = list(dict.fromkeys(GM_PRIMARY + GM_OFFMESA + GM_NCALM + AP_PRIMARY + AP_PAMAP + AP1_3DEP + TTV_PRIMARY))   # 73 (2.1)
MD_TILES = ["TTV1", "TTV2", "TTV3"]                                                                                      # decision 6
GROUP_OF_SPEC_TILE = {k: v for k, v in GROUPS.items()}


def run_dir(name: str | None = None) -> str:
    rid = os.environ.get("A19_RUN") or name
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a19_hilltops_heads_slope"
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def out_path(*parts: str) -> str:
    p = os.path.join(run_dir(), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def a18_specs() -> dict[str, dict]:
    return load_json(A18_SPECS)["specs"]


def md_table1() -> list[dict]:
    with open(MD_TABLE1) as f:
        return [{"row": int(r["row"]), "S": float(r["local_valley_slope_m_per_m"]), "A": float(r["source_area_m2"]),
                 "L": float(r["source_basin_length_m"]), "W": float(r["colluvial_fill_width_m"])} for r in csv.DictReader(f)]


# ----------------------------------------------------------------------------- statistics

def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(v, kind="stable")
    cw = np.cumsum(w[order])
    if cw[-1] == 0:
        return float("nan")
    return float(v[order][np.searchsorted(cw, 0.5 * cw[-1])])


def segment_bootstrap_median(vals: np.ndarray, seg: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, min_segments: int = 5) -> dict[str, Any]:
    """Median with the segment bootstrap of addendum 8 section 2.2 (segments resampled with replacement, weighted median)."""
    vals = np.asarray(vals, dtype=float)
    seg = np.asarray(seg)
    ok = np.isfinite(vals)
    vals, seg = vals[ok], seg[ok]
    if vals.size == 0:
        return {"median": float("nan"), "n": 0, "n_segments": 0}
    uniq, inv = np.unique(seg, return_inverse=True)
    rec: dict[str, Any] = {"median": float(np.median(vals)), "n": int(vals.size), "n_segments": int(uniq.size)}
    if n_boot > 0 and uniq.size >= min_segments:
        rng = np.random.default_rng(seed)
        bs = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.integers(0, uniq.size, uniq.size)
            counts = np.bincount(pick, minlength=uniq.size)
            bs[i] = weighted_median(vals, counts[inv].astype(float))
        rec["ci68"] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
        rec["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return rec


def ratio_interval(E: float, c: dict[str, Any]) -> dict[str, Any]:
    """D0 = E / C_ht with the interval propagated (upper C -> lower D)."""
    out = {"D0": float(E / c["median"]) if c.get("median") and c["median"] > 0 else None}
    if c.get("ci68") and c["ci68"][0] > 0:
        out["D0_ci68"] = [float(E / c["ci68"][1]), float(E / c["ci68"][0])]
    if c.get("ci95") and c["ci95"][0] > 0:
        out["D0_ci95"] = [float(E / c["ci95"][1]), float(E / c["ci95"][0])]
    return out


def ols_loglog(A: np.ndarray, S: np.ndarray) -> dict[str, float]:
    """log10 A = log10 a - b log10 S by ordinary least squares with A dependent (M&D 1989 eq. 10)."""
    x, y = np.log10(S), np.log10(A)
    X = np.column_stack([np.ones_like(x), -x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"log10_a": float(coef[0]), "a": float(10 ** coef[0]), "b": float(coef[1]), "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "n": int(x.size)}


def ols_loglog_bootstrap(A: np.ndarray, S: np.ndarray, n_boot: int = N_BOOT_FITS, seed: int = SEED) -> dict[str, Any]:
    fit = ols_loglog(A, S)
    rng = np.random.default_rng(seed)
    n = A.size
    bs_a, bs_b = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(S[idx]).size < 2:
            bs_a[i], bs_b[i] = np.nan, np.nan
            continue
        f = ols_loglog(A[idx], S[idx])
        bs_a[i], bs_b[i] = f["a"], f["b"]
    fit["a_ci95"] = [float(np.nanpercentile(bs_a, 2.5)), float(np.nanpercentile(bs_a, 97.5))]
    fit["b_ci95"] = [float(np.nanpercentile(bs_b, 2.5)), float(np.nanpercentile(bs_b, 97.5))]
    fit["a_ci68"] = [float(np.nanpercentile(bs_a, 16)), float(np.nanpercentile(bs_a, 84))]
    fit["b_ci68"] = [float(np.nanpercentile(bs_b, 16)), float(np.nanpercentile(bs_b, 84))]
    fit["n_boot"] = n_boot
    fit["seed"] = seed
    return fit


def pct(v: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return {"n": int(v.size)} | ({f"p{q}": float(np.percentile(v, q)) for q in qs} if v.size else {})


# ----------------------------------------------------------------------------- hilltop set on a fixed network (decision 1)

def network_basins(Z, A, FD, dx, dy, A_c: float):
    """Channel network at A_c and the outlet-basin lookup of its pixels (the `measure` relabelling)."""
    from lemcvt.measure.chi import channel_network, channel_pixel_basin_lookup
    net = channel_network(Z, A, FD, A_c, dx, dy)
    return net, channel_pixel_basin_lookup(net)


def hilltop_set(Z, A, FD, dx, dy, A_c: float, w_f: int = W_F, road_mask: np.ndarray | None = None, segment_min_px: int = SEGMENT_MIN_PX,
                C_at_Lf: np.ndarray | None = None) -> dict[str, Any]:
    """The decision-1 hilltop set on the network A >= A_c, with every stage kept so that the filters' effects can be reported.

    Returns masks: ``divide_between_basins`` (before any drop), ``geom`` (edge margin, slope, L_f/2 distance: the Part B definition on this
    network), ``roadfree`` (geom minus the road buffer), ``segfilt`` (roadfree minus segments under ``segment_min_px``), ``counted``
    (segfilt minus C_ht(L_f) >= 0), ``unf`` (geom minus C >= 0: no segment or road filter), and the distance grid to the network.
    """
    from scipy.ndimage import distance_transform_edt, label

    from lemcvt.measure.hilltop_curvature import hilltop_mask
    net, lookup = network_basins(Z, A, FD, dx, dy, A_c)
    mask0, diag = hilltop_mask(Z, A, FD, A_c, dx, dy, edge_margin=EDGE_MARGIN, max_slope=SLOPE_MAX, basin_of_channel=lookup)
    channel = np.isfinite(A) & (A >= A_c)
    dist = distance_transform_edt(~channel, sampling=(dy, dx))
    L_f = (2 * w_f + 1) * dx
    geom = mask0 & (dist >= 0.5 * L_f)
    out: dict[str, Any] = {"A_c": float(A_c), "L_f_m": float(L_f), "n_channel_px": int(channel.sum()), "n_divide": int(diag["n_divide"]),
                           "n_divide_between_basins": int(diag["n_divide_between_basins"]), "n_removed_steep": int(diag.get("n_removed_steep", 0)),
                           "n_before_exclusion": int(mask0.sum()), "n_geom": int(geom.sum()), "n_basins": int(net.outlet_index.size), "n_edge_outlets": int(net.n_edge_outlets)}
    roadfree = geom.copy()
    if road_mask is not None:
        roadfree &= ~road_mask
    out["n_removed_road"] = int(geom.sum() - roadfree.sum())
    out["road_filter_applied"] = road_mask is not None
    seg, n_seg = label(roadfree, structure=np.ones((3, 3)))
    sizes = np.bincount(seg.ravel(), minlength=n_seg + 1)
    small = sizes < segment_min_px
    small[0] = False
    segfilt = roadfree & ~small[seg]
    out["n_segments_before"] = int(n_seg)
    out["n_segments_removed"] = int(small.sum())
    out["n_removed_segment"] = int(roadfree.sum() - segfilt.sum())
    if C_at_Lf is None:
        from lemcvt.measure.hilltop_curvature import total_curvature
        C_at_Lf = -total_curvature(Z, w_f, dx, dy)
    convex = np.isfinite(C_at_Lf) & (C_at_Lf > 0)
    counted = segfilt & convex
    unf = geom & convex
    out["n_removed_nonconvex"] = int(segfilt.sum() - counted.sum())
    out["n_counted"] = int(counted.sum())
    out["n_counted_without_convex_drop"] = int(segfilt.sum())
    out["n_unf"] = int(unf.sum())
    # labelled sensitivities of the segment filter (report only): the threshold, and the segment measured on the divide set before the
    # geometric drops (the crest's own length, Hurst et al. 2012's meaning) instead of on the surviving fragments
    alt: dict[str, Any] = {}
    seg0, n0 = label(mask0, structure=np.ones((3, 3)))
    sizes0 = np.bincount(seg0.ravel(), minlength=n0 + 1)
    for thr in (0, 10, 20, 50):
        small_f = sizes < thr
        small_f[0] = False
        m_f = roadfree & ~small_f[seg] & convex
        small_p = sizes0 < thr
        small_p[0] = False
        m_p = roadfree & ~small_p[seg0] & convex
        alt[f"fragment_{thr}"] = {"n": int(m_f.sum()), "C_ht_Lf": float(np.median(C_at_Lf[m_f])) if m_f.any() else None}
        alt[f"parent_{thr}"] = {"n": int(m_p.sum()), "C_ht_Lf": float(np.median(C_at_Lf[m_p])) if m_p.any() else None}
        if thr == 50:
            out["masks_parent50"] = m_p
    out["segment_filter_sensitivity"] = alt
    out["segment_sizes_fragments_px"] = {"median": float(np.median(sizes[1:])) if n_seg else None, "max": int(sizes[1:].max()) if n_seg else None, "n": int(n_seg)}
    out["segment_sizes_parent_px"] = {"median": float(np.median(sizes0[1:])) if n0 else None, "max": int(sizes0[1:].max()) if n0 else None, "n": int(n0)}
    masks = {"divide_between_basins": mask0, "geom": geom, "roadfree": roadfree, "segfilt": segfilt, "counted": counted, "unf": unf, "parent50": out.pop("masks_parent50")}
    return out | {"masks": masks, "dist": dist, "channel": channel, "net": net}


def segments_of(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label
    seg, _ = label(mask, structure=np.ones((3, 3)))
    return seg


def crest_width_stats(dist: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    d = dist[mask]
    return {str(q): float(2 * np.percentile(d, q)) for q in (5, 10, 25, 50, 75, 90, 95)} if d.size else None


def largest_window_not_above(width_m: float, half_widths=TEST_HALF_WIDTHS, dx: float = 1.0, w_f: int = W_F) -> int:
    """L_w: the largest tested window (nominal metres) not above the crest width, at least L_f."""
    cands = [w for w in half_widths if (2 * w + 1) <= width_m + 1e-9 and w >= w_f]
    return max(cands) if cands else w_f


# ----------------------------------------------------------------------------- D8 tracing to a target mask

def trace_to_mask(Z, FD, dx, dy, rows, cols, target: np.ndarray) -> dict[str, np.ndarray]:
    """D8 path from each (row, col) to the first cell of ``target``: path length L_H, drop R, the end cell, whether the path left the
    tile first (no receiver), and the Euclidean start-end distance. Same stepping as `er_star.trace_to_channel`."""
    from lemcvt.hydro.d8 import d8_receiver_arrays
    r_to, c_to, good = d8_receiver_arrays(FD)
    n = rows.size
    cur_r, cur_c = rows.astype(np.int64).copy(), cols.astype(np.int64).copy()
    L = np.zeros(n)
    reached = target[rows, cols].copy()
    left = np.zeros(n, dtype=bool)
    active = ~reached
    z0 = Z[rows, cols]
    diag_len = math.hypot(dx, dy)
    for _ in range(Z.shape[0] * Z.shape[1]):
        if not active.any():
            break
        ok = active & good[cur_r, cur_c]
        left |= active & ~ok
        active &= ok
        nr, nc = r_to[cur_r, cur_c], c_to[cur_r, cur_c]
        dr, dc = nr - cur_r, nc - cur_c
        step = np.where((dr != 0) & (dc != 0), diag_len, np.where(dr != 0, dy, dx))
        L[ok] += step[ok]
        cur_r[ok], cur_c[ok] = nr[ok], nc[ok]
        hit = ok & target[cur_r, cur_c]
        reached |= hit
        active &= ~hit
    R = z0 - Z[cur_r, cur_c]
    eucl = np.hypot((cur_c - cols) * dx, (cur_r - rows) * dy)
    return {"L_H": L, "R": R, "reached": reached, "left_tile": left, "end_row": cur_r, "end_col": cur_c, "L_euclid": eucl}


def channel_heads(channel: np.ndarray, FD) -> np.ndarray:
    """Channel cells with no channel donor (the most upstream channel cells)."""
    from lemcvt.hydro.d8 import d8_receiver_arrays
    r_to, c_to, good = d8_receiver_arrays(FD)
    ny, nx = channel.shape
    donors = np.zeros((ny, nx), dtype=np.int64)
    src = channel & good
    np.add.at(donors, (r_to[src], c_to[src]), 1)
    return channel & (donors == 0)


# ----------------------------------------------------------------------------- roads

def road_layers() -> dict[str, Any] | None:
    p = os.path.join(run_dir(), "roads", "roads.json")
    return load_json(p) if os.path.exists(p) else None


def road_mask_for_tile(td, buffer_m: float = ROAD_BUFFER_M) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Cells within ``buffer_m`` of a TIGER road/trail line, on the tile lattice (lines reprojected to the tile CRS)."""
    from osgeo import ogr, osr
    from scipy.ndimage import distance_transform_edt
    rec = road_layers()
    site = td.cfg.site
    if rec is None or not rec["sites"].get(site, {}).get("files"):
        return None, {"applied": False, "reason": "no road layer for the site"}
    ogr.UseExceptions()
    osr.UseExceptions()
    tr = td.filled.transform
    ny, nx = td.Z.shape
    x0, y1 = tr.c, tr.f
    dx, dy = td.dx, td.filled.dy
    x1, y0 = x0 + nx * dx, y1 - ny * dy
    pad = buffer_m + 2 * max(dx, dy)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(int(td.cfg.raw["grids"].get("crs_epsg") or td.site.crs_epsg))
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    hit = np.zeros((ny, nx), dtype=bool)
    n_lines, n_cells = 0, 0
    mtfcc: dict[str, int] = {}
    for f in rec["sites"][site]["files"]:
        ds = ogr.Open(f["path"])
        lyr = ds.GetLayer(0)
        src = lyr.GetSpatialRef()
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(src, dst)
        inv = osr.CoordinateTransformation(dst, src)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for X, Y in ((x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad), (x0 - pad, y1 + pad), (x0 - pad, y0 - pad)):
            ring.AddPoint(X, Y)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        poly.Transform(inv)
        lyr.SetSpatialFilter(poly)
        for feat in lyr:
            g = feat.GetGeometryRef().Clone()
            g.Transform(ct)
            g.Segmentize(0.5 * min(dx, dy))
            code = feat.GetField("MTFCC") if feat.GetFieldIndex("MTFCC") >= 0 else "?"
            lines = [g] if g.GetGeometryType() in (ogr.wkbLineString, ogr.wkbLineString25D) else [g.GetGeometryRef(i) for i in range(g.GetGeometryCount())]
            any_in = False
            for ln in lines:
                pts = np.array(ln.GetPoints())
                if pts.size == 0:
                    continue
                c = np.floor((pts[:, 0] - x0) / dx).astype(int)
                r = np.floor((y1 - pts[:, 1]) / dy).astype(int)
                ok = (c >= 0) & (c < nx) & (r >= 0) & (r < ny)
                if ok.any():
                    hit[r[ok], c[ok]] = True
                    any_in = True
            if any_in:
                n_lines += 1
                mtfcc[code] = mtfcc.get(code, 0) + 1
        lyr.SetSpatialFilter(None)
    if not hit.any():
        return np.zeros((ny, nx), dtype=bool), {"applied": True, "n_lines_in_tile": 0, "n_cells_within_buffer": 0, "buffer_m": buffer_m, "mtfcc": mtfcc}
    dist = distance_transform_edt(~hit, sampling=(dy, dx))
    mask = dist <= buffer_m
    n_cells = int(mask.sum())
    return mask, {"applied": True, "n_lines_in_tile": n_lines, "n_cells_within_buffer": n_cells, "buffer_m": buffer_m, "mtfcc": mtfcc}


# ----------------------------------------------------------------------------- tiles

def metre_tile(tile: str):
    from lemcvt.measure.pipeline import _metre_tile
    return _metre_tile(tile)


def tile_crs_epsg(cfg) -> int:
    return int(cfg.raw["grids"].get("crs_epsg") or cfg.site_crs_epsg)


def tile_bounds(tile: str) -> dict[str, float]:
    import rasterio

    from lemcvt.config import data_root, load_tile_config
    cfg = load_tile_config(tile)
    with rasterio.open(data_root() / cfg.grids["filled"]) as ds:
        b = ds.bounds
        epsg = int(cfg.raw["grids"].get("crs_epsg") or (ds.crs.to_epsg() if ds.crs else 0))
        return {"left": b.left, "bottom": b.bottom, "right": b.right, "top": b.top, "crs_epsg": epsg,
                "width": ds.width, "height": ds.height, "transform": list(ds.transform)[:6]}


def spec_groups_of(tile: str, specs: dict[str, dict]) -> list[str]:
    return [k for k, s in specs.items() if s["tile"] == tile]
