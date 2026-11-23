#!/usr/bin/env python3
"""Addendum 21 (the Tennessee Valley channel-head test): shared constants and helpers for the a21_* scripts.

Directive `reports/directives/2026-11-23T1030PST_v2_addendum_21_ttv_channel_head_test.md`. Decisions of its section 1 as constants,
rules of its section 6 as thresholds, all fixed before any result. No package code is changed: the mask route of decision 1.3 is
`PartitionedProfiler(channel_stat=mask)` with the threshold 0.5 (the class already takes a per-cell statistic in place of the area),
checked against the area route at A_crit = 7,400 m2 on TTV2 (`a21_tile.py`).

Choices made in applying the directive (every one is listed in the report's state check):
  - The observed criterion network (decision 1.3) is built on the full-site 12 m lattice (block means of the stage B 1 m grid from its
    origin, 529495 / 4198505), so areas at tile edges are full-grid areas; the tiles are offset 1-9 m from that lattice. Decision 1.4
    (head extraction from every surface) uses each surface's own lattice (the tile origin, 83 x 83 full blocks; the 4 m remainder is
    dropped), so the observed lidar and the model surfaces are treated identically. H0 is decided on the full-site network; the
    tile-lattice extraction of the observed surface (the H1 baseline) is reported beside it.
  - A 12 m head cell is seeded at the 1 m cell of largest full-grid D8 area in its block and traced downstream on the full-grid D8;
    for the per-tile model the trace is reduced to its first cell inside the tile (the entry cell) and completed on the tile's own D8
    (the tile grids are cut from the same full-grid routing). Candidate 12 m cells of a tile are those whose 12 m D8 path enters the
    tile box with a 36 m margin (3 cells); their 1 m paths are followed for at most 60,000 steps.
  - S_12 is measured on the pit-filled 12 m surface (the surface that was routed); the two-cell sensitivity uses the largest-area
    donor of the largest-area donor.
  - New free fits (PF-A, PF-C, SJ-C) use the addendum 18 box in (U/K, U/D), i.e. the configured K and D bounds times 65/77
    (decision 1.1, "bounds scale the same way"), so that they are exactly comparable with the rescaled P4 and AD.
  - The Perron et al. (2012) valley-head sensitivity follows the paper's Methods as the directive condenses it (sigma = 10 m Gaussian;
    contour curvature above the mean + 2 sd; 8-connected clusters of at least 10 cells; flow paths from each cluster's largest-area
    cell to the tile edge; closing with a 10 m disk; thinning; spurs under 10 m pruned; heads = skeleton end points off the tile
    edge). The paper's extra condition that a cluster hold a high-area cell is not in the directive and is not applied.
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
os.environ.setdefault("A19_RUN", "20261120T160830Z_a19_hilltops_heads_slope")
os.environ.setdefault("A20_RUN", "20261120T185052Z_a20_perron_D0_heads_partB")
from a19_fixed import fits_from_reduction  # noqa: E402,F401
from a19_hilltops import md_polygon_mask  # noqa: E402,F401
from a20_common import (  # noqa: E402,F401
    A18,
    A18_SUMMARY,
    HEADS_CSV,
    HEADS_SHA256,
    MD_EQ10,
    MD_GPKG,
    MD_TABLE1_MEDIAN_M2,
    MD_TILES,
    RESULTS_ROOT,
    SCRATCH,
    TTV_FULL,
    TTV_PRIMARY,
    a18_specs,
    a20_run_dir,  # noqa: E402
    channel_heads,
    dump,
    hillshade,
    load_json,
    md_table1,
    metre_tile,
    ols_loglog_bootstrap,
    pct,
    segment_bootstrap_median,
    sha256_file,
    slurm_record,
    tile_bounds,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DIRECTIVE = "reports/directives/2026-11-23T1030PST_v2_addendum_21_ttv_channel_head_test.md"
A20_DIR = a20_run_dir()
A20_REPORT = "reports/v2/20261121T0630PST_perron_D0_heads_partB.md"
FIG_DIR = os.path.join(REPO, "reports", "v2", "figures", "a21")
JOB_DIR = os.path.join(SCRATCH, "lemcvt_jobs", "a21")
FULL_ORIGIN = (529495.0, 4198505.0)          # x0, y1 of the stage B full-site grid (EPSG:6339)

# ----------------------------------------------------------------------------- decision 1.1 (parameter values)
U_NEW = 6.5e-5                      # m/yr, Heimsath et al. 1999 Table 1, sub-basin 1 stream sediment (64 +- 12, 66 +- 13 m/Myr)
U_OLD = 7.7e-5                      # the configured value of addenda 17-20 (Heimsath's P0)
U_SENS = {"sub_basin_2": 1.02e-4, "legacy_P0": 7.7e-5}
RESCALE = U_NEW / U_OLD             # 65/77: K and D of a fit at U_OLD convert to U_NEW by this factor; MSE unchanged
E_RIDGE = 2.7e-5                    # median, soil-covered greywacke noses (Table 1)
E_RIDGE_MEAN = 3.5e-5
M_SITE = 0.33
M_SENS = 0.30
D_PHYS = 0.0025                     # rock-equivalent, rho_s/rho_r = 0.5 x 50 cm2/yr
D_PHYS_MEASURED_RHO = 50e-4 * 1.4 / 2.2
K_CAL = 2.4e-5                      # m^0.34/yr at m = 0.33 (A20 section 12.4)
A_F_TABLE1 = MD_TABLE1_MEDIAN_M2    # 7,400 m2
RESCALE_TOL = 1e-8                  # decision 1.1: the direct evaluation at the new U
MASK_TOL = 1e-12                    # decision 1.3: the mask route against the area route (the same channel set, so bit-exact)

# ----------------------------------------------------------------------------- decision 1.2 (map-scale heads)
BLOCK_M = 12
BLOCK_SENS_M = 10
ALPHA_PRIMARY = 2.0
ALPHA_SENS = 1.65
SEED = 20261123
N_BOOT_FITS = 10000
N_BOOT_SEG = 2000
PICK_EXCLUDED_SENS = 49
MARGIN_CELLS = 3                    # decision 1.3 candidates: the tile box plus 36 m
MAX_TRACE_STEPS = 60000

# ----------------------------------------------------------------------------- decision 1.4 (extraction) and 1.5 (runs)
LINE_TOL_M = 15.0
PERRON_SIGMA_M = 10.0
PERRON_K_SD = 2.0
PERRON_MIN_CELLS = 10
PERRON_CLOSE_R_M = 10.0
PERRON_PRUNE_M = 10.0
SJC_THIN = 400
SJC_REFINE_MAX = 400
SJC_A_RANGE = (1e3, 1e5)
HOLDOUT_A_C = 1e3
FIXED_D = ("phys", "form", "relief")
MODEL_ORDER = ["obs", "F_phys", "F_form", "F_relief", "F2_phys", "F2_form", "F2_relief", "PF_A", "PF_C", "SJ_C", "P4", "AD"]

# ----------------------------------------------------------------------------- rules (section 6)
HOLD_FRAC = 0.80
H0_DIST_M = 25.0
H0_RATIO = (0.67, 1.5)
H0B_RATIO = (0.5, 2.0)
K1_RANGE = (0.5, 2.0)
DX_RANGE = (0.67, 1.5)
SITE_MIN = 9                        # of 11 tiles (K1, DX, M1)
F0_PREDICTIONS = {"F_relief_fits_best": "F-relief fits best of the three", "F_phys_fits_worst": "F-phys fits worst, with crests too sharp"}
CREST_CURV_MEASURED = 0.016         # m^-1, the planner's figure for the measured crest curvature (F0 prediction)


def hold_n(n: int) -> int:
    return int(math.ceil(HOLD_FRAC * n - 1e-9)) if n > 0 else 0


# ----------------------------------------------------------------------------- run directory

def a21_run_dir() -> str:
    rid = os.environ.get("A21_RUN")
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a21_ttv_channel_head_test"
        os.environ["A21_RUN"] = rid
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def out_path(*parts: str) -> str:
    p = os.path.join(a21_run_dir(), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def a20_path(*parts: str) -> str:
    return os.path.join(A20_DIR, *parts)


def hms(h: float) -> str:
    s = int(round(h * 3600))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def picks() -> list[dict]:
    assert sha256_file(HEADS_CSV) == HEADS_SHA256, "heads CSV checksum differs from the copy of record"
    with open(HEADS_CSV) as f:
        rows = list(csv.DictReader(f))
    assert [r["pick_id"] for r in rows] == [str(i) for i in range(1, 65)]
    return [{"pick_id": int(r["pick_id"]), "x": float(r["x_epsg6339"]), "y": float(r["y_epsg6339"])} for r in rows]


# ----------------------------------------------------------------------------- D8 on any grid

CODES = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
DROW = np.zeros(256, dtype=np.int64)
DCOL = np.zeros(256, dtype=np.int64)
for _code, (_dr, _dc) in CODES.items():
    DROW[_code], DCOL[_code] = _dr, _dc


def block_mean(Z: np.ndarray, b: int) -> np.ndarray:
    """Mean of each full b x b block (NaN-aware); the remainder rows and columns are dropped."""
    if b == 1:
        return np.asarray(Z, dtype=float).copy()
    ny, nx = (Z.shape[0] // b) * b, (Z.shape[1] // b) * b
    with np.errstate(invalid="ignore"), _quiet():
        return np.nanmean(Z[:ny, :nx].reshape(ny // b, b, nx // b, b), axis=(1, 3))


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def coarse_grid(Z1: np.ndarray, b: int, dx: float = 1.0) -> dict[str, Any]:
    """Block-mean a 1 m surface to b m, pit-fill it and route D8 (`border='outlets'`): Zb, Zf, FD, A (m2), valid."""
    from lemcvt.hydro.d8 import d8_area, d8_flow_direction
    from lemcvt.hydro.fill import priority_flood_fill
    Zb = block_mean(Z1, b)
    valid = np.isfinite(Zb)
    Zf = priority_flood_fill(np.where(valid, Zb, np.nan), float(b) * dx)
    FD = d8_flow_direction(Zf, border="outlets")
    A = d8_area(FD, Zf, float(b) * dx)
    A[~valid] = np.nan
    return {"Zb": Zb, "Zf": Zf, "FD": FD, "A": A, "valid": valid, "b": b, "dx": float(b) * dx}


def receiver_flat(FD: np.ndarray) -> np.ndarray:
    """Flat index of every cell's receiver (-1 for none)."""
    ny, nx = FD.shape
    rows, cols = np.indices((ny, nx))
    nr, nc = rows + DROW[FD], cols + DCOL[FD]
    good = (FD != 0) & (nr >= 0) & (nr < ny) & (nc >= 0) & (nc < nx)
    return np.where(good, nr * nx + nc, -1).ravel()


def max_area_donor(A: np.ndarray, FD: np.ndarray) -> np.ndarray:
    """Flat index of the largest-area donor of every cell (-1 where none)."""
    ny, nx = A.shape
    recv = receiver_flat(FD)
    af = np.asarray(A, dtype=float).ravel()
    src = np.nonzero((recv >= 0) & np.isfinite(af))[0]
    order = np.argsort(af[src], kind="stable")                # ascending area: the last write per receiver is the largest donor
    best = np.full(ny * nx, -1, dtype=np.int64)
    best[recv[src[order]]] = src[order]
    return best


def donor_slope(Zf: np.ndarray, A: np.ndarray, FD: np.ndarray, cell_m: float, n_cells: int = 1) -> dict[str, np.ndarray]:
    """Slope from each cell up its largest-area donor chain over ``n_cells`` cells: (z_up - z) / centre-to-centre path length.
    Cells with no donor have NaN; ``short`` marks chains shorter than ``n_cells``."""
    ny, nx = Zf.shape
    best = max_area_donor(A, FD)
    z = np.asarray(Zf, dtype=float).ravel()
    start = np.arange(ny * nx)
    cur = start.copy()
    L = np.zeros(ny * nx)
    ok = np.ones(ny * nx, dtype=bool)
    steps = np.zeros(ny * nx, dtype=np.int64)
    for _ in range(n_cells):
        d = best[cur]
        has = ok & (d >= 0)
        dr = d[has] // nx - cur[has] // nx
        dc = d[has] % nx - cur[has] % nx
        L[has] += cell_m * np.hypot(dr, dc)
        cur[has] = d[has]
        steps[has] += 1
        ok = has
    used = steps > 0
    S = np.full(ny * nx, np.nan)
    S[used] = (z[cur[used]] - z[start[used]]) / L[used]
    return {"S": S.reshape(ny, nx), "L": L.reshape(ny, nx), "short": (used & (steps < n_cells)).reshape(ny, nx), "n_steps": steps.reshape(ny, nx),
            "up_flat": np.where(used, cur, -1).reshape(ny, nx)}


def head_statistic_coarse(A: np.ndarray, S: np.ndarray, FD: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Decision 1.3 on a coarse grid: T = A S^alpha (NaN where the cell has no donor: divides never start a channel), M = the upstream
    maximum of T along D8, Mdon = the largest M among the cell's donors (-inf where none). Channel at C: M >= C; head at C: M >= C
    and Mdon < C."""
    from lemcvt.analyses.slope_threshold import upstream_max
    ok = np.isfinite(A) & np.isfinite(S)
    Spos = np.where(ok & (S > 0), S, 0.0)
    T = np.where(ok, A * (Spos ** alpha if alpha != 0 else 1.0), np.nan)
    M = upstream_max(T, A, FD)
    ny, nx = A.shape
    recv = receiver_flat(FD)
    Mf = M.ravel()
    src = np.nonzero((recv >= 0) & np.isfinite(Mf))[0]
    Mdon = np.full(ny * nx, -np.inf)
    np.maximum.at(Mdon, recv[src], Mf[src])
    return {"T": T, "M": M, "Mdon": Mdon.reshape(ny, nx)}


def criterion_sets(M: np.ndarray, Mdon: np.ndarray, C: float) -> tuple[np.ndarray, np.ndarray]:
    channel = np.isfinite(M) & (M >= C)
    heads = channel & ~(Mdon >= C)
    return channel, heads


def channel_count_C(M: np.ndarray, A: np.ndarray, a_target: float) -> float:
    """The C at which the channel set M >= C has as many cells as A >= a_target (on the same grid)."""
    m = np.sort(M[np.isfinite(M)])[::-1]
    n = int(np.sum(A[np.isfinite(A)] >= a_target))
    n = min(max(n, 1), m.size)
    return float(m[n - 1])


def trace_paths(FD: np.ndarray, rows: np.ndarray, cols: np.ndarray, target: np.ndarray | None = None, stop: np.ndarray | None = None,
                max_steps: int = MAX_TRACE_STEPS, target_box: tuple[int, int, int, int] | None = None) -> dict[str, np.ndarray]:
    """Follow D8 from each (row, col) until a cell of ``target`` (a mask, or the box ``(r0, r1, c0, c1)``) is reached (its position is
    returned), the path stops (code 0, grid edge) or a cell of ``stop`` is entered. Vectorised over paths; the full-grid FD is read
    in place (no receiver arrays)."""
    ny, nx = FD.shape
    n = rows.size
    if target_box is not None:
        tb = target_box

        class _Box:
            def __getitem__(self, rc):
                r, c = rc
                return (r >= tb[0]) & (r < tb[1]) & (c >= tb[2]) & (c < tb[3])
        target = _Box()  # type: ignore[assignment]
    cur_r, cur_c = rows.astype(np.int64).copy(), cols.astype(np.int64).copy()
    end_r = np.full(n, -1, dtype=np.int64)
    end_c = np.full(n, -1, dtype=np.int64)
    steps = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    stopped = np.zeros(n, dtype=bool)
    if target is not None:
        hit = target[cur_r, cur_c]
        end_r[hit], end_c[hit] = cur_r[hit], cur_c[hit]
        active &= ~hit
    for _ in range(max_steps):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        code = FD[cur_r[idx], cur_c[idx]]
        nr, nc = cur_r[idx] + DROW[code], cur_c[idx] + DCOL[code]
        ok = (code != 0) & (nr >= 0) & (nr < ny) & (nc >= 0) & (nc < nx)
        active[idx[~ok]] = False
        j = idx[ok]
        cur_r[j], cur_c[j] = nr[ok], nc[ok]
        steps[j] += 1
        if target is not None:
            hit = target[cur_r[j], cur_c[j]]
            end_r[j[hit]], end_c[j[hit]] = cur_r[j[hit]], cur_c[j[hit]]
            active[j[hit]] = False
            j = j[~hit]
        if stop is not None and j.size:
            st = stop[cur_r[j], cur_c[j]]
            stopped[j[st]] = True
            active[j[st]] = False
    return {"end_row": end_r, "end_col": end_c, "reached": end_r >= 0, "steps": steps, "unfinished": active, "stopped": stopped, "last_row": cur_r, "last_col": cur_c}


def block_view(arr: np.ndarray, b: int) -> np.ndarray:
    """A strided view [R, C, i, j] of the full b x b blocks of ``arr`` (no copy)."""
    from numpy.lib.stride_tricks import as_strided
    nb_r, nb_c = arr.shape[0] // b, arr.shape[1] // b
    s0, s1 = arr.strides
    return as_strided(arr, shape=(nb_r, nb_c, b, b), strides=(b * s0, b * s1, s0, s1), writeable=False)


def drains_to(FD: np.ndarray, box: np.ndarray, max_iter: int = 20000) -> tuple[np.ndarray, int]:
    """Cells whose D8 path enters ``box`` (fixed point of flag |= flag[receiver])."""
    recv = receiver_flat(FD)
    flag = box.ravel().copy()
    has = recv >= 0
    r = np.where(has, recv, 0)
    for it in range(max_iter):
        new = flag | (has & flag[r])
        if np.array_equal(new, flag):
            return new.reshape(FD.shape), it
        flag = new
    return flag.reshape(FD.shape), max_iter


def mask_from_entries(entry_rows: np.ndarray, entry_cols: np.ndarray, A: np.ndarray, FD: np.ndarray) -> np.ndarray:
    """The 1 m channel set of a tile: every valid cell at or downstream of an entry cell (downstream propagation along D8)."""
    from lemcvt.analyses.slope_threshold import upstream_max
    T = np.where(np.isfinite(A), 0.0, np.nan)
    if entry_rows.size:
        T[entry_rows, entry_cols] = 1.0
    M = upstream_max(T, A, FD)
    return np.isfinite(M) & (M >= 0.5)


# ----------------------------------------------------------------------------- the picks on a coarse grid

def pick_cells(xs: np.ndarray, ys: np.ndarray, x0: float, y1: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    return np.floor((y1 - ys) / b).astype(int), np.floor((xs - x0) / b).astype(int)


def pick_area_cell(A: np.ndarray, r: int, c: int) -> tuple[int, int, float]:
    """The largest-area cell of the 3 x 3 block around (r, c) and its area (decision 1.2)."""
    r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, A.shape[0]), max(c - 1, 0), min(c + 2, A.shape[1])
    sub = A[r0:r1, c0:c1]
    if not np.isfinite(sub).any():
        return r, c, float("nan")
    k = int(np.nanargmax(sub))
    rr, cc = r0 + k // sub.shape[1], c0 + k % sub.shape[1]
    return int(rr), int(cc), float(A[rr, cc])


# ----------------------------------------------------------------------------- measures against the picks and the M&D lines (decision 1.4)

def md_lines_mask(td) -> np.ndarray:
    """The M&D (1989) traced channel lines rasterised on the tile's 1 m lattice (ALL_TOUCHED), inside the study polygon."""
    from osgeo import gdal, ogr
    gdal.UseExceptions()
    ogr.UseExceptions()
    ny, nx = td.Z.shape
    tr = td.filled.transform
    ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((tr.c, tr.a, tr.b, tr.f, tr.d, tr.e))
    ds.SetProjection(td.filled.crs.to_wkt())
    src = ogr.Open(MD_GPKG)
    gdal.RasterizeLayer(ds, [1], src.GetLayerByName("channels"), burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    lines = ds.GetRasterBand(1).ReadAsArray().astype(bool)
    return lines & md_polygon_mask(td)


def coverage_1m(mask_b: np.ndarray, b: int, shape: tuple[int, int], r_off: int = 0, c_off: int = 0) -> np.ndarray:
    """The 1 m cells covered by the b m cells of ``mask_b`` (lattice origin at (r_off, c_off) of the 1 m grid)."""
    out = np.zeros(shape, dtype=bool)
    big = np.repeat(np.repeat(mask_b, b, axis=0), b, axis=1)
    r0, c0 = max(r_off, 0), max(c_off, 0)
    br0, bc0 = r0 - r_off, c0 - c_off
    h = min(big.shape[0] - br0, shape[0] - r0)
    w = min(big.shape[1] - bc0, shape[1] - c0)
    if h > 0 and w > 0:
        out[r0:r0 + h, c0:c0 + w] = big[br0:br0 + h, bc0:bc0 + w]
    return out


def head_measures(channel_cov: np.ndarray, head_rc_1m: tuple[np.ndarray, np.ndarray], pick_rc: tuple[np.ndarray, np.ndarray], pick_ids: list[int],
                  poly: np.ndarray, lines: np.ndarray, dx: float = 1.0) -> dict[str, Any]:
    """Distance from each pick to the nearest extracted channel cell (1 m coverage), the ratio of extracted heads to picks inside the
    polygon, and the traced-line overlap both ways (cells within LINE_TOL_M)."""
    from scipy.ndimage import distance_transform_edt
    pr, pc = pick_rc
    hr, hc = head_rc_1m
    rec: dict[str, Any] = {"n_channel_1m_cells": int(channel_cov.sum()), "n_picks": int(pr.size), "n_picks_in_polygon": int(poly[pr, pc].sum()) if pr.size else 0}
    if channel_cov.any():
        dist = distance_transform_edt(~channel_cov, sampling=(dx, dx))
    else:
        dist = np.full(channel_cov.shape, np.inf)
    d = dist[pr, pc] if pr.size else np.array([])
    rec["pick_dist_m"] = {"median": float(np.median(d)) if d.size else None, "q25": float(np.percentile(d, 25)) if d.size else None,
                          "q75": float(np.percentile(d, 75)) if d.size else None, "frac_within_25m": float(np.mean(d <= H0_DIST_M)) if d.size else None,
                          "frac_inside_channel_cell": float(np.mean(d == 0)) if d.size else None, "values": dict(zip(map(str, pick_ids), d.tolist(), strict=True))}
    n_heads_poly = int(poly[hr, hc].sum()) if hr.size else 0
    rec["n_heads_tile"] = int(hr.size)
    rec["n_heads_in_polygon"] = n_heads_poly
    rec["ratio_heads_to_picks"] = (n_heads_poly / rec["n_picks_in_polygon"]) if rec["n_picks_in_polygon"] else None
    if lines.any():
        rec["line_cells"] = int(lines.sum())
        rec["frac_line_within_15m_of_channel"] = float(np.mean(dist[lines] <= LINE_TOL_M))
        dl = distance_transform_edt(~lines, sampling=(dx, dx))
        chp = channel_cov & poly
        rec["frac_channel_within_15m_of_line"] = float(np.mean(dl[chp] <= LINE_TOL_M)) if chp.any() else None
        rec["n_channel_cells_in_polygon"] = int(chp.sum())
    return rec


def h0_reading(med: float | None, ratio: float | None) -> dict[str, Any]:
    a = med is not None and med <= H0_DIST_M
    b = ratio is not None and H0_RATIO[0] <= ratio <= H0_RATIO[1]
    if a and b:
        txt = "locates the mapped heads"
    elif ratio is not None and ratio < H0_RATIO[0]:
        txt = "too few heads" + ("" if a else ", distances large")
    elif ratio is not None and ratio > H0_RATIO[1]:
        txt = "too many heads" + ("" if a else ", distances large")
    else:
        txt = "count in range, distances large" if not a else "undetermined"
    return {"median_dist_ok": bool(a), "ratio_ok": bool(b), "holds": bool(a and b), "reading": txt}


# ----------------------------------------------------------------------------- the extraction of decision 1.4 on any surface

def extract_coarse(Z1: np.ndarray, dx: float, field: dict[str, float], b: int = BLOCK_M, alphas=(0.0, ALPHA_PRIMARY)) -> dict[str, Any]:
    """Block-mean a surface to b m, route it, and apply the criterion with its own A and S (alpha = 0 at A_F, alpha = 2 at C_F)."""
    g = coarse_grid(Z1, b, dx)
    sl = donor_slope(g["Zf"], g["A"], g["FD"], g["dx"], 1)
    out: dict[str, Any] = {"grid": g, "S": sl["S"], "sets": {}}
    for al in alphas:
        hs = head_statistic_coarse(g["A"], sl["S"], g["FD"], al)
        C = field["A_F"] if al == 0 else field["C_F"]
        ch, hd = criterion_sets(hs["M"], hs["Mdon"], C)
        out["sets"][f"a{al:g}"] = {"alpha": al, "C": C, "channel": ch, "heads": hd, "M": hs["M"], "Mdon": hs["Mdon"]}
    return out


def extract_and_measure(Zs: np.ndarray, dx: float, field: dict[str, float], pick_rc, pick_ids, poly: np.ndarray, lines: np.ndarray,
                        A1: np.ndarray | None = None, FD1: np.ndarray | None = None, perron: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Decision 1.4 on one surface: the 12 m criterion sets (alpha = 0 at A_F, alpha = 2 at C_F) with the measures against the picks
    and the M&D lines, and the Perron (2012) valley-head sensitivity at 1 m. Returns the record and the arrays to store."""
    t1 = time.perf_counter()
    ex = extract_coarse(Zs, dx, field, BLOCK_M)
    rec: dict[str, Any] = {"n_valid_12m": int(ex["grid"]["valid"].sum())}
    arrays: dict[str, np.ndarray] = {"Z12": ex["grid"]["Zb"].astype(np.float32)}
    for key, st in ex["sets"].items():
        cov = coverage_1m(st["channel"], BLOCK_M, Zs.shape)
        hr, hc = np.nonzero(st["heads"])
        hm = head_measures(cov, (hr * BLOCK_M + BLOCK_M // 2, hc * BLOCK_M + BLOCK_M // 2), pick_rc, pick_ids, poly, lines, dx)
        hm.update({"alpha": st["alpha"], "C": st["C"], "n_channel_12m": int(st["channel"].sum()), "n_heads_12m": int(hr.size)})
        hm["H"] = h0_reading(hm["pick_dist_m"]["median"], hm["ratio_heads_to_picks"]) if pick_rc[0].size else None
        rec[key] = hm
        arrays[f"{key}_channel"] = st["channel"]
        arrays[f"{key}_heads"] = st["heads"]
    if perron:
        t2 = time.perf_counter()
        pv = perron_valley_heads(Zs, dx, A1, FD1)
        hm = head_measures(pv["skeleton"], (pv["head_rows"], pv["head_cols"]), pick_rc, pick_ids, poly, lines, dx)
        hm.update({k: pv[k] for k in ("kc_threshold", "n_clusters", "n_cluster_cells", "n_heads", "n_skeleton_cells")})
        hm["H"] = h0_reading(hm["pick_dist_m"]["median"], hm["ratio_heads_to_picks"]) if pick_rc[0].size else None
        hm["wall_s"] = time.perf_counter() - t2
        rec["perron"] = hm
        arrays["perron_skeleton"] = np.packbits(pv["skeleton"])
        arrays["perron_heads_rc"] = np.stack([pv["head_rows"], pv["head_cols"]]).astype(np.int32)
    rec["wall_s"] = time.perf_counter() - t1
    return rec, arrays


def picks_on_tile(td) -> tuple[tuple[np.ndarray, np.ndarray], list[int]]:
    """The picks inside the tile as (rows, cols) on its 1 m lattice, with their ids."""
    pk = picks()
    tr = td.filled.transform
    xs, ys = np.array([q["x"] for q in pk]), np.array([q["y"] for q in pk])
    c = np.floor((xs - tr.c) / tr.a).astype(int)
    r = np.floor((ys - tr.f) / tr.e).astype(int)
    ny, nx = td.Z.shape
    on = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
    return (r[on], c[on]), [q["pick_id"] for q, o in zip(pk, on, strict=True) if o]


# ----------------------------------------------------------------------------- Perron et al. (2012) valley heads (sensitivity)

def _fill_nan_nearest(Z: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    nan = ~np.isfinite(Z)
    if not nan.any():
        return Z
    idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
    return Z[tuple(idx)]


def contour_curvature(Z: np.ndarray, dx: float, sigma_m: float = PERRON_SIGMA_M) -> np.ndarray:
    """Perron et al. (2012) eq. (1) on the Gaussian-smoothed surface; positive = convergent (valley)."""
    from scipy.ndimage import gaussian_filter
    Zs = gaussian_filter(_fill_nan_nearest(np.asarray(Z, dtype=float)), sigma_m / dx, mode="nearest")
    zy, zx = np.gradient(Zs, dx)
    zxy, zxx = np.gradient(zx, dx)
    zyy, _ = np.gradient(zy, dx)
    g2 = zx ** 2 + zy ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        kc = (zy ** 2 * zxx - 2 * zx * zy * zxy + zx ** 2 * zyy) / g2 ** 1.5
    kc[~np.isfinite(Z)] = np.nan
    return kc


def _neighbours8(m: np.ndarray) -> np.ndarray:
    p = np.pad(m.astype(np.int8), 1)
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, :-2] + p[1:-1, 2:] + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])


def thin_zhang_suen(m: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Zhang-Suen thinning (vectorised) to a one-pixel-wide skeleton."""
    img = m.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2, P3, P4, P5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            P6, P7, P8, P9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            Aa = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1, c2 = P2 * P4 * P6, P4 * P6 * P8
            else:
                c1, c2 = P2 * P4 * P8, P2 * P6 * P8
            rem = (img == 1) & (B >= 2) & (B <= 6) & (Aa == 1) & (c1 == 0) & (c2 == 0)
            if rem.any():
                img[rem] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def crossing_number(m: np.ndarray) -> np.ndarray:
    """Number of 0 -> 1 transitions around each cell's 8-neighbour ring (1 at an end point, 2 along a line, >= 3 at a junction)."""
    p = np.pad(m.astype(np.uint8), 1)
    ring = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
    ring.append(ring[0])
    return sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))


def skeleton_endpoints(sk: np.ndarray) -> np.ndarray:
    return sk & (_neighbours8(sk) == 1)


def prune_spurs(sk: np.ndarray, min_len: int, rounds: int = 5) -> np.ndarray:
    """Remove end branches shorter than ``min_len`` cells: walk from each end point (one skeleton neighbour) along the line, stepping to
    unvisited skeleton neighbours, to the first junction (crossing number >= 3, kept) or to the other end of an isolated piece (removed
    whole). Staircase corners of a one-pixel line have crossing number 2 and are not junctions."""
    sk = sk.copy()
    ny, nx = sk.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for _ in range(rounds):
        X = crossing_number(sk)
        junction = sk & (X >= 3)
        ends = np.argwhere(skeleton_endpoints(sk))
        removed = False
        for r, c in ends:
            start = (int(r), int(c))
            if not sk[start]:
                continue
            path: list[tuple[int, int]] = []
            visited = {start}
            cur = start
            keep_branch = False
            while True:
                if cur != start and junction[cur]:
                    break
                path.append(cur)
                if len(path) >= min_len:
                    keep_branch = True
                    break
                nbrs = [(cur[0] + dr, cur[1] + dc) for dr, dc in offs
                        if 0 <= cur[0] + dr < ny and 0 <= cur[1] + dc < nx and sk[cur[0] + dr, cur[1] + dc] and (cur[0] + dr, cur[1] + dc) not in visited]
                if not nbrs:
                    break
                jn = [q for q in nbrs if junction[q]]
                cur = jn[0] if jn else nbrs[0]
                visited.update(nbrs)
            if not keep_branch:
                for q in path:
                    sk[q] = False
                removed = True
        if not removed:
            break
    return sk


def perron_valley_heads(Z1: np.ndarray, dx: float, A1: np.ndarray | None = None, FD1: np.ndarray | None = None) -> dict[str, Any]:
    """The directive's condensation of Perron et al. (2012): clusters of contour curvature > mean + 2 sd, >= 10 cells, joined by flow
    paths from each cluster's largest-area cell, closed (10 m disk), thinned, pruned (< 10 m); heads = end points off the tile edge."""
    from scipy.ndimage import binary_closing, label
    kc = contour_curvature(Z1, dx)
    v = kc[np.isfinite(kc)]
    thr = float(v.mean() + PERRON_K_SD * v.std())
    valley = np.isfinite(kc) & (kc > thr)
    lab, n = label(valley, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= PERRON_MIN_CELLS
    keep[0] = False
    clusters = keep[lab]
    if A1 is None or FD1 is None:
        g = coarse_grid(Z1, 1, dx)
        A1, FD1 = g["A"], g["FD"]
    # flow path from each cluster's largest-area cell
    ids = np.nonzero(keep)[0]
    rows, cols = [], []
    Af = np.where(np.isfinite(A1), A1, -1.0)
    for k in ids:
        rr, cc = np.nonzero(lab == k)
        j = int(np.argmax(Af[rr, cc]))
        rows.append(rr[j])
        cols.append(cc[j])
    net = clusters.copy()
    if rows:
        starts = (np.array(rows), np.array(cols))
        net |= mask_from_entries(starts[0], starts[1], A1, FD1)
    r = int(round(PERRON_CLOSE_R_M / dx))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r * r
    closed = binary_closing(net, structure=disk, border_value=0) | net
    sk = thin_zhang_suen(closed)
    sk = prune_spurs(sk, int(round(PERRON_PRUNE_M / dx)))
    ends = skeleton_endpoints(sk)
    ny, nx = sk.shape
    edge = np.zeros_like(sk)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    heads = ends & ~edge
    hr, hc = np.nonzero(heads)
    return {"kc_threshold": thr, "n_clusters": int(keep.sum()), "n_cluster_cells": int(clusters.sum()), "skeleton": sk, "heads": heads, "head_rows": hr, "head_cols": hc,
            "n_heads": int(hr.size), "n_skeleton_cells": int(sk.sum())}


# ----------------------------------------------------------------------------- surfaces

def partitioned_surface(prof, red, K: float, D: float, Z_obs: np.ndarray) -> tuple[np.ndarray, int]:
    """The model surface Z = a + (U/K) b + (U/D) c on the grid; cells without chi (channel cells not reaching the boundary) take the
    observed elevation; their count is returned."""
    flat = red.Z_flat(K, D)
    Z = prof.to_grid(flat)
    hole = np.isfinite(Z_obs) & ~np.isfinite(Z)
    Z[hole] = Z_obs[hole]
    return Z, int(hole.sum())


def ad_surface(Z0: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, m: float, U: float, K: float, D: float) -> tuple[np.ndarray, float, dict[str, Any]]:
    """The AD surface at r = K/D and y = U/D from one LU (`ADProfiler.reduce`), with its MSE on the valid cells."""
    from lemcvt.models.ad import advection_diffusion_model
    from lemcvt.models.common import extract_grid_data
    from lemcvt.optimize.v2 import ADProfiler
    ad = advection_diffusion_model(Z0, A, FD, dx, m)
    gd = extract_grid_data(Z0, A, FD, dx)
    obs = Z0[gd["rev_j"], gd["rev_i"]].astype(float)
    prof = ADProfiler(ad, obs, U)
    t0 = time.perf_counter()
    red = prof.reduce(K / D)
    y = U / D
    flat = red["h"] + y * red["g"]
    mse = prof.mse_at(red, y)
    Z = np.full(Z0.shape, np.nan)
    Z[gd["rev_j"], gd["rev_i"]] = flat
    return Z, float(mse), {"r": K / D, "y": y, "wall_s": time.perf_counter() - t0}


def residual_stats(Z_model: np.ndarray, Z_obs: np.ndarray) -> dict[str, float]:
    r = Z_model - Z_obs
    ok = np.isfinite(r)
    v = r[ok]
    return {"mse": float(np.mean(v * v)), "mean": float(v.mean()), "rmse": float(np.sqrt(np.mean(v * v))), "n": int(ok.sum()),
            "q05": float(np.percentile(v, 5)), "q95": float(np.percentile(v, 95))}


def rescaled(fit: dict[str, Any], D_hi: float) -> dict[str, Any]:
    """A fit stored at U_OLD expressed at U_NEW (decision 1.1): K and D scale by 65/77, the MSE does not."""
    out = {k: fit.get(k) for k in ("mse", "A_crit", "C", "alpha", "where", "n_channel", "r")}
    out["K"] = fit["K"] * RESCALE
    out["D"] = fit["D"] * RESCALE
    out["D_at_bound_rescaled"] = bool(abs(out["D"] / D_hi - 1) < 1e-9) if D_hi else None
    out["U"] = U_NEW
    return out


def json_ready(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_ready(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


_ = (json, A18, fits_from_reduction, md_polygon_mask, channel_heads, dump, hillshade, load_json, md_table1, metre_tile, ols_loglog_bootstrap, pct,
     segment_bootstrap_median, slurm_record, tile_bounds, a18_specs, A18_SUMMARY, TTV_PRIMARY, MD_TILES, TTV_FULL, MD_EQ10)
