"""Addendum 8, section 3: a slope-dependent channel threshold A S^alpha >= C (replaces Phase 7 Test 1).

3.1 Definitions
    S      the observed local valley slope over a 10 m reach along D8 from the metre DEM: the drop from the cell to the
           first cell at or beyond 10 m of D8 path length, over that path length (fixed: the reduction stays exact).
           Cells whose path leaves the tile before 10 m use the length available (flagged); cells with no receiver have
           no slope and never start a channel.
    heads  cells where A S^alpha >= C first holds along a flow path; every cell downstream of a head is a channel.
           Equivalently, with M = the maximum of A S^alpha over the cell and every cell upstream of it, the channel
           set is M >= C, monotone in C, and its staircase changes exactly at the unique values of M. With alpha = 0,
           M = A (area increases downstream) and the model is the Phase 6 A_crit model with C = A_crit (tested).
           Head cells must drain to the tile boundary: channel cells whose D8 path does not reach it have no chi and
           drop out of the misfit; their number at the optimum is reported.

3.2 The linear joint profile over C at alpha in {0, 0.5, 1, 1.5, 2, 2.5}, at the site m: thin to 2000 unique values of
    M (between the C values that give the channel-cell counts of A_crit = 1e2 and 1e5 m^2), refine around the minima.
    Per alpha: C, K, D, MSE, channel-cell count, the 1 % interval, D / D0.

3.3 Hold-out, from the same reductions: the tile's basins (outlets on the tile edge, network at 1e3 m^2) are split
    into two halves by alternating rank of basin area; every cell is assigned to its basin's half (cells reaching no
    1e3 m^2 channel pixel are in neither half and enter only the 'all' fit). At each C the joint (K, D) optimum is found
    on each half and scored on the other half; the hold-out profile takes, for each fitting half, the C that minimises
    its own MSE and reports the score on the other half; the two scores are averaged (:func:`summarize`).
    Rule: a slope-dependent threshold is preferred at a site if the best alpha > 0 lowers the held-out MSE by more
    than 2 % on >= 4 of 5 tiles.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, cast

import numpy as np
import pandas as pd

from lemcvt.analyses.seam_test import ACRIT_RANGE, _terms
from lemcvt.config import apply_d0_variant, config_snapshot, load_tile
from lemcvt.hydro.d8 import d8_receiver_arrays
from lemcvt.measure.chi import channel_network, channel_pixel_basin_lookup
from lemcvt.measure.hilltop_curvature import basin_labels
from lemcvt.optimize.v1 import Trajectory
from lemcvt.optimize.v2 import PartitionedProfiler, box_quadratic_min, identifiability, thin_log_uniform
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
REACH_M = 10.0
HOLDOUT_A_C = 1e3
RULE = {"holdout_improvement": 0.02, "min_tiles": 4}
MD_AS2_RANGE = (500.0, 4000.0)      # Montgomery & Dietrich (1992), A S^2 at mapped channel heads (m^2)


# ----------------------------------------------------------------------------- slope and the head statistic

def valley_slope_reach(Z: np.ndarray, FD: np.ndarray, dx: float, dy: float, reach_m: float = REACH_M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slope over the D8 path from each cell to the first cell at or beyond ``reach_m``: (S, path length, left_tile)."""
    r_to, c_to, good = d8_receiver_arrays(FD)
    ny, nx = Z.shape
    idx = np.indices((ny, nx))
    cur_r: np.ndarray = idx[0].astype(np.int64)
    cur_c: np.ndarray = idx[1].astype(np.int64)
    L = np.zeros((ny, nx))
    active = good & np.isfinite(Z)
    left = np.zeros((ny, nx), dtype=bool)
    diag_len = math.hypot(dx, dy)
    n_steps = int(math.ceil(reach_m / min(dx, dy))) + 2
    for _ in range(n_steps):
        if not active.any():
            break
        ok = active & good[cur_r, cur_c]
        left |= active & ~ok
        active &= ok
        nr, nc = r_to[cur_r, cur_c], c_to[cur_r, cur_c]
        dr, dc = nr - cur_r, nc - cur_c
        step = np.where((dr != 0) & (dc != 0), diag_len, np.where(dr != 0, dy, dx))
        L = np.where(ok, L + step, L)
        cur_r = np.where(ok, nr, cur_r)
        cur_c = np.where(ok, nc, cur_c)
        active &= L < reach_m
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(L > 0, (Z - Z[cur_r, cur_c]) / np.where(L > 0, L, 1.0), np.nan)
    return S, L, left


def _upstream_pass_py(order, recv, M) -> int:
    n = 0
    for p in order.tolist():
        q = recv[p]
        if q >= 0 and M[p] > M[q]:
            M[q] = M[p]
            n += 1
    return n


try:
    from numba import njit  # type: ignore

    @njit(cache=True)
    def _upstream_pass(order, recv, M):  # pragma: no cover (compiled)
        n = 0
        for p in order:
            q = recv[p]
            if q >= 0 and M[p] > M[q]:
                M[q] = M[p]
                n += 1
        return n
except Exception:  # pragma: no cover - numba missing
    _upstream_pass = _upstream_pass_py  # type: ignore[assignment]


def upstream_max(T: np.ndarray, A: np.ndarray, FD: np.ndarray) -> np.ndarray:
    """M(cell) = max(T(cell), max over donors of M(donor)): cells visited in ascending drainage area, ties by ascending
    flat index. A receiver's area exceeds each donor's (it includes them), so every donor is visited first."""
    r_to, c_to, good = d8_receiver_arrays(FD)
    ny, nx = T.shape
    M = np.where(np.isfinite(T), T, -np.inf).ravel()
    af = np.where(np.isfinite(A), A, np.inf).ravel()
    order = np.lexsort((np.arange(af.size), af))
    recv = np.where(good, r_to * nx + c_to, -1).ravel()
    # one pass is exact where every receiver's area exceeds its donors'; repeat until nothing changes, so cells whose
    # stored area is inconsistent with their donors (the AP1 seam rows, numerics_changes.md N9) are also exact
    for _ in range(1000):
        if _upstream_pass(order, recv, M) == 0:
            break
    M2 = M.reshape(ny, nx)
    return np.where(np.isfinite(M2), M2, np.nan)


def head_statistic(Z: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, dy: float, alpha: float, reach_m: float = REACH_M) -> dict[str, Any]:
    S, L, left = valley_slope_reach(Z, FD, dx, dy, reach_m)
    if alpha == 0.0:
        T = np.where(np.isfinite(A), A, np.nan)
    else:
        Spos = np.where(np.isfinite(S) & (S > 0), S, 0.0)
        T = np.where(np.isfinite(A) & np.isfinite(S), A * Spos ** alpha, np.nan)
    M = upstream_max(T, A, FD)
    if alpha == 0.0:
        M = np.where(np.isfinite(A), A, np.nan)        # exactly the area (also what the max gives; keeps the bits identical)
    return {"S": S, "L": L, "left": left, "T": T, "M": M}


def c_range_from_area_counts(M: np.ndarray, A: np.ndarray, a_lo: float, a_hi: float) -> tuple[float, float]:
    """C values at which the channel set M >= C has as many cells as A >= a_lo and A >= a_hi."""
    m = np.sort(M[np.isfinite(M)])[::-1]
    n_lo, n_hi = int(np.sum(A[np.isfinite(A)] >= a_lo)), int(np.sum(A[np.isfinite(A)] >= a_hi))
    n_lo, n_hi = min(max(n_lo, 1), m.size), min(max(n_hi, 1), m.size)
    return float(m[n_lo - 1]), float(m[n_hi - 1])


# ----------------------------------------------------------------------------- basin halves

def basin_halves(Z: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, dy: float, A_c: float = HOLDOUT_A_C) -> tuple[np.ndarray, dict[str, Any]]:
    """Half (1 or 2) of every cell by alternating rank of its basin's outlet area; 0 for cells in no basin."""
    net = channel_network(Z, A, FD, A_c, dx, dy)
    lab = basin_labels(A, FD, net.mask)                      # channel pixel (row-major, 1-based) first reached, 0 = none
    lookup = channel_pixel_basin_lookup(net)                 # outlet-basin id per channel pixel (row-major order)
    basin = np.where(lab > 0, lookup[np.maximum(lab - 1, 0)], -1)
    outlets = net.outlet_index
    ny, nx = Z.shape
    o_rows, o_cols = net.rows[outlets], net.cols[outlets]
    on_edge = (o_rows == 0) | (o_rows == ny - 1) | (o_cols == 0) | (o_cols == nx - 1)
    o_area = net.area[outlets]
    order = np.argsort(-o_area, kind="stable")
    half_of_basin = {int(outlets[k]): 1 + (r % 2) for r, k in enumerate(order)}
    half = np.zeros((ny, nx), dtype=np.int8)
    for b, h in half_of_basin.items():
        half[basin == b] = h
    half[~np.isfinite(A)] = 0
    n_valid = int(np.isfinite(A).sum())
    info = {"A_c": A_c, "n_basins": int(outlets.size), "n_edge_outlets": int(on_edge.sum()), "n_interior_outlets": int((~on_edge).sum()),
            "basin_areas_m2": {str(int(outlets[k])): float(o_area[k]) for k in order},
            "n_cells_half1": int((half == 1).sum()), "n_cells_half2": int((half == 2).sum()), "n_cells_no_basin": int(((half == 0) & np.isfinite(A)).sum()),
            "frac_cells_no_basin": float(((half == 0) & np.isfinite(A)).sum() / n_valid) if n_valid else float("nan")}
    return half, info


# ----------------------------------------------------------------------------- the profile over C at one alpha

def run_slope_threshold(tile: str, alpha: float, thin_n: int = 2000, refine_max: int = 400, reach_m: float = REACH_M, run_id: str | None = None,
                        results_root=None, crop=None, c_range: tuple[float, float] | None = None) -> RunRecorder:
    td = load_tile(tile, crop=crop, sanity=crop is None)
    mp = apply_d0_variant(td.cfg.model, "A")
    Z0, A, FD, dx, dy, U = td.Z, td.A, td.FD, td.dx, td.filled.dy, mp.U
    rec = RunRecorder("slope_threshold", f"{tile}_a{alpha:g}", run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    rec.add_params("tile", tile)
    rec.add_params("alpha", float(alpha))
    rec.add_params("inputs", td.provenance())
    rec.add_params("model", {"m": mp.m, "U": U, "K0_partitioned": mp.K0_partitioned, "D0": mp.D0, "D0_B": mp.D0_B, "K_bounds": mp.K_bounds,
                             "D_bounds": mp.D_bounds, "z_units_working": td.z_units_working})
    hs = head_statistic(Z0, A, FD, dx, dy, alpha, reach_m)
    S, M = hs["S"], hs["M"]
    valid = np.isfinite(A)
    slope_info = {"reach_m": reach_m, "n_valid": int(valid.sum()), "n_slope_defined": int((valid & np.isfinite(S)).sum()),
                  "n_path_left_tile_before_reach": int((valid & hs["left"]).sum()), "n_slope_zero": int((valid & (S <= 0)).sum()),
                  "S_median": float(np.nanmedian(S[valid])), "S_q16_q84": [float(np.nanpercentile(S[valid], 16)), float(np.nanpercentile(S[valid], 84))],
                  "path_length_median_m": float(np.median(hs["L"][valid & np.isfinite(S)])) if (valid & np.isfinite(S)).any() else float("nan")}
    rec.add_params("slope", slope_info)
    lo_c, hi_c = c_range if c_range is not None else c_range_from_area_counts(M, A, *ACRIT_RANGE)
    half, hinfo = basin_halves(Z0, A, FD, dx, dy)
    rec.add_params("holdout_split", hinfo)
    prof = PartitionedProfiler(Z0, A, FD, dx, U, m_values=[mp.m], channel_stat=M[np.isfinite(A)])
    gd = prof.gd
    half_flat = half[gd["rev_j"], gd["rev_i"]]
    keep = {"all": np.ones(prof.n, dtype=bool), "h1": half_flat == 1, "h2": half_flat == 2}
    lo = np.array([U / mp.K_bounds[1], U / mp.D_bounds[1]])
    hi = np.array([U / mp.K_bounds[0], U / mp.D_bounds[0]])
    mv = M[np.isfinite(M)]
    uv = np.unique(mv[(mv >= lo_c) & (mv <= hi_c)])
    grid = thin_log_uniform(uv, thin_n)
    rec.add_params("setup", {"C_range": [lo_c, hi_c], "C_range_rule": f"channel-cell counts of A_crit = {ACRIT_RANGE[0]:g} and {ACRIT_RANGE[1]:g} m^2",
                             "n_unique_in_range": int(uv.size), "thin_n": thin_n, "refine_max": refine_max,
                             "n_channel_at_C_lo": int(np.sum(mv >= lo_c)), "n_channel_at_C_hi": int(np.sum(mv >= hi_c))})
    traj = Trajectory("slope_threshold_partitioned", ("log10_C", "log10_D", "log10_K", "m", "kind"))
    rows: list[dict[str, Any]] = []

    def _eval(c: float, kind: str) -> None:
        r_ = prof.reduce(float(c), mp.m)
        t_all = r_.mse_terms(prof.Z_obs_flat)
        t1 = _terms(r_, prof.Z_obs_flat, keep["h1"])
        t2 = _terms(r_, prof.Z_obs_flat, keep["h2"])
        v_all, w_all = box_quadratic_min(t_all["G"], t_all["g"], lo, hi)
        v1, w1 = box_quadratic_min(t1["G"], t1["g"], lo, hi)
        v2, w2 = box_quadratic_min(t2["G"], t2["g"], lo, hi)
        ch = r_.channel_mask
        row: dict[str, Any] = {"C": float(c), "m": mp.m, "kind": kind, "n_channel": int(ch.sum()), "n_hillslope": r_.n_hillslope,
                               "n_channel_no_chi": int(np.sum(ch & ~np.isfinite(r_.b))), "n_misfit": int(t_all["n"]),
                               "mse_joint": r_.mse_from_terms(t_all, v_all[0], v_all[1]), "K_opt": float(U / v_all[0]), "D_opt": float(U / v_all[1]), "where": w_all,
                               "mse_stage2": r_.mse_from_terms(t_all, U / mp.K0_partitioned, U / mp.D0),
                               "mse_h1_fit": r_.mse_from_terms(t1, v1[0], v1[1]), "K_h1": float(U / v1[0]), "D_h1": float(U / v1[1]), "where_h1": w1,
                               "mse_h2_at_h1": r_.mse_from_terms(t2, v1[0], v1[1]),
                               "mse_h2_fit": r_.mse_from_terms(t2, v2[0], v2[1]), "K_h2": float(U / v2[0]), "D_h2": float(U / v2[1]), "where_h2": w2,
                               "mse_h1_at_h2": r_.mse_from_terms(t1, v2[0], v2[1]), "n_h1": int(t1["n"]), "n_h2": int(t2["n"]), "wall_s": r_.wall_s}
        rows.append(row)
        traj.record({"log10_C": math.log10(c), "log10_D": math.log10(row["D_opt"]), "log10_K": math.log10(row["K_opt"]), "m": mp.m, "kind": kind},
                    row["mse_joint"], cached=False)

    for i, c in enumerate(grid):
        _eval(c, "grid")
        if i % 200 == 0 or i == grid.size - 1:
            b = min(rows, key=lambda r: r["mse_joint"])
            logger.info("slope threshold alpha=%g %d/%d: best C %.4g (n_channel %d) K %.3e D %.3e mse %.4f [%.0f s]", alpha, i + 1, grid.size, b["C"],
                        b["n_channel"], b["K_opt"], b["D_opt"], b["mse_joint"], time.perf_counter() - t0)
    extra: set[float] = set()
    for col in ("mse_joint", "mse_h1_fit", "mse_h2_fit"):
        j = int(np.argmin([r[col] for r in rows]))
        lo_a, hi_a = grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]
        extra |= set(uv[(uv > lo_a) & (uv < hi_a)].tolist())
    extra -= set(grid.tolist())
    extra_a = np.array(sorted(extra))
    if extra_a.size > refine_max:
        extra_a = thin_log_uniform(extra_a, refine_max)
    for c in extra_a:
        _eval(c, "refine")
    rec.add_trajectory(traj)
    df = pd.DataFrame(rows).sort_values("C").reset_index(drop=True)
    rec.write_table(df, "c_profile")
    recs = cast(list[dict[str, Any]], df.to_dict("records"))
    jj = int(np.argmin(df.mse_joint.to_numpy()))
    j1 = int(np.argmin(df.mse_h1_fit.to_numpy()))
    j2 = int(np.argmin(df.mse_h2_fit.to_numpy()))
    opt: dict[str, Any] = {"C": float(df.C[jj]), "K": float(df.K_opt[jj]), "D": float(df.D_opt[jj]), "mse": float(df.mse_joint[jj]), "where": df["where"][jj],
           "n_channel": int(df.n_channel[jj]), "n_channel_no_chi": int(df.n_channel_no_chi[jj]), "index": jj}
    hold: dict[str, Any] = {"h1_fit": {"C": float(df.C[j1]), "K": float(df.K_h1[j1]), "D": float(df.D_h1[j1]), "mse_fit": float(df.mse_h1_fit[j1]), "mse_heldout_h2": float(df.mse_h2_at_h1[j1])},
            "h2_fit": {"C": float(df.C[j2]), "K": float(df.K_h2[j2]), "D": float(df.D_h2[j2]), "mse_fit": float(df.mse_h2_fit[j2]), "mse_heldout_h1": float(df.mse_h1_at_h2[j2])},
            "mse_heldout_mean": 0.5 * (float(df.mse_h2_at_h1[j1]) + float(df.mse_h1_at_h2[j2]))}
    out: dict[str, Any] = {"joint_optimum": opt, "joint_identifiability": identifiability(recs, "mse_joint", key="C"), "holdout": hold, "n_evaluations": int(df.shape[0]),
           "D_over_D0_A": opt["D"] / mp.D0, "D_over_D0_B": (opt["D"] / mp.D0_B) if mp.D0_B else None, "K_over_K0": opt["K"] / mp.K0_partitioned,
           "AS2_equivalent_note": "at alpha = 2, C is A S^2 in m^2 (Montgomery & Dietrich 1992: 500-4000 m^2 at mapped heads with a field slope)"}
    if alpha == 0.0:
        out["phase6_check"] = _phase6_check(tile, df)
    rec.add_params("result", out)
    rec.add_params("wall_s", time.perf_counter() - t0)
    logger.info("%s alpha=%g: C %.4g, K %.3e, D %.3e (D/D0_A %.2f), MSE %.4f, n_channel %d (%d without chi); hold-out %.4f / %.4f -> mean %.4f [%.0f s]", tile, alpha,
                opt["C"], opt["K"], opt["D"], out["D_over_D0_A"], opt["mse"], opt["n_channel"], opt["n_channel_no_chi"], hold["h1_fit"]["mse_heldout_h2"],
                hold["h2_fit"]["mse_heldout_h1"], hold["mse_heldout_mean"], time.perf_counter() - t0)
    rec.finalize(config_snapshot(td.cfg, {"crop": crop, "alpha": alpha, "thin_n": thin_n, "refine_max": refine_max}))
    return rec


def _phase6_check(tile: str, df: pd.DataFrame) -> dict[str, Any]:
    """alpha = 0 against the Phase 6 refined partitioned run at the common A_crit values (the same rows must result)."""
    try:
        from lemcvt.analyses.curvature_compare import _runs_by_stage

        runs = _runs_by_stage(tile)
        if "joint" not in runs:
            return {"available": False}
        r, p = runs["joint"]
        ref = pd.read_parquet(r / "acrit_profile.parquet")
        a = df.set_index("C")
        b = ref.set_index("A_crit")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            return {"available": True, "n_common": 0}
        da = a.loc[common, "mse_joint"].to_numpy()
        db = b.loc[common, "mse_joint"].to_numpy()
        jo = p["v2"]["partitioned"]["joint_optimum"]
        return {"available": True, "run": r.name, "n_common": int(len(common)), "max_rel_diff_mse": float(np.max(np.abs(da / db - 1))),
                "max_rel_diff_D": float(np.max(np.abs(a.loc[common, "D_opt"].to_numpy() / b.loc[common, "D_opt"].to_numpy() - 1))),
                "phase6_joint_optimum": {k: jo[k] for k in ("A_crit", "K", "D", "mse")}}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


# ----------------------------------------------------------------------------- summary over alphas and tiles

def collect(tiles) -> dict[str, dict[float, tuple[Any, dict[str, Any]]]]:
    out: dict[str, dict[float, tuple[Any, dict[str, Any]]]] = {}
    for r in find_runs("slope_threshold"):
        try:
            p = json.loads((r / "params.json").read_text())
            rj = json.loads((r / "run.json").read_text())
        except (OSError, ValueError):
            continue
        if (rj.get("config") or {}).get("crop") or p.get("tile") not in tiles or "result" not in p:
            continue
        out.setdefault(p["tile"], {})[float(p["alpha"])] = (r, p)       # newest wins (find_runs is sorted)
    return out


def summarize(tiles, run_id: str | None = None, results_root=None) -> RunRecorder:
    """The rule of 3.3 per tile and site from the latest run of every (tile, alpha), written as a run."""
    runs = collect(tiles)
    rec = RunRecorder("slope_threshold_summary", f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    rows = []
    per_tile: dict[str, Any] = {}
    for tile in tiles:
        if tile not in runs:
            continue
        base = runs[tile].get(0.0)
        alphas = sorted(runs[tile])
        tab: dict[float, dict[str, Any]] = {}
        for a in alphas:
            r, p = runs[tile][a]
            res = p["result"]
            tab[a] = {"run": r.name, "C": res["joint_optimum"]["C"], "K": res["joint_optimum"]["K"], "D": res["joint_optimum"]["D"], "mse": res["joint_optimum"]["mse"],
                      "n_channel": res["joint_optimum"]["n_channel"], "n_channel_no_chi": res["joint_optimum"]["n_channel_no_chi"], "D_over_D0_A": res["D_over_D0_A"],
                      "D_over_D0_B": res.get("D_over_D0_B"), "K_over_K0": res["K_over_K0"], "mse_heldout_mean": res["holdout"]["mse_heldout_mean"],
                      "mse_heldout_h2_at_h1": res["holdout"]["h1_fit"]["mse_heldout_h2"], "mse_heldout_h1_at_h2": res["holdout"]["h2_fit"]["mse_heldout_h1"],
                      "C_h1": res["holdout"]["h1_fit"]["C"], "C_h2": res["holdout"]["h2_fit"]["C"],
                      "interval_1pct": res["joint_identifiability"]["level_0.01"]["intervals"], "n_evaluations": res["n_evaluations"]}
            rows.append({"tile": tile, "alpha": a} | {k: v for k, v in tab[a].items() if k != "interval_1pct"})
        h0 = tab[0.0]["mse_heldout_mean"] if base is not None else float("nan")
        best_a = min((a for a in alphas if a > 0), key=lambda a: tab[a]["mse_heldout_mean"], default=None)
        imp = (1.0 - tab[best_a]["mse_heldout_mean"] / h0) if (best_a is not None and np.isfinite(h0)) else float("nan")
        best_fit_a = min(alphas, key=lambda a: tab[a]["mse"])
        per_tile[tile] = {"alphas": alphas, "table": {str(a): v for a, v in tab.items()}, "heldout_alpha0": h0, "best_alpha_heldout": best_a,
                          "heldout_improvement_best": imp, "preferred": bool(np.isfinite(imp) and imp > RULE["holdout_improvement"]),
                          "best_alpha_fit": best_fit_a, "fit_improvement_best": (1.0 - tab[best_fit_a]["mse"] / tab[0.0]["mse"]) if base is not None else float("nan"),
                          "complete": len(alphas) == len(ALPHAS)}
    sites: dict[str, Any] = {}
    for site in ("AP", "GM"):
        ts = [t for t in per_tile if t.startswith(site)]
        if not ts:
            continue
        n_pref = sum(per_tile[t]["preferred"] for t in ts)
        best = [per_tile[t]["best_alpha_heldout"] for t in ts]
        sites[site] = {"n_tiles": len(ts), "n_preferred": n_pref, "preferred": bool(n_pref >= RULE["min_tiles"]), "best_alpha_per_tile": dict(zip(ts, best, strict=True)),
                       "n_complete": sum(per_tile[t]["complete"] for t in ts)}
    rec.add_params("rule", RULE)
    rec.add_params("tiles", per_tile)
    rec.add_params("sites", sites)
    rec.add_params("MD_AS2_range_m2", list(MD_AS2_RANGE))
    if rows:
        rec.write_table(pd.DataFrame(rows), "summary")
    rec.finalize({"tiles": list(tiles)})
    return rec
