"""``lemcvt tile run --optimizer v2``: the Phase 4 optimizer on one tile (directive Phase 4, addendum 3 section 5).

Stages (``--stages``; ``acrit`` and ``joint`` share one computation and are both produced by
``partitioned``):

``ad``           profile of MSE*(r) over log10 r = log10(K/D) on a 200-point grid spanning the
                 bounds, closed-form U/D at each r, bounded Brent refinement; optimality
                 certificate at the v1 AD optimum (same code: direct solve, MSE gradient by
                 central differences as in 02d).
``partitioned``  the A_crit grid (every unique drainage area between 1e2 and 1e5 m^2, or the
                 2000 values nearest a uniform log grid when the budget rule says so); at every
                 A_crit the MSE at (K0, D0) (v1's stage 2) and the exact box-constrained
                 minimum over (U/K, U/D) (v1's joint stage); identifiability intervals at 1 %
                 and 5 %; certificates at the v1 stage-2 and joint optima with the MSE gradient.
``nonlinear``    the nonlinear model on a 200-value log-thinned A_crit grid at (K0, D0, S_c),
                 then the unique values inside the bracket around the minimum (<= 60 more);
                 Picard record per evaluation; certificate at the v1 nonlinear optimum.
``m_profile``    addendum 3 section 5.2: MSE(A_crit, m) for m = 0.20 ... 0.70 (step 0.01) on 200
                 log-thinned A_crit values, with the exact (U/K, U/D) minimum at each point.

Every misfit evaluation goes to ``trajectory.parquet`` (``stage`` and ``kind`` columns); the
optima, profiles, certificates and identifiability go to ``params.json``; the grids at the v2
optima are written like the v1 pipeline's so that ``lemcvt tile forward --params`` works.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.analyses.outputs import write_model_outputs
from lemcvt.analyses.tile_pipeline import V1_PREFIX, mse
from lemcvt.config import TileData, config_snapshot, data_root, load_tile, load_tile_config
from lemcvt.io.grids import read_geotiff
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.models.partitioned_nonlinear import partitioned_threshold_model_nonlinear
from lemcvt.optimize.v1 import Trajectory
from lemcvt.optimize.v2 import (
    ADProfiler,
    PartitionedProfiler,
    ad_profile,
    exponent_profile,
    identifiability,
    partitioned_profile,
    thin_log_uniform,
    unique_areas,
)
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)

STAGES_V2 = ("ad", "partitioned", "nonlinear", "m_profile")
ACRIT_RANGE = (1e2, 1e5)
THIN_N = 2000
NL_GRID_N = 200
NL_REFINE_MAX = 60
M_GRID = np.round(np.arange(0.20, 0.7001, 0.01), 2)
M_ACRIT_N = 200
BUDGET_CORE_HOURS = 300.0
# per-evaluation wall times from the Phase 3 measurements (03_performance.md, G2; seconds, full tile)
COST_S = {"partitioned": {"AP": 7.0, "GM": 6.0}, "ad": {"AP": 27.0, "GM": 25.0}, "nonlinear": {"AP": 330.0, "GM": 45.0},
          "m_profile_per_acrit": {"AP": 7.0 + 51 * 1.0, "GM": 6.0 + 51 * 1.0}}


# ----------------------------------------------------------------------------- estimate

def estimate(tiles: Sequence[str], thin_n: int = THIN_N, nl_n: int = NL_GRID_N, m_acrit_n: int = M_ACRIT_N) -> dict[str, Any]:
    """Unique-area counts and the core-hour estimate for the full and the thinned A_crit grids (budget rule)."""
    rows = []
    for t in tiles:
        cfg = load_tile_config(t)
        A = read_geotiff(data_root() / cfg.grids["area"]).data
        ua = unique_areas(A, *ACRIT_RANGE)
        site = t[:2]
        rows.append({"tile": t, "n_unique_areas": int(ua.size), "n_thinned": int(thin_log_uniform(ua, thin_n).size),
                     "n_cells_in_range": int(np.sum((A >= ACRIT_RANGE[0]) & (A <= ACRIT_RANGE[1]))),
                     "partitioned_full_h": ua.size * COST_S["partitioned"][site] / 3600,
                     "partitioned_thin_h": min(ua.size, thin_n) * COST_S["partitioned"][site] / 3600,
                     "ad_h": 220 * COST_S["ad"][site] / 3600,
                     "nonlinear_h": (nl_n + NL_REFINE_MAX) * COST_S["nonlinear"][site] / 3600,
                     "m_profile_h": m_acrit_n * COST_S["m_profile_per_acrit"][site] / 3600})
    df = pd.DataFrame(rows)
    tot_full = float(df.partitioned_full_h.sum() + df.ad_h.sum() + df.nonlinear_h.sum() + df.m_profile_h.sum())
    tot_thin = float(df.partitioned_thin_h.sum() + df.ad_h.sum() + df.nonlinear_h.sum() + df.m_profile_h.sum())
    decision = "full" if tot_full <= BUDGET_CORE_HOURS else "thin"
    return {"rows": rows, "total_core_hours_full_grid": tot_full, "total_core_hours_thinned_grid": tot_thin,
            "partitioned_full_grid_core_hours": float(df.partitioned_full_h.sum()),
            "budget_core_hours": BUDGET_CORE_HOURS, "decision": decision, "thin_n": thin_n,
            "cost_assumptions_s": COST_S, "table": df.to_string(index=False, float_format=lambda v: f"{v:.1f}")}


# ----------------------------------------------------------------------------- helpers

def _grid_for(td: TileData, mode: str, thin_n: int) -> tuple[np.ndarray, dict[str, Any]]:
    ua = unique_areas(td.A, *ACRIT_RANGE)
    if mode == "full":
        return ua, {"mode": "full", "n_unique": int(ua.size), "n_used": int(ua.size)}
    thinned = thin_log_uniform(ua, thin_n)
    return thinned, {"mode": "thin", "n_unique": int(ua.size), "n_used": int(thinned.size), "thin_n": thin_n,
                     "rule": "the unique drainage areas nearest a uniform log grid (addendum 3, section 5.1 budget rule)"}


def _ad_gradient_direct(ad, U, K, D, Z0, rel_step=1e-6) -> dict[str, float]:
    """dMSE/dlog10 K, dMSE/dlog10 D by central differences of the direct AD solve (as 02d section 2.1b)."""
    dlog = math.log10(1 + rel_step) - math.log10(1 - rel_step)
    out = {}
    for name in ("K", "D"):
        if name == "K":
            p, m = mse(ad(U, K * (1 + rel_step), D), Z0), mse(ad(U, K * (1 - rel_step), D), Z0)
        else:
            p, m = mse(ad(U, K, D * (1 + rel_step)), Z0), mse(ad(U, K, D * (1 - rel_step)), Z0)
        out[name] = float((p - m) / dlog)
    return out


# ----------------------------------------------------------------------------- driver

def run_tile_v2(tile: str, stages: Sequence[str] = STAGES_V2, acrit_grid: str = "thin", thin_n: int = THIN_N,
                nl_n: int = NL_GRID_N, nl_refine_max: int = NL_REFINE_MAX, m_values=M_GRID, m_acrit_n: int = M_ACRIT_N,
                save_grids: bool = True, derived_grids: bool = False, crop: tuple[int, int, int, int] | None = None,
                run_id: str | None = None, results_root=None) -> RunRecorder:
    stages = tuple(s for s in STAGES_V2 if s in set(stages) | ({"partitioned"} if {"acrit", "joint"} & set(stages) else set()))
    if not stages:
        raise ValueError(f"no valid stage in {stages}; choose from {STAGES_V2} (acrit/joint -> partitioned)")
    td = load_tile(tile, crop=crop, sanity=crop is None)
    mp = td.cfg.model
    Z0, A, FD, dx, U = td.Z, td.A, td.FD, td.dx, mp.U
    ref = td.cfg.v1_reference.get("stages", {})
    if run_id is None:
        from lemcvt.results import new_run_id

        run_id = new_run_id("tile_run", f"{tile}_v2_{'-'.join(stages)}")
    rec = RunRecorder("tile_run", tile, run_id=run_id, root=results_root)
    rec.add_params("tile", tile)
    rec.add_params("optimizer", "v2")
    rec.add_params("stages", list(stages))
    rec.add_params("inputs", td.provenance())
    rec.add_params("model", {"m": mp.m, "U": U, "n": mp.n, "S_c": mp.S_c, "K0_ad": mp.K0_ad, "K0_partitioned": mp.K0_partitioned,
                             "D0": mp.D0, "K_bounds": mp.K_bounds, "D_bounds": mp.D_bounds, "A_crit_0": mp.A_crit_0,
                             "A_crit_bounds": mp.A_crit_bounds})
    stage_results: dict[str, Any] = {}
    v2: dict[str, Any] = {}
    forward_runs: list[dict[str, Any]] = []

    def _save(Z, key, suffix=""):
        if not save_grids:
            return []
        return write_model_outputs(rec, td.filled, Z, f"{td.cfg.site}_1m_best_tile_{V1_PREFIX[key]}{suffix}", derived=derived_grids)

    # ------------------------------------------------------------------ AD
    if "ad" in stages:
        t0 = time.perf_counter()
        ad = advection_diffusion_model(Z0, A, FD, dx, mp.m)
        from lemcvt.models.common import extract_grid_data

        gd = extract_grid_data(Z0, A, FD, dx)
        prof = ADProfiler(ad, Z0[gd["rev_j"], gd["rev_i"]], U)
        traj = Trajectory("ad", ("log10_K", "log10_D", "log10_r", "kind"))
        res = ad_profile(prof, mp.K_bounds, mp.D_bounds, n_grid=200, traj=traj)
        opt = res["optimum"]
        Z = ad(U, opt["K"], opt["D"])
        mse_direct = mse(Z, Z0)
        cert: dict[str, Any] = {"v2_optimum_direct_solve": {"K": opt["K"], "D": opt["D"], "mse_direct": mse_direct,
                                                           "mse_reduction": opt["mse"], "rel_diff": abs(mse_direct - opt["mse"]) / opt["mse"],
                                                           "gradient_log10": _ad_gradient_direct(ad, U, opt["K"], opt["D"], Z0)}}
        if "ad" in ref:
            Kv, Dv = float(ref["ad"]["K"]), float(ref["ad"]["D"])
            Zv = ad(U, Kv, Dv)
            cert["v1_optimum"] = {"K": Kv, "D": Dv, "mse_same_code": mse(Zv, Z0), "mse_printed": ref["ad"].get("MSE"),
                                  "gradient_log10": _ad_gradient_direct(ad, U, Kv, Dv, Z0)}
            traj.record({"log10_K": math.log10(Kv), "log10_D": math.log10(Dv), "log10_r": math.log10(Kv / Dv), "kind": "certificate_v1"},
                        cert["v1_optimum"]["mse_same_code"], cached=False)
        rec.add_trajectory(traj)
        forward_runs.append({"key": "ad_opt", "model": "AD", "params": {"U": U, "K": opt["K"], "D": opt["D"]}, "mse": mse_direct,
                             "files": _save(Z, "ad_opt")})
        stage_results["ad"] = {"params": {"K": opt["K"], "D": opt["D"]}, "mse": opt["mse"], "mse_direct": mse_direct,
                               "n_fev": res["n_evaluations"], "n_solves": res["n_evaluations"], "n_iter": res.get("brent", {}).get("nfev"),
                               "wall_s": time.perf_counter() - t0, "success": True, "message": "profile over r + bounded Brent",
                               "where": opt.get("where"), "x_log10": [math.log10(opt["K"]), math.log10(opt["D"])]}
        v2["ad"] = {k: v for k, v in res.items() if k != "grid_rows"} | {"certificate": cert, "grid_rows": res["grid_rows"]}
        logger.info("AD v2: K=%.4e D=%.4e MSE=%.4f (%s; v1 %.4f) in %.0f s", opt["K"], opt["D"], opt["mse"], opt.get("where"),
                    cert.get("v1_optimum", {}).get("mse_same_code", float("nan")), time.perf_counter() - t0)

    # ------------------------------------------------------------------ partitioned (acrit + joint)
    prof_p = None
    if "partitioned" in stages or "m_profile" in stages:
        prof_p = PartitionedProfiler(Z0, A, FD, dx, U, m_values=[mp.m])
    if "partitioned" in stages:
        t0 = time.perf_counter()
        grid, ginfo = _grid_for(td, acrit_grid, thin_n)
        extras = {}
        if "acrit" in ref:
            extras["v1_stage2"] = (float(ref["acrit"]["A_crit"]), mp.K0_partitioned, mp.D0)
        if "joint" in ref:
            extras["v1_joint"] = (float(ref["joint"]["A_crit"]), float(ref["joint"]["K"]), float(ref["joint"]["D"]))
        if "nonlinear" in ref:
            extras["v1_nonlinear_acrit_linear_model"] = (float(ref["nonlinear"]["A_crit"]), mp.K0_partitioned, mp.D0)
        traj = Trajectory("partitioned", ("log10_A_crit", "log10_D", "log10_K", "m", "kind"))
        res = partitioned_profile(prof_p, grid, mp.m, mp.K_bounds, mp.D_bounds, mp.K0_partitioned, mp.D0, traj=traj,
                                  extra_points=extras)
        rec.add_trajectory(traj)
        rows = res.pop("rows")
        rec.write_table(pd.DataFrame(rows), "acrit_profile")
        s2, jo = res["stage2_optimum"], res["joint_optimum"]
        # direct forward solves at the v2 optima: the reduction against the v1-path model
        pm = partitioned_threshold_model(Z0, A, FD, dx, mp.m, U, mp.K0_partitioned)
        Z2, mask2 = pm(s2["A_crit"], mp.D0)
        mse2 = mse(Z2, Z0)
        pmj = partitioned_threshold_model(Z0, A, FD, dx, mp.m, U, jo["K"])
        Zj, maskj = pmj(jo["A_crit"], jo["D"])
        msej = mse(Zj, Z0)
        red = prof_p.reduce(jo["A_crit"], mp.m)
        Zj_red = prof_p.to_grid(red.Z_flat(jo["K"], jo["D"]))
        relief = float(np.nanmax(Z0) - np.nanmin(Z0))
        res["direct_check"] = {"stage2": {"mse_direct": mse2, "mse_reduction": s2["mse"], "rel_diff": abs(mse2 - s2["mse"]) / s2["mse"]},
                               "joint": {"mse_direct": msej, "mse_reduction": jo["mse"], "rel_diff": abs(msej - jo["mse"]) / jo["mse"],
                                         "max_abs_dz_over_relief": float(np.nanmax(np.abs(Zj - Zj_red)) / relief)}}
        res["grid"] = ginfo
        for key, mse_v in (("v1_stage2", ref.get("acrit", {}).get("MSE")), ("v1_joint", ref.get("joint", {}).get("MSE"))):
            if key in res.get("certificate", {}):
                res["certificate"][key]["mse_printed"] = mse_v
        forward_runs.append({"key": "acrit_opt", "model": "partitioned", "params": {"U": U, "K": mp.K0_partitioned, "D": mp.D0, "A_crit": s2["A_crit"]},
                             "mse": mse2, "files": _save(Z2, "acrit_opt"), "n_channel": int(mask2.sum())})
        forward_runs.append({"key": "joint_opt", "model": "partitioned", "params": {"U": U, "K": jo["K"], "D": jo["D"], "A_crit": jo["A_crit"]},
                             "mse": msej, "files": _save(Zj, "joint_opt"), "n_channel": int(maskj.sum())})
        wall = time.perf_counter() - t0
        stage_results["acrit"] = {"params": {"A_crit": s2["A_crit"]}, "mse": s2["mse"], "mse_direct": mse2, "n_fev": res["n_evaluations"],
                                  "n_solves": res["n_evaluations"], "n_iter": None, "wall_s": wall, "success": True,
                                  "message": f"profile over {ginfo['n_used']} A_crit values ({ginfo['mode']})", "K": mp.K0_partitioned,
                                  "D": mp.D0, "x_log10": [math.log10(s2["A_crit"])]}
        stage_results["joint"] = {"params": {"A_crit": jo["A_crit"], "D": jo["D"], "K": jo["K"]}, "mse": jo["mse"], "mse_direct": msej,
                                  "n_fev": res["n_evaluations"], "n_solves": res["n_evaluations"], "n_iter": None, "wall_s": wall,
                                  "success": True, "message": f"exact box-constrained (U/K, U/D) at each of {ginfo['n_used']} A_crit values; {jo['where']}",
                                  "where": jo["where"], "x_log10": [math.log10(jo["A_crit"]), math.log10(jo["D"]), math.log10(jo["K"])]}
        v2["partitioned"] = res
        logger.info("partitioned v2: stage2 A_crit=%.1f MSE=%.4f; joint A_crit=%.1f K=%.4e D=%.4e MSE=%.4f (%s) in %.0f s", s2["A_crit"],
                    s2["mse"], jo["A_crit"], jo["K"], jo["D"], jo["mse"], jo["where"], wall)

    # ------------------------------------------------------------------ nonlinear
    if "nonlinear" in stages:
        t0 = time.perf_counter()
        pmnl = partitioned_threshold_model_nonlinear(Z0, A, FD, dx, mp.m, U, mp.K0_partitioned)
        ua = unique_areas(A, *ACRIT_RANGE)
        grid = thin_log_uniform(ua, nl_n)
        traj = Trajectory("nonlinear", ("log10_A_crit", "log10_D", "log10_K", "kind"))
        rows = []

        def _eval(a_c, kind):
            t1 = time.perf_counter()
            Z, mask = pmnl(a_c, mp.D0, mp.S_c, mp.nonlinear_max_iter, mp.nonlinear_tol)
            m_ = mse(Z, Z0)
            pic = pmnl.last_picard
            row = {"A_crit": float(a_c), "mse": m_, "n_channel": int(mask.sum()), "picard_iterations": pic["n_iter"],
                   "converged": pic["converged"], "final_update": pic["iterations"][-1]["update_max_abs"] if pic["iterations"] else None,
                   "kind": kind, "wall_s": time.perf_counter() - t1}
            rows.append(row)
            traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(mp.D0), "log10_K": math.log10(mp.K0_partitioned), "kind": kind},
                        m_, cached=False)
            return row, Z, mask

        for i, a_c in enumerate(grid):
            row, _, _ = _eval(a_c, "grid")
            if i % 20 == 0 or i == grid.size - 1:
                logger.info("nonlinear grid %d/%d A_crit=%.1f mse=%.4f (%d Picard, %s) [%.0f s]", i + 1, grid.size, a_c, row["mse"],
                            row["picard_iterations"], "converged" if row["converged"] else "CAP", time.perf_counter() - t0)
        # refine: every unique area inside the bracket around the grid minimum, thinned to <= nl_refine_max
        gm = np.array([r["mse"] for r in rows])
        j = int(np.argmin(gm))
        lo = grid[max(j - 1, 0)]
        hi = grid[min(j + 1, grid.size - 1)]
        inside = ua[(ua > lo) & (ua < hi)]
        inside = inside[~np.isin(inside, grid)]
        refine = thin_log_uniform(inside, nl_refine_max) if inside.size > nl_refine_max else inside
        for a_c in refine:
            _eval(a_c, "refine")
        logger.info("nonlinear refine: %d values in (%.1f, %.1f)", refine.size, lo, hi)
        cert = {}
        if "nonlinear" in ref:
            row, _, _ = _eval(float(ref["nonlinear"]["A_crit"]), "certificate_v1")
            cert["v1_optimum"] = {**row, "mse_printed": ref["nonlinear"].get("MSE")}
        rec.add_trajectory(traj)
        rec.write_table(pd.DataFrame(rows), "nonlinear_profile")
        prof_rows = [r for r in rows if r["kind"] in ("grid", "refine")]
        jb = int(np.argmin([r["mse"] for r in prof_rows]))
        best = prof_rows[jb]
        Zb, maskb = pmnl(best["A_crit"], mp.D0, mp.S_c, mp.nonlinear_max_iter, mp.nonlinear_tol)
        ident = identifiability([{"A_crit": r["A_crit"], "mse": r["mse"]} for r in prof_rows], "mse")
        forward_runs.append({"key": "nonlinear_opt", "model": "partitioned_nonlinear",
                             "params": {"U": U, "K": mp.K0_partitioned, "D": mp.D0, "S_c": mp.S_c, "A_crit": best["A_crit"]},
                             "mse": mse(Zb, Z0), "files": _save(Zb, "nonlinear_opt"), "n_channel": int(maskb.sum()),
                             "picard_iterations": pmnl.last_n_iter, "picard": pmnl.last_picard})
        stage_results["nonlinear"] = {"params": {"A_crit": best["A_crit"]}, "mse": best["mse"], "n_fev": len(rows), "n_solves": len(rows),
                                      "n_iter": None, "wall_s": time.perf_counter() - t0, "success": True,
                                      "message": f"grid of {grid.size} + {refine.size} refinement values", "K": mp.K0_partitioned,
                                      "D": mp.D0, "S_c": mp.S_c, "x_log10": [math.log10(best["A_crit"])]}
        v2["nonlinear"] = {"optimum": best, "identifiability": ident, "certificate": cert, "n_grid": int(grid.size),
                           "n_refine": int(refine.size), "n_unique_in_range": int(ua.size),
                           "n_hit_cap": int(sum(1 for r in rows if not r["converged"])), "n_evaluations": len(rows),
                           "wall_s": time.perf_counter() - t0}
        logger.info("nonlinear v2: A_crit=%.1f MSE=%.4f (%d of %d evaluations hit the Picard cap) in %.0f s", best["A_crit"],
                    best["mse"], v2["nonlinear"]["n_hit_cap"], len(rows), time.perf_counter() - t0)

    # ------------------------------------------------------------------ m profile (5.2)
    if "m_profile" in stages:
        t0 = time.perf_counter()
        ua = unique_areas(A, *ACRIT_RANGE)
        grid = thin_log_uniform(ua, m_acrit_n)
        traj = Trajectory("m_profile", ("log10_A_crit", "log10_D", "log10_K", "m", "kind"))
        res = exponent_profile(prof_p, grid, np.asarray(m_values, float), mp.K_bounds, mp.D_bounds, mp.K0_partitioned, mp.D0, traj=traj)
        rec.add_trajectory(traj)
        rows = res.pop("rows")
        rec.write_table(pd.DataFrame(rows), "m_profile")
        o = res["optimum"]
        res["grid"] = {"n_acrit": int(grid.size), "m_values": [float(m) for m in m_values], "n_unique": int(ua.size)}
        res["config_m_row"] = min((r for r in rows if abs(r["m"] - mp.m) < 5e-3), key=lambda r: r["mse_joint"], default=None)
        v2["m_profile"] = res
        stage_results["m_profile"] = {"params": {"A_crit": o["A_crit"], "m": o["m"], "K": o["K_opt"], "D": o["D_opt"]}, "mse": o["mse_joint"],
                                      "n_fev": res["n_evaluations"], "n_solves": res["n_evaluations"], "n_iter": None,
                                      "wall_s": time.perf_counter() - t0, "success": True, "message": "exact (U/K, U/D) on the (A_crit, m) grid",
                                      "x_log10": [math.log10(o["A_crit"]), o["m"], math.log10(o["D_opt"]), math.log10(o["K_opt"])]}
        logger.info("m profile: optimum A_crit=%.1f m=%.2f K=%.4e D=%.4e MSE=%.4f in %.0f s", o["A_crit"], o["m"], o["K_opt"], o["D_opt"],
                    o["mse_joint"], time.perf_counter() - t0)

    rec.add_params("stages_results", stage_results)
    rec.add_params("forward_runs", forward_runs)
    rec.add_params("v2", v2)
    rec.add_params("v1_reference", ref)
    rec.finalize(config_snapshot(td.cfg, {"crop": crop, "acrit_grid": acrit_grid, "thin_n": thin_n, "nl_n": nl_n,
                                          "m_acrit_n": m_acrit_n}))
    return rec


def refine_partitioned(tile: str, source_run: str | Path, max_extra: int = 400, run_id: str | None = None,
                       results_root=None) -> RunRecorder:
    """Staircase refinement of a thinned partitioned profile (addendum 3, 5.1 budget rule follow-up).

    The thinned grid samples one unique drainage area in about 20; the staircase can have its minimum at a
    skipped step (AP5: v1's joint A_crit = 1886.99 lies between two grid values and is 0.07 % lower in MSE).
    This evaluates every unique area inside the bracket (previous grid value, next grid value) around the
    stage-2 and joint minima of ``source_run``, and around the v1 stage-2 and joint A_crit, and writes a new run
    with the merged profile, the updated optima, identifiability and certificates.
    """
    src = Path(source_run)
    if not src.is_absolute():
        from lemcvt.config import results_root as _rr

        src = _rr() / src
    p0 = json.loads((src / "params.json").read_text())
    prof0 = pd.read_parquet(src / "acrit_profile.parquet")
    td = load_tile(tile)
    mp = td.cfg.model
    Z0, A, FD, dx, U = td.Z, td.A, td.FD, td.dx, mp.U
    ref = td.cfg.v1_reference.get("stages", {})
    if run_id is None:
        from lemcvt.results import new_run_id

        run_id = new_run_id("tile_run", f"{tile}_v2_partitioned-refined")
    rec = RunRecorder("tile_run", tile, run_id=run_id, root=results_root)
    for k in ("tile", "optimizer", "inputs", "model", "v1_reference"):
        rec.add_params(k, p0[k])
    rec.add_params("stages", ["partitioned"])
    rec.add_params("refined_from", src.name)
    grid = np.sort(prof0.A_crit.to_numpy())
    ua = unique_areas(A, *ACRIT_RANGE)
    centres = [float(prof0.A_crit[prof0.mse_stage2.idxmin()]), float(prof0.A_crit[prof0.mse_joint.idxmin()])]
    for key in ("acrit", "joint"):
        if key in ref:
            centres.append(float(ref[key]["A_crit"]))
    extra = set()
    for c in centres:
        j = int(np.searchsorted(grid, c))
        lo = grid[max(j - 1, 0)]
        hi = grid[min(j + 1, grid.size - 1)]
        if c in grid:
            lo = grid[max(j - 1, 0)]
            hi = grid[min(j + 1, grid.size - 1)]
        inside = ua[(ua >= lo) & (ua <= hi)]
        extra |= set(inside.tolist())
    extra -= set(grid.tolist())
    extra_a = np.array(sorted(extra))
    if extra_a.size > max_extra:
        extra_a = thin_log_uniform(extra_a, max_extra)
    prof = PartitionedProfiler(Z0, A, FD, dx, U, m_values=[mp.m])
    traj = Trajectory("partitioned_refine", ("log10_A_crit", "log10_D", "log10_K", "m", "kind"))
    extras = {}
    if "acrit" in ref:
        extras["v1_stage2"] = (float(ref["acrit"]["A_crit"]), mp.K0_partitioned, mp.D0)
    if "joint" in ref:
        extras["v1_joint"] = (float(ref["joint"]["A_crit"]), float(ref["joint"]["K"]), float(ref["joint"]["D"]))
    t0 = time.perf_counter()
    res = partitioned_profile(prof, extra_a, mp.m, mp.K_bounds, mp.D_bounds, mp.K0_partitioned, mp.D0, traj=traj,
                              extra_points=extras, log_every=50)
    rec.add_trajectory(traj)
    rows_new = res.pop("rows")
    merged = pd.concat([prof0, pd.DataFrame(rows_new)], ignore_index=True).sort_values("A_crit").reset_index(drop=True)
    rec.write_table(merged, "acrit_profile")
    rows = merged.to_dict("records")
    j2 = int(np.argmin([r["mse_stage2"] for r in rows]))
    jj = int(np.argmin([r["mse_joint"] for r in rows]))
    s2 = {"A_crit": rows[j2]["A_crit"], "mse": rows[j2]["mse_stage2"], "K": mp.K0_partitioned, "D": mp.D0, "index": j2,
          "n_channel": rows[j2]["n_channel"]}
    jo = {"A_crit": rows[jj]["A_crit"], "mse": rows[jj]["mse_joint"], "K": rows[jj]["K_opt"], "D": rows[jj]["D_opt"],
          "where": rows[jj]["where"], "index": jj, "n_channel": rows[jj]["n_channel"]}
    out = dict(p0["v2"]["partitioned"])
    out.update({"stage2_optimum": s2, "joint_optimum": jo, "stage2_identifiability": identifiability(rows, "mse_stage2"),
                "joint_identifiability": identifiability(rows, "mse_joint"), "certificate": res.get("certificate", out.get("certificate")),
                "refinement": {"source_run": src.name, "centres": centres, "n_extra": int(extra_a.size), "wall_s": time.perf_counter() - t0,
                               "stage2_before": p0["v2"]["partitioned"]["stage2_optimum"], "joint_before": p0["v2"]["partitioned"]["joint_optimum"]},
                "n_evaluations": int(p0["v2"]["partitioned"]["n_evaluations"] + extra_a.size),
                "grid": {**p0["v2"]["partitioned"]["grid"], "n_used": int(merged.shape[0]), "mode": "thin + staircase refinement"}})
    # direct checks at the new optima
    pm = partitioned_threshold_model(Z0, A, FD, dx, mp.m, U, mp.K0_partitioned)
    Z2, mask2 = pm(s2["A_crit"], mp.D0)
    pmj = partitioned_threshold_model(Z0, A, FD, dx, mp.m, U, jo["K"])
    Zj, maskj = pmj(jo["A_crit"], jo["D"])
    mse2, msej = mse(Z2, Z0), mse(Zj, Z0)
    red = prof.reduce(jo["A_crit"], mp.m)
    relief = float(np.nanmax(Z0) - np.nanmin(Z0))
    out["direct_check"] = {"stage2": {"mse_direct": mse2, "mse_reduction": s2["mse"], "rel_diff": abs(mse2 - s2["mse"]) / s2["mse"]},
                           "joint": {"mse_direct": msej, "mse_reduction": jo["mse"], "rel_diff": abs(msej - jo["mse"]) / jo["mse"],
                                     "max_abs_dz_over_relief": float(np.nanmax(np.abs(Zj - prof.to_grid(red.Z_flat(jo["K"], jo["D"])))) / relief)}}
    forward_runs = [
        {"key": "acrit_opt", "model": "partitioned", "params": {"U": U, "K": mp.K0_partitioned, "D": mp.D0, "A_crit": s2["A_crit"]},
         "mse": mse2, "files": write_model_outputs(rec, td.filled, Z2, f"{td.cfg.site}_1m_best_tile_{V1_PREFIX['acrit_opt']}", derived=False),
         "n_channel": int(mask2.sum())},
        {"key": "joint_opt", "model": "partitioned", "params": {"U": U, "K": jo["K"], "D": jo["D"], "A_crit": jo["A_crit"]},
         "mse": msej, "files": write_model_outputs(rec, td.filled, Zj, f"{td.cfg.site}_1m_best_tile_{V1_PREFIX['joint_opt']}", derived=False),
         "n_channel": int(maskj.sum())}]
    sr = dict(p0["stages_results"])
    sr["acrit"] = {**sr["acrit"], "params": {"A_crit": s2["A_crit"]}, "mse": s2["mse"], "mse_direct": mse2, "n_fev": out["n_evaluations"],
                   "message": f"thinned profile + {extra_a.size} staircase values around the minima", "x_log10": [math.log10(s2["A_crit"])]}
    sr["joint"] = {**sr["joint"], "params": {"A_crit": jo["A_crit"], "D": jo["D"], "K": jo["K"]}, "mse": jo["mse"], "mse_direct": msej,
                   "n_fev": out["n_evaluations"], "where": jo["where"], "message": f"exact (U/K, U/D) at each of {merged.shape[0]} A_crit values; {jo['where']}",
                   "x_log10": [math.log10(jo["A_crit"]), math.log10(jo["D"]), math.log10(jo["K"])]}
    rec.add_params("stages_results", sr)
    rec.add_params("forward_runs", forward_runs)
    rec.add_params("v2", {"partitioned": out})
    logger.info("refined %s: stage2 %.6g -> %.6g at A_crit %.2f; joint %.6g -> %.6g at A_crit %.2f (%d extra evaluations, %.0f s)", tile,
                p0["v2"]["partitioned"]["stage2_optimum"]["mse"], s2["mse"], s2["A_crit"], p0["v2"]["partitioned"]["joint_optimum"]["mse"],
                jo["mse"], jo["A_crit"], extra_a.size, time.perf_counter() - t0)
    rec.finalize(config_snapshot(td.cfg, {"refined_from": src.name, "max_extra": max_extra}))
    return rec
