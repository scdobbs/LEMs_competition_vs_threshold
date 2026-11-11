"""Addendum 6, section 3: does the AP1 seam matter? (pre-registered)

The affected set S on AP1 is the union of rows 385-395, every cell downstream (D8, stored directions) of the 345
seam cells, and the cells whose stored area is inconsistent with their in-tile donors (|ext| > 0.5 cell). The linear
joint profile, the stage-2 profile at D0 and the AD profile are evaluated twice at the same A_crit (or r) values:
with the misfit over all valid cells, and with the misfit over all valid cells except S. The reductions are shared
(one LU per A_crit, one per r); only the Gram matrix of the misfit changes with the mask.

Rule (addendum 6, 3.4): the seam is negligible if the masked joint A_crit lies inside the unmasked 1 % interval, K moves
by less than 2 %, D by less than 5 %, and the AD r by less than 0.1 decade.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, cast

import numpy as np
import pandas as pd

from lemcvt.analyses.seamfix import classify_seam, external_inflow, receiver_rc
from lemcvt.config import apply_d0_variant, config_snapshot, load_tile
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.common import extract_grid_data
from lemcvt.optimize.v1 import Trajectory
from lemcvt.optimize.v2 import (
    ADProfiler,
    PartitionedProfiler,
    _row,
    identifiability,
    thin_log_uniform,
    unique_areas,
)
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)

SEAM_ROWS = (385, 395)
ACRIT_RANGE = (1e2, 1e5)


def affected_set(Z: np.ndarray, A: np.ndarray, fd0: np.ndarray, rows=SEAM_ROWS) -> tuple[np.ndarray, dict[str, Any]]:
    """The set S of addendum 6, 3.1, as a 2-D boolean mask, with the size of each component."""
    ny, nx = fd0.shape
    S_rows = np.zeros((ny, nx), dtype=bool)
    S_rows[rows[0]:rows[1] + 1, :] = True
    cls = classify_seam(Z, None, fd0)
    down = np.zeros((ny, nx), dtype=bool)
    for r, c in cls["seam_rc"].tolist():
        cur: tuple[int, int] | None = (int(r), int(c))
        steps = 0
        while cur is not None and steps < ny * nx:
            if down[cur]:
                break
            down[cur] = True
            cur = receiver_rc(int(fd0[cur]), cur[0], cur[1], ny, nx)
            steps += 1
    cell = float(np.nanmin(A))
    ext = external_inflow(fd0, A, Z, math.sqrt(cell))
    interior = np.zeros((ny, nx), dtype=bool)
    interior[1:-1, 1:-1] = True
    incons = interior & np.isfinite(ext) & (np.abs(ext) > 0.5 * cell)
    S = (S_rows | down | incons) & np.isfinite(A)
    info = {"rows": list(rows), "n_rows_band": int((S_rows & np.isfinite(A)).sum()), "n_seam_cells": cls["n_seam"],
            "n_downstream_of_seam": int(down.sum()), "n_inconsistent": int(incons.sum()), "n_S": int(S.sum()),
            "frac_S_of_valid": float(int(S.sum()) / int(np.isfinite(A).sum())), "n_valid": int(np.isfinite(A).sum())}
    return S, info


def _terms(red, Z_obs_flat, keep):
    """Gram matrix and vector of the quadratic MSE over the cells in ``keep`` (the reduction's cache is bypassed)."""
    r = Z_obs_flat - red.a
    ok = keep & np.isfinite(r) & np.isfinite(red.b) & np.isfinite(red.c)
    b, c, r = red.b[ok], red.c[ok], r[ok]
    n = int(ok.sum())
    return {"G": np.array([[b @ b, b @ c], [b @ c, c @ c]]) / n, "g": np.array([b @ r, c @ r]) / n, "rr": float(r @ r) / n, "n": n}


def run_seam_test(tile: str = "AP1", thin_n: int = 2000, refine_max: int = 400, n_grid_ad: int = 200, run_id: str | None = None,
                  results_root=None, crop=None) -> RunRecorder:
    td = load_tile(tile, crop=crop, sanity=crop is None)
    mp = apply_d0_variant(td.cfg.model, "A")
    Z0, A, FD, dx, U = td.Z, td.A, td.FD, td.dx, mp.U
    rec = RunRecorder("seam_test", tile, run_id=run_id, root=results_root)
    rec.add_params("tile", tile)
    rec.add_params("inputs", td.provenance())
    rec.add_params("model", {"m": mp.m, "U": U, "K0_partitioned": mp.K0_partitioned, "D0": mp.D0, "D0_B": mp.D0_B, "K_bounds": mp.K_bounds,
                             "D_bounds": mp.D_bounds, "z_units_working": td.z_units_working})
    t0 = time.perf_counter()
    S, s_info = affected_set(Z0, A, FD.astype(np.int64))
    rec.add_params("affected_set", s_info)
    rec.write_grid(td.filled.with_data(S.astype(np.float64)), f"{tile}_affected_set_S")
    logger.info("%s: |S| = %d cells (%.2f %% of valid): rows band %d, downstream of seam %d, inconsistent %d", tile, s_info["n_S"],
                100 * s_info["frac_S_of_valid"], s_info["n_rows_band"], s_info["n_downstream_of_seam"], s_info["n_inconsistent"])

    # ---------------------------------------------------------------- linear partitioned: stage 2 and joint, both masks
    prof = PartitionedProfiler(Z0, A, FD, dx, U, m_values=[mp.m])
    gd = prof.gd
    keep_all = np.ones(prof.n, dtype=bool)
    keep_masked = ~S[gd["rev_j"], gd["rev_i"]]
    lo = np.array([U / mp.K_bounds[1], U / mp.D_bounds[1]])
    hi = np.array([U / mp.K_bounds[0], U / mp.D_bounds[0]])
    ua = unique_areas(A, *ACRIT_RANGE)
    grid = thin_log_uniform(ua, thin_n)
    traj = Trajectory("seam_test_partitioned", ("log10_A_crit", "log10_D", "log10_K", "m", "kind"))
    rows_all: list[dict[str, Any]] = []
    rows_m: list[dict[str, Any]] = []

    def _eval(a_c: float, kind: str) -> None:
        red = prof.reduce(float(a_c), mp.m)
        for keep, rows, tag in ((keep_all, rows_all, "all"), (keep_masked, rows_m, "masked")):
            row: dict[str, Any] = _row(red, _terms(red, prof.Z_obs_flat, keep), U, mp.K0_partitioned, mp.D0, lo, hi)
            row["kind"] = kind
            rows.append(row)
            traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(row["D_opt"]), "log10_K": math.log10(row["K_opt"]), "m": mp.m,
                         "kind": f"{tag}_{kind}"}, row["mse_joint"], cached=False)

    for i, a_c in enumerate(grid):
        _eval(a_c, "grid")
        if i % 200 == 0 or i == grid.size - 1:
            ja = min(rows_all, key=lambda r: r["mse_joint"])
            jm = min(rows_m, key=lambda r: r["mse_joint"])
            logger.info("seam test %d/%d: joint all A_crit %.1f mse %.4f | masked A_crit %.1f mse %.4f [%.0f s]", i + 1, grid.size, ja["A_crit"],
                        ja["mse_joint"], jm["A_crit"], jm["mse_joint"], time.perf_counter() - t0)
    extra = set()
    for rows in (rows_all, rows_m):
        for col in ("mse_stage2", "mse_joint"):
            j = int(np.argmin([r[col] for r in rows]))
            lo_a, hi_a = grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]
            extra |= set(ua[(ua > lo_a) & (ua < hi_a)].tolist())
    extra -= set(grid.tolist())
    extra_a = np.array(sorted(extra))
    if extra_a.size > refine_max:
        extra_a = thin_log_uniform(extra_a, refine_max)
    for a_c in extra_a:
        _eval(a_c, "refine")
    rec.add_trajectory(traj)
    df_all = pd.DataFrame(rows_all).sort_values("A_crit").reset_index(drop=True)
    df_m = pd.DataFrame(rows_m).sort_values("A_crit").reset_index(drop=True)
    df_all["mask"] = "all"
    df_m["mask"] = "masked"
    rec.write_table(pd.concat([df_all, df_m], ignore_index=True), "acrit_profile")

    def _summ(df: pd.DataFrame) -> dict[str, Any]:
        rows = cast(list[dict[str, Any]], df.to_dict("records"))
        j2, jj = int(np.argmin(df.mse_stage2.to_numpy())), int(np.argmin(df.mse_joint.to_numpy()))
        return {"stage2_optimum": {"A_crit": float(df.A_crit[j2]), "mse": float(df.mse_stage2[j2])},
                "joint_optimum": {"A_crit": float(df.A_crit[jj]), "K": float(df.K_opt[jj]), "D": float(df.D_opt[jj]), "mse": float(df.mse_joint[jj]),
                                  "where": df["where"][jj]},
                "stage2_identifiability": identifiability(rows, "mse_stage2"), "joint_identifiability": identifiability(rows, "mse_joint")}

    part: dict[str, Any] = {"all": _summ(df_all), "masked": _summ(df_m), "n_evaluations": int(df_all.shape[0]), "grid": {"thin_n": thin_n, "n_refine": int(extra_a.size), "n_unique": int(ua.size)},
            "n_cells_misfit_all": int(keep_all.sum()), "n_cells_misfit_masked": int(keep_masked.sum())}
    shape = {"joint_profile_max_rel_shape_diff": float(np.nanmax(np.abs((df_all.mse_joint / df_all.mse_joint.min()) / (df_m.mse_joint / df_m.mse_joint.min()) - 1))),
             "stage2_profile_max_rel_shape_diff": float(np.nanmax(np.abs((df_all.mse_stage2 / df_all.mse_stage2.min()) / (df_m.mse_stage2 / df_m.mse_stage2.min()) - 1)))}
    part["profile_shape"] = shape

    # ---------------------------------------------------------------- AD: both masks on the same r grid, then Brent each
    ad = advection_diffusion_model(Z0, A, FD, dx, mp.m)
    gd2 = extract_grid_data(Z0, A, FD, dx)
    obs = Z0[gd2["rev_j"], gd2["rev_i"]]
    prof_ad = ADProfiler(ad, obs, U)
    r_lo, r_hi = mp.K_bounds[0] / mp.D_bounds[1], mp.K_bounds[1] / mp.D_bounds[0]
    rgrid = np.logspace(math.log10(r_lo), math.log10(r_hi), n_grid_ad)
    ad_rows: dict[str, list[dict[str, Any]]] = {"all": [], "masked": []}
    traj_ad = Trajectory("seam_test_ad", ("log10_K", "log10_D", "log10_r", "kind"))

    def _best(red: dict[str, Any], keep: np.ndarray) -> dict[str, Any] | None:
        r = prof_ad.obs - red["h"]
        g = red["g"]
        ok = keep & np.isfinite(r) & np.isfinite(g)
        gg = float(g[ok] @ g[ok])
        D_lo = max(mp.D_bounds[0], mp.K_bounds[0] / red["r"])
        D_hi = min(mp.D_bounds[1], mp.K_bounds[1] / red["r"])
        if D_lo > D_hi:
            return None
        y_lo, y_hi = U / D_hi, U / D_lo
        y = float(g[ok] @ r[ok]) / gg if gg > 0 else y_lo
        where = "interior"
        if y < y_lo:
            y, where = y_lo, "y_lo"
        elif y > y_hi:
            y, where = y_hi, "y_hi"
        res = r[ok] - y * g[ok]
        D = U / y
        return {"r": red["r"], "D": float(D), "K": float(red["r"] * D), "mse": float(res @ res / ok.sum()), "where": where}

    def _eval_ad(r: float, kind: str) -> dict[str, Any]:
        red = prof_ad.reduce(float(r))
        out: dict[str, Any] = {}
        for tag, keep in (("all", keep_all), ("masked", keep_masked)):
            q = _best(red, keep)
            if q is None:
                continue
            q["kind"] = kind
            ad_rows[tag].append(q)
            traj_ad.record({"log10_K": math.log10(q["K"]), "log10_D": math.log10(q["D"]), "log10_r": math.log10(r), "kind": f"{tag}_{kind}"}, q["mse"], cached=False)
            out[tag] = q
        return out

    for i, r in enumerate(rgrid):
        _eval_ad(r, "grid")
        if i % 50 == 0 or i == rgrid.size - 1:
            logger.info("seam test AD %d/%d [%.0f s]", i + 1, rgrid.size, time.perf_counter() - t0)
    from scipy.optimize import minimize_scalar

    ad_res: dict[str, Any] = {}
    for tag in ("all", "masked"):
        rows = sorted(ad_rows[tag], key=lambda q: q["r"])
        feas = [q for q in rows if q["kind"] == "grid"]
        j = int(np.argmin([q["mse"] for q in feas]))
        lg = np.log10([q["r"] for q in feas])
        a, b = lg[max(j - 1, 0)], lg[min(j + 1, len(lg) - 1)]
        res = minimize_scalar(lambda lr: _eval_ad(10 ** lr, f"brent_{tag}")[tag]["mse"], bounds=(a, b), method="bounded", options={"xatol": 1e-6})
        fin = _eval_ad(10 ** res.x, f"brent_{tag}")[tag]
        best = fin if fin["mse"] <= feas[j]["mse"] else feas[j]
        grid_rows = [{"r": q["r"], "mse": q["mse"], "K_opt": q["K"], "D_opt": q["D"]} for q in feas]
        ad_res[tag] = {"optimum": best, "identifiability": identifiability(grid_rows, "mse", key="r"), "n_evaluations": len(ad_rows[tag])}
    rec.add_trajectory(traj_ad)
    rec.write_table(pd.DataFrame([dict(q, mask=tag) for tag in ("all", "masked") for q in ad_rows[tag]]), "ad_profile")

    # ---------------------------------------------------------------- the rule
    ja, jm = part["all"]["joint_optimum"], part["masked"]["joint_optimum"]
    i1 = part["all"]["joint_identifiability"]["level_0.01"]
    inside = any(lo_ <= jm["A_crit"] <= hi_ for lo_, hi_ in i1["intervals"])
    dK = abs(jm["K"] / ja["K"] - 1)
    dD = abs(jm["D"] / ja["D"] - 1)
    ra, rm = ad_res["all"]["optimum"]["r"], ad_res["masked"]["optimum"]["r"]
    dr = abs(math.log10(rm / ra))
    verdict = {"masked_joint_A_crit_inside_unmasked_1pct_interval": bool(inside), "unmasked_1pct_intervals": i1["intervals"],
               "joint_A_crit": {"all": ja["A_crit"], "masked": jm["A_crit"]}, "K_rel_change": dK, "D_rel_change": dD,
               "ad_r": {"all": ra, "masked": rm}, "ad_r_log10_change": dr,
               "thresholds": {"K_rel": 0.02, "D_rel": 0.05, "ad_r_log10": 0.1},
               "negligible": bool(inside and dK < 0.02 and dD < 0.05 and dr < 0.1),
               "stage2_A_crit": {"all": part["all"]["stage2_optimum"]["A_crit"], "masked": part["masked"]["stage2_optimum"]["A_crit"]}}
    rec.add_params("partitioned", part)
    rec.add_params("ad", ad_res)
    rec.add_params("verdict", verdict)
    rec.add_params("wall_s", time.perf_counter() - t0)
    logger.info("%s seam test: negligible = %s (A_crit %.1f -> %.1f inside 1%% %s; K %.2f %%; D %.2f %%; r %.3f dec)", tile, verdict["negligible"],
                ja["A_crit"], jm["A_crit"], inside, 100 * dK, 100 * dD, dr)
    rec.finalize(config_snapshot(td.cfg, {"crop": crop, "thin_n": thin_n, "refine_max": refine_max}))
    return rec
