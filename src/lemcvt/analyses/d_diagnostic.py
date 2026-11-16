"""Addendum 7, section 2: where does the fitted D come from? (pre-registered diagnostic)

Hypothesis under test (not assumed): in the partitioned model every cell with A < A_crit is a linear-diffusion
hillslope cell with -laplacian z = U/D, convex everywhere, while the real surface between the slope-area rollover and
A_crit holds colluvial hollows that are concave or nearly planar; a least-squares fit over all hillslope cells averages
convex divides with non-convex hollows, lowers the fitted convexity U/D and so raises D.

2.1 The model hillslope cells at the tile's Phase 6 linear joint optimum (A < A_crit; boundary and channel cells
excluded) are split into *divergent* (A < A_roll, the tile's slope-area rollover of the `02b`/`02c` diagnostics),
*hollow* (A_roll <= A < A_crit) and *hilltop* (the option (iii) pixel set, a subset of divergent, reported separately).
For each class: count and share, mean and median observed -laplacian at the 3-cell window (the model's 5-point stencil)
and at L_f = 11 m (the quadratic-surface estimator), the fraction of concave cells, and the share of the joint-optimum
hillslope SSR. The mean over all hillslope cells is reported against U/D_fit.

2.2 The linear joint profile is rerun with the misfit over (1) divergent cells plus channel cells and (2) hollow cells
plus channel cells; (3) all cells is the Phase 6 run and is evaluated here from the same reductions as a check. The
channel cells of every A_crit stay in each mask so that K stays identified; the classes are fixed at the Phase 6
optimum. One reduction (LU + solve) per A_crit serves the three masks (only the Gram matrix of the misfit changes), as
in the seam test.

2.3 Rule, per tile here and per site in the report (supported at a site if all three hold on >= 4 of 5 tiles):
(a) the mean observed -laplacian over all hillslope cells at the 3-cell window is within +-30 % of U/D_fit;
(b) the hollow class's mean -laplacian is below half of the divergent class's;
(c) D from the divergent-only refit is at least a factor 2 closer to D0_A than D_fit is, in log ratio, read as
    |log10(D_div / D0_A)| <= |log10(D_fit / D0_A)| - log10(2) (the ratio to D0_A shrinks by at least a factor 2);
    the stricter reading, that the log distance halves, is reported beside it.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, cast

import numpy as np
import pandas as pd

from lemcvt.analyses.curvature_compare import _runs_by_stage, laplacian5
from lemcvt.analyses.seam_test import ACRIT_RANGE, _terms
from lemcvt.config import apply_d0_variant, config_snapshot, load_tile
from lemcvt.measure.hilltop_curvature import total_curvature
from lemcvt.measure.phase6_inputs import W_F, option_iii_mask
from lemcvt.optimize.v1 import Trajectory
from lemcvt.optimize.v2 import PartitionedProfiler, _row, identifiability, thin_log_uniform, unique_areas
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

CLASSES = ("divergent", "hollow", "hilltop")
MASKS = ("all", "divergent", "hollow")
RULE = {"mean_lap_within": 0.30, "hollow_below_fraction_of_divergent": 0.5, "closer_factor": 2.0}


def rollover_area(tile: str) -> tuple[float, str]:
    """The tile's slope-area rollover A_roll from the latest `measure_diagnostics` run (`scaling_break.A_rollover`)."""
    for r in reversed(find_runs("measure_diagnostics")):
        try:
            p = json.loads((r / "params.json").read_text())
            v = p["tiles"][tile]["scaling_break"]["A_rollover"]
            if v is not None and np.isfinite(v):
                return float(v), r.name
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise FileNotFoundError(f"{tile}: no measure_diagnostics run with scaling_break.A_rollover")


def curvature_network_threshold(tile: str) -> tuple[float, str]:
    """A_c of the rollover network the option (iii) set is built on (the `measure` run's curvature block)."""
    for r in reversed(find_runs("measure")):
        try:
            p = json.loads((r / "params.json").read_text())
            v = p["tiles"][tile]["curvature"]["A_c"]
            return float(v), r.name
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise FileNotFoundError(f"{tile}: no measure run with a curvature A_c")


def _stats(v: np.ndarray) -> dict[str, float | int]:
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "frac_concave": float("nan")}
    return {"n": int(v.size), "mean": float(np.mean(v)), "median": float(np.median(v)), "frac_concave": float(np.mean(v < 0))}


def run_d_diagnostic(tile: str, thin_n: int = 2000, refine_max: int = 400, A_roll: float | None = None, A_c: float | None = None,
                     run_id: str | None = None, results_root=None, crop=None, joint_run: str | None = None) -> RunRecorder:
    td = load_tile(tile, crop=crop, sanity=crop is None)
    mp = apply_d0_variant(td.cfg.model, "A")
    Z0, A, FD, dx, dy, U = td.Z, td.A, td.FD, td.dx, td.filled.dy, mp.U
    rec = RunRecorder("d_diagnostic", tile, run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    # ------------------------------------------------------------------ the Phase 6 joint optimum and the inputs
    if joint_run is None:
        runs = _runs_by_stage(tile)
        if "joint" not in runs:
            raise FileNotFoundError(f"{tile}: no Phase 6 refined partitioned run")
        jr, jp = runs["joint"]
    else:
        from pathlib import Path

        jr = Path(joint_run)
        jp = json.loads((jr / "params.json").read_text())
    jo = jp["v2"]["partitioned"]["joint_optimum"]
    A_crit_opt, K_fit, D_fit = float(jo["A_crit"]), float(jo["K"]), float(jo["D"])
    roll_src = acs = ""
    if A_roll is None:
        A_roll, roll_src = rollover_area(tile)
    if A_c is None:
        A_c, acs = curvature_network_threshold(tile)
    rec.add_params("tile", tile)
    rec.add_params("inputs", td.provenance())
    rec.add_params("model", {"m": mp.m, "U": U, "K0_partitioned": mp.K0_partitioned, "D0": mp.D0, "D0_B": mp.D0_B, "K_bounds": mp.K_bounds,
                             "D_bounds": mp.D_bounds, "z_units_working": td.z_units_working})
    rec.add_params("setup", {"joint_run": jr.name, "joint_optimum": jo, "A_roll": A_roll, "A_roll_source": roll_src, "A_c_rollover_network": A_c,
                             "A_c_source": acs, "w_f": W_F, "thin_n": thin_n, "refine_max": refine_max, "A_crit_range": list(ACRIT_RANGE), "rule": RULE})

    # ------------------------------------------------------------------ 2.1 classes on the model's cell index
    prof = PartitionedProfiler(Z0, A, FD, dx, U, m_values=[mp.m])
    gd = prof.gd
    rj, ri = gd["rev_j"], gd["rev_i"]
    red = prof.reduce(A_crit_opt, mp.m)
    channel_opt, hs_mask, _hs, _to_red = prof.masks(A_crit_opt)
    A_flat = prof.A_flat
    mask3, _m0, _dist, diag3 = option_iii_mask(Z0, A, FD, dx, dy, float(A_c), W_F)
    cls = {"divergent": hs_mask & (A_flat < A_roll), "hollow": hs_mask & (A_flat >= A_roll), "hilltop": hs_mask & mask3[rj, ri]}
    Zm = red.Z_flat(K_fit, D_fit)
    resid2 = (prof.Z_obs_flat - Zm) ** 2
    lap3 = -laplacian5(Z0, dx, dy)[rj, ri]                   # observed -laplacian, 3-cell window
    c11 = -total_curvature(Z0, W_F, dx, dy)[rj, ri]           # observed -laplacian of the quadratic surface at L_f = 11 m
    lap3_model = -laplacian5(prof.to_grid(Zm), dx, dy)[rj, ri]
    n_hs = int(hs_mask.sum())
    ssr_hs = float(np.nansum(resid2[hs_mask]))
    classes: dict[str, Any] = {}
    for name in CLASSES:
        k = cls[name]
        classes[name] = {"n": int(k.sum()), "share_of_hillslope": float(k.sum() / n_hs) if n_hs else float("nan"),
                         "lap3": _stats(lap3[k]), "lap11": _stats(c11[k]), "model_lap3": _stats(lap3_model[k]),
                         "ssr": float(np.nansum(resid2[k])), "ssr_share_of_hillslope": float(np.nansum(resid2[k]) / ssr_hs) if ssr_hs else float("nan"),
                         "mse": float(np.nanmean(resid2[k])) if k.any() else float("nan")}
    all_hs: dict[str, Any] = {"n": n_hs, "lap3": _stats(lap3[hs_mask]), "lap11": _stats(c11[hs_mask]), "model_lap3": _stats(lap3_model[hs_mask]), "ssr": ssr_hs,
              "mse": float(ssr_hs / n_hs) if n_hs else float("nan"), "U_over_D_fit": U / D_fit,
              "mean_lap3_over_U_over_D_fit": float(np.nanmean(lap3[hs_mask]) / (U / D_fit)), "median_lap3_over_U_over_D_fit": float(np.nanmedian(lap3[hs_mask]) / (U / D_fit)),
              "mean_lap11_over_U_over_D_fit": float(np.nanmean(c11[hs_mask]) / (U / D_fit))}
    n_ch = int(channel_opt.sum())
    curv = {"classes": classes, "all_hillslope": all_hs, "n_channel": n_ch, "n_valid": int(prof.n),
            "channel": {"ssr": float(np.nansum(resid2[channel_opt])), "mse": float(np.nanmean(resid2[channel_opt])) if n_ch else float("nan")},
            "mse_all_check": float(np.nanmean(resid2)), "option_iii_set": diag3}
    rec.add_params("curvature_by_class", curv)
    cls_grid = np.zeros(Z0.shape, dtype=np.float64)
    for code, name in ((1, "divergent"), (2, "hollow")):
        cls_grid[rj[cls[name]], ri[cls[name]]] = code
    cls_grid[rj[cls["hilltop"]], ri[cls["hilltop"]]] = 3
    cls_grid[rj[channel_opt], ri[channel_opt]] = 4
    rec.write_grid(td.filled.with_data(cls_grid), f"{tile}_hillslope_classes")
    logger.info("%s: hillslope %d cells (channel %d): divergent %d (%.1f %%), hollow %d (%.1f %%), hilltop %d; mean -lap3 all %.4g vs U/D_fit %.4g (%.2f); "
                "divergent %.4g, hollow %.4g (ratio %.2f); concave %.2f / %.2f [%.0f s]", tile, n_hs, n_ch, classes["divergent"]["n"],
                100 * classes["divergent"]["share_of_hillslope"], classes["hollow"]["n"], 100 * classes["hollow"]["share_of_hillslope"], classes["hilltop"]["n"],
                all_hs["lap3"]["mean"], U / D_fit, all_hs["mean_lap3_over_U_over_D_fit"], classes["divergent"]["lap3"]["mean"], classes["hollow"]["lap3"]["mean"],
                classes["hollow"]["lap3"]["mean"] / classes["divergent"]["lap3"]["mean"] if classes["divergent"]["lap3"]["mean"] else float("nan"),
                classes["divergent"]["lap3"]["frac_concave"], classes["hollow"]["lap3"]["frac_concave"], time.perf_counter() - t0)

    # ------------------------------------------------------------------ 2.2 the three masks on one set of reductions
    lo = np.array([U / mp.K_bounds[1], U / mp.D_bounds[1]])
    hi = np.array([U / mp.K_bounds[0], U / mp.D_bounds[0]])
    ua = unique_areas(A, *ACRIT_RANGE)
    grid = thin_log_uniform(ua, thin_n)
    keep_fixed = {"all": np.ones(prof.n, dtype=bool), "divergent": cls["divergent"], "hollow": cls["hollow"]}
    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in MASKS}
    traj = Trajectory("d_diagnostic_partitioned", ("log10_A_crit", "log10_D", "log10_K", "m", "kind"))

    def _eval(a_c: float, kind: str) -> None:
        r_ = prof.reduce(float(a_c), mp.m)
        for tag in MASKS:
            keep = keep_fixed[tag] | r_.channel_mask        # channel cells of this A_crit stay in every mask
            row: dict[str, Any] = _row(r_, _terms(r_, prof.Z_obs_flat, keep), U, mp.K0_partitioned, mp.D0, lo, hi, D0_B=mp.D0_B)
            row["kind"] = kind
            row["n_misfit"] = int(keep.sum())
            rows[tag].append(row)
            traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(row["D_opt"]), "log10_K": math.log10(row["K_opt"]), "m": mp.m,
                         "kind": f"{tag}_{kind}"}, row["mse_joint"], cached=False)

    for i, a_c in enumerate(grid):
        _eval(a_c, "grid")
        if i % 200 == 0 or i == grid.size - 1:
            msg = " | ".join(f"{t} A_crit {min(rows[t], key=lambda r: r['mse_joint'])['A_crit']:.0f} D {min(rows[t], key=lambda r: r['mse_joint'])['D_opt']:.3e}" for t in MASKS)
            logger.info("d diagnostic %d/%d: %s [%.0f s]", i + 1, grid.size, msg, time.perf_counter() - t0)
    extra: set[float] = set()
    for tag in MASKS:
        j = int(np.argmin([r["mse_joint"] for r in rows[tag]]))
        lo_a, hi_a = grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]
        extra |= set(ua[(ua > lo_a) & (ua < hi_a)].tolist())
    extra -= set(grid.tolist())
    extra_a = np.array(sorted(extra))
    if extra_a.size > refine_max:
        extra_a = thin_log_uniform(extra_a, refine_max)
    for a_c in extra_a:
        _eval(a_c, "refine")
    rec.add_trajectory(traj)
    frames = []
    refits: dict[str, Any] = {}
    for tag in MASKS:
        df = pd.DataFrame(rows[tag]).sort_values("A_crit").reset_index(drop=True)
        df["mask"] = tag
        frames.append(df)
        recs = cast(list[dict[str, Any]], df.to_dict("records"))
        jj = int(np.argmin(df.mse_joint.to_numpy()))
        D = float(df.D_opt[jj])
        refits[tag] = {"joint_optimum": {"A_crit": float(df.A_crit[jj]), "K": float(df.K_opt[jj]), "D": D, "mse": float(df.mse_joint[jj]), "where": df["where"][jj],
                                         "n_misfit": int(df.n_misfit[jj])},
                       "joint_identifiability": identifiability(recs, "mse_joint"), "n_evaluations": int(df.shape[0]),
                       "D_over_D0_A": D / mp.D0, "D_over_D0_B": (D / mp.D0_B) if mp.D0_B else None, "K_over_K0": float(df.K_opt[jj]) / mp.K0_partitioned}
    rec.write_table(pd.concat(frames, ignore_index=True), "acrit_profile")
    refits["all_vs_phase6"] = {"phase6": {"A_crit": A_crit_opt, "K": K_fit, "D": D_fit, "mse": jo["mse"]},
                               "A_crit_rel_diff": refits["all"]["joint_optimum"]["A_crit"] / A_crit_opt - 1, "K_rel_diff": refits["all"]["joint_optimum"]["K"] / K_fit - 1,
                               "D_rel_diff": refits["all"]["joint_optimum"]["D"] / D_fit - 1, "mse_rel_diff": refits["all"]["joint_optimum"]["mse"] / jo["mse"] - 1}
    rec.add_params("refits", refits)

    # ------------------------------------------------------------------ 2.3 the rule, per tile
    D_div = refits["divergent"]["joint_optimum"]["D"]
    gap_fit, gap_div = abs(math.log10(D_fit / mp.D0)), abs(math.log10(D_div / mp.D0))
    mean_ratio = all_hs["mean_lap3_over_U_over_D_fit"]
    hol_over_div = classes["hollow"]["lap3"]["mean"] / classes["divergent"]["lap3"]["mean"] if classes["divergent"]["lap3"]["mean"] else float("nan")
    c_a = bool(abs(mean_ratio - 1) <= RULE["mean_lap_within"])
    c_b = bool(hol_over_div < RULE["hollow_below_fraction_of_divergent"])
    c_c = bool(gap_div <= gap_fit - math.log10(RULE["closer_factor"]))
    verdict = {"a_mean_lap3_over_U_over_D_fit": mean_ratio, "a_pass": c_a, "b_hollow_mean_over_divergent_mean": hol_over_div, "b_pass": c_b,
               "c_log10_gap_fit": gap_fit, "c_log10_gap_divergent": gap_div, "c_gap_reduction_log10": gap_fit - gap_div, "c_pass": c_c,
               "c_pass_strict_half_log_distance": bool(gap_div <= 0.5 * gap_fit), "D_fit": D_fit, "D_divergent": D_div,
               "D_hollow": refits["hollow"]["joint_optimum"]["D"], "D0_A": mp.D0, "D0_B": mp.D0_B, "all_pass": bool(c_a and c_b and c_c)}
    rec.add_params("verdict", verdict)
    rec.add_params("wall_s", time.perf_counter() - t0)
    logger.info("%s verdict: (a) mean -lap3 / (U/D_fit) = %.2f -> %s; (b) hollow/divergent = %.2f -> %s; (c) gap %.3f -> %.3f dec (D_div %.3e, D_fit %.3e, D0_A %.3e) -> %s; all %s",
                tile, mean_ratio, c_a, hol_over_div, c_b, gap_fit, gap_div, D_div, D_fit, mp.D0, c_c, verdict["all_pass"])
    rec.finalize(config_snapshot(td.cfg, {"crop": crop, "thin_n": thin_n, "refine_max": refine_max}))
    return rec
