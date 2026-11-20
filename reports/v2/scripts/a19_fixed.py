#!/usr/bin/env python3
"""Addendum 19, section 5.2 (decision 5) and gate 4.1.3: the field-fixed thresholds on the 11 TTV primary tiles at the site m = 0.33.

PF  [K, D]      area threshold fixed at A_F = median A_3 of the picks (primary), at 7,400 m2 (the Table 1 median, sensitivity) and, labelled,
                at the median A_10 of the picks (the picks sit on hillslope cells: A_cell median 4 m2; see heads.json);
PF3 [D]         the same thresholds with K = K0;
SF  [K, D]      A S^alpha >= C with C fixed at C_F = median of A_3 S_down^alpha over the picks, at alpha = 0, b_hat (rounded to 0.05) and 2
                (`slope_threshold.head_statistic`, unchanged; at alpha = 0, C_F = A_F and SF must equal PF exactly: checked).
Each is one exact reduction of the partitioned model (`PartitionedProfiler.reduce`) and a box-quadratic fit from its Gram terms, on the whole
tile and on the two held-out halves (the addendum 18 greedy split) for the record. The timing of one PF and one SF evaluation (gate 3) is
written; `--sj-pilot` times the slope-threshold module itself on a 20-value profile for the SJ estimate.
usage: a19_fixed.py --tile TTV2 [--sj-pilot]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_common import (  # noqa: E402
    MD_TABLE1_MEDIAN_M2,
    a18_specs,
    channel_heads,
    dump,
    load_json,
    out_path,
    slurm_record,
)

from lemcvt.analyses.sensitivity import _fit, _q, _terms_masked, greedy_basin_halves  # noqa: E402
from lemcvt.analyses.slope_threshold import head_statistic  # noqa: E402
from lemcvt.config import load_tile  # noqa: E402
from lemcvt.optimize.v2 import PartitionedProfiler  # noqa: E402

HOLDOUT_A_C = 1e3


def fits_from_reduction(prof, red, keep, U, K0, lo, hi) -> dict:
    terms = {p: _terms_masked(red, prof.Z_obs_flat, k) for p, k in keep.items()}
    out: dict = {"n_channel": int(red.channel_mask.sum()), "n_hillslope": int(red.n_hillslope), "n_channel_no_chi": int(np.sum(red.channel_mask & ~np.isfinite(red.b))), "wall_s": red.wall_s}
    x0 = U / K0
    for name, kind in (("free_KD", "joint"), ("K_fixed", "fix_x")):
        t = terms["all"]
        x, y, where = _fit(t, kind, lo, hi, x0, None)
        out[name] = {"mse": _q(t, x, y), "K": U / x, "D": U / y, "where": where, "K_over_K0": (U / x) / K0}
        for part, other in (("h1", "h2"), ("h2", "h1")):
            tp = terms[part]
            if tp.get("n", 0) == 0 or not np.isfinite(tp.get("rr", np.nan)):
                continue
            xp, yp, wp = _fit(tp, kind, lo, hi, x0, None)
            out[name][f"fit_{part}"] = {"mse": _q(tp, xp, yp), "K": U / xp, "D": U / yp, "where": wp, f"score_{other}": _q(terms[other], xp, yp)}
        if "fit_h1" in out[name] and "fit_h2" in out[name]:
            out[name]["mse_heldout_mean"] = 0.5 * (out[name]["fit_h1"]["score_h2"] + out[name]["fit_h2"]["score_h1"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    p.add_argument("--sj-pilot", action="store_true")
    a = p.parse_args()
    tile = a.tile
    t0 = time.time()
    out_dir = os.path.dirname(out_path("fixed", f"{tile}.json"))
    slurm_record(out_dir)
    heads = load_json(out_path("heads", "heads.json"))
    fv = heads["field_values"]
    spec = a18_specs()[f"{tile}:primary"]
    td = load_tile(tile)
    mp = td.cfg.model
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    U, m, K0 = float(spec["E"]), float(spec["m"]), float(spec["K0"])
    assert abs(U - td.site.U) < 1e-15 and abs(m - mp.m) < 1e-12 and abs(K0 - mp.K0_partitioned) / K0 < 1e-9, (U, m, K0, mp.m, mp.K0_partitioned)
    lo = np.array([U / mp.K_bounds[1], U / mp.D_bounds[1]])
    hi = np.array([U / mp.K_bounds[0], U / mp.D_bounds[0]])
    half, hinfo = greedy_basin_halves(Z, A, FD, dx, dy, A_c=HOLDOUT_A_C)
    rec: dict = {"tile": tile, "spec": spec, "model": {"m": m, "U": U, "K0": K0, "D0_A": spec.get("D0A"), "D0_B": spec.get("D0B"), "K_bounds": mp.K_bounds, "D_bounds": mp.D_bounds},
                 "split": {k: hinfo[k] for k in ("share_half1_of_split", "share_half2_of_split", "n_basins_edge")}, "field_values": fv, "PF": {}, "SF": {}, "timing": {}}
    # PF / PF3 (area thresholds)
    prof = PartitionedProfiler(Z, A, FD, dx, U, m_values=[m])
    gd = prof.gd
    half_flat = half[gd["rev_j"], gd["rev_i"]]
    keep = {"all": np.ones(prof.n, dtype=bool), "h1": half_flat == 1, "h2": half_flat == 2}
    thresholds = {"A_F": fv["A_F_m2"], "A_F_table1": MD_TABLE1_MEDIAN_M2, "A_F_without_49": fv["A_F_without_49_m2"], "A_F_A10_sensitivity": float(heads["distributions"]["A_10"]["p50"])}
    for name, a_f in thresholds.items():
        t1 = time.perf_counter()
        red = prof.reduce(float(a_f), m)
        r = fits_from_reduction(prof, red, keep, U, K0, lo, hi)
        r["A_crit"] = float(a_f)
        r["n_heads"] = int(channel_heads(np.isfinite(A) & (A >= a_f), FD).sum())
        r["eval_wall_s"] = time.perf_counter() - t1
        rec["PF"][name] = r
        print(f"{tile} PF at {name} = {a_f:.0f} m2: n_channel {r['n_channel']}, heads {r['n_heads']}; PF mse {r['free_KD']['mse']:.3f} K/K0 {r['free_KD']['K_over_K0']:.3f} D {r['free_KD']['D']:.4g} ({r['free_KD']['where']}); "
              f"PF3 mse {r['K_fixed']['mse']:.3f} D {r['K_fixed']['D']:.4g} ({r['K_fixed']['where']}) [{r['eval_wall_s']:.1f} s]", flush=True)
    rec["timing"]["PF_eval_s"] = rec["PF"]["A_F"]["eval_wall_s"]
    # SF (slope-dependent thresholds at fixed C_F)
    valid = np.isfinite(A)
    for al in fv["alphas"]:
        key = f"{al:g}"
        t1 = time.perf_counter()
        hs = head_statistic(Z, A, FD, dx, dy, float(al))
        M = hs["M"]
        t_stat = time.perf_counter() - t1
        prof_s = PartitionedProfiler(Z, A, FD, dx, U, m_values=[m], channel_stat=M[valid]) if al != 0 else prof
        for cname, C_F in (("C_F", fv["C_F"][key]), ("C_F_without_49", fv["C_F_without_49"][key])):
            t2 = time.perf_counter()
            red = prof_s.reduce(float(C_F), m)
            r = fits_from_reduction(prof_s, red, keep, U, K0, lo, hi)
            r["C"] = float(C_F)
            r["alpha"] = float(al)
            ch = np.isfinite(M) & (M >= C_F) if al != 0 else (valid & (A >= C_F))
            r["n_heads"] = int(channel_heads(ch, FD).sum())
            r["eval_wall_s"] = time.perf_counter() - t2
            r["head_statistic_wall_s"] = t_stat
            rec["SF"][f"alpha_{key}:{cname}"] = r
            print(f"{tile} SF alpha {al:g} C_F {C_F:.4g} ({cname}): n_channel {r['n_channel']}, heads {r['n_heads']}; mse {r['free_KD']['mse']:.3f} K/K0 {r['free_KD']['K_over_K0']:.3f} D {r['free_KD']['D']:.4g} ({r['free_KD']['where']}) [{r['eval_wall_s']:.1f} s]", flush=True)
        if al == 0:
            a_, b_ = rec["PF"]["A_F"]["free_KD"]["mse"], rec["SF"]["alpha_0:C_F"]["free_KD"]["mse"]
            rec["alpha0_equals_PF"] = {"mse_PF": a_, "mse_SF0": b_, "rel_diff": abs(a_ / b_ - 1), "n_channel_equal": rec["PF"]["A_F"]["n_channel"] == rec["SF"]["alpha_0:C_F"]["n_channel"], "pass": bool(abs(a_ / b_ - 1) < 1e-12)}
    rec["timing"]["SF_eval_s"] = rec["SF"][f"alpha_{fv['alphas'][-1]:g}:C_F"]["eval_wall_s"]
    rec["timing"]["head_statistic_s"] = rec["SF"][f"alpha_{fv['alphas'][-1]:g}:C_F"]["head_statistic_wall_s"]
    if a.sj_pilot:
        from lemcvt.analyses.slope_threshold import run_slope_threshold
        t1 = time.perf_counter()
        rr = run_slope_threshold(tile, 2.0, thin_n=20, refine_max=0, run_id=f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_a19_sj_pilot_{tile}_a2")
        wall = time.perf_counter() - t1
        pj = load_json(str(rr.dir / "params.json"))
        n_ev = pj["result"]["n_evaluations"]
        rec["timing"]["SJ_pilot"] = {"run": rr.run_id, "n_evaluations": n_ev, "wall_s": wall, "s_per_evaluation": wall / n_ev, "module_wall_s": pj.get("wall_s"),
                                     "expected_evaluations_full": 2000 + 400, "expected_h_full_run": (wall / n_ev) * 2400 / 3600.0}
        print(f"{tile} SJ pilot: {n_ev} evaluations in {wall:.0f} s -> {wall / n_ev:.2f} s each; a full run (2,400) ~ {rec['timing']['SJ_pilot']['expected_h_full_run']:.2f} h", flush=True)
    rec["wall_s"] = time.time() - t0
    dump(rec, out_path("fixed", f"{tile}.json"))


if __name__ == "__main__":
    main()
