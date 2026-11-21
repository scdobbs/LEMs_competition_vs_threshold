#!/usr/bin/env python3
"""Addendum 20, section 5.2 (decision 6a, 6b): the field-fixed thresholds on one TTV primary tile at the site m = 0.33.

PF  [K, D]   area threshold fixed at A_F (median A_P10 of the snapped picks, primary), 7,400 m2 (Table 1 median), 2,848 m2 (addendum 19
             median A_10) and 611 m2 (addendum 19 median A_3); one evaluation each;
PF3 [D]      the same four thresholds with K = K0;
SF  [K, D]   A S^alpha >= C_F with C_F = median A_P10 S_down^alpha over the picks, at alpha = 0, 1.65, 2 (`slope_threshold.head_statistic`,
             unchanged); at alpha = 0, C_F = A_F and SF must equal PF at A_F exactly (checked, rel. 1e-12).
Each is one exact reduction of the partitioned model (`PartitionedProfiler.reduce`) and the box-quadratic fit from its Gram terms (the
addendum 19 `fits_from_reduction`), also on the addendum 18 greedy halves for the record. Channel-head counts per model.
usage: a20_fixed.py --tile TTV2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_fixed import HOLDOUT_A_C, fits_from_reduction  # noqa: E402
from a20_common import (  # noqa: E402
    A19_A3_MEDIAN_M2,
    A19_A10_MEDIAN_M2,
    MD_TABLE1_MEDIAN_M2,
    a18_specs,
    channel_heads,
    dump,
    load_json,
    out_path,
    slurm_record,
)

from lemcvt.analyses.sensitivity import greedy_basin_halves  # noqa: E402
from lemcvt.analyses.slope_threshold import head_statistic  # noqa: E402
from lemcvt.config import load_tile  # noqa: E402
from lemcvt.optimize.v2 import PartitionedProfiler  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    a = p.parse_args()
    tile = a.tile
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("fixed", f"{tile}.json")))
    fv = load_json(out_path("heads", "heads.json"))["field_values"]
    spec = a18_specs()[f"{tile}:primary"]
    td = load_tile(tile)
    mp = td.cfg.model
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    U, m, K0 = float(spec["E"]), float(spec["m"]), float(spec["K0"])
    assert abs(U - td.site.U) < 1e-15 and abs(m - mp.m) < 1e-12 and abs(K0 - mp.K0_partitioned) / K0 < 1e-9, (U, m, K0, mp.m, mp.K0_partitioned)
    lo = np.array([U / mp.K_bounds[1], U / mp.D_bounds[1]])
    hi = np.array([U / mp.K_bounds[0], U / mp.D_bounds[0]])
    half, hinfo = greedy_basin_halves(Z, A, FD, dx, dy, A_c=HOLDOUT_A_C)
    rec: dict = {"tile": tile, "spec": spec, "model": {"m": m, "U": U, "K0": K0, "K_bounds": mp.K_bounds, "D_bounds": mp.D_bounds},
                 "split": {k: hinfo[k] for k in ("share_half1_of_split", "share_half2_of_split", "n_basins_edge")}, "field_values": fv, "PF": {}, "SF": {}}
    prof = PartitionedProfiler(Z, A, FD, dx, U, m_values=[m])
    gd = prof.gd
    half_flat = half[gd["rev_j"], gd["rev_i"]]
    keep = {"all": np.ones(prof.n, dtype=bool), "h1": half_flat == 1, "h2": half_flat == 2}
    thresholds = {"A_F": fv["A_F_m2"], "A_F_table1": MD_TABLE1_MEDIAN_M2, "A_F_A10": A19_A10_MEDIAN_M2, "A_F_A3": A19_A3_MEDIAN_M2, "A_F_without_49": fv["A_F_without_49_m2"]}
    valid = np.isfinite(A)
    for name, a_f in thresholds.items():
        t1 = time.perf_counter()
        red = prof.reduce(float(a_f), m)
        r = fits_from_reduction(prof, red, keep, U, K0, lo, hi)
        r["A_crit"] = float(a_f)
        r["n_heads"] = int(channel_heads(valid & (A >= a_f), FD).sum())
        r["eval_wall_s"] = time.perf_counter() - t1
        rec["PF"][name] = r
        print(f"{tile} PF {name} = {a_f:.0f} m2: n_channel {r['n_channel']}, heads {r['n_heads']}; PF mse {r['free_KD']['mse']:.3f} K/K0 {r['free_KD']['K_over_K0']:.3f} "
              f"D {r['free_KD']['D']:.4g} ({r['free_KD']['where']}); PF3 mse {r['K_fixed']['mse']:.3f} D {r['K_fixed']['D']:.4g} ({r['K_fixed']['where']}) [{r['eval_wall_s']:.1f} s]", flush=True)
    for al in fv["alphas_SF"]:
        key = f"{al:g}"
        hs = head_statistic(Z, A, FD, dx, dy, float(al))
        M = hs["M"]
        prof_s = PartitionedProfiler(Z, A, FD, dx, U, m_values=[m], channel_stat=M[valid]) if al != 0 else prof
        for cname, C_F in (("C_F", fv["C_F"][key]), ("C_F_without_49", fv["C_F_without_49"][key])):
            t2 = time.perf_counter()
            red = prof_s.reduce(float(C_F), m)
            r = fits_from_reduction(prof_s, red, keep, U, K0, lo, hi)
            r["C"] = float(C_F)
            r["alpha"] = float(al)
            ch = (np.isfinite(M) & (M >= C_F)) if al != 0 else (valid & (A >= C_F))
            r["n_heads"] = int(channel_heads(ch, FD).sum())
            r["eval_wall_s"] = time.perf_counter() - t2
            rec["SF"][f"alpha_{key}:{cname}"] = r
            print(f"{tile} SF alpha {key} C {C_F:.4g} ({cname}): n_channel {r['n_channel']}, heads {r['n_heads']}; mse {r['free_KD']['mse']:.3f} K/K0 {r['free_KD']['K_over_K0']:.3f} "
                  f"D {r['free_KD']['D']:.4g} ({r['free_KD']['where']}) [{r['eval_wall_s']:.1f} s]", flush=True)
        if al == 0:
            a_, b_ = rec["PF"]["A_F"]["free_KD"]["mse"], rec["SF"]["alpha_0:C_F"]["free_KD"]["mse"]
            rec["alpha0_equals_PF"] = {"mse_PF": a_, "mse_SF0": b_, "rel_diff": abs(a_ / b_ - 1), "pass": bool(abs(a_ / b_ - 1) < 1e-12)}
    rec["wall_s"] = time.time() - t0
    dump(rec, out_path("fixed", f"{tile}.json"))


if __name__ == "__main__":
    main()
