#!/usr/bin/env python3
"""Addendum 21, Part B item 4: SJ-C, the headline partitioned model with K, D and the map-scale head criterion all free (decision 1.5).

At alpha = 2 the channel network of the 1 m model is the 12 m criterion network at C (decision 1.3: 12 m heads where A_12 S_12^2 >= C
first holds, seeded into the 1 m grid). Profile over the unique values of the 12 m head statistic M among the tile's candidate cells
(those with an entry into the tile), between the C values whose 1 m networks have as many cells as A >= 1e3 and A >= 1e5 m2; thinned to
at most 400 values, refined around the minima (all unique values between the grid neighbours of the whole-tile and the two half
minima, at most 400); at each C one exact reduction (the mask route) and the closed-form box minimum in (U/K, U/D) on the whole tile and
on the addendum 18 basin halves (the hold-out: fit C, K, D on one half, score on the other). The optimum's surface then goes through the
decision 1.4 extraction. `--pilot N` evaluates N grid values only and prints the time per evaluation and the full-run estimate.
Outputs <run>/sjc/<tile>.json, <tile>_c_profile.parquet, <tile>_heads.npz, <tile>_residual.npz (pilot: <tile>_pilot.json).
usage: a21_sjc.py --tile TTV2 [--pilot 20]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import (  # noqa: E402
    ALPHA_PRIMARY,
    HOLDOUT_A_C,
    M_SITE,
    RESCALE,
    SJC_A_RANGE,
    SJC_REFINE_MAX,
    SJC_THIN,
    U_NEW,
    a18_specs,
    channel_heads,
    dump,
    extract_and_measure,
    fits_from_reduction,
    json_ready,
    load_json,
    mask_from_entries,
    md_lines_mask,
    md_polygon_mask,
    metre_tile,
    out_path,
    partitioned_surface,
    picks_on_tile,
    residual_stats,
    slurm_record,
)
from a21_tile import reduce_mask  # noqa: E402

from lemcvt.analyses.sensitivity import greedy_basin_halves  # noqa: E402
from lemcvt.optimize.v2 import PartitionedProfiler, identifiability, thin_log_uniform  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    p.add_argument("--pilot", type=int, default=0)
    p.add_argument("--thin", type=int, default=SJC_THIN)
    p.add_argument("--refine-max", type=int, default=SJC_REFINE_MAX)
    a = p.parse_args()
    tile = a.tile
    t0 = time.time()
    tag = f"{tile}_pilot" if a.pilot else tile
    slurm_record(os.path.dirname(out_path("sjc", f"{tag}.json")))
    td = metre_tile(tile)
    mp = td.cfg.model
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    U, m = float(mp.U), float(mp.m)
    assert abs(U - U_NEW) < 1e-15 and abs(m - M_SITE) < 1e-12
    spec = a18_specs()[f"{tile}:primary"]
    K0 = float(mp.K0_partitioned)
    Kb = (mp.K_bounds[0] * RESCALE, mp.K_bounds[1] * RESCALE)
    Db = (mp.D_bounds[0] * RESCALE, mp.D_bounds[1] * RESCALE)
    lo = np.array([U / Kb[1], U / Db[1]])
    hi = np.array([U / Kb[0], U / Db[0]])
    sd = np.load(out_path("grid12", f"{tile}_seeds.npz"))
    fv = load_json(out_path("grid12", "grid12.json"))["field_values"]["12m"]
    M, Md = sd[f"cand_M_a{ALPHA_PRIMARY:g}"], sd[f"cand_Mdon_a{ALPHA_PRIMARY:g}"]
    er, ec = sd["entry_r"], sd["entry_c"]
    has_entry = er >= 0

    def mask_at(C: float) -> np.ndarray:
        use = np.isfinite(M) & (M >= C) & ~(Md >= C) & has_entry
        return mask_from_entries(er[use], ec[use], A, FD)

    uv_all = np.unique(M[has_entry & np.isfinite(M)])
    valid = np.isfinite(A)
    n_lo, n_hi = int(np.sum(A[valid] >= SJC_A_RANGE[0])), int(np.sum(A[valid] >= SJC_A_RANGE[1]))
    t1 = time.perf_counter()
    counts: dict[float, int] = {}

    def n_at(C: float) -> int:
        if C not in counts:
            counts[C] = int(mask_at(C).sum())
        return counts[C]

    def search(target: int, want_ge: bool) -> float:
        """The largest C with n(C) >= target (want_ge) or the smallest C with n(C) <= target, by bisection on the unique values."""
        lo_i, hi_i = 0, uv_all.size - 1
        while hi_i - lo_i > 1:
            mid = (lo_i + hi_i) // 2
            n = n_at(float(uv_all[mid]))
            if (n >= target) if want_ge else (n > target):
                lo_i = mid
            else:
                hi_i = mid
        if want_ge:
            return float(uv_all[lo_i])
        return float(uv_all[hi_i])

    C_lo, C_hi = search(n_lo, True), search(n_hi, False)
    if C_hi <= C_lo:
        C_hi = float(uv_all[min(np.searchsorted(uv_all, C_lo) + 1, uv_all.size - 1)])
    uv = uv_all[(uv_all >= C_lo) & (uv_all <= C_hi)]
    grid = thin_log_uniform(uv, a.pilot if a.pilot else a.thin)
    setup = {"C_range": [C_lo, C_hi], "n_channel_targets": [n_lo, n_hi], "n_channel_at_C_lo": n_at(C_lo), "n_channel_at_C_hi": n_at(C_hi), "n_unique_all": int(uv_all.size),
             "n_unique_in_range": int(uv.size), "n_grid": int(grid.size), "thin": a.thin, "refine_max": a.refine_max, "n_bisection_evals": len(counts), "range_wall_s": time.perf_counter() - t1,
             "n_candidates": int(M.size), "n_candidates_with_entry": int(has_entry.sum()),
             "monotone_check": bool(all(counts[c1] >= counts[c2] for c1, c2 in zip(sorted(counts)[:-1], sorted(counts)[1:], strict=True)))}
    print(f"{tile}: C range [{C_lo:.4g}, {C_hi:.4g}] (cells {setup['n_channel_at_C_lo']} .. {setup['n_channel_at_C_hi']}, targets {n_lo} .. {n_hi}); {uv.size} unique values, grid {grid.size} "
          f"[{time.time() - t0:.0f} s]", flush=True)
    prof = PartitionedProfiler(Z, A, FD, dx, U, m_values=[m])
    gd = prof.gd
    half, hinfo = greedy_basin_halves(Z, A, FD, dx, dy, A_c=HOLDOUT_A_C)
    half_flat = half[gd["rev_j"], gd["rev_i"]]
    keep = {"all": np.ones(prof.n, dtype=bool), "h1": half_flat == 1, "h2": half_flat == 2}
    rows: list[dict] = []
    done: set[float] = set()
    walls: list[float] = []

    def _eval(C: float, kind: str) -> None:
        C = float(C)
        if C in done:
            return
        t2 = time.perf_counter()
        mk = mask_at(C)
        red = reduce_mask(prof, mk, m)
        r = fits_from_reduction(prof, red, keep, U, K0, lo, hi)
        f = r["free_KD"]
        row = {"C": C, "kind": kind, "n_channel": r["n_channel"], "n_hillslope": r["n_hillslope"], "n_channel_no_chi": r["n_channel_no_chi"], "n_heads_12m": int(np.sum(np.isfinite(M) & (M >= C) & ~(Md >= C) & has_entry)),
               "mse_joint": f["mse"], "K_opt": f["K"], "D_opt": f["D"], "where": f["where"], "mse_K_fixed": r["K_fixed"]["mse"], "D_K_fixed": r["K_fixed"]["D"]}
        for part, other in (("h1", "h2"), ("h2", "h1")):
            fp = f.get(f"fit_{part}")
            if fp:
                row.update({f"mse_{part}_fit": fp["mse"], f"K_{part}": fp["K"], f"D_{part}": fp["D"], f"where_{part}": fp["where"], f"mse_{other}_at_{part}": fp[f"score_{other}"]})
        row["wall_s"] = time.perf_counter() - t2
        rows.append(row)
        done.add(C)
        walls.append(row["wall_s"])

    for i, C in enumerate(grid):
        _eval(C, "grid")
        if i % 50 == 0 or i == grid.size - 1:
            b = min(rows, key=lambda r: r["mse_joint"])
            print(f"{tile}: {i + 1}/{grid.size} best C {b['C']:.4g} (n_channel {b['n_channel']}, heads {b['n_heads_12m']}) K {b['K_opt']:.3e} D {b['D_opt']:.3e} mse {b['mse_joint']:.3f}; "
                  f"{np.mean(walls):.1f} s/eval [{time.time() - t0:.0f} s]", flush=True)
    rec: dict = {"tile": tile, "U": U, "m": m, "K0": K0, "bounds_rescaled": {"K": list(Kb), "D": list(Db)}, "spec": spec, "setup": setup, "split": {k: hinfo[k] for k in ("share_half1_of_split", "share_half2_of_split", "n_basins_edge")}}
    if a.pilot:
        s_eval = float(np.mean(walls))
        est_h = s_eval * (a.thin + a.refine_max) / 3600.0
        rec["pilot"] = {"n_evaluations": len(rows), "s_per_evaluation": s_eval, "expected_evaluations_full": a.thin + a.refine_max, "expected_h_full_run": est_h, "rows": rows}
        rec["wall_s"] = time.time() - t0
        dump(json_ready(rec), out_path("sjc", f"{tag}.json"))
        print(f"{tile} SJ-C pilot: {len(rows)} evaluations at {s_eval:.1f} s each -> a full run ({a.thin} + <= {a.refine_max}) ~ {est_h:.2f} h", flush=True)
        return
    # refinement around the minima of the whole-tile and the two half objectives
    extra: set[float] = set()
    for col in ("mse_joint", "mse_h1_fit", "mse_h2_fit"):
        cand = [r for r in rows if col in r]
        if not cand:
            continue
        j = int(np.argmin([r[col] for r in cand]))
        cj = cand[j]["C"]
        gi = int(np.searchsorted(grid, cj))
        lo_a, hi_a = grid[max(gi - 1, 0)], grid[min(gi + 1, grid.size - 1)]
        extra |= set(uv[(uv > lo_a) & (uv < hi_a)].tolist())
    extra -= done
    extra_a = np.array(sorted(extra))
    n_targets = int(extra_a.size)
    if extra_a.size > a.refine_max:
        extra_a = thin_log_uniform(extra_a, a.refine_max)
    for C in extra_a:
        _eval(C, "refine")
    rec["refine"] = {"n_targets": n_targets, "n_evaluated": int(extra_a.size), "capped": bool(n_targets > a.refine_max)}
    df = pd.DataFrame(rows).sort_values("C").reset_index(drop=True)
    df.to_parquet(out_path("sjc", f"{tile}_c_profile.parquet"), index=False)
    recs = df.to_dict("records")
    jj = int(np.argmin(df.mse_joint.to_numpy()))
    opt = {"C": float(df.C[jj]), "K": float(df.K_opt[jj]), "D": float(df.D_opt[jj]), "mse": float(df.mse_joint[jj]), "where": df["where"][jj], "n_channel": int(df.n_channel[jj]),
           "n_channel_no_chi": int(df.n_channel_no_chi[jj]), "n_heads_12m": int(df.n_heads_12m[jj]), "K_over_K0": float(df.K_opt[jj]) / K0, "index": jj, "n_params": 3}
    hold = None
    if "mse_h1_fit" in df and "mse_h2_fit" in df:
        j1, j2 = int(np.argmin(df.mse_h1_fit.to_numpy())), int(np.argmin(df.mse_h2_fit.to_numpy()))
        hold = {"h1_fit": {"C": float(df.C[j1]), "K": float(df.K_h1[j1]), "D": float(df.D_h1[j1]), "mse_fit": float(df.mse_h1_fit[j1]), "mse_heldout_h2": float(df.mse_h2_at_h1[j1])},
                "h2_fit": {"C": float(df.C[j2]), "K": float(df.K_h2[j2]), "D": float(df.D_h2[j2]), "mse_fit": float(df.mse_h2_fit[j2]), "mse_heldout_h1": float(df.mse_h1_at_h2[j2])},
                "mse_heldout_mean": 0.5 * (float(df.mse_h2_at_h1[j1]) + float(df.mse_h1_at_h2[j2]))}
    rec["joint_optimum"] = opt
    rec["identifiability"] = identifiability(recs, "mse_joint", key="C")
    rec["holdout"] = hold
    rec["n_evaluations"] = int(df.shape[0])
    rec["s_per_evaluation"] = float(np.mean(walls))
    rec["D_at_bound"] = bool(np.isclose(opt["D"], Db[0], rtol=1e-9) or np.isclose(opt["D"], Db[1], rtol=1e-9))
    print(f"{tile} SJ-C: C {opt['C']:.4g} (n_channel {opt['n_channel']}, 12 m heads {opt['n_heads_12m']}), K {opt['K']:.3e} (K/K0 {opt['K_over_K0']:.2f}), D {opt['D']:.3e} ({opt['where']}), "
          f"mse {opt['mse']:.3f}; 1 % interval {rec['identifiability']['level_0.01']['intervals']}; hold-out {hold['mse_heldout_mean'] if hold else None} [{time.time() - t0:.0f} s]", flush=True)
    rec["wall_s_profile"] = time.time() - t0
    dump(json_ready(rec), out_path("sjc", f"{tile}.json"))          # the profile and optimum are on disk before the extraction
    # the optimum's surface through the decision 1.4 extraction
    mk = mask_at(opt["C"])
    red = reduce_mask(prof, mk, m)
    Zs, holes = partitioned_surface(prof, red, opt["K"], opt["D"], Z)
    poly = md_polygon_mask(td)
    lines = md_lines_mask(td) if poly.any() else np.zeros(Z.shape, dtype=bool)
    pick_rc, pick_ids = picks_on_tile(td)
    e_rec, arrs = extract_and_measure(Zs, dx, fv, pick_rc, pick_ids, poly, lines)
    e_rec["n_holes_1m"] = holes
    e_rec["residual"] = residual_stats(Zs, Z)
    rec["extraction"] = e_rec
    rec["model"] = opt | {"n_heads_1m": int(channel_heads(mk, FD).sum()), "surface_holes": holes}
    np.savez_compressed(out_path("sjc", f"{tile}_heads.npz"), **{f"SJ_C_{k}": v for k, v in arrs.items()}, net_SJ_C_1m=np.packbits(mk))
    np.savez_compressed(out_path("sjc", f"{tile}_residual.npz"), SJ_C=(Zs - Z).astype(np.float32))
    rec["wall_s"] = time.time() - t0
    dump(json_ready(rec), out_path("sjc", f"{tile}.json"))
    a2 = e_rec["a2"]
    print(f"{tile} SJ-C extraction: a2 channels {a2['n_channel_12m']} heads {a2['n_heads_12m']} (poly {a2['n_heads_in_polygon']}/{a2['n_picks_in_polygon']}), pick dist {a2['pick_dist_m']['median']} [{time.time() - t0:.0f} s]", flush=True)
    _ = math


if __name__ == "__main__":
    main()
