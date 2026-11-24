#!/usr/bin/env python3
"""Addendum 22, the models whose refit needs a profile: AD (over r = K/D), the split model with the area x slope^2 threshold fitted
(SJ-C, over C) and the split model with the area threshold fitted (P4, over A_crit), each with the misfit over the M&D polygon only.

At every profile point the reduction is made once and both Gram-term sets are formed (every cell; polygon cells), so the whole-tile
objective is profiled on the same points: its minimum must reproduce the addendum 21 fit (the decision 2.1 mask check; the addendum 21
optimum is added to the grid so that the check is exact when the grids agree) and the whole-tile misfit of the polygon fit is read off.
  - AD: the addendum 18 procedure (200 log-uniform r between the bounds' ratio limits, closed-form y = U/D on the polygon cells, bounded
    Brent in log10 r around the grid minimum), run for both objectives from the same LU factorisations.
  - SJ-C: the addendum 21 procedure (unique 12 m head statistics among the tile's candidate cells between the C values whose 1 m networks
    have as many cells as A >= 1e3 and A >= 1e5 m2; thinned to 400; refined around the minima, at most 400 more).
  - P4: the addendum 18 procedure (unique drainage areas in [1e2, 1e5] m2 thinned to 2000; staircase refinement around the minima, at most
    1000 more; the 1 % fill step is not run).
The optimum's surface goes through the 12 m products of decision 2.3. `--pilot N` evaluates N grid points and prints the time per
evaluation and the estimate for the full run.
Outputs <run>/profiles/<model>_<tile>.json, _profile.parquet, _products.npz (pilot: <model>_<tile>_pilot.json).
usage: a22_profile.py --model SJ_C --tile TTV2 [--pilot 20]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a22_common import (  # noqa: E402
    AD_N_GRID,
    ALPHA_PRIMARY,
    BLOCK_M,
    C_F,
    MASK_CHECK_TOL,
    P4_ACRIT_RANGE,
    P4_REFINE_MAX,
    P4_THIN,
    SJC_REFINE_MAX,
    SJC_THIN,
    a21_path,
    coarse_products,
    dump,
    fit_terms,
    json_ready,
    keep_masks,
    mask_from_entries,
    mse_at,
    out_path,
    partitioned_surface,
    poly12_centres,
    reduce_mask,
    slurm_record,
    terms_of,
    tile_setup,
)

from lemcvt.models.ad import advection_diffusion_model  # noqa: E402
from lemcvt.models.common import extract_grid_data  # noqa: E402
from lemcvt.optimize.v2 import ADProfiler, PartitionedProfiler, identifiability, thin_log_uniform, unique_areas  # noqa: E402


def rel(a: float, b: float) -> float:
    return abs(a / b - 1.0) if b else float("inf")


def products_of(name: str, Zs: np.ndarray, dx: float, poly: np.ndarray, tag: str) -> dict[str, Any]:
    nb = Zs.shape[0] // BLOCK_M
    P = coarse_products(Zs, dx)
    ch2 = np.isfinite(P["M_a2"]) & (P["M_a2"] >= C_F)
    hd2 = ch2 & ~(P["Mdon_a2"] >= C_F)
    inside = poly12_centres(poly, nb)
    np.savez_compressed(out_path("profiles", f"{tag}_products.npz"), **{f"{name}_{k}": v for k, v in P.items()})
    return {"n_valid_12m": int(np.isfinite(P["Z12"]).sum()), "n_channel_a2": int(ch2.sum()), "n_heads_a2": int(hd2.sum()), "n_heads_a2_in_polygon": int((hd2 & inside).sum())}


# ----------------------------------------------------------------------------- AD

def run_ad(S: dict, tag: str, pilot: int, t0: float) -> dict[str, Any]:
    Z, A, FD, dx, U, m = S["Z"], S["A"], S["FD"], S["dx"], S["U"], S["m"]
    Kb, Db = S["Kb"], S["Db"]
    poly = S["poly"]
    adm = advection_diffusion_model(Z, A, FD, dx, m)
    gd = extract_grid_data(Z, A, FD, dx)
    obs_all = Z[gd["rev_j"], gd["rev_i"]].astype(float)
    obs_poly = np.where(poly[gd["rev_j"], gd["rev_i"]], obs_all, np.nan)
    pa, pp = ADProfiler(adm, obs_all, U), ADProfiler(adm, obs_poly, U)
    r_lo, r_hi = Kb[0] / Db[1], Kb[1] / Db[0]
    grid = np.logspace(math.log10(r_lo), math.log10(r_hi), AD_N_GRID)
    a21 = S["a21"]["rescaled"]["AD"]
    rows: list[dict[str, Any]] = []
    cache: dict[float, dict[str, Any]] = {}
    walls: list[float] = []

    def evaluate(r: float, kind: str) -> dict[str, Any]:
        r = float(r)
        if r in cache:
            return cache[r]
        red = pa.reduce(r)
        D_lo, D_hi = max(Db[0], Kb[0] / r), min(Db[1], Kb[1] / r)
        rec = {"r": r, "kind": kind, "wall_s": red["wall_s"], "feasible": bool(D_lo <= D_hi)}
        if rec["feasible"]:
            for lab, prof in (("all", pa), ("poly", pp)):
                y, mse, where = prof.best_y(red, U / D_hi, U / D_lo)
                rec.update({f"mse_{lab}": mse, f"D_{lab}": U / y, f"K_{lab}": r * U / y, f"where_{lab}": where})
            rec["mse_all_at_poly"] = pa.mse_at(red, U / rec["D_poly"])
            rec["mse_poly_at_all"] = pp.mse_at(red, U / rec["D_all"])
        else:
            rec.update({"mse_all": float("inf"), "mse_poly": float("inf")})
        cache[r] = rec
        rows.append(rec)
        walls.append(red["wall_s"])
        return rec

    n_do = pilot if pilot else grid.size
    for i, r in enumerate(grid[:n_do]):
        evaluate(r, "grid")
        if i % 25 == 0 or i == n_do - 1:
            b = min((q for q in rows if q["feasible"]), key=lambda q: q["mse_poly"], default=None)
            print(f"{tag}: {i + 1}/{n_do} r {r:.3e}; best polygon mse {b['mse_poly'] if b else None}; {np.mean(walls):.1f} s/eval [{time.time() - t0:.0f} s]", flush=True)
    rec: dict[str, Any] = {"model": "AD", "tile": S["a21"]["tile"], "r_range": [r_lo, r_hi], "n_grid": AD_N_GRID, "s_per_evaluation": float(np.mean(walls))}
    if pilot:
        rec["pilot"] = {"n_evaluations": len(rows), "expected_evaluations_full": AD_N_GRID + 30, "expected_h_full_run": float(np.mean(walls)) * (AD_N_GRID + 30) / 3600.0}
        print(f"{tag} pilot: {len(rows)} evaluations at {rec['s_per_evaluation']:.1f} s each -> full run ~ {rec['pilot']['expected_h_full_run']:.2f} h", flush=True)
        return rec
    from scipy.optimize import minimize_scalar
    opt: dict[str, Any] = {}
    for lab in ("all", "poly"):
        feas = sorted((q for q in rows if q["feasible"] and q["kind"] == "grid"), key=lambda q: q["r"])
        jj = int(np.argmin([q[f"mse_{lab}"] for q in feas]))
        lg = np.log10([q["r"] for q in feas])
        a_, b_ = lg[max(jj - 1, 0)], lg[min(jj + 1, len(lg) - 1)]
        res = minimize_scalar(lambda lr: evaluate(10 ** lr, f"brent_{lab}")[f"mse_{lab}"], bounds=(a_, b_), method="bounded", options={"xatol": 1e-6})
        fin = evaluate(10 ** res.x, f"brent_{lab}")
        best = fin if fin[f"mse_{lab}"] <= feas[jj][f"mse_{lab}"] else feas[jj]
        opt[lab] = {"r": best["r"], "K": best[f"K_{lab}"], "D": best[f"D_{lab}"], "mse": best[f"mse_{lab}"], "where": best[f"where_{lab}"], "brent_nfev": int(res.nfev),
                    "mse_other": best["mse_poly_at_all"] if lab == "all" else best["mse_all_at_poly"]}
    chk = {"a21_mse": a21["mse"], "all_cells_mse": opt["all"]["mse"], "rel_diff_mse": rel(opt["all"]["mse"], a21["mse"]), "rel_diff_K": rel(opt["all"]["K"], a21["K"]), "rel_diff_D": rel(opt["all"]["D"], a21["D"])}
    chk["pass"] = bool(chk["rel_diff_mse"] <= MASK_CHECK_TOL)
    chk["all_cells_min_below_a21"] = bool(opt["all"]["mse"] < a21["mse"] * (1 - MASK_CHECK_TOL))
    rec.update({"optimum_poly": opt["poly"], "optimum_all": opt["all"], "mask_check": chk, "n_evaluations": len(rows),
                "D_at_bound_poly": bool(np.isclose(opt["poly"]["D"], Db[0], rtol=1e-9) or np.isclose(opt["poly"]["D"], Db[1], rtol=1e-9)),
                "K_over_K0_poly": opt["poly"]["K"] / S["K0"]})
    grid_rows = [{"r": q["r"], "mse": q["mse_poly"], "K_opt": q["K_poly"], "D_opt": q["D_poly"]} for q in rows if q["feasible"]]
    rec["identifiability_poly"] = identifiability(grid_rows, "mse", key="r")
    pd.DataFrame(rows).sort_values("r").to_parquet(out_path("profiles", f"{tag}_profile.parquet"), index=False)
    print(f"{tag}: polygon optimum K {opt['poly']['K']:.3e} (K/K0 {rec['K_over_K0_poly']:.2f}) D {opt['poly']['D']:.4g} ({opt['poly']['where']}) mse_poly {opt['poly']['mse']:.2f}, whole tile at it {opt['poly']['mse_other']:.2f}; "
          f"all-cells optimum mse {opt['all']['mse']:.4f} vs a21 {a21['mse']:.4f} (rel {chk['rel_diff_mse']:.1e}, {'PASS' if chk['pass'] else 'FAIL'}) [{time.time() - t0:.0f} s]", flush=True)
    red = pa.reduce(opt["poly"]["r"])
    flat = red["h"] + (U / opt["poly"]["D"]) * red["g"]
    Zs = np.full(Z.shape, np.nan)
    Zs[gd["rev_j"], gd["rev_i"]] = flat
    rec["products"] = products_of("AD", Zs, dx, poly, tag)
    return rec


# ----------------------------------------------------------------------------- the partitioned profiles (SJ-C over C, P4 over A_crit)

def run_partitioned(S: dict, model: str, tag: str, pilot: int, t0: float) -> dict[str, Any]:
    Z, A, FD, dx, U, m, K0, lo, hi = S["Z"], S["A"], S["FD"], S["dx"], S["U"], S["m"], S["K0"], S["lo"], S["hi"]
    poly = S["poly"]
    prof = PartitionedProfiler(Z, A, FD, dx, U, m_values=[m])
    keep = keep_masks(prof, poly)
    rec: dict[str, Any] = {"model": model, "tile": S["a21"]["tile"], "U": U, "m": m, "K0": K0, "bounds_rescaled": {"K": list(S["Kb"]), "D": list(S["Db"])}}
    if model == "SJ_C":
        sd = np.load(a21_path("grid12", f"{S['a21']['tile']}_seeds.npz"))
        M, Md = sd[f"cand_M_a{ALPHA_PRIMARY:g}"], sd[f"cand_Mdon_a{ALPHA_PRIMARY:g}"]
        er, ec = sd["entry_r"], sd["entry_c"]
        has_entry = er >= 0

        def mask_at(C: float) -> np.ndarray:
            use = np.isfinite(M) & (M >= C) & ~(Md >= C) & has_entry
            return mask_from_entries(er[use], ec[use], A, FD)

        def reduce_at(C: float):
            return reduce_mask(prof, mask_at(C), m)

        def n_heads_at(C: float) -> int:
            return int(np.sum(np.isfinite(M) & (M >= C) & ~(Md >= C) & has_entry))

        a21_setup = S["a21_sjc"]["setup"]
        C_lo, C_hi = a21_setup["C_range"]                      # the addendum 21 range (the same seeds, the same targets)
        uv_all = np.unique(M[has_entry & np.isfinite(M)])
        uv = uv_all[(uv_all >= C_lo) & (uv_all <= C_hi)]
        grid = thin_log_uniform(uv, SJC_THIN)
        a21_opt = S["a21_sjc"]["joint_optimum"]
        thin_n, refine_max, key, a21_thr = SJC_THIN, SJC_REFINE_MAX, "C", float(a21_opt["C"])
        rec["setup"] = {"C_range": [C_lo, C_hi], "n_unique_in_range": int(uv.size), "n_grid": int(grid.size), "a21_C_range_reused": True}
    else:
        uv = unique_areas(A, *P4_ACRIT_RANGE)
        grid = thin_log_uniform(uv, P4_THIN)

        def reduce_at(a_c: float):
            return prof.reduce(float(a_c), m)

        def n_heads_at(a_c: float) -> int:
            return -1

        a21_opt = S["a21"]["rescaled"]["P4"]
        thin_n, refine_max, key, a21_thr = P4_THIN, P4_REFINE_MAX, "A_crit", float(a21_opt["A_crit"])
        rec["setup"] = {"A_crit_range": list(P4_ACRIT_RANGE), "n_unique_in_range": int(uv.size), "n_grid": int(grid.size)}
    rows: list[dict[str, Any]] = []
    done: set[float] = set()
    walls: list[float] = []

    def _eval(v: float, kind: str) -> None:
        v = float(v)
        if v in done:
            return
        t2 = time.perf_counter()
        red = reduce_at(v)
        T = {k: terms_of(red, prof.Z_obs_flat, kk) for k, kk in keep.items()}
        row: dict[str, Any] = {key: v, "kind": kind, "n_channel": int(red.channel_mask.sum()), "n_hillslope": int(red.n_hillslope), "n_channel_no_chi": int(np.sum(red.channel_mask & ~np.isfinite(red.b))),
                               "n_heads_12m": n_heads_at(v)}
        for lab in ("all", "poly"):
            f = fit_terms(T[lab], U, K0, lo, hi)["free_KD"]
            row.update({f"mse_{lab}": f["mse"], f"K_{lab}": f["K"], f"D_{lab}": f["D"], f"where_{lab}": f["where"]})
        row["mse_all_at_poly"] = mse_at(T["all"], U, row["K_poly"], row["D_poly"])
        row["mse_poly_at_all"] = mse_at(T["poly"], U, row["K_all"], row["D_all"])
        row["wall_s"] = time.perf_counter() - t2
        rows.append(row)
        done.add(v)
        walls.append(row["wall_s"])

    n_do = pilot if pilot else grid.size
    for i, v in enumerate(grid[:n_do]):
        _eval(v, "grid")
        if i % 100 == 0 or i == n_do - 1:
            b = min(rows, key=lambda q: q["mse_poly"])
            print(f"{tag}: {i + 1}/{n_do} best {key} {b[key]:.4g} (n_channel {b['n_channel']}) K {b['K_poly']:.3e} D {b['D_poly']:.3e} mse_poly {b['mse_poly']:.3f}; {np.mean(walls):.1f} s/eval [{time.time() - t0:.0f} s]", flush=True)
    rec["s_per_evaluation"] = float(np.mean(walls))
    if pilot:
        rec["pilot"] = {"n_evaluations": len(rows), "expected_evaluations_full": thin_n + refine_max + 1, "expected_h_full_run": float(np.mean(walls)) * (thin_n + refine_max + 1) / 3600.0}
        print(f"{tag} pilot: {len(rows)} evaluations at {rec['s_per_evaluation']:.1f} s each -> full run (<= {thin_n + refine_max + 1}) ~ {rec['pilot']['expected_h_full_run']:.2f} h", flush=True)
        return rec
    _eval(a21_thr, "a21_optimum")                                   # the addendum 21 optimum itself (the mask check is exact there)
    gs = np.sort(np.append(grid, a21_thr))
    extra: set[float] = set()
    for col in ("mse_poly", "mse_all"):
        j = int(np.argmin([q[col] for q in rows]))
        vj = rows[j][key]
        gi = int(np.searchsorted(gs, vj))
        lo_v, hi_v = gs[max(gi - 1, 0)], gs[min(gi + 1, gs.size - 1)]
        extra |= set(uv[(uv > lo_v) & (uv < hi_v)].tolist())
    extra -= done
    extra_a = np.array(sorted(extra))
    n_targets = int(extra_a.size)
    if extra_a.size > refine_max:
        extra_a = thin_log_uniform(extra_a, refine_max)
    for v in extra_a:
        _eval(v, "refine")
    rec["refine"] = {"n_targets": n_targets, "n_evaluated": int(extra_a.size), "capped": bool(n_targets > refine_max)}
    df = pd.DataFrame(rows).sort_values(key).reset_index(drop=True)
    df.to_parquet(out_path("profiles", f"{tag}_profile.parquet"), index=False)
    opt: dict[str, Any] = {}
    for lab in ("all", "poly"):
        j = int(np.argmin(df[f"mse_{lab}"].to_numpy()))
        opt[lab] = {key: float(df[key][j]), "K": float(df[f"K_{lab}"][j]), "D": float(df[f"D_{lab}"][j]), "mse": float(df[f"mse_{lab}"][j]), "where": str(df[f"where_{lab}"][j]),
                    "n_channel": int(df.n_channel[j]), "n_heads_12m": int(df.n_heads_12m[j]), "kind": str(df.kind[j]), "K_over_K0": float(df[f"K_{lab}"][j]) / K0,
                    "mse_other": float(df["mse_poly_at_all"][j] if lab == "all" else df["mse_all_at_poly"][j])}
    chk = {"a21_mse": a21_opt["mse"], "a21_threshold": a21_thr, "all_cells_mse": opt["all"]["mse"], "all_cells_threshold": opt["all"][key], "rel_diff_mse": rel(opt["all"]["mse"], a21_opt["mse"]),
           "rel_diff_K": rel(opt["all"]["K"], a21_opt["K"]), "rel_diff_D": rel(opt["all"]["D"], a21_opt["D"])}
    row21 = df[df[key] == a21_thr]
    chk["mse_all_at_a21_threshold"] = float(row21.mse_all.iloc[0]) if len(row21) else None
    chk["rel_diff_at_a21_threshold"] = rel(chk["mse_all_at_a21_threshold"], a21_opt["mse"]) if chk["mse_all_at_a21_threshold"] is not None else None
    chk["all_cells_min_below_a21"] = bool(opt["all"]["mse"] < a21_opt["mse"] * (1 - MASK_CHECK_TOL))
    chk["pass"] = bool(chk["rel_diff_at_a21_threshold"] is not None and chk["rel_diff_at_a21_threshold"] <= MASK_CHECK_TOL and not chk["all_cells_min_below_a21"])
    rec.update({"optimum_poly": opt["poly"], "optimum_all": opt["all"], "mask_check": chk, "n_evaluations": int(df.shape[0]),
                "identifiability_poly": identifiability([{key: q[key], "mse": q["mse_poly"], "K_opt": q["K_poly"], "D_opt": q["D_poly"]} for q in df.to_dict("records")], "mse", key=key),
                "D_at_bound_poly": bool(np.isclose(opt["poly"]["D"], S["Db"][0], rtol=1e-9) or np.isclose(opt["poly"]["D"], S["Db"][1], rtol=1e-9))})
    print(f"{tag}: polygon optimum {key} {opt['poly'][key]:.4g} (n_channel {opt['poly']['n_channel']}, 12 m heads {opt['poly']['n_heads_12m']}) K/K0 {opt['poly']['K_over_K0']:.2f} D {opt['poly']['D']:.4g} ({opt['poly']['where']}) "
          f"mse_poly {opt['poly']['mse']:.3f}, whole tile at it {opt['poly']['mse_other']:.3f}; all-cells optimum {key} {opt['all'][key]:.4g} mse {opt['all']['mse']:.4f} vs a21 {a21_opt['mse']:.4f} at {a21_thr:.4g} "
          f"(at a21 threshold rel {chk['rel_diff_at_a21_threshold']}, below a21 {chk['all_cells_min_below_a21']}; {'PASS' if chk['pass'] else 'FAIL'}) [{time.time() - t0:.0f} s]", flush=True)
    dump(json_ready(rec), out_path("profiles", f"{tag}.json"))
    red = reduce_at(opt["poly"][key])
    Zs, holes = partitioned_surface(prof, red, opt["poly"]["K"], opt["poly"]["D"], Z)
    rec["products"] = products_of(model, Zs, dx, poly, tag) | {"surface_holes": holes}
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["AD", "SJ_C", "P4"])
    p.add_argument("--tile", required=True)
    p.add_argument("--pilot", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    tag = f"{a.model}_{a.tile}" + ("_pilot" if a.pilot else "")
    slurm_record(os.path.dirname(out_path("profiles", f"{tag}.json")))
    S = tile_setup(a.tile)
    rec = run_ad(S, tag, a.pilot, t0) if a.model == "AD" else run_partitioned(S, a.model, tag, a.pilot, t0)
    rec["tile"] = a.tile
    rec["wall_s"] = time.time() - t0
    dump(json_ready(rec), out_path("profiles", f"{tag}.json"))
    print(f"{tag}: done [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
