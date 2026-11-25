#!/usr/bin/env python3
"""Addendum 23, Part B item 1 (decision 2.6): the five models on the domain, U = 65 m/Myr, m = 0.33, misfit inside the M&D polygon,
fitted on the 1 m grid (outlets fixed, no-flux divides: the domain options of the package).

  --model fixed  the split models with M&D's rules (K, D free; polygon Gram terms, closed-form box minimum), the nothing-fitted
                 model (K = the domain K0, D = the domain relief D) and the whole-domain misfit of every fit beside the polygon one;
  --model AD     the AD profile over r = K/D (AD_N_GRID log-uniform points between the bounds' ratio limits, closed-form y = U/D on the
                 polygon cells, bounded Brent in log10 r), both objectives from the same LU;
  --model SJ_C   the split model with the area x slope^2 threshold fitted: the profile over C on the observed criterion statistic
                 (the 12 m full-site statistic through the 12 m head cells' seeds, as addenda 21-22; or the accepted 1 m finder's
                 statistic when heads/accepted.json names one), thinned to SJC_THIN, refined (<= SJC_REFINE_MAX) around the minima.
Thresholds: M&D's 876 and 6,120 m2 through the 12 m networks of a23_domain.py unless a 1 m finder was accepted, in which case the
networks are rebuilt from its statistic at the thresholds re-measured at the picks (decision 2.6). `--pilot N` evaluates N points and
prints the time per evaluation and the estimate. Outputs <run>/fits/<model>.json, <model>_surfaces.npz (float32 surfaces), the SJ_C
and AD profiles as parquet. usage: a23_fit.py --model fixed|AD|SJ_C [--pilot N]
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
from a23_common import (  # noqa: E402
    A_F,
    AD_N_GRID,
    BLOCK_M,
    C_F,
    SJC_A_RANGE,
    SJC_REFINE_MAX,
    SJC_THIN,
    M,
    U,
    a21_path,
    bounds_box,
    criterion_sets,
    domain_ad_model,
    domain_profiler,
    dump,
    fit_terms,
    json_ready,
    load_domain,
    load_json,
    mask_from_entries,
    mse_at,
    out_path,
    partitioned_surface,
    slurm_record,
    terms_of,
)

from lemcvt.optimize.v2 import ADProfiler, identifiability, thin_log_uniform  # noqa: E402


def accepted_finder() -> dict[str, Any] | None:
    p = out_path("heads", "accepted.json")
    if not os.path.exists(p):
        return None
    return load_json(p).get("accepted")


def finder_statistic(name: str) -> dict[str, np.ndarray]:
    z = np.load(out_path("heads", "heads1m.npz"))
    return {"M": z[f"{name}_M"].astype(float), "Mdon": z[f"{name}_Mdon"].astype(float), "A": z[f"{name}_A"].astype(float)}


def networks_for_fit(dom: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The channel masks of the two M&D rules: the 12 m networks of a23_domain.py, or the accepted 1 m finder's at the re-measured
    thresholds (channel = upstream maximum of its statistic >= C; area rule: its area >= A)."""
    acc = accepted_finder()
    if acc is None:
        return {"C": dom["net_C"], "A": dom["net_A"]}, {"route": "12 m full-site statistic (addendum 21) through the 12 m head cells' seeds", "C": C_F, "A": A_F, "finder": None}
    st = finder_statistic(acc["candidate"])
    from lemcvt.analyses.slope_threshold import upstream_max
    netC = np.isfinite(st["M"]) & (st["M"] >= acc["C_all_picks"]) & dom["domain"]
    MA = upstream_max(np.where(np.isfinite(st["A"]), st["A"], np.nan), dom["A"], dom["FD"])
    netA = np.isfinite(MA) & (MA >= acc["A_all_picks"]) & dom["domain"]
    return {"C": netC, "A": netA}, {"route": f"accepted 1 m finder {acc['candidate']}", "C": acc["C_all_picks"], "A": acc["A_all_picks"], "finder": acc["candidate"]}


def keep_masks(prof, dom: dict[str, Any]) -> dict[str, np.ndarray]:
    gd = prof.gd
    return {"all": np.ones(prof.n, dtype=bool), "poly": dom["poly"][gd["rev_j"], gd["rev_i"]]}


def record_fit(f: dict[str, Any], T: dict[str, dict[str, float]], K0: float) -> dict[str, Any]:
    fp = f["free_KD"]
    return {"K": fp["K"], "D": fp["D"], "K_over_K0": fp["K"] / K0, "where": fp["where"], "mse_poly": fp["mse"], "mse_all_at_poly_fit": mse_at(T["all"], U, fp["K"], fp["D"]),
            "n_cells_poly": T["poly"]["n"], "n_cells_all": T["all"]["n"], "K_fixed_poly": f["K_fixed"]}


# ----------------------------------------------------------------------------- fixed-rule models

def run_fixed(dom: dict[str, Any], t0: float) -> None:
    rec = dom["rec"]
    K0, D_relief = float(rec["measured"]["K0"]), float(rec["measured"]["D_relief"])
    lo, hi, Kb, Db = bounds_box(rec)
    prof = domain_profiler(dom)
    keep = keep_masks(prof, dom)
    nets, ninfo = networks_for_fit(dom)
    out: dict[str, Any] = {"U": U, "m": M, "K0": K0, "D_relief": D_relief, "bounds_rescaled": {"K": list(Kb), "D": list(Db)}, "networks": ninfo, "n_cells": int(prof.n),
                           "n_cells_poly": int(keep["poly"].sum()), "n_outlets": int(dom["outlets"].sum()), "models": {}}
    print(f"fixed: profiler {prof.n:,} cells ({keep['poly'].sum():,} in polygon), chi anchors {prof.bc_inds.size}; networks {ninfo['route']} [{time.time() - t0:.0f} s]", flush=True)
    surfaces: dict[str, np.ndarray] = {}
    for net, name in (("C", "PF_C"), ("A", "PF_A")):
        t1 = time.perf_counter()
        from a23_common import reduce_mask
        red = reduce_mask(prof, nets[net], M)
        T = {k: terms_of(red, prof.Z_obs_flat, v) for k, v in keep.items()}
        f = fit_terms(T["poly"], U, K0, lo, hi)
        f_all = fit_terms(T["all"], U, K0, lo, hi)["free_KD"]
        m = record_fit(f, T, K0) | {"fitted": "K, D", "network": net, "threshold": ninfo[net], "n_channel": int(red.channel_mask.sum()),
                                    "n_channel_no_chi": int(np.sum(red.channel_mask & ~np.isfinite(red.b))),
                                    "whole_domain_fit": {"K": f_all["K"], "D": f_all["D"], "mse_all": f_all["mse"], "mse_poly": mse_at(T["poly"], U, f_all["K"], f_all["D"]), "K_over_K0": f_all["K"] / K0},
                                    "wall_s": time.perf_counter() - t1}
        Zs, holes = partitioned_surface(prof, red, m["K"], m["D"], dom["Z"])
        m["surface_holes"] = holes
        surfaces[name] = Zs
        out["models"][name] = m
        print(f"fixed: {name} K/K0 {m['K_over_K0']:.3f} D {m['D']:.4g} ({m['where']}) mse_poly {m['mse_poly']:.2f} (whole domain at these {m['mse_all_at_poly_fit']:.2f}; whole-domain fit "
              f"{f_all['mse']:.2f}) [{time.time() - t0:.0f} s]", flush=True)
        if net == "C":
            m2 = {"fitted": "none", "network": "C", "threshold": ninfo["C"], "K": K0, "D": D_relief, "K_over_K0": 1.0, "where": "fixed", "mse_poly": mse_at(T["poly"], U, K0, D_relief),
                  "mse_all_at_poly_fit": mse_at(T["all"], U, K0, D_relief), "n_channel": int(red.channel_mask.sum()), "n_cells_poly": T["poly"]["n"], "n_cells_all": T["all"]["n"],
                  "D_inside_bounds": bool(Db[0] <= D_relief <= Db[1])}
            Zs2, holes2 = partitioned_surface(prof, red, K0, D_relief, dom["Z"])
            m2["surface_holes"] = holes2
            surfaces["F2_relief"] = Zs2
            out["models"]["F2_relief"] = m2
            print(f"fixed: F2_relief (K0 {K0:.4g}, D_relief {D_relief:.4g}) mse_poly {m2['mse_poly']:.2f}, whole domain {m2['mse_all_at_poly_fit']:.2f} [{time.time() - t0:.0f} s]", flush=True)
        del red, T
    out["wall_s"] = time.time() - t0
    np.savez_compressed(out_path("fits", "fixed_surfaces.npz"), **{k: v.astype(np.float32) for k, v in surfaces.items()})
    dump(json_ready(out), out_path("fits", "fixed.json"))
    print(f"fixed: done [{time.time() - t0:.0f} s]", flush=True)


# ----------------------------------------------------------------------------- AD

def run_ad(dom: dict[str, Any], pilot: int, t0: float) -> None:
    rec = dom["rec"]
    K0 = float(rec["measured"]["K0"])
    lo, hi, Kb, Db = bounds_box(rec)
    adm = domain_ad_model(dom)
    from lemcvt.models.common import extract_grid_data
    gd = extract_grid_data(dom["Z"], dom["A"], dom["FD"], 1.0, outlets=dom["outlets"], noflux_outside=True)
    obs_all = dom["Z"][gd["rev_j"], gd["rev_i"]].astype(float)
    obs_poly = np.where(dom["poly"][gd["rev_j"], gd["rev_i"]], obs_all, np.nan)
    pa, pp = ADProfiler(adm, obs_all, U), ADProfiler(adm, obs_poly, U)
    r_lo, r_hi = Kb[0] / Db[1], Kb[1] / Db[0]
    grid = np.logspace(math.log10(r_lo), math.log10(r_hi), AD_N_GRID)
    rows: list[dict[str, Any]] = []
    cache: dict[float, dict[str, Any]] = {}
    walls: list[float] = []

    def evaluate(r: float, kind: str) -> dict[str, Any]:
        r = float(r)
        if r in cache:
            return cache[r]
        red = pa.reduce(r)
        D_lo, D_hi = max(Db[0], Kb[0] / r), min(Db[1], Kb[1] / r)
        q: dict[str, Any] = {"r": r, "kind": kind, "wall_s": red["wall_s"], "feasible": bool(D_lo <= D_hi)}
        if q["feasible"]:
            for lab, prof in (("all", pa), ("poly", pp)):
                y, mse, where = prof.best_y(red, U / D_hi, U / D_lo)
                q.update({f"mse_{lab}": mse, f"D_{lab}": U / y, f"K_{lab}": r * U / y, f"where_{lab}": where})
            q["mse_all_at_poly"] = pa.mse_at(red, U / q["D_poly"])
            q["mse_poly_at_all"] = pp.mse_at(red, U / q["D_all"])
        else:
            q.update({"mse_all": float("inf"), "mse_poly": float("inf")})
        cache[r] = q
        rows.append(q)
        walls.append(red["wall_s"])
        return q

    n_do = pilot if pilot else grid.size
    for i, r in enumerate(grid[:n_do]):
        evaluate(r, "grid")
        if i % 25 == 0 or i == n_do - 1:
            b = min((q for q in rows if q["feasible"]), key=lambda q: q["mse_poly"], default=None)
            print(f"AD: {i + 1}/{n_do} r {r:.3e}; best polygon mse {b['mse_poly'] if b else None}; {np.mean(walls):.1f} s/eval [{time.time() - t0:.0f} s]", flush=True)
    out: dict[str, Any] = {"model": "AD", "U": U, "m": M, "K0": K0, "r_range": [r_lo, r_hi], "n_grid": AD_N_GRID, "s_per_evaluation": float(np.mean(walls)), "n_cells": int(gd["n"]),
                           "n_cells_poly": int(np.isfinite(obs_poly).sum())}
    if pilot:
        out["pilot"] = {"n_evaluations": len(rows), "expected_evaluations_full": AD_N_GRID + 30, "expected_h_full_run": float(np.mean(walls)) * (AD_N_GRID + 30) / 3600.0}
        print(f"AD pilot: {len(rows)} evaluations at {out['s_per_evaluation']:.1f} s each -> full run ~ {out['pilot']['expected_h_full_run']:.2f} h", flush=True)
        dump(json_ready(out), out_path("fits", "AD_pilot.json"))
        return
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
    out.update({"optimum_poly": opt["poly"], "optimum_all": opt["all"], "n_evaluations": len(rows), "K_over_K0_poly": opt["poly"]["K"] / K0,
                "D_at_bound_poly": bool(np.isclose(opt["poly"]["D"], Db[0], rtol=1e-9) or np.isclose(opt["poly"]["D"], Db[1], rtol=1e-9))})
    out["identifiability_poly"] = identifiability([{"r": q["r"], "mse": q["mse_poly"], "K_opt": q["K_poly"], "D_opt": q["D_poly"]} for q in rows if q["feasible"]], "mse", key="r")
    pd.DataFrame(rows).sort_values("r").to_parquet(out_path("fits", "AD_profile.parquet"), index=False)
    red = pa.reduce(opt["poly"]["r"])
    flat = red["h"] + (U / opt["poly"]["D"]) * red["g"]
    Zs = np.full(dom["Z"].shape, np.nan)
    Zs[gd["rev_j"], gd["rev_i"]] = flat
    np.savez_compressed(out_path("fits", "AD_surfaces.npz"), AD=Zs.astype(np.float32))
    out["wall_s"] = time.time() - t0
    dump(json_ready(out), out_path("fits", "AD.json"))
    print(f"AD: polygon optimum K {opt['poly']['K']:.3e} (K/K0 {out['K_over_K0_poly']:.2f}) D {opt['poly']['D']:.4g} ({opt['poly']['where']}) mse_poly {opt['poly']['mse']:.2f}, whole domain at it "
          f"{opt['poly']['mse_other']:.2f}; whole-domain optimum K {opt['all']['K']:.3e} D {opt['all']['D']:.4g} mse {opt['all']['mse']:.2f} [{time.time() - t0:.0f} s]", flush=True)


# ----------------------------------------------------------------------------- SJ_C (the area x slope^2 threshold fitted)

def sjc_candidates_12m(dom: dict[str, Any]) -> dict[str, np.ndarray]:
    """The 12 m full-site statistic (alpha = 2) of every 12 m cell touching the domain, with the seed (largest-area 1 m domain cell of
    the block) and the 12 m area, as a23_domain.py builds the fixed networks."""
    g12 = np.load(a21_path("grid12", "grid12_arrays.npz"))
    M12, Md12, A12 = g12["M_a2"], g12["Mdon_a2"], g12["A"]
    fr0, fc0 = dom["rec"]["box"]["full_rows"][0], dom["rec"]["box"]["full_cols"][0]
    ny, nx = dom["shape"]
    A = dom["A"]
    R0, R1 = fr0 // BLOCK_M, (fr0 + ny) // BLOCK_M + 1
    C0, C1 = fc0 // BLOCK_M, (fc0 + nx) // BLOCK_M + 1
    Ms, Mds, As, sr, sc = [], [], [], [], []
    for rb in range(R0, min(R1, M12.shape[0])):
        for cb in range(C0, min(C1, M12.shape[1])):
            if not np.isfinite(M12[rb, cb]):
                continue
            r_lo, r_hi = rb * BLOCK_M - fr0, rb * BLOCK_M - fr0 + BLOCK_M
            c_lo, c_hi = cb * BLOCK_M - fc0, cb * BLOCK_M - fc0 + BLOCK_M
            if r_hi <= 0 or c_hi <= 0 or r_lo >= ny or c_lo >= nx:
                continue
            r_lo, r_hi, c_lo, c_hi = max(r_lo, 0), min(r_hi, ny), max(c_lo, 0), min(c_hi, nx)
            sub = A[r_lo:r_hi, c_lo:c_hi]
            if not np.isfinite(sub).any():
                continue
            k = int(np.nanargmax(sub))
            Ms.append(M12[rb, cb])
            Mds.append(Md12[rb, cb])
            As.append(A12[rb, cb])
            sr.append(r_lo + k // sub.shape[1])
            sc.append(c_lo + k % sub.shape[1])
    return {"M": np.array(Ms), "Mdon": np.array(Mds), "A12": np.array(As), "seed_r": np.array(sr, dtype=int), "seed_c": np.array(sc, dtype=int)}


def run_sjc(dom: dict[str, Any], pilot: int, t0: float) -> None:
    rec = dom["rec"]
    K0 = float(rec["measured"]["K0"])
    lo, hi, Kb, Db = bounds_box(rec)
    A, FD = dom["A"], dom["FD"]
    acc = accepted_finder()
    out: dict[str, Any] = {"model": "SJ_C", "U": U, "m": M, "K0": K0, "bounds_rescaled": {"K": list(Kb), "D": list(Db)}}
    if acc is None:
        cand = sjc_candidates_12m(dom)
        Mc, Mdc, er, ec = cand["M"], cand["Mdon"], cand["seed_r"], cand["seed_c"]
        prof = domain_profiler(dom)

        def mask_at(C: float) -> np.ndarray:
            use = np.isfinite(Mc) & (Mc >= C) & ~(Mdc >= C)
            return mask_from_entries(er[use], ec[use], A, FD)

        def n_heads_at(C: float) -> int:
            return int(np.sum(np.isfinite(Mc) & (Mc >= C) & ~(Mdc >= C)))

        from a23_common import reduce_mask

        def reduce_at(C: float):
            return reduce_mask(prof, mask_at(C), M)

        # the C range: the 1 m channel set as large as A >= 1e3 and A >= 1e5 m2 (bisection on C over the unique statistics; addendum 21)
        uv_all = np.unique(Mc[np.isfinite(Mc)])
        targets = [int(np.sum(np.isfinite(A) & (A >= a))) for a in SJC_A_RANGE]
        bounds = []
        for tgt in targets:
            lo_i, hi_i = 0, uv_all.size - 1
            while hi_i - lo_i > 1:
                mid = (lo_i + hi_i) // 2
                n = int(mask_at(uv_all[mid]).sum())
                if n > tgt:
                    lo_i = mid
                else:
                    hi_i = mid
            bounds.append(float(uv_all[hi_i]))
        C_lo, C_hi = min(bounds), max(bounds)
        uv = uv_all[(uv_all >= C_lo) & (uv_all <= C_hi)]
        out["setup"] = {"route": "12 m full-site statistic through the head cells' seeds", "C_range": [C_lo, C_hi], "n_channel_targets": targets, "n_candidates": int(Mc.size),
                        "n_unique_all": int(uv_all.size), "n_unique_in_range": int(uv.size), "thin": SJC_THIN, "refine_max": SJC_REFINE_MAX}
        key_desc = "C (m2, area x slope^2 on the 12 m statistic)"
    else:
        st = finder_statistic(acc["candidate"])
        Mflat_grid = st["M"]
        prof = domain_profiler(dom, channel_stat=np.where(np.isfinite(Mflat_grid), Mflat_grid, -np.inf)[np.isfinite(dom["A"])])

        def reduce_at(C: float):
            return prof.reduce(float(C), M)

        def n_heads_at(C: float) -> int:
            _ch, hd = criterion_sets(st["M"], st["Mdon"], C)
            return int(hd.sum())

        uv_all = np.unique(st["M"][np.isfinite(st["M"]) & dom["domain"]])
        targets = [int(np.sum(np.isfinite(A) & (A >= a))) for a in SJC_A_RANGE]
        ms = np.sort(st["M"][np.isfinite(st["M"]) & dom["domain"]])[::-1]
        C_lo, C_hi = float(ms[min(targets[0], ms.size) - 1]), float(ms[min(targets[1], ms.size) - 1])
        uv = uv_all[(uv_all >= C_lo) & (uv_all <= C_hi)]
        out["setup"] = {"route": f"accepted 1 m finder {acc['candidate']}: channel where its upstream maximum >= C", "C_range": [C_lo, C_hi], "n_channel_targets": targets,
                        "n_unique_all": int(uv_all.size), "n_unique_in_range": int(uv.size), "thin": SJC_THIN, "refine_max": SJC_REFINE_MAX}
        key_desc = f"C (m2, area x slope^2 on the {acc['candidate']} statistic)"
    grid = thin_log_uniform(uv, SJC_THIN)
    keep = keep_masks(prof, dom)
    out["setup"]["n_grid"] = int(grid.size)
    print(f"SJ_C: {out['setup']} [{time.time() - t0:.0f} s]", flush=True)
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
        row: dict[str, Any] = {"C": v, "kind": kind, "n_channel": int(red.channel_mask.sum()), "n_hillslope": int(red.n_hillslope), "n_channel_no_chi": int(np.sum(red.channel_mask & ~np.isfinite(red.b))),
                               "n_heads": n_heads_at(v)}
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
        if i % 50 == 0 or i == n_do - 1:
            b = min(rows, key=lambda q: q["mse_poly"])
            print(f"SJ_C: {i + 1}/{n_do} best C {b['C']:.4g} (n_channel {b['n_channel']}) K {b['K_poly']:.3e} D {b['D_poly']:.3e} mse_poly {b['mse_poly']:.3f}; {np.mean(walls):.1f} s/eval [{time.time() - t0:.0f} s]", flush=True)
    out["s_per_evaluation"] = float(np.mean(walls))
    if pilot:
        out["pilot"] = {"n_evaluations": len(rows), "expected_evaluations_full": SJC_THIN + SJC_REFINE_MAX, "expected_h_full_run": float(np.mean(walls)) * (SJC_THIN + SJC_REFINE_MAX) / 3600.0}
        print(f"SJ_C pilot: {len(rows)} evaluations at {out['s_per_evaluation']:.1f} s each -> full run (<= {SJC_THIN + SJC_REFINE_MAX}) ~ {out['pilot']['expected_h_full_run']:.2f} h", flush=True)
        dump(json_ready(out), out_path("fits", "SJ_C_pilot.json"))
        return
    _eval(C_F, "md_value")                                          # M&D's 876 m2 itself is on every profile
    gs = np.sort(np.append(grid, C_F))
    extra: set[float] = set()
    for col in ("mse_poly", "mse_all"):
        j = int(np.argmin([q[col] for q in rows]))
        vj = rows[j]["C"]
        gi = int(np.searchsorted(gs, vj))
        lo_v, hi_v = gs[max(gi - 1, 0)], gs[min(gi + 1, gs.size - 1)]
        extra |= set(uv[(uv > lo_v) & (uv < hi_v)].tolist())
    extra -= done
    extra_a = np.array(sorted(extra))
    n_targets = int(extra_a.size)
    if extra_a.size > SJC_REFINE_MAX:
        extra_a = thin_log_uniform(extra_a, SJC_REFINE_MAX)
    for v in extra_a:
        _eval(v, "refine")
    out["refine"] = {"n_targets": n_targets, "n_evaluated": int(extra_a.size), "capped": bool(n_targets > SJC_REFINE_MAX)}
    df = pd.DataFrame(rows).sort_values("C").reset_index(drop=True)
    df.to_parquet(out_path("fits", "SJ_C_profile.parquet"), index=False)
    opt: dict[str, Any] = {}
    for lab in ("all", "poly"):
        j = int(np.argmin(df[f"mse_{lab}"].to_numpy()))
        opt[lab] = {"C": float(df.C[j]), "K": float(df[f"K_{lab}"][j]), "D": float(df[f"D_{lab}"][j]), "mse": float(df[f"mse_{lab}"][j]), "where": str(df[f"where_{lab}"][j]),
                    "n_channel": int(df.n_channel[j]), "n_heads": int(df.n_heads[j]), "kind": str(df.kind[j]), "K_over_K0": float(df[f"K_{lab}"][j]) / K0,
                    "mse_other": float(df["mse_poly_at_all"][j] if lab == "all" else df["mse_all_at_poly"][j])}
    out.update({"optimum_poly": opt["poly"], "optimum_all": opt["all"], "n_evaluations": int(df.shape[0]), "key": key_desc,
                "identifiability_poly": identifiability([{"C": q["C"], "mse": q["mse_poly"], "K_opt": q["K_poly"], "D_opt": q["D_poly"]} for q in df.to_dict("records")], "mse", key="C"),
                "D_at_bound_poly": bool(np.isclose(opt["poly"]["D"], Db[0], rtol=1e-9) or np.isclose(opt["poly"]["D"], Db[1], rtol=1e-9)),
                "at_md_value": df[df.C == C_F].iloc[0].to_dict() if (df.C == C_F).any() else None})
    red = reduce_at(opt["poly"]["C"])
    Zs, holes = partitioned_surface(prof, red, opt["poly"]["K"], opt["poly"]["D"], dom["Z"])
    out["surface_holes"] = holes
    np.savez_compressed(out_path("fits", "SJ_C_surfaces.npz"), SJ_C=Zs.astype(np.float32))
    out["wall_s"] = time.time() - t0
    dump(json_ready(out), out_path("fits", "SJ_C.json"))
    print(f"SJ_C: polygon optimum C {opt['poly']['C']:.4g} (n_channel {opt['poly']['n_channel']}, heads {opt['poly']['n_heads']}) K/K0 {opt['poly']['K_over_K0']:.2f} D {opt['poly']['D']:.4g} "
          f"({opt['poly']['where']}) mse_poly {opt['poly']['mse']:.3f}; whole-domain optimum C {opt['all']['C']:.4g} mse {opt['all']['mse']:.3f} [{time.time() - t0:.0f} s]", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["fixed", "AD", "SJ_C"])
    p.add_argument("--pilot", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("fits", "x")))
    dom = load_domain()
    print(f"domain {dom['shape']}: {dom['domain'].sum():,} cells, {dom['outlets'].sum()} outlets, K0 {dom['rec']['measured']['K0']:.4g}, D_relief {dom['rec']['measured']['D_relief']:.4g} [{time.time() - t0:.0f} s]", flush=True)
    if a.model == "fixed":
        run_fixed(dom, t0)
    elif a.model == "AD":
        run_ad(dom, a.pilot, t0)
    else:
        run_sjc(dom, a.pilot, t0)


if __name__ == "__main__":
    main()
