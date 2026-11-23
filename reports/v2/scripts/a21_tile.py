#!/usr/bin/env python3
"""Addendum 21, per tile (Part A items 1-2 and 4; Part B items 1-3 and 5, all but SJ-C): at U = 6.5e-5 m/yr, m = 0.33.

  - the rescaling checks (decision 1.1): the addendum 18 P4 and AD fits at K, D x 65/77 evaluated directly at the new U (P4: one exact
    reduction at its A_crit; AD: one LU at its r), against the stored MSE (tolerance 1e-8);
  - the mask route (decision 1.3): `PartitionedProfiler` with the 0/1 mask A >= 7,400 m2 as `channel_stat` against the area route at
    A_crit = 7,400 m2 (the same channel set, the same Gram terms);
  - the observed criterion networks on the 1 m grid (decision 1.3) from the Part A seeds: alpha = 0 at A_F, alpha = 2 at C_F (alpha = 1.65
    at C_F,1.65 as a sensitivity), and the mapped-head network (TTV1-TTV3);
  - D_form = U / |C_crest| (the A20 crest curvature), D_relief = U L_H^2 / (2 R) traced from the A20 crest pixels to the mapped-head
    network (TTV1-TTV3, pixels inside the M&D polygon) or to the criterion network (the other tiles, flagged), with the segment bootstrap;
  - the fixed runs F (alpha = 0 network) and F2 (alpha = 2 network) at K0 and D in {D_phys, D_form, D_relief}, and PF-A, PF-C (K, D free,
    closed form from the same reductions, with the addendum 18 basin halves);
  - every surface (observed, F x 3, F2 x 3, PF-A, PF-C, P4, AD) block-meaned to 12 m, routed, and the criterion applied with its own
    A and S (decision 1.4; alpha = 2 at C_F and alpha = 0 at A_F), plus the Perron et al. (2012) valley-head sensitivity at 1 m; the
    measures against the picks and the M&D traced lines on TTV1-TTV3.
Outputs <run>/tiles/<tile>.json, <tile>_heads.npz (12 m channel/head sets per surface), <tile>_residuals.npz (P4, AD).
usage: a21_tile.py --tile TTV2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import (  # noqa: E402
    A18_SUMMARY,
    A_F_TABLE1,
    ALPHA_PRIMARY,
    ALPHA_SENS,
    BLOCK_M,
    D_PHYS,
    D_PHYS_MEASURED_RHO,
    E_RIDGE,
    E_RIDGE_MEAN,
    HOLDOUT_A_C,
    K_CAL,
    M_SENS,
    M_SITE,
    MASK_TOL,
    MD_TILES,
    N_BOOT_SEG,
    RESCALE,
    RESCALE_TOL,
    SEED,
    U_NEW,
    U_SENS,
    a18_specs,
    a20_path,
    ad_surface,
    channel_heads,
    coverage_1m,
    criterion_sets,
    dump,
    extract_and_measure,
    fits_from_reduction,
    h0_reading,
    head_measures,
    json_ready,
    load_json,
    mask_from_entries,
    md_lines_mask,
    md_polygon_mask,
    metre_tile,
    out_path,
    partitioned_surface,
    picks_on_tile,
    rescaled,
    residual_stats,
    segment_bootstrap_median,
    slurm_record,
)

from lemcvt.analyses.er_star import trace_to_channel  # noqa: E402
from lemcvt.analyses.sensitivity import _fit, _q, _terms_masked, greedy_basin_halves  # noqa: E402
from lemcvt.optimize.v2 import PartitionedProfiler  # noqa: E402

SURFACE_MODELS = ["obs", "F_phys", "F_form", "F_relief", "F2_phys", "F2_form", "F2_relief", "PF_A", "PF_C", "P4", "AD"]


def reduce_mask(prof: PartitionedProfiler, mask: np.ndarray, m: float):
    """One exact reduction of the partitioned model with an arbitrary channel mask (decision 1.3): the mask as the per-cell statistic."""
    gd = prof.gd
    saved = prof.stat
    prof.stat = mask[gd["rev_j"], gd["rev_i"]].astype(float)
    try:
        return prof.reduce(0.5, m)
    finally:
        prof.stat = saved


def net_from_seeds(sd, alpha: float, C: float, A: np.ndarray, FD: np.ndarray) -> tuple[np.ndarray, dict]:
    """The 1 m network of the 12 m criterion at (alpha, C): heads among the candidate 12 m cells, their entry cells, downstream propagation."""
    M, Md = sd[f"cand_M_a{alpha:g}"], sd[f"cand_Mdon_a{alpha:g}"]
    er, ec = sd["entry_r"], sd["entry_c"]
    head = np.isfinite(M) & (M >= C) & ~(Md >= C)
    use = head & (er >= 0)
    mask = mask_from_entries(er[use], ec[use], A, FD)
    return mask, {"alpha": alpha, "C": float(C), "n_head_cells_12m": int(head.sum()), "n_entries": int(use.sum()), "n_channel_1m": int(mask.sum()),
                  "n_heads_1m": int(channel_heads(mask, FD).sum())}


def d_relief_block(U: float, L: np.ndarray, R: np.ndarray, reached: np.ndarray, seg: np.ndarray, label: str) -> dict:
    used = reached & (L > 0) & (R > 0)
    rec: dict = {"network": label, "n_pixels": int(L.size), "n_reached": int(reached.sum()), "n_used": int(used.sum()), "n_R_nonpositive": int(np.sum(reached & ~(R > 0)))}
    if not used.any():
        return rec
    D = np.where(used, U * L ** 2 / (2 * np.where(R > 0, R, np.nan)), np.nan)
    rec.update({"L_H_median_m": float(np.median(L[used])), "L_H_q16_q84_m": [float(np.percentile(L[used], 16)), float(np.percentile(L[used], 84))],
                "R_median_m": float(np.median(R[used])), "Sbar_median": float(np.median(R[used] / L[used])),
                "D_relief": segment_bootstrap_median(D, seg, N_BOOT_SEG, seed=SEED)})
    rec["D_relief_U_sens"] = {k: rec["D_relief"]["median"] * v / U for k, v in U_SENS.items()}
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    p.add_argument("--no-perron", action="store_true")
    a = p.parse_args()
    tile = a.tile
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("tiles", f"{tile}.json")))
    td = metre_tile(tile)
    mp = td.cfg.model
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    U, m = float(mp.U), float(mp.m)
    assert abs(U - U_NEW) < 1e-15 and abs(m - M_SITE) < 1e-12, (U, m)
    spec = a18_specs()[f"{tile}:primary"]
    U_old, K0_old = float(spec["E"]), float(spec["K0"])
    K0 = float(mp.K0_partitioned)
    assert abs(K0 / (K0_old * RESCALE) - 1) < 1e-9, (K0, K0_old)
    Kb = (mp.K_bounds[0] * RESCALE, mp.K_bounds[1] * RESCALE)
    Db = (mp.D_bounds[0] * RESCALE, mp.D_bounds[1] * RESCALE)
    lo = np.array([U / Kb[1], U / Db[1]])
    hi = np.array([U / Kb[0], U / Db[0]])
    fits18 = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))[f"{tile}:primary"]
    g12 = load_json(out_path("grid12", "grid12.json"))
    fv = g12["field_values"]["12m"]
    sd = np.load(out_path("grid12", f"{tile}_seeds.npz"))
    partA = load_json(a20_path("site", "partA.json"))["tiles"][tile]
    crest = pd.read_parquet(a20_path("tiles", f"{tile}_divide_pixels.parquet"), columns=["row", "col", "segment", "C_Lf", "in_md_polygon"] if tile in MD_TILES else ["row", "col", "segment", "C_Lf"])
    rec: dict = {"tile": tile, "md_tile": tile in MD_TILES, "U": U, "U_old": U_old, "rescale": RESCALE, "m": m, "K0": K0, "K0_old": K0_old, "K_cal": K_CAL, "K0_over_K_cal": K0 / K_CAL,
                 "bounds_rescaled": {"K": list(Kb), "D": list(Db), "box_xy_equals_a18": bool(abs((U / Kb[1]) / (U_old / mp.K_bounds[1]) - 1) < 1e-12 and abs((U / Db[1]) / (U_old / mp.D_bounds[1]) - 1) < 1e-12)},
                 "field_values": {k: fv[k] for k in ("A_F", "C_F", "C_F_a1.65", "A_F_without_49", "C_F_without_49")}, "inputs": td.provenance()}
    # ---------------------------------------------------------------- rescaled addendum 18 / 20 fits
    rec["rescaled"] = {"P4": rescaled(fits18["P4"], Db[1]), "AD": rescaled(fits18["AD"], Db[1]), "P3": rescaled(fits18["P3"], Db[1]),
                       "heldout_P4_mean": fits18.get("heldout_part", {}).get("P4", {}).get("mean"), "heldout_AD_mean": fits18.get("heldout_AD", {}).get("mean")}
    fx20 = a20_path("fixed", f"{tile}.json")
    if os.path.exists(fx20):
        f20 = load_json(fx20)["PF"]["A_F_table1"]["free_KD"] | {"A_crit": A_F_TABLE1}
        rec["rescaled"]["PF_7400_a20"] = rescaled(f20, Db[1])
    # ---------------------------------------------------------------- the profiler, the halves, the mask-route check (decision 1.3)
    prof = PartitionedProfiler(Z, A, FD, dx, U, m_values=[m])
    gd = prof.gd
    half, hinfo = greedy_basin_halves(Z, A, FD, dx, dy, A_c=HOLDOUT_A_C)
    half_flat = half[gd["rev_j"], gd["rev_i"]]
    keep = {"all": np.ones(prof.n, dtype=bool), "h1": half_flat == 1, "h2": half_flat == 2}
    rec["split"] = {k: hinfo[k] for k in ("share_half1_of_split", "share_half2_of_split", "n_basins_edge")}
    t1 = time.perf_counter()
    red_a = prof.reduce(A_F_TABLE1, m)
    ra = fits_from_reduction(prof, red_a, keep, U, K0, lo, hi)
    red_m = reduce_mask(prof, np.isfinite(A) & (A >= A_F_TABLE1), m)
    rm = fits_from_reduction(prof, red_m, keep, U, K0, lo, hi)
    same_set = bool(np.array_equal(red_a.channel_mask, red_m.channel_mask))
    rel = abs(ra["free_KD"]["mse"] / rm["free_KD"]["mse"] - 1)
    rec["mask_route_check"] = {"A_crit": A_F_TABLE1, "same_channel_set": same_set, "mse_area_route": ra["free_KD"]["mse"], "mse_mask_route": rm["free_KD"]["mse"], "rel_diff": rel,
                               "max_abs_diff_b": float(np.nanmax(np.abs(red_a.b - red_m.b))), "pass": bool(same_set and rel <= MASK_TOL), "wall_s": time.perf_counter() - t1}
    rec["PF_7400_new_U"] = ra["free_KD"] | {"A_crit": A_F_TABLE1}
    print(f"{tile}: mask route {'PASS' if rec['mask_route_check']['pass'] else 'FAIL'} (rel {rel:.1e}, same set {same_set}); PF(7,400) K/K0 {ra['free_KD']['K_over_K0']:.3f} D {ra['free_KD']['D']:.4g} [{time.time() - t0:.0f} s]", flush=True)
    # ---------------------------------------------------------------- rescaling checks (decision 1.1): P4 and AD at the new U
    p4 = rec["rescaled"]["P4"]
    red4 = prof.reduce(float(p4["A_crit"]), m)
    t4 = red4.mse_terms(prof.Z_obs_flat)
    mse4 = red4.mse_from_terms(t4, U / p4["K"], U / p4["D"])
    rec["rescale_check"] = {"P4": {"A_crit": p4["A_crit"], "stored_mse": p4["mse"], "direct_mse_new_U": mse4, "rel_diff": abs(mse4 / p4["mse"] - 1), "pass": bool(abs(mse4 / p4["mse"] - 1) <= RESCALE_TOL)}}
    surfaces: dict[str, np.ndarray] = {"obs": Z}
    models: dict[str, dict] = {}
    Z4, holes4 = partitioned_surface(prof, red4, p4["K"], p4["D"], Z)
    surfaces["P4"] = Z4
    models["P4"] = {"K": p4["K"], "D": p4["D"], "mse": p4["mse"], "mse_direct": mse4, "A_crit": p4["A_crit"], "where": p4["where"], "n_channel": int(red4.channel_mask.sum()),
                    "n_heads_1m": int(channel_heads(np.isfinite(A) & (A >= p4["A_crit"]), FD).sum()), "surface_holes": holes4,
                    "n_params": 3, "heldout_mean": rec["rescaled"]["heldout_P4_mean"]}
    ad = rec["rescaled"]["AD"]
    Zad, mse_ad, adinfo = ad_surface(Z, A, FD, dx, m, U, ad["K"], ad["D"])
    rec["rescale_check"]["AD"] = {"r": adinfo["r"], "y": adinfo["y"], "stored_mse": ad["mse"], "direct_mse_new_U": mse_ad, "rel_diff": abs(mse_ad / ad["mse"] - 1), "pass": bool(abs(mse_ad / ad["mse"] - 1) <= RESCALE_TOL), "wall_s": adinfo["wall_s"]}
    surfaces["AD"] = Zad
    models["AD"] = {"K": ad["K"], "D": ad["D"], "mse": ad["mse"], "mse_direct": mse_ad, "where": ad["where"], "n_params": 2, "heldout_mean": rec["rescaled"]["heldout_AD_mean"],
                    "D_at_bound": ad["D_at_bound_rescaled"]}
    print(f"{tile}: rescale checks P4 rel {rec['rescale_check']['P4']['rel_diff']:.1e}, AD rel {rec['rescale_check']['AD']['rel_diff']:.1e} [{time.time() - t0:.0f} s]", flush=True)
    # ---------------------------------------------------------------- the networks (decision 1.3)
    nets: dict[str, np.ndarray] = {}
    rec["networks"] = {}
    for name, al, C in (("A", 0.0, fv["A_F"]), ("C", ALPHA_PRIMARY, fv["C_F"]), ("C165", ALPHA_SENS, fv["C_F_a1.65"]), ("A_w49", 0.0, fv["A_F_without_49"]), ("C_w49", ALPHA_PRIMARY, fv["C_F_without_49"])):
        nets[name], rec["networks"][name] = net_from_seeds(sd, al, C, A, FD)
    # the seed check: the tile's area grid equals the full-grid area at the seeds inside the tile
    r0, r1, c0, c1 = (int(v) for v in sd["tile_1m_full"])
    ins = (sd["seed_r"] >= r0) & (sd["seed_r"] < r1) & (sd["seed_c"] >= c0) & (sd["seed_c"] < c1)
    if ins.any():
        a_tile = A[sd["seed_r"][ins] - r0, sd["seed_c"][ins] - c0]
        rec["networks"]["seed_area_check"] = {"n": int(ins.sum()), "max_rel_diff": float(np.nanmax(np.abs(a_tile / sd["seed_A"][ins] - 1))), "n_nan": int(np.isnan(a_tile).sum())}
    pe_r, pe_c = sd["pick_entry_r"], sd["pick_entry_c"]
    use = pe_r >= 0
    if tile in MD_TILES and use.any():
        nets["mapped"] = mask_from_entries(pe_r[use], pe_c[use], A, FD)
        rec["networks"]["mapped"] = {"n_pick_entries": int(use.sum()), "pick_ids": sd["pick_id"][use].tolist(), "n_channel_1m": int(nets["mapped"].sum()), "n_heads_1m": int(channel_heads(nets["mapped"], FD).sum())}
    print(f"{tile}: networks " + "; ".join(f"{k}: {v['n_channel_1m']} cells, {v.get('n_heads_1m')} heads" for k, v in rec["networks"].items() if "n_channel_1m" in v) + f" [{time.time() - t0:.0f} s]", flush=True)
    # ---------------------------------------------------------------- D values (section 2.2)
    C_crest = float(partA["C_ht_Lf"])
    rec["D"] = {"C_crest": C_crest, "D0_N_A_a20": partA["D0_N_A"], "D_phys": D_PHYS, "D_phys_measured_rho": D_PHYS_MEASURED_RHO,
                "D_form": U / C_crest, "D_form_U_sens": {k: v / C_crest for k, v in U_SENS.items()},
                "D_crest_ridge_median": E_RIDGE / C_crest, "D_crest_ridge_mean": E_RIDGE_MEAN / C_crest,
                "D_form_over_D_phys": (U / C_crest) / D_PHYS, "U_over_E_ridge": U / E_RIDGE, "U_over_E_ridge_mean": U / E_RIDGE_MEAN,
                "model_crest_curvature_at_D_phys": U / D_PHYS}
    rows, cols, seg = crest["row"].to_numpy(), crest["col"].to_numpy(), crest["segment"].to_numpy()
    inpoly = crest["in_md_polygon"].to_numpy() if "in_md_polygon" in crest else np.zeros(rows.size, dtype=bool)
    rec["D"]["relief"] = {}
    for name in ("mapped", "A", "C"):
        if name not in nets:
            continue
        trc = trace_to_channel(Z, nets[name].astype(float), FD, dx, dy, rows, cols, 0.5)
        sel = inpoly if name == "mapped" else np.ones(rows.size, dtype=bool)
        blk = d_relief_block(U, np.where(sel, trc["L_H"], np.nan), np.where(sel, trc["R"], np.nan), trc["reached"] & sel, seg, name)
        blk["n_left_tile"] = int((trc["left_tile"] & sel).sum())
        blk["pixels"] = "inside the M&D polygon" if name == "mapped" else "all counted crest pixels"
        rec["D"]["relief"][name] = blk
        if name == "A" and tile in MD_TILES and inpoly.any():
            blk2 = d_relief_block(U, np.where(inpoly, trc["L_H"], np.nan), np.where(inpoly, trc["R"], np.nan), trc["reached"] & inpoly, seg, "A_in_polygon")
            rec["D"]["relief"]["A_in_polygon"] = blk2
    primary = "mapped" if ("mapped" in rec["D"]["relief"] and rec["D"]["relief"]["mapped"].get("D_relief")) else "A"
    rec["D"]["D_relief_network"] = primary
    rec["D"]["D_relief_flagged"] = primary != "mapped"
    rec["D"]["D_relief"] = rec["D"]["relief"][primary]["D_relief"]["median"]
    rec["D"]["D_relief_ci68"] = rec["D"]["relief"][primary]["D_relief"].get("ci68")
    rec["D"]["D_relief_over_D_form"] = rec["D"]["D_relief"] / rec["D"]["D_form"]
    rec["D"]["decomposition"] = {"D_relief_over_D_form_times_U_over_E_ridge": rec["D"]["D_relief_over_D_form"] * U / E_RIDGE}
    print(f"{tile}: D_phys {D_PHYS}, D_form {rec['D']['D_form']:.4g}, D_relief {rec['D']['D_relief']:.4g} ({primary}; L_H {rec['D']['relief'][primary]['L_H_median_m']:.0f} m) [{time.time() - t0:.0f} s]", flush=True)
    # ---------------------------------------------------------------- fixed runs and PF (decision 1.5)
    Dvals = {"phys": D_PHYS, "form": rec["D"]["D_form"], "relief": rec["D"]["D_relief"]}
    rec["fixed"], rec["PF"] = {}, {}
    for net_name, label in (("A", "F"), ("C", "F2")):
        t1 = time.perf_counter()
        red = reduce_mask(prof, nets[net_name], m)
        terms = _terms_masked(red, prof.Z_obs_flat, None)
        pf = fits_from_reduction(prof, red, keep, U, K0, lo, hi)
        pf_name = "PF_A" if net_name == "A" else "PF_C"
        rec["PF"][pf_name] = pf | {"network": net_name, "C": rec["networks"][net_name]["C"], "alpha": rec["networks"][net_name]["alpha"], "n_params": 2, "eval_wall_s": time.perf_counter() - t1}
        Zpf, holes = partitioned_surface(prof, red, pf["free_KD"]["K"], pf["free_KD"]["D"], Z)
        surfaces[pf_name] = Zpf
        models[pf_name] = {"K": pf["free_KD"]["K"], "D": pf["free_KD"]["D"], "mse": pf["free_KD"]["mse"], "where": pf["free_KD"]["where"], "n_channel": pf["n_channel"], "n_heads_1m": rec["networks"][net_name]["n_heads_1m"],
                           "surface_holes": holes, "n_params": 2, "heldout_mean": pf["free_KD"].get("mse_heldout_mean"), "network": net_name}
        for dname, Dv in Dvals.items():
            x0, y0 = U / K0, U / Dv
            mse = _q(terms, x0, y0)
            xk, yk, wk = _fit(terms, "fix_y", lo, hi, x0, y0)
            key = f"{label}_{dname}"
            rec["fixed"][key] = {"network": net_name, "D": Dv, "K": K0, "mse": mse, "n_channel": pf["n_channel"], "n_params": 0, "D_inside_bounds": bool(Db[0] <= Dv <= Db[1]),
                                 "K_free_at_this_D": {"K": U / xk, "mse": _q(terms, xk, yk), "where": wk, "K_over_K0": (U / xk) / K0}}
            Zf_, holes_ = partitioned_surface(prof, red, K0, Dv, Z)
            surfaces[key] = Zf_
            models[key] = {"K": K0, "D": Dv, "mse": mse, "n_channel": pf["n_channel"], "n_heads_1m": rec["networks"][net_name]["n_heads_1m"], "surface_holes": holes_, "n_params": 0, "network": net_name}
        # the m = 0.30 sensitivity of the free fit on the same network
        red30 = reduce_mask(prof, nets[net_name], M_SENS)
        pf30 = fits_from_reduction(prof, red30, keep, U, K0, lo, hi)
        rec["PF"][pf_name]["m_0.30_sensitivity"] = {"m": M_SENS, "free_KD": pf30["free_KD"]}
        print(f"{tile}: {pf_name} K/K0 {pf['free_KD']['K_over_K0']:.3f} D {pf['free_KD']['D']:.4g} ({pf['free_KD']['where']}) mse {pf['free_KD']['mse']:.2f}; " +
              ", ".join(f"{label}-{d}: {rec['fixed'][f'{label}_{d}']['mse']:.1f}" for d in Dvals) + f" [{time.time() - t0:.0f} s]", flush=True)
    del red, red4, red_a, red_m
    # ---------------------------------------------------------------- head extraction on every surface (decision 1.4)
    poly = md_polygon_mask(td)
    lines = md_lines_mask(td) if poly.any() else np.zeros(Z.shape, dtype=bool)
    pick_rc, pick_ids = picks_on_tile(td)
    rec["picks_on_tile"] = {"n": int(pick_rc[0].size), "ids": pick_ids, "n_in_polygon": int(poly[pick_rc].sum()) if pick_rc[0].size else 0, "polygon_px": int(poly.sum()), "line_px": int(lines.sum())}
    nb = Z.shape[0] // BLOCK_M
    heads_arrays: dict[str, np.ndarray] = {}
    rec["extraction"] = {}
    for name in SURFACE_MODELS:
        Zs = surfaces[name]
        e_rec, arrs = extract_and_measure(Zs, dx, fv, pick_rc, pick_ids, poly, lines, A if name == "obs" else None, FD if name == "obs" else None, perron=not a.no_perron)
        e_rec["n_holes_1m"] = models.get(name, {}).get("surface_holes", 0)
        if name != "obs":
            e_rec["residual"] = residual_stats(Zs, Z)
        rec["extraction"][name] = e_rec
        for k, v in arrs.items():
            heads_arrays[f"{name}_{k}"] = v
        a2 = e_rec["a2"]
        print(f"{tile}: {name}: a2 channels {a2['n_channel_12m']} heads {a2['n_heads_12m']} (poly {a2['n_heads_in_polygon']}/{a2['n_picks_in_polygon']}), pick dist {a2['pick_dist_m']['median']}; "
              f"a0 heads {e_rec['a0']['n_heads_12m']}; perron heads {e_rec.get('perron', {}).get('n_heads')} [{e_rec['wall_s']:.0f} s]", flush=True)
    # the observed criterion on the full-site lattice (the decision 1.3 network itself), for H0
    mR0, mR1, mC0, mC1 = (int(v) for v in sd["box12_margin"])
    r_off, c_off = mR0 * BLOCK_M - r0, mC0 * BLOCK_M - c0
    rec["extraction"]["obs_full_lattice"] = {}
    for key, al, C in (("a0", 0.0, fv["A_F"]), ("a2", ALPHA_PRIMARY, fv["C_F"]), ("a1.65", ALPHA_SENS, fv["C_F_a1.65"])):
        ch, hd = criterion_sets(sd[f"box_M_a{al:g}"], sd[f"box_Mdon_a{al:g}"], C)
        cov = coverage_1m(ch, BLOCK_M, Z.shape, r_off, c_off)
        hr, hc = np.nonzero(hd)
        hr1, hc1 = hr * BLOCK_M + BLOCK_M // 2 + r_off, hc * BLOCK_M + BLOCK_M // 2 + c_off
        inside = (hr1 >= 0) & (hr1 < Z.shape[0]) & (hc1 >= 0) & (hc1 < Z.shape[1])
        hm = head_measures(cov, (hr1[inside], hc1[inside]), pick_rc, pick_ids, poly, lines, dx)
        hm.update({"alpha": al, "C": C, "n_channel_12m_box": int(ch.sum()), "n_heads_12m_box": int(hr.size), "n_heads_12m_in_tile": int(inside.sum()), "lattice": "full-site", "offset_1m": [r_off, c_off]})
        hm["H"] = h0_reading(hm["pick_dist_m"]["median"], hm["ratio_heads_to_picks"]) if pick_rc[0].size else None
        rec["extraction"]["obs_full_lattice"][key] = hm
        heads_arrays[f"obsfull_{key}_channel"] = ch
        heads_arrays[f"obsfull_{key}_heads"] = hd
    heads_arrays["obsfull_offset_1m"] = np.array([r_off, c_off])
    heads_arrays["poly12"] = (np.mean(poly[:nb * BLOCK_M, :nb * BLOCK_M].reshape(nb, BLOCK_M, nb, BLOCK_M), axis=(1, 3)) >= 0.5)
    heads_arrays["pick_rc"] = np.stack(pick_rc).astype(np.int32) if pick_rc[0].size else np.zeros((2, 0), np.int32)
    heads_arrays["pick_ids"] = np.array(pick_ids, dtype=np.int32)
    heads_arrays["lines_1m"] = np.packbits(lines)
    heads_arrays["poly_1m"] = np.packbits(poly)
    for nname, mk in nets.items():
        heads_arrays[f"net_{nname}_1m"] = np.packbits(mk)
    np.savez_compressed(out_path("tiles", f"{tile}_heads.npz"), **heads_arrays)
    np.savez_compressed(out_path("tiles", f"{tile}_residuals.npz"), P4=(surfaces["P4"] - Z).astype(np.float32), AD=(surfaces["AD"] - Z).astype(np.float32),
                        PF_C=(surfaces["PF_C"] - Z).astype(np.float32), F_relief=(surfaces["F_relief"] - Z).astype(np.float32))
    rec["models"] = models
    rec["mse_summary"] = {k: v["mse"] for k, v in models.items()}
    rec["wall_s"] = time.time() - t0
    dump(json_ready(rec), out_path("tiles", f"{tile}.json"))
    print(f"{tile}: done, MSE " + ", ".join(f"{k} {v:.1f}" for k, v in rec["mse_summary"].items()) + f" [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
