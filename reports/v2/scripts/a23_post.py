#!/usr/bin/env python3
"""Addendum 23 post-processing: the head metrics on every surface (the 12 m method and, if accepted, the 1 m finder), the readings of
section 7 (H-A, Road, C-1, misfit / K / D), the road counts per surface, and the report tables.

Inputs: <run>/domain, heads, roads, fits, measures. Outputs <run>/post/metrics.json, rules.json, tables.md. usage: a23_post.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a23_common import (  # noqa: E402
    A_F,
    BLOCK_M,
    C_F,
    CANDIDATE_PLAIN,
    CANDIDATES,
    D_PHYS,
    D_RELIEF_RANGE,
    FITTED,
    FITTED_MODELS,
    K_PHYS_RANGE,
    MODELS,
    PLAIN,
    ROAD_HEAD_BUFFER_M,
    T_PRIMARY_M,
    TOLERANCES_M,
    WHAT,
    criterion_sets,
    dump,
    json_ready,
    load_domain,
    load_json,
    match_heads,
    out_path,
)

WHAT23 = dict(WHAT) | {"F2_relief": "M&D area × slope² rule, K = measured K₀ (domain), D = relief D (domain)"}


def load_fits() -> dict[str, Any]:
    F: dict[str, Any] = {"models": {}}
    fx = load_json(out_path("fits", "fixed.json"))
    F.update({k: fx[k] for k in ("K0", "D_relief", "bounds_rescaled", "networks", "n_cells", "n_cells_poly")})
    F["models"].update(fx["models"])
    for name in ("AD", "SJ_C"):
        p = out_path("fits", f"{name}.json")
        if not os.path.exists(p):
            continue
        pr = load_json(p)
        o = pr["optimum_poly"]
        F["models"][name] = {"fitted": FITTED[name], "K": o["K"], "D": o["D"], "K_over_K0": o["K"] / fx["K0"], "where": o["where"], "mse_poly": o["mse"], "mse_all_at_poly_fit": o["mse_other"],
                             "threshold": o.get("C"), "n_channel": o.get("n_channel"), "whole_domain_fit": pr["optimum_all"], "D_at_bound": pr.get("D_at_bound_poly"),
                             "identifiability_poly": pr.get("identifiability_poly"), "n_evaluations": pr.get("n_evaluations"), "s_per_evaluation": pr.get("s_per_evaluation"), "wall_s": pr.get("wall_s"),
                             "setup": pr.get("setup"), "at_md_value": pr.get("at_md_value")}
    F["models"]["obs"] = {"fitted": "none", "K": None, "D": None, "K_over_K0": None, "mse_poly": 0.0, "mse_all_at_poly_fit": 0.0, "threshold": None}
    return F


def head_metrics_12m(dom: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """The addendum 22 metrics on the 12 m products of every surface (box lattice), heads inside the domain and the polygon."""
    P = np.load(out_path("measures", "surfaces12.npz"))
    inside12 = P["inside12_centre_in_domain_and_polygon"]
    pk = dom["picks"]
    use = pk["in_domain"] & pk["in_poly"]
    mapped_xy = np.stack([pk["row"][use], pk["col"][use]], axis=1).astype(float)
    road = dom["road_dist"]
    out: dict[str, Any] = {"n_mapped": int(use.sum()), "mapped_ids": pk["id"][use].tolist(), "models": {}}
    for name in names:
        if f"{name}_M_a2" not in P:
            continue
        rec: dict[str, Any] = {"rules": {}}
        for rule, key, C in (("primary", "a2", C_F), ("area_rule", "a0", A_F)):
            ch, hd = criterion_sets(P[f"{name}_M_{key}"], P[f"{name}_Mdon_{key}"], C)
            hr, hc = np.nonzero(hd & inside12)
            r1, c1 = hr * BLOCK_M + BLOCK_M // 2, hc * BLOCK_M + BLOCK_M // 2
            r1c, c1c = np.clip(r1, 0, road.shape[0] - 1), np.clip(c1, 0, road.shape[1] - 1)
            model_xy = np.stack([r1, c1], axis=1).astype(float)
            rr: dict[str, Any] = {"C": C, "n_heads_in_domain_polygon": int(hr.size), "n_channel_cells": int((ch & inside12).sum()), "by_t": {}}
            for t in TOLERANCES_M:
                mt = match_heads(mapped_xy, model_xy, t)
                extra = np.array([j not in {p[1] for p in mt["pairs"]} for j in range(model_xy.shape[0])], dtype=bool) if model_xy.shape[0] else np.zeros(0, bool)
                d_extra = road[r1c[extra], c1c[extra]] if extra.any() else np.zeros(0)
                rr["by_t"][f"{t:g}"] = {k: mt[k] for k in ("n_matched", "n_missed", "n_extra", "share_found", "share_real")} | {
                    "n_extra_within_15m_of_road": int(np.sum(d_extra <= ROAD_HEAD_BUFFER_M)), "pairs": [{"mapped_id": out["mapped_ids"][i], "model_rc": [int(r1[j]), int(c1[j])], "dist_m": d} for i, j, d in mt["pairs"]],
                    "extra_rc": [[int(r1[j]), int(c1[j])] for j in np.nonzero(extra)[0]]}
            rec["rules"][rule] = rr
        out["models"][name] = rec
    obs = out["models"].get("obs")
    if obs:
        for rec in out["models"].values():
            for rule in rec["rules"]:
                for k, b in rec["rules"][rule]["by_t"].items():
                    o = obs["rules"][rule]["by_t"][k]
                    b["share_found_vs_obs"] = b["share_found"] / o["share_found"] if o["share_found"] else None
                    b["share_real_vs_obs"] = b["share_real"] / o["share_real"] if o["share_real"] else None
    return out


def head_metrics_1m(dom: dict[str, Any], acc: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """The accepted 1 m finder on every surface at its all-picks threshold (heads inside the domain and the polygon)."""
    H = np.load(out_path("heads", "heads1m.npz"))
    HM = np.load(out_path("measures", "heads1m_models.npz")) if os.path.exists(out_path("measures", "heads1m_models.npz")) else {}
    cand = acc["candidate"]
    C = acc["C_all_picks"]
    pk = dom["picks"]
    use = pk["in_domain"] & pk["in_poly"]
    mapped_xy = np.stack([pk["row"][use], pk["col"][use]], axis=1).astype(float)
    keep = dom["domain"] & dom["poly"]
    road = dom["road_dist"]
    out: dict[str, Any] = {"candidate": cand, "C": C, "n_mapped": int(use.sum()), "models": {}}
    for name in names:
        Mk, Mdk = (f"{cand}_M", f"{cand}_Mdon") if name == "obs" else (f"{name}_M", f"{name}_Mdon")
        src = H if name == "obs" else HM
        if Mk not in getattr(src, "files", src.keys() if isinstance(src, dict) else []):
            continue
        _ch, hd = criterion_sets(src[Mk].astype(float), src[Mdk].astype(float), C)
        hr, hc = np.nonzero(hd & keep)
        model_xy = np.stack([hr, hc], axis=1).astype(float)
        rec: dict[str, Any] = {"n_heads_in_domain_polygon": int(hr.size), "by_t": {}}
        for t in TOLERANCES_M:
            mt = match_heads(mapped_xy, model_xy, t)
            extra = np.array([j not in {p[1] for p in mt["pairs"]} for j in range(model_xy.shape[0])], dtype=bool) if model_xy.shape[0] else np.zeros(0, bool)
            d_extra = road[hr[extra], hc[extra]] if extra.any() else np.zeros(0)
            rec["by_t"][f"{t:g}"] = {k: mt[k] for k in ("n_matched", "n_missed", "n_extra", "share_found", "share_real")} | {"n_extra_within_15m_of_road": int(np.sum(d_extra <= ROAD_HEAD_BUFFER_M))}
        out["models"][name] = rec
    return out


def readings(F: dict[str, Any], MEAS: dict[str, Any], H1: dict[str, Any], RD: dict[str, Any]) -> dict[str, Any]:
    R: dict[str, Any] = {"H_A": {}, "road": RD["reading"], "C_1": {}, "misfit": {}, "K_physical": {}, "D_relief": {}, "planner_expectation_C1": None}
    for c in CANDIDATES:
        q = H1["candidates"].get(c)
        if q is None:
            continue
        R["H_A"][c] = {"accepted": q["H_A"], "halves": {h: {"share_found": v["by_t"][f"{T_PRIMARY_M:g}"]["share_found"], "share_real": v["by_t"][f"{T_PRIMARY_M:g}"]["share_real"], "holds": v["H_A_half"]}
                                                       for h, v in q["halves"].items()}, "pooled": q["pooled"][f"{T_PRIMARY_M:g}"]}
    R["H_A"]["accepted_finder"] = H1["accepted"]
    ad = F["models"].get("AD")
    for name in FITTED_MODELS + ["F2_relief"]:
        m = F["models"].get(name)
        if m is None:
            continue
        R["misfit"][name] = {"mse_poly": m["mse_poly"], "mse_poly_AD": ad["mse_poly"] if ad else None, "ratio_to_AD": m["mse_poly"] / ad["mse_poly"] if ad else None,
                             "beats_AD": bool(ad and m["mse_poly"] < ad["mse_poly"])}
        if name != "F2_relief":
            R["K_physical"][name] = {"K_over_K0": m["K_over_K0"], "physical": bool(K_PHYS_RANGE[0] <= m["K_over_K0"] <= K_PHYS_RANGE[1])}
            R["D_relief"][name] = {"D": m["D"], "D_over_D_relief": m["D"] / F["D_relief"], "D_over_D_phys": m["D"] / D_PHYS,
                                   "explained_by_relief": bool(D_RELIEF_RANGE[0] <= m["D"] / F["D_relief"] <= D_RELIEF_RANGE[1])}
    sp = MEAS["spectrum"]["per_surface"]
    vs = MEAS["valley"]["per_sigma"]["10"]["per_surface"]
    for name in MODELS:
        if name == "obs" or name not in sp:
            continue
        spec_ok = sp[name]["ratio_in_reading_band"]["within_band_every_bin"]
        val_ok = vs[name]["within_range"]
        R["C_1"][name] = {"spectrum_within_0.5_2_in_20_200m": spec_ok, "spectrum_ratio_min_max": [sp[name]["ratio_in_reading_band"]["min"], sp[name]["ratio_in_reading_band"]["max"]],
                          "valley_share_ratio": vs[name]["ratio_to_lidar"], "valley_within_0.67_1.5": val_ok, "valley_share_ratio_sigma5": MEAS["valley"]["per_sigma"]["5"]["per_surface"][name]["ratio_to_lidar"],
                          "matches_lidar_channelization": bool(spec_ok and val_ok)}
    # the planner's expectation: AD more power than the lidar at short wavelengths and a larger valley share; the M&D-rule split models closer
    if "AD" in R["C_1"]:
        ad_short = sp["AD"]["ratio_20_60m_median"]
        ad_more_power = bool(ad_short > 1.0)
        ad_more_valley = bool(R["C_1"]["AD"]["valley_share_ratio"] > 1.0)
        split_closer = all(abs(np.log(sp[n]["ratio_20_60m_median"])) < abs(np.log(ad_short)) and abs(np.log(vs[n]["ratio_to_lidar"])) < abs(np.log(vs["AD"]["ratio_to_lidar"]))
                           for n in ("PF_C", "PF_A") if n in sp)
        R["planner_expectation_C1"] = {"AD_more_power_20_60m": ad_more_power, "AD_ratio_20_60m_median": ad_short, "AD_larger_valley_share": ad_more_valley,
                                       "md_rule_split_models_closer_than_AD_on_both": split_closer, "held": bool(ad_more_power and ad_more_valley and split_closer)}
    return R


def fmt(v, nd=2, none="–"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return none
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) < 0.001 and v != 0:
        return f"{v:.2e}"
    if abs(v) < 0.01 and v != 0:
        return f"{v:.3f}"
    return f"{v:.{nd}f}"


def tables(dom: dict[str, Any], F: dict[str, Any], MEAS: dict[str, Any], H1: dict[str, Any], RD: dict[str, Any], M12: dict[str, Any], M1: dict[str, Any] | None, R: dict[str, Any]) -> str:
    D = dom["rec"]
    me = D["measured"]
    tp = f"{T_PRIMARY_M:g}"
    L: list[str] = []
    L.append("### Model key\n")
    L.append("| Plain name | What it is | Fitted parameters | internal code |\n|---|---|---|---|")
    for n in MODELS:
        L.append(f"| **{PLAIN[n]}** | {WHAT23[n]} | {FITTED[n]} | {n} |")
    L.append("\n### The domain\n")
    d, o, p = D["domain"], D["outlets"], D["polygon"]
    L.append("| quantity | value |\n|---|---|")
    L.append(f"| domain area | {d['area_km2']:.3f} km² ({d['n_cells']:,} 1 m cells) |")
    L.append(f"| outlets (polygon exits with area ≥ 10⁴ m²) | {o['n_exits_at_or_above_A_min']} ({o['n_fixed']} fixed, {o['n_nested']} nested; polygon exit cells of any size {o['n_polygon_exit_cells_all_sizes']:,}) |")
    L.append(f"| largest outlet area | {fmt(o['largest_outlet_area_m2'])} m² |")
    L.append(f"| M&D polygon (valid cells) | {p['n_valid_cells']:,} m² (vector area {p['area_m2_vector']:,.0f} m²) |")
    L.append(f"| share of the polygon the domain covers | {d['share_of_polygon_covered']:.3f} |")
    L.append(f"| hillslope fringe (polygon cells draining out below 10⁴ m²) | {d['hillslope_fringe_cells']:,} cells, {d['hillslope_fringe_share_of_polygon']:.3f} of the polygon |")
    L.append(f"| share of the domain outside the polygon | {d['share_of_domain_outside_polygon']:.3f} |")
    L.append(f"| box | {D['box']['shape'][0]} × {D['box']['shape'][1]} cells, origin x {D['box']['x0']:.0f}, y {D['box']['y1']:.0f} (EPSG:6339) |")
    L.append(f"| picks in the domain / in the polygon / both | {D['picks']['n_in_domain']} / {D['picks']['n_in_polygon']} / {D['picks']['n_in_domain_and_polygon']} of {D['picks']['n']} (per tile in domain {D['picks']['per_tile']}) |")
    L.append(f"| M&D traced channel length inside the polygon | {p['md_traced_length_in_polygon_m']:,.0f} m |")
    L.append(f"| TIGER features on the box | {D['roads']['n_features']} ({D['roads']['mtfcc']}); trail classes present: {D['roads']['trail_classes_present'] or 'none'} |")
    L.append("\n### Re-measured inputs on the domain, beside the per-tile values\n")
    L.append("| quantity | domain | TTV1 | TTV2 | TTV3 |\n|---|---|---|---|---|")
    pt = me["per_tile_a21"]
    L.append(f"| k_s (χ–z at m = 0.33, A ≥ 10³ m²) | {me['ks']:.3f} (n basins {me['ks_1000']['n_basins']}, 68 % {fmt(me['ks_1000'].get('ks_ci68', [None, None])[0], 3)}–{fmt(me['ks_1000'].get('ks_ci68', [None, None])[1], 3)}) | "
             + " | ".join(f"{pt[t]['ks']:.3f}" for t in ("TTV1", "TTV2", "TTV3")) + " |")
    L.append(f"| K₀ = U / k_s (m^0.34/yr) | {me['K0']:.3e} | " + " | ".join(f"{pt[t]['K0']:.3e}" for t in ("TTV1", "TTV2", "TTV3")) + " |")
    cr = me["crest"]
    L.append(f"| crest curvature at 11 m (m⁻¹) | {cr['C_ht_Lf']['median']:.4f} (n {cr['n_counted']:,}; 68 % {fmt(cr['C_ht_Lf'].get('ci68', [None, None])[0], 4)}–{fmt(cr['C_ht_Lf'].get('ci68', [None, None])[1], 4)}) | "
             + " | ".join(f"{pt[t]['C_crest']:.4f}" for t in ("TTV1", "TTV2", "TTV3")) + " |")
    L.append(f"| D_form = U / crest curvature (m²/yr) | {cr['D_form']:.4f} | " + " | ".join(f"{pt[t]['D_form']:.4f}" for t in ("TTV1", "TTV2", "TTV3")) + " |")
    L.append(f"| physical-D check: measured crest curvature ÷ the curvature a linear model needs at D = 0.0025 (U/D = {cr['physical_D_check']['model_crest_curvature_at_D_phys']:.3f} m⁻¹) | {cr['physical_D_check']['measured_over_model_at_D_phys']:.2f} | – | – | – |")
    rl = me["relief"]
    L.append(f"| relief D on the mapped-head network (m²/yr) | {me['D_relief']:.4f} (n {rl['D_relief']['n']:,} crest pixels in the polygon; 68 % {fmt(rl['D_relief'].get('ci68', [None, None])[0], 4)}–{fmt(rl['D_relief'].get('ci68', [None, None])[1], 4)}; L_H median {rl['L_H_median_m']:.0f} m) | "
             + " | ".join(f"{pt[t]['D_relief']:.4f}" for t in ("TTV1", "TTV2", "TTV3")) + " |")
    L.append("\n### The road check (Addendum 22 TTV2 surfaces)\n")
    L.append("| quantity | 5 m | 15 m | 25 m |\n|---|---|---|---|")
    w = RD["wall_heads_near_road"]
    L.append(f"| share of the {RD['n_wall_heads']} too-steep-wall extra AD heads within the distance of a TIGER road | " + " | ".join(f"{fmt(w[b]['share'])} ({w[b]['n_within']})" for b in ("5", "15", "25")) + " |")
    e = RD["extra_heads_near_road"]
    L.append(f"| share of all {RD['n_extra_heads']} extra AD heads | " + " | ".join(f"{fmt(e[b]['share'])} ({e[b]['n_within']})" for b in ("5", "15", "25")) + " |")
    tr = RD["trunk"]["near_road"]
    L.append(f"| share of the observed trunk path (A ≥ 10⁵ m², {RD['trunk']['n_cells_in_polygon']:,} cells in the polygon) | " + " | ".join(f"{fmt(tr[b]['share'])}" for b in ("5", "15", "25")) + " |")
    L.append(f"\nMedian distance of the wall heads to a road {fmt(w['15'].get('median_distance_m'), 0)} m; of the trunk cells {fmt(RD['trunk']['median_distance_m'], 0)} m. "
             f"TIGER features on TTV2: {RD['roads']['n_features']} ({RD['roads']['mtfcc']}); trail classes: {RD['trail_classes_present'] or 'none'}. Reading: road-influenced = {fmt(RD['reading']['road_influenced'])}.\n")
    L.append("\n### The head-finder candidates on the lidar (reading H-A: at 25 m, share found ≥ 0.6 and share real ≥ 0.5 on both test halves)\n")
    L.append("| candidate | calibrated on TTV2 (34 picks): threshold | tested on TTV1 + TTV3: found / real (heads) | calibrated on TTV1 + TTV3: threshold | tested on TTV2: found / real (heads) | pooled found / real | F1-optimal threshold on the calibration half (TTV2 / TTV1+3) and its test-half found / real | H-A |\n|---|---|---|---|---|---|---|---|")
    for c in CANDIDATES:
        q = H1["candidates"].get(c)
        if q is None:
            continue
        h2, h13 = q["halves"]["TTV2"], q["halves"]["TTV13"]
        b2, b13 = h2["by_t"][tp], h13["by_t"][tp]
        f2, f13 = h2.get("F1_optimal_on_calibration_half") or {}, h13.get("F1_optimal_on_calibration_half") or {}
        L.append(f"| {CANDIDATE_PLAIN[c]} | {fmt(h2['C'], 0)} m² | {fmt(b2['share_found'])} / {fmt(b2['share_real'])} ({h2['n_heads_on_test_ground']}) | {fmt(h13['C'], 0)} m² | {fmt(b13['share_found'])} / {fmt(b13['share_real'])} ({h13['n_heads_on_test_ground']}) | "
                 f"{fmt(q['pooled'][tp]['share_found'])} / {fmt(q['pooled'][tp]['share_real'])} | {fmt(f2.get('C'), 0)} → {fmt(f2.get('test_half_at_this_C', {}).get('share_found'))} / {fmt(f2.get('test_half_at_this_C', {}).get('share_real'))}; "
                 f"{fmt(f13.get('C'), 0)} → {fmt(f13.get('test_half_at_this_C', {}).get('share_found'))} / {fmt(f13.get('test_half_at_this_C', {}).get('share_real'))} | {fmt(q['H_A'])} |")
    L.append("\n### Head-finder candidates at every tolerance (pooled over both test halves; share found / share real)\n")
    L.append("| candidate | " + " | ".join(f"{t:g} m" for t in TOLERANCES_M) + " |\n|---|" + "---|" * len(TOLERANCES_M))
    for c in CANDIDATES:
        q = H1["candidates"].get(c)
        if q is None:
            continue
        L.append(f"| {CANDIDATE_PLAIN[c]} | " + " | ".join(f"{fmt(q['pooled'][f'{t:g}']['share_found'])} / {fmt(q['pooled'][f'{t:g}']['share_real'])}" for t in TOLERANCES_M) + " |")
    L.append("\nThe statistic at the picks (largest area × slope² within 3 m; median over the 61 picks in the domain and polygon), and the median largest drainage area within 3 m:\n")
    L.append("| candidate | median area × slope² (m²) | 25th–75th | median area (m²) | heads in the domain ∩ polygon at that threshold |\n|---|---|---|---|---|")
    for c in CANDIDATES:
        q = H1["candidates"].get(c)
        if q is None:
            continue
        pk3 = q["picks"]["3"]
        L.append(f"| {CANDIDATE_PLAIN[c]} | {fmt(pk3['median_T_all'], 0)} | {fmt(pk3['T_pct']['25'], 0)}–{fmt(pk3['T_pct']['75'], 0)} | {fmt(pk3['median_A_all'], 0)} | {q['n_heads_domain_polygon_at_C_all_picks']} |")
    L.append("\n### Misfit, K and D on the domain (misfit inside the M&D polygon; m² mean squared elevation error)\n")
    L.append("| model | misfit in polygon | ÷ AD | whole-domain misfit at these parameters | K ÷ K₀ | K physical (0.5–2) | D (m²/yr) | D ÷ relief D | explained by relief (0.67–1.5) | D ÷ Heimsath's | fitted threshold | channel cells |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in MODELS:
        m = F["models"].get(n)
        if m is None or n == "obs":
            continue
        rm = R["misfit"].get(n, {})
        thr = "–"
        if n == "SJ_C":
            thr = f"{m['threshold']:,.0f} m² area × slope² (M&D 876)"
        elif n in ("PF_C", "F2_relief"):
            thr = f"{F['networks']['C']:,.0f} m² area × slope² (fixed)"
        elif n == "PF_A":
            thr = f"{F['networks']['A']:,.0f} m² area (fixed)"
        Dr = F["D_relief"]
        L.append(f"| {PLAIN[n]} | {fmt(m['mse_poly'])} | {fmt(rm.get('ratio_to_AD'))} | {fmt(m['mse_all_at_poly_fit'])} | {fmt(m['K_over_K0'])} | {fmt(R['K_physical'].get(n, {}).get('physical')) if n in R['K_physical'] else '–'} | "
                 f"{fmt(m['D'], 4)} | {fmt(m['D'] / Dr)} | {fmt(R['D_relief'].get(n, {}).get('explained_by_relief')) if n in R['D_relief'] else '–'} | {fmt(m['D'] / D_PHYS, 1)} | {thr} | {fmt(m.get('n_channel'))} |")
    L.append("\n### Channelization (reading C-1)\n")
    sp = MEAS["spectrum"]
    ro = sp["per_surface"]["obs"]["rolloff"]
    L.append(f"Square of side {sp['square']['side_m']} m inside the domain ({sp['square']['share_in_polygon']:.2f} of it in the polygon). Lidar spectrum: background power-law exponent {fmt(ro.get('background_exponent'))} "
             f"(over ≥ {ro.get('fit_min_m')} m: {fmt(ro.get('long_wavelength_exponent_at_or_above_fit_min'))}); characteristic wavelength (peak of the normalised spectrum) {fmt(ro.get('characteristic_wavelength_m'), 0)} m "
             f"({fmt(ro.get('normalised_peak'))} × the background); roll-off above the peak {fmt(ro.get('rolloff_m'), 0)} m.\n")
    lam = np.array(MEAS["spectrum"]["per_surface"]["obs"]["power"])
    edges = np.array(sp["bin_edges_m"])
    centres = np.sqrt(edges[:-1] * edges[1:])
    L.append("| model | " + " | ".join(f"{c:.0f} m" for c in centres) + " | within 0.5–2 in every bin 20–200 m |\n|---|" + "---|" * centres.size + "---|")
    for n in MODELS:
        q = sp["per_surface"].get(n)
        if q is None or n == "obs":
            continue
        L.append(f"| {PLAIN[n]} | " + " | ".join(fmt(v) for v in q["ratio_to_lidar"]) + f" | {fmt(q['ratio_in_reading_band']['within_band_every_bin'])} |")
    _ = lam
    L.append("\n| model | valley share (σ = 10 m) | ÷ lidar | within 0.67–1.5 | valley share (σ = 5 m) | ÷ lidar | matches the lidar's channelization (C-1) |\n|---|---|---|---|---|---|---|")
    v10, v5 = MEAS["valley"]["per_sigma"]["10"]["per_surface"], MEAS["valley"]["per_sigma"]["5"]["per_surface"]
    for n in MODELS:
        if n not in v10:
            continue
        c1 = R["C_1"].get(n, {})
        L.append(f"| {PLAIN[n]} | {v10[n]['share_domain']:.3f} | {fmt(v10[n]['ratio_to_lidar'])} | {fmt(v10[n]['within_range']) if n != 'obs' else '–'} | {v5[n]['share_domain']:.3f} | {fmt(v5[n]['ratio_to_lidar'])} | {fmt(c1.get('matches_lidar_channelization')) if n != 'obs' else '–'} |")
    pe = R.get("planner_expectation_C1") or {}
    L.append(f"\nPlanner's expectation (AD more power than the lidar at 20–60 m: {fmt(pe.get('AD_more_power_20_60m'))}, median ratio {fmt(pe.get('AD_ratio_20_60m_median'))}; AD larger valley share: {fmt(pe.get('AD_larger_valley_share'))}; "
             f"M&D-rule split models closer than AD on both: {fmt(pe.get('md_rule_split_models_closer_than_AD_on_both'))}): held = {fmt(pe.get('held'))}.\n")
    L.append("\n### Heads on every surface through the 12 m method (heads inside the domain and the polygon; 25 m; the Addendum 22 metrics)\n")
    L.append("| model | heads | matched | missed | extra | share found | ÷ lidar | share real | ÷ lidar | extra heads within 15 m of a road |\n|---|---|---|---|---|---|---|---|---|---|")
    for n in MODELS:
        q = M12["models"].get(n)
        if q is None:
            continue
        b = q["rules"]["primary"]["by_t"][tp]
        L.append(f"| {PLAIN[n]} | {q['rules']['primary']['n_heads_in_domain_polygon']} | {b['n_matched']} | {b['n_missed']} | {b['n_extra']} | {fmt(b['share_found'])} | {fmt(b.get('share_found_vs_obs'))} | {fmt(b['share_real'])} | {fmt(b.get('share_real_vs_obs'))} | {b['n_extra_within_15m_of_road']} |")
    L.append(f"\nMapped heads inside the domain and the polygon: {M12['n_mapped']}. Area rule (6,120 m²) at 25 m, share found / share real: " +
             "; ".join(f"{PLAIN[n]} {fmt(q['rules']['area_rule']['by_t'][tp]['share_found'])} / {fmt(q['rules']['area_rule']['by_t'][tp]['share_real'])}" for n, q in M12["models"].items()) + ".\n")
    if M1:
        L.append(f"\n### Heads on every surface through the accepted 1 m finder ({CANDIDATE_PLAIN[M1['candidate']]}, threshold {M1['C']:.0f} m²; 25 m)\n")
        L.append("| model | heads | matched | extra | share found | share real | extra within 15 m of a road |\n|---|---|---|---|---|---|---|")
        for n, q in M1["models"].items():
            b = q["by_t"][tp]
            L.append(f"| {PLAIN[n]} | {q['n_heads_in_domain_polygon']} | {b['n_matched']} | {b['n_extra']} | {fmt(b['share_found'])} | {fmt(b['share_real'])} | {b['n_extra_within_15m_of_road']} |")
    L.append("\n### Appendix: the domain check (TTV2 cut as before, its edge ring as the outlets, through the domain route)\n")
    dc = D.get("domain_check", {})
    L.append("| check | Addendum 22 value | domain route | relative difference | pass |\n|---|---|---|---|---|")
    for k, c in dc.get("checks", {}).items():
        a22v = c.get("a22", {}).get("mse_poly", c.get("a22_mse_poly"))
        new = c.get("domain_route", {}).get("mse_poly") if isinstance(c.get("domain_route"), dict) else c.get("domain_route")
        L.append(f"| {k} | {a22v:.6f} | {new:.6f} | {c['rel_mse']:.1e} | {fmt(c['pass'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.time()
    dom = load_domain()
    F = load_fits()
    MEAS = load_json(out_path("measures", "measures.json"))
    H1 = load_json(out_path("heads", "heads1m.json"))
    RD = load_json(out_path("roads", "road_check.json"))
    names = [n for n in MODELS if n in F["models"]]
    M12 = head_metrics_12m(dom, names)
    acc = H1.get("accepted")
    M1 = head_metrics_1m(dom, acc, names) if acc else None
    R = readings(F, MEAS, H1, RD)
    dump(json_ready({"fits": F, "heads12": M12, "heads1m_models": M1}), out_path("post", "metrics.json"))
    dump(json_ready({"readings": R}), out_path("post", "rules.json"))
    md = tables(dom, F, MEAS, H1, RD, M12, M1, R)
    with open(out_path("post", "tables.md"), "w") as f:
        f.write(md)
    print(md[:3000])
    print(f"post: done [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
