#!/usr/bin/env python3
"""Addendum 21, section 6 rules and the report tables, from the run directory (grid12/, tiles/, sjc/).

H0b (the picks against M&D eq. 10 at map scale), H0 (the observed criterion locates the mapped heads; the full-site network decides,
the tile-lattice extraction is beside it), F0 (the fixed-step predictions), K1 (K physical: K/K0 within 0.5-2 on >= 9 of 11), M1 (SJ-C
beats AD on >= 9 of 11; the same reported for PF-A, PF-C and the fixed runs), H1 (model heads by the H0 conditions, alpha = 2 at C_F;
alpha = 0 and Perron beside, no verdict), H2 (SJ-C against AD, pooled TTV1-TTV3), DX (D_fit / D_relief within 0.67-1.5 on >= 9 of 11,
with the decomposition table). Every rule reports its count and denominator. Writes <run>/rules/rules.json and <run>/rules/tables.md.
usage: a21_rules.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import (  # noqa: E402
    CREST_CURV_MEASURED,
    D_PHYS,
    DX_RANGE,
    E_RIDGE,
    E_RIDGE_MEAN,
    H0_DIST_M,
    H0_RATIO,
    K1_RANGE,
    K_CAL,
    MD_TILES,
    PICK_EXCLUDED_SENS,
    SITE_MIN,
    TTV_PRIMARY,
    U_NEW,
    U_SENS,
    dump,
    h0_reading,
    hold_n,
    json_ready,
    load_json,
    out_path,
)

FREE_D = ["PF_A", "PF_C", "SJ_C", "P4"]
MODELS_K = ["PF_A", "PF_C", "SJ_C", "P4", "AD"]
FIXED = ["F_phys", "F_form", "F_relief", "F2_phys", "F2_form", "F2_relief"]
HEAD_MODELS = ["obs", "F_phys", "F_form", "F_relief", "F2_phys", "F2_form", "F2_relief", "PF_A", "PF_C", "SJ_C", "P4", "AD"]
OUT: list[str] = []


def W(s: str = "") -> None:
    OUT.append(s)


def f(x, fmt=".4g"):
    if x is None:
        return "-"
    if isinstance(x, bool | np.bool_):
        return "yes" if x else "no"
    if isinstance(x, int | np.integer):
        return f"{x:,}"
    if isinstance(x, float | np.floating):
        return format(x, fmt) if np.isfinite(x) else "-"
    return str(x)


def ci(v, fmt=".3g"):
    return f"[{f(v[0], fmt)}, {f(v[1], fmt)}]" if v else "-"


def table(h, rows):
    W("| " + " | ".join(h) + " |")
    W("|" + "---|" * len(h))
    for r in rows:
        W("| " + " | ".join(f(x) if not isinstance(x, str) else x for x in r) + " |")
    W()


def pooled_heads(tiles: dict, getter, drop_pick: int | None = None) -> dict:
    """Pooled pick distances and head ratio over TTV1-TTV3 for one extraction (getter(tile_record) -> measure dict)."""
    d, heads, picks_n, n_tiles = [], 0, 0, 0
    for t in MD_TILES:
        if t not in tiles:
            continue
        m = getter(tiles[t])
        if m is None:
            continue
        n_tiles += 1
        vals = m["pick_dist_m"]["values"]
        for k, v in vals.items():
            if drop_pick is not None and int(k) == drop_pick:
                continue
            d.append(v)
        heads += m["n_heads_in_polygon"]
        picks_n += m["n_picks_in_polygon"] - (1 if (drop_pick is not None and str(drop_pick) in vals) else 0)
    if not d:
        return {"n_tiles": n_tiles, "median_dist_m": None, "ratio": None, "n_heads": heads, "n_picks": picks_n, "H": None}
    d = np.array(d)
    med = float(np.median(d))
    ratio = heads / picks_n if picks_n else None
    return {"n_tiles": n_tiles, "complete": n_tiles == len(MD_TILES), "median_dist_m": med, "q25": float(np.percentile(d, 25)), "q75": float(np.percentile(d, 75)),
            "frac_within_25m": float(np.mean(d <= H0_DIST_M)), "n_picks": picks_n, "n_heads": heads, "ratio": ratio, "H": h0_reading(med, ratio)}


def main() -> None:
    t0 = time.time()
    g12 = load_json(out_path("grid12", "grid12.json"))
    tiles = {t: load_json(out_path("tiles", f"{t}.json")) for t in TTV_PRIMARY if os.path.exists(out_path("tiles", f"{t}.json"))}
    sjc = {t: load_json(out_path("sjc", f"{t}.json")) for t in TTV_PRIMARY if os.path.exists(out_path("sjc", f"{t}.json"))}
    N = len(tiles)
    rules: dict = {"n_tiles": N, "n_sjc": len(sjc), "missing_tiles": [t for t in TTV_PRIMARY if t not in tiles], "missing_sjc": [t for t in TTV_PRIMARY if t not in sjc]}
    fv = g12["field_values"]["12m"]
    # ------------------------------------------------------------------ per-tile model table
    per: dict[str, dict] = {}
    for t, r in tiles.items():
        mods = {k: dict(v) for k, v in r["models"].items()}
        if t in sjc:
            s = sjc[t]
            mods["SJ_C"] = {"K": s["joint_optimum"]["K"], "D": s["joint_optimum"]["D"], "mse": s["joint_optimum"]["mse"], "C": s["joint_optimum"]["C"], "where": s["joint_optimum"]["where"],
                            "n_channel": s["joint_optimum"]["n_channel"], "n_params": 3, "heldout_mean": (s.get("holdout") or {}).get("mse_heldout_mean"), "D_at_bound": s.get("D_at_bound"),
                            "interval_1pct_C": s["identifiability"]["level_0.01"]["intervals"], "n_heads_12m": s["joint_optimum"]["n_heads_12m"]}
        for v in mods.values():
            v["K_over_K0"] = v["K"] / r["K0"]
            v["K_over_K_cal"] = v["K"] / K_CAL
            v["D_over_D_phys"] = v["D"] / D_PHYS
            v["D_over_D_form"] = v["D"] / r["D"]["D_form"]
            v["D_over_D_relief"] = v["D"] / r["D"]["D_relief"]
        per[t] = {"K0": r["K0"], "models": mods, "D": r["D"], "mse_AD": mods["AD"]["mse"], "mse_SJC": mods.get("SJ_C", {}).get("mse")}
    rules["per_tile"] = per
    # ------------------------------------------------------------------ H0b
    rules["H0b"] = g12["rule_H0b"] | {"fits": {k: {kk: v[kk] for kk in ("a", "b", "b_ci95", "R2", "n_used")} | {"median_ratio_to_eq10": v["ratio_to_eq10"]["median"], "frac_in_0.5_2": v["ratio_to_eq10"]["frac_in_0.5_2"]} for k, v in g12["fits"].items()}}
    # ------------------------------------------------------------------ H0 (the observed criterion), pooled TTV1-TTV3
    rules["H0"] = {}
    for key in ("a2", "a0", "a1.65"):
        rules["H0"][f"full_lattice_{key}"] = pooled_heads(tiles, lambda r, k=key: r["extraction"]["obs_full_lattice"].get(k))
        rules["H0"][f"full_lattice_{key}_without_49"] = pooled_heads(tiles, lambda r, k=key: r["extraction"]["obs_full_lattice"].get(k), PICK_EXCLUDED_SENS)
    for key in ("a2", "a0", "perron"):
        rules["H0"][f"tile_lattice_{key}"] = pooled_heads(tiles, lambda r, k=key: r["extraction"]["obs"].get(k))
    rules["H0"]["verdict_a2"] = rules["H0"]["full_lattice_a2"]["H"]
    rules["H0"]["verdict_a0"] = rules["H0"]["full_lattice_a0"]["H"]
    # ------------------------------------------------------------------ H1, H2 (model heads)
    rules["H1"] = {}
    for mname in HEAD_MODELS:
        def get(r, mn=mname, k="a2"):
            if mn == "SJ_C":
                return sjc[r["tile"]]["extraction"].get(k) if r["tile"] in sjc else None
            return r["extraction"].get(mn, {}).get(k)
        rules["H1"][mname] = {k: pooled_heads(tiles, lambda r, mn=mname, kk=k: get(r, mn, kk)) for k in ("a2", "a0", "perron")}
    sj, ad = rules["H1"].get("SJ_C", {}).get("a2"), rules["H1"]["AD"]["a2"]
    if sj and ad and sj.get("median_dist_m") is not None and ad.get("median_dist_m") is not None and sj.get("ratio") and ad.get("ratio"):
        a = sj["median_dist_m"] < ad["median_dist_m"]
        b = abs(np.log(sj["ratio"])) < abs(np.log(ad["ratio"]))
        rules["H2"] = {"SJC_median_dist": sj["median_dist_m"], "AD_median_dist": ad["median_dist_m"], "SJC_ratio": sj["ratio"], "AD_ratio": ad["ratio"], "dist_smaller": bool(a), "ratio_closer_to_1_log": bool(b),
                       "holds": bool(a and b), "complete": bool(sj.get("complete") and ad.get("complete"))}
    else:
        rules["H2"] = {"holds": None, "note": "SJ-C or AD extraction missing on TTV1-TTV3"}
    # ------------------------------------------------------------------ K1
    rules["K1"] = {}
    for mname in MODELS_K:
        rows = {t: per[t]["models"][mname]["K_over_K0"] for t in per if mname in per[t]["models"]}
        n_ok = sum(K1_RANGE[0] <= v <= K1_RANGE[1] for v in rows.values())
        rules["K1"][mname] = {"K_over_K0": rows, "K_over_K_cal": {t: per[t]["models"][mname]["K_over_K_cal"] for t in rows}, "n_physical": n_ok, "n": len(rows), "need": SITE_MIN,
                              "holds": bool(len(rows) == N and n_ok >= SITE_MIN), "median_K_over_K0": float(np.median(list(rows.values()))) if rows else None,
                              "median_K_over_K_cal": float(np.median([per[t]["models"][mname]["K_over_K_cal"] for t in rows])) if rows else None}
    # ------------------------------------------------------------------ M1
    rules["M1"] = {}
    for mname in ["SJ_C", "PF_A", "PF_C"] + FIXED + ["P4"]:
        rows = {t: per[t]["models"][mname]["mse"] / per[t]["mse_AD"] for t in per if mname in per[t]["models"]}
        n_ok = sum(v < 1 for v in rows.values())
        rules["M1"][mname] = {"mse_over_AD": rows, "n_beats_AD": n_ok, "n": len(rows), "need": SITE_MIN, "holds": bool(len(rows) == N and n_ok >= SITE_MIN),
                              "median_ratio": float(np.median(list(rows.values()))) if rows else None}
    held = {t: (per[t]["models"]["SJ_C"]["heldout_mean"], per[t]["models"]["P4"]["heldout_mean"]) for t in per if "SJ_C" in per[t]["models"]}
    rules["M1"]["heldout_SJC_vs_P4"] = {t: {"SJ_C": v[0], "P4": v[1], "ratio": (v[0] / v[1]) if (v[0] and v[1]) else None} for t, v in held.items()}
    rules["M1"]["heldout_SJC_vs_P4"]["n_SJC_better"] = sum(1 for v in held.values() if v[0] and v[1] and v[0] < v[1])
    rules["M1"]["heldout_SJC_vs_P4"]["n"] = len(held)
    # ------------------------------------------------------------------ F0
    rules["F0"] = {"per_tile": {}, "predictions": {}}
    best_relief, worst_phys, crest_sharper = 0, 0, 0
    for t in per:
        mods = per[t]["models"]
        rec = {}
        for nm in FIXED:
            rec[nm] = {"mse": mods[nm]["mse"], "over_AD": mods[nm]["mse"] / per[t]["mse_AD"], "over_SJC": (mods[nm]["mse"] / per[t]["mse_SJC"]) if per[t]["mse_SJC"] else None}
        for pre in ("F", "F2"):
            trio = {d: mods[f"{pre}_{d}"]["mse"] for d in ("phys", "form", "relief")}
            rec[f"{pre}_best"] = min(trio, key=trio.get)
            rec[f"{pre}_worst"] = max(trio, key=trio.get)
        best_relief += rec["F_best"] == "relief"
        worst_phys += rec["F_worst"] == "phys"
        rec["crest_curvature_measured"] = per[t]["D"]["C_crest"]
        rec["model_crest_curvature_at_D_phys"] = U_NEW / D_PHYS
        crest_sharper += (U_NEW / D_PHYS) > per[t]["D"]["C_crest"]
        rules["F0"]["per_tile"][t] = rec
    rules["F0"]["predictions"] = {"F_relief_fits_best": {"n": best_relief, "N": N, "holds": bool(N and best_relief >= hold_n(N))},
                                  "F_phys_fits_worst": {"n": worst_phys, "N": N, "holds": bool(N and worst_phys >= hold_n(N))},
                                  "model_crest_sharper_than_measured_at_D_phys": {"n": crest_sharper, "N": N, "model_curvature": U_NEW / D_PHYS, "planner_measured": CREST_CURV_MEASURED,
                                                                                  "measured_median": float(np.median([per[t]["D"]["C_crest"] for t in per])) if per else None},
                                  "F2_relief_fits_best": {"n": sum(rules["F0"]["per_tile"][t]["F2_best"] == "relief" for t in per), "N": N},
                                  "F2_phys_fits_worst": {"n": sum(rules["F0"]["per_tile"][t]["F2_worst"] == "phys" for t in per), "N": N}}
    # ------------------------------------------------------------------ DX
    rules["DX"] = {}
    for mname in FREE_D:
        rows = {t: per[t]["models"][mname]["D_over_D_relief"] for t in per if mname in per[t]["models"]}
        n_ok = sum(DX_RANGE[0] <= v <= DX_RANGE[1] for v in rows.values())
        n_above = sum(v > DX_RANGE[1] for v in rows.values())
        rules["DX"][mname] = {"D_over_D_relief": rows, "n_explained": n_ok, "n_above": n_above, "n_below": sum(v < DX_RANGE[0] for v in rows.values()), "n": len(rows), "need": SITE_MIN,
                              "holds": bool(len(rows) == N and n_ok >= SITE_MIN), "median": float(np.median(list(rows.values()))) if rows else None,
                              "flagged_tiles": [t for t in rows if per[t]["D"]["D_relief_flagged"]]}
    rules["DX"]["reading"] = ("explained by relief (SJ-C and PF-C)" if (rules["DX"].get("SJ_C", {}).get("holds") and rules["DX"].get("PF_C", {}).get("holds"))
                              else ("excess not explained by relief (D_fit > 1.5 D_relief on most tiles)" if any(rules["DX"][m]["n_above"] > N / 2 for m in ("SJ_C", "PF_C") if m in rules["DX"]) else "mixed"))
    rules["DX"]["decomposition"] = {t: {"D_relief_over_D_form": per[t]["D"]["D_relief_over_D_form"], "U_over_E_ridge": U_NEW / E_RIDGE, "product": per[t]["D"]["D_relief_over_D_form"] * U_NEW / E_RIDGE,
                                        "product_E_mean": per[t]["D"]["D_relief_over_D_form"] * U_NEW / E_RIDGE_MEAN,
                                        **{f"{m}_D_over_D_phys": per[t]["models"][m]["D_over_D_phys"] for m in FREE_D + ["AD"] if m in per[t]["models"]}} for t in per}
    rules["DX"]["AD"] = {k: {t: per[t]["models"]["AD"][k] for t in per} for k in ("D_over_D_phys", "D_over_D_form", "D_over_D_relief")}
    # ------------------------------------------------------------------ checks and sensitivities
    rules["checks"] = {"mask_route": {t: tiles[t]["mask_route_check"]["pass"] for t in tiles}, "rescale_P4": {t: tiles[t]["rescale_check"]["P4"]["rel_diff"] for t in tiles},
                       "rescale_AD": {t: tiles[t]["rescale_check"]["AD"]["rel_diff"] for t in tiles}, "D_bound_hits": {t: {m: per[t]["models"][m].get("D_at_bound") for m in ("AD", "SJ_C") if m in per[t]["models"]} for t in per},
                       "PF_where": {t: {m: per[t]["models"][m]["where"] for m in ("PF_A", "PF_C", "P4", "AD", "SJ_C") if m in per[t]["models"]} for t in per},
                       "seed_area": {t: tiles[t]["networks"].get("seed_area_check") for t in tiles}}
    rules["sensitivities"] = {"U": {k: {"factor_on_K_D": v / U_NEW, "U_over_E_ridge": v / E_RIDGE} for k, v in U_SENS.items()}, "E_ridge_mean": {"U_over_E_ridge_mean": U_NEW / E_RIDGE_MEAN},
                              "m_0.30": {t: {m: tiles[t]["PF"][m]["m_0.30_sensitivity"]["free_KD"] for m in ("PF_A", "PF_C")} for t in tiles},
                              "grid_10m": g12["field_values"]["10m"], "dinf_12m": g12["field_values"]["12m_dinf"], "two_cell": g12["field_values"]["12m_S2"],
                              "pick_49": {"A_F_without_49": fv["A_F_without_49"], "C_F_without_49": fv["C_F_without_49"], "networks": {t: {k: tiles[t]["networks"][k] for k in ("A_w49", "C_w49")} for t in tiles}}}
    rules["wall_s"] = time.time() - t0
    dump(json_ready(rules), out_path("rules", "rules.json"))
    # ------------------------------------------------------------------ tables
    W("### Parameter values per tile (section 2)")
    table(["tile", "U", "K0 (m^0.34/yr)", "K0/K_cal", "C_crest (1/m)", "D_phys", "D_form", "D_form/D_phys", "D_relief", "68 %", "network", "L_H (m)", "R (m)", "D_relief/D_form", "D_crest at E_ridge"],
          [[t, f(r["U"], ".3g"), f(r["K0"], ".3g"), f(r["K0_over_K_cal"], ".2f"), f(r["D"]["C_crest"], ".4f"), f(r["D"]["D_phys"]), f(r["D"]["D_form"], ".4f"), f(r["D"]["D_form_over_D_phys"], ".2f"),
            f(r["D"]["D_relief"], ".4f"), ci(r["D"]["D_relief_ci68"], ".4f"), r["D"]["D_relief_network"] + (" (flagged)" if r["D"]["D_relief_flagged"] else ""),
            f(r["D"]["relief"][r["D"]["D_relief_network"]]["L_H_median_m"], ".0f"), f(r["D"]["relief"][r["D"]["D_relief_network"]]["R_median_m"], ".1f"), f(r["D"]["D_relief_over_D_form"], ".2f"),
            f(r["D"]["D_crest_ridge_median"], ".4f")] for t, r in tiles.items()])
    W("### Map-scale heads: fits (section 3)")
    table(["fit", "n", "a (m2)", "b", "b 95 %", "R2", "median A/(1978 S^-1.65)", "share in 0.5-2", "H0b"],
          [[k, v["n_used"], f(v["a"], ".0f"), f(v["b"], ".3f"), ci(v["b_ci95"], ".2f"), f(v["R2"], ".2f"), f(v["ratio_to_eq10"]["median"], ".2f"), f(v["ratio_to_eq10"]["frac_in_0.5_2"], ".2f"),
            f(v["H0b"]["consistent"])] for k, v in g12["fits"].items()])
    W("### Field values")
    table(["set", "A_F (m2)", "C_F alpha 2 (m2)", "C_F alpha 1.65 (m2)", "A_F w/o 49", "C_F w/o 49", "S nan/<=0"],
          [[k, f(v["A_F"], ".0f"), f(v["C_F"], ".1f"), f(v["C_F_a1.65"], ".1f"), f(v["A_F_without_49"], ".0f"), f(v["C_F_without_49"], ".1f"), v["n_S_nonpositive_or_nan"]] for k, v in g12["field_values"].items()])
    W("### H0: the observed criterion against the picks (TTV1-TTV3 pooled)")
    table(["extraction", "tiles", "median dist (m)", "q25-q75", "share <= 25 m", "heads in polygon", "picks", "ratio", "reading"],
          [[k, v["n_tiles"], f(v["median_dist_m"], ".1f"), f"{f(v.get('q25'), '.1f')}-{f(v.get('q75'), '.1f')}", f(v.get("frac_within_25m"), ".2f"), v["n_heads"], v["n_picks"], f(v["ratio"], ".2f"),
            v["H"]["reading"] if v["H"] else "-"] for k, v in rules["H0"].items() if isinstance(v, dict) and "n_tiles" in v])
    W("### H1: model heads (TTV1-TTV3 pooled; alpha = 2 at C_F decides; alpha = 0 and Perron beside)")
    rows = []
    for mname in HEAD_MODELS:
        h = rules["H1"][mname]
        a2, a0, pe = h["a2"], h["a0"], h["perron"]
        rows.append([mname, f(a2["median_dist_m"], ".1f"), f(a2["ratio"], ".2f"), f"{a2['n_heads']}/{a2['n_picks']}", a2["H"]["reading"] if a2["H"] else "-",
                     f(a0["median_dist_m"], ".1f"), f(a0["ratio"], ".2f"), f(pe["median_dist_m"], ".1f"), f(pe["ratio"], ".2f")])
    table(["model", "a2 median dist (m)", "a2 ratio", "heads/picks", "H1 reading", "a0 dist", "a0 ratio", "Perron dist", "Perron ratio"], rows)
    W("### K1 and M1 per tile")
    rows = []
    for t in per:
        mods = per[t]["models"]
        rows.append([t] + [f(mods[m]["K_over_K0"], ".2f") if m in mods else "-" for m in MODELS_K] + [f(mods[m]["mse"], ".1f") if m in mods else "-" for m in MODELS_K] +
                    [f(mods["SJ_C"]["mse"] / mods["AD"]["mse"], ".2f") if "SJ_C" in mods else "-", f(mods["PF_A"]["mse"] / mods["AD"]["mse"], ".2f"), f(mods["PF_C"]["mse"] / mods["AD"]["mse"], ".2f")])
    table(["tile"] + [f"K/K0 {m}" for m in MODELS_K] + [f"MSE {m}" for m in MODELS_K] + ["SJ-C/AD", "PF-A/AD", "PF-C/AD"], rows)
    W("### F0: the fixed step per tile (MSE / MSE_AD; MSE / MSE_SJ-C in brackets)")
    rows = []
    for t in per:
        rec = rules["F0"]["per_tile"][t]
        rows.append([t] + [f"{f(rec[nm]['over_AD'], '.2f')} ({f(rec[nm]['over_SJC'], '.2f')})" for nm in FIXED] + [rec["F_best"], rec["F_worst"], f(rec["crest_curvature_measured"], ".4f")])
    table(["tile"] + FIXED + ["F best", "F worst", "C_crest"], rows)
    W("### DX: D_fit / D_relief and the decomposition")
    rows = []
    for t in per:
        mods, dc = per[t]["models"], rules["DX"]["decomposition"][t]
        rows.append([t, f(per[t]["D"]["D_relief"], ".4f"), per[t]["D"]["D_relief_network"]] + [f(mods[m]["D_over_D_relief"], ".2f") if m in mods else "-" for m in FREE_D] +
                    [f(mods[m]["D_over_D_phys"], ".1f") if m in mods else "-" for m in FREE_D] + [f(dc["product"], ".2f"), f(mods["AD"]["D_over_D_phys"], ".3g"), f(mods["AD"]["D_over_D_relief"], ".3g")])
    table(["tile", "D_relief", "network"] + [f"{m} D/D_relief" for m in FREE_D] + [f"{m} D/D_phys" for m in FREE_D] + ["(D_relief/D_form)(U/E_ridge)", "AD D/D_phys", "AD D/D_relief"], rows)
    W("### Rule verdicts")
    table(["rule", "count", "need", "holds / reading"],
          [["H0 (alpha 2, full-site network)", "-", "-", rules["H0"]["verdict_a2"]["reading"] if rules["H0"]["verdict_a2"] else "-"],
           ["H0 (alpha 0)", "-", "-", rules["H0"]["verdict_a0"]["reading"] if rules["H0"]["verdict_a0"] else "-"],
           ["H0b (primary)", "-", "-", f(rules["H0b"]["primary"]["consistent"])],
           ["H2", "-", "-", f(rules["H2"].get("holds"))]] +
          [[f"K1 {m}", f"{v['n_physical']}/{v['n']}", v["need"], f(v["holds"])] for m, v in rules["K1"].items()] +
          [[f"M1 {m}", f"{v['n_beats_AD']}/{v['n']}", v["need"], f(v["holds"])] for m, v in rules["M1"].items() if isinstance(v, dict) and "n_beats_AD" in v] +
          [[f"DX {m}", f"{v['n_explained']}/{v['n']}", v["need"], f(v["holds"])] for m, v in rules["DX"].items() if isinstance(v, dict) and "n_explained" in v] +
          [[f"F0 {k}", f"{v['n']}/{v['N']}", hold_n(N), f(v.get("holds"))] for k, v in rules["F0"]["predictions"].items()])
    with open(out_path("rules", "tables.md"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")
    print("\n".join(OUT[-40:]))
    print(f"rules written [{time.time() - t0:.0f} s]; H0 a2: {rules['H0']['verdict_a2']}; H2: {rules['H2']}; K1: {{m: v['holds'] for m, v in rules['K1'].items()}}; M1 SJ-C: {rules['M1']['SJ_C']['holds']}; DX: {rules['DX']['reading']}")
    _ = H0_RATIO


if __name__ == "__main__":
    main()
