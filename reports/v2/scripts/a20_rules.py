#!/usr/bin/env python3
"""Addendum 20, Part B summary (sections 5.1-5.3; rules R2, R6, R7, R9, R10, R14; the model-network D_R; the SJ alpha = 0 check).

Reads <run>/site/partA.json (D0_P, D0_N, the brackets, R11P), <run>/dstat/*.json, <run>/fixed/*.json, the a20 slope-threshold runs, the
addendum 18 summary (per_tile_fits.json, rules.json) and the D0_N pixel tables. "Holds" = at least 80 % of N, rounded up.
R2: F3, F4, E1, E2 (P2), H2 as addendum 18 section 6 with D0_P (the tile's D_C) as the primary D0, D0_N beside it, the addendum 18 D0_R outcome
as the third column; at a site where R11P fails, the lower-end statements (F3, F4) carry no verdict (section 4.2). R10: each fit's D against
[D_C,lo, D_R,hi] (bracket_P) and against the D0_N lower end (bracket_N); "above" depends only on D_R and is the same under both.
usage: a20_rules.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a20_common import (  # noqa: E402
    A18_SUMMARY,
    CLEAN,
    E_SITE,
    F4_LOG10,
    N_BOOT,
    PRIMARY,
    R6_MIN,
    R7_IMPROVEMENT,
    R7_MIN,
    R10_FRAC,
    R14_FRAC,
    RESULTS_ROOT,
    TTV_PRIMARY,
    a18_specs,
    dump,
    hold_n,
    load_json,
    metre_tile,
    out_path,
    segment_bootstrap_median,
    slurm_record,
    trace_to_mask,
)

from lemcvt.analyses.slope_threshold import head_statistic  # noqa: E402
from lemcvt.results import find_runs  # noqa: E402

PF_KEYS = {"A_F": "PF", "A_F_table1": "PF_7400", "A_F_A10": "PF_2848", "A_F_A3": "PF_611"}


def l10(x):
    return float(np.log10(x)) if x and x > 0 else float("nan")


def count(recs, key):
    sel = [r for r in recs if r.get(key) is not None]
    n = sum(1 for r in sel if r[key])
    N = len(sel)
    return {"n": n, "N": N, "needed": hold_n(N), "holds": bool(N > 0 and n >= hold_n(N)), "tiles_fail": [r["tile"] for r in sel if not r[key]]}


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def sj_runs() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in find_runs("slope_threshold"):
        if "_a20_sj_" not in r.name:
            continue
        try:
            p = json.loads((r / "params.json").read_text())
        except (OSError, ValueError):
            continue
        if "result" not in p:
            continue
        out.setdefault(p["tile"], {})[f"{float(p['alpha']):g}"] = {"run": r.name, "params": p}
    return out


def bracket_class(D, at_bound, bracket, determined=True) -> str:
    if D is None or bracket is None or not determined:
        return "undetermined"
    lo, hi = bracket
    if at_bound and lo <= D <= hi:
        return "undetermined"
    if D < lo:
        return "below"
    if D > hi:
        return "above"
    return "inside"


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("rules", "partB.json")))
    partA = load_json(out_path("site", "partA.json"))
    specs = a18_specs()
    fits18 = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    rules18 = load_json(os.path.join(A18_SUMMARY, "rules.json"))
    R11P = partA["rules"]["R11P"]
    gate_fail = {s: (s in ("GM", "AP") and not R11P[s]["pass"]) for s in ("GM", "AP", "TTV")}
    dstat = {}
    for f in glob.glob(out_path("dstat", "*.json")):
        if os.path.basename(f) != "job.json":
            d = load_json(f)
            if "spec" in d:
                dstat[d["spec"]] = d
    fixed = {t: load_json(out_path("fixed", f"{t}.json")) for t in TTV_PRIMARY if os.path.exists(out_path("fixed", f"{t}.json"))}
    sj = sj_runs()
    out: dict = {"n_dstat": len(dstat), "n_fixed": len(fixed), "n_sj_tiles": len(sj), "n_sj_runs": sum(len(v) for v in sj.values()), "R11P_gate": {s: not v for s, v in gate_fail.items()},
                 "R2": {}, "R6": {}, "R7": {}, "R9": {}, "R10": {}, "R14": {}, "model_network_D_R": {}, "checks": {}}
    # ------------------------------------------------------------------ R2
    by_group: dict[tuple[str, str], list[dict]] = {}
    recon_fail, gram_fail = [], []
    for key, s in specs.items():
        d, f = dstat.get(key), fits18.get(key)
        if not f or not f.get("P4") or not f.get("AD"):
            continue
        rec = {"spec": key, "tile": s["tile"], "group": s["group"], "site": s["site"], "has_D0_R": bool(s.get("has_D0")), "ends": []}
        if d and d.get("usable"):
            if d.get("gram_route_check", {}).get("pass") is False:
                gram_fail.append(key)
            for end, D0 in d["D0_used"].items():
                rec["ends"].append(end)
                rec[f"D0_{end}"] = D0
                rec[f"P4_D_over_D0{end}"] = f["P4"]["D"] / D0
                rec[f"AD_D_over_D0{end}"] = f["AD"]["D"] / D0
                rec[f"F3_{end}"] = bool(abs(l10(f["P4"]["D"] / D0)) < abs(l10(f["AD"]["D"] / D0)))
                p1, p2 = d["partitioned"][end].get("P1"), d["partitioned"][end].get("P2")
                a2 = d["ad"].get(f"A2_{end}", {}).get("optimum")
                rec[f"P1_{end}"], rec[f"P2_{end}"], rec[f"A2_{end}"] = p1, p2, a2
                if p1 and a2 and f.get("A1"):
                    rec[f"E1_{end}"] = bool(p1["mse"] < min(f["A1"]["mse"], a2["mse"]))
                if p2:
                    rec[f"E2_{end}"] = bool(p2["mse"] < f["AD"]["mse"])
                    if p2.get("heldout", {}).get("mse_heldout_mean") is not None and f.get("heldout_AD"):
                        rec[f"H2_{end}"] = bool(p2["heldout"]["mse_heldout_mean"] < f["heldout_AD"]["mean"])
                if d["ad"].get(f"A2_{end}", {}).get("reconstruction_check", {}).get("pass") is False:
                    recon_fail.append(f"{key}:{end}")
        rec["has_P"], rec["has_N"] = "P" in rec["ends"], "N" in rec["ends"]
        rec["lopsided"] = f.get("lopsided")
        rec["old"] = {k: f.get(k) for k in ("F3_A", "E1_A", "E2_P2_A", "H2_A")}
        by_group.setdefault((s["site"], s["group"]), []).append(rec)
    out["checks"]["A2_reconstruction_failed"] = recon_fail
    out["checks"]["gram_route_failed"] = gram_fail
    out["checks"]["gram_route_checked"] = sum(1 for d in dstat.values() if d.get("gram_route_check", {}).get("pass") is not None)
    for (site, group), recs in by_group.items():
        g: dict = {"N": len(recs), "N_P": sum(r["has_P"] for r in recs), "N_N": sum(r["has_N"] for r in recs), "N_D0_R": sum(r["has_D0_R"] for r in recs),
                   "lower_end_verdicts_withheld_by_R11P": gate_fail[site], "verdicts": group == "primary"}
        for end in ("P", "N"):
            us = [r for r in recs if end in r["ends"]]
            g[f"F3_{end}"] = count(us, f"F3_{end}") | {"P4_D_over_D0_median": med([r.get(f"P4_D_over_D0{end}") for r in us]), "AD_D_over_D0_median": med([r.get(f"AD_D_over_D0{end}") for r in us])}
            g[f"E1_{end}"] = count(us, f"E1_{end}")
            g[f"E2_{end}"] = count(us, f"E2_{end}")
            g[f"H2_{end}"] = {"all": count(us, f"H2_{end}"), "balanced": count([r for r in us if not r.get("lopsided")], f"H2_{end}")}
            v = [abs(l10(r[f"AD_D_over_D0{end}"])) for r in us if r.get(f"AD_D_over_D0{end}")]
            g[f"F4_{end}"] = {"AD_median_abs_log10": med(v), "holds": bool(v and med(v) >= F4_LOG10), "n": len(v)}
        if group == "primary" and site in rules18:
            old = rules18[site]["primary"]
            ch = []
            for k_new, k_old in (("F3_P", "F3_A"), ("E1_P", "E1_A"), ("E2_P", "E2_P2_A")):
                if g[k_new]["holds"] != old[k_old]["holds"]:
                    ch.append(f"{k_old} (D0_R): {old[k_old]['n']}/{old[k_old]['N']} {'holds' if old[k_old]['holds'] else 'fails'} -> {k_new} (D0_P): {g[k_new]['n']}/{g[k_new]['N']} {'holds' if g[k_new]['holds'] else 'fails'}")
            if g["H2_P"]["all"]["holds"] != old["H2_A"]["all"]["holds"]:
                ch.append(f"H2_A (D0_R): {old['H2_A']['all']['n']}/{old['H2_A']['all']['N']} -> H2_P: {g['H2_P']['all']['n']}/{g['H2_P']['all']['N']}")
            if g["F4_P"]["holds"] != old["F4"]["holds"]:
                ch.append(f"F4: {'holds' if old['F4']['holds'] else 'fails'} (D0_R) -> {'holds' if g['F4_P']['holds'] else 'fails'} (D0_P)")
            g["changed_verdicts_vs_a18"] = ch
            g["a18"] = {k: old.get(k) for k in ("F3_A", "F4", "E1_A", "E2_P2_A", "H2_A")}
            cl = [r for r in recs if r["tile"] in CLEAN[site]]
            g["clean"] = {k: count([r for r in cl if k.split("_")[1] in r["ends"]], k) for k in ("F3_P", "E1_P", "E2_P", "H2_P", "F3_N", "E1_N", "E2_N", "H2_N")}
        g["tiles"] = recs
        out["R2"][f"{site}:{group}"] = g
    # ------------------------------------------------------------------ R6 and the PF/SF tables
    if fixed:
        for key, nm in PF_KEYS.items():
            rows = []
            for t, fx in fixed.items():
                f18 = fits18[f"{t}:primary"]
                pf = fx["PF"][key]
                d = dstat.get(f"{t}:primary", {})
                a2p = (d.get("ad", {}).get("A2_P", {}).get("optimum") or {}).get("mse") if d.get("usable") else None
                rows.append({"tile": t, "A_crit": pf["A_crit"], "mse_PF": pf["free_KD"]["mse"], "mse_AD": f18["AD"]["mse"], "mse_P4": f18["P4"]["mse"], "PF_beats_AD": bool(pf["free_KD"]["mse"] < f18["AD"]["mse"]),
                             "PF_over_P4": pf["free_KD"]["mse"] / f18["P4"]["mse"], "PF_over_AD": pf["free_KD"]["mse"] / f18["AD"]["mse"], "K_over_K0": pf["free_KD"]["K_over_K0"], "D": pf["free_KD"]["D"],
                             "where": pf["free_KD"]["where"], "mse_PF3": pf["K_fixed"]["mse"], "D_PF3": pf["K_fixed"]["D"], "where_PF3": pf["K_fixed"]["where"], "mse_A1": f18["A1"]["mse"],
                             "mse_A2_R": (f18.get("A2_A") or {}).get("mse"), "mse_A2_P": a2p,
                             "PF3_beats_A1": bool(pf["K_fixed"]["mse"] < f18["A1"]["mse"]),
                             "PF3_beats_A2_R": bool(pf["K_fixed"]["mse"] < f18["A2_A"]["mse"]) if f18.get("A2_A") else None,
                             "PF3_beats_A2_P": bool(pf["K_fixed"]["mse"] < a2p) if a2p else None, "n_channel": pf["n_channel"], "n_heads": pf["n_heads"]})
            n = sum(1 for r in rows if r["PF_beats_AD"])
            out["R6"][nm] = {"A_crit": rows[0]["A_crit"] if rows else None, "decides": key == "A_F", "n_PF_beats_AD": n, "N": len(rows), "needed": R6_MIN, "holds": bool(n >= R6_MIN) if key == "A_F" else None,
                             "n_beats_AD_report": n, "PF3_vs_A1": count(rows, "PF3_beats_A1"), "PF3_vs_A2_R": count(rows, "PF3_beats_A2_R"), "PF3_vs_A2_P": count(rows, "PF3_beats_A2_P"),
                             "PF_over_P4_median": med([r["PF_over_P4"] for r in rows]), "PF_over_AD_median": med([r["PF_over_AD"] for r in rows]), "K_over_K0_median": med([r["K_over_K0"] for r in rows]),
                             "D_median": med([r["D"] for r in rows]), "n_D_at_bound": sum(1 for r in rows if r["where"] != "interior"), "n_PF3_D_at_bound": sum(1 for r in rows if r["where_PF3"] != "interior"),
                             "tiles": rows}
        sf = {}
        for al in fixed[next(iter(fixed))]["field_values"]["alphas_SF"]:
            rows = []
            for t, fx in fixed.items():
                f18 = fits18[f"{t}:primary"]
                s = fx["SF"][f"alpha_{al:g}:C_F"]
                rows.append({"tile": t, "C": s["C"], "mse_SF": s["free_KD"]["mse"], "mse_AD": f18["AD"]["mse"], "SF_beats_AD": bool(s["free_KD"]["mse"] < f18["AD"]["mse"]),
                             "SF_over_P4": s["free_KD"]["mse"] / f18["P4"]["mse"], "K_over_K0": s["free_KD"]["K_over_K0"], "D": s["free_KD"]["D"], "where": s["free_KD"]["where"],
                             "n_channel": s["n_channel"], "n_heads": s["n_heads"]})
            n = sum(1 for r in rows if r["SF_beats_AD"])
            sf[f"{al:g}"] = {"C_F": rows[0]["C"] if rows else None, "n_SF_beats_AD": n, "N": len(rows), "needed": R6_MIN, "holds": bool(n >= R6_MIN), "SF_over_P4_median": med([r["SF_over_P4"] for r in rows]),
                             "K_over_K0_median": med([r["K_over_K0"] for r in rows]), "D_median": med([r["D"] for r in rows]), "n_D_at_bound": sum(1 for r in rows if r["where"] != "interior"), "tiles": rows}
        out["R6"]["SF"] = sf
        out["checks"]["SF_alpha0_equals_PF"] = {t: fx.get("alpha0_equals_PF", {}).get("pass") for t, fx in fixed.items()}
    # ------------------------------------------------------------------ R7 and the alpha = 0 check against addendum 18
    r7_rows = []
    for t in TTV_PRIMARY:
        runs = sj.get(t, {})
        if "0" not in runs:
            continue
        tab = {}
        for al, r in runs.items():
            res = r["params"]["result"]
            o = res["joint_optimum"]
            tab[al] = {"run": r["run"], "C": o["C"], "K": o["K"], "D": o["D"], "mse": o["mse"], "where": o["where"], "n_channel": o["n_channel"], "n_channel_no_chi": o["n_channel_no_chi"],
                       "heldout_mean": res["holdout"]["mse_heldout_mean"], "K_over_K0": res["K_over_K0"], "n_evaluations": res["n_evaluations"]}
        h0 = tab["0"]["heldout_mean"]
        pos = [al for al in tab if float(al) > 0]
        best = min(pos, key=lambda al: tab[al]["heldout_mean"]) if pos else None
        imp = (1 - tab[best]["heldout_mean"] / h0) if best else None
        row = {"tile": t, "alphas": sorted(tab, key=float), "table": tab, "heldout_alpha0": h0, "best_alpha": best, "heldout_improvement_best": imp, "preferred": bool(imp is not None and imp > R7_IMPROVEMENT),
               "insample_ratio_to_alpha0": {al: tab[al]["mse"] / tab["0"]["mse"] for al in tab}, "complete": all(f"{a:g}" in tab for a in (0.0, 1.65, 2.0))}
        f18 = fits18[f"{t}:primary"]
        try:
            prof = pd.read_parquet(os.path.join(RESULTS_ROOT, runs["0"]["run"], "c_profile.parquet")).set_index("C")
            a18 = pd.read_parquet(os.path.join(RESULTS_ROOT, f18["run_part"], "acrit_profile.parquet")).set_index("A_crit")
            common = prof.index.intersection(a18.index)
            rel = np.abs(prof.loc[common, "mse_joint"].to_numpy() / a18.loc[common, "mse_P4_all"].to_numpy() - 1) if len(common) else np.array([np.nan])
            row["alpha0_check"] = {"n_common_rows": int(len(common)), "max_rel_diff_mse_common_rows": float(np.nanmax(rel)) if len(common) else None,
                                   "a18_P4": {k: f18["P4"][k] for k in ("A_crit", "K", "D", "mse")}, "sj0": {k: tab["0"][k] for k in ("C", "K", "D", "mse")},
                                   "optimum_rel_diff_mse": abs(tab["0"]["mse"] / f18["P4"]["mse"] - 1), "same_threshold": bool(abs(tab["0"]["C"] / f18["P4"]["A_crit"] - 1) < 1e-9),
                                   "pass_rows_1e-9": bool(len(common) and np.nanmax(rel) < 1e-9)}
        except Exception as exc:  # noqa: BLE001
            row["alpha0_check"] = {"error": repr(exc)}
        r7_rows.append(row)
    if r7_rows:
        n_pref = sum(1 for r in r7_rows if r["preferred"])
        out["R7"] = {"N": len(r7_rows), "n_preferred": n_pref, "needed": R7_MIN, "preferred_at_TTV": bool(n_pref >= R7_MIN), "best_alpha_per_tile": {r["tile"]: r["best_alpha"] for r in r7_rows},
                     "improvement_median": med([r["heldout_improvement_best"] for r in r7_rows]), "n_complete": sum(1 for r in r7_rows if r["complete"]), "tiles": r7_rows,
                     "insample_ratio_median": {al: med([r["insample_ratio_to_alpha0"].get(al) for r in r7_rows]) for al in ("1.65", "2")},
                     "alpha0_rows_pass_all": all(r.get("alpha0_check", {}).get("pass_rows_1e-9") for r in r7_rows),
                     "alpha0_same_optimum": sum(1 for r in r7_rows if r.get("alpha0_check", {}).get("same_threshold"))}
    # ------------------------------------------------------------------ fits list, R9, R10
    def fit_list(tile: str, site: str) -> list[dict]:
        f18 = fits18.get(f"{tile}:primary")
        fl = []
        if f18:
            for name in ("AD", "P4", "P3", "A1"):
                o = f18.get(name)
                if o:
                    fl.append({"model": name, "D": o["D"], "at_bound": bool(o.get("D_at_bound") or o.get("where") in ("y_lo", "y_hi")), "K": o["K"], "mse": o["mse"],
                               "A_crit": o.get("A_crit")})
        if site == "TTV" and tile in fixed:
            fx = fixed[tile]
            for key, nm in PF_KEYS.items():
                o = fx["PF"][key]
                fl.append({"model": nm, "D": o["free_KD"]["D"], "at_bound": o["free_KD"]["where"] != "interior", "K": o["free_KD"]["K"], "mse": o["free_KD"]["mse"], "A_crit": o["A_crit"]})
                fl.append({"model": nm + "3", "D": o["K_fixed"]["D"], "at_bound": o["K_fixed"]["where"] != "interior", "K": o["K_fixed"]["K"], "mse": o["K_fixed"]["mse"], "A_crit": o["A_crit"]})
            for al in fx["field_values"]["alphas_SF"]:
                o = fx["SF"][f"alpha_{al:g}:C_F"]
                fl.append({"model": f"SF_a{al:g}", "D": o["free_KD"]["D"], "at_bound": o["free_KD"]["where"] != "interior", "K": o["free_KD"]["K"], "mse": o["free_KD"]["mse"], "C": o["C"], "alpha": al})
        if site == "TTV" and tile in sj:
            for al, r in sj[tile].items():
                o = r["params"]["result"]["joint_optimum"]
                fl.append({"model": f"SJ_a{al}", "D": o["D"], "at_bound": o["where"] != "interior", "K": o["K"], "mse": o["mse"], "C": o["C"], "alpha": float(al)})
        return fl

    for site in ("GM", "AP", "TTV"):
        per_model: dict[str, dict] = {}
        for tile in PRIMARY[site]:
            tl = partA["tiles"].get(tile)
            if not tl:
                continue
            det = bool(tl["usable_N"])
            for f in fit_list(tile, site):
                cP = bracket_class(f["D"], f["at_bound"], tl.get("bracket_P"), det)
                cN = bracket_class(f["D"], f["at_bound"], tl.get("bracket_N"), det)
                cB = bracket_class(f["D"], f["at_bound"], tl.get("bracket_P_block"), det)
                pm = per_model.setdefault(f["model"], {"P": {"inside": 0, "below": 0, "above": 0, "undetermined": 0}, "N": {"inside": 0, "below": 0, "above": 0, "undetermined": 0},
                                                       "P_block": {"inside": 0, "below": 0, "above": 0, "undetermined": 0}, "tiles": {}})
                pm["P"][cP] += 1
                pm["N"][cN] += 1
                pm["P_block"][cB] += 1
                pm["tiles"][tile] = {"D": f["D"], "class_P": cP, "class_N": cN, "class_P_block": cB, "bracket_P": tl.get("bracket_P"), "bracket_N": tl.get("bracket_N"), "at_bound": f["at_bound"]}
                if site == "TTV":
                    out["R9"].setdefault(f["model"], {})[tile] = {"D": f["D"], "D_over_D0P": f["D"] / tl["D0_P"] if tl.get("D0_P") else None, "D_over_D0N": f["D"] / tl["D0_N_A"] if tl.get("D0_N_A") else None,
                                                                  "D_over_D0R": f["D"] / tl["D0_R_A"] if tl.get("D0_R_A") else None, "D_over_D_R": f["D"] / tl["D_R"] if tl.get("D_R") else None,
                                                                  "at_bound": f["at_bound"], "K": f["K"], "mse": f["mse"]}
        for name, pm in per_model.items():
            for end in ("P", "N", "P_block"):
                c = pm[end]
                det = c["inside"] + c["below"] + c["above"]
                c["n_determined"] = det
                c["needed"] = hold_n(det)
                c["consistent_with_linear_diffusion"] = bool(det > 0 and c["inside"] >= hold_n(det))
            pm["above_same_under_both_ends"] = bool(pm["P"]["above"] == pm["N"]["above"])
            c = pm["P"]
            det = c["n_determined"]
            verdict_allowed = not gate_fail[site]
            pm["verdict"] = None
            if verdict_allowed:
                pm["verdict"] = "consistent" if c["consistent_with_linear_diffusion"] else "not consistent"
            else:
                pm["verdict"] = "withheld (R11P fails): only the upper-end reading stands"
            pm["reading"] = None
            if site == "TTV" and name not in ("AD", "A1"):
                if det and c["above"] >= hold_n(det):
                    pm["reading"] = "above on >= 80 % of determined tiles: the excess D is not a linear model compensating for nonlinear hillslopes; read with R5-R8 as a sign that the area-only threshold is mis-specified"
                elif c["consistent_with_linear_diffusion"]:
                    pm["reading"] = "inside: the excess over D_C is what a linear model needs to match the measured relief; the F3 failure at TTV is not by itself evidence against the model"
                elif det and c["below"] >= hold_n(det):
                    pm["reading"] = "below (no reading set)"
                else:
                    pm["reading"] = "mixed classes: no reading applies"
            if gate_fail[site]:
                pm["upper_end_reading"] = f"above on {c['above']} of {det} determined tiles"
        out["R10"][site] = {"lower_end_verdict_withheld": gate_fail[site], "models": per_model, "frac_rule": R10_FRAC}
    for name, d in list(out["R9"].items()):
        out["R9"][name] = {"tiles": d, **{f"median_{k}": med([v[k] for v in d.values()]) for k in ("D_over_D0P", "D_over_D0N", "D_over_D0R", "D_over_D_R")},
                           "n_at_bound": sum(1 for v in d.values() if v["at_bound"]), "N": len(d)}
    # ------------------------------------------------------------------ model-network D_R (TTV fits; GM and AP P4 is in the tile records)
    for tile in TTV_PRIMARY:
        pix_p = out_path("tiles", f"{tile}_divide_pixels.parquet")
        if not os.path.exists(pix_p) or (tile not in fixed and tile not in sj):
            continue
        pix = pd.read_parquet(pix_p, columns=["row", "col", "segment", "C_Lf"])
        if pix.shape[0] == 0:
            continue
        td = metre_tile(tile)
        Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
        rows, cols, seg, C = (pix[k].to_numpy() for k in ("row", "col", "segment", "C_Lf"))
        E = E_SITE["TTV"]
        nets: dict[str, np.ndarray] = {}
        Mc: dict[float, np.ndarray] = {}
        for f in fit_list(tile, "TTV"):
            if f["model"] in ("AD", "A1", "P3") or f["model"].endswith("3") and f["model"].startswith("PF"):
                continue
            if "A_crit" in f and f.get("A_crit"):
                nets[f["model"]] = np.isfinite(A) & (A >= f["A_crit"])
            elif "alpha" in f:
                al = float(f["alpha"])
                if al == 0:
                    nets[f["model"]] = np.isfinite(A) & (A >= f["C"])
                else:
                    if al not in Mc:
                        Mc[al] = head_statistic(Z, A, FD, dx, dy, al)["M"]
                    nets[f["model"]] = np.isfinite(Mc[al]) & (Mc[al] >= f["C"])
        rec = {}
        fl = {f["model"]: f for f in fit_list(tile, "TTV")}
        for name, target in nets.items():
            tr = trace_to_mask(Z, FD, dx, dy, rows, cols, target)
            used = tr["reached"] & (tr["L_H"] > 0) & (tr["R"] > 0) & (C > 0)
            D_R = np.where(used, E * tr["L_H"] ** 2 / (2 * np.where(tr["R"] > 0, tr["R"], np.nan)), np.nan)
            b = segment_bootstrap_median(D_R, seg, N_BOOT)
            fit_D = fl[name]["D"]
            rec[name] = {"n_used": int(used.sum()), "n_left_tile": int(tr["left_tile"].sum()), "L_H_median_m": float(np.median(tr["L_H"][used])) if used.any() else None, "D_R": b,
                         "fit_D": fit_D, "fit_D_over_D_R_model": fit_D / b["median"] if b.get("median") else None}
        out["model_network_D_R"][tile] = rec
        print(tile, "model-network D_R:", {k: (v["D_R"].get("median") and round(v["D_R"]["median"], 4), v["fit_D_over_D_R_model"] and round(v["fit_D_over_D_R_model"], 2)) for k, v in rec.items()},
              f"[{time.time() - t0:.0f} s]", flush=True)
    # ------------------------------------------------------------------ R14
    if fixed:
        for key, nm in PF_KEYS.items():
            rows = []
            for t in TTV_PRIMARY:
                if t not in fixed:
                    continue
                cls = out["R10"]["TTV"]["models"].get(nm, {}).get("tiles", {}).get(t, {})
                pf = next(r for r in out["R6"][nm]["tiles"] if r["tile"] == t)
                mn = out["model_network_D_R"].get(t, {}).get(nm, {})
                rows.append({"tile": t, "MSE_over_P4": pf["PF_over_P4"], "MSE_over_AD": pf["PF_over_AD"], "beats_AD": pf["PF_beats_AD"], "class_P": cls.get("class_P"), "class_N": cls.get("class_N"),
                             "D": pf["D"], "D_R_model": mn.get("D_R", {}).get("median"), "D_over_D_R_model": mn.get("fit_D_over_D_R_model"), "L_H_model": mn.get("L_H_median_m")})
            det = [r for r in rows if r["class_P"] in ("inside", "below", "above")]
            n_in = sum(1 for r in det if r["class_P"] == "inside")
            n_beat = sum(1 for r in rows if r["beats_AD"])
            reading = None
            if key == "A_F":
                ok = bool(det and n_in >= hold_n(len(det)) and n_beat >= R6_MIN)
                reading = ("a field-measured threshold gives physically consistent D at a fit cost that still beats AD" if ok else "the reading's conditions do not both hold") + \
                          f" (inside {n_in}/{len(det)} determined, needed {hold_n(len(det)) if det else '-'}; beats AD {n_beat}/{len(rows)}, needed {R6_MIN})"
            out["R14"][nm] = {"tiles": rows, "n_inside": n_in, "n_determined": len(det), "n_beats_AD": n_beat, "MSE_over_P4_median": med([r["MSE_over_P4"] for r in rows]),
                              "MSE_over_AD_median": med([r["MSE_over_AD"] for r in rows]), "reading": reading, "frac": R14_FRAC}
    out["wall_s"] = time.time() - t0
    dump(out, out_path("rules", "partB.json"))
    print("R2 primary:", {k: {kk: (v[kk]["n"], v[kk]["N"], v[kk]["holds"]) for kk in ("F3_P", "E1_P", "E2_P", "F3_N")} for k, v in out["R2"].items() if k.endswith(":primary")}, flush=True)
    print("R6:", {k: (v.get("n_PF_beats_AD"), v.get("holds")) for k, v in out["R6"].items() if k != "SF"}, "SF:", {k: (v["n_SF_beats_AD"], v["holds"]) for k, v in out["R6"].get("SF", {}).items()})
    print("R7:", {k: out["R7"].get(k) for k in ("n_preferred", "N", "preferred_at_TTV", "best_alpha_per_tile")} if out["R7"] else None)
    print("R10:", {s: {m: (v["P"]["inside"], v["P"]["below"], v["P"]["above"], v["P"]["undetermined"]) for m, v in r["models"].items()} for s, r in out["R10"].items()})
    print("R14:", {k: v["reading"] for k, v in out["R14"].items() if v["reading"]})


if __name__ == "__main__":
    main()
