#!/usr/bin/env python3
"""Addendum 19, Part B summary (sections 5.1-5.3, rules R2, R6, R7, R9, R10; the model-network D_R of decision 12; the SJ alpha = 0 check).

Reads <run>/site/partA.json (D0_N, brackets, R11), <run>/dstat/*.json (P1_N, P2_N, A2_N per addendum 18 spec), <run>/fixed/*.json (PF, PF3, SF on
the TTV tiles), the a19 slope-threshold runs (SJ), the addendum 18 summary (per_tile_fits.json, rules.json) and the hilltop pixel tables.
Writes <run>/rules/partB.json. Rules as written in the directive's section 6 ("holds" = at least 80 % of N, rounded up).
usage: a19_rules.py
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
from a19_common import (  # noqa: E402
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


def l10(x):
    return float(np.log10(x)) if x and x > 0 else float("nan")


def count(recs, key, denom=None):
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
        if "a19_sj_" not in r.name or "pilot" in r.name:
            continue
        try:
            p = json.loads((r / "params.json").read_text())
        except (OSError, ValueError):
            continue
        if "result" not in p:
            continue
        out.setdefault(p["tile"], {})[f"{float(p['alpha']):g}"] = {"run": r.name, "params": p}
    return out


def bracket_class(D: float | None, at_bound: bool, bracket) -> str:
    if D is None or bracket is None:
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
    out_dir = os.path.dirname(out_path("rules", "partB.json"))
    slurm_record(out_dir)
    partA = load_json(out_path("site", "partA.json"))
    specs = a18_specs()
    fits18 = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    rules18 = load_json(os.path.join(A18_SUMMARY, "rules.json"))
    R11 = partA["rules"]["R11"]
    dstat = {}
    for f in glob.glob(out_path("dstat", "*.json")):
        if os.path.basename(f) == "job.json":
            continue
        d = load_json(f)
        if "spec" in d:
            dstat[d["spec"]] = d
    fixed = {t: load_json(out_path("fixed", f"{t}.json")) for t in TTV_PRIMARY if os.path.exists(out_path("fixed", f"{t}.json"))}
    sj = sj_runs()
    out: dict = {"n_dstat": len(dstat), "n_fixed": len(fixed), "n_sj_tiles": len(sj), "R11_gate": R11, "R2": {}, "R6": {}, "R7": {}, "R9": {}, "R10": {}, "model_network_D_R": {}, "sj_alpha0_check": {}}
    # ------------------------------------------------------------------ R2: the addendum 18 D statements with D0_N (decision 7)
    recs_by_group: dict[tuple[str, str], list[dict]] = {}
    for key, s in specs.items():
        d = dstat.get(key)
        f = fits18.get(key)
        if not f or not f.get("P4") or not f.get("AD"):
            continue
        rec = {"spec": key, "tile": s["tile"], "group": s["group"], "site": s["site"], "has_D0_N": bool(d and d.get("usable")), "has_D0_R": bool(s.get("has_D0"))}
        if d and d.get("usable"):
            for end in ("A", "B"):
                D0 = d[f"D0_N_{end}"]
                rec[f"D0_N_{end}"] = D0
                rec[f"P4_D_over_D0N_{end}"] = f["P4"]["D"] / D0
                rec[f"AD_D_over_D0N_{end}"] = f["AD"]["D"] / D0
                rec[f"F3_N_{end}"] = bool(abs(l10(f["P4"]["D"] / D0)) < abs(l10(f["AD"]["D"] / D0)))
                p1 = d["partitioned"][end].get("P1")
                p2 = d["partitioned"][end].get("P2")
                a2 = d["ad"].get(f"A2_N_{end}", {}).get("optimum")
                rec[f"P1_N_{end}"], rec[f"P2_N_{end}"], rec[f"A2_N_{end}"] = p1, p2, a2
                if p1 and a2 and f.get("A1"):
                    rec[f"E1_N_{end}"] = bool(p1["mse"] < min(f["A1"]["mse"], a2["mse"]))
                if p2:
                    rec[f"E2_N_{end}"] = bool(p2["mse"] < f["AD"]["mse"])
                    if p2.get("heldout", {}).get("mse_heldout_mean") is not None and f.get("heldout_AD"):
                        rec[f"H2_N_{end}"] = bool(p2["heldout"]["mse_heldout_mean"] < f["heldout_AD"]["mean"])
                rec[f"A2_N_{end}_recon_check"] = d["ad"].get(f"A2_N_{end}", {}).get("reconstruction_check", {}).get("pass")
        rec["lopsided"] = f.get("lopsided")
        rec["old"] = {k: f.get(k) for k in ("F3_A", "F3_B", "E1_A", "E1_B", "E2_P2_A", "E2_P2_B", "H2_A", "H2_B")}
        recs_by_group.setdefault((s["site"], s["group"]), []).append(rec)
    for (site, group), recs in recs_by_group.items():
        us = [r for r in recs if r["has_D0_N"]]
        g: dict = {"N": len(recs), "N_D0_N": len(us), "N_D0_R": sum(1 for r in recs if r["has_D0_R"]), "tiles_no_D0_N": [r["tile"] for r in recs if not r["has_D0_N"]],
                   "tiles_gained_D0": [r["tile"] for r in recs if r["has_D0_N"] and not r["has_D0_R"]], "tiles_lost_D0": [r["tile"] for r in recs if r["has_D0_R"] and not r["has_D0_N"]],
                   "verdicts": group == "primary", "withheld_by_R11": bool(site in ("GM", "AP") and not R11[site]["pass"])}
        for end in ("A", "B"):
            g[f"F3_N_{end}"] = count(us, f"F3_N_{end}") | {"P4_D_over_D0N_median": med([r.get(f"P4_D_over_D0N_{end}") for r in us]), "AD_D_over_D0N_median": med([r.get(f"AD_D_over_D0N_{end}") for r in us])}
            g[f"E1_N_{end}"] = count(us, f"E1_N_{end}")
            g[f"E2_N_{end}"] = count(us, f"E2_N_{end}")
            g[f"H2_N_{end}"] = {"all": count(us, f"H2_N_{end}"), "balanced": count([r for r in us if not r.get("lopsided")], f"H2_N_{end}")}
            v = [abs(l10(r[f"AD_D_over_D0N_{end}"])) for r in us if r.get(f"AD_D_over_D0N_{end}")]
            g[f"F4_N_{end}"] = {"AD_median_abs_log10": med(v), "ge_1": bool(v and med(v) >= F4_LOG10), "n": len(v)}
        g["F4_N"] = {"holds": bool(g["F4_N_A"]["ge_1"] and g["F4_N_B"]["ge_1"])}
        g["recon_checks_failed"] = [r["tile"] for r in us for end in ("A", "B") if r.get(f"A2_N_{end}_recon_check") is False]
        # changed verdicts against addendum 18 (D0_R), primary and clean
        if group == "primary":
            old = rules18[site]["primary"]
            ch = []
            for k_new, k_old in (("F3_N_A", "F3_A"), ("F3_N_B", "F3_B"), ("E1_N_A", "E1_A"), ("E1_N_B", "E1_B"), ("E2_N_A", "E2_P2_A"), ("E2_N_B", "E2_P2_B")):
                if g[k_new]["holds"] != old[k_old]["holds"]:
                    ch.append(f"{k_old}: {old[k_old]['n']}/{old[k_old]['N']} ({'holds' if old[k_old]['holds'] else 'fails'}) -> {k_new}: {g[k_new]['n']}/{g[k_new]['N']} ({'holds' if g[k_new]['holds'] else 'fails'})")
            for end in ("A", "B"):
                if g[f"H2_N_{end}"]["all"]["holds"] != old[f"H2_{end}"]["all"]["holds"]:
                    ch.append(f"H2_{end}: {old[f'H2_{end}']['all']['n']}/{old[f'H2_{end}']['all']['N']} -> {g[f'H2_N_{end}']['all']['n']}/{g[f'H2_N_{end}']['all']['N']}")
            if g["F4_N"]["holds"] != old["F4"]["holds"]:
                ch.append(f"F4: {'holds' if old['F4']['holds'] else 'fails'} -> {'holds' if g['F4_N']['holds'] else 'fails'}")
            g["changed_verdicts"] = ch
            # clean subset
            cl = [r for r in us if r["tile"] in CLEAN[site]]
            g["clean"] = {k: count(cl, k) for k in ("F3_N_A", "F3_N_B", "E1_N_A", "E1_N_B", "E2_N_A", "E2_N_B", "H2_N_A", "H2_N_B")}
        g["tiles"] = recs
        out["R2"][f"{site}:{group}"] = g
    # ------------------------------------------------------------------ R6: PF, PF3, SF against AD / A1, A2 at equal count (TTV)
    r6: dict = {}
    if fixed:
        for key, label in (("A_F", "PF at A_F (median A_3)"), ("A_F_table1", "PF at 7,400 m2 (Table 1 median)"), ("A_F_A10_sensitivity", "PF at the median A_10 (labelled sensitivity)"), ("A_F_without_49", "PF at A_F without pick 49")):
            rows = []
            for t, fx in fixed.items():
                f18 = fits18[f"{t}:primary"]
                pf = fx["PF"][key]
                d = dstat.get(f"{t}:primary", {})
                a2n = {end: (d.get("ad", {}).get(f"A2_N_{end}", {}).get("optimum") or {}).get("mse") for end in ("A", "B")} if d.get("usable") else {}
                rows.append({"tile": t, "A_crit": pf["A_crit"], "mse_PF": pf["free_KD"]["mse"], "mse_AD": f18["AD"]["mse"], "PF_beats_AD": bool(pf["free_KD"]["mse"] < f18["AD"]["mse"]),
                             "mse_PF3": pf["K_fixed"]["mse"], "mse_A1": f18["A1"]["mse"], "mse_A2_R_A": (f18.get("A2_A") or {}).get("mse"), "mse_A2_N_A": a2n.get("A"), "mse_A2_N_B": a2n.get("B"),
                             "PF3_beats_A1_A2R": bool(pf["K_fixed"]["mse"] < min(f18["A1"]["mse"], (f18.get("A2_A") or {}).get("mse", np.inf))) if f18.get("A2_A") else None,
                             "PF3_beats_A1_A2N": bool(pf["K_fixed"]["mse"] < min(f18["A1"]["mse"], a2n["A"])) if a2n.get("A") else None,
                             "mse_P4": f18["P4"]["mse"], "PF_over_P4": pf["free_KD"]["mse"] / f18["P4"]["mse"], "K_over_K0": pf["free_KD"]["K_over_K0"], "D": pf["free_KD"]["D"], "where": pf["free_KD"]["where"],
                             "D_PF3": pf["K_fixed"]["D"], "where_PF3": pf["K_fixed"]["where"], "n_channel": pf["n_channel"], "n_heads": pf["n_heads"]})
            n = sum(1 for r in rows if r["PF_beats_AD"])
            r6[key] = {"label": label, "n_PF_beats_AD": n, "N": len(rows), "needed": R6_MIN, "holds": bool(n >= R6_MIN), "PF3_vs_A1_A2R": count(rows, "PF3_beats_A1_A2R") | {"needed": R6_MIN, "holds": bool(sum(1 for r in rows if r["PF3_beats_A1_A2R"]) >= R6_MIN)},
                       "PF3_vs_A1_A2N": count(rows, "PF3_beats_A1_A2N") | {"needed": R6_MIN, "holds": bool(sum(1 for r in rows if r["PF3_beats_A1_A2N"]) >= R6_MIN)},
                       "PF_over_P4_median": med([r["PF_over_P4"] for r in rows]), "K_over_K0_median": med([r["K_over_K0"] for r in rows]), "D_median": med([r["D"] for r in rows]),
                       "n_D_at_bound": sum(1 for r in rows if r["where"] != "interior"), "tiles": rows}
        sf: dict = {}
        alphas = fixed[next(iter(fixed))]["field_values"]["alphas"]
        for al in alphas:
            rows = []
            for t, fx in fixed.items():
                f18 = fits18[f"{t}:primary"]
                s = fx["SF"][f"alpha_{al:g}:C_F"]
                rows.append({"tile": t, "C": s["C"], "mse_SF": s["free_KD"]["mse"], "mse_AD": f18["AD"]["mse"], "SF_beats_AD": bool(s["free_KD"]["mse"] < f18["AD"]["mse"]), "SF_over_P4": s["free_KD"]["mse"] / f18["P4"]["mse"],
                             "K_over_K0": s["free_KD"]["K_over_K0"], "D": s["free_KD"]["D"], "where": s["free_KD"]["where"], "n_channel": s["n_channel"], "n_heads": s["n_heads"]})
            n = sum(1 for r in rows if r["SF_beats_AD"])
            sf[f"{al:g}"] = {"n_SF_beats_AD": n, "N": len(rows), "needed": R6_MIN, "holds": bool(n >= R6_MIN), "SF_over_P4_median": med([r["SF_over_P4"] for r in rows]), "K_over_K0_median": med([r["K_over_K0"] for r in rows]),
                             "D_median": med([r["D"] for r in rows]), "n_D_at_bound": sum(1 for r in rows if r["where"] != "interior"), "tiles": rows}
        r6["SF"] = sf
        r6["alpha0_equals_PF"] = {t: fx.get("alpha0_equals_PF", {}).get("pass") for t, fx in fixed.items()}
    out["R6"] = r6
    # ------------------------------------------------------------------ R7: SJ hold-out rule, and the alpha = 0 check against the addendum 18 rows
    r7_rows = []
    for t in TTV_PRIMARY:
        runs = sj.get(t, {})
        if "0" not in runs:
            continue
        tab = {}
        for al, r in runs.items():
            res = r["params"]["result"]
            tab[al] = {"run": r["run"], "C": res["joint_optimum"]["C"], "K": res["joint_optimum"]["K"], "D": res["joint_optimum"]["D"], "mse": res["joint_optimum"]["mse"], "where": res["joint_optimum"]["where"],
                       "n_channel": res["joint_optimum"]["n_channel"], "n_channel_no_chi": res["joint_optimum"]["n_channel_no_chi"], "heldout_mean": res["holdout"]["mse_heldout_mean"],
                       "K_over_K0": res["K_over_K0"], "n_evaluations": res["n_evaluations"], "interval_1pct": res["joint_identifiability"]["level_0.01"]["intervals"]}
        h0 = tab["0"]["heldout_mean"]
        pos = [al for al in tab if float(al) > 0]
        best = min(pos, key=lambda al: tab[al]["heldout_mean"]) if pos else None
        imp = (1 - tab[best]["heldout_mean"] / h0) if best else None
        row = {"tile": t, "alphas": sorted(tab, key=float), "table": tab, "heldout_alpha0": h0, "best_alpha": best, "heldout_improvement_best": imp, "preferred": bool(imp is not None and imp > R7_IMPROVEMENT),
               "insample_ratio_to_alpha0": {al: tab[al]["mse"] / tab["0"]["mse"] for al in tab}, "complete": len(tab) == 3}
        # alpha = 0 against the addendum 18 P4 rows (same tile, same m, K0: mse_joint at common A_crit vs mse_P4_all)
        f18 = fits18[f"{t}:primary"]
        try:
            prof = pd.read_parquet(os.path.join(RESULTS_ROOT, runs["0"]["run"], "c_profile.parquet")).set_index("C")
            a18rows = pd.read_parquet(os.path.join(RESULTS_ROOT, f18["run_part"], "acrit_profile.parquet")).set_index("A_crit")
            common = prof.index.intersection(a18rows.index)
            if len(common):
                rel = np.abs(prof.loc[common, "mse_joint"].to_numpy() / a18rows.loc[common, "mse_P4_all"].to_numpy() - 1)
                relD = np.abs(prof.loc[common, "D_opt"].to_numpy() / a18rows.loc[common, "D_P4_all"].to_numpy() - 1)
                row["alpha0_check"] = {"n_common": int(len(common)), "max_rel_diff_mse": float(np.nanmax(rel)), "max_rel_diff_D": float(np.nanmax(relD)), "a18_P4": {k: f18["P4"][k] for k in ("A_crit", "K", "D", "mse")},
                                       "sj0_optimum": {k: tab["0"][k] for k in ("C", "K", "D", "mse")}, "optimum_rel_diff_mse": abs(tab["0"]["mse"] / f18["P4"]["mse"] - 1),
                                       "same_A_crit": bool(abs(tab["0"]["C"] - f18["P4"]["A_crit"]) < 1e-9), "pass_rows_1e-9": bool(np.nanmax(rel) < 1e-9)}
            else:
                row["alpha0_check"] = {"n_common": 0}
        except Exception as exc:  # noqa: BLE001
            row["alpha0_check"] = {"error": repr(exc)}
        r7_rows.append(row)
    if r7_rows:
        n_pref = sum(1 for r in r7_rows if r["preferred"])
        out["R7"] = {"N": len(r7_rows), "n_preferred": n_pref, "needed": R7_MIN, "preferred_at_TTV": bool(n_pref >= R7_MIN), "best_alpha_per_tile": {r["tile"]: r["best_alpha"] for r in r7_rows},
                     "improvement_median": med([r["heldout_improvement_best"] for r in r7_rows]), "n_complete": sum(1 for r in r7_rows if r["complete"]), "tiles": r7_rows,
                     "alpha0_check_pass_all": all(r.get("alpha0_check", {}).get("pass_rows_1e-9") for r in r7_rows), "alpha0_same_optimum_all": all(r.get("alpha0_check", {}).get("same_A_crit") for r in r7_rows)}
        out["sj_alpha0_check"] = {r["tile"]: r.get("alpha0_check") for r in r7_rows}
    # ------------------------------------------------------------------ R9 (TTV D ratios) and R10 (the bracket) for every fit
    def fit_list(tile: str, site: str) -> list[dict]:
        f18 = fits18.get(f"{tile}:primary") or fits18.get(f"{tile}:offmesa")
        fl = []
        if f18:
            for name in ("AD", "P4", "P3", "A1"):
                o = f18.get(name)
                if o:
                    at_b = bool(o.get("D_at_bound") or o.get("where") in ("y_lo", "y_hi"))
                    fl.append({"model": name, "D": o["D"], "at_bound": at_b, "K": o["K"], "mse": o["mse"]})
        if site == "TTV" and tile in fixed:
            fx = fixed[tile]
            for key, nm in (("A_F", "PF"), ("A_F_table1", "PF_7400"), ("A_F_A10_sensitivity", "PF_A10")):
                o = fx["PF"][key]
                fl.append({"model": nm, "D": o["free_KD"]["D"], "at_bound": o["free_KD"]["where"] != "interior", "K": o["free_KD"]["K"], "mse": o["free_KD"]["mse"]})
                fl.append({"model": nm + "3", "D": o["K_fixed"]["D"], "at_bound": o["K_fixed"]["where"] != "interior", "K": o["K_fixed"]["K"], "mse": o["K_fixed"]["mse"]})
            for al in fx["field_values"]["alphas"]:
                o = fx["SF"][f"alpha_{al:g}:C_F"]
                fl.append({"model": f"SF_a{al:g}", "D": o["free_KD"]["D"], "at_bound": o["free_KD"]["where"] != "interior", "K": o["free_KD"]["K"], "mse": o["free_KD"]["mse"]})
        if site == "TTV" and tile in sj:
            for al, r in sj[tile].items():
                o = r["params"]["result"]["joint_optimum"]
                fl.append({"model": f"SJ_a{al}", "D": o["D"], "at_bound": o["where"] != "interior", "K": o["K"], "mse": o["mse"]})
        return fl

    r10: dict = {}
    r9: dict = {}
    for site in ("GM", "AP", "TTV"):
        per_model: dict[str, dict] = {}
        for tile in PRIMARY[site]:
            tl = partA["tiles"].get(tile)
            if not tl:
                continue
            br = tl.get("bracket")
            for f in fit_list(tile, site):
                cls = bracket_class(f["D"], f["at_bound"], br) if tl["usable"] else "undetermined"
                pm = per_model.setdefault(f["model"], {"inside": 0, "below": 0, "above": 0, "undetermined": 0, "tiles": {}})
                pm[cls] += 1
                pm["tiles"][tile] = {"D": f["D"], "class": cls, "bracket": br, "at_bound": f["at_bound"]}
                if site == "TTV":
                    r9.setdefault(f["model"], {})[tile] = {"D": f["D"], "D_over_D0N_A": f["D"] / tl["D0_N_A"] if tl.get("D0_N_A") else None, "D_over_D0N_B": f["D"] / tl["D0_N_B"] if tl.get("D0_N_B") else None,
                                                           "D_over_D0R_A": f["D"] / tl["D0_R_A"] if tl.get("D0_R_A") else None, "D_over_D0R_B": f["D"] / tl["D0_R_B"] if tl.get("D0_R_B") else None,
                                                           "D_over_D_R": f["D"] / tl["D_R"] if tl.get("D_R") else None, "at_bound": f["at_bound"], "K": f["K"], "mse": f["mse"]}
        for name, pm in per_model.items():
            det = pm["inside"] + pm["below"] + pm["above"]
            pm["n_determined"] = det
            pm["needed"] = hold_n(det)
            pm["consistent_with_linear_diffusion"] = bool(det > 0 and pm["inside"] >= hold_n(det))
            pm["frac_above_of_determined"] = pm["above"] / det if det else None
            pm["reading"] = None
            if site == "TTV" and name not in ("AD", "A1"):
                if det and pm["above"] >= hold_n(det):
                    pm["reading"] = "above on >= 80 % of determined tiles: the excess D is not a linear model compensating for nonlinear hillslopes; read with R5-R8 as a sign that the area-only threshold is mis-specified"
                elif pm["consistent_with_linear_diffusion"]:
                    pm["reading"] = "inside: the excess over D_C is what a linear model needs to match the measured relief; the F3 failure at TTV is not by itself evidence against the model"
                elif det and pm["below"] >= hold_n(det):
                    pm["reading"] = "below (no reading set)"
                else:
                    pm["reading"] = "mixed classes: no reading applies"
        r10[site] = {"withheld_by_R11": bool(site in ("GM", "AP") and not R11[site]["pass"]), "models": per_model, "frac_rule": R10_FRAC}
    out["R10"] = r10
    for name, d in r9.items():
        out["R9"][name] = {"tiles": d, "median_D_over_D0N_A": med([v["D_over_D0N_A"] for v in d.values()]), "median_D_over_D0N_B": med([v["D_over_D0N_B"] for v in d.values()]),
                           "median_D_over_D0R_A": med([v["D_over_D0R_A"] for v in d.values()]), "median_D_over_D_R": med([v["D_over_D_R"] for v in d.values()]), "n_at_bound": sum(1 for v in d.values() if v["at_bound"])}
    # ------------------------------------------------------------------ model-network D_R for the TTV fits (decision 12, report only)
    for tile in TTV_PRIMARY:
        if tile not in fixed and tile not in sj:
            continue
        pix_path = out_path("hilltops", f"{tile}_pixels.parquet")
        if not os.path.exists(pix_path):
            continue
        pix = pd.read_parquet(pix_path)
        if pix.shape[0] == 0:
            continue
        td = metre_tile(tile)
        Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
        rows, cols, seg, C = pix["row"].to_numpy(), pix["col"].to_numpy(), pix["segment"].to_numpy(), pix["C_Lf"].to_numpy()
        E = E_SITE["TTV"]
        nets: dict[str, np.ndarray] = {}
        M_cache: dict[float, np.ndarray] = {}
        if tile in fixed:
            fx = fixed[tile]
            for key, nm in (("A_F", "PF"), ("A_F_table1", "PF_7400")):
                nets[nm] = np.isfinite(A) & (A >= fx["PF"][key]["A_crit"])
            for al in fx["field_values"]["alphas"]:
                M_cache[al] = head_statistic(Z, A, FD, dx, dy, float(al))["M"]
                nets[f"SF_a{al:g}"] = np.isfinite(M_cache[al]) & (M_cache[al] >= fx["SF"][f"alpha_{al:g}:C_F"]["C"])
        for al, r in sj.get(tile, {}).items():
            a_ = float(al)
            if a_ not in M_cache:
                M_cache[a_] = head_statistic(Z, A, FD, dx, dy, a_)["M"]
            nets[f"SJ_a{al}"] = np.isfinite(M_cache[a_]) & (M_cache[a_] >= r["params"]["result"]["joint_optimum"]["C"])
        rec = {}
        for name, target in nets.items():
            tr = trace_to_mask(Z, FD, dx, dy, rows, cols, target)
            used = tr["reached"] & (tr["L_H"] > 0) & (tr["R"] > 0) & (C > 0)
            D_R = np.where(used, E * tr["L_H"] ** 2 / (2 * np.where(tr["R"] > 0, tr["R"], np.nan)), np.nan)
            rec[name] = {"n_used": int(used.sum()), "n_left_tile": int(tr["left_tile"].sum()), "L_H_median_m": float(np.median(tr["L_H"][used])) if used.any() else None,
                         "D_R": segment_bootstrap_median(D_R, seg, N_BOOT)}
            fit_D = next((f["D"] for f in fit_list(tile, "TTV") if f["model"] == name), None)
            rec[name]["fit_D"] = fit_D
            rec[name]["fit_D_over_D_R_model"] = (fit_D / rec[name]["D_R"]["median"]) if (fit_D and rec[name]["D_R"].get("median")) else None
        out["model_network_D_R"][tile] = rec
        print(tile, "model-network D_R:", {k: (round(v["D_R"]["median"], 4) if v["D_R"].get("median") else None, v["fit_D_over_D_R_model"]) for k, v in rec.items()}, f"[{time.time() - t0:.0f} s]", flush=True)
    out["wall_s"] = time.time() - t0
    dump(out, out_path("rules", "partB.json"))
    print("R2 primary:", {k: {kk: (v[kk]["n"], v[kk]["N"], v[kk]["holds"]) for kk in ("F3_N_A", "E1_N_A", "E2_N_A")} for k, v in out["R2"].items() if k.endswith(":primary")})
    print("R6:", {k: (v.get("n_PF_beats_AD"), v.get("holds")) for k, v in r6.items() if k != "SF" and isinstance(v, dict) and "holds" in v}, "SF:", {k: (v["n_SF_beats_AD"], v["holds"]) for k, v in r6.get("SF", {}).items()})
    print("R7:", {k: v for k, v in out["R7"].items() if k in ("n_preferred", "preferred_at_TTV", "alpha0_check_pass_all")} if out["R7"] else None)
    print("R10 TTV:", {k: (v["inside"], v["below"], v["above"], v["undetermined"]) for k, v in r10.get("TTV", {}).get("models", {}).items()})


if __name__ == "__main__":
    main()
