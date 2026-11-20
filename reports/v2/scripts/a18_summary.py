#!/usr/bin/env python3
"""Addendum 18, sections 5-6 and 8: collect every a18 fit run, apply the rules written in section 6, the continuity (5.5), trunk (5.4),
TTV per-tile m (5.6) and channel-head (5.7) comparisons, and draw the Figs. 4-6 equivalents.

Reads <run>/prep/*.json, the latest a18_part / a18_ad run of every spec, and writes <run>/summary/: per_tile_fits.json and .csv
(every variant's optimum per spec), rules.json (F1-F4, E1, E2, H1, H2 per site on the primary and clean groups; counts without verdicts on
the off-mesa and continuity groups), trunk.json, continuity.json, ttv_mt.json, heads.json, cost.json; figures under
reports/v2/figures/a18/ (<site>_fig4_mse.png, <site>_fig5_K.png, <site>_fig6_D.png).
usage: a18_summary.py [--no-figures]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_common import (  # noqa: E402
    AP1_3DEP,
    AP_PAMAP,
    CLEAN,
    F4_LOG10,
    FIG_DIR,
    FIT_STEPS,
    GM_NCALM,
    GM_OFFMESA,
    JOB_DIR,
    LOPSIDED_FRAC,
    M_IMMATERIAL_REL,
    PRIMARY,
    SITE_M,
    TIER2,
    TRUNK_MATERIAL_REL,
    TTV_M_T_EDGE,
    TTV_PRIMARY,
    dump,
    hold_n,
    run_dir,
    spec_key,
)

from lemcvt.results import find_runs  # noqa: E402

VARIANTS_P = ("P1_A", "P1_B", "P2_A", "P2_B", "P3", "P4")
VARIANTS_A = ("AD", "A1", "A2_A", "A2_B")
COL = {"part": "#0072B2", "ad": "#D55E00", "grid": "#d9d9d9", "ink": "#222222", "muted": "#777777"}
MARK = {"primary": ("o", "primary (clean)"), "tier2": ("o", "primary, Tier 2 (not in the clean subset)"), "offmesa": ("^", "off-mesa"),
        "continuity": ("D", "continuity (NCALM / PAMAP / AP1_3dep)")}


def l10(x):
    return math.log10(x) if x is not None and x > 0 else None


def latest_run(step: str, label: str):
    runs = [r for r in find_runs(step, label) if "_pilot_" not in r.name]
    if not runs:
        return None
    r = runs[-1]
    p = json.load(open(r / "params.json"))
    rj = json.load(open(r / "run.json"))
    return r.name, p, rj


# ============================================================================= per-spec records

def collect(specs: dict) -> dict[str, dict]:
    out = {}
    for key, s in specs.items():
        label = f"{s['tile']}_{s['group']}"
        pr = latest_run(FIT_STEPS["part"], label)
        ar = latest_run(FIT_STEPS["ad"], label)
        rec: dict = {"spec": key, "tile": s["tile"], "group": s["group"], "site": s["site"], "m": s["m"], "K0": s["K0"], "D0A": s["D0A"], "D0B": s["D0B"],
                     "has_D0": s["has_D0"], "trunk": s["trunk"], "tier2": s["tile"] in TIER2, "run_part": pr[0] if pr else None, "run_ad": ar[0] if ar else None,
                     "complete": bool(pr and ar)}
        K0, D0A, D0B = s["K0"], s["D0A"], s["D0B"]
        if pr:
            res = pr[1]["result"]
            ins = res["insample"]
            for n in VARIANTS_P:
                o = ins.get(n)
                if not o:
                    continue
                rec[n] = {"A_crit": o["A_crit"], "mse": o["mse"], "K": o["K"], "D": o["D"], "where": o["where"], "n_channel": o["n_channel"], "K_over_K0": o["K"] / K0,
                          "D_over_D0A": (o["D"] / D0A) if D0A else None, "D_over_D0B": (o["D"] / D0B) if D0B else None}
                idt = res["identifiability"].get(n)
                if idt:
                    for lev, tag in (("0.01", "i1"), ("0.05", "i5")):
                        r = idt[f"level_{lev}"]
                        rec[n][tag] = [r["A_crit_min"], r["A_crit_max"]]
                        rec[n][f"{tag}_n_intervals"] = r["n_intervals"]
                        rec[n][f"{tag}_intervals"] = r["intervals"]
            rec["heldout_part"] = {n: {"mean": h["mse_heldout_mean"], "h1_fit_score_h2": h["fit_h1"]["score_h2"], "h2_fit_score_h1": h["fit_h2"]["score_h1"],
                                       "A_crit_h1": h["fit_h1"]["A_crit"], "A_crit_h2": h["fit_h2"]["A_crit"]} for n, h in res["heldout"].items()}
            sp = pr[1]["split"]
            rec["split"] = {"share_h1": sp["share_half1_of_split"], "share_h2": sp["share_half2_of_split"], "frac_valid_in_split": sp["frac_valid_in_split"],
                            "n_basins_edge": sp["n_basins_edge"], "largest_basin_share": sp["largest_basin_share_of_split"], "n_basins_h1": sp["n_basins_half1"], "n_basins_h2": sp["n_basins_half2"]}
            rec["lopsided"] = bool(min(sp["share_half1_of_split"], sp["share_half2_of_split"]) < LOPSIDED_FRAC)
            rec["phase6_check_part"] = res.get("phase6_check")
            rec["union_P4"] = (res.get("insample_union_phase6") or {}).get("P4")
            rec["trunk_info"] = pr[1].get("trunk")
            rec["n_evaluations_part"] = res["n_evaluations"]
            rec["fill"] = {k: res["fill"].get(k) for k in ("n_targets", "n_evaluated", "capped")}
            rec["timing_part"] = res["timing"]
            rec["git_part"] = pr[2].get("git_hash")
            rec["job_part"] = pr[2].get("slurm_job_id")
        if ar:
            res = ar[1]["result"]
            ins = res["insample"]
            for n in VARIANTS_A:
                o = ins.get(n)
                if not o:
                    continue
                rec[n] = {"mse": o["mse"], "K": o["K"], "D": o["D"], "r": o["r"], "where": o.get("where"), "K_at_bound": o["K_at_bound"], "D_at_bound": o["D_at_bound"],
                          "K_over_K0": o["K"] / K0, "D_over_D0A": (o["D"] / D0A) if D0A else None, "D_over_D0B": (o["D"] / D0B) if D0B else None}
            h = res["heldout"].get("AD")
            rec["heldout_AD"] = {"mean": h["mse_heldout_mean"], "h1_fit_score_h2": h["fit_h1"]["score_h2"], "h2_fit_score_h1": h["fit_h2"]["score_h1"]} if h else None
            rec["phase6_check_ad"] = res.get("phase6_check")
            rec["n_evaluations_ad"] = res["n_evaluations"]
            rec["timing_ad"] = res["timing"]
            rec["git_ad"] = ar[2].get("git_hash")
            rec["job_ad"] = ar[2].get("slurm_job_id")
        if rec.get("P4") and rec.get("AD"):
            rec["R_MSE"] = rec["AD"]["mse"] / rec["P4"]["mse"]
            rec["F1"] = bool(rec["P4"]["mse"] < rec["AD"]["mse"])
            rec["F2"] = bool(abs(l10(rec["P4"]["K_over_K0"])) < abs(l10(rec["AD"]["K_over_K0"])))
            for end in ("A", "B"):
                if rec["P4"].get(f"D_over_D0{end}") and rec["AD"].get(f"D_over_D0{end}"):
                    rec[f"F3_{end}"] = bool(abs(l10(rec["P4"][f"D_over_D0{end}"])) < abs(l10(rec["AD"][f"D_over_D0{end}"])))
                p1, a2 = rec.get(f"P1_{end}"), rec.get(f"A2_{end}")
                if p1 and a2 and rec.get("A1"):
                    rec[f"E1_{end}"] = bool(p1["mse"] < min(rec["A1"]["mse"], a2["mse"]))
                p2 = rec.get(f"P2_{end}")
                if p2:
                    rec[f"E2_P2_{end}"] = bool(p2["mse"] < rec["AD"]["mse"])
            rec["E2_P3"] = bool(rec["P3"]["mse"] < rec["AD"]["mse"])
            if rec.get("heldout_AD") and rec.get("heldout_part", {}).get("P4"):
                rec["H1"] = bool(rec["heldout_part"]["P4"]["mean"] < rec["heldout_AD"]["mean"])
                for end in ("A", "B"):
                    if rec["heldout_part"].get(f"P2_{end}"):
                        rec[f"H2_{end}"] = bool(rec["heldout_part"][f"P2_{end}"]["mean"] < rec["heldout_AD"]["mean"])
        out[key] = rec
    return out


# ============================================================================= rules (section 6)

def _count(recs: list[dict], key: str, denom_key: str | None = None) -> dict:
    """{n, N, needed, holds} over the records that carry ``key`` (and, when given, that have a usable D0)."""
    sel = [r for r in recs if key in r and (denom_key is None or r.get(denom_key))]
    n = sum(1 for r in sel if r[key])
    N = len(sel)
    return {"n": n, "N": N, "needed": hold_n(N), "holds": bool(N > 0 and n >= hold_n(N)), "tiles_fail": [r["tile"] for r in sel if not r[key]], "tiles_missing": [r["tile"] for r in recs if key not in r]}


def _med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _rng(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return [float(min(v)), float(max(v))] if v else None


def site_rules(recs: list[dict], verdicts: bool = True) -> dict:
    recs = [r for r in recs if r.get("P4") and r.get("AD")]
    out: dict = {"N": len(recs), "N_D0": sum(1 for r in recs if r["has_D0"]), "tiles": [r["tile"] for r in recs]}
    out["F1"] = _count(recs, "F1") | {"R_MSE_median": _med([r["R_MSE"] for r in recs]), "R_MSE_range": _rng([r["R_MSE"] for r in recs])}
    out["F2"] = _count(recs, "F2") | {"K_over_K0_median": {"P4": _med([r["P4"]["K_over_K0"] for r in recs]), "AD": _med([r["AD"]["K_over_K0"] for r in recs])},
                                      "K_over_K0_range": {"P4": _rng([r["P4"]["K_over_K0"] for r in recs]), "AD": _rng([r["AD"]["K_over_K0"] for r in recs])}}
    for end in ("A", "B"):
        d = _count(recs, f"F3_{end}", "has_D0")
        d["D_over_D0_median"] = {"P4": _med([r["P4"].get(f"D_over_D0{end}") for r in recs if r["has_D0"]]), "AD": _med([r["AD"].get(f"D_over_D0{end}") for r in recs if r["has_D0"]])}
        d["D_over_D0_range"] = {"P4": _rng([r["P4"].get(f"D_over_D0{end}") for r in recs if r["has_D0"]]), "AD": _rng([r["AD"].get(f"D_over_D0{end}") for r in recs if r["has_D0"]])}
        out[f"F3_{end}"] = d
    f4 = {}
    for end in ("A", "B"):
        v = [abs(l10(r["AD"][f"D_over_D0{end}"])) for r in recs if r["has_D0"] and r["AD"].get(f"D_over_D0{end}")]
        f4[end] = {"AD_median_abs_log10_D_over_D0": _med(v), "n": len(v), "ge_1": bool(v and _med(v) >= F4_LOG10)}
    f4["holds"] = bool(f4["A"]["ge_1"] and f4["B"]["ge_1"])
    out["F4"] = f4
    for end in ("A", "B"):
        out[f"E1_{end}"] = _count(recs, f"E1_{end}", "has_D0")
        out[f"E2_P2_{end}"] = _count(recs, f"E2_P2_{end}", "has_D0")
    out["E2_P3"] = _count(recs, "E2_P3")
    bal = [r for r in recs if not r.get("lopsided")]
    out["n_lopsided"] = sum(1 for r in recs if r.get("lopsided"))
    out["lopsided_tiles"] = [r["tile"] for r in recs if r.get("lopsided")]
    out["H1"] = {"all": _count(recs, "H1"), "balanced": _count(bal, "H1")}
    for end in ("A", "B"):
        out[f"H2_{end}"] = {"all": _count(recs, f"H2_{end}", "has_D0"), "balanced": _count(bal, f"H2_{end}", "has_D0")}
    if not verdicts:
        for _k, v in list(out.items()):
            if isinstance(v, dict) and "holds" in v:
                v["holds"] = None
                v["verdict"] = "report only"
    return out


def rule_diff(a: dict, b: dict) -> list[str]:
    diffs = []
    for k in ("F1", "F2", "F3_A", "F3_B", "E1_A", "E1_B", "E2_P2_A", "E2_P2_B", "E2_P3"):
        if a.get(k, {}).get("holds") != b.get(k, {}).get("holds"):
            diffs.append(f"{k}: primary {a[k]['n']}/{a[k]['N']} ({'holds' if a[k]['holds'] else 'fails'}) -> clean {b[k]['n']}/{b[k]['N']} ({'holds' if b[k]['holds'] else 'fails'})")
    if a["F4"]["holds"] != b["F4"]["holds"]:
        diffs.append("F4: " + ("holds" if a["F4"]["holds"] else "fails") + " -> " + ("holds" if b["F4"]["holds"] else "fails"))
    for k in ("H1", "H2_A", "H2_B"):
        for sub in ("all", "balanced"):
            if a.get(k, {}).get(sub, {}).get("holds") != b.get(k, {}).get(sub, {}).get("holds"):
                diffs.append(f"{k} ({sub}): {a[k][sub]['n']}/{a[k][sub]['N']} -> {b[k][sub]['n']}/{b[k][sub]['N']}")
    return diffs


# ============================================================================= trunk, continuity, TTV, heads

def trunk_compare(fits: dict, specs: dict) -> dict:
    out = {}
    for key, s in specs.items():
        if s["group"] != "trunk":
            continue
        t = s["tile"]
        base_key = next(k for k in (spec_key(t, "primary"), spec_key(t, "ncalm"), spec_key(t, "pamap"), spec_key(t, "ap1_3dep")) if k in fits)
        b, r = fits[base_key], fits[key]
        if not (b.get("P4") and r.get("P4") and b.get("AD") and r.get("AD")):
            out[t] = {"incomplete": True, "base": base_key}
            continue
        ch = {"K_over_K0": r["P4"]["K_over_K0"] / b["P4"]["K_over_K0"] - 1, "A_crit": r["P4"]["A_crit"] / b["P4"]["A_crit"] - 1, "R_MSE": r["R_MSE"] / b["R_MSE"] - 1}
        if b["P4"].get("D_over_D0A") and r["P4"].get("D_over_D0A"):
            ch["D_over_D0A"] = r["P4"]["D_over_D0A"] / b["P4"]["D_over_D0A"] - 1
        flip = b["F1"] != r["F1"]
        out[t] = {"base": base_key, "trunk_cells": r["trunk_info"], "base_P4": b["P4"], "trunk_P4": r["P4"], "base_AD": b["AD"], "trunk_AD": r["AD"], "base_R_MSE": b["R_MSE"], "trunk_R_MSE": r["R_MSE"],
                  "rel_changes": ch, "F1_base": b["F1"], "F1_trunk": r["F1"], "F1_flips": flip,
                  "material": bool(flip or any(abs(v) > TRUNK_MATERIAL_REL for v in ch.values())), "runs": {"part": r["run_part"], "ad": r["run_ad"]}}
    return out


def continuity(fits: dict, gate: dict | None) -> dict:
    out: dict = {"GM": {"pairs": [], "note": "NCALM GM1-GM7 (group ncalm, m 0.29) against FEMA GM1_fema-GM7_fema (primary); ratio = FEMA / NCALM"},
                 "AP": {"pairs": [], "note": "PAMAP AP1-AP5 (group pamap, m 0.47) against 3DEP AP1_3dep (ap1_3dep) and AP2_3dep-AP5_3dep (primary); ratio = 3DEP / PAMAP"}}
    keys = ("K", "D", "A_crit", "MSE_P4", "MSE_AD", "R_MSE")

    def vals(r):
        return {"K": r["P4"]["K"], "D": r["P4"]["D"], "A_crit": r["P4"]["A_crit"], "MSE_P4": r["P4"]["mse"], "MSE_AD": r["AD"]["mse"], "R_MSE": r["R_MSE"],
                "AD_K": r["AD"]["K"], "AD_D": r["AD"]["D"]}

    for site, pairs in (("GM", [(spec_key(f"GM{n}", "ncalm"), spec_key(f"GM{n}_fema", "primary")) for n in range(1, 8)]),
                        ("AP", [(spec_key("AP1", "pamap"), spec_key("AP1_3dep", "ap1_3dep"))] + [(spec_key(f"AP{n}", "pamap"), spec_key(f"AP{n}_3dep", "primary")) for n in range(2, 6)])):
        for old, new in pairs:
            o, n = fits.get(old), fits.get(new)
            if not (o and n and o.get("P4") and n.get("P4") and o.get("AD") and n.get("AD")):
                out[site]["pairs"].append({"old": old, "new": new, "incomplete": True})
                continue
            vo, vn = vals(o), vals(n)
            out[site]["pairs"].append({"old": old, "new": new, "old_values": vo, "new_values": vn, "ratios": {k: vn[k] / vo[k] for k in keys} | {"AD_K": vn["AD_K"] / vo["AD_K"], "AD_D": vn["AD_D"] / vo["AD_D"]}})
        done = [p for p in out[site]["pairs"] if not p.get("incomplete")]
        out[site]["median_ratio"] = {k: _med([p["ratios"][k] for p in done]) for k in keys + ("AD_K", "AD_D")}
        out[site]["largest_single_tile_abs_log10"] = {k: (max(done, key=lambda p: abs(l10(p["ratios"][k])))["old"], max(abs(l10(p["ratios"][k])) for p in done)) if done else None for k in keys}
    # v2.0.0 match (5.5): GM1-GM5 at 0.31 (v2m) and AP1-AP5 at 0.47 (pamap)
    v2 = {}
    for key in [spec_key(f"GM{n}", "v2m") for n in range(1, 6)] + [spec_key(f"AP{n}", "pamap") for n in range(1, 6)]:
        r = fits.get(key)
        if not r:
            continue
        pc, ac = r.get("phase6_check_part") or {}, r.get("phase6_check_ad") or {}
        rec = {"m": r["m"], "part_run": r["run_part"], "ad_run": r["run_ad"]}
        if pc.get("available") and pc.get("same_model"):
            rec["P4_rel_diff_union"] = pc["rel_diff_union_vs_phase6"]
            rec["P4_rel_diff_new_rows_only"] = pc["rel_diff_new_rows_only_vs_phase6"]
            rec["P4_match_1e-9"] = bool(all(v <= 1e-9 for v in pc["rel_diff_union_vs_phase6"].values()))
            rec["phase6_run_part"] = pc["run"]
        if ac.get("available"):
            rec["AD_rel_diff"] = ac["rel_diff"]
            rec["AD_match_1e-9"] = bool(all(v <= 1e-9 for v in ac["rel_diff"].values()))
            rec["phase6_run_ad"] = ac["run"]
        v2[key] = rec
    out["v2_match"] = v2
    out["gate"] = gate
    return out


def ttv_mt(fits: dict, ttv_prep: dict) -> dict:
    rows = []
    for t in TTV_PRIMARY:
        a, b = fits.get(spec_key(t, "primary")), fits.get(spec_key(t, "mt"))
        if not (a and b and a.get("P4") and b.get("P4") and a.get("AD") and b.get("AD")):
            rows.append({"tile": t, "incomplete": True})
            continue
        for which, r in (("site", a), ("m_t", b)):
            rows.append({"tile": t, "which": which, "m": r["m"], "K0": r["K0"], "A_crit": r["P4"]["A_crit"], "i1": r["P4"].get("i1"), "K_over_K0": r["P4"]["K_over_K0"],
                         "D_over_D0A": r["P4"].get("D_over_D0A"), "D_over_D0B": r["P4"].get("D_over_D0B"), "mse_P4": r["P4"]["mse"], "mse_AD": r["AD"]["mse"], "R_MSE": r["R_MSE"],
                         "AD_K_over_K0": r["AD"]["K_over_K0"], "AD_D_over_D0A": r["AD"].get("D_over_D0A"), "F1": r["F1"], "m_t_on_grid_edge": t in TTV_M_T_EDGE and which == "m_t",
                         "run_part": r["run_part"], "run_ad": r["run_ad"]})
    done = [r for r in rows if not r.get("incomplete")]
    site_rows = [r for r in done if r["which"] == "site"]
    mt_rows = [r for r in done if r["which"] == "m_t"]
    f1 = _count([{"tile": r["tile"], "F1": r["F1"]} for r in mt_rows], "F1")
    med = {}
    for col in ("D_over_D0A", "AD_K_over_K0", "K_over_K0", "AD_D_over_D0A", "D_over_D0B"):
        m0, m1 = _med([r[col] for r in site_rows]), _med([r[col] for r in mt_rows])
        med[col] = {"site_m": m0, "m_t": m1, "rel_change": (m1 / m0 - 1) if (m0 and m1) else None}
    c2 = bool(med["D_over_D0A"]["rel_change"] is not None and abs(med["D_over_D0A"]["rel_change"]) < M_IMMATERIAL_REL
              and med["AD_K_over_K0"]["rel_change"] is not None and abs(med["AD_K_over_K0"]["rel_change"]) < M_IMMATERIAL_REL)
    return {"rows": rows, "rule_6_9": {"F1_at_m_t": f1, "medians": med, "cond_F1": f1["holds"], "cond_medians": c2, "immaterial": bool(f1["holds"] and c2)},
            "prep": ttv_prep}


def heads(fits: dict, heads_prep: dict) -> dict:
    t1 = heads_prep["table1"]["dist"]
    d77 = heads_prep["digitised"]["dist_max_within_3m"]
    iqr1, iqr77 = t1["iqr"], d77["iqr"]

    def classify(i1, iqr):
        lo, hi = i1
        if lo > iqr[1]:
            return "above"
        if hi < iqr[0]:
            return "below"
        return "consistent"

    out: dict = {"table1_iqr": iqr1, "digitised_iqr": iqr77, "classification": {}}
    for which, group in (("site_m", "primary"), ("m_t", "mt")):
        rows = []
        for t in TTV_PRIMARY:
            r = fits.get(spec_key(t, group))
            if not (r and r.get("P4") and r["P4"].get("i1")):
                rows.append({"tile": t, "incomplete": True})
                continue
            rows.append({"tile": t, "m": r["m"], "A_crit": r["P4"]["A_crit"], "i1": r["P4"]["i1"], "i5": r["P4"].get("i5"), "class_table1": classify(r["P4"]["i1"], iqr1),
                         "class_77": classify(r["P4"]["i1"], iqr77), "md_study_area": t in ("TTV1", "TTV2", "TTV3")})
        done = [x for x in rows if not x.get("incomplete")]
        counts = {c: sum(1 for x in done if x["class_table1"] == c) for c in ("consistent", "above", "below")}
        md = [x for x in done if x["md_study_area"]]
        counts_md = {"table1": {c: sum(1 for x in md if x["class_table1"] == c) for c in ("consistent", "above", "below")},
                     "digitised_77": {c: sum(1 for x in md if x["class_77"] == c) for c in ("consistent", "above", "below")}, "N": len(md)}
        n = len(done)
        reading = ("agrees with the field channel heads" if counts["consistent"] >= 8 else "lies above the field channel heads at the 1 km tile scale" if counts["above"] >= 6 else "split")
        out["classification"][which] = {"rows": rows, "counts": counts, "N": n, "of_11": 11, "md_study_area": counts_md, "reading": reading,
                                        "rule": "consistent >= 8 of 11 -> agrees; above >= 6 of 11 -> above; otherwise the split is reported"}
    out["field"] = {"table1": t1, "digitised": {k: d77[k] for k in d77}, "digitised_at_cell": heads_prep["digitised"]["dist_at_head_cell"], "comparison": heads_prep["comparison"]}
    return out


# ============================================================================= figures (section 8)

def figures(fits: dict, out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": COL["muted"], "axes.linewidth": 0.8, "xtick.color": COL["ink"], "ytick.color": COL["ink"]})

    def kind_of(r):
        if r["group"] == "primary":
            return "tier2" if r["tier2"] else "primary"
        if r["group"] == "offmesa":
            return "offmesa"
        return "continuity"

    for site in ("GM", "AP", "TTV"):
        groups = ["primary", "offmesa", "ncalm"] if site == "GM" else ["primary", "pamap", "ap1_3dep"] if site == "AP" else ["primary"]
        recs = [r for r in fits.values() if r["site"] == site and r["group"] in groups and r.get("P4") and r.get("AD")]
        if not recs:
            continue
        # Fig. 4: MSE_AD against MSE_P4
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        allv = [r["P4"]["mse"] for r in recs] + [r["AD"]["mse"] for r in recs]
        lo, hi = min(allv) * 0.7, max(allv) * 1.4
        ax.plot([lo, hi], [lo, hi], color=COL["grid"], lw=1, zorder=1)
        used = set()
        for r in recs:
            k = kind_of(r)
            used.add(k)
            ax.scatter(r["P4"]["mse"], r["AD"]["mse"], marker=MARK[k][0], s=34, facecolor=COL["part"] if k != "tier2" else "white", edgecolor=COL["part"], linewidths=1.0, zorder=3)
            if r["group"] == "primary" and (r["R_MSE"] == max(x["R_MSE"] for x in recs if x["group"] == "primary") or r["R_MSE"] == min(x["R_MSE"] for x in recs if x["group"] == "primary")):
                ax.annotate(r["tile"], (r["P4"]["mse"], r["AD"]["mse"]), xytext=(4, 3), textcoords="offset points", fontsize=7, color=COL["ink"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("MSE of the partitioned joint fit P4 (m²)")
        ax.set_ylabel("MSE of the AD fit (m²)")
        ax.grid(True, color=COL["grid"], lw=0.8)
        ax.set_axisbelow(True)
        prim = [r for r in recs if r["group"] == "primary"]
        ax.set_title(f"{site}: AD against partitioned MSE (1:1 line)\nprimary tiles: P4 < AD on {sum(r['F1'] for r in prim)}/{len(prim)}, "
                     f"median R_MSE {np.median([r['R_MSE'] for r in prim]):.2f}", fontsize=8)
        ax.legend(handles=[Line2D([], [], marker=MARK[k][0], color="none", markerfacecolor=COL["part"] if k != "tier2" else "white", markeredgecolor=COL["part"], markersize=6, label=MARK[k][1]) for k in MARK if k in used],
                  fontsize=7, frameon=False, loc="upper left")
        p = os.path.join(out_dir, f"{site}_fig4_mse.png")
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)
        # Fig. 5: K/K0 per tile, both models; Fig. 6: D/D0 at both ends
        order = sorted(recs, key=lambda r: (groups.index(r["group"]), r["tile"]))
        for fig_name, key, ylabel, ends in (("fig5_K", "K_over_K0", "K / K₀", [None]), ("fig6_D", "D_over_D0", "D / D₀", ["A", "B"])):
            nrow = len(ends)
            fig, axes = plt.subplots(nrow, 1, figsize=(max(6.0, 0.28 * len(order) + 1.5), 3.0 * nrow + 0.6), sharex=True)
            axes = np.atleast_1d(axes)
            x = np.arange(len(order))
            for ax, end in zip(axes, ends, strict=True):
                col = key if end is None else f"{key}{end}"
                for model, c, dxo in (("P4", COL["part"], -0.17), ("AD", COL["ad"], 0.17)):
                    for i, r in enumerate(order):
                        v = r[model].get(col)
                        if v is None:
                            continue
                        k = kind_of(r)
                        ax.scatter(x[i] + dxo, v, marker=MARK[k][0], s=26, facecolor=c if k != "tier2" else "white", edgecolor=c, linewidths=0.9, zorder=3)
                ax.axhline(1.0, color=COL["grid"], lw=1, zorder=1)
                ax.set_yscale("log")
                ax.set_ylabel(ylabel + (f",{end}" if end else ""))
                ax.grid(True, axis="y", color=COL["grid"], lw=0.8)
                ax.set_axisbelow(True)
                pr = [r for r in order if r["group"] == "primary" and r[("P4")].get(col)]
                ax.set_title(f"{site}: {ylabel}{',' + end if end else ''} per tile; primary medians P4 {np.median([r['P4'][col] for r in pr]):.2f}, AD {np.median([r['AD'][col] for r in pr]):.2f} (n = {len(pr)})", fontsize=8)
                if end == "A" or end is None:
                    hs = [Line2D([], [], marker="o", color="none", markerfacecolor=COL["part"], markeredgecolor=COL["part"], markersize=6, label="partitioned P4"),
                          Line2D([], [], marker="o", color="none", markerfacecolor=COL["ad"], markeredgecolor=COL["ad"], markersize=6, label="AD")]
                    hs += [Line2D([], [], marker=MARK[k][0], color="none", markerfacecolor="grey" if k != "tier2" else "white", markeredgecolor="grey", markersize=6, label=MARK[k][1]) for k in MARK if any(kind_of(r) == k for r in order)]
                    ax.legend(handles=hs, fontsize=7, frameon=False, ncol=3, loc="upper left")
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels([r["tile"] + (" [T2]" if r["tier2"] else "") for r in order], rotation=90, fontsize=6.5)
            # mark the group boundaries
            for ax in axes:
                for i in range(1, len(order)):
                    if order[i]["group"] != order[i - 1]["group"]:
                        ax.axvline(i - 0.5, color=COL["muted"], lw=0.6, ls=":")
            p = os.path.join(out_dir, f"{site}_{fig_name}.png")
            fig.tight_layout()
            fig.savefig(p, dpi=200)
            plt.close(fig)
            paths.append(p)
    return paths


# ============================================================================= cost

def cost(jobs_a: list[str], jobs_b: dict) -> dict:
    ids = list(jobs_a) + [v["job_id"] for v in jobs_b.values()]
    if not ids:
        return {}
    out = subprocess.check_output(["sacct", "-j", ",".join(ids), "-X", "-P", "-n", "--format=JobID,JobName,State,ElapsedRaw,AllocCPUS,TimelimitRaw,MaxRSS"], text=True)
    rows = []
    for line in out.strip().splitlines():
        jid, name, st, el, cpus, lim, rss = (line.split("|") + [""] * 7)[:7]
        rows.append({"job": jid, "name": name, "state": st, "elapsed_s": float(el or 0), "cpus": int(cpus or 0), "limit_min": float(lim or 0)})
    used = sum(r["elapsed_s"] * r["cpus"] for r in rows) / 3600
    lim = sum(r["limit_min"] / 60 * r["cpus"] for r in rows)
    by = {}
    for r in rows:
        b = by.setdefault(r["name"], {"tasks": 0, "used_h": 0.0, "limits_h": 0.0, "states": {}})
        b["tasks"] += 1
        b["used_h"] += r["elapsed_s"] * r["cpus"] / 3600
        b["limits_h"] += r["limit_min"] / 60 * r["cpus"]
        b["states"][r["state"]] = b["states"].get(r["state"], 0) + 1
    return {"used_core_hours": used, "limits_core_hours": lim, "n_tasks": len(rows), "by_job_name": by, "jobs_part_a": jobs_a, "jobs_part_b": jobs_b}


# ============================================================================= main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    rd = run_dir()
    sd = os.path.join(rd, "summary")
    os.makedirs(sd, exist_ok=True)
    pre = {n: json.load(open(os.path.join(rd, "prep", f"{n}.json"))) for n in ("fit_specs", "trunk", "ttv_mt", "heads", "samepixel", "gm22", "configs")}
    specs = pre["fit_specs"]["specs"]
    gate = json.load(open(os.path.join(rd, "gate", "gate.json"))) if os.path.exists(os.path.join(rd, "gate", "gate.json")) else None
    fits = collect(specs)
    dump(fits, os.path.join(sd, "per_tile_fits.json"))
    flat = []
    for _k, r in fits.items():
        row = {c: r.get(c) for c in ("spec", "tile", "group", "site", "m", "K0", "D0A", "D0B", "has_D0", "trunk", "tier2", "complete", "R_MSE", "F1", "F2", "F3_A", "F3_B", "E1_A", "E1_B",
                                     "E2_P2_A", "E2_P2_B", "E2_P3", "H1", "H2_A", "H2_B", "lopsided", "run_part", "run_ad")}
        for n in VARIANTS_P:
            if r.get(n):
                row.update({f"{n}_A_crit": r[n]["A_crit"], f"{n}_mse": r[n]["mse"], f"{n}_K": r[n]["K"], f"{n}_D": r[n]["D"], f"{n}_K_over_K0": r[n]["K_over_K0"],
                            f"{n}_D_over_D0A": r[n]["D_over_D0A"], f"{n}_D_over_D0B": r[n]["D_over_D0B"]})
        if r.get("P4"):
            row.update({"P4_i1_lo": r["P4"].get("i1", [None, None])[0], "P4_i1_hi": r["P4"].get("i1", [None, None])[1], "P4_i5_lo": r["P4"].get("i5", [None, None])[0],
                        "P4_i5_hi": r["P4"].get("i5", [None, None])[1]})
        for n in VARIANTS_A:
            if r.get(n):
                row.update({f"{n}_mse": r[n]["mse"], f"{n}_K": r[n]["K"], f"{n}_D": r[n]["D"], f"{n}_K_over_K0": r[n]["K_over_K0"], f"{n}_D_over_D0A": r[n]["D_over_D0A"], f"{n}_D_over_D0B": r[n]["D_over_D0B"]})
        if r.get("heldout_part"):
            for n, h in r["heldout_part"].items():
                row[f"heldout_{n}"] = h["mean"]
        if r.get("heldout_AD"):
            row["heldout_AD"] = r["heldout_AD"]["mean"]
        if r.get("split"):
            row.update({"share_h1": r["split"]["share_h1"], "share_h2": r["split"]["share_h2"]})
        flat.append(row)
    pd.DataFrame(flat).to_csv(os.path.join(sd, "per_tile_fits.csv"), index=False)
    # rules per site: primary (verdicts), clean (recomputed), off-mesa / continuity (counts only)
    rules: dict = {}
    for site in ("GM", "AP", "TTV"):
        prim = [fits[spec_key(t, "primary")] for t in PRIMARY[site] if spec_key(t, "primary") in fits]
        clean = [r for r in prim if r["tile"] in CLEAN[site]]
        rp, rc = site_rules(prim), site_rules(clean)
        rules[site] = {"primary": rp, "clean": rc, "clean_differences": rule_diff(rp, rc)}
        if site == "GM":
            rules[site]["offmesa"] = site_rules([fits[spec_key(t, "offmesa")] for t in GM_OFFMESA if spec_key(t, "offmesa") in fits], verdicts=False)
            rules[site]["ncalm"] = site_rules([fits[spec_key(t, "ncalm")] for t in GM_NCALM if spec_key(t, "ncalm") in fits], verdicts=False)
        if site == "AP":
            rules[site]["pamap"] = site_rules([fits[spec_key(t, "pamap")] for t in AP_PAMAP if spec_key(t, "pamap") in fits], verdicts=False)
            rules[site]["ap1_3dep"] = site_rules([fits[spec_key(t, "ap1_3dep")] for t in AP1_3DEP if spec_key(t, "ap1_3dep") in fits], verdicts=False)
    dump(rules, os.path.join(sd, "rules.json"))
    dump(trunk_compare(fits, specs), os.path.join(sd, "trunk.json"))
    dump(continuity(fits, gate), os.path.join(sd, "continuity.json"))
    dump(ttv_mt(fits, pre["ttv_mt"]), os.path.join(sd, "ttv_mt.json"))
    dump(heads(fits, pre["heads"]), os.path.join(sd, "heads.json"))
    jobs_a = [kv.split("=")[1] for kv in open(os.path.join(JOB_DIR, "JOBS_partA.txt")).read().split()] if os.path.exists(os.path.join(JOB_DIR, "JOBS_partA.txt")) else []
    jobs_b = json.load(open(os.path.join(JOB_DIR, "JOBS_partB.json"))) if os.path.exists(os.path.join(JOB_DIR, "JOBS_partB.json")) else {}
    dump(cost(jobs_a, jobs_b), os.path.join(sd, "cost.json"))
    incomplete = [k for k, r in fits.items() if not r["complete"]]
    if not a.no_figures:
        figures(fits, FIG_DIR)
    dump({"n_specs": len(specs), "n_complete": len(specs) - len(incomplete), "incomplete": incomplete, "wall_s": time.time() - t0, "site_m": SITE_M}, os.path.join(sd, "status.json"))
    print(f"summary: {len(specs) - len(incomplete)}/{len(specs)} specs complete; incomplete {incomplete[:20]}{'...' if len(incomplete) > 20 else ''}", flush=True)
    for site, r in rules.items():
        rp = r["primary"]
        print(site, "F1", rp["F1"]["n"], "/", rp["F1"]["N"], "F2", rp["F2"]["n"], "F3_A", rp["F3_A"]["n"], "/", rp["F3_A"]["N"], "E2_P3", rp["E2_P3"]["n"], "H1", rp["H1"]["all"]["n"], "clean diffs", r["clean_differences"])


if __name__ == "__main__":
    main()
