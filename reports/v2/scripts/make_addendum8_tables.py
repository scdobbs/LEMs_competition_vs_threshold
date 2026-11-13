"""Appendix tables for the addendum 8 report (D1-D8), from the run directories only.

Usage: PYTHONNOUSERSITE=1 LEMCVT_RESULTS=... python reports/v2/scripts/make_addendum8_tables.py > appendix.md
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ.get("LEMCVT_RESULTS", "/home/groups/hilley/lemcvt_results"))
TILES = ["AP1", "AP2", "AP3", "AP4", "AP5", "GM1", "GM2", "GM3", "GM4", "GM5"]
NETS = ("primary_1e3", "sensitivity_1e4", "rollover")
out: list[str] = []


def latest(pattern: str) -> Path:
    c = sorted(d for d in root.glob(pattern) if (d / "run.json").exists())
    if not c:
        raise FileNotFoundError(pattern)
    return c[-1]


def P(d: Path) -> dict:
    return json.loads((d / "params.json").read_text())


def ci(v, f=".3f"):
    return f"[{v[0]:{f}}, {v[1]:{f}}]" if v else "-"


# ------------------------------------------------------------------ D1 curve verification
er = latest("*_er_star_10tiles")
pe = P(er)
cv = pe["curve_verification"]
out.append(f"## D1. Section 2.1: the steady nonlinear E*-R* curve against the integrated profile\n\nRun `{er.name}`. Max relative difference {cv['max_rel_diff']:.1e} "
           f"(tolerance {cv['tol']:g}); R*/E* at E* = 1e-3: {cv['linear_limit_R_over_E']:.7f} (the linear limit is E*/4).\n")
out.append("| E* | R* published form | R* integrated | relative difference |\n|---|---|---|---|")
for r in cv["rows"]:
    out.append(f"| {r['E_star']:g} | {r['R_published']:.10f} | {r['R_integrated']:.10f} | {r['rel_diff']:.1e} |")
out.append("")

# ------------------------------------------------------------------ D2 per tile and network
px = pd.read_parquet(er / "hilltop_pixels.parquet")
out.append("## D2. Sections 2.2-2.3: per tile and network\n\nratio = 2 Sbar / (C_ht L_H) (1 on a linear hillslope), median with segment-bootstrap 68 % interval; literal = Sbar / (C_ht L_H). "
           "S_c by least squares of log10 R* against the curve, with the segment-bootstrap 68 % interval and full range. Pass = ratio median < 0.67 with 68 % upper bound < 1, and S_c inside its 68 % interval and < 1.27.\n")
out.append("| tile | network | A_c (m^2) | hilltop px | used | left tile | L_H median (m) | Sbar median | C_ht median (1/m) | ratio (68 %) | literal | S_c (68 %) | S_c range | log10 rms | E* median at S_c | frac R* > 1 | pass |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for net in NETS:
    for t in TILES:
        n = pe["tiles"][t]["networks"][net]
        s = px[(px.tile == t) & (px.network == net) & px.used]
        sc = n["S_c_fit"]
        out.append(f"| {t} | {net} | {n['A_c']:.4g} | {n['n_pixels']} | {n['n_used']} | {n['n_left_tile']} | {n['L_H_median_m']:.1f} | {n['Sbar_median']:.3f} | "
                   f"{np.nanmedian(s.C_ht):.4f} | {n['ratio_normalised']['median']:.3f} {ci(n['ratio_normalised'].get('ci68'))} | {n['ratio_literal']['median']:.3f} | "
                   f"{sc['S_c']:.3f} {ci(sc.get('S_c_ci68'))} | {ci(sc.get('S_c_bootstrap_range'))} | {sc['misfit_log10_rms']:.3f} | {n['E_star_median_at_fit']:.2f} | "
                   f"{float(np.mean(s.Sbar > sc['S_c'])):.3f} | {'yes' if n['verdict']['pass'] else 'no'} |")
out.append("")
out.append("Site rule (>= 4 of 5 tiles) and the site-pooled values:\n")
out.append("| network | site | tiles passing | (a) ratio | (b) S_c | supported | pooled ratio (68 %) | pooled S_c (68 %) |\n|---|---|---|---|---|---|---|---|")
for net in NETS:
    for site in ("AP", "GM"):
        r = pe["site_rule"][net][site]
        pl = pe["pooled"][site][net]
        out.append(f"| {net} | {site} | {r['n_pass']} | {r['n_a_pass']} | {r['n_b_pass']} | **{r['supported']}** | {pl['ratio_normalised']['median']:.3f} {ci(pl['ratio_normalised'].get('ci68'))} | "
                   f"{pl['S_c_fit']['S_c']:.3f} {ci(pl['S_c_fit'].get('S_c_ci68'))} |")
out.append("")

# ------------------------------------------------------------------ D3 nonlinear joint at fitted S_c
out.append("## D3. Section 2.4: nonlinear joint profile at the fitted S_c (stopped by the pre-registered cap rule)\n\n"
           "K fixed at the linear joint optimum; A_crit grid as addendum 6 (30 values, split into 6 interleaved parts); D: 22 log-spaced values 1e-3 to 5e-2 m^2/yr; "
           "Picard cap 100. Each part stops after an A_crit row when more than 30 % of its evaluations hit the cap. 'Best converged' is the lowest MSE among "
           "evaluations that converged; it is reported for the record and is not an optimum (6 of 30 A_crit values evaluated).\n")
out.append("| tile | S_c | evaluated / grid | at cap | parts stopped | lowest D converging everywhere | best converged: A_crit, D, MSE, D/D0_A, >0.5 S_c | linear joint MSE |")
out.append("|---|---|---|---|---|---|---|---|")
for t in TILES:
    d = latest(f"*_tile_run_{t}_v2_nonlinear_joint_Sc*_merged")
    v = P(d)["v2"]["nonlinear_joint"]
    st = v["setup"]
    df = pd.read_parquet(d / "nonlinear_joint_profile.parquet")
    df["Dr"] = df.D.round(12)
    g = df.groupby("Dr").converged.all()
    dmin = g[g].index.min() if g.any() else float("nan")
    c = df[df.converged]
    b = c.loc[c.mse.idxmin()] if len(c) else None
    lin = st["linear_joint_optimum"]
    d0 = P(d)["model"]["D0"]
    bb = f"{b.A_crit:.0f}, {b.D:.3e}, {b.mse:.2f}, {b.D / d0:.2f}, {b['frac_hillslope_gt_0.5Sc']:.2f}" if b is not None else "-"
    out.append(f"| {t} | {st['S_c']:.3f} | {v['n_evaluations']} / {len(st['A_crit_grid']) * len(st['D_grid'])} | {v['n_hit_cap']} ({v['frac_hit_cap']:.0%}) | "
               f"{len(v.get('stopped_by_cap_rule_parts', {}))} of 6 | {dmin:.3e} | {bb} | {lin['mse']:.2f} |")
out.append("")

# ------------------------------------------------------------------ D4 slope threshold per alpha
sm = latest("*_slope_threshold_summary_10tiles")
ps = P(sm)
out.append(f"## D4. Section 3.2: the linear joint profile over C at each alpha\n\nSummary run `{sm.name}`. C in m^2 (at alpha = 2, A S^2 in m^2). Hold-out = mean of the two "
           "half-on-half scores at each half's own best C.\n")
out.append("| tile | alpha | C | K (1/yr) | D (m^2/yr) | D/D0_A | MSE | channel cells | 1 % interval of C | held-out MSE | C fit on half 1 / half 2 | run |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TILES:
    for a, r in ps["tiles"][t]["table"].items():
        iv = "; ".join(f"{x:.4g}-{y:.4g}" for x, y in r["interval_1pct"][:3]) + (" ..." if len(r["interval_1pct"]) > 3 else "")
        out.append(f"| {t} | {float(a):g} | {r['C']:.4g} | {r['K']:.3e} | {r['D']:.3e} | {r['D_over_D0_A']:.2f} | {r['mse']:.3f} | {r['n_channel']} | {iv} | "
                   f"{r['mse_heldout_mean']:.3f} | {r['C_h1']:.4g} / {r['C_h2']:.4g} | `{r['run']}` |")
out.append("")

# ------------------------------------------------------------------ D5 rule and split
out.append("## D5. Section 3.3: the hold-out rule, the split, and the alpha = 0 check against Phase 6\n")
out.append("| tile | held-out alpha 0 | best alpha > 0 | held-out there | improvement | preferred (> 2 %) | best in-sample alpha | in-sample improvement | basins | cells half 1 / half 2 / none | alpha 0 vs Phase 6: common C, max rel diff MSE |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for t in TILES:
    v = ps["tiles"][t]
    a0 = latest(f"*_slope_threshold_{t}_a0")
    p0 = P(a0)
    h = p0["holdout_split"]
    c = p0["result"]["phase6_check"]
    ba = v["best_alpha_heldout"]
    out.append(f"| {t} | {v['heldout_alpha0']:.3f} | {ba:g} | {v['table'][str(ba)]['mse_heldout_mean']:.3f} | {v['heldout_improvement_best']:+.1%} | {v['preferred']} | "
               f"{v['best_alpha_fit']:g} | {v['fit_improvement_best']:+.1%} | {h['n_basins']} | {h['n_cells_half1']} / {h['n_cells_half2']} / {h['n_cells_no_basin']} | "
               f"{c.get('n_common')}, {c.get('max_rel_diff_mse', float('nan')):.1e} |")
out.append("\n| site | tiles preferring alpha > 0 | preferred at the site |\n|---|---|---|")
for site, s in ps["sites"].items():
    out.append(f"| {site} | {s['n_preferred']} of {s['n_tiles']} | **{s['preferred']}** |")
out.append("")

# ------------------------------------------------------------------ D6 Phase 7 port check
pc = latest("*_phase7_port_check_all")
ck = P(pc)["checks"]
out.append(f"## D6. Phase 7: ported analyses against the feature-branch outputs\n\nRun `{pc.name}`. GM grids are identical in both (metres); AP were in US survey feet on the branches and are in metres here.\n")
out.append("| analysis | site | points new / branch | matched | quantity | identical (of matched) | median new | median branch |\n|---|---|---|---|---|---|---|---|")
for step, cols in (("width", ["W_primary", "W_inner_bankbreak", "W_stage_0.5", "local_slope", "inner_relief_m"]), ("width_constrained", ["W_trench", "W_bed", "trench_depth_m", "local_slope"])):
    for site in ("GM", "AP"):
        s = ck[step][site]
        m = s["n_matched_on_tile_row_col"]
        for c in cols:
            q = s[c]
            out.append(f"| {step} | {site} | {s['n_new']} / {s['n_old']} | {m} | {c} | {q['n_identical_or_both_nan']} | {q['median_new']:.4g} | {q['median_old']:.4g} |")
out.append("")
fo, fn = ck["width"]["fits_old"], ck["width"]["fits_new"]
out.append("| width fit | site | variant | branch c, b, n | new c, b, n |\n|---|---|---|---|---|")
for site in ("GM", "AP"):
    for var in ("all", "no_steep"):
        o, n = fo[site][var], fn[site][var]
        out.append(f"| Phase 3 Tobit | {site} | {var} | {o['tobit']['c']:.3g}, {o['tobit']['b']:.3f}, {o['n_channelized']} | {n['tobit']['c']:.3g}, {n['tobit']['b']:.3f}, {n['n_channelized']} |")
fo, fn = ck["width_constrained"]["fits_old"], ck["width_constrained"]["fits_new"]
for site in ("GM", "AP"):
    o, n = fo[site], fn[site]
    out.append(f"| constrained Tobit | {site} | default window | {o['default']['tobit']['c']:.3g}, {o['default']['tobit']['b']:.3f} {ci(o['default']['b_ci'])}, {o['resolved']['n']} resolved | "
               f"{n['default']['tobit']['c']:.3g}, {n['default']['tobit']['b']:.3f} {ci(n['default']['b_ci'])}, {n['resolved']['n']} resolved |")
for site in ("GM", "AP"):
    s = ck["width_constrained"][site]
    out.append(f"\nConstrained classes {site}: branch {s['class_counts_old']}; new {s['class_counts_new']}.")
out.append("\nHead extraction (Method P, kt = 0.1), retained heads branch / new, heads on the same pixel, hollow-class heads branch / new, hollow median A (m^2) branch / new:\n")
out.append("| tile | OWF | OWF + DT0.5 | OWF + DT2 | OWF + DT5 |\n|---|---|---|---|---|")
for t, v in ck["heads"]["per_tile"].items():
    cells = []
    for smo in ("OWF", "OWF+DT0.5", "OWF+DT2", "OWF+DT5"):
        x = v[smo]
        cells.append(f"{x['n_retained_old']}/{x['n_retained_new']}, {x['n_same_pixel']}, {x['n_hollow_old']}/{x['n_hollow_new']}, "
                     f"{x['hollow_A_median_old'] and round(x['hollow_A_median_old'])}/{x['hollow_A_median_new'] and round(x['hollow_A_median_new'])}")
    out.append(f"| {t} | " + " | ".join(cells) + " |")
out.append("")

# ------------------------------------------------------------------ D7 Phase 7 results on metres
wr = latest("*_width_AP_GM_full")
wc = latest("*_width_constrained_full")
hd = latest("*_heads_10tiles")
out.append(f"## D7. Phase 7 results on the metre grids\n\nWidth `{wr.name}`, constrained `{wc.name}`, heads `{hd.name}` (outputs and figures under each run's `outputs/` or `head_extraction/`).\n")
ex = pd.read_csv(wc / "outputs" / "exceedance.csv")
ex = ex[ex.bin_index == -1]
out.append("| site | measure | > 1 m | > 2 m | > 5 m | > 10 m |\n|---|---|---|---|---|---|")
for (site, meas), s in ex.groupby(["site", "measure"]):
    out.append(f"| {site} (n={int(s.n.iloc[0])}) | {meas} | " + " | ".join(f"{r.fraction:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]" for r in s.sort_values("threshold_m").itertuples()) + " |")
cs = pd.read_csv(hd / "head_extraction" / "class_summary.csv")
cs = cs[cs.method == "P"]
out.append("\n| tile | smoothing | retained | hollow | valley_floor | bench_plateau | rill | road_ditch | unclassified | hollow A median (m^2) | hollow S_up_10 median |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in cs.sort_values(["site", "tile", "smoothing"]).itertuples():
    out.append(f"| {r.site}{r.tile} | {r.smoothing} | {r.n_retained} | {r.n_hollow} | {r.n_valley_floor} | {r.n_bench_plateau} | {r.n_rill} | {r.n_road_ditch} | {r.n_unclassified} | "
               f"{r.hollow_A_median_m2:.4g} | {r.hollow_S_up_10_median:.3f} |")
out.append("")

# ------------------------------------------------------------------ D8 accounting
out.append("## D8. Slurm accounting of the addendum 8 jobs\n")
txt = subprocess.run(["sacct", "-S", "2026-11-12T16:20", "-E", "2026-11-13T06:00", "-P", "-X", "-n", "-o", "JobID,JobName,State,Elapsed,Timelimit,AllocCPUS"],
                     capture_output=True, text=True, timeout=60).stdout


def hours(s: str) -> float:
    d = 0
    if "-" in s:
        dd, s = s.split("-")
        d = int(dd)
    p = [int(x) for x in s.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return d * 24 + p[0] + p[1] / 60 + p[2] / 3600


agg: dict[str, dict] = {}
for line in txt.splitlines():
    jid, name, state, el, tl, cpus = line.split("|")
    if not (name.startswith("a8_") or name.startswith("lemcvt_a8")):
        continue
    a = agg.setdefault(name, {"n": 0, "states": {}, "used": 0.0, "limit": 0.0})
    a["n"] += 1
    a["states"][state.split()[0]] = a["states"].get(state.split()[0], 0) + 1
    a["used"] += hours(el) * int(cpus)
    a["limit"] += hours(tl) * int(cpus)
out.append("| job name | tasks | states | limit core-h | used core-h |\n|---|---|---|---|---|")
for k, a in sorted(agg.items()):
    out.append(f"| {k} | {a['n']} | {a['states']} | {a['limit']:.1f} | {a['used']:.1f} |")
out.append(f"| total | | | {sum(a['limit'] for a in agg.values()):.1f} | {sum(a['used'] for a in agg.values()):.1f} |")
print("\n".join(out))
