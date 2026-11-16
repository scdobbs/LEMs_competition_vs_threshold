"""Tables for reports/v2/02e_audits.md (addendum 3, sections 3 and 4).

Usage: PYTHONNOUSERSITE=1 LEMCVT_RESULTS=... python reports/v2/scripts/make_02e_tables.py > reports/v2/02e_tables.md
Reads the `picard_audit_*` runs (section 3) and the latest `measure_audit_10tiles` run (section 4).
Every number comes from a run's params.json; run ids are printed with each table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

root = Path(os.environ.get("LEMCVT_RESULTS", "/home/groups/hilley/lemcvt_results"))
TILES = ["AP1", "AP2", "AP3", "AP4", "AP5", "GM1", "GM2", "GM3", "GM4", "GM5"]
STEPS = [100.0, 199.526, 398.107, 794.328, 1000.0, 1995.26, 3981.07, 7943.28, 10000.0, 19952.6, 39810.7, 100000.0]
out: list[str] = []


def load(d: Path):
    return json.loads((d / "params.json").read_text()), json.loads((d / "run.json").read_text())


def fmt(v, f=".3g"):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, f)


# ============================================================================ section 3: Picard
runs = {}
for d in sorted(root.glob("*_picard_audit_*")):
    if (d / "params.json").exists():
        p, r = load(d)
        runs[(p["tile"], p["variant"]["label"])] = (d.name, p, r)

out.append("## P1. Picard iteration at the v1 printed nonlinear optimum, AP in feet as v1 ran (`lemcvt audit picard`)\n")
out.append("| tile | run | z unit | iterations | ended by | first update max\\|dz\\| | final update max\\|dz\\| (tol 1e-6) | final nonlinear residual (rel. L2) | hillslope cells | clipped at 0.99 S_c | MSE | max\\|dz\\| vs step-5 gate grid / relief |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TILES:
    k = (t, "feet_as_v1")
    if k not in runs:
        out.append(f"| {t} | (not finished) | | | | | | | | | | |")
        continue
    name, p, r = runs[k]
    res = p["result"]
    pic = res["picard"]
    it = pic["iterations"]
    ref = res.get("reference_comparison") or {}
    out.append(f"| {t} | {name} | {p['variant']['z_units_working']} | {pic['n_iter']} | {'tolerance' if pic['converged'] else '**cap (50)**'} | "
               f"{it[0]['update_max_abs']:.2e} | {it[-1]['update_max_abs']:.2e} | {it[-1].get('residual_rel_l2', float('nan')):.1e} | "
               f"{res['n_hillslope']} | {it[-1]['frac_ratio_clipped']:.3f} | {res['mse']:.4f} | {fmt(ref.get('max_abs_dz_over_relief'), '.1e')} |")
out.append("")
out.append("## P2. Hillslope slope statistics (the model's own slope stencil on the hillslope cells): fraction with \\|grad z\\| > 0.5 S_c, > 0.9 S_c, and max \\|grad z\\|/S_c, for the nonlinear solution, the linear solution the iteration starts from, and the observed grid on the same cells\n")
out.append("| tile | variant | nonlinear final: > 0.5 / > 0.9 / max | linear start: > 0.5 / > 0.9 / max | observed DEM: > 0.5 / > 0.9 / max | hillslope cells |")
out.append("|---|---|---|---|---|---|")
for (t, lab), (name, p, r) in sorted(runs.items(), key=lambda kv: (TILES.index(kv[0][0]), kv[0][1] != "feet_as_v1", kv[0][1])):
    res = p["result"]
    nl, li, de = res["slopes_nonlinear_final"], res["slopes_linear_start"], res["slopes_dem_same_cells"]
    out.append(f"| {t} | {lab} | {nl['frac_gt_0.5Sc']:.3f} / {nl['frac_gt_0.9Sc']:.3f} / {nl['max_ratio']:.1f} | "
               f"{li['frac_gt_0.5Sc']:.3f} / {li['frac_gt_0.9Sc']:.3f} / {li['max_ratio']:.1f} | "
               f"{de['frac_gt_0.5Sc']:.3f} / {de['frac_gt_0.9Sc']:.4f} / {de['max_ratio']:.2f} | {res['n_hillslope']} |")
out.append("")
out.append("## P3. AP1 and AP5 in metres (`--metres`, config unchanged): the three scalings of K0 and D0\n")
out.append("| tile | variant | K factor | D factor | iterations | ended by | final update | final residual | clipped | nonlinear > 0.5 / > 0.9 S_c | MSE (m^2) | MSE of the linear start (m^2) |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in ("AP1", "AP5"):
    for lab in ("feet_as_v1", "metres_KD_x3.2808", "metres_D_x3.2808", "metres_D_div3.2808"):
        k = (t, lab)
        if k not in runs:
            continue
        name, p, r = runs[k]
        res = p["result"]
        pic = res["picard"]
        it = pic["iterations"]
        nl = res["slopes_nonlinear_final"]
        v = p["variant"]
        out.append(f"| {t} | {lab} | {v['scale_K']:.4f} | {v['scale_D']:.4f} | {pic['n_iter']} | {'tolerance' if pic['converged'] else '**cap**'} | "
                   f"{it[-1]['update_max_abs']:.2e} | {it[-1].get('residual_rel_l2', float('nan')):.1e} | {it[-1]['frac_ratio_clipped']:.3f} | "
                   f"{nl['frac_gt_0.5Sc']:.3f} / {nl['frac_gt_0.9Sc']:.3f} | {res['mse']:.2f}{' (ft^2)' if lab == 'feet_as_v1' else ''} | {res['mse_linear']:.2f} |")
out.append("")
out.append("## P4. Convergence histories (max\\|dz\\| per iteration, z unit of the run) for the five AP feet runs and AP1 / AP5 in metres (K, D x 3.2808)\n")
out.append("| run | iteration 1 | 2 | 3 | 5 | 10 | 20 | 30 | 40 | 50 |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for t in TILES[:5]:
    for lab in ("feet_as_v1", "metres_KD_x3.2808"):
        k = (t, lab)
        if k not in runs:
            continue
        it = runs[k][1]["result"]["picard"]["iterations"]
        cells = []
        for i in (1, 2, 3, 5, 10, 20, 30, 40, 50):
            cells.append(f"{it[i - 1]['update_max_abs']:.1e}" if i <= len(it) else "-")
        out.append(f"| {t} {lab} | " + " | ".join(cells) + " |")
out.append("")

# ============================================================================ section 4: measure audit
aruns = sorted(d for d in root.glob("*_measure_audit_10tiles*") if (d / "params.json").exists())
if aruns:
    run = aruns[-1]
    p, meta = load(run)
    tiles = list(p["tiles"])
    out.append(f"# Section 4 tables (run `{run.name}`, git {meta['git_hash'][:8]}, Slurm job {meta.get('slurm_job_id')})\n")

    def step(lad, a):
        d = {round(s["A_c"]): s for s in lad}
        return d.get(round(a))

    def th(e, key="theta_star"):
        if not e or e.get(key) is None:
            return "-"
        s = f"{e[key]:.2f}"
        ci = e.get("theta_ci68")
        if ci:
            s += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
        return s

    # A1 n_eff
    out.append("## A1. n_eff of the pooled through-origin estimator along the free ladder (centred chi-spread weights; through-origin leverage in parentheses); basins used\n")
    out.append("| tile | " + " | ".join(f"{a:.3g}" for a in STEPS) + " |")
    out.append("|---|" + "---|" * len(STEPS))
    for t in tiles + [f"{s}_pooled" for s in p["pooled"]]:
        lad = p["tiles"][t]["free_ladder"] if t in p["tiles"] else p["pooled"][t.split("_")[0]]["ladder"]
        cells = []
        for a in STEPS:
            s = step(lad, a)
            e = (s or {}).get("pooled")
            if not e:
                cells.append("-")
            else:
                flag = "**" if e.get("n_eff_lt_3") else ""
                cells.append(f"{flag}{e['n_eff_centred']:.1f}{flag} ({e['n_eff_through_origin']:.1f}) / {s['n_basins']}")
        out.append(f"| {'**' + t + '**' if 'pooled' in t else t} | " + " | ".join(cells) + " |")
    out.append("\n(bold: n_eff < 3, the bootstrap interval is not a usable uncertainty)\n")

    # A2 three estimators, free ladder
    out.append("## A2. Free ladder: theta* by the pooled (02c), equal-weight and median estimators [68 % basin bootstrap]\n")
    for est, title in (("pooled", "pooled through-origin SSR (02c estimator)"), ("equal", "equal-weight: sum_i SSR_i / SSR_i,null"), ("median", "median over basins of the per-basin best theta")):
        out.append(f"\n### A2.{est}: {title}\n")
        out.append("| tile | " + " | ".join(f"{a:.3g}" for a in STEPS) + " |")
        out.append("|---|" + "---|" * len(STEPS))
        for t in tiles + [f"{s}_pooled" for s in p["pooled"]]:
            lad = p["tiles"][t]["free_ladder"] if t in p["tiles"] else p["pooled"][t.split("_")[0]]["ladder"]
            cells = []
            for a in STEPS:
                s = step(lad, a)
                e = (s or {}).get(est)
                cells.append(th(e) + (f" ({s['n_basins']})" if e and s.get("flag_few_basins") else "") if e else "-")
            out.append(f"| {'**' + t + '**' if 'pooled' in t else t} | " + " | ".join(cells) + " |")
    out.append("")

    # A3 basin-fixed against free, all three estimators
    out.append("## A3. Basin-fixed ladder against the free ladder: max |theta_fixed - theta_free| over the 1e2-1e4 steps, per estimator; basins kept / at 1e4\n")
    out.append("| tile | kept / at 1e4 | pooled | equal | median | fixed-set theta at 1e2 -> 1e4 (pooled / equal / median) |")
    out.append("|---|---|---|---|---|---|")
    for t in tiles:
        r = p["tiles"][t]
        fx = {round(s["A_c"]): s for s in r["fixed_ladder"]}
        fr = {round(s["A_c"]): s for s in r["free_ladder"]}
        cells = []
        for est in ("pooled", "equal", "median"):
            diffs = [abs(fr[a][est]["theta_star"] - fx[a][est]["theta_star"]) for a in fx
                     if fr.get(a, {}).get(est) and fx[a].get(est) and fr[a][est].get("theta_star") is not None and fx[a][est].get("theta_star") is not None]
            cells.append(f"{max(diffs):.2f}" if diffs else "-")
        lo, hi = fx.get(100), fx.get(10000)
        trend = " / ".join(f"{(lo or {}).get(est, {}).get('theta_star', float('nan')):.2f} -> {(hi or {}).get(est, {}).get('theta_star', float('nan')):.2f}"
                           for est in ("pooled", "equal", "median")) if lo and hi and lo.get("pooled") and hi.get("pooled") else "-"
        out.append(f"| {t} | {r['fixed_basins']['n_kept']} / {r['fixed_basins']['n_basins_at_ref']} | " + " | ".join(cells) + f" | {trend} |")
    out.append("")

    # A4 truncated vs complete at 1e3
    out.append("## A4. Truncated against complete basins at A_c = 1e3 m^2 (free ladder's basins): theta* by estimator [68 %], basins\n")
    out.append("| tile | truncated: pooled / equal / median (n) | complete: pooled / equal / median (n) |")
    out.append("|---|---|---|")
    for t in tiles:
        tc = p["tiles"][t]["truncated_vs_complete_1e3"]
        cells = []
        for key in ("truncated", "complete"):
            e = tc.get(key) or {}
            if not e.get("pooled"):
                cells.append(f"- (n = {e.get('n_basins', 0)})")
            else:
                cells.append(f"{th(e['pooled'])} / {th(e['equal'])} / {th(e['median'])} (n = {e['n_basins']})")
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    out.append("")

    # A5 T8 with both estimators and n_eff
    out.append("## A5. T8 revisited: theta / k_s / K0 = U/k_s per tile at the three threshold options, pooled and equal-weight estimators, with n_eff and basins\n")
    out.append("| tile | option | A_c | pooled: theta / k_s / K0 | n_eff (basins) | equal: theta / k_s(pooled at theta) / K0 | k_s basin median | bootstrap usable |")
    out.append("|---|---|---|---|---|---|---|---|")
    for t in tiles:
        U = p["tiles"][t]["U"]
        site = t[:2]
        for name, a in (("(a) 1e3", 1000.0), ("(b) 1e4", 10000.0), ("(c) site break", 1995.26 if site == "AP" else 398.107)):
            s = step(p["tiles"][t]["free_ladder"], a)
            if not s or not s.get("pooled"):
                out.append(f"| {t} | {name} | {a:.3g} | - | - | - | - | - |")
                continue
            po, eq = s["pooled"], s["equal"]
            out.append(f"| {t} | {name} | {a:.3g} | {po['theta_star']:.2f} / {po['ks']:.3g} / {U / po['ks']:.3g} | {po['n_eff_centred']:.1f} ({s['n_basins']}) | "
                       f"{eq['theta_star']:.2f} / {eq['ks_pooled_at_theta']:.3g} / {U / eq['ks_pooled_at_theta']:.3g} | {fmt(eq.get('ks_basin_median'))} | "
                       f"{'no (n_eff < 3)' if po.get('n_eff_lt_3') else 'yes'} |")
    out.append("")

    # A6 hilltops
    out.append("## A6. Hilltop curvature with and without the near-channel exclusion (drop pixels closer than L*/2 to a channel pixel), networks at A_c = 1e3 and 1e4 m^2 and the rollover network the measure run used; C_ht at the fixed L* [segment-bootstrap 68 %]; D0 = U/C_ht\n")
    out.append("| tile | L* (m) | network | set | pixels | segments | C_ht (1/m) [68 %] | D0 (m^2/yr) [68 %] | L* re-selected (m) | C_ht at re-selected L* | width 2d: 25 / 50 % (m) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tiles:
        h = p["tiles"][t]["hilltops"]
        for key in ("1000", "10000", "rollover"):
            net = h["networks"].get(key)
            if not net:
                continue
            for s in ("all", "excluded"):
                r = net[s]
                a = r["at_fixed_L"]
                ci = a.get("C_ht_ci68")
                dci = a.get("D0_ci68")
                w = r.get("width_2x_distance_m") or {}
                out.append(f"| {t} | {h['L_star_m']:.0f} | {key if key == 'rollover' else key + ' m^2'}{' (' + format(net['A_c'], '.3g') + ')' if key == 'rollover' else ''} | {s} | "
                           f"{r['n_pixels']} | {r['n_segments']} | {a['C_ht']:.3e}" + (f" [{ci[0]:.2e}, {ci[1]:.2e}]" if ci else "") + f" | {fmt(a.get('D0'))}"
                           + (f" [{dci[0]:.3g}, {dci[1]:.3g}]" if dci else "") + f" | {fmt(r['L_star_reselected_m'], '.0f')} | {fmt(r['at_reselected_L']['C_ht'], '.3e')} | "
                           f"{fmt(w.get('25'), '.0f')} / {fmt(w.get('50'), '.0f')} |")
        mr = h.get("measure_run_C_ht", {})
        out.append(f"| {t} | | measure run (02b): C_ht {fmt(mr.get('C_ht'), '.3e')} at A_c {fmt(mr.get('A_c'))}, {mr.get('n_hilltop')} pixels | | | | | | | | |")
    out.append("")

    # A7 synthetic ridges post exclusion
    out.append("## A7. Synthetic-ridge bias at each tile's L*, post-exclusion width distribution (network at 1e4 m^2)\n")
    out.append("| tile | L* (m) | post-exclusion width 2d: 25 / 50 % (m) | bias at median width | bias at 25 % width | variant | c_true (1/m) |")
    out.append("|---|---|---|---|---|---|---|")
    for site, sy in p.get("synthetic_ridges_post_exclusion", {}).items():
        for t, b in sy.get("bias_post_exclusion", {}).items():
            out.append(f"| {t} | {p['tiles'][t]['hilltops']['L_star_m']:.0f} | {fmt(b['width_q25_m'], '.0f')} / {fmt(b['width_median_m'], '.0f')} | "
                       f"{b['bias_at_median_width'] * 100:+.1f} % | {b['bias_at_q25_width'] * 100:+.0f} % | {b['variant']} | {sy['c_true']:.3g} |")
    out.append("")
else:
    out.append("\n(no measure_audit run found yet)\n")

print("\n".join(out))
