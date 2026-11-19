#!/usr/bin/env python3
"""Addendum 17 Part B: appendix tables of the measurement report, generated from the run directory.
usage: make_a17b_tables.py --run DIR [--jobs id,id,...]"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17b_common import BEDROCK_PASS_FRAC, E_M_PER_YR, GAPFILL_MAX_FRAC, TIER1, TIER2, V2_SITE_M  # noqa: E402


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, bool | np.bool_):
        return "yes" if v else "no"
    if isinstance(v, int | np.integer):
        return f"{v:,}"
    if isinstance(v, float | np.floating):
        if abs(v) >= 1e5:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.3e}"
        return f"{v:.{d}f}"
    return str(v)


def pct(v):
    return "-" if v is None or not np.isfinite(v) else f"{100 * v:.1f} %"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def ci(lo, hi, d=3):
    return "-" if lo is None or hi is None else f"[{fmt(lo, d)}, {fmt(hi, d)}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--jobs", default="")
    a = p.parse_args()
    R = a.run
    out = []
    W = out.append
    # ---------------- B1 exclusions and tiers
    W("### B1. Exclusions and tiers (go-ahead section 1)\n")
    W("Tier 1 (not measured, except AP1_3dep for the survey comparison; registered):\n")
    W(table(["tile", "reason"], [[t, r] for t, r in TIER1.items()]))
    W("Tier 2 (measured, in the primary set; the clean subset leaves them out):\n")
    W(table(["tile", "reason"], [[t, r] for t, r in TIER2.items()]))
    # ---------------- B2 screens
    b = load(os.path.join(R, "screens", "bedrock_screen.json"))
    W("### B2. The GM bedrock screen (go-ahead 3.1), from screens/bedrock_screen.json\n")
    if b:
        W(f"Map: {b['map']['path']} ({b['map']['purl']}; sha256 of GEOLOGY.shp {b['map']['sha256_shp'][:16]}..., GEOLOGY.dbf {b['map']['sha256_dbf'][:16]}...; CRS {b['map']['crs']}; {b['map']['n_features']:,} polygons). "
          f"Reference unit: **{b['reference_unit']['GEN_LAB']} ({b['reference_unit']['GEN_NAME']})**, {100 * b['reference_unit']['combined_fraction_over_controls'] / 7:.1f} % of the seven control footprints combined "
          f"(candidates: {', '.join(f'{k} {100 * v / 7:.1f} %' for k, v in sorted(b['reference_unit']['candidates'].items(), key=lambda kv: -kv[1]))}). "
          f"Counted Quaternary units: {', '.join(b['rule']['quaternary_counted'])}; not counted: landslide deposits ({', '.join(b['rule']['landslide_not_counted'])}), fill, eolian, dune, beach, coastal terrace, Aromas Sand. Pass at >= {100 * BEDROCK_PASS_FRAC:.0f} %.\n")
        W(table(["tile", "kind", "reference unit", "counted Quaternary", "**rule fraction**", "landslide (not counted)", "other bedrock", "other Quaternary", "map coverage", "pass", "Tier 1", "other units in the footprint"],
                [[r["tile"], r["kind"], pct(r["frac_reference"]), pct(r["frac_quaternary_counted"]), pct(r["frac_rule"]), pct(r["frac_landslide"]), pct(r["frac_other_bedrock"]), pct(r["frac_other_quaternary"]), pct(r["coverage"]), r["pass"], r["tier1"], r["other_units"]] for r in b["table"]]))
        W(f"Fail (off-mesa): {b['fail'] or 'none'}; control tiles failing: {b['control_fail'] or 'none'}.\n")
    g = load(os.path.join(R, "screens", "gapfill_screen.json"))
    W("### B3. The gap-fill screen on the measurement pixels (go-ahead 3.2), from screens/gapfill_screen.json\n")
    if g:
        W(f"Filled cells = nnfill_mask == 1 (nearest-neighbour fill within 10 m) or gapfill_mask == 1 (enclosed gap filled for routing). (a) = {g['rule']['a']}; (b) = {g['rule']['b']}; a tile leaves the primary set above {100 * GAPFILL_MAX_FRAC:.0f} % on either.\n")
        W(table(["tile", "site", "Tier 2", "channel px (kept basins)", "basins kept / all", "**(a) filled on channel px**", "hilltop px", "**(b) filled on hilltop px**", "filled, whole tile", "rollover A_c (m2)", "measure run", "pass"],
                [[r["tile"], r["site"], r["tier2"], r["n_channel_px_kept"], f"{r['n_basins_kept']} / {r['n_basins']}", pct(r["frac_fill_channel"]), r["n_hilltop_px"], pct(r["frac_fill_hilltop"]), pct(r["frac_fill_tile"]), r["A_c_rollover"], r["measure_run"], r["pass"]] for r in g["table"]]))
        W(f"Fail (gap-filled): {g['fail'] or 'none'}.\n")
    ps = load(os.path.join(R, "screens", "primary_set.json"))
    W("### B4. The primary set, fixed before the m-profile runs, from screens/primary_set.json\n")
    if ps:
        W(f"Fixed at {ps['fixed_at_utc']} UTC. " + "; ".join(f"**{s}: {n}** (expected {ps['expected_before_screens'][s]})" for s, n in ps["counts"].items()) + ".\n")
        W(table(["site", "primary tiles", "removed by the screens", "clean subset (minus Tier 2)"],
                [[s, ", ".join(ps["primary"][s]), "; ".join(f"{r['tile']}: {r['reason']}" for r in ps["removed_by_screens"][s]) or "none", ", ".join(ps["clean_subset"][s])] for s in ps["primary"]]))
    # ---------------- B5 S(m)
    m = load(os.path.join(R, "measure", "measurement.json"))
    W("### B5. The site exponent profiles S(m) (section 5.2), from measure/measurement.json\n")
    if m:
        rows = []
        for site, S in m["sites"].items():
            for gname, prof in S["S_m"].items():
                if "m_opt" not in prof:
                    rows.append([site, gname, len(prof.get("tiles", [])), "-", "-", "-", "-", prof.get("note", "")])
                    continue
                i1, i5 = prof["interval_0.01"], prof["interval_0.05"]
                rows.append([site, gname, prof["n_tiles"], prof["m_opt"], prof["S_min"] / prof["n_tiles"], f"[{i1['m_min']:.2f}, {i1['m_max']:.2f}]" + (" (edge)" if i1["touches_grid_edge"] else ""),
                             f"[{i5['m_min']:.2f}, {i5['m_max']:.2f}]" + (" (edge)" if i5["touches_grid_edge"] else ""), ""])
        W(table(["site", "group", "tiles", "**m = argmin S**", "S_min / n", "1 % interval", "5 % interval", "note"], rows))
        for site, S in m["sites"].items():
            prof = S["S_m"]["primary"]
            if "leave_one_out" in prof:
                W(f"{site} primary set, leave-one-out minima: " + ", ".join(f"{t} {v:.2f}" for t, v in prof["leave_one_out"].items()) + ".\n")
                W(f"{site} per-tile optima (m, A_crit): " + ", ".join(f"{t} ({prof['tile_optima_m'][t]:.2f}, {prof['tile_optima_A_crit'][t]:,.0f})" for t in prof["tile_optima_m"]) + ".\n")
        W("Rule 6.2 (site m stability): " + "; ".join(f"{s}: v2 {r['v2_m']:.2f}, primary {r['primary_m']:.2f} (moved > 0.05: {fmt(r['primary_moved_gt_0.05'])})" + (f", NCALM GM1-GM7 {fmt(r.get('ncalm_m'))} (moved: {fmt(r.get('ncalm_moved_gt_0.05'))})" if s == "GM" else f", PAMAP AP1-AP5 recomputed {fmt(r.get('pamap_m_recomputed'))}") for s, r in m["rule_62"].items()) + ".\n")
    # ---------------- B6 per tile
    pt = os.path.join(R, "measure", "per_tile.csv")
    W("### B6. Per-tile measurements at the site m (sections 5.1, 5.3, 5.4), from measure/per_tile.csv\n")
    if os.path.exists(pt):
        df = pd.read_csv(pt)
        rows = []
        for _, r in df.iterrows():
            rows.append([r.tile, r.site, r.group + (" (Tier 2)" if r.tier2 else ""), r.m_site, r.theta_pooled, ci(r.theta_pooled_lo95, r.theta_pooled_hi95, 2), r.theta_equal, r.theta_1e4_pooled, r.n_basins_1e3, r.theta_rollover, r.A_c_rollover,
                         r.ks_1e3, ci(r.ks_1e3_lo95, r.ks_1e3_hi95), r.n_eff_1e3, r.K0, r.ks_1e4, r.C_ht_Lf, ci(r.C_ht_Lf_lo95, r.C_ht_Lf_hi95, 4), r.D0_A, r.L_w_m, r.C_ht_Lw, r.D0_B, r.n_hilltop_px, r.n_segments,
                         r.median_crest_width_m, r.windows_tested, r.max_abs_dev, r.window_pass])
        W(table(["tile", "site", "group", "m_site", "theta (1e3, pooled)", "95 %", "theta (1e3, equal)", "theta (1e4, pooled)", "basins (1e3)", "theta at rollover", "A_c rollover",
                 "k_s (1e3)", "95 %", "n_eff", "K0 = E/k_s", "k_s (1e4)", "C_ht(L_f)", "95 %", "D0_A", "L_w (m)", "C_ht(L_w)", "D0_B", "hilltop px", "segments", "median crest width (m)", "windows tested", "max |dev|", "window test"], rows))
    # ---------------- B7 site summaries and rule 6.1
    W("### B7. Site summaries per group: pooled theta, site-pooled k_s and K0, D0, the window-stability test (rule 6.1)\n")
    if m:
        rows = []
        for site, S in m["sites"].items():
            for gname, s in S.items():
                if gname == "S_m":
                    continue
                th3 = (s.get("theta_site_1000") or {})
                th4 = (s.get("theta_site_10000") or {})
                k3 = s.get("ks_site_pooled_1000") or {}
                wt = s["window_test"]
                rows.append([site, gname, s["n"], s["m_site"], (th3.get("pooled") or {}).get("theta_star"), ci(*((th3.get("pooled") or {}).get("theta_ci95") or [None, None]), 2),
                             (th3.get("equal") or {}).get("theta_star"), ci(*((th3.get("equal") or {}).get("theta_ci95") or [None, None]), 2), th3.get("n_basins"),
                             (th4.get("pooled") or {}).get("theta_star"), ci(*((th4.get("pooled") or {}).get("theta_ci95") or [None, None]), 2), (th4.get("equal") or {}).get("theta_star"), th4.get("n_basins"),
                             k3.get("ks"), ci(*(k3.get("ks_ci95") or [None, None])), k3.get("n_eff_centred"), s.get("K0_site_pooled_1000"), s["K0_median"], f"{fmt(s['K0_min'])}-{fmt(s['K0_max'])}",
                             s["D0_A_median"], f"{fmt(s['D0_A_min'])}-{fmt(s['D0_A_max'])}", s["D0_B_median"], f"{wt['n_pass']} / {wt['n_tiles']}", wt["stands"], s["D0_status"], ", ".join(wt["trivial_only_Lf"]) or "-"])
        W(table(["site", "group", "n", "m_site", "theta site-pooled (1e3)", "95 %", "theta equal-weight (1e3)", "95 %", "basins", "theta pooled (1e4)", "95 %", "theta equal (1e4)", "basins (1e4)",
                 "k_s site-pooled (1e3)", "95 %", "n_eff", "K0 site-pooled", "K0 median of tiles", "K0 range", "D0_A median", "D0_A range", "D0_B median", "window test pass", "stands (>= 80 %)", "D0 status", "tiles with only L_f in range"], rows))
    # ---------------- B8 survey comparisons
    W("### B8. Survey comparisons: AP PAMAP against 3DEP (rule 6.3) and GM NCALM against FEMA (rule 6.4, report only)\n")
    if m:
        r63 = m["rule_63"]
        W(table(["tile", "k_s PAMAP", "k_s 3DEP", "ratio 3DEP/PAMAP", "C_ht PAMAP", "C_ht 3DEP", "ratio", "theta PAMAP", "theta 3DEP", "ratio", "3DEP tile excluded (Tier 1)"],
                [[r["tile"], r["ks_pamap"], r["ks_3dep"], r["ks_ratio_3dep_over_pamap"], r["C_ht_pamap"], r["C_ht_3dep"], r["C_ht_ratio"], r["theta_pamap"], r["theta_3dep"], r["theta_ratio"], r["excluded"]] for r in r63["rows"]]))
        W("AP medians: k_s ratio " + fmt(r63["ks"]["median_ratio"]) + ", C_ht ratio " + fmt(r63["C_ht"]["median_ratio"]) + ", theta ratio " + fmt(r63["theta"]["median_ratio"]) + ".\n")
        r64 = m["rule_64"]
        W(table(["tile", "k_s NCALM", "k_s FEMA", "ratio FEMA/NCALM", "C_ht NCALM", "C_ht FEMA", "ratio", "theta NCALM", "theta FEMA"],
                [[r["tile"], r["ks_ncalm"], r["ks_fema"], r["ks_ratio_fema_over_ncalm"], r["C_ht_ncalm"], r["C_ht_fema"], r["C_ht_ratio"], r["theta_ncalm"], r["theta_fema"]] for r in r64["rows"]]))
        W(f"GM: median k_s ratio {fmt(r64['ks']['median_ratio'])} (within +-15 %: {fmt(r64['ks']['median_within_15pct'])}; largest single deviation {pct(r64['ks']['max_single_abs_dev'])}, all within +-30 %: {fmt(r64['ks']['all_within_30pct'])}); "
          f"median |C_ht| ratio {fmt(r64['C_ht']['median_ratio'])} (within +-15 %: {fmt(r64['C_ht']['median_within_15pct'])}; largest {pct(r64['C_ht']['max_single_abs_dev'])}, all within +-30 %: {fmt(r64['C_ht']['all_within_30pct'])}); "
          f"**criteria of rule 6.4 met: {fmt(r64['criteria_met'])}** ({r64['note']}).\n")
    # ---------------- B9 runs and checksums
    W("### B9. Run ids and input checksums\n")
    if m:
        W(table(["tile", "m-profile run", "measure run", "sha256 filled (16)", "sha256 area (16)", "sha256 fd (16)"],
                [[t, m["m_profile_runs"].get(t, "-"), v["measure_run"]["run"], v["inputs"]["sha256"]["filled"][:16] + "...", v["inputs"]["sha256"]["area"][:16] + "...", v["inputs"]["sha256"]["fd"][:16] + "..."] for t, v in m["tiles"].items()]))
        W(f"E (m/yr): {E_M_PER_YR}; v2 site m: {V2_SITE_M}; settings: {json.dumps({k: v for k, v in m['settings'].items() if k != 'thetas'})}; theta grid 0.00-0.90 step 0.01.\n")
    # ---------------- B10 slurm
    W("### B10. Slurm accounting of the Part B jobs\n")
    if a.jobs:
        try:
            txt = subprocess.run(["sacct", "-j", a.jobs, "-X", "-o", "JobID,JobName%22,Partition,State,Elapsed,Timelimit,AllocCPUS,MaxRSS,TotalCPU", "-P"], capture_output=True, text=True, check=True).stdout
            lines = [line.split("|") for line in txt.strip().splitlines()]
            body = lines[1:]
            # collapse the arrays: one row per job id with the count of tasks and the elapsed sum
            W(table(lines[0], body))
            txt2 = subprocess.run(["sacct", "-j", a.jobs, "-o", "JobID,MaxRSS,Elapsed,AllocCPUS", "-P"], capture_output=True, text=True, check=True).stdout
            peak = {}
            for line in txt2.strip().splitlines()[1:]:
                jid, rss, el, cpus = line.split("|")
                base = jid.split(".")[0].split("_")[0]
                for suf, f in (("K", 1e6), ("M", 1e3), ("G", 1.0)):
                    if rss.endswith(suf):
                        peak[base] = max(peak.get(base, 0), float(rss[:-1]) / f)
            W("Peak memory per job or array (GB): " + "; ".join(f"{k}: {v:.1f}" for k, v in peak.items()) + "\n")

            def secs(s):
                d, _, hms = s.rpartition("-")
                h, mi, se = (int(x) for x in hms.split(":"))
                return (int(d) * 86400 if d else 0) + h * 3600 + mi * 60 + se
            used = sum(secs(r[4]) * int(r[6]) for r in body if r[6].isdigit()) / 3600
            lim = sum(secs(r[5]) * int(r[6]) for r in body if r[6].isdigit()) / 3600
            W(f"Core-hours used (elapsed x CPUs over all tasks): {used:.1f}; sum of CPU-weighted limits: {lim:.1f}; tasks: {len(body)}.\n")
        except Exception as e:  # pragma: no cover
            W(f"sacct failed: {e!r}\n")
    txt = "\n".join(out)
    with open(os.path.join(R, "appendix_tables.md"), "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
