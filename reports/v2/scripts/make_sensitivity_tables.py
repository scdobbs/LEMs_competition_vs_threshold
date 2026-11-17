"""Appendix tables of the addendum 12 report (reports/v2/<ts>PST_sensitivity.md, section 7), from the latest
``a12_summary``, ``a12_inputs`` and ``a12_estimate`` runs. Prints markdown to stdout.

Usage: python reports/v2/scripts/make_sensitivity_tables.py [--summary RUN] > /path/to/appendix.md
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from lemcvt.config import results_root
from lemcvt.results import find_runs


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e4 or a < 1e-3:
            return f"{v:.{digits - 1}e}"
        return f"{v:.{digits}g}"
    return str(v)


def md(df: pd.DataFrame, cols: list[str] | None = None, digits: int = 4) -> str:
    cols = cols or list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], digits) if c in r else "-" for c in cols) + " |")
    return "\n".join(out)


def _load(step: str, name: str | None = None):
    runs = find_runs(step)
    if name:
        runs = [r for r in runs if r.name == name]
    if not runs:
        raise SystemExit(f"no {step} run")
    r = runs[-1]
    return r, json.loads((r / "params.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()
    root = results_root()
    srun, s = _load("a12_summary", args.summary)
    irun, inp = _load("a12_inputs", s.get("inputs_run"))
    erun, est = _load("a12_estimate")
    ins = pd.read_parquet(srun / "insample.parquet")
    held = pd.read_parquet(srun / "heldout.parquet")
    split = pd.read_parquet(srun / "split.parquet")
    side = pd.read_parquet(srun / "side_by_side.parquet")
    p = print

    p("## B1. Section 3: the core-hour estimate\n")
    p(f"Run `{erun.name}`; expected core-hours from the Phase 6 seconds per evaluation of the same computations.\n")
    e = pd.read_parquet(erun / "estimate.parquet")
    p(md(e, ["tile", "n_unique", "s_per_part_eval", "s_per_ad_eval", "n_joint_5pct_at_m_t", "equal_part_h", "equal_ad_h", "mt_part_h", "mt_ad_h"], 3))
    p("\nTotals (expected core-hours): " + ", ".join(f"{k} {v:.1f}" for k, v in est["totals_expected_core_hours"].items()) + "\n")

    p("## B2. Section 2.2: the greedy basin split (edge-outlet basins of the 10³ m² network)\n")
    p(md(split, ["tile", "n_basins_edge", "n_interior_outlets", "n_basins_half1", "n_basins_half2", "n_cells_half1", "n_cells_half2", "share_half1",
                 "share_half2", "frac_valid_in_split", "largest_basin_share"], 4))

    p("\n## B3. Section 2.1: in-sample MSE (m²) of every variant at the site m\n")
    p("Partitioned variants: minimum over the union of the Phase 6 refined A_crit rows (stored Gram terms) and this run's rows; "
      "`P*_mse_new_rows` = the minimum over this run's rows only (thinned grid + staircase + 1 % fill).\n")
    cols = ["tile", "P1_A_mse", "P1_B_mse", "P2_A_mse", "P2_B_mse", "P3_mse", "P4_mse", "A1_mse", "A2_A_mse", "A2_B_mse", "AD_mse"]
    p(md(ins, [c for c in cols if c in ins.columns], 5))
    p("\n### B3b. Parameters of each variant's optimum\n")
    rows = []
    for _, r in ins.iterrows():
        for n in ("P1_A", "P1_B", "P2_A", "P2_B", "P3", "P4"):
            if f"{n}_mse" in r and pd.notna(r.get(f"{n}_mse")):
                rows.append({"tile": r["tile"], "variant": n, "A_crit": r[f"{n}_A_crit"], "K": r[f"{n}_K"], "D": r[f"{n}_D"], "mse": r[f"{n}_mse"],
                             "mse_new_rows": r[f"{n}_mse_new_rows"]})
        for n in ("A1", "A2_A", "A2_B", "AD"):
            if f"{n}_mse" in r and pd.notna(r.get(f"{n}_mse")):
                rows.append({"tile": r["tile"], "variant": n, "A_crit": None, "K": r[f"{n}_K"], "D": r[f"{n}_D"], "mse": r[f"{n}_mse"],
                             "K_at_bound": r[f"{n}_K_at_bound"], "D_at_bound": r[f"{n}_D_at_bound"]})
    p(md(pd.DataFrame(rows), ["tile", "variant", "A_crit", "K", "D", "mse", "mse_new_rows", "K_at_bound", "D_at_bound"], 4))

    p("\n## B4. Section 2.3: equal-count comparisons per tile (True = the partitioned variant is lower than every AD variant of its count)\n")
    wcols = [c for c in ("count1_A_P1_wins", "count1_B_P1_wins", "count2_A_P2_wins", "count2_B_P2_wins", "count2_P3_wins", "P4_beats_AD") if c in ins.columns]
    p(md(ins, ["tile"] + wcols))
    p("\nPer site (rule: >= 4 of 5 tiles):\n")
    rr = []
    for site, r in s["site_rules"].items():
        for k, v in r.items():
            if k == "m_rule":
                continue
            rr.append({"site": site, "comparison": k, "tiles": f"{v['n_tiles']} of {v['of']}", "wins": v["wins"]})
    p(md(pd.DataFrame(rr), ["site", "comparison", "tiles", "wins"]))

    p("\n## B5. Section 2.2: held-out MSE (m²): fit on one half, score on the other, both ways, averaged\n")
    hc = ["tile", "P4_heldout", "P2_A_heldout", "P2_B_heldout", "AD_heldout", "P4_beats_AD", "P2_A_beats_AD", "P2_B_beats_AD"]
    p(md(held, [c for c in hc if c in held.columns], 5))
    p("\n### B5b. The two directions and the A_crit chosen by each fitting half\n")
    hc2 = ["tile", "P4_h1_fit_score_h2", "P4_h2_fit_score_h1", "P4_A_crit_h1", "P4_A_crit_h2", "AD_h1_fit_score_h2", "AD_h2_fit_score_h1",
           "P2_A_h1_fit_score_h2", "P2_A_h2_fit_score_h1", "P2_A_A_crit_h1", "P2_A_A_crit_h2", "P2_B_h1_fit_score_h2", "P2_B_h2_fit_score_h1"]
    p(md(held, [c for c in hc2 if c in held.columns], 5))

    p("\n## B6. Section 1.1: per-tile m_t, its intervals, and the measured site theta\n")
    p(f"m-profile runs of the 4b/5 checkpoint (grid 0.05-0.90, step 0.01; 200 A_crit values); inputs run `{irun.name}`.\n")
    rows = []
    for t, v in inp["tiles"].items():
        rows.append({"tile": t, "m_site": v["m_site"], "m_t": v["m_t"], "m_t on grid edge": v["m_t_on_grid_edge"],
                     "1 % interval": f"{v['interval_0.01']['m_min']:.2f}-{v['interval_0.01']['m_max']:.2f} ({v['interval_0.01']['n_intervals']})",
                     "5 % interval": f"{v['interval_0.05']['m_min']:.2f}-{v['interval_0.05']['m_max']:.2f} ({v['interval_0.05']['n_intervals']})",
                     "5 % reaches edge": v["interval_0.05"]["touches_grid_edge"], "MSE* at m_t": v["mse_at_m_t"], "A_crit at m_t": v["A_crit_at_m_t"],
                     "run": v["run"]})
    p(md(pd.DataFrame(rows)))
    p("\nSite-pooled theta (run `" + inp["site_theta"]["run"] + "`, 95 % basin-bootstrap band):\n")
    rows = []
    for site in ("AP", "GM"):
        for ac in ("1000", "10000"):
            d = inp["site_theta"][site][ac]
            for est_ in ("pooled", "equal"):
                lo, hi = d[est_]["ci95"]
                rows.append({"site": site, "A_c (m²)": ac, "estimator": est_, "theta": d[est_]["theta"], "95 %": f"[{lo:.2f}, {hi:.2f}]", "basins": d["n_basins"]})
    p(md(pd.DataFrame(rows)))
    p("\nInside the band? (m_t / site m)\n")
    rows = []
    for t, v in inp["tiles"].items():
        row = {"tile": t}
        for k, x in v["inside_theta_bands"].items():
            row[k] = f"{'yes' if x['m_t_inside'] else 'no'} / {'yes' if x['m_site_inside'] else 'no'}"
        rows.append(row)
    p(md(pd.DataFrame(rows)))

    p("\n## B7. Section 1.2.1: k_s and K0 at theta = m_t (pooled through-origin chi-z above 10³ m²)\n")
    rows = []
    for t, v in inp["tiles"].items():
        k3, k4 = v["ks"]["1000"], v["ks"]["10000"]
        rows.append({"tile": t, "m_t": v["m_t"], "k_s 1e3": k3["ks"], "95 %": f"[{k3['ks_ci95'][0]:.4g}, {k3['ks_ci95'][1]:.4g}]" if k3.get("ks_ci95") else "-",
                     "basins": k3["n_basins"], "n_eff": k3.get("n_eff_centred"), "K0(m_t)": v["K0_m_t"], "K0(site m)": v["K0_site_m"],
                     "k_s 1e4 (reported)": k4["ks"], "check: k_s at site m / config - 1": v["ks_check"]["rel_diff"]})
    p(md(pd.DataFrame(rows)))

    p("\n## B8. Sections 1.2-1.3: site m against m_t\n")
    p(md(side, ["tile", "which", "m", "K0", "A_crit", "K_over_K0", "D_over_D0_A", "D_over_D0_B", "mse_partitioned", "mse_AD", "ratio_AD_over_part",
                "AD_K_over_K0", "AD_D_over_D0_A", "AD_D_over_D0_B", "joint_where", "AD_where"], 4))
    p("\nRuns:\n")
    p(md(side, ["tile", "which", "run_partitioned", "run_ad"]))

    p("\n## B9. Section 1.4: the rule per site\n")
    rows = []
    for site, r in s["site_rules"].items():
        m = r.get("m_rule")
        if not m:
            continue
        for col, v in m["medians"].items():
            rows.append({"site": site, "median of": col, "site m": v["site_m"], "m_t": v["m_t"], "relative change": v["rel_change"]})
        rows.append({"site": site, "median of": f"partitioned MSE < AD MSE at m_t: {m['n_part_below_AD_at_m_t']} of {m['of']}",
                     "site m": None, "m_t": None, "relative change": None})
        rows.append({"site": site, "median of": f"immaterial: {m['immaterial']} (MSE condition {m['cond_mse']}, medians condition {m['cond_medians']})",
                     "site m": None, "m_t": None, "relative change": None})
    p(md(pd.DataFrame(rows), ["site", "median of", "site m", "m_t", "relative change"]))

    p("\n## B10. Checks against Phase 6\n")
    p(md(ins, ["tile", "phase6_check_part", "phase6_check_ad"], 3))
    p("\n`phase6_check_part` = largest relative difference of the P4 MSE between this run's rows and the Phase 6 rows at the same A_crit; "
      "`phase6_check_ad` = AD optimum MSE / Phase 6 AD optimum MSE - 1.\n")
    p("## B11. Runs used\n")
    p(md(pd.DataFrame([{"tile": t, **v} for t, v in s["runs_used"].items()])))
    p(f"\nSummary run `{srun.name}` under `{root}`.")


if __name__ == "__main__":
    main()
