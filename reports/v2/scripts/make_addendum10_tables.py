"""Appendix tables of the addendum 10 wrap-up report, from the run directories only.

    python reports/v2/scripts/make_addendum10_tables.py > reports/v2/_appendix_a10.md

F1 pilot and estimate; F2 the section 2 summary per tile and S_c (both outcome counts); F3 the 2.3 rule; F4 the
morphology score; F5-F9 Test 2 (6a, 6b, AS^2, 6d, sensitivity); F10 Slurm accounting of the addendum 10 jobs.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from lemcvt.config import results_root
from lemcvt.results import find_runs

HEADS_RUN = "20261113T220600Z_heads_analyse_10tiles"
JOBS_FILE = Path(__file__).with_name("a10_jobs.txt")


def fmt(v, nd=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        a = abs(v)
        if a != 0 and (a < 1e-2 or a >= 1e5):
            return f"{v:.{nd}e}"
        return f"{v:.{nd}g}" if a < 1000 else f"{v:.0f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x, nd) for x in v) + "]"
    return str(v)


def md(df: pd.DataFrame, nd=3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)


def latest(step: str, label: str | None = None) -> Path | None:
    hit = None
    for r in find_runs(step, label):
        if (r / "params.json").exists():
            hit = r
    return hit


def f1() -> str:
    out = ["## F1. Section 2.2: the pilot rows and the estimate", ""]
    for t in ("GM1", "GM3"):
        r = latest("a10_flux", f"{t}_1.25_rows")
        if r is None:
            continue
        df = pd.read_parquet(r / "flux_rows.parquet")
        out += [f"Run `{r.name}` (S_c 1.25, A_crit row {df.A_crit.iloc[0]:.1f} m^2, the 12-value grid value nearest the linear joint A_crit).", "",
                md(df[["D", "outcome", "outcome_a9", "n_iter", "stop_reason", "frac_clipped", "frac_clipped_interior", "frac_hillslope_gt_0.5Sc", "mse", "wall_s"]]), ""]
    e = latest("a10_flux_estimate")
    if e is not None:
        p = json.loads((e / "params.json").read_text())["estimate"]
        out += [f"Estimate run `{e.name}`: {p['evaluations_per_tile_and_S_c']} evaluations per tile and S_c; pilot {p['pilot_wall_per_eval_s']:.0f} s per "
                f"evaluation, {p['pilot_wall_per_iter_s']:.1f} s per Newton iteration, {p['pilot_iter_mean']:.1f} iterations on average; "
                f"expected {p['total_expected']:.0f} core-hours, {p['total_limits']:.0f} of limits at the estimate's time factor ({p['time_factor']}).", ""]
        sub = pd.DataFrame([{"array": k, **v} for k, v in p["submissions"].items()])
        out += [md(sub), ""]
        tk = pd.read_parquet(e / "estimate_tasks.parquet")
        out += [md(tk), ""]
    return "\n".join(out)


def f2_f3() -> str:
    r = latest("a10_flux_summary")
    if r is None:
        return "## F2. Section 2 summary\n\nNo summary run yet.\n"
    p = json.loads((r / "params.json").read_text())
    df = pd.read_parquet(r / "flux_summary.parquet")
    cols = ["tile", "sc_label", "S_c", "A_crit", "D", "mse", "linear_joint_mse", "mse_ratio", "D0_A", "D_over_D0_A", "n_evaluations", "n_planned",
            "n_converged", "n_no_steady_solution", "n_not_converged", "n_converged_a9rule", "n_no_steady_solution_a9rule", "n_not_converged_a9rule",
            "stop_rule_applied", "frac_gt_0.5Sc", "frac_clipped", "frac_clipped_interior", "fit", "D_ok", "pass"]
    out = ["## F2. Section 2: the flux-form profile per tile and S_c", "",
           f"Summary run `{r.name}`. Optimum over `converged` evaluations (addendum 10, 1.4 rule); counts by the 1.4 rule and by the addendum 9 rule "
           "(every hillslope cell counted). mse_ratio = optimum MSE / linear joint MSE; D0_A = the configs' D0 (option (iii), 11 m).", "",
           md(df[[c for c in cols if c in df.columns]]), "", "Contours (1 % and 5 % above the optimum, converged evaluations):", "",
           md(df[["tile", "sc_label", "contour1_A", "contour1_D", "contour5_A", "contour5_D", "edge", "run"]]), "",
           "## F3. Section 2.3: the rule", "", "```json", json.dumps({"by_S_c": p["by_S_c"], "verdict": p["verdict"], "rule": p["rule"]}, indent=1), "```", ""]
    return "\n".join(out)


def f4() -> str:
    r = latest("a9_morphology")
    if r is None:
        return ""
    p = json.loads((r / "params.json").read_text())
    rows = []
    for t, v in p["per_tile"].items():
        o = v["surfaces"]["observed"]
        rows.append({"tile": t, "surface": "observed", "label": "", "MSE": None, "C_ht": o["C_ht_median"], "ER_ratio": o["ratio_median"], "grad_median": o["grad_median"],
                     "grad_p90": o["grad_p90"], "C/obs": None, "ER/obs": None, "g50/obs": None, "g90/obs": None, "reproduces": None})
        for k, s in v["surfaces"].items():
            if k == "observed":
                continue
            q = s["ratios_model_over_observed"]
            rows.append({"tile": t, "surface": k, "label": s.get("label", ""), "MSE": s["mse"], "C_ht": s["C_ht_median"], "ER_ratio": s["ratio_median"],
                         "grad_median": s["grad_median"], "grad_p90": s["grad_p90"], "C/obs": q["C_ht"], "ER/obs": q["ER_ratio"], "g50/obs": q["grad_median"],
                         "g90/obs": q["grad_p90"], "reproduces": s["reproduces_form"]})
    out = ["## F4. Section 3: morphology score of every surface (rule: all four ratios within x1.25)", "",
           f"Run `{r.name}`. Gradients: the model's hillslope cells, the models' slope stencil; g50/obs and g90/obs compare with the observed DEM on the same cells.", "",
           md(pd.DataFrame(rows)), "", "Counts per site:", "", "```json", json.dumps(p["counts"], indent=1), "```", ""]
    return "\n".join(out)


def f5_f9() -> str:
    d = results_root() / HEADS_RUN / "head_analysis"
    if not d.exists():
        return ""
    out = [f"## F5. Section 4 (Test 2, 6a): head drainage area against the A_crit bracket (run `{HEADS_RUN}`)", ""]
    a = pd.read_csv(d / "head_areas_6a.csv")
    a = a[a.routing == "A_D8_m2"]
    out += [md(a[["smoothing", "tile", "site", "n_heads", "A_p05", "A_q1", "A_median", "A_q3", "A_p95", "A_crit_stage2", "A_crit_stage2_B", "A_crit_joint",
                  "joint_0.05_min", "joint_0.05_max", "bracket_overlaps_IQR", "ratio_joint_over_median"]]), ""]
    b = pd.read_csv(d / "area_slope_fits_6b.csv")
    out += ["## F6. Test 2, 6b: log A against log S_up_10 (OLS and RMA, 95 % bootstrap over heads within tiles)", "",
            md(b[["smoothing", "scope", "n", "b_ols", "b_ols_lo", "b_ols_hi", "b_rma", "b_rma_lo", "b_rma_hi", "r2", "ols_ci_includes_-2", "ols_ci_includes_-1.65", "ols_ci_includes_0"]]), ""]
    e = pd.read_csv(d / "AS2_envelope.csv")
    out += ["## F7. Test 2: A S^2 against the Montgomery & Dietrich (1992) envelope (500-4000 m^2), and the fitted A_crit at the median head slope", "",
            md(e[["smoothing", "scope", "n", "AS2_q1", "AS2_median", "AS2_q3", "frac_below_500", "frac_in_envelope", "frac_above_4000", "S_median",
                  "A_crit_stage2_x_Smed2", "A_crit_joint_x_Smed2", "A_crit_joint_x_Smed2_in_envelope"]]), ""]
    s = pd.read_csv(d / "spatial_correspondence_6d.csv")
    out += ["## F8. Test 2, 6d: spatial correspondence within 20 m", "",
            md(s[["smoothing", "tile", "site", "stage", "A_crit_m2", "n_heads", "n_partition_heads_interior", "frac_heads_with_partition_head",
                  "frac_partition_heads_with_head", "chance_frac_random_point_with_partition_head"]]), ""]
    v = pd.read_csv(d / "sensitivity.csv")
    v = v[v.scope.str.contains("pooled")]
    out += ["## F9. Test 2: sensitivity of the pooled exponent and median head area", "",
            md(v[["smoothing", "site", "slope", "routing", "n", "b_ols", "b_ols_lo", "b_ols_hi", "r2", "median_A"]]), ""]
    return "\n".join(out)


def f10() -> str:
    if not JOBS_FILE.exists():
        return ""
    ids = [ln.split()[0] for ln in JOBS_FILE.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    what = {ln.split()[0]: " ".join(ln.split()[1:]) for ln in JOBS_FILE.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    rows = []
    for j in ids:
        res = subprocess.run(["sacct", "-j", j, "-X", "-P", "-n", "--format=JobID,JobName,State,Elapsed,AllocCPUS,Timelimit,ElapsedRaw,TimelimitRaw"],
                             capture_output=True, text=True, check=False).stdout
        for ln in res.splitlines():
            f = ln.split("|")
            if len(f) < 8:
                continue
            used = float(f[6] or 0) * int(f[4] or 0) / 3600.0
            lim = float(f[7] or 0) * int(f[4] or 0) / 60.0 if f[7].isdigit() else float("nan")
            rows.append({"job": f[0], "name": f[1], "what": what.get(j, ""), "state": f[2], "elapsed": f[3], "CPUs": int(f[4] or 0), "time limit": f[5],
                         "core-h used": used, "core-h limit": lim})
    df = pd.DataFrame(rows)
    tot = f"Total: {df['core-h used'].sum():.1f} core-hours used, {df['core-h limit'].sum():.0f} of limits." if len(df) else ""
    return "\n".join(["## F10. Slurm accounting (sacct) of the addendum 10 jobs", "", md(df, nd=3), "", tot, ""])


def main() -> None:
    parts = [f1(), f2_f3(), f4(), f5_f9(), f10()]
    sys.stdout.write("\n".join(p for p in parts if p))


if __name__ == "__main__":
    main()
