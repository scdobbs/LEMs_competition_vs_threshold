"""Appendix tables for the addendum 9 report (E1-E6), from the run directories only.

Usage: PYTHONNOUSERSITE=1 LEMCVT_RESULTS=... python reports/v2/scripts/make_addendum9_tables.py > appendix.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ.get("LEMCVT_RESULTS", "/home/groups/hilley/lemcvt_results"))
out: list[str] = []


def latest(pattern: str) -> Path:
    c = sorted(d for d in root.glob(pattern) if (d / "run.json").exists())
    if not c:
        raise FileNotFoundError(pattern)
    return c[-1]


def P(d: Path) -> dict:
    return json.loads((d / "params.json").read_text())


def g(x, f=".3g"):
    return "-" if x is None or (isinstance(x, float) and x != x) else format(x, f)


# ------------------------------------------------------------------ E1 strip
s2 = latest("*_a9_section2_tests_strip_jacobian")
st = pd.read_parquet(s2 / "strip.parquet")
out.append(f"## E1. Section 2.2 test 2: the 1-D ridge strip, both forms\n\nRun `{s2.name}`. Ridge strip with channels (z = 0) at distance L = 10 m "
           "from the crest, solved by the 2-D model with Newton; U = 1e-4 m/yr, S_c = 0.5, D = 2 U L/(E* S_c). Roering = `integrate_profile_R` "
           "(the integrated steady curve of the divergence form); centre-form curve = (2/E*) ln cosh(E*/2), the exact 1-D curve of the v1 equation.\n")
out.append("| form | E* | dx (m) | cells | R* | R* Roering | rel. error vs Roering | R* centre-form curve | rel. error vs centre-form curve | outcome | clipped | Newton it |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in st.iterrows():
    out.append(f"| {r.form} | {r.E_star:g} | {r.dx:g} | {r.n_cells} | {r.R_star:.6f} | {r.R_roering:.6f} | {r.rel_err_vs_roering:+.2e} | {r.R_centre_form:.6f} | "
               f"{r.rel_err_vs_centre_form:+.2e} | {r.outcome} | {r.frac_clipped:.3f} | {r.n_iter} |")
out.append("")

# ------------------------------------------------------------------ E2 Jacobian
jac = pd.read_parquet(s2 / "jacobian.parquet")
out.append(f"## E2. Section 2.2 test 3: analytic Jacobian against central finite differences\n\nRun `{s2.name}`. Synthetic 40 x 40 tile "
           "(`tests/synthetic_tile.synthetic_dem`, seed 3), A_crit 200 m^2, D 5e-3, S_c 3 (nonlinearity engaged, nothing clipped), z perturbed by N(0, 0.05 m). "
           "The error grows as 1/h below h = 1e-4, i.e. it is finite-difference roundoff.\n")
out.append("| form | h (m) | unknowns | max slope / S_c | max abs error / max abs J | max entrywise rel. error (entries > 1e-3 max) |")
out.append("|---|---|---|---|---|---|")
for _, r in jac.iterrows():
    out.append(f"| {r.form} | {r.h:g} | {r.n_unknowns} | {r.max_slope_ratio:.3f} | {r.max_err_over_max_J:.1e} | {r['max_entrywise_rel_err_above_1e-3']:.1e} |")
out.append("")

# ------------------------------------------------------------------ E3 test 1, E4 pilot rows
summ = latest("*_a9_pilot_summary_*")
ps = P(summ)
out.append(f"## E3. Section 2.2 test 1: Newton against Picard at the Phase 6 setting (S_c = 1.27, v1 form)\n\nPilot summary `{summ.name}`. "
           "Picard at the configured cap (100) and tolerance (max update 1e-6 m); 'tight' = Picard to 1e-10 m with cap 2000; Newton to relative residual 1e-10 and max update 1e-6 m.\n")
out.append("| case | A_crit | D | Picard: outcome, it, wall s | tight Picard: outcome, it | Newton: outcome, it, wall s, rel. residual | max abs dz Newton-Picard (m) | Newton-tight (m) | pass (1e-6 m) |")
out.append("|---|---|---|---|---|---|---|---|---|")
for k, v in ps["per_task"].items():
    for c in v.get("test1", []):
        out.append(f"| {c['label']} | {c['A_crit']:.1f} | {c['D']:.3e} | {c['picard']['outcome']}, {c['picard']['n_iter']}, {c['picard']['wall_s']:.0f} | "
                   f"{c['picard_tight']['outcome']}, {c['picard_tight']['n_iter']} | {c['newton']['outcome']}, {c['newton']['n_iter']}, {c['newton']['wall_s']:.0f}, "
                   f"{c['newton']['rel_residual']:.1e} | {c['max_abs_dz_newton_picard']:.1e} | {c['max_abs_dz_newton_picard_tight']:.1e} | {c['pass_1e-6']} |")
out.append("")
out.append(f"## E4. Section 2.2 test 4 and the pilot rows (5.1)\n\nPilot summary `{summ.name}`. One A_crit row per tile (the addendum 6 grid value "
           "nearest the linear joint A_crit), K at the linear joint optimum, S_c = the addendum 8 primary-network fit; Newton with continuation from the largest D. "
           "clipped = fraction of hillslope cells with |grad z|/S_c > 0.99 at the last iterate.\n")
out.append("| tile | solver/form | D | outcome | iterations | stop | clipped | > 0.5 S_c | MSE | wall s | run |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for k, v in ps["per_task"].items():
    df = pd.read_parquet(root / v["run"] / "pilot_rows.parquet").sort_values(["solver", "form", "D"])
    for _, r in df.iterrows():
        out.append(f"| {r.tile} | {r.solver}/{r.form} | {r.D:.3e} | {r.outcome} | {r.n_iter} | {r.stop_reason} | {r.frac_clipped:.4f} | "
                   f"{g(r['frac_hillslope_gt_0.5Sc'], '.3f')} | {r.mse:.2f} | {r.wall_s:.0f} | `{v['run']}` |")
out.append("")
out.append("Stop rule (2.2) and estimate (5.2):\n")
out.append("```json\n" + json.dumps({"stop_rule_2_2": ps["stop_rule_2_2"], "estimate_section3": ps["estimate_section3"]}, indent=1) + "\n```\n")

# ------------------------------------------------------------------ E5 slope exceedance
se = latest("*_a9_slope_exceedance_10tiles")
d = pd.read_parquet(se / "slope_exceedance.parquet")
out.append(f"## E5. Hillslope cells above the clip at the fitted S_c (diagnostic)\n\nRun `{se.name}`. Hillslope cells = A below the linear joint A_crit, "
           "tile edge excluded; gradient from the models' cell-centre slope stencil on the 1 m grid. S_c: the addendum 8 primary-network fit and its 68 % interval.\n")
out.append("| tile | surface | cells | median slope | p90 slope | S_c | > 0.99 S_c | > 0.5 S_c | > 0.99 S_c at S_c p16 / p84 | share of clipped cells next to a channel or boundary cell |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in d.iterrows():
    out.append(f"| {r.tile} | {r.surface} | {r.n_hillslope} | {r.slope_median:.3f} | {r.slope_p90:.3f} | {r.S_c_central:.3f} | {r['frac_gt_0.99Sc_central']:.3f} | "
               f"{r['frac_gt_0.5Sc_central']:.3f} | {r['frac_gt_0.99Sc_p16']:.3f} / {r['frac_gt_0.99Sc_p84']:.3f} | {r.frac_of_clipped_adjacent_to_fixed_central:.3f} |")
out.append("")

# ------------------------------------------------------------------ E6 accounting (sacct)
import subprocess  # noqa: E402

JOBS = {"52186353": "design probe, AP1 (scancel issued during the flux solve)", "52196422": "design probe, AP1, flux first (cancelled)",
        "52196932": "srun diagnostic: 300 x 300 crop, GMRES vs direct", "52198633": "srun diagnostic: crop continuation row (stopped by me)",
        "52198921": "srun diagnostic: slope exceedance draft", "52199605": "ruff, mypy, fast suite, slope exceedance",
        "52199606": "pilot, centre part (AP1, GM1)", "52199672": "pilot, flux part (AP1, GM1)", "52199689": "pilot summary and figure",
        "52202379": "recorded section 2.2 tests 2-3", "52230236": "final ruff, mypy, fast suite"}
txt = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(JOBS), "-o", "JobID,State,Elapsed,AllocCPUS,CPUTimeRAW,Timelimit"],
                     capture_output=True, text=True).stdout
out.append("## E6. Jobs (sacct)\n")
out.append("| job | what | state | elapsed | CPUs | core-h used | time limit |\n|---|---|---|---|---|---|---|")
tot = 0.0
for line in txt.strip().splitlines():
    jid, state, el, cpus, cpu_s, lim = line.split("|")
    base = jid.split("_")[0]
    ch = int(cpu_s or 0) / 3600.0
    tot += ch
    out.append(f"| {jid} | {JOBS.get(base, '')} | {state} | {el} | {cpus} | {ch:.2f} | {lim} |")
out.append(f"\nTotal used: {tot:.1f} core-hours.\n")
print("\n".join(out))
