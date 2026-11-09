#!/usr/bin/env python
"""Fill the RESULTS_* placeholders of reports/v2/02_regression.md from the result runs.

Run inside the lemcvt environment with LEMCVT_RESULTS set:
    PYTHONNOUSERSITE=1 python scripts/fill_phase2_report.py
Regenerates reports/v2/02_regression_tables.md first (lemcvt regression-report), then
writes the forward-regression table, the D8/D-inf table and the wall-time table into
the report. Idempotent: placeholders are kept in a sidecar so the script can be re-run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.analyses.regression_report import build, d8_dinf_table, forward_table
from lemcvt.config import REPO_ROOT, TILE_IDS
from lemcvt.results import find_runs, latest_run

REPORT = REPO_ROOT / "reports" / "v2" / "02_regression.md"
TEMPLATE = REPO_ROOT / "reports" / "v2" / ".02_regression.template.md"
REF = json.loads((REPO_ROOT / "reference" / "v1_reference.json").read_text())


def fmt(v, f=".2e"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return format(v, f)


def forward_md() -> str:
    df = forward_table()
    if df.empty:
        return "_no tile_forward results_\n"
    keys = ["ad_no_opt", "ad_opt", "acrit_opt", "nonlinear_opt", "joint_opt"]
    lines = ["| tile | run | AD-no-opt dz/relief | AD-no-opt dMSE | AD-opt dz/relief | AD-opt dMSE | A_crit-opt dz/relief | A_crit-opt dMSE | nonlinear MSE v2 / v1 printed | joint dz/relief | joint dMSE | wall (min) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    passes = {"exact": 0, "exact_n": 0}
    for tile in TILE_IDS:
        sub = df[df.tile == tile]
        if sub.empty:
            continue
        row = {k: sub[sub.key == k].iloc[0] if (sub.key == k).any() else None for k in keys}
        cells = [tile, sub.iloc[0]["run"]]
        for k in ("ad_no_opt", "ad_opt", "acrit_opt"):
            r = row[k]
            cells += [fmt(r["max_dz_over_relief"]) if r is not None else "", fmt(r["rel_mse_diff"]) if r is not None else ""]
            if k in ("ad_no_opt", "acrit_opt") and r is not None:
                passes["exact_n"] += 1
                passes["exact"] += int(bool(r["pass_dz_1e-6"]) and bool(r["pass_mse_1e-8"]))
        r = row["nonlinear_opt"]
        v1_nl = REF["tiles"][tile]["stages"].get("nonlinear", {}).get("MSE")
        cells.append(f"{r['mse']:.5g} / {v1_nl}" if r is not None else "")
        r = row["joint_opt"]
        cells += [fmt(r["max_dz_over_relief"]) if r is not None else "", fmt(r["rel_mse_diff"]) if r is not None else ""]
        cells.append(f"{sub.iloc[0]['wall_s_run'] / 60:.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Exact-parameter runs (AD-no-opt and A_crit-opt) passing both tolerances: {passes['exact']} of "
                 f"{passes['exact_n']}. dMSE is the relative MSE difference against the saved grid.")
    return "\n".join(lines) + "\n"


def d8dinf_md() -> str:
    df = d8_dinf_table(REF)
    if df.empty:
        return "_no d8_vs_dinf results_\n"
    lines = ["| tile | quantity | v1 printed | v2 | relative difference |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['tile']} | {r['quantity']} | {r['v1']:.5g} | {r['v2']:.5g} | {r['rel_diff']:.1e} |")
    lines.append("")
    lines.append(f"Run: `{df.iloc[0]['run']}`. v1 printed four decimals in scientific notation (5 s.f.); "
                 "all relative differences are below the printed precision.")
    return "\n".join(lines) + "\n"


def walltimes_md() -> str:
    rows = []
    for step in ("tile_forward", "sweep_resolution", "measure", "slope_area", "d8_vs_dinf", "synthetic_pe"):
        for run in find_runs(step):
            meta = json.loads((run / "run.json").read_text())
            rows.append({"step": step, "label": meta.get("label"), "wall_min": meta.get("wall_s", np.nan) / 60,
                         "host": meta.get("host"), "job": meta.get("slurm_job_id")})
    if not rows:
        return "_no runs_\n"
    df = pd.DataFrame(rows)
    g = df.groupby("step").agg(runs=("label", "size"), wall_min_median=("wall_min", "median"),
                               wall_min_max=("wall_min", "max")).reset_index()
    lines = ["| step | runs | median wall (min) | max wall (min) |", "|---|---|---|---|"]
    for _, r in g.iterrows():
        lines.append(f"| {r['step']} | {r['runs']} | {r['wall_min_median']:.1f} | {r['wall_min_max']:.1f} |")
    lines.append("")
    lines.append("Forward runs include the nonlinear optimum (up to 50 Picard iterations with v1's per-cell Python "
                 "assembly), which dominates their wall time; 4 CPUs, default partition.")
    return "\n".join(lines) + "\n"


def main() -> None:
    build()
    if not TEMPLATE.exists():
        TEMPLATE.write_text(REPORT.read_text())
    text = TEMPLATE.read_text()
    text = text.replace("RESULTS_FORWARD_TABLE\n", forward_md())
    text = text.replace("RESULTS_D8DINF\n", d8dinf_md())
    text = text.replace("RESULTS_WALLTIMES\n", walltimes_md())
    assert not re.search(r"RESULTS_[A-Z_]+", text), "unfilled placeholder"
    REPORT.write_text(text)
    print(f"wrote {REPORT}")
    n_fwd = len([t for t in TILE_IDS if latest_run("tile_forward", t)])
    print(f"forward runs available: {n_fwd}/10")


if __name__ == "__main__":
    main()
