"""Phase 2 regression summary: compare lemcvt results with the frozen v1 reference.

Reads ``reference/v1_reference.json`` and the latest results of each step and writes
``reports/v2/02_regression_tables.md`` (markdown tables) plus a CSV of every compared
number. Used by ``lemcvt regression-report``.

Comparisons
-----------
* tile_forward: per tile and run key, max|dz|/relief and relative MSE difference against
  the saved v1 grids, with pass/fail at the directive tolerances (1e-6, 1e-8) and the
  attainable tolerance for runs at printed (rounded) parameters;
* tile_run (if any): parameters within 1 % and MSE within 0.5 % of the printed v1 values;
* sweep_resolution: ratio_d8 / ratio_dinf against the printed ``[OK]`` rows (3 decimals);
* slope_area: Ks, theta, R2 against the printed table; d8_vs_dinf: MSEs and area ranges;
* synthetic_pe: local-k_s medians and IQRs against the printed summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.config import REPO_ROOT, TILE_IDS
from lemcvt.results import find_runs, latest_run

REF = REPO_ROOT / "reference" / "v1_reference.json"


def _load_ref() -> dict:
    return json.loads(REF.read_text())


def _params(run: Path) -> dict:
    return json.loads((run / "params.json").read_text())


def _run_meta(run: Path) -> dict:
    return json.loads((run / "run.json").read_text())


def forward_table() -> pd.DataFrame:
    rows = []
    for tile in TILE_IDS:
        run = latest_run("tile_forward", tile)
        if run is None:
            continue
        p = _params(run)
        meta = _run_meta(run)
        for fr in p["forward_runs"]:
            c = fr.get("comparison")
            exact = fr["key"] in ("ad_no_opt",)
            row = {"tile": tile, "key": fr["key"], "mse": fr["mse"], "run": run.name, "wall_s_run": meta.get("wall_s"),
                   "exact_params": exact}
            if c and "max_abs_dz" in c:
                row.update({"max_dz_over_relief": c["max_abs_dz_over_relief"], "rel_mse_diff": c["rel_mse_diff"],
                            "pass_dz_1e-6": c["pass_dz"], "pass_mse_1e-8": c["pass_mse"],
                            "mse_saved": c["mse_saved"], "n_diff_cells": c["n_diff_cells"]})
            rows.append(row)
    return pd.DataFrame(rows)


def tile_run_table(ref: dict) -> pd.DataFrame:
    rows = []
    for tile in TILE_IDS:
        run = latest_run("tile_run", tile)
        if run is None:
            continue
        p = _params(run)
        st = p.get("stages_results", {})
        rs = ref["tiles"][tile]["stages"]
        for stage, keys in (("ad", ("K", "D")), ("acrit", ("A_crit",)), ("nonlinear", ("A_crit",)),
                            ("joint", ("A_crit", "D", "K"))):
            if stage not in st or stage not in rs:
                continue
            for k in keys:
                new, old = st[stage]["params"][k], rs[stage][k]
                rows.append({"tile": tile, "stage": stage, "quantity": k, "v1": old, "v2": new,
                             "rel_diff": (new - old) / old, "pass_1pct": abs(new - old) / abs(old) <= 0.01,
                             "wall_s_v2": st[stage]["wall_s"], "wall_s_v1": rs[stage].get("duration_s")})
            new, old = st[stage]["mse"], rs[stage]["MSE"]
            rows.append({"tile": tile, "stage": stage, "quantity": "MSE", "v1": old, "v2": new,
                         "rel_diff": (new - old) / old, "pass_0.5pct": abs(new - old) / abs(old) <= 0.005,
                         "wall_s_v2": st[stage]["wall_s"], "wall_s_v1": rs[stage].get("duration_s")})
    return pd.DataFrame(rows)


def sweep_table(ref: dict) -> pd.DataFrame:
    runs = find_runs("sweep_resolution")
    if not runs:
        return pd.DataFrame()
    frames = []
    for r in runs:
        t = r / "table.parquet"
        if t.exists():
            df = pd.read_parquet(t)
            df["run"] = r.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    new = pd.concat(frames, ignore_index=True).sort_values("run").drop_duplicates(["tile", "dx"], keep="last")
    v1_d8 = {(r["site"] + str(r["tile"]), round(r["dx"], 2)): r for r in ref["figure_notebooks"]["supp_fig6"]["parsed"]["ok_rows"]}
    v1_di = {(r["site"] + str(r["tile"]), round(r["dx"], 2)): r for r in ref["figure_notebooks"]["resolution_study_dinf"]["parsed"]["ok_rows"]}
    rows = []
    for _, rec in new.iterrows():
        key = (rec["tile"], round(float(rec["dx"]), 2))
        a, b = v1_d8.get(key), v1_di.get(key)
        rows.append({
            "tile": rec["tile"], "dx": rec["dx"],
            "ratio_d8_v2": rec.get("ratio_d8"), "ratio_d8_v1": a["ratio"] if a else np.nan,
            "Ks_obs_v2": rec.get("Ks_obs"), "Ks_obs_v1": a["Ks_obs"] if a else np.nan,
            "Ks_mod_v2": rec.get("Ks_mod"), "Ks_mod_v1": a["Ks_mod"] if a else np.nan,
            "ratio_dinf_v2": rec.get("ratio_dinf"), "ratio_dinf_v1": b["ratio_dinf"] if b else np.nan,
            "n_obs_v2": rec.get("npts_obs"), "n_obs_v1": a["n_obs"] if a else np.nan,
        })
    df = pd.DataFrame(rows)
    for c in ("ratio_d8", "ratio_dinf", "Ks_obs", "Ks_mod"):
        df[f"{c}_absdiff"] = (df[f"{c}_v2"] - df[f"{c}_v1"]).abs()
    return df


def slope_area_table(ref: dict) -> pd.DataFrame:
    runs = find_runs("slope_area")
    run = latest_run("slope_area", "AP1_GM2") or (runs[-1] if runs else None)
    if run is None:
        return pd.DataFrame()
    p = _params(run)
    v1 = ref["figure_notebooks"]["fig_slope_area"]["parsed"]["tiles"]
    names = {"orig": "ORIGINAL", "AD_no_opt": "AD no-opt", "AD_opt": "AD opt", "part": "PARTITIONED"}
    rows = []
    for tile, res in p["tiles"].items():
        for k, v1k in names.items():
            if k not in res or tile not in v1 or v1k not in v1[tile]:
                continue
            for q in ("Ks", "theta", "R2"):
                old, new = v1[tile][v1k][q], res[k][q]
                rows.append({"tile": tile, "series": k, "quantity": q, "v1": old, "v2": new,
                             "rel_diff": (new - old) / old if old else np.nan, "run": run.name})
    return pd.DataFrame(rows)


def d8_dinf_table(ref: dict) -> pd.DataFrame:
    runs = find_runs("d8_vs_dinf")
    run = runs[-1] if runs else None
    if run is None:
        return pd.DataFrame()
    p = _params(run)
    v1 = ref["figure_notebooks"]["d8_vs_dinf_comparison"]["parsed"]["locations"]
    lab = {"AP1": "Allegheny Plateau (tile 1)", "GM1": "Gabilan Mesa (tile 1)"}
    rows = []
    for tile, res in p["tiles"].items():
        ref_t = v1.get(lab.get(tile, ""), {})
        for q_new, q_old in (("mse_ad_d8", "MSE_AD_D8"), ("mse_ad_dinf", "MSE_AD_Dinf"), ("mse_pt_d8", "MSE_PT_D8"),
                             ("mse_pt_dinf", "MSE_PT_Dinf")):
            if q_old in ref_t:
                rows.append({"tile": tile, "quantity": q_new, "v1": ref_t[q_old], "v2": res[q_new],
                             "rel_diff": (res[q_new] - ref_t[q_old]) / ref_t[q_old], "run": run.name})
        if "A_dinf_range" in ref_t:
            rows.append({"tile": tile, "quantity": "A_dinf_max", "v1": ref_t["A_dinf_range"][1], "v2": res["A_dinf_range"][1],
                         "rel_diff": (res["A_dinf_range"][1] - ref_t["A_dinf_range"][1]) / ref_t["A_dinf_range"][1], "run": run.name})
    return pd.DataFrame(rows)


def synthetic_table(ref: dict) -> pd.DataFrame:
    runs = find_runs("synthetic_pe")
    run = runs[-1] if runs else None
    if run is None:
        return pd.DataFrame()
    p = _params(run)
    v1 = ref["figure_notebooks"]["supp_fig5"]["parsed"]["summary"]
    new = {round(r["dx"], 3): r for r in p["summary_local_ks"]}
    rows = []
    for r1 in v1:
        k = min(new, key=lambda kk: abs(kk - r1["dx"]))
        r2 = new[k]
        rows.append({"dx": r1["dx"], "median_v1": r1["median_ks_over_UK"], "median_v2": r2["ks_med_norm"],
                     "q25_v1": r1["q25"], "q25_v2": r2["ks_q25_norm"], "q75_v1": r1["q75"], "q75_v2": r2["ks_q75_norm"],
                     "n_v1": r1["n_good"], "n_v2": r2["n_good"],
                     "rel_diff_median": (r2["ks_med_norm"] - r1["median_ks_over_UK"]) / r1["median_ks_over_UK"], "run": run.name})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    if df.empty:
        return "_no results yet_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def build(out_md: Path | None = None, out_dir: Path | None = None) -> Path:
    ref = _load_ref()
    out_dir = out_dir or (REPO_ROOT / "reports" / "v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_md or (out_dir / "02_regression_tables.md")
    parts = ["# Phase 2 regression tables (generated by `lemcvt regression-report`)\n"]
    fwd = forward_table()
    parts.append("## 2.1 Forward regression against the saved v1 grids\n")
    if not fwd.empty:
        show = fwd[["tile", "key", "exact_params", "max_dz_over_relief", "rel_mse_diff", "pass_dz_1e-6", "pass_mse_1e-8",
                    "wall_s_run"]].copy() if "max_dz_over_relief" in fwd else fwd
        parts.append(_md(show, ".3e"))
        fwd.to_csv(out_dir / "02_forward_regression.csv", index=False)
    else:
        parts.append("_no tile_forward results_\n")
    tr = tile_run_table(ref)
    parts.append("## 2.2 Optimizer regression (tile_run against printed v1 optima)\n")
    parts.append(_md(tr, ".4g"))
    if not tr.empty:
        tr.to_csv(out_dir / "02_optimizer_regression.csv", index=False)
    sw = sweep_table(ref)
    parts.append("## 2.3a Resolution sweep (ratios against the printed [OK] rows; v1 printed 3 decimals)\n")
    if not sw.empty:
        summ = sw.groupby("tile").agg(n=("dx", "size"), max_absdiff_ratio_d8=("ratio_d8_absdiff", "max"),
                                      max_absdiff_ratio_dinf=("ratio_dinf_absdiff", "max"),
                                      max_absdiff_Ks_obs=("Ks_obs_absdiff", "max")).reset_index()
        parts.append(_md(summ, ".3e"))
        sw.to_csv(out_dir / "02_resolution_sweep_regression.csv", index=False)
    else:
        parts.append("_no sweep results_\n")
    sa = slope_area_table(ref)
    parts.append("## 2.3b fig_slope_area regressions\n")
    parts.append(_md(sa, ".4g"))
    dd = d8_dinf_table(ref)
    parts.append("## 2.3c D8 vs D-inf MSE table\n")
    parts.append(_md(dd, ".5g"))
    sy = synthetic_table(ref)
    parts.append("## 2.3d supp_fig5 local k_s summary\n")
    parts.append(_md(sy, ".4g"))
    out_md.write_text("\n".join(parts))
    return out_md
