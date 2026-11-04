"""One command runs everything:

    python -m acrit_slope.run            # outputs/acrit_slope/
    python -m acrit_slope.run --out DIR  # elsewhere (e.g. a quick test)

Steps: load the ten tiles, detect and measure the partition heads, apply the
exclusions, per-tile summary, across-tile fits with the two-level bootstrap,
AP tile 5 range from its misfit profile, sensitivity, supplementary S(A)
structure, figures and QC, run_info.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from width_analysis.network import FlowGraph
from width_analysis.tiles import REPO_ROOT, load_tile

from . import figures
from .config import A_CRIT_M2, A_CRIT_STAGE2_M2, ACRIT_PROFILE_CSV, CONFIG, THETA, TILE_KEYS
from .fits import (bootstrap_prediction_interval, ci, fit_common_slope, fit_ols, fit_rma,
                   ols_prediction_interval, two_level_bootstrap)
from .heads import EXCLUSION_REASONS, curvature_grid, measure_heads

HEAD_COLUMNS = ["site", "tile", "row", "col", "x", "y", "area_m2", "a_crit_m2", "kept", "reason",
                "S_up_5", "S_up_10", "S_up_20", "S_centred_5", "S_centred_10", "S_centred_20", "S_pixel", "AS2",
                "plan_curvature", "plan_curvature_sign", "convergent", "edge_distance_m", "upslope_path_m",
                "downslope_path_m", "trace_slope_20m", "flag_tile_edge", "flag_path_leaves_tile", "flag_flat_fill"]


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def measure_all(cfg: dict, out_dir: Path, rng: np.random.Generator, qc: bool = True):
    """Heads at A_crit (main), at the stage-2 A_crit and at the pseudo-head thresholds (supplementary)."""
    heads, pseudo, qc_dirs = [], [], {}
    if qc:
        qc_dirs = {k: out_dir / "figures" / k for k in ("qc_maps", "qc_profiles")}
        for d in qc_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
    for site, t in TILE_KEYS:
        t0 = time.time()
        tile = load_tile(site, t)
        graph = FlowGraph(tile, A_CRIT_M2[(site, t)], 0.0)
        curv = curvature_grid(tile, cfg["curvature_block_m"])
        recs = measure_heads(tile, graph, A_CRIT_M2[(site, t)], cfg, curv)
        for r in recs:
            r["head_set"] = "A_crit"
        heads.extend(recs)
        df_t = pd.DataFrame(recs)
        n_kept = int(df_t.kept.sum())
        print(f"[{site}{t}] A_crit={A_CRIT_M2[(site, t)]:.4g}: {len(recs)} heads, {n_kept} kept; "
              f"curvature block {curv[1]} px = {curv[2]:.2f} m", flush=True)
        for label, thr in [("stage2", A_CRIT_STAGE2_M2[(site, t)])] + [("pseudo", a) for a in cfg["pseudo_head_thresholds_m2"]]:
            rr = measure_heads(tile, graph, thr, cfg, curv)
            for r in rr:
                r["head_set"] = label
                r["threshold_m2"] = thr
            pseudo.extend(rr)
        if qc:
            figures.fig_qc_map(tile, df_t, cfg, qc_dirs["qc_maps"] / f"{site}{t}.png")
            figures.fig_qc_profiles(tile, graph, df_t, cfg, rng, qc_dirs["qc_profiles"] / f"{site}{t}.png")
        print(f"[{site}{t}] done in {time.time() - t0:.0f} s", flush=True)
        del tile, graph
    return pd.DataFrame(heads), pd.DataFrame(pseudo)


def summarize(heads: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    Ld = f"S_up_{cfg['L_default_m']:g}"
    rows = []
    for site, t in TILE_KEYS:
        sub = heads[(heads.site == site) & (heads.tile == t)]
        k = sub[sub.kept]
        row = dict(site=site, tile=t, A_crit_m2=A_CRIT_M2[(site, t)], theta=THETA[(site, t)],
                   n_heads_detected=len(sub), n_heads_kept=len(k))
        for r in EXCLUSION_REASONS:
            row[f"n_excl_{r}"] = int((sub.reason == r).sum())
        q = k[Ld].quantile([0.25, 0.5, 0.75])
        row.update({f"{Ld}_median": q[0.5], f"{Ld}_q25": q[0.25], f"{Ld}_q75": q[0.75], f"{Ld}_iqr": q[0.75] - q[0.25]})
        for c in ("S_up_5", "S_up_20", "S_centred_10", "S_pixel"):
            row[f"{c}_median"] = k[c].median()
        row["A_median_m2"] = k.area_m2.median()
        row["A_median_over_A_crit"] = row["A_median_m2"] / row["A_crit_m2"]
        row["AS2_median"] = k.AS2.median()
        row["convergent_fraction"] = k.convergent.mean() if len(k) else np.nan
        row["n_curvature_defined"] = int(np.isfinite(k.plan_curvature).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------

def _tile_inputs(summary: pd.DataFrame, heads: pd.DataFrame, slope_col: str, keys):
    logA = {k: float(np.log10(A_CRIT_M2[k])) for k in keys}
    site = {k: k[0] for k in keys}
    hs = {k: heads[(heads.site == k[0]) & (heads.tile == k[1]) & heads.kept][slope_col].dropna().to_numpy() for k in keys}
    x = np.array([np.log10(np.median(hs[k])) for k in keys])
    y = np.array([logA[k] for k in keys])
    g = np.array([site[k] for k in keys])
    return logA, site, hs, x, y, g


def run_variant(name, method, keys, summary, heads, slope_col, cfg, rng, label="", resample_tiles=True) -> dict:
    logA, site, hs, x, y, g = _tile_inputs(summary, heads, slope_col, keys)
    f = {"OLS": fit_ols, "RMA": fit_rma}[method](x, y) if method != "common_slope" else fit_common_slope(x, y, g)
    boot = two_level_bootstrap(keys, logA, site, hs, method, cfg["bootstrap_n"], rng, resample_tiles)
    cib = ci(boot["b"], cfg["ci_percentiles"])
    out = dict(name=name, method=method, slope_col=slope_col, tiles=[f"{k[0]}{k[1]}" for k in keys], n=len(keys),
               b=f["b"], ci_b=cib, includes_minus2=bool(cib[0] <= cfg["reference_exponent"] <= cib[1]),
               sigma=f["sigma"], r2=f["r2"], residuals=dict(zip([f"{k[0]}{k[1]}" for k in keys], f["residuals"])),
               boot_n_ok=boot["n_ok"], boot_n_skipped=boot["n_skipped"], label=label,
               x_log10_slope=dict(zip([f"{k[0]}{k[1]}" for k in keys], x.tolist())))
    if method == "common_slope":
        out["intercepts"] = f["intercepts"]
        out["ci_intercepts"] = {lv: ci(v, cfg["ci_percentiles"]) for lv, v in boot["intercepts"].items()}
        out["intercept_difference_AP_minus_GM"] = f["intercepts"]["AP"] - f["intercepts"]["GM"]
        if "AP" in boot["intercepts"] and "GM" in boot["intercepts"]:
            m = min(len(boot["intercepts"]["AP"]), len(boot["intercepts"]["GM"]))
            out["ci_intercept_difference"] = ci(boot["intercepts"]["AP"][:m] - boot["intercepts"]["GM"][:m], cfg["ci_percentiles"])
    else:
        out["a"] = f["a"]
        out["ci_a"] = ci(boot["a"], cfg["ci_percentiles"])
    out["_boot"] = boot
    out["_fit"] = f
    out["_xy"] = (x, y, g, hs)
    return out


def all_fits(summary: pd.DataFrame, heads: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict:
    Ld = f"S_up_{cfg['L_default_m']:g}"
    keys = list(TILE_KEYS)
    ap5 = ("AP", 5)
    no5 = [k for k in keys if k != ap5]
    V = {}
    V["v1_all_OLS"] = run_variant("v1_all_OLS", "OLS", keys, summary, heads, Ld, cfg, rng, "all ten tiles, one line")
    V["v1_all_RMA"] = run_variant("v1_all_RMA", "RMA", keys, summary, heads, Ld, cfg, rng, "all ten tiles, one line")
    V["v2_common_slope"] = run_variant("v2_common_slope", "common_slope", keys, summary, heads, Ld, cfg, rng, "common slope, separate site intercepts")
    for s in ("AP", "GM"):
        ks = [k for k in keys if k[0] == s]
        for m in ("OLS", "RMA"):
            V[f"v3_{s}_{m}"] = run_variant(f"v3_{s}_{m}", m, ks, summary, heads, Ld, cfg, rng, f"{s} alone (n = 5): low power")
    V["v4_noAP5_OLS"] = run_variant("v4_noAP5_OLS", "OLS", no5, summary, heads, Ld, cfg, rng, "nine tiles without AP5, one line")
    V["v4_noAP5_RMA"] = run_variant("v4_noAP5_RMA", "RMA", no5, summary, heads, Ld, cfg, rng, "nine tiles without AP5, one line")
    V["v4_noAP5_common_slope"] = run_variant("v4_noAP5_common_slope", "common_slope", no5, summary, heads, Ld, cfg, rng,
                                             "nine tiles without AP5, common slope, site intercepts")

    # AP tile 5 against the fits that exclude it
    hs5 = heads[(heads.site == "AP") & (heads.tile == 5) & heads.kept][Ld].dropna().to_numpy()
    x5, y5 = float(np.log10(np.median(hs5))), float(np.log10(A_CRIT_M2[ap5]))
    x5_samples = np.array([np.median(rng.choice(hs5, len(hs5), replace=True)) for _ in range(cfg["bootstrap_n"])])
    ap5_out = dict(log10_median_slope=x5, log10_A_crit=y5, median_slope=10**x5, A_crit=A_CRIT_M2[ap5])
    for name in ("v4_noAP5_OLS", "v4_noAP5_RMA", "v4_noAP5_common_slope"):
        v = V[name]
        f = v["_fit"]
        pred = (f["a"] + f["b"] * x5) if v["method"] != "common_slope" else (f["intercepts"]["AP"] + f["b"] * x5)
        resid = y5 - pred
        entry = dict(predicted_log10_A=pred, predicted_A=10**pred, residual_log10=resid, residual_factor=10**resid,
                     residual_over_sigma=resid / f["sigma"] if f["sigma"] else np.nan)
        bpi = bootstrap_prediction_interval(v["_boot"], f["residuals"], np.log10(x5_samples), rng, cfg["ci_percentiles"],
                                            intercept_key="AP" if v["method"] == "common_slope" else None)
        entry["bootstrap_PI_log10"] = [bpi["lo"], bpi["hi"]]
        entry["bootstrap_PI_A"] = [10**bpi["lo"], 10**bpi["hi"]]
        entry["inside_bootstrap_PI"] = bool(bpi["lo"] <= y5 <= bpi["hi"])
        if v["method"] == "OLS":
            x, y, _, _ = v["_xy"]
            cpi = ols_prediction_interval(x, y, x5)
            entry["classical_PI_log10"] = [cpi["lo"], cpi["hi"]]
            entry["classical_PI_A"] = [10**cpi["lo"], 10**cpi["hi"]]
            entry["inside_classical_PI"] = bool(cpi["lo"] <= y5 <= cpi["hi"])
        ap5_out[name] = entry

    # sensitivity: variant 1 with the other slope definitions
    sens = {}
    for col in ("S_up_5", Ld, "S_up_20", "S_centred_10"):
        sens[col] = {}
        for m in ("OLS", "RMA"):
            v = run_variant(f"sens_{col}_{m}", m, keys, summary, heads, col, cfg, rng, "sensitivity")
            sens[col][m] = {k: v[k] for k in ("b", "ci_b", "includes_minus2", "r2")}
    b10 = {m: sens[Ld][m]["b"] for m in ("OLS", "RMA")}
    for col in sens:
        for m in ("OLS", "RMA"):
            sens[col][m]["change_from_default"] = sens[col][m]["b"] - b10[m]
    return dict(slope_col=Ld, variants=V, AP5=ap5_out, sensitivity=sens)


def ap5_acrit_range(cfg: dict) -> dict:
    """A_crit values of AP tile 5 whose misfit is within misfit_tolerance of the profile minimum."""
    p = REPO_ROOT / ACRIT_PROFILE_CSV[("AP", 5)]
    df = pd.read_csv(p).sort_values("A_crit_m2")
    a, m = df.A_crit_m2.to_numpy(), df.mse.to_numpy()
    k = int(np.argmin(m))
    thr = (1 + cfg["misfit_tolerance"]) * m[k]
    inside = np.nonzero(m <= thr)[0]
    la = np.log10(a)
    # interpolated crossings in log A on either side of the minimum
    lo = la[inside[0]]
    if inside[0] > 0:
        i = inside[0]
        lo = np.interp(thr, [m[i], m[i - 1]], [la[i], la[i - 1]])
    hi = la[inside[-1]]
    if inside[-1] < len(a) - 1:
        i = inside[-1]
        hi = np.interp(thr, [m[i], m[i + 1]], [la[i], la[i + 1]])
    return dict(profile=str(ACRIT_PROFILE_CSV[("AP", 5)]), A_min=float(a[k]), mse_min=float(m[k]), mse_threshold=float(thr),
                grid_values_within=a[inside].tolist(), lo=float(10**lo), hi=float(10**hi),
                note="lo/hi are log-linear interpolations of the misfit profile at the threshold; grid_values_within are the sampled A_crit inside it")


def supplementary(summary: pd.DataFrame, heads: pd.DataFrame, pseudo: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict:
    """Beyond the directive: the within-tile S(A) structure near A_crit and fits that do not inherit it."""
    Ld = f"S_up_{cfg['L_default_m']:g}"
    keys = list(TILE_KEYS)
    # per-tile median slope of kept pseudo-heads at each threshold
    rows = []
    for site, t in keys:
        for thr in cfg["pseudo_head_thresholds_m2"]:
            k = pseudo[(pseudo.site == site) & (pseudo.tile == t) & (pseudo.head_set == "pseudo") & (pseudo.threshold_m2 == thr) & pseudo.kept]
            rows.append(dict(site=site, tile=t, threshold_m2=thr, n_kept=len(k), **{f"{Ld}_median": k[Ld].median(),
                             f"{Ld}_q25": k[Ld].quantile(0.25), f"{Ld}_q75": k[Ld].quantile(0.75)}))
    ps = pd.DataFrame(rows)
    # pooled within-tile log-log slope of median S against threshold (tile intercepts)
    ok = np.isfinite(ps[f"{Ld}_median"])
    f = fit_common_slope(np.log10(ps.threshold_m2[ok]), np.log10(ps[f"{Ld}_median"][ok]), (ps.site[ok] + ps.tile[ok].astype(str)).to_numpy())
    theta_h = -f["b"]
    within = dict(theta_h=theta_h, null_exponent=-1.0 / theta_h if theta_h else np.nan, r2=f["r2"],
                  note="log10 median S_up_10 = a_tile - theta_h log10 A_threshold over the pseudo-head thresholds; "
                       "if tiles shared one S(A) curve, any placement of A_crit would yield b = -1/theta_h")
    # slope at a fixed reference area: no mechanical dependence on A_crit
    aref = cfg["reference_area_m2"]
    ref = {}
    hs_ref = {k: pseudo[(pseudo.site == k[0]) & (pseudo.tile == k[1]) & (pseudo.head_set == "pseudo") & (pseudo.threshold_m2 == aref) & pseudo.kept][Ld].dropna().to_numpy() for k in keys}
    logA = {k: float(np.log10(A_CRIT_M2[k])) for k in keys}
    site = {k: k[0] for k in keys}
    x = np.array([np.log10(np.median(hs_ref[k])) for k in keys])
    y = np.array([logA[k] for k in keys])
    for m in ("OLS", "RMA"):
        ff = {"OLS": fit_ols, "RMA": fit_rma}[m](x, y)
        boot = two_level_bootstrap(keys, logA, site, hs_ref, m, cfg["bootstrap_n"], rng)
        cib = ci(boot["b"], cfg["ci_percentiles"])
        ref[m] = dict(b=ff["b"], a=ff["a"], ci_b=cib, r2=ff["r2"], includes_minus2=bool(cib[0] <= cfg["reference_exponent"] <= cib[1]),
                      includes_zero=bool(cib[0] <= 0 <= cib[1]), boot_n_ok=boot["n_ok"])
    ref["x_log10_slope_at_A_ref"] = {f"{k[0]}{k[1]}": float(v) for k, v in zip(keys, x)}
    ref["note"] = f"log10 A_crit against log10 median S_up_10 of heads detected at a fixed A = {aref:g} m² on every tile"
    # stage-2 A_crit (K, D held at the measured values)
    st = {}
    logA2 = {k: float(np.log10(A_CRIT_STAGE2_M2[k])) for k in keys}
    hs2 = {k: pseudo[(pseudo.site == k[0]) & (pseudo.tile == k[1]) & (pseudo.head_set == "stage2") & pseudo.kept][Ld].dropna().to_numpy() for k in keys}
    x2 = np.array([np.log10(np.median(hs2[k])) if len(hs2[k]) else np.nan for k in keys])
    y2 = np.array([logA2[k] for k in keys])
    okk = np.isfinite(x2)
    for m in ("OLS", "RMA"):
        ff = {"OLS": fit_ols, "RMA": fit_rma}[m](x2[okk], y2[okk])
        kk = [k for k, o in zip(keys, okk) if o]
        boot = two_level_bootstrap(kk, logA2, site, hs2, m, cfg["bootstrap_n"], rng)
        cib = ci(boot["b"], cfg["ci_percentiles"])
        st[m] = dict(b=ff["b"], a=ff["a"], ci_b=cib, r2=ff["r2"], includes_minus2=bool(cib[0] <= cfg["reference_exponent"] <= cib[1]), n=int(okk.sum()))
    st["A_crit_stage2"] = {f"{k[0]}{k[1]}": A_CRIT_STAGE2_M2[k] for k in keys}
    st["n_kept_heads"] = {f"{k[0]}{k[1]}": int(len(hs2[k])) for k in keys}
    st["median_slope"] = {f"{k[0]}{k[1]}": float(10**v) if np.isfinite(v) else None for k, v in zip(keys, x2)}
    st["ratio_stage4_over_stage2"] = {f"{k[0]}{k[1]}": A_CRIT_M2[k] / A_CRIT_STAGE2_M2[k] for k in keys}
    st["note"] = "same test with the stage-2 A_crit (A_crit optimized alone, K and D at the per-tile measured values)"
    return dict(pseudo_table=ps, within_tile_slope_area=within, reference_area_fit=ref, stage2_fit=st)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--n-boot", type=int, default=None)
    p.add_argument("--no-qc", action="store_true")
    a = p.parse_args(argv)
    cfg = dict(CONFIG)
    if a.n_boot:
        cfg["bootstrap_n"] = a.n_boot
    out_dir = Path(a.out) if a.out else REPO_ROOT / cfg["output_dir"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    info = dict(git=_git_hash(), python=sys.version, platform=platform.platform(),
                started=time.strftime("%Y-%m-%d %H:%M:%S"), config=cfg,
                A_crit_m2={f"{k[0]}{k[1]}": v for k, v in A_CRIT_M2.items()}, theta={f"{k[0]}{k[1]}": v for k, v in THETA.items()})
    rng = np.random.default_rng(cfg["random_seed"])

    heads, pseudo = measure_all(cfg, out_dir, rng, qc=not a.no_qc)
    cols = [c for c in HEAD_COLUMNS if c in heads.columns] + [c for c in heads.columns if c not in HEAD_COLUMNS]
    heads = heads[cols]
    heads.to_csv(out_dir / "heads.csv", index=False)
    heads[~heads.kept].to_csv(out_dir / "excluded_heads.csv", index=False)
    pseudo.to_csv(out_dir / "pseudo_heads.csv", index=False)

    summary = summarize(heads, cfg)
    summary.to_csv(out_dir / "tile_summary.csv", index=False)
    print(summary.to_string(), flush=True)

    fits = all_fits(summary, heads, cfg, rng)
    fits["AP5_acrit_range"] = ap5_acrit_range(cfg)
    supp = supplementary(summary, heads, pseudo, cfg, rng)
    supp["pseudo_table"].to_csv(out_dir / "pseudo_head_summary.csv", index=False)
    fits["supplementary"] = {k: v for k, v in supp.items() if k != "pseudo_table"}
    with open(out_dir / "fits.json", "w") as fh:
        json.dump(_json_safe(fits), fh, indent=1)
    for name, v in fits["variants"].items():
        print(f"{name:26s} b = {v['b']:7.3f}  95% [{v['ci_b'][0]:7.3f}, {v['ci_b'][1]:7.3f}]  -2 inside: {v['includes_minus2']}  r2={v['r2']:.2f}", flush=True)

    boot = fits["variants"]["v1_all_RMA"]["_boot"]
    boot_ab = np.column_stack([boot["a"], boot["b"]]) if boot["a"] is not None else None
    figures.fig_main(summary, _json_safe(fits), boot_ab, fits["AP5_acrit_range"], cfg, fig_dir / "fig1_acrit_vs_slope.png")
    figures.fig_boxes(heads, summary, fig_dir / "fig2_head_slopes_by_tile.png")
    figures.fig_sensitivity(fits["sensitivity"], cfg, fig_dir / "fig3_sensitivity.png")
    figures.fig_slope_area_structure(supp["pseudo_table"], summary, supp, cfg, fig_dir / "fig6_supp_slope_area_structure.png")

    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["elapsed_s"] = round(time.time() - t_start, 1)
    info["n_heads"] = int(len(heads))
    info["n_kept"] = int(heads.kept.sum())
    with open(out_dir / "run_info.json", "w") as fh:
        json.dump(_json_safe(info), fh, indent=1)
    print(f"wrote {out_dir} in {info['elapsed_s']} s", flush=True)


if __name__ == "__main__":
    main()
