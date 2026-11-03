"""Constrained re-run driver (A >= 1e5 m^2, five transects per point,
constrained bank pick).

    python -m width_analysis.run_constrained --stage checkpoint   # first 2 per bin = 10 per site, QC frames, stop
    python -m width_analysis.run_constrained --stage full         # 10 per bin per site, sensitivity, exceedance, fits, figures

Outputs go to outputs/widths_constrained/ (checkpoint/ for the checkpoint).
Nothing in outputs/widths/ is touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures_constrained as figs
from .config import A_CRIT_M2, CONFIG, CONSTRAINED, SITES, TILES
from .constrained import (exceedance_table, measure_point, multi_transect, sensitivity_table,
                          window_is_binding)
from .fits import bootstrap_tobit, tobit_fit
from .network import FlowGraph
from .qc_constrained import plot_point_constrained
from .regime import channel_pixels, regime_boundary_table, regime_fraction_table, sweep_dx
from .run import _git_hash, _json_safe, write_gpkg
from .sampling import draw_site_sample
from .tiles import REPO_ROOT, load_tile

POINT_COLUMNS = [
    "point_id", "site", "tile", "row", "col", "x", "y", "crs", "area_m2", "local_slope", "slope_method",
    "point_class", "W_bank", "W_bank_n_valid", "W_bank_spread", "W_stage_bank", "ratio_bank_over_stage",
    "h_bank_mean", "relief_left_m", "relief_right_m", "left_class", "right_class", "n_transects",
    "n_inner_channel_transects", "class_counts", "fit_censor", "W_fit", "censor_left_bound", "censor_right_bound",
    "shape_class", "W_stage_0.25", "W_stage_0.5", "W_stage_1.0", "qc_flags",
]


def sampling_config(ccfg: dict) -> dict:
    return dict(CONFIG, a_min_sample_m2=ccfg["a_min_sample_m2"], n_area_bins=ccfg["n_area_bins"],
                full_points_per_bin=ccfg["full_points_per_bin"])


def measure(stage: str, out_dir: Path, cfg: dict, ccfg: dict, qc_dir: Path | None):
    scfg = sampling_config(ccfg)
    points, excluded, transects = [], [], []
    edges_by_site, store, pixel_sizes, map_data = {}, {}, {}, {}
    for si, site in enumerate(list(SITES)):
        rng = np.random.default_rng([cfg["random_seed"], si])
        print(f"[{site}] loading tiles", flush=True)
        tiles = {t: load_tile(site, t) for t in TILES}
        graphs = {t: FlowGraph(tiles[t], scfg["a_min_sample_m2"], scfg["confluence_min_fraction"]) for t in TILES}
        accepted, excl, edges = draw_site_sample(site, tiles, graphs, scfg, "full", rng)
        excluded.extend(excl)
        edges_by_site[site] = edges.tolist()
        if stage == "checkpoint":
            keep, seen = [], {}
            for c in accepted:                     # accepted is in bin order; first k per bin
                if seen.get(c.bin_index, 0) < ccfg["checkpoint_per_bin"]:
                    keep.append(c)
                    seen[c.bin_index] = seen.get(c.bin_index, 0) + 1
            accepted = keep
        short = [e for e in excl if e["reason"] == "bin_exhausted"]
        print(f"[{site}] {len(accepted)} points ({stage}); {len(excl)} exclusions logged; bins short: {len(short)}", flush=True)
        for k, c in enumerate(accepted):
            tile = tiles[c.tile]
            pid = f"{site}{c.tile}_{k:03d}"
            mt = multi_transect(tile, graphs[c.tile], c.row, c.col, cfg, ccfg)
            agg, per = measure_point(mt, tile.pixel_size, cfg, ccfg)
            rec = dict(point_id=pid, site=site, tile=c.tile, row=c.row, col=c.col, x=c.x, y=c.y, crs=tile.crs,
                       area_m2=c.area_m2, bin_index=c.bin_index, local_slope=c.local_slope, slope_method=c.slope_method,
                       a_crit_tile_m2=A_CRIT_M2[(site, c.tile)], pixel_size_m=tile.pixel_size)
            rec.update(agg)
            points.append(rec)
            store[pid] = mt
            pixel_sizes[pid] = tile.pixel_size
            for off, tr in zip(mt.offsets, mt.transects):
                if tr is None:
                    continue
                for sv, zv, xv, yv in zip(tr.s, tr.z, tr.x, tr.y):
                    transects.append(dict(point_id=pid, offset_m=off, s=float(sv), z=float(zv), x=float(xv), y=float(yv)))
            if qc_dir is not None:
                plot_point_constrained(tile, rec, mt, per, agg, str(qc_dir / f"{pid}.png"), cfg, ccfg)
        for t in TILES:
            map_data[(site, t)] = dict(chan=channel_pixels(tiles[t]), pixel_size=tiles[t].pixel_size)
        del tiles, graphs
    df = pd.DataFrame(points)
    rest = [c for c in df.columns if c not in POINT_COLUMNS]
    df = df[[c for c in POINT_COLUMNS if c in df.columns] + rest]
    return df, pd.DataFrame(excluded), pd.DataFrame(transects), edges_by_site, store, pixel_sizes, map_data


def fit_constrained(df: pd.DataFrame, cfg: dict, label: str, rng: np.random.Generator, binding: bool, binding_reasons: list) -> dict:
    """Censored MLE on inner_channel (observed), wide_or_terrace (right) and unresolved_narrow (left) points."""
    d = df[df.fit_censor != "excluded"]
    logA = np.log10(d.area_m2.to_numpy(float))
    W = d.W_fit.to_numpy(float)
    left = (d.fit_censor == "left").to_numpy()
    right = (d.fit_censor == "right").to_numpy()
    logW = np.log10(np.where(W > 0, W, np.nan))
    lb = np.log10(np.where(left, d.censor_left_bound, np.where(right, d.censor_right_bound, 1.0)).astype(float))
    t = tobit_fit(logA, logW, left, lb, right)
    boot = bootstrap_tobit(logA, logW, left, lb, cfg["bootstrap_n"], rng, right) if len(d) >= 3 else np.zeros((0, 2))
    out = dict(label=label, n=int(len(d)), n_inner=int((d.fit_censor == "none").sum()), n_right=int(right.sum()), n_left=int(left.sum()),
               tobit=t, bootstrap_n_ok=int(len(boot)))
    if len(boot):
        p = cfg["ci_percentiles"]
        out["c_ci"] = [float(10 ** v) for v in np.percentile(boot[:, 0], p)]
        out["b_ci"] = [float(v) for v in np.percentile(boot[:, 1], p)]
    else:
        out["c_ci"] = out["b_ci"] = [np.nan, np.nan]
    b_ok = np.isfinite(t["b"]) and len(boot) > 0 and 0.0 <= out["b_ci"][0] and out["b_ci"][1] <= 1.0
    out["qualifies"] = bool(b_ok and not binding)
    notes = []
    if not b_ok:
        notes.append("95 % CI on b not within [0, 1]")
    if binding:
        notes.append("window binding: " + "; ".join(binding_reasons))
    out["qualify_note"] = "; ".join(notes) if notes else "qualifies"
    out["_boot_ab"] = boot
    return out


def run(stage: str, out_dir: Path, cfg: dict = CONFIG, ccfg: dict = CONSTRAINED) -> pd.DataFrame:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_all = out_dir / ("qc" if stage == "checkpoint" else "qc_all")
    qc_all.mkdir(exist_ok=True)
    df, excluded, transects, edges_by_site, store, pixel_sizes, map_data = measure(stage, out_dir, cfg, ccfg, qc_all)
    df.to_csv(out_dir / "points.csv", index=False)
    excluded.to_csv(out_dir / "excluded.csv", index=False)
    transects.to_csv(out_dir / "transects.csv", index=False)
    write_gpkg(df, out_dir / "points.gpkg")
    with open(out_dir / "bins.json", "w") as f:
        json.dump({s: dict(edges_m2=e, n_points=int((df.site == s).sum())) for s, e in edges_by_site.items()}, f, indent=2)
    info = dict(stage=stage, git=_git_hash(), python=sys.version, started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                config=cfg, constrained=ccfg)
    print(df.point_class.value_counts().to_string(), flush=True)

    if stage == "full":
        rng = np.random.default_rng([cfg["random_seed"], 7])
        # window sensitivity
        sens, per_setting = sensitivity_table(store, pixel_sizes, cfg, ccfg)
        sens.to_csv(out_dir / "window_sensitivity.csv", index=False)
        binding, reasons = window_is_binding(sens, ccfg)
        # exceedance
        exc = exceedance_table(df, edges_by_site, ccfg["exceedance_thresholds_m"])
        exc.to_csv(out_dir / "exceedance.csv", index=False)
        # fits at default and the two extreme windows
        fits = {}
        extremes = [(min(ccfg["sensitivity_offsets_m"]), min(ccfg["sensitivity_reliefs_m"])),
                    (max(ccfg["sensitivity_offsets_m"]), max(ccfg["sensitivity_reliefs_m"]))]
        for site in SITES:
            fits[site] = {}
            fits[site]["default"] = fit_constrained(df[df.site == site], cfg, "default window", rng, binding, reasons)
            for (off, rel) in extremes:
                dd = per_setting[(off, rel)]
                dd = dd.merge(df[["point_id", "site", "area_m2"]], on="point_id")
                fits[site][f"offset{off:g}_relief{rel:g}"] = fit_constrained(dd[dd.site == site], cfg, f"offset {off:g} m, relief {rel:g} m", rng, binding, reasons)
            fits[site]["class_counts"] = df[df.site == site].point_class.value_counts().to_dict()
            fits[site]["n_asymmetric_banks"] = int(df[df.site == site].qc_flags.fillna("").str.contains("asymmetric_banks").sum())
            fits[site]["n_inconsistent_along_stream"] = int(df[df.site == site].qc_flags.fillna("").str.contains("inconsistent_along_stream").sum())
            ic = df[(df.site == site) & (df.point_class == "inner_channel")]
            fits[site]["ratio_bank_over_stage"] = dict(median=float(ic.ratio_bank_over_stage.median()) if len(ic) else np.nan,
                                                       frac_within_25pct=float(((ic.ratio_bank_over_stage > 0.8) & (ic.ratio_bank_over_stage < 1.25)).mean()) if len(ic) else np.nan,
                                                       n=int(len(ic)))
        fits["window_binding"] = dict(binding=binding, reasons=reasons)
        with open(out_dir / "fits.json", "w") as f:
            json.dump(_json_safe(fits), f, indent=2)
        # regime outputs only if the default fit qualifies at both sites
        dxs = sweep_dx()
        qualifying = [s for s in SITES if fits[s]["default"]["qualifies"]]
        if qualifying:
            ft, bt = [], []
            for site in qualifying:
                fdict = dict(fits[site]["default"], label="default")
                chan = [map_data[(site, t)]["chan"] for t in TILES]
                ft.append(regime_fraction_table(site, chan, fdict, dxs, cfg["ci_percentiles"]))
                bt.append(regime_boundary_table(site, fdict, dxs, cfg["ci_percentiles"]))
            pd.concat(ft).to_csv(out_dir / "regime_fraction_vs_dx.csv", index=False)
            pd.concat(bt).to_csv(out_dir / "regime_boundary.csv", index=False)
        with open(out_dir / "regime_outputs_note.txt", "w") as f:
            f.write("regime outputs written for: " + (", ".join(qualifying) if qualifying else "no site") + "\n")
            for s in SITES:
                f.write(f"{s}: {fits[s]['default']['qualify_note']}\n")
        # figures
        figdir = out_dir / "figures"
        figdir.mkdir(exist_ok=True)
        figs.fig_width_vs_area(df, fits, figdir / "fig1_width_vs_area.png")
        figs.fig_exceedance(exc, edges_by_site, figdir / "fig2_exceedance.png")
        figs.fig_consistency(df, figdir / "fig3_consistency_Wbank_vs_Wstagebank.png")
        figs.fig_sensitivity(sens, figdir / "fig4_window_sensitivity.png")
        # validation export
        val = df[df.point_class.isin(["inner_channel", "wide_or_terrace"])].copy()
        val["W_imagery_m"] = np.nan
        val["imagery_source"] = ""
        vcols = ["point_id", "site", "tile", "area_m2", "local_slope", "point_class", "W_bank", "W_stage_bank",
                 "ratio_bank_over_stage", "censor_right_bound", "qc_flags", "W_imagery_m", "imagery_source"]
        write_gpkg(val, out_dir / "validation_points.gpkg", columns=vcols)
        val[["point_id", "site", "tile", "x", "y"] + vcols[3:]].to_csv(out_dir / "validation_points.csv", index=False)
        # random gallery
        qc = out_dir / "qc"
        qc.mkdir(exist_ok=True)
        chosen = []
        for site, sub in df.groupby("site"):
            ids = sub.point_id.to_numpy()
            pick = ids[rng.choice(len(ids), min(ccfg["qc_gallery_per_site"], len(ids)), replace=False)]
            for pid in sorted(pick):
                shutil.copy2(qc_all / f"{pid}.png", qc / f"{pid}.png")
                chosen.append(pid)
        with open(qc / "gallery_points.txt", "w") as f:
            f.write("\n".join(chosen) + "\n")
    info["elapsed_s"] = round(time.time() - t0, 1)
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"wrote {out_dir} in {info['elapsed_s']} s", flush=True)
    return df


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="checkpoint", choices=["checkpoint", "full"])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    base = REPO_ROOT / CONSTRAINED["output_dir"]
    out = Path(a.out) if a.out else (base / "checkpoint" if a.stage == "checkpoint" else base)
    run(a.stage, out)


if __name__ == "__main__":
    main()
