"""Command-line driver.

    python -m width_analysis.run --site all --mode pilot    # 10 points per site: points, exclusions, QC
    python -m width_analysis.run --site all --mode full     # stratified sample, fits, figures, regime outputs

Pilot output goes to outputs/widths/pilot/, full output to outputs/widths/.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import figures
from .config import CONFIG, SITES, TILES, acrit_m2
from .fits import fit_site_channelized, fit_site_widths
from .network import FlowGraph
from .qc import hillshade, plot_point
from .regime import channel_pixels, regime_boundary_table, regime_fraction_table, sweep_dx
from .sampling import draw_site_sample
from .tiles import REPO_ROOT, load_tile
from .transect import make_transect
from .widths import measure_cross_section, stage_label

POINT_COLUMNS = [
    "point_id", "site", "tile", "row", "col", "x", "y", "crs", "area_m2", "local_slope", "slope_method",
    "steep_rill_flag", "W_primary", "W_primary_source", "censored", "censor_bound",
    "W_inner_bankbreak", "inner_relief_m", "channelized", "W_valley_floor",
    "W_stage_0.25", "W_stage_0.5", "W_stage_1.0", "shape_class", "resolved_width", "qc_flags",
]
VALIDATION_COLUMNS = [
    "point_id", "site", "tile", "area_m2", "local_slope", "channelized", "W_primary", "W_primary_source",
    "censored", "W_inner_bankbreak", "inner_relief_m", "W_valley_floor", "W_stage_0.25", "W_stage_0.5",
    "W_stage_1.0", "shape_class", "flow_azimuth_deg", "qc_flags",
]


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def measure(sites, mode, out_dir: Path, cfg: dict, qc_dir: Path | None):
    """Sample and measure every site. Returns (points, excluded, transects, edges_by_site, map_data)."""
    points, excluded, transects = [], [], []
    _ACRIT = acrit_m2()
    edges_by_site, map_data = {}, {}
    for si, site in enumerate(list(SITES)):
        if site not in sites:
            continue
        rng = np.random.default_rng([cfg["random_seed"], si])
        print(f"[{site}] loading tiles", flush=True)
        tiles = {t: load_tile(site, t) for t in TILES}
        graphs = {t: FlowGraph(tiles[t], cfg["a_min_sample_m2"], cfg["confluence_min_fraction"]) for t in TILES}
        print(f"[{site}] drawing sample ({mode})", flush=True)
        accepted, excl, edges = draw_site_sample(site, tiles, graphs, cfg, mode, rng)
        excluded.extend(excl)
        edges_by_site[site] = edges.tolist()
        print(f"[{site}] accepted {len(accepted)} points, logged {len(excl)} exclusions", flush=True)
        for k, c in enumerate(accepted):
            tile = tiles[c.tile]
            pid = f"{site}{c.tile}_{k:03d}"
            tr = c.transect if c.transect is not None else make_transect(tile, graphs[c.tile], c.row, c.col, cfg)
            m = measure_cross_section(tr.s, tr.z, tile.pixel_size, cfg)
            rec = dict(point_id=pid, site=site, tile=c.tile, row=c.row, col=c.col, x=c.x, y=c.y, crs=tile.crs,
                       area_m2=c.area_m2, bin_index=c.bin_index, local_slope=c.local_slope,
                       slope_method=c.slope_method, steep_rill_flag=bool(c.local_slope > cfg["steep_slope_flag"]),
                       reach_median_slope=c.reach_median_slope, trace_slope=c.extra.get("trace_slope", np.nan),
                       a_crit_tile_m2=_ACRIT[(site, c.tile)], pixel_size_m=tile.pixel_size,
                       flow_azimuth_deg=tr.azimuth_deg)
            rec.update(m)
            points.append(rec)
            for sv, zv, xv, yv in zip(tr.s, tr.z, tr.x, tr.y, strict=False):
                transects.append(dict(point_id=pid, s=float(sv), z=float(zv), x=float(xv), y=float(yv)))
            if qc_dir is not None:
                plot_point(tile, rec, tr, m, str(qc_dir / f"{pid}.png"), cfg)
        for t in TILES:
            tile = tiles[t]
            map_data[(site, t)] = dict(
                hillshade=hillshade(tile.z, tile.dx, tile.dy, cfg["hillshade_azimuth_deg"], cfg["hillshade_altitude_deg"]).astype(np.float32),
                chan=channel_pixels(tile), pixel_size=tile.pixel_size)
        del tiles, graphs
    df = pd.DataFrame(points)
    rest = [c for c in df.columns if c not in POINT_COLUMNS]
    df = df[[c for c in POINT_COLUMNS if c in df.columns] + rest]
    return df, pd.DataFrame(excluded), pd.DataFrame(transects), edges_by_site, map_data


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def write_gpkg(df: pd.DataFrame, path: Path, columns=None, layer_suffix="") -> None:
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not available; skipping GeoPackage", flush=True)
        return
    if path.exists():
        path.unlink()
    for site, sub in df.groupby("site"):
        crs = sub["crs"].iloc[0]
        cols = [c for c in (columns or sub.columns) if c in sub.columns and c != "crs"]
        g = gpd.GeoDataFrame(sub[cols].copy(), geometry=gpd.points_from_xy(sub.x, sub.y), crs=crs)
        g.to_file(path, layer=f"{site}{layer_suffix}", driver="GPKG")


def validation_subset(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Channelized points with A >= validation_min_area_m2 (priority 1) plus a few random others (priority 2)."""
    out = []
    for _site, sub in df.groupby("site"):
        pri = sub[sub.channelized & (sub.area_m2 >= cfg["validation_min_area_m2"])].copy()
        pri["validation_priority"] = 1
        rest = sub.drop(pri.index)
        n_extra = min(cfg["validation_extra_per_site"], len(rest))
        extra = rest.iloc[rng.choice(len(rest), n_extra, replace=False)].copy() if n_extra else rest.iloc[:0].copy()
        extra["validation_priority"] = 2
        out.append(pd.concat([pri, extra]))
    return pd.concat(out) if out else df.iloc[:0]


def copy_gallery(df: pd.DataFrame, qc_all: Path, qc_dir: Path, n_per_site: int, rng: np.random.Generator) -> list[str]:
    qc_dir.mkdir(exist_ok=True)
    chosen = []
    for _site, sub in df.groupby("site"):
        ids = sub.point_id.to_numpy()
        pick = ids[rng.choice(len(ids), min(n_per_site, len(ids)), replace=False)]
        for pid in sorted(pick):
            shutil.copy2(qc_all / f"{pid}.png", qc_dir / f"{pid}.png")
            chosen.append(pid)
    return chosen


def run(sites: list[str], mode: str, out_dir: Path, cfg: dict = CONFIG, make_qc: bool = True) -> pd.DataFrame:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_all = out_dir / ("qc" if mode == "pilot" else "qc_all")
    if make_qc:
        qc_all.mkdir(exist_ok=True)

    df, excluded, transects, edges_by_site, map_data = measure(sites, mode, out_dir, cfg, qc_all if make_qc else None)
    df.to_csv(out_dir / "points.csv", index=False)
    excluded.to_csv(out_dir / "excluded.csv", index=False)
    transects.to_csv(out_dir / "transects.csv", index=False)
    write_gpkg(df, out_dir / "points.gpkg")
    with open(out_dir / "bins.json", "w") as f:
        json.dump({s: dict(edges_m2=e, n_points=int((df.site == s).sum())) for s, e in edges_by_site.items()}, f, indent=2)
    info: dict[str, Any] = dict(mode=mode, sites=sites, git=_git_hash(), python=sys.version, platform=platform.platform(),
                started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), config=cfg,
                fixed_stage_columns=[f"W_stage_{stage_label(h)}" for h in cfg["fixed_stages_m"]])
    print(f"measured {len(df)} points in {time.time() - t0:.0f} s", flush=True)

    if mode == "full":
        rng = np.random.default_rng([cfg["random_seed"], 99])
        # validation export weighted to channelized points above 1e5 m^2
        val = validation_subset(df, cfg, rng)
        write_gpkg(val, out_dir / "validation_points.gpkg", columns=VALIDATION_COLUMNS + ["validation_priority"])
        val[["point_id", "site", "tile", "x", "y", "area_m2", "validation_priority"]].to_csv(out_dir / "validation_points.csv", index=False)
        # fits
        fits: dict[str, Any] = {}
        for site in sites:
            d = df[df.site == site]
            fits[site] = dict(
                all=fit_site_widths(d, cfg, rng, "all"),
                no_steep=fit_site_widths(d[~d.steep_rill_flag], cfg, rng, "no_steep"),
                channelized_logistic=fit_site_channelized(d, cfg, rng),
                n_points=int(len(d)), n_channelized=int(d.channelized.sum()), n_steep=int(d.steep_rill_flag.sum()),
                resolved_fraction=float(d.resolved_width.mean()), flat_bed_fraction=float((d.shape_class == "flat_bed").mean()),
            )
            print(f"[{site}] tobit all: c={fits[site]['all']['tobit']['c']:.3g} b={fits[site]['all']['tobit']['b']:.3f} "
                  f"(n={fits[site]['all']['n_channelized']}, censored {fits[site]['all']['n_censored']}); "
                  f"A50={fits[site]['channelized_logistic']['A50']:.3g}", flush=True)
        with open(out_dir / "fits.json", "w") as f:
            json.dump(_json_safe(fits), f, indent=2)
        # regime outputs
        dxs = sweep_dx()
        frac_tabs, bound_tabs = [], []
        for site in sites:
            chan = [map_data[(site, t)]["chan"] for t in TILES]
            for variant in ("all", "no_steep"):
                frac_tabs.append(regime_fraction_table(site, chan, fits[site][variant], dxs, cfg["ci_percentiles"]))
                bound_tabs.append(regime_boundary_table(site, fits[site][variant], dxs, cfg["ci_percentiles"]))
        frac_table = pd.concat(frac_tabs)
        frac_table.to_csv(out_dir / "regime_fraction_vs_dx.csv", index=False)
        pd.concat(bound_tabs).to_csv(out_dir / "regime_boundary.csv", index=False)
        # figures
        figdir = out_dir / "figures"
        figdir.mkdir(exist_ok=True)
        figures.fig_width_vs_area(df, fits, figdir / "fig1_width_vs_area.png")
        figures.fig_resolved_fraction(df, edges_by_site, figdir / "fig2_resolved_fraction.png")
        figures.fig_wd_maps(map_data, fits, figdir / "fig3_W_over_delta_maps.png")
        figures.fig_regime_fraction(frac_table, figdir / "fig4_regime_fraction_vs_dx.png")
        figures.fig_shape_class(df, edges_by_site, figdir / "fig5_shape_class.png")
        figures.fig_channelized_fraction(df, fits, edges_by_site, figdir / "fig6_channelized_fraction.png")
        figures.fig_bankbreak_vs_stage(df, figdir / "fig7_bankbreak_vs_stage025.png")
        if make_qc:
            chosen = copy_gallery(df, qc_all, out_dir / "qc", cfg["qc_gallery_per_site"], rng)
            with open(out_dir / "qc" / "gallery_points.txt", "w") as f:
                f.write("\n".join(chosen) + "\n")
    info["elapsed_s"] = round(time.time() - t0, 1)
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"wrote outputs to {out_dir} in {info['elapsed_s']} s", flush=True)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="Channel-width extraction on the 1 m lidar tiles")
    p.add_argument("--site", default="all", choices=["all", *SITES])
    p.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    p.add_argument("--out", default=None, help="output dir (default outputs/widths/pilot or outputs/widths)")
    p.add_argument("--no-qc", action="store_true")
    a = p.parse_args(argv)
    sites = list(SITES) if a.site == "all" else [a.site]
    from lemcvt.config import results_root

    out = Path(a.out) if a.out else (results_root() / "width_manual" / ("pilot" if a.mode == "pilot" else "full"))
    run(sites, a.mode, out, make_qc=not a.no_qc)


if __name__ == "__main__":
    main()
