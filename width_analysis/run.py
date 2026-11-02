"""Command-line driver.

    python -m width_analysis.run --site all --mode pilot
    python -m width_analysis.run --site all --mode full

Writes points.csv, excluded.csv, transects.csv, points.gpkg, bins.json,
run_info.json and a QC PNG per point under the output directory.
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

from .config import A_CRIT_M2, CONFIG, SITES, TILES
from .network import FlowGraph
from .qc import plot_point
from .sampling import draw_site_sample
from .tiles import REPO_ROOT, load_tile
from .transect import make_transect
from .widths import measure_cross_section, stage_label

POINT_COLUMNS = [
    "point_id", "site", "tile", "row", "col", "x", "y", "crs", "area_m2", "local_slope",
    "W_hand_bankfull", "h_bankfull", "W_hand_0.25", "W_hand_0.5", "W_hand_1.0",
    "W_bankbreak", "W_valley", "censored", "censor_bound", "shape_class", "qc_flags",
]


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run(sites: list[str], mode: str, out_dir: Path, cfg: dict = CONFIG, make_qc: bool = True) -> pd.DataFrame:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "qc"
    if make_qc:
        qc_dir.mkdir(exist_ok=True)

    points: list[dict] = []
    excluded: list[dict] = []
    transects: list[dict] = []
    bins_info: dict = {}

    for si, site in enumerate(sites):
        rng = np.random.default_rng([cfg["random_seed"], si])
        print(f"[{site}] loading tiles", flush=True)
        tiles = {t: load_tile(site, t) for t in TILES}
        graphs = {t: FlowGraph(tiles[t], cfg["a_min_sample_m2"], cfg["confluence_min_fraction"]) for t in TILES}
        print(f"[{site}] drawing sample ({mode})", flush=True)
        accepted, excl, edges = draw_site_sample(site, tiles, graphs, cfg, mode, rng)
        excluded.extend(excl)
        bins_info[site] = dict(edges_m2=edges.tolist(), n_accepted=len(accepted),
                               n_excluded=len([e for e in excl if e["reason"] != "bin_exhausted"]))
        print(f"[{site}] accepted {len(accepted)} points, logged {len(excl)} exclusions", flush=True)

        for k, c in enumerate(accepted):
            tile = tiles[c.tile]
            pid = f"{site}{c.tile}_{k:03d}"
            tr = c.transect if c.transect is not None else make_transect(tile, graphs[c.tile], c.row, c.col, cfg)
            m = measure_cross_section(tr.s, tr.z, tile.pixel_size, cfg)
            rec = dict(point_id=pid, site=site, tile=c.tile, row=c.row, col=c.col, x=c.x, y=c.y, crs=tile.crs,
                       area_m2=c.area_m2, bin_index=c.bin_index, local_slope=c.local_slope,
                       reach_median_slope=c.reach_median_slope, slope_method=c.slope_method,
                       trace_slope=c.extra.get("trace_slope", np.nan), a_crit_tile_m2=A_CRIT_M2[(site, c.tile)],
                       pixel_size_m=tile.pixel_size, flow_azimuth_deg=tr.azimuth_deg)
            rec.update({kk: vv for kk, vv in m.items() if kk not in ("hand_h", "hand_W")})
            points.append(rec)
            for sv, zv, xv, yv in zip(tr.s, tr.z, tr.x, tr.y):
                transects.append(dict(point_id=pid, s=float(sv), z=float(zv), x=float(xv), y=float(yv)))
            if make_qc:
                plot_point(tile, rec, tr, m, str(qc_dir / f"{pid}.png"), cfg)
        del tiles, graphs

    df = pd.DataFrame(points)
    # directive's column order first, then everything else
    rest = [c for c in df.columns if c not in POINT_COLUMNS]
    df = df[[c for c in POINT_COLUMNS if c in df.columns] + rest]
    df.to_csv(out_dir / "points.csv", index=False)
    pd.DataFrame(excluded).to_csv(out_dir / "excluded.csv", index=False)
    pd.DataFrame(transects).to_csv(out_dir / "transects.csv", index=False)
    _write_gpkg(df, out_dir / "points.gpkg")
    with open(out_dir / "bins.json", "w") as f:
        json.dump(bins_info, f, indent=2)
    info = dict(mode=mode, sites=sites, git=_git_hash(), python=sys.version, platform=platform.platform(),
                started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                elapsed_s=round(time.time() - t0, 1), config=cfg,
                fixed_stage_columns=[f"W_hand_{stage_label(h)}" for h in cfg["fixed_stages_m"]])
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"wrote {len(df)} points to {out_dir} in {info['elapsed_s']} s", flush=True)
    return df


def _write_gpkg(df: pd.DataFrame, path: Path) -> None:
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not available; skipping GeoPackage", flush=True)
        return
    if path.exists():
        path.unlink()
    for site, sub in df.groupby("site"):
        crs = sub["crs"].iloc[0]
        g = gpd.GeoDataFrame(sub.drop(columns=["crs"]), geometry=gpd.points_from_xy(sub.x, sub.y), crs=crs)
        g.to_file(path, layer=site, driver="GPKG")


def main(argv=None):
    p = argparse.ArgumentParser(description="Channel-width extraction on the 1 m lidar tiles")
    p.add_argument("--site", default="all", choices=["all", *SITES])
    p.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    p.add_argument("--out", default=None, help="output dir (default outputs/widths/<mode>)")
    p.add_argument("--no-qc", action="store_true")
    a = p.parse_args(argv)
    sites = list(SITES) if a.site == "all" else [a.site]
    out = Path(a.out) if a.out else REPO_ROOT / "outputs" / "widths" / a.mode
    run(sites, a.mode, out, make_qc=not a.no_qc)


if __name__ == "__main__":
    main()
