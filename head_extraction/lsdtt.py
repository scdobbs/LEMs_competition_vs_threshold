"""Method D: DrEICH (Clubb et al. 2014) through lsdtt-channel-extraction, plus
LSDTopoTools' own Wiener/Pelletier heads for comparison.

Writes the tile's filled DEM as an ENVI .bil, a driver file, runs the binary
with a wall-clock cap, and parses the source CSVs it writes (WGS84 lat/lon ->
tile CRS -> row/col). Everything LSDTT does with area parameters is recorded
in ``run_info``.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from width_analysis.tiles import Tile


def binary_available(cfg: dict) -> str | None:
    p = Path(cfg["bin_dir"]) / "lsdtt-channel-extraction"
    return str(p) if p.exists() else None


def export_bil(tile: Tile, work: Path, name: str) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{name}.bil"
    z = np.where(np.isfinite(tile.z), tile.z, -9999.0).astype(np.float32)
    tr = from_origin(tile.x0, tile.y0, tile.dx, tile.dy)
    with rasterio.open(path, "w", driver="ENVI", height=z.shape[0], width=z.shape[1], count=1, dtype="float32",
                       crs=tile.crs, transform=tr, nodata=-9999.0) as ds:
        ds.write(z, 1)
    return path


def write_driver(work: Path, name: str, m_over_n: float, cfg: dict, tcp: int, wiener: bool) -> Path:
    lines = [
        f"read path: {work}", f"write path: {work}", f"read fname: {name}", f"write fname: {name}_tcp{tcp}",
        "raster_is_filled: true", f"min_slope_for_fill: {cfg['min_slope_for_fill']}",
        "print_area_threshold_channels: false", "print_dreich_channels: true",
        f"print_wiener_channels: {'true' if wiener else 'false'}", "print_pelletier_channels: false",
        f"m_over_n: {m_over_n}", f"A_0: {cfg['A_0']}", f"number_of_junctions_dreich: {cfg['number_of_junctions_dreich']}",
        f"threshold_contributing_pixels: {tcp}", f"pruning_drainage_area: {cfg['pruning_drainage_area']}",
        f"connected_components_threshold: {cfg['connected_components_threshold']}",
        f"curvature_threshold: {cfg['curvature_threshold']}", f"surface_fitting_radius: {cfg['surface_fitting_radius']}",
        "print_sources_to_csv: true", "print_channels_to_csv: false", "print_curvature_raster: false",
    ]
    p = work / f"{name}_tcp{tcp}.driver"
    p.write_text("\n".join(lines) + "\n")
    return p


def run_tile(tile: Tile, m_over_n: float, cfg: dict, work_root: Path, tcps=None, deadline: float | None = None) -> dict:
    """Run DrEICH (and LSDTT Wiener) on one tile for each threshold_contributing_pixels value.

    Returns dict(heads=DataFrame[method, tcp, row, col, x, y], log=list, elapsed_s).
    """
    exe = binary_available(cfg)
    name = f"{tile.site}{tile.tile}"
    work = (Path(work_root) / name).resolve()      # LSDTT runs with the directory as an argument: must be absolute
    out = dict(heads=pd.DataFrame(columns=["method", "tcp", "row", "col", "x", "y"]), log=[], elapsed_s=0.0, exe=exe)
    if exe is None:
        out["log"].append("lsdtt-channel-extraction not found; Method D skipped")
        return out
    t0 = time.time()
    export_bil(tile, work, name)
    frames = []
    for k, tcp in enumerate(tcps or [cfg["threshold_contributing_pixels"]]):
        if deadline is not None and time.time() > deadline:
            out["log"].append(f"time cap reached before tcp={tcp}")
            break
        drv = write_driver(work, name, m_over_n, cfg, tcp, cfg["print_wiener_channels"] and k == 0)
        log = work / f"{name}_tcp{tcp}.log"
        with open(log, "w") as fh:
            try:
                # LSDTT takes two arguments: the directory (with trailing slash) and the driver file name
                proc = subprocess.run([exe, str(work) + "/", drv.name], cwd=work, stdout=fh, stderr=subprocess.STDOUT, check=False,
                                      timeout=max(60, (deadline - time.time()) if deadline else 3600))
                out["log"].append(f"tcp={tcp}: exit code {proc.returncode}")
            except subprocess.TimeoutExpired:
                out["log"].append(f"tcp={tcp}: timed out")
                continue
        found = sorted(glob.glob(str(work / f"{name}_tcp{tcp}*sources*.csv")))
        out["log"].append(f"tcp={tcp}: outputs {[os.path.basename(f) for f in sorted(glob.glob(str(work / f'{name}_tcp{tcp}*')))]}")
        for f in found:
            df = parse_sources(f, tile)
            if df is None or len(df) == 0:
                continue
            base = os.path.basename(f).lower()
            # LSDTT names: <fname>_Wsources.csv (wiener), _Dsources.csv (dreich), _Psources.csv (pelletier), _ATsources.csv (area threshold)
            method = "D" if "_dsources" in base else ("W" if "_wsources" in base else ("Pl" if "_psources" in base else "?"))
            if method == "?":
                continue
            df["method"] = method
            df["tcp"] = tcp
            df["source_file"] = os.path.basename(f)
            frames.append(df)
    if frames:
        out["heads"] = pd.concat(frames, ignore_index=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def parse_sources(path: str, tile: Tile) -> pd.DataFrame | None:
    """Read an LSDTT sources CSV (latitude/longitude WGS84 or x/y) into tile row/col."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    if "latitude" in cols and "longitude" in cols:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", tile.crs, always_xy=True)
        x, y = tr.transform(df[cols["longitude"]].to_numpy(), df[cols["latitude"]].to_numpy())
    elif "x" in cols and "y" in cols:
        x, y = df[cols["x"]].to_numpy(float), df[cols["y"]].to_numpy(float)
    else:
        return None
    col = np.floor((x - tile.x0) / tile.dx).astype(int)
    row = np.floor((tile.y0 - y) / tile.dy).astype(int)
    ok = (row >= 0) & (row < tile.shape[0]) & (col >= 0) & (col < tile.shape[1])
    return pd.DataFrame(dict(row=row[ok], col=col[ok], x=x[ok], y=y[ok]))


def cleanup(work_root: Path) -> None:
    shutil.rmtree(work_root, ignore_errors=True)
