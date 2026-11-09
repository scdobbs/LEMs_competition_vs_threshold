#!/usr/bin/env python3
"""Build reference/v1_grids_manifest.csv and reference/v1_saved_grid_mse.csv.

Phase 0, step 4 of reports/directives/refactor_v2_directive.md.

For every grid under data/{AP,GM}/tile*/ and data/{AP,GM}/tile*/outputs/ record
sha256 of the file bytes, GeoTIFF metadata (shape, dtype, nodata, transform, CRS),
and array statistics (min, max, finite count, NaN count).  The grids are GeoTIFFs
written by TopoAnalysis through GDAL, so rasterio reads them directly.

Also computes, for every saved ``*_elevation`` output grid, the nan-aware MSE
against the tile's observed filled grid, exactly as the v1 notebooks define the
misfit (``np.nanmean((Z_mod - dem._griddata) ** 2)``).  These are the
forward-regression targets for Phase 2.

Run inside a Slurm job (reads ~1.2 GB from $HOME).  Needs rasterio + numpy.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_MANIFEST = ROOT / "reference" / "v1_grids_manifest.csv"
OUT_MSE = ROOT / "reference" / "v1_saved_grid_mse.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def grid_record(path: Path) -> tuple[dict, np.ndarray | None]:
    rec = {
        "path": str(path.relative_to(ROOT)),
        "site": path.relative_to(DATA).parts[0],
        "tile": path.relative_to(DATA).parts[1],
        "kind": "output" if path.parent.name == "outputs" else "input",
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    arr = None
    try:
        with rasterio.open(path) as src:
            rec.update({
                "driver": src.driver,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs_epsg": src.crs.to_epsg() if src.crs else None,
                "crs_wkt_sha256": hashlib.sha256(src.crs.to_wkt().encode()).hexdigest()[:16] if src.crs else None,
                "gt_x0": src.transform.c, "gt_dx": src.transform.a, "gt_rx": src.transform.b,
                "gt_y0": src.transform.f, "gt_ry": src.transform.d, "gt_dy": src.transform.e,
            })
            arr = src.read(1)
            a = arr.astype(np.float64, copy=False)
            finite = np.isfinite(a)
            if rec["nodata"] is not None and np.isfinite(rec["nodata"]):
                finite &= a != rec["nodata"]
            rec.update({
                "n_cells": int(a.size),
                "n_finite": int(finite.sum()),
                "n_nan": int(np.isnan(a).sum()),
                "min": float(a[finite].min()) if finite.any() else None,
                "max": float(a[finite].max()) if finite.any() else None,
                "mean": float(a[finite].mean()) if finite.any() else None,
            })
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec, arr


def main() -> int:
    paths = sorted(p for p in DATA.glob("*/tile*/*") if p.is_file()) + \
        sorted(p for p in DATA.glob("*/tile*/outputs/*") if p.is_file())
    print(f"{len(paths)} grid files under {DATA.relative_to(ROOT)}")

    records = []
    filled: dict[tuple[str, str], np.ndarray] = {}
    elevations: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {}
    for p in paths:
        rec, arr = grid_record(p)
        records.append(rec)
        key = (rec["site"], rec["tile"])
        if arr is not None:
            if rec["kind"] == "input" and rec["name"].endswith("_filled"):
                filled[key] = arr.astype(np.float64)
            elif rec["kind"] == "output" and rec["name"].endswith("_elevation"):
                elevations.setdefault(key, []).append((rec["name"], arr.astype(np.float64)))
        print(f"  {rec['path']}: {rec.get('dtype')} {rec.get('height')}x{rec.get('width')} "
              f"min={rec.get('min')} max={rec.get('max')} sha={rec['sha256'][:12]} {rec.get('error', '')}")

    fields = sorted({k for r in records for k in r}, key=lambda k: (k != "path", k))
    with OUT_MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"wrote {OUT_MANIFEST.relative_to(ROOT)} ({len(records)} rows)")

    # Saved-grid MSE against observed filled DEM (v1 misfit definition)
    rows = []
    for key in sorted(elevations):
        Z_obs = filled.get(key)
        if Z_obs is None:
            print(f"  no filled grid for {key}; skipping MSE")
            continue
        relief = float(np.nanmax(Z_obs) - np.nanmin(Z_obs))
        for name, Z in sorted(elevations[key]):
            if Z.shape != Z_obs.shape:
                rows.append({"site": key[0], "tile": key[1], "grid": name, "error": f"shape {Z.shape} vs {Z_obs.shape}"})
                continue
            diff = Z - Z_obs
            mse = float(np.nanmean(diff ** 2))
            rows.append({
                "site": key[0], "tile": key[1], "grid": name,
                "mse": repr(mse),
                "rmse": repr(float(np.sqrt(mse))),
                "max_abs_diff": repr(float(np.nanmax(np.abs(diff)))),
                "n_compared": int(np.isfinite(diff).sum()),
                "n_nan_model": int(np.isnan(Z).sum()),
                "obs_relief": repr(relief),
                "obs_min": repr(float(np.nanmin(Z_obs))),
                "obs_max": repr(float(np.nanmax(Z_obs))),
                "model_min": repr(float(np.nanmin(Z))),
                "model_max": repr(float(np.nanmax(Z))),
            })
            print(f"  MSE {key[0]}/{key[1]} {name}: {mse:.6e} (n={rows[-1]['n_compared']}, nan_model={rows[-1]['n_nan_model']})")
    fields = ["site", "tile", "grid", "mse", "rmse", "max_abs_diff", "n_compared", "n_nan_model",
              "obs_relief", "obs_min", "obs_max", "model_min", "model_max", "error"]
    with OUT_MSE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_MSE.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
