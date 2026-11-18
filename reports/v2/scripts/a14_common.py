"""Addendum 14 (tile expansion, stage A): shared constants and helpers for the a14_* scripts.

Every path is read from the environment ($GROUP_HOME, $SCRATCH); nothing here downloads anything.
Units: 1 ftUS = 1200/3937 m (decision 10). Heights are converted only where a file's header or metadata states feet.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

FT_US = 1200.0 / 3937.0
GROUP_HOME = os.environ.get("GROUP_HOME", "/home/groups/hilley")
SCRATCH = os.environ.get("SCRATCH", "/scratch/users/scdobbs")
SOURCE = os.path.join(GROUP_HOME, "lemcvt_data", "source")
LEM_TOPO = os.path.join(SCRATCH, "lem_topography")           # Steve's staging copies (same sha256 as the record copies)
RESULTS_ROOT = os.path.join(GROUP_HOME, "lemcvt_results")

# The selection references of the original tile search (decision 6): legacy/measurement_notebooks/
# updated_best_fit_tile_finder.ipynb, cell 1, dict TAYLOR_REF (curvature stored negative there; magnitudes here).
LEGACY_REF = {
    "GM": {"theta": 0.292, "ks": 1.9, "C_ht": 1.28e-2, "A_min": 1.0e4, "U": 1.47e-4, "E_m_per_Myr": 147.0, "K_per_Myr": 100.2},
    "AP": {"theta": 0.54, "ks": 47.2, "C_ht": 5.93e-3, "A_min": 5.04e4, "U": 5.5e-5, "E_m_per_Myr": 55.0, "K_per_Myr": 12.5},
}
# Perron et al. (2012) SI Table S2 (hilltop Laplacian magnitude, 1/m; D cm2/yr; D/K m^(2m+1); K m^(1-2m)/Myr; m)
PERRON_S2 = {
    "AP": {"E": 55.0, "lap_h": 5.5e-3, "D_cm2_yr": 100.0, "D_over_K": 802.0, "K": 12.5, "m": 0.37},
    "GM": {"E": 147.0, "lap_h": 11.8e-3, "D_cm2_yr": 124.0, "D_over_K": 124.0, "K": 100.2, "m": 0.35},
}

# TTV working lattice for the coarse (10 m NED) grid: NAD83(2011) / UTM 10N (EPSG:6339), integer-metre origin.
TTV_LATTICE = {"epsg": 6339, "x0": 530000.0, "y1": 4198000.0, "x1": 548000.0, "y0": 4180000.0, "dx": 10.0}

SITES = {
    "GM": {"coarse": os.environ.get("A14_GM_COARSE", os.path.join(SOURCE, "Perron_coarse", "GM_Lidar_overlap_10m.tif")),
           "lidar": os.path.join(LEM_TOPO, "Gabilan_mesa.tif"), "lidar_record": os.path.join(SOURCE, "GM_original", "Gabilan_mesa.tif"),
           "lidar_z_scale": 1.0, "lidar_z_units": "metre (inferred, addendum 13)", "epsg": 32610, "dx_nominal": 10.0,
           "tiles": ["GM1", "GM2", "GM3", "GM4", "GM5"], "ned_offset_expect_m": (30.0, 34.0)},
    "AP": {"coarse": os.path.join(SOURCE, "Perron_coarse", "AP_Lidar_overlap_30m.tif"),
           "lidar": os.path.join(LEM_TOPO, "PAMAP_DEM_FIXED_REPROJECT_12-2025.tif"),
           "lidar_record": os.path.join(SOURCE, "PA_original", "PAMAP_DEM_FIXED_REPROJECT_12-2025.tif"),
           "lidar_z_scale": FT_US, "lidar_z_units": "US survey foot (established addendum 13, ratio 3.28083 against the metre tiles)",
           "epsg": 26917, "dx_nominal": 30.0, "tiles": ["AP1", "AP2", "AP3", "AP4", "AP5"], "ned_offset_expect_m": (-5.0, 5.0)},
    "TTV": {"coarse": None,   # written by a14_ttv_inventory.py into the run directory (TTV_NED_10m_utm10n.tif)
            "lidar": os.path.join(LEM_TOPO, "output_be.tif"), "lidar_record": os.path.join(SOURCE, "TTV", "NCALM_2006_be", "output_be.tif"),
            "lidar_z_scale": 1.0, "lidar_z_units": "metre (header: NAD83 / UTM 10N; vertical not stated)", "epsg": 6339, "dx_nominal": 10.0,
            "tiles": [], "ned_offset_expect_m": (-5.0, 5.0)},
}


def run_dir(name: str | None = None) -> str:
    rid = os.environ.get("A14_RUN") or name
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a14_stage_a"
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dump(obj, path: str) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=default)


def slurm_record(out_dir: str) -> None:
    jid = os.environ.get("SLURM_JOB_ID")
    rec = {"slurm_job_id": jid, "host": os.uname().nodename, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(out_dir, "job.json"), "a") as f:
        f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------------------------- rasters

def write_tif(path: str, arr: np.ndarray, transform, crs, nodata=None, dtype=None) -> None:
    import rasterio
    dtype = dtype or str(arr.dtype)
    prof = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1, "dtype": dtype, "crs": crs,
            "transform": transform, "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def bin_to_lattice(src_path: str, transform, shape: tuple[int, int], z_scale: float = 1.0, block_rows: int = 512,
                   valid_fn=None, band: int = 1) -> dict[str, np.ndarray]:
    """Block-mean a fine raster onto a coarse lattice by binning cell centres (handles non-integer coarse cells).

    Returns n_total (fine cells whose centre falls in the coarse cell), n_valid, sum_z and sum_z2 of the valid cells
    (z multiplied by z_scale). Nothing is interpolated; a coarse cell's mean is the plain mean of the fine cells inside it.
    """
    import rasterio
    from rasterio.windows import Window
    ny, nx = shape
    X0, DX, Y1, DY = transform.c, transform.a, transform.f, -transform.e
    n_tot = np.zeros(ny * nx, np.int64)
    n_val = np.zeros(ny * nx, np.int64)
    s_z = np.zeros(ny * nx, np.float64)
    s_z2 = np.zeros(ny * nx, np.float64)
    with rasterio.open(src_path) as ds:
        nod = ds.nodata
        t = ds.transform
        xs = t.c + (np.arange(ds.width) + 0.5) * t.a
        cols = np.floor((xs - X0) / DX).astype(np.int64)
        col_ok = (cols >= 0) & (cols < nx)
        for r0 in range(0, ds.height, block_rows):
            r1 = min(ds.height, r0 + block_rows)
            ys = t.f + (np.arange(r0, r1) + 0.5) * t.e
            rows = np.floor((Y1 - ys) / DY).astype(np.int64)
            row_ok = (rows >= 0) & (rows < ny)
            if not row_ok.any() or not col_ok.any():
                continue
            a = ds.read(band, window=Window(0, r0, ds.width, r1 - r0)).astype(np.float64)
            valid = np.isfinite(a)
            if nod is not None and np.isfinite(nod):
                valid &= a != nod
            if valid_fn is not None:
                valid &= valid_fn(a)
            idx = (rows[:, None] * nx + cols[None, :])
            inb = row_ok[:, None] & col_ok[None, :]
            flat = idx[inb]
            n_tot += np.bincount(flat, minlength=ny * nx)
            v = valid & inb
            fv = idx[v]
            n_val += np.bincount(fv, minlength=ny * nx)
            z = a[v] * z_scale
            s_z += np.bincount(fv, weights=z, minlength=ny * nx)
            s_z2 += np.bincount(fv, weights=z * z, minlength=ny * nx)
    return {"n_total": n_tot.reshape(ny, nx), "n_valid": n_val.reshape(ny, nx), "sum_z": s_z.reshape(ny, nx), "sum_z2": s_z2.reshape(ny, nx)}


def theil_sen_sub(x: np.ndarray, y: np.ndarray, n_sub: int = 4000, n_rep: int = 7, seed: int = 0) -> dict:
    """Theil-Sen slope of y on x as the median over random subsamples (the addendum 13 recipe), plus OLS."""
    from scipy.stats import theilslopes
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    slopes, inters = [], []
    for _ in range(n_rep):
        sel = rng.choice(n, size=min(n_sub, n), replace=False) if n > n_sub else np.arange(n)
        s, i, _lo, _hi = theilslopes(y[sel], x[sel])
        slopes.append(float(s))
        inters.append(float(i))
    ols = np.polyfit(x, y, 1) if n >= 3 else [np.nan, np.nan]
    return {"theil_sen_slope": float(np.median(slopes)), "theil_sen_slope_range": [float(min(slopes)), float(max(slopes))],
            "theil_sen_intercept": float(np.median(inters)), "ols_slope": float(ols[0]), "ols_intercept": float(ols[1]), "n": int(n)}


def dz_stats(dz: np.ndarray) -> dict:
    dz = np.asarray(dz, float)
    dz = dz[np.isfinite(dz)]
    if dz.size == 0:
        return {"n": 0}
    q = np.percentile(dz, [5, 25, 50, 75, 95])
    return {"n": int(dz.size), "median": float(q[2]), "p05": float(q[0]), "p25": float(q[1]), "p75": float(q[3]), "p95": float(q[4]),
            "mean": float(dz.mean()), "rms": float(np.sqrt(np.mean(dz * dz))), "frac_abs_gt_1m": float(np.mean(np.abs(dz) > 1)),
            "frac_abs_gt_2m": float(np.mean(np.abs(dz) > 2))}
