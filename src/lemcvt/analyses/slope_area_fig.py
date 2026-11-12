"""Binned slope-area regressions of observed and model grids (``fig_slope_area``). Exact port.

v1 regressed the observed filled grid and the saved *filled* grids of three model runs
(AD-no-opt, AD-opt, joint partitioned) with the tile's area and D8 grids, log-binned
(24 bins, >= 10 points per bin, medians, count-weighted least squares), using
undocumented area thresholds (AP tile 1: 8.89e3 m^2; GM tile 2: 5.49e3 m^2). Those
values are kept as ``V1_MIN_AREA``; other tiles default to the site's value.

Model grids come from a ``tile_run`` result when given (``run_dirs``), otherwise from
the saved v1 output grids under ``$LEMCVT_DATA`` (Phase 2 regression).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.config import config_snapshot, data_root, load_tile
from lemcvt.io.grids import read_geotiff
from lemcvt.io.units import z_scale_factor
from lemcvt.measure.slope_area import slope_area_regression_binned, weighted_r2
from lemcvt.results import RunRecorder, latest_run

logger = logging.getLogger(__name__)

V1_MIN_AREA = {"AP": 8.89e3, "GM": 5.49e3}
V1_TILES = ("AP1", "GM2")
MODEL_KEYS = {
    "AD_no_opt": "AD-no-opt",
    "AD_opt": "AD-opt",
    "part": "Partitioned-model-a_crit-k-d-opt",
}


def _run_z_units(run_dir: Path, default: str) -> str:
    """Vertical unit of a run's output grids: the run's working unit (``inputs.z_units_working``), else ``default``."""
    try:
        p = json.loads((Path(run_dir) / "params.json").read_text())
        return str(p.get("inputs", {}).get("z_units_working") or default)
    except (OSError, ValueError):
        return default


def convert_z(z: np.ndarray, from_units: str, to_units: str) -> np.ndarray:
    """``z`` taken from ``from_units`` to ``to_units`` (identity when they are equal)."""
    if from_units == to_units:
        return z
    return z * (z_scale_factor(from_units) / z_scale_factor(to_units))


def _model_filled_grid(td, key: str, run_dir: Path | None) -> np.ndarray:
    """The *filled* model grid in the tile's working unit, from a tile_run result or from the saved v1 outputs.

    Addendum 7, 2b: a run's grids are in the run's working unit (metres for every v2 run; the stored unit, feet on AP,
    for ``reproduce_v1_units`` runs), the saved v1 outputs in the stored unit. Each is converted from its own unit to
    the tile's working unit. Before this, the tile's stored-to-working factor (0.3048 on AP) was applied to every grid,
    so metre grids of the Phase 6 runs were converted a second time (fig_slope_area, AP: model slopes 3.28x too low).
    """
    site = td.cfg.site
    name = f"{site}_1m_best_tile_{MODEL_KEYS[key]}_filled.tif"
    if run_dir is not None:
        p = Path(run_dir) / name
        if not p.exists():
            raise FileNotFoundError(f"{p} not in run (run tile_run with derived grids)")
        from_units = _run_z_units(run_dir, default=td.cfg.z_units)
    else:
        p = data_root() / site / f"tile{td.cfg.index}" / "outputs" / name
        from_units = td.cfg.z_units
    return convert_z(read_geotiff(p).data, from_units, td.z_units_working)


def _phase6_dirs(tile: str) -> dict[str, Path]:
    """Phase 6 (addendum 6): AD-no-opt and AD-opt from the latest v2 AD run, the joint grid from the refined partitioned run."""
    from lemcvt.analyses.curvature_compare import _runs_by_stage

    runs = _runs_by_stage(tile)
    out = {}
    if "ad" in runs:
        out["AD_no_opt"] = runs["ad"][0]
        out["AD_opt"] = runs["ad"][0]
    if "joint" in runs:
        out["part"] = runs["joint"][0]
    return out


def run_slope_area(tiles: Sequence[str] = V1_TILES, run_dirs: dict[str, Path] | None = None, use_latest_runs: bool = False,
                   vertical_interval: float = 10, nbins: int = 24, run_id: str | None = None,
                   results_root=None, phase6: bool = False) -> RunRecorder:
    rec = RunRecorder("slope_area", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    out = {}
    frames = []
    snaps = {}
    for tile in tiles:
        td = load_tile(tile)
        snaps[tile] = config_snapshot(td.cfg)
        min_area = V1_MIN_AREA[td.cfg.site]
        run_dir = None
        per_key: dict[str, Path] = {}
        if phase6:
            per_key = _phase6_dirs(tile)
            missing = [k for k in MODEL_KEYS if k not in per_key]
            if missing:
                raise FileNotFoundError(f"{tile}: no Phase 6 run for {missing}")
        elif run_dirs and tile in run_dirs:
            run_dir = Path(run_dirs[tile])
        elif use_latest_runs:
            run_dir = latest_run("tile_run", tile)
        grids = {"orig": td.Z}
        for key in MODEL_KEYS:
            grids[key] = _model_filled_grid(td, key, per_key.get(key, run_dir))
        res: dict[str, Any] = {"min_area": min_area, "model_source": ({k: str(v) for k, v in per_key.items()} if phase6 else (str(run_dir) if run_dir else "v1 saved grids"))}
        for key, z in grids.items():
            Ks, theta, dbg = slope_area_regression_binned(z, td.A, td.FD, td.dx, min_area=min_area,
                                                          vertical_interval=vertical_interval, nbins=nbins,
                                                          min_per_bin=10, agg="median", require_weighted=True,
                                                          relax_if_sparse=True)
            bA, bS, bN = dbg["bin_logA"], dbg["bin_logS"], dbg["bin_counts"].astype(float)
            R2 = weighted_r2(bA, bS, np.log10(Ks) - theta * bA, bN)
            res[key] = {"Ks": float(Ks), "theta": float(theta), "R2": float(R2), "n_points": int(dbg["A"].size)}
            frames.append(pd.DataFrame({"tile": tile, "series": key, "kind": "bin", "logA": bA, "logS": bS, "count": bN}))
            frames.append(pd.DataFrame({"tile": tile, "series": key, "kind": "raw", "logA": dbg["logA"],
                                        "logS": dbg["logS"], "count": np.nan}))
            logger.info("%s %-10s Ks=%.3e theta=%.3f R2=%.3f", tile, key, Ks, theta, R2)
        out[tile] = res
    rec.add_params("tiles", out)
    rec.write_table(pd.concat(frames, ignore_index=True), "slope_area")
    rec.finalize({"tiles": snaps})
    return rec


def run_slope_area_units_check(tiles: Sequence[str], run_id: str | None = None, results_root=None) -> RunRecorder:
    """Addendum 7, 2b.1-2b.3: for each tile, the Phase 6 joint grid as the figure read it before the fix (stored-to-working
    factor applied) and as written (raw): relief and mean against the observed working-unit grid, the MSE of each against
    the observed grid beside the run's joint MSE, and the binned channel slopes of the raw grid above A_crit against
    (U/K) A^-m at the bin centres. Written as a run record for the report."""
    rec = RunRecorder("slope_area_check", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    out: dict[str, Any] = {}
    rows = []
    snaps = {}
    for tile in tiles:
        td = load_tile(tile)
        snaps[tile] = config_snapshot(td.cfg)
        per_key = _phase6_dirs(tile)
        if "part" not in per_key:
            raise FileNotFoundError(f"{tile}: no Phase 6 refined partitioned run")
        run_dir = per_key["part"]
        p = json.loads((run_dir / "params.json").read_text())
        jo = p["v2"]["partitioned"]["joint_optimum"]
        U, m = td.site.U, p["model"]["m"]
        name = f"{td.cfg.site}_1m_best_tile_{MODEL_KEYS['part']}_filled.tif"
        z_raw = read_geotiff(run_dir / name).data
        z_old = z_raw * td.z_factor                       # what _model_filled_grid did before addendum 7
        z_new = _model_filled_grid(td, "part", run_dir)
        Z = td.Z
        ok = np.isfinite(Z) & np.isfinite(z_raw)

        def rel(a):
            return float(np.nanmax(a[ok]) - np.nanmin(a[ok]))

        res: dict[str, Any] = {"run": run_dir.name, "run_z_units": _run_z_units(run_dir, td.cfg.z_units), "tile_stored_z_units": td.cfg.z_units,
                               "tile_working_z_units": td.z_units_working, "z_factor_stored_to_working": td.z_factor, "joint_mse_in_run": jo["mse"],
                               "A_crit": jo["A_crit"], "K": jo["K"], "D": jo["D"], "m": m, "U": U,
                               "relief_obs": rel(Z), "relief_raw": rel(z_raw), "relief_old_path": rel(z_old), "relief_new_path": rel(z_new),
                               "relief_ratio_raw_over_obs": rel(z_raw) / rel(Z), "relief_ratio_old_over_obs": rel(z_old) / rel(Z),
                               "mean_obs": float(np.nanmean(Z[ok])), "mean_raw": float(np.nanmean(z_raw[ok])), "mean_old_path": float(np.nanmean(z_old[ok])),
                               "mse_raw_vs_obs": float(np.mean((Z[ok] - z_raw[ok]) ** 2)), "mse_old_path_vs_obs": float(np.mean((Z[ok] - z_old[ok]) ** 2)),
                               "mse_new_path_vs_obs": float(np.mean((Z[ok] - z_new[ok]) ** 2))}
        # binned channel slopes of the raw model grid above A_crit against the model's own channel law
        _Ks, _theta, dbg = slope_area_regression_binned(z_raw, td.A, td.FD, td.dx, min_area=float(jo["A_crit"]), vertical_interval=10, nbins=24,
                                                        min_per_bin=10, agg="median", require_weighted=True, relax_if_sparse=True)
        bA, bS = 10 ** dbg["bin_logA"], 10 ** dbg["bin_logS"]
        law = (U / jo["K"]) * bA ** (-m)
        _Ks_o, _theta_o, dbg_o = slope_area_regression_binned(z_old, td.A, td.FD, td.dx, min_area=float(jo["A_crit"]), vertical_interval=10, nbins=24,
                                                              min_per_bin=10, agg="median", require_weighted=True, relax_if_sparse=True)
        bS_old = np.interp(dbg["bin_logA"], dbg_o["bin_logA"], 10 ** dbg_o["bin_logS"]) if dbg_o["bin_logA"].size else np.full(bA.size, np.nan)
        for a, s, s_old, lw in zip(bA, bS, bS_old, law, strict=True):
            rows.append({"tile": tile, "bin_A_m2": float(a), "binned_S_raw": float(s), "binned_S_old_path": float(s_old), "law_U_over_K_A_minus_m": float(lw),
                         "ratio_raw_over_law": float(s / lw), "ratio_old_over_law": float(s_old / lw)})
        rr = np.array([r["ratio_raw_over_law"] for r in rows if r["tile"] == tile])
        ro = np.array([r["ratio_old_over_law"] for r in rows if r["tile"] == tile])
        res["channel_bins"] = {"n_bins": int(bA.size), "median_ratio_raw_over_law": float(np.nanmedian(rr)), "median_ratio_old_over_law": float(np.nanmedian(ro)),
                               "raw_fit": {"Ks": float(_Ks), "theta": float(_theta)}, "old_path_fit": {"Ks": float(_Ks_o), "theta": float(_theta_o)}}
        out[tile] = res
        logger.info("%s: relief raw/obs %.4f, old path/obs %.4f; MSE raw %.3f (joint %.3f), old path %.1f; channel bins raw/law %.3f, old/law %.3f", tile,
                    res["relief_ratio_raw_over_obs"], res["relief_ratio_old_over_obs"], res["mse_raw_vs_obs"], jo["mse"], res["mse_old_path_vs_obs"],
                    res["channel_bins"]["median_ratio_raw_over_law"], res["channel_bins"]["median_ratio_old_over_law"])
    rec.add_params("tiles", out)
    rec.write_table(pd.DataFrame(rows), "channel_bins")
    rec.finalize({"tiles": snaps})
    return rec
