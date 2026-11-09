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

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.config import config_snapshot, data_root, load_tile
from lemcvt.io.grids import read_geotiff
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


def _model_filled_grid(td, key: str, run_dir: Path | None) -> np.ndarray:
    """The *filled* model grid, from a tile_run result or from the saved v1 outputs."""
    site = td.cfg.site
    name = f"{site}_1m_best_tile_{MODEL_KEYS[key]}_filled.tif"
    if run_dir is not None:
        p = Path(run_dir) / name
        if not p.exists():
            raise FileNotFoundError(f"{p} not in run (run tile_run with derived grids)")
    else:
        p = data_root() / site / f"tile{td.cfg.index}" / "outputs" / name
    z = read_geotiff(p).data
    return z * td.z_factor if td.z_factor != 1.0 else z


def run_slope_area(tiles: Sequence[str] = V1_TILES, run_dirs: dict[str, Path] | None = None, use_latest_runs: bool = False,
                   vertical_interval: float = 10, nbins: int = 24, run_id: str | None = None,
                   results_root=None) -> RunRecorder:
    rec = RunRecorder("slope_area", "_".join(tiles), run_id=run_id, root=results_root)
    out = {}
    frames = []
    snaps = {}
    for tile in tiles:
        td = load_tile(tile)
        snaps[tile] = config_snapshot(td.cfg)
        min_area = V1_MIN_AREA[td.cfg.site]
        run_dir = None
        if run_dirs and tile in run_dirs:
            run_dir = Path(run_dirs[tile])
        elif use_latest_runs:
            run_dir = latest_run("tile_run", tile)
        grids = {"orig": td.Z}
        for key in MODEL_KEYS:
            grids[key] = _model_filled_grid(td, key, run_dir)
        res = {"min_area": min_area, "model_source": str(run_dir) if run_dir else "v1 saved grids"}
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
