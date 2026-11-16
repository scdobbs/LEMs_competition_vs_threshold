"""Resolution sweep: k_s of modelled against observed topography as a function of pixel
size (``supp_fig6`` for D8; ``resolution_study_dinf`` adds D-infinity). Exact port.

Per (tile, dx): resample the 1 m *filled* grid to dx (NaN-aware bilinear zoom),
recompute hydrology (priority flood -> D8 -> area), measure Ks_obs with fixed theta
(A >= 1e5 m^2, 10 m vertical interval), run the AD model at (U, K, D) on the
resampled (unfilled) grid, measure Ks_mod the same way, and record ratios.

``params="v1"`` uses the K, D (2 significant figures) and theta tables hard-coded in
the v1 figure notebooks; they are the initial K0 = U/k_s and D0 = U/C_ht rounded,
not optimized values (``reports/v2/00_reference.md`` 3.5). ``params="config"`` uses
the tile configs' K0_partitioned, D0 and m instead.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from lemcvt.config import config_snapshot, load_tile
from lemcvt.hydro.d8 import d8_area, d8_flow_direction
from lemcvt.hydro.dinf import compute_dinf_area, compute_dinf_flow_direction
from lemcvt.hydro.fill import priority_flood_fill
from lemcvt.measure.slope_area import ks_mod_from_modeled, ks_obs_from_observed
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.dinf_models import advection_diffusion_model_dinf
from lemcvt.results import RunRecorder
from lemcvt.utils import resample_from_1m

logger = logging.getLogger(__name__)

# v1 figure-notebook tables (supp_fig6.ipynb / resolution_study_dinf.ipynb)
V1_U = {"GM": 1.47e-4, "AP": 5.5e-5}
V1_KD = {
    "GM": {1: (2.2e-05, 1.0e-02), 2: (6.1e-05, 1.0e-02), 3: (1.9e-05, 2.1e-02), 4: (3.5e-05, 9.8e-03), 5: (5.9e-05, 9.0e-03)},
    "AP": {1: (4.0e-07, 2.8e-03), 2: (2.9e-07, 2.8e-03), 3: (4.9e-07, 2.7e-03), 4: (2.0e-06, 2.5e-03), 5: (3.6e-06, 2.5e-03)},
}
V1_THETA = {
    "GM": {1: 0.379, 2: 0.308, 3: 0.417, 4: 0.313, 5: 0.300},
    "AP": {1: 0.562, 2: 0.592, 3: 0.542, 4: 0.438, 5: 0.400},
}
MIN_AREA = 1e5
VERTICAL_INTERVAL = 10
RESAMPLE_ORDER = 1


def dx_targets(dx_min: float = 5.0, dx_max: float = 100.0, n: int = 25) -> list[float]:
    t = np.logspace(np.log10(dx_min), np.log10(dx_max), n)
    return np.unique(np.sort(t)).tolist()


def run_at_target_dx(filled_1m, dx_target: float, U: float, K: float, D: float, theta: float,
                     routings: Sequence[str]) -> dict:
    """One (tile, dx) evaluation; ``filled_1m`` is the tile's filled Grid."""
    if np.isclose(dx_target, 1.0):
        dem_dx = filled_1m
    else:
        dem_dx = resample_from_1m(filled_1m, dx_target=float(dx_target), order=RESAMPLE_ORDER)
    Zr = dem_dx.data
    dx_val = dem_dx.dx

    zf = priority_flood_fill(Zr, dx_val)
    fd = d8_flow_direction(zf)
    area = d8_area(fd, zf, dx_val)

    Ks_obs, r2_obs, n_obs, Amax_obs = ks_obs_from_observed(zf, area, fd, dx_val, theta,
                                                           min_area=MIN_AREA, vertical_interval=VERTICAL_INTERVAL)
    rec = {"dx": float(dx_target), "dx_grid": dx_val, "ny": Zr.shape[0], "nx": Zr.shape[1],
           "U": U, "K": K, "D": D, "theta": theta,
           "Ks_obs": float(Ks_obs), "r2_obs": float(r2_obs), "npts_obs": int(n_obs), "Amax_obs": float(Amax_obs)}

    if "d8" in routings:
        solver = advection_diffusion_model(Zr, area, fd, dx_val, theta)
        Z_model = solver(U, K, D)
        Ks_mod, r2_mod, n_mod, Amax_mod = ks_mod_from_modeled(Z_model, area, fd, dx_val, theta,
                                                              min_area=MIN_AREA, vertical_interval=VERTICAL_INTERVAL)
        ratio = float(Ks_mod / Ks_obs) if (np.isfinite(Ks_mod) and np.isfinite(Ks_obs) and Ks_mod > 0 and Ks_obs > 0) else np.nan
        rec.update({"Ks_mod": float(Ks_mod), "r2_mod": float(r2_mod), "npts_mod": int(n_mod),
                    "Amax_mod": float(Amax_mod), "ratio_d8": ratio})
    if "dinf" in routings:
        angles, r_facet, facet_idx = compute_dinf_flow_direction(zf, dx_val, FD_d8=fd)
        A_dinf = compute_dinf_area(zf, dx_val, angles, r_facet, facet_idx)
        solver_dinf = advection_diffusion_model_dinf(Zr, area, fd, dx_val, theta, angles, r_facet, facet_idx, A_dinf)
        Z_dinf = solver_dinf(U, K, D)
        Ks_dinf, r2_dinf, n_dinf, Amax_dinf = ks_mod_from_modeled(Z_dinf, area, fd, dx_val, theta,
                                                                  min_area=MIN_AREA, vertical_interval=VERTICAL_INTERVAL)
        ratio_dinf = float(Ks_dinf / Ks_obs) if (np.isfinite(Ks_dinf) and np.isfinite(Ks_obs) and Ks_obs > 0) else np.nan
        rec.update({"Ks_dinf": float(Ks_dinf), "r2_dinf": float(r2_dinf), "npts_dinf": int(n_dinf),
                    "Amax_dinf": float(Amax_dinf), "ratio_dinf": ratio_dinf})
    return rec


def run_resolution_sweep(tiles: Sequence[str], routings: Sequence[str] = ("d8", "dinf"), dx_min: float = 5.0,
                         dx_max: float = 100.0, n_dx: int = 25, params: str = "v1", run_id: str | None = None,
                         results_root=None, d0_variant: str = "A") -> RunRecorder:
    from lemcvt.config import apply_d0_variant

    label = ("_".join(tiles) if len(tiles) <= 3 else f"{tiles[0]}-{tiles[-1]}_{len(tiles)}tiles") + ("_D0B" if d0_variant.upper() == "B" else "")
    rec = RunRecorder("sweep_resolution", label, run_id=run_id, root=results_root)
    rec.add_params("d0_variant", d0_variant.upper())
    targets = dx_targets(dx_min, dx_max, n_dx)
    rec.add_params("dx_targets", targets)
    rec.add_params("routings", list(routings))
    rec.add_params("params_source", params)
    rec.add_params("min_area", MIN_AREA)
    rec.add_params("vertical_interval", VERTICAL_INTERVAL)
    rows = []
    errors = []
    snapshots = {}
    for tile in tiles:
        td = load_tile(tile)
        cfg = td.cfg
        snapshots[tile] = config_snapshot(cfg)
        site, idx = cfg.site, cfg.index
        if params == "v1":
            U = V1_U[site]
            K, D = V1_KD[site][idx]
            theta = V1_THETA[site][idx]
        else:
            mp = apply_d0_variant(cfg.model, d0_variant)
            U, K, D, theta = mp.U, mp.K0_partitioned, mp.D0, mp.m
        for dx_t in targets:
            t0 = time.perf_counter()
            try:
                r = run_at_target_dx(td.filled, dx_t, U, K, D, theta, routings)
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s dx=%g failed", tile, dx_t)
                errors.append({"tile": tile, "dx": dx_t, "error": repr(exc)})
                continue
            r.update({"site": site, "tile": tile, "tile_index": idx, "wall_s": time.perf_counter() - t0,
                      "z_units_working": td.z_units_working})
            rows.append(r)
            logger.info("[OK] %s dx=%6.2f | ratio_d8=%s ratio_dinf=%s (%.1f s)", tile, dx_t,
                        f"{r.get('ratio_d8', float('nan')):.3f}", f"{r.get('ratio_dinf', float('nan')):.3f}", r["wall_s"])
    df = pd.DataFrame(rows)
    rec.write_table(df, "table")
    rec.add_params("n_rows", len(rows))
    rec.add_params("errors", errors)
    rec.add_params("tiles", list(tiles))
    rec.finalize({"tiles": snapshots})
    return rec
