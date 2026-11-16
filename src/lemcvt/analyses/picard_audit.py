"""Audit of the Picard iteration of the nonlinear partitioned model (addendum 3, section 3).

``lemcvt audit picard --tile AP1`` runs the nonlinear forward model once, at the v1 printed
nonlinear optimum (K0_partitioned, D0, S_c, A_crit_nonlinear), with the instrumented
iteration (``partitioned_threshold_model_nonlinear(..., diagnostics=True)``), and records:

* the iterations used, every iteration's update norm and nonlinear residual, and whether
  the stopping test or the cap ended the loop;
* the fraction of hillslope cells with |grad z| > 0.5 S_c and > 0.9 S_c and the maximum
  |grad z| / S_c, for the final nonlinear solution, for the linear solution the iteration
  starts from, and for the observed DEM on the same cells (the model's own slope stencil);
* the MSE against the observed grid and, when a reference ``tile_forward`` run is given, the
  maximum |dz| against its ``nonlinear_opt`` grid (bit identity of the instrumented loop).

``--metres`` loads the tile with ``reproduce_v1_units: false`` (the config is not changed;
the metre copy exists only in memory). ``--scale-K`` and ``--scale-D`` multiply K0 and D0,
so that the metre run can be made equivalent to the feet run in its linear part
(K and D both x 3.2808, the Phase 5.4 invariance) or run with the factor addendum 3
section 3.3 states (D / 3.2808); both are reported.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from lemcvt.analyses.outputs import write_model_outputs
from lemcvt.analyses.tile_pipeline import V1_PREFIX, mse
from lemcvt.config import config_snapshot, load_tile, load_tile_config
from lemcvt.io.grids import read_geotiff
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.models.partitioned_nonlinear import partitioned_threshold_model_nonlinear
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)


def slope_stats(slopes: np.ndarray, S_c: float) -> dict[str, float]:
    s = np.asarray(slopes)
    s = s[np.isfinite(s)]
    r = s / S_c
    return {"n": int(s.size), "frac_gt_0.5Sc": float(np.mean(r > 0.5)), "frac_gt_0.9Sc": float(np.mean(r > 0.9)),
            "frac_gt_0.99Sc": float(np.mean(r > 0.99)), "frac_gt_Sc": float(np.mean(r > 1.0)),
            "max_ratio": float(np.max(r)) if r.size else float("nan"), "median_ratio": float(np.median(r)) if r.size else float("nan"),
            "p90_ratio": float(np.percentile(r, 90)) if r.size else float("nan")}


def run_picard_audit(tile: str, metres: bool = False, scale_K: float = 1.0, scale_D: float = 1.0,
                     label: str | None = None, reference_run: str | Path | None = None, save_grids: bool = True,
                     run_id: str | None = None, results_root=None, max_iter: int | None = None) -> RunRecorder:
    # metres=True: the loader converts z and scales K0, D0 by c (Phase 5); metres=False: the config's own flag.
    # Before Phase 5 the AP configs reproduced v1 (feet), so `--metres` alone was the K, D x 1 variant and the
    # unit-consistent one needed --scale-k/--scale-d 3.2808; since Phase 5 `--metres` is unit-consistent by itself.
    cfg = load_tile_config(tile, reproduce_v1_units=False if metres else None)
    td = load_tile(cfg)
    mp = td.cfg.model
    ref = td.cfg.v1_reference.get("stages", {}).get("nonlinear")
    if not ref:
        raise ValueError(f"{tile}: no v1 nonlinear reference in the tile config")
    K = mp.K0_partitioned * scale_K
    D = mp.D0 * scale_D
    A_crit = float(ref["A_crit"])
    S_c = mp.S_c
    n_it = int(max_iter or mp.nonlinear_max_iter)
    lab = label or ("metres" if metres else "feet_as_v1")
    rec = RunRecorder("picard_audit", f"{tile}_{lab}", run_id=run_id, root=results_root)
    rec.add_params("tile", tile)
    rec.add_params("variant", {"label": lab, "metres": metres, "scale_K": scale_K, "scale_D": scale_D,
                               "z_units_working": td.z_units_working, "z_factor": td.z_factor})
    rec.add_params("inputs", td.provenance())
    rec.add_params("params", {"U": mp.U, "K": K, "D": D, "S_c": S_c, "A_crit": A_crit, "m": mp.m, "max_iter": n_it,
                              "tol": mp.nonlinear_tol, "K0_partitioned": mp.K0_partitioned, "D0": mp.D0})
    Z0, A, FD, dx = td.Z, td.A, td.FD, td.dx

    t0 = time.perf_counter()
    f = partitioned_threshold_model_nonlinear(Z0, A, FD, dx, mp.m, mp.U, K)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    Z, mask = f(A_crit, D, S_c, n_it, mp.nonlinear_tol, diagnostics=True)
    t_solve = time.perf_counter() - t0
    pic = f.last_picard
    hs = f.last_hillslope_indices
    gd = f.grid_data
    Z0_flat = Z0[gd["rev_j"], gd["rev_i"]]
    out: dict[str, Any] = {
        "wall_build_s": t_build, "wall_solve_s": t_solve, "mse": mse(Z, Z0), "n_channel": int(mask.sum()),
        "n_hillslope": int(hs.size), "picard": pic,
        "slopes_nonlinear_final": slope_stats(f.last_slopes, S_c),
        "slopes_dem_same_cells": slope_stats(f.compute_slopes(Z0_flat, hs), S_c),
    }
    # the linear solution the iteration starts from (iteration 0 of the Picard loop), same cells
    lin = partitioned_threshold_model(Z0, A, FD, dx, mp.m, mp.U, K)
    Z_lin, _ = lin(A_crit, D)
    out["slopes_linear_start"] = slope_stats(f.compute_slopes(Z_lin[gd["rev_j"], gd["rev_i"]], hs), S_c)
    out["mse_linear"] = mse(Z_lin, Z0)
    if reference_run:
        prefix = f"{td.cfg.site}_1m_best_tile_{V1_PREFIX['nonlinear_opt']}"
        p = Path(reference_run) / f"{prefix}_elevation.tif"
        if p.exists():
            Zr = read_geotiff(p).data
            relief = float(np.nanmax(Z0) - np.nanmin(Z0))
            d = np.nanmax(np.abs(Z - Zr))
            out["reference_comparison"] = {"reference_grid": str(p), "max_abs_dz": float(d), "max_abs_dz_over_relief": float(d / relief),
                                           "mse_reference": mse(Zr, Z0), "bit_identical": bool(d == 0.0)}
        else:
            out["reference_comparison"] = {"reference_grid": str(p), "status": "missing"}
    if save_grids:
        out["files"] = write_model_outputs(rec, td.filled, Z, f"{td.cfg.site}_1m_best_tile_{V1_PREFIX['nonlinear_opt']}_{lab}",
                                           derived=False)
    rec.add_params("result", out)
    logger.info("%s %s: %d iterations (%s), final update %.3e, residual %.3e, MSE %.6e, hillslope |grad z| > 0.5 S_c: %.3f, "
                "> 0.9 S_c: %.3f, max ratio %.3f (%.0f s)", tile, lab, pic["n_iter"], pic["stop_reason"],
                pic["iterations"][-1]["update_max_abs"] if pic["iterations"] else float("nan"),
                pic["iterations"][-1].get("residual_rel_l2", float("nan")) if pic["iterations"] else float("nan"),
                out["mse"], out["slopes_nonlinear_final"]["frac_gt_0.5Sc"], out["slopes_nonlinear_final"]["frac_gt_0.9Sc"],
                out["slopes_nonlinear_final"]["max_ratio"], t_solve)
    rec.finalize(config_snapshot(td.cfg, {"metres": metres, "scale_K": scale_K, "scale_D": scale_D,
                                          "reference_run": str(reference_run) if reference_run else None}))
    return rec
