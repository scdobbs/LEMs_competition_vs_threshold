"""The per-tile calibration pipeline: the four v1 stages, in v1 order, with v1 semantics.

Stage semantics (from ``notebooks/model_runs/*_complete.ipynb``, see
``reports/v2/00_reference.md`` section 3):

``ad``         forward AD run at (K0_ad, D0) saved as ``*_AD-no-opt``; L-BFGS-B over
               (log10 K, log10 D); forward at the optimum saved as ``*_AD-opt``.
``acrit``      partitioned model built with K0_partitioned; Nelder-Mead over
               log10 A_crit with D = D0; forward at the optimum saved as
               ``*_Partitioned-model-a_crit-opt``.
``nonlinear``  nonlinear partitioned model (K0_partitioned, S_c); Nelder-Mead over
               log10 A_crit with D = D0; forward at the optimum (v1 plotted but did
               not save it; saved here as ``*_Partitioned-model-nonlinear-a_crit-opt``).
``joint``      Nelder-Mead over (log10 A_crit, log10 D, log10 K) starting from
               (A_crit_0, D0, K0_partitioned), rebuilding the channel network per K;
               forward at the optimum saved as ``*_Partitioned-model-a_crit-k-d-opt``.

The misfit everywhere is ``nanmean((Z_model - Z_obs)**2)`` over all cells.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from lemcvt.analyses.outputs import write_model_outputs
from lemcvt.config import TileData, config_snapshot, load_tile
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.models.partitioned_nonlinear import partitioned_threshold_model_nonlinear
from lemcvt.optimize import v1 as opt_v1
from lemcvt.results import RunRecorder
from lemcvt.utils import compute_slope_magnitude

logger = logging.getLogger(__name__)

STAGE_ORDER = ("ad", "acrit", "nonlinear", "joint")
V1_PREFIX = {
    "ad_no_opt": "AD-no-opt",
    "ad_opt": "AD-opt",
    "acrit_opt": "Partitioned-model-a_crit-opt",
    "nonlinear_opt": "Partitioned-model-nonlinear-a_crit-opt",
    "joint_opt": "Partitioned-model-a_crit-k-d-opt",
}
EXPLORATION_ACRIT_LINEAR = (5e3, 1e3, 5e2, 1e2)
EXPLORATION_ACRIT_NONLINEAR = (1e3, 1e4)


def mse(Z_model: np.ndarray, Z_obs: np.ndarray) -> float:
    return float(np.nanmean((Z_model - Z_obs) ** 2))


def _slope_histogram(Z_model: np.ndarray, hillslope_mask: np.ndarray) -> dict[str, Any]:
    """v1 cells 17-18: histogram of hillslope slope magnitudes (unit pixel spacing)."""
    s = compute_slope_magnitude(Z_model)
    counts, edges = np.histogram(s[hillslope_mask], bins=np.arange(0, 2, 0.1))
    return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def _prefix(td: TileData, key: str) -> str:
    return f"{td.cfg.site}_1m_best_tile_{V1_PREFIX[key]}"


def run_tile(
    tile: str,
    stages: Sequence[str] = STAGE_ORDER,
    optimizer: str = "v1",
    save_grids: bool = True,
    derived_grids: bool = True,
    exploration_runs: bool = False,
    crop: tuple[int, int, int, int] | None = None,
    run_id: str | None = None,
    results_root=None,
) -> RunRecorder:
    """Run the requested stages on one tile and write a results directory.

    Parameters
    ----------
    tile
        Tile id, e.g. ``"GM1"``.
    stages
        Subset of ``("ad", "acrit", "nonlinear", "joint")``; run in v1 order.
    optimizer
        ``"v1"`` (exact port). ``"v2"`` arrives in Phase 4.
    save_grids
        Write the forward-run grids (elevation plus derived set) as GeoTIFF.
    derived_grids
        Also write filled/flow-direction/area/log-area/hillshade like v1.
    exploration_runs
        Also perform v1's plotted-only forward runs (A_crit = 5e3, 1e3, 5e2, 1e2 and
        1e3, 1e4 nonlinear) and record their MSEs.
    crop
        Pixel crop for smoke tests.
    """
    if optimizer != "v1":
        raise NotImplementedError("only --optimizer v1 exists in Phase 1; v2 is Phase 4")
    bad = [s for s in stages if s not in STAGE_ORDER]
    if bad:
        raise ValueError(f"unknown stages {bad}; choose from {STAGE_ORDER}")
    stages = tuple(s for s in STAGE_ORDER if s in stages)

    td = load_tile(tile, crop=crop, sanity=crop is None)
    mp = td.cfg.model
    Z0, A, FD, dx, U = td.Z, td.A, td.FD, td.dx, mp.U
    if run_id is None and len(stages) < len(STAGE_ORDER):
        from lemcvt.results import new_run_id

        run_id = new_run_id("tile_run", f"{tile}_{'-'.join(stages)}")   # one stage per job: name it
    rec = RunRecorder("tile_run", tile, run_id=run_id, root=results_root)
    rec.add_params("tile", tile)
    rec.add_params("optimizer", optimizer)
    rec.add_params("stages", list(stages))
    rec.add_params("inputs", td.provenance())
    rec.add_params("model", {
        "m": mp.m, "U": U, "n": mp.n, "S_c": mp.S_c, "K0_ad": mp.K0_ad, "K0_partitioned": mp.K0_partitioned,
        "D0": mp.D0, "K_bounds": mp.K_bounds, "D_bounds": mp.D_bounds, "A_crit_0": mp.A_crit_0,
        "A_crit_bounds": mp.A_crit_bounds,
    })
    stage_results: dict[str, Any] = {}
    forward_runs: list[dict[str, Any]] = []
    template = td.filled

    def _save(Z, key):
        if save_grids:
            return write_model_outputs(rec, template, Z, _prefix(td, key), derived=derived_grids)
        return []

    # ---------------------------------------------------------------- AD stage
    if "ad" in stages:
        t0 = time.perf_counter()
        ad = advection_diffusion_model(Z0, A, FD, dx, mp.m)
        t_build = time.perf_counter() - t0
        Z = ad(U, mp.K0_ad, mp.D0)
        forward_runs.append({"key": "ad_no_opt", "model": "AD", "params": {"U": U, "K": mp.K0_ad, "D": mp.D0},
                             "mse": mse(Z, Z0), "files": _save(Z, "ad_no_opt")})
        res = opt_v1.optimize_advection_diffusion(ad, Z0, U, mp.K0_ad, mp.D0, mp.K_bounds, mp.D_bounds,
                                                  method=mp.methods["ad"], return_result=True)
        rec.add_trajectory(res.trajectory)
        Z = ad(U, res.params["K"], res.params["D"])
        forward_runs.append({"key": "ad_opt", "model": "AD", "params": {"U": U, **res.params},
                             "mse": mse(Z, Z0), "files": _save(Z, "ad_opt")})
        stage_results["ad"] = _stage_record(res, {"build_s": t_build, "K0": mp.K0_ad, "D0": mp.D0})
        logger.info("AD: K=%.4e D=%.4e MSE=%.4e (%.1f s)", res.params["K"], res.params["D"], res.mse, res.wall_s)

    # ----------------------------------------------------- partitioned (linear)
    need_pm = {"acrit"} & set(stages)
    if need_pm:
        t0 = time.perf_counter()
        pm = partitioned_threshold_model(Z0, A, FD, dx, mp.m, U, mp.K0_partitioned)
        t_build = time.perf_counter() - t0
        if exploration_runs:
            for a_c in EXPLORATION_ACRIT_LINEAR:
                Z, _ = pm(a_c, mp.D0)
                forward_runs.append({"key": f"acrit_explore_{a_c:g}", "model": "partitioned",
                                     "params": {"U": U, "K": mp.K0_partitioned, "D": mp.D0, "A_crit": a_c},
                                     "mse": mse(Z, Z0), "files": []})
        res = opt_v1.optimize_threshold_acrit(pm, Z0, mp.D0, mp.A_crit_0, mp.A_crit_bounds,
                                              method=mp.methods["acrit"], return_result=True)
        rec.add_trajectory(res.trajectory)
        Z, channel_mask = pm(res.params["A_crit"], mp.D0)
        forward_runs.append({"key": "acrit_opt", "model": "partitioned",
                             "params": {"U": U, "K": mp.K0_partitioned, "D": mp.D0, "A_crit": res.params["A_crit"]},
                             "mse": mse(Z, Z0), "files": _save(Z, "acrit_opt"),
                             "n_channel": int(channel_mask.sum())})
        stage_results["acrit"] = _stage_record(res, {"build_s": t_build, "K": mp.K0_partitioned, "D": mp.D0,
                                                     "hillslope_slope_histogram": _slope_histogram(Z, ~channel_mask)})
        logger.info("A_crit: %.4e MSE=%.4e (%.1f s)", res.params["A_crit"], res.mse, res.wall_s)

    # -------------------------------------------------------------- nonlinear
    if "nonlinear" in stages:
        t0 = time.perf_counter()
        pmnl = partitioned_threshold_model_nonlinear(Z0, A, FD, dx, mp.m, U, mp.K0_partitioned)
        t_build = time.perf_counter() - t0
        if exploration_runs:
            for a_c in EXPLORATION_ACRIT_NONLINEAR:
                Z, _ = pmnl(a_c, mp.D0, mp.S_c, mp.nonlinear_max_iter, mp.nonlinear_tol)
                forward_runs.append({"key": f"nonlinear_explore_{a_c:g}", "model": "partitioned_nonlinear",
                                     "params": {"U": U, "K": mp.K0_partitioned, "D": mp.D0, "S_c": mp.S_c, "A_crit": a_c},
                                     "mse": mse(Z, Z0), "files": []})

        def pmnl_v1(A_crit, D, Sc, _f=pmnl, _mi=mp.nonlinear_max_iter, _tol=mp.nonlinear_tol):
            return _f(A_crit, D, Sc, _mi, _tol)

        res = opt_v1.optimize_threshold_acrit_nonlinear(pmnl_v1, Z0, mp.D0, mp.S_c, mp.A_crit_0, mp.A_crit_bounds,
                                                        method=mp.methods["nonlinear"], return_result=True)
        rec.add_trajectory(res.trajectory)
        Z, channel_mask = pmnl(res.params["A_crit"], mp.D0, mp.S_c, mp.nonlinear_max_iter, mp.nonlinear_tol)
        forward_runs.append({"key": "nonlinear_opt", "model": "partitioned_nonlinear",
                             "params": {"U": U, "K": mp.K0_partitioned, "D": mp.D0, "S_c": mp.S_c,
                                        "A_crit": res.params["A_crit"]},
                             "mse": mse(Z, Z0), "files": _save(Z, "nonlinear_opt"),
                             "picard_iterations": int(getattr(pmnl, "last_n_iter", 0)),
                             "picard": getattr(pmnl, "last_picard", None)})
        stage_results["nonlinear"] = _stage_record(res, {"build_s": t_build, "K": mp.K0_partitioned, "D": mp.D0,
                                                         "S_c": mp.S_c})
        logger.info("nonlinear A_crit: %.4e MSE=%.4e (%.1f s)", res.params["A_crit"], res.mse, res.wall_s)

    # ------------------------------------------------------------------ joint
    if "joint" in stages:
        res = opt_v1.optimize_threshold_full(Z0, A, FD, dx, mp.m, U, Z0, mp.K0_partitioned, mp.D0, mp.A_crit_0,
                                             mp.K_bounds, mp.D_bounds, mp.A_crit_bounds,
                                             method=mp.methods["joint"], return_result=True)
        rec.add_trajectory(res.trajectory)
        pm_all = partitioned_threshold_model(Z0, A, FD, dx, mp.m, U, res.params["K"])
        Z, channel_mask = pm_all(res.params["A_crit"], res.params["D"])
        forward_runs.append({"key": "joint_opt", "model": "partitioned", "params": {"U": U, **res.params},
                             "mse": mse(Z, Z0), "files": _save(Z, "joint_opt"), "n_channel": int(channel_mask.sum())})
        stage_results["joint"] = _stage_record(res, {"K0": mp.K0_partitioned, "D0": mp.D0, "A_crit_0": mp.A_crit_0})
        logger.info("joint: A_crit=%.4e D=%.4e K=%.4e MSE=%.4e (%.1f s)", res.params["A_crit"], res.params["D"],
                    res.params["K"], res.mse, res.wall_s)

    rec.add_params("stages_results", stage_results)
    rec.add_params("forward_runs", forward_runs)
    rec.add_params("v1_reference", td.cfg.v1_reference.get("stages", {}))
    rec.finalize(config_snapshot(td.cfg, {"crop": crop, "exploration_runs": exploration_runs}))
    return rec


def _stage_record(res: opt_v1.OptResult, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": res.params,
        "mse": res.mse,
        "n_iter": res.n_iter,
        "n_fev": res.n_fev,
        "n_solves": res.n_solves,
        "wall_s": res.wall_s,
        "success": res.success,
        "message": res.message,
        "x_log10": res.scipy_x,
        **extra,
    }
