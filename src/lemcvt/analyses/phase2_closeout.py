"""Phase 2 close-out diagnostics (addendum 2, sections 2.1-2.3).

``lemcvt closeout linearized --tile GM5`` tests the Phase 2 explanation that the
AD-opt and joint forward runs differ from the saved v1 grids only because the v1
notebooks printed the optimum to five significant figures:

1. **Linearized recovery.** For the AD-opt and joint runs, J = dZ/d(log10 K, log10 D)
   is computed by central differences (relative parameter step 1e-6, channel mask
   fixed at the printed A_crit) and ``min || Z_saved - Z_v2 - J delta ||_2`` is solved
   for delta. The run passes when every |delta_i| lies inside the rounding box of the
   printed value (half a unit in the 5th significant figure, in log10 space) and the
   residual's max |.| is <= 1e-8 x relief.
2. **MSE stationarity.** The gradient g = dMSE/d(log10 K, log10 D) at the printed
   parameters is reported with the first-order prediction g . delta of the MSE
   difference, so a non-zero gradient (an optimizer that stopped short of the
   stationary point) is distinguished from a mask change.
3. **Rounding-interval counts.** For every A_crit-bearing run, the number of cells whose
   D8 area lies inside the rounding interval of the printed A_crit (the only cells whose
   channel membership the rounding could change).
4. **Difference counts** replacing ``n_diff_cells``: cells with |dz| > 1e-9 x relief,
   cells with |dz| > 1e-6 x relief, and the rounding-interval count above as the count of
   cells whose channel mask may differ (v1 saved no mask).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from lemcvt.analyses.forward import v1_forward_params
from lemcvt.analyses.tile_pipeline import V1_PREFIX, mse
from lemcvt.config import TileData, config_snapshot, data_root, load_tile
from lemcvt.io.grids import read_geotiff
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.results import RunRecorder, latest_run

logger = logging.getLogger(__name__)

REL_STEP = 1e-6
TOL_RESIDUAL_REL_RELIEF = 1e-8
SIG_FIGS_PRINTED = 5


def half_unit(value: float, sig_figs: int = SIG_FIGS_PRINTED) -> float:
    """Half a unit in the last printed significant figure of ``value``."""
    e = math.floor(math.log10(abs(value)))
    return 0.5 * 10 ** (e - (sig_figs - 1))


def rounding_interval(value: float, sig_figs: int = SIG_FIGS_PRINTED) -> tuple[float, float]:
    """``[lo, hi)``: every number that prints as ``value`` at ``sig_figs`` significant figures."""
    h = half_unit(value, sig_figs)
    return value - h, value + h


def log10_box(value: float, sig_figs: int = SIG_FIGS_PRINTED) -> tuple[float, float]:
    """The rounding interval as offsets in log10 from the printed value (lo < 0 < hi)."""
    lo, hi = rounding_interval(value, sig_figs)
    lv = math.log10(value)
    return math.log10(lo) - lv, math.log10(hi) - lv


def saved_v1_grid(td: TileData, key: str) -> np.ndarray | None:
    saved = td.cfg.v1_reference.get("saved_grids", {})
    if key not in saved:
        return None
    ref = read_geotiff(data_root() / saved[key]).data
    if td.z_factor != 1.0:
        ref = ref * td.z_factor
    return ref


def difference_counts(Z: np.ndarray, ref: np.ndarray, relief: float, A: np.ndarray | None = None,
                      A_crit_printed: float | None = None) -> dict[str, Any]:
    """Counts that replace ``n_diff_cells`` (addendum 2, 2.3)."""
    diff = np.abs(Z - ref)
    out: dict[str, Any] = {
        "n_dz_gt_1e-9_relief": int(np.nansum(diff > 1e-9 * relief)),
        "n_dz_gt_1e-6_relief": int(np.nansum(diff > 1e-6 * relief)),
        "n_valid": int(np.sum(np.isfinite(diff))),
    }
    if A is not None and A_crit_printed is not None:
        lo, hi = rounding_interval(A_crit_printed)
        out["n_mask_may_differ"] = int(np.sum((A >= lo) & (A < hi)))
        out["A_crit_rounding_interval"] = [lo, hi]
    return out


def rounding_interval_count(A: np.ndarray, A_crit_printed: float) -> dict[str, Any]:
    """Cells whose D8 area lies inside the rounding interval of the printed A_crit."""
    lo, hi = rounding_interval(A_crit_printed)
    finite = A[np.isfinite(A)]
    inside = (finite >= lo) & (finite < hi)
    below = finite[finite < lo]
    above = finite[finite >= hi]
    return {
        "A_crit_printed": A_crit_printed,
        "interval": [lo, hi],
        "n_cells_in_interval": int(inside.sum()),
        "n_channel_at_printed": int(np.sum(finite >= A_crit_printed)),
        "nearest_area_below": float(below.max()) if below.size else None,
        "nearest_area_above": float(above.min()) if above.size else None,
        "cell_area": float(finite.min()),
    }


def _evaluate(td: TileData, spec: dict[str, Any], K: float, D: float, cache: dict[Any, Any]) -> np.ndarray:
    """Z at (K, D) for the model of ``spec`` with the channel mask fixed by ``spec['A_crit']``."""
    mp = td.cfg.model
    if spec["model"] == "AD":
        f = cache.get("AD")
        if f is None:
            f = cache["AD"] = advection_diffusion_model(td.Z, td.A, td.FD, td.dx, mp.m)
        return f(spec["U"], K, D)
    if spec["model"] == "partitioned":
        key = ("partitioned", float(K))
        f = cache.get(key)
        if f is None:
            f = cache[key] = partitioned_threshold_model(td.Z, td.A, td.FD, td.dx, mp.m, spec["U"], K)
        Z, _mask = f(spec["A_crit"], D)
        return Z
    raise ValueError(f"linearized recovery is defined for AD and partitioned runs, not {spec['model']!r}")


def linearized_recovery(td: TileData, key: str, spec: dict[str, Any], ref: np.ndarray,
                        rel_step: float = REL_STEP) -> dict[str, Any]:
    """Recover the rounding offsets (log10 K, log10 D) that explain ``ref - Z_v2`` to first order."""
    cache: dict[Any, Any] = {}
    K, D = float(spec["K"]), float(spec["D"])
    Z_base = _evaluate(td, spec, K, D, cache)
    Z_obs = td.Z
    relief = float(np.nanmax(Z_obs) - np.nanmin(Z_obs))

    dlog = math.log10(1 + rel_step) - math.log10(1 - rel_step)
    J_cols = []
    for name in ("K", "D"):
        if name == "K":
            Zp = _evaluate(td, spec, K * (1 + rel_step), D, cache)
            Zm = _evaluate(td, spec, K * (1 - rel_step), D, cache)
        else:
            Zp = _evaluate(td, spec, K, D * (1 + rel_step), cache)
            Zm = _evaluate(td, spec, K, D * (1 - rel_step), cache)
        J_cols.append((Zp - Zm) / dlog)
    J = np.stack([c.ravel() for c in J_cols], axis=1)                    # (n_cells, 2), per log10 unit
    r = (ref - Z_base).ravel()
    valid = np.isfinite(r) & np.all(np.isfinite(J), axis=1)
    delta, *_ = np.linalg.lstsq(J[valid], r[valid], rcond=None)
    resid = r[valid] - J[valid] @ delta

    boxes = {"K": log10_box(K), "D": log10_box(D)}
    in_box = {n: bool(boxes[n][0] <= d <= boxes[n][1]) for n, d in zip(("K", "D"), delta, strict=True)}
    max_resid = float(np.max(np.abs(resid)))
    pass_resid = max_resid <= TOL_RESIDUAL_REL_RELIEF * relief

    # MSE stationarity at the printed parameters
    e = (Z_base - Z_obs).ravel()
    ok = np.isfinite(e) & valid
    n_mse = int(np.sum(np.isfinite(e)))
    grad = 2.0 * np.array([np.sum(e[ok] * J[ok, i]) for i in range(2)]) / n_mse
    Jd = J[valid] @ delta
    pred_first = float(grad @ delta)
    pred_second = float(np.sum(Jd * Jd) / n_mse)
    mse_base, mse_ref = mse(Z_base, Z_obs), mse(ref, Z_obs)

    # the same recovery with a +A_crit column is not possible (staircase); report the mask size instead
    out = {
        "key": key,
        "model": spec["model"],
        "params_printed": {k: spec[k] for k in ("K", "D", *(["A_crit"] if "A_crit" in spec else []))},
        "rel_step": rel_step,
        "n_cells_fit": int(valid.sum()),
        "relief": relief,
        "delta_log10": {"K": float(delta[0]), "D": float(delta[1])},
        "delta_relative": {"K": float(10 ** delta[0] - 1), "D": float(10 ** delta[1] - 1)},
        "recovered_params": {"K": K * 10 ** delta[0], "D": D * 10 ** delta[1]},
        "rounding_box_log10": boxes,
        "in_box": in_box,
        "max_abs_diff_before": float(np.max(np.abs(r[valid]))),
        "max_abs_diff_before_over_relief": float(np.max(np.abs(r[valid])) / relief),
        "rms_diff_before": float(np.sqrt(np.mean(r[valid] ** 2))),
        "max_abs_residual": max_resid,
        "max_abs_residual_over_relief": max_resid / relief,
        "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
        "pass_box": all(in_box.values()),
        "pass_residual_1e-8_relief": bool(pass_resid),
        "pass": bool(all(in_box.values()) and pass_resid),
        "mse_v2_printed_params": mse_base,
        "mse_saved_grid": mse_ref,
        "dmse_measured": mse_ref - mse_base,
        "dmse_measured_relative": (mse_ref - mse_base) / mse_ref,
        "mse_gradient_log10": {"K": float(grad[0]), "D": float(grad[1])},
        "dmse_predicted_first_order": pred_first,
        "dmse_predicted_second_order": pred_second,
        "dmse_predicted_total": pred_first + pred_second,
        "J_column_norms_rms": [float(np.sqrt(np.mean(J[valid, i] ** 2))) for i in range(2)],
    }
    logger.info("%s %s: delta_log10 K=%.3e (box %s) D=%.3e (box %s); max|resid|/relief=%.2e; dMSE meas %.3e pred %.3e",
                td.cfg.tile, key, delta[0], boxes["K"], delta[1], boxes["D"], max_resid / relief,
                out["dmse_measured"], out["dmse_predicted_total"])
    return out


def forward_run_grid(run_dir: Path, td: TileData, key: str) -> np.ndarray | None:
    """The elevation grid a Phase 2 ``tile_forward`` run wrote for ``key``."""
    prefix = f"{td.cfg.site}_1m_best_tile_{V1_PREFIX.get(key, key)}"
    p = run_dir / f"{prefix}_elevation.tif"
    if not p.exists():
        return None
    return read_geotiff(p).data


def run_closeout(tile: str, keys: tuple[str, ...] = ("ad_opt", "joint_opt"), rel_step: float = REL_STEP,
                 run_id: str | None = None, results_root=None, forward_run: Path | None = None) -> RunRecorder:
    """Linearized recovery, rounding-interval counts and difference counts for one tile."""
    td = load_tile(tile)
    specs = v1_forward_params(td)
    rec = RunRecorder("closeout", tile, run_id=run_id, root=results_root)
    rec.add_params("tile", tile)
    rec.add_params("inputs", td.provenance())
    relief = float(np.nanmax(td.Z) - np.nanmin(td.Z))

    # 2.2.1 rounding-interval counts for every A_crit-bearing run
    counts = {k: rounding_interval_count(td.A, float(s["A_crit"])) for k, s in specs.items() if "A_crit" in s}
    rec.add_params("rounding_interval_counts", counts)
    for k, c in counts.items():
        logger.info("%s %s: %d cells with area in %s (n_channel=%d)", tile, k, c["n_cells_in_interval"], c["interval"],
                    c["n_channel_at_printed"])

    # 2.3 difference counts from the Phase 2 forward run grids (no recomputation)
    fwd = forward_run or latest_run("tile_forward", tile)
    diff_counts: dict[str, Any] = {}
    if fwd is not None:
        for k, s in specs.items():
            ref = saved_v1_grid(td, k)
            Z = forward_run_grid(Path(fwd), td, k)
            if ref is None or Z is None:
                continue
            diff_counts[k] = difference_counts(Z, ref, relief, td.A, s.get("A_crit"))
            diff_counts[k]["forward_run"] = Path(fwd).name
    rec.add_params("difference_counts", diff_counts)

    # 2.1 linearized recovery
    recovery: dict[str, Any] = {}
    for k in keys:
        if k not in specs:
            continue
        ref = saved_v1_grid(td, k)
        if ref is None:
            recovery[k] = {"status": "no saved grid"}
            continue
        recovery[k] = linearized_recovery(td, k, specs[k], ref, rel_step=rel_step)
    rec.add_params("linearized_recovery", recovery)
    rec.add_params("tolerances", {"residual_rel_relief": TOL_RESIDUAL_REL_RELIEF, "sig_figs_printed": SIG_FIGS_PRINTED})
    rec.finalize(config_snapshot(td.cfg, {"keys": list(keys), "rel_step": rel_step}))
    return rec
