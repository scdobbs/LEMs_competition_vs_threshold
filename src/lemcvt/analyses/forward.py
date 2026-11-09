"""Forward runs at given parameters, and the regression comparison against saved grids.

``lemcvt tile forward --tile GM1 --params <params.json|v1>`` evaluates the AD and
partitioned models at the supplied parameters and writes the elevation grids.
With ``--compare`` it also reads the saved v1 output grids (``v1_reference.saved_grids``
in the tile config) and reports ``max|dz|`` and MSE differences against the
Phase 2 tolerances (``max|dz| <= 1e-6 * relief``, ``|dMSE/MSE| <= 1e-8``).

Note on attainable agreement: the v1 notebooks print optimized parameters with
five significant figures, so forward runs at the *printed* optimum cannot match
the saved grids better than the parameter rounding allows (about 1e-5 relative
in MSE). The ``ad_no_opt`` run uses exactly reproducible parameters (K0, D0) and
is the bit-level target; the staircase A_crit stages are exact whenever the
rounded A_crit selects the same channel mask.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lemcvt.analyses.outputs import write_model_outputs
from lemcvt.analyses.tile_pipeline import V1_PREFIX, mse
from lemcvt.config import TileData, config_snapshot, data_root, load_tile
from lemcvt.io.grids import read_geotiff
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.models.partitioned_nonlinear import partitioned_threshold_model_nonlinear
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)

TOL_DZ_REL_RELIEF = 1e-6
TOL_MSE_REL = 1e-8


def v1_forward_params(td: TileData) -> dict[str, dict[str, Any]]:
    """Parameter sets for the four saved v1 runs, from the tile config (printed v1 optima)."""
    mp = td.cfg.model
    ref = td.cfg.v1_reference.get("stages", {})
    out: dict[str, dict[str, Any]] = {
        "ad_no_opt": {"model": "AD", "U": mp.U, "K": mp.K0_ad, "D": mp.D0},
    }
    if "ad" in ref:
        out["ad_opt"] = {"model": "AD", "U": mp.U, "K": ref["ad"]["K"], "D": ref["ad"]["D"]}
    if "acrit" in ref:
        out["acrit_opt"] = {"model": "partitioned", "U": mp.U, "K": mp.K0_partitioned, "D": mp.D0,
                            "A_crit": ref["acrit"]["A_crit"]}
    if "nonlinear" in ref:
        out["nonlinear_opt"] = {"model": "partitioned_nonlinear", "U": mp.U, "K": mp.K0_partitioned, "D": mp.D0,
                                "S_c": mp.S_c, "A_crit": ref["nonlinear"]["A_crit"]}
    if "joint" in ref:
        out["joint_opt"] = {"model": "partitioned", "U": mp.U, "K": ref["joint"]["K"], "D": ref["joint"]["D"],
                            "A_crit": ref["joint"]["A_crit"]}
    return out


def params_from_run(params_json: Path) -> dict[str, dict[str, Any]]:
    """Parameter sets from a ``tile_run`` params.json (its ``forward_runs`` entries)."""
    p = json.loads(Path(params_json).read_text())
    out: dict[str, dict[str, Any]] = {}
    for fr in p.get("forward_runs", []):
        if fr["key"].endswith("_opt") or fr["key"] == "ad_no_opt":
            out[fr["key"]] = {"model": fr["model"], **fr["params"]}
    return out


def evaluate(td: TileData, spec: dict[str, Any], models_cache: dict[Any, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate one forward run; model factories are cached per (model, K)."""
    mp = td.cfg.model
    Z0, A, FD, dx = td.Z, td.A, td.FD, td.dx
    model = spec["model"]
    info: dict[str, Any] = {}
    if model == "AD":
        f = models_cache.setdefault("AD", advection_diffusion_model(Z0, A, FD, dx, mp.m))
        Z = f(spec["U"], spec["K"], spec["D"])
    elif model == "partitioned":
        key = ("partitioned", float(spec["K"]))
        f = models_cache.get(key)
        if f is None:
            f = models_cache[key] = partitioned_threshold_model(Z0, A, FD, dx, mp.m, spec["U"], spec["K"])
        Z, mask = f(spec["A_crit"], spec["D"])
        info["n_channel"] = int(mask.sum())
    elif model == "partitioned_nonlinear":
        key = ("nonlinear", float(spec["K"]))
        f = models_cache.get(key)
        if f is None:
            f = models_cache[key] = partitioned_threshold_model_nonlinear(Z0, A, FD, dx, mp.m, spec["U"], spec["K"])
        Z, mask = f(spec["A_crit"], spec["D"], spec["S_c"], mp.nonlinear_max_iter, mp.nonlinear_tol)
        info["n_channel"] = int(mask.sum())
        info["picard_iterations"] = int(getattr(f, "last_n_iter", 0))
    else:
        raise ValueError(f"unknown model {model!r}")
    return Z, info


def compare_with_saved(td: TileData, key: str, Z: np.ndarray, spec: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Compare a forward grid with the saved v1 grid of the same run key, if it exists.

    Besides max|dz| and the MSE difference, the comparison counts cells with
    |dz| > 1e-9 x relief and |dz| > 1e-6 x relief, and, for A_crit-bearing runs, the
    cells whose D8 area lies inside the rounding interval of the printed A_crit (the
    only cells whose channel membership can differ between the saved grid and this run;
    v1 saved no mask). These replace the former ``n_diff_cells`` (addendum 2, 2.3).
    """
    from lemcvt.analyses.phase2_closeout import difference_counts

    saved = td.cfg.v1_reference.get("saved_grids", {})
    if key not in saved:
        return None
    path = data_root() / saved[key]
    if not path.exists():
        return {"saved_grid": str(path), "status": "missing"}
    ref = read_geotiff(path).data
    if td.z_factor != 1.0:
        ref = ref * td.z_factor  # saved grids are in the stored unit
    Z0 = td.Z
    relief = float(np.nanmax(Z0) - np.nanmin(Z0))
    diff = Z - ref
    max_abs = float(np.nanmax(np.abs(diff)))
    mse_new, mse_ref = mse(Z, Z0), mse(ref, Z0)
    rel_mse = abs(mse_new - mse_ref) / mse_ref
    a_crit = float(spec["A_crit"]) if spec and "A_crit" in spec else None
    return {
        "saved_grid": str(path),
        "relief": relief,
        "max_abs_dz": max_abs,
        "max_abs_dz_over_relief": max_abs / relief,
        "mse_new": mse_new,
        "mse_saved": mse_ref,
        "rel_mse_diff": rel_mse,
        **difference_counts(Z, ref, relief, td.A, a_crit),
        "pass_dz": bool(max_abs <= TOL_DZ_REL_RELIEF * relief),
        "pass_mse": bool(rel_mse <= TOL_MSE_REL),
    }


def run_forward(tile: str, params: str | Path = "v1", compare: bool = False, save_grids: bool = True,
                derived_grids: bool = False, keys: list[str] | None = None,
                crop: tuple[int, int, int, int] | None = None, run_id: str | None = None,
                results_root=None) -> RunRecorder:
    """Forward runs for one tile; ``params="v1"`` uses the tile config's v1 reference."""
    td = load_tile(tile, crop=crop, sanity=crop is None)
    specs = v1_forward_params(td) if str(params) == "v1" else params_from_run(Path(params))
    if keys:
        specs = {k: v for k, v in specs.items() if k in keys}
    rec = RunRecorder("tile_forward", tile, run_id=run_id, root=results_root)
    rec.add_params("tile", tile)
    rec.add_params("params_source", str(params))
    rec.add_params("inputs", td.provenance())
    cache: dict[Any, Any] = {}
    runs = []
    for key, spec in specs.items():
        Z, info = evaluate(td, spec, cache)
        entry = {"key": key, "params": spec, "mse": mse(Z, td.Z), **info}
        if save_grids:
            prefix = f"{td.cfg.site}_1m_best_tile_{V1_PREFIX.get(key, key)}"
            entry["files"] = write_model_outputs(rec, td.filled, Z, prefix, derived=derived_grids)
        if compare and crop is None:
            entry["comparison"] = compare_with_saved(td, key, Z, spec)
        runs.append(entry)
        logger.info("forward %s %s: MSE=%.6e %s", tile, key, entry["mse"],
                    ("cmp " + json.dumps(entry.get("comparison"), default=str)) if compare else "")
    rec.add_params("forward_runs", runs)
    rec.finalize(config_snapshot(td.cfg, {"compare": compare, "crop": crop}))
    return rec
