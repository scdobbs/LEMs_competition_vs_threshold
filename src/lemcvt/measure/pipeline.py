"""``lemcvt measure``: theta, k_s, C_ht and D0 = U/C_ht for each tile from its own grid.

Runs on metre grids (AP converted from feet regardless of ``reproduce_v1_units``
unless ``force_metres=False``), uses the tile's D8 area grid, no model output.

Per tile the run records:

* slope-area rollover curve and the primary threshold A_c* (``thresholds``);
* chi fits (theta misfit curves, theta*, k_s, bootstrap CIs) at A_c* and on the
  threshold ladder (``chi_fits``);
* binned slope-area regression at A_c* as a cross-check (``slope_area``);
* hilltop curvature scale curve, L*, C_ht, D0 with CIs (``curvature``);
* comparisons with the v1 hard-coded values (AP converted to metres), the regional
  values and the re-run v1 method (``comparisons``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.config import TileConfig, config_snapshot, load_tile, load_tile_config
from lemcvt.io.units import FT_US_IN_M
from lemcvt.measure.chi import channel_network
from lemcvt.measure.concavity import DEFAULT_THETAS, fit_chi, ks_at_theta
from lemcvt.measure.hilltop_curvature import measure_hilltop_curvature
from lemcvt.measure.legacy_v1 import legacy_chi, legacy_curvature
from lemcvt.measure.slope_area import slope_area_regression_binned, weighted_r2
from lemcvt.measure.thresholds import THRESHOLD_LADDER, slope_area_rollover, theta_plateau_threshold
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)


def _metre_tile(tile: str, crop=None):
    """Load a tile forcing z to metres (AP) even when the config reproduces v1 units."""
    cfg = load_tile_config(tile)
    if cfg.reproduce_v1_units:
        cfg = TileConfig(**{**cfg.__dict__, "reproduce_v1_units": False})
    return load_tile(cfg, crop=crop, sanity=crop is None)


def measure_tile(tile: str, force_metres: bool = True, n_boot: int = 200, thetas=DEFAULT_THETAS,
                 ladder=THRESHOLD_LADDER, run_legacy: bool = True, seed: int = 0, crop=None) -> dict[str, Any]:
    t0 = time.perf_counter()
    td = _metre_tile(tile, crop) if force_metres else load_tile(tile, crop=crop, sanity=crop is None)
    site = td.site
    Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    out: dict[str, Any] = {"tile": tile, "site": site.site, "z_units_working": td.z_units_working, "U": site.U,
                           "inputs": td.provenance()}

    # 1. threshold
    roll = slope_area_rollover(Z, A, FD, dx, dy)
    A_c = roll.A_rollover if np.isfinite(roll.A_rollover) else 1e4
    out["thresholds"] = {"rollover": roll.as_dict(), "A_c_primary": A_c, "ladder": list(ladder),
                         "rule": "slope-area rollover (max binned median D8 slope, 0.1-decade bins, 3-bin running median)"}
    logger.info("%s: rollover A_c = %.3g m^2 %s", tile, A_c, roll.note)

    # 2. chi fits at A_c* and on the ladder
    fits = {}
    nets = {}
    for a_c in [A_c, *ladder]:
        key = "primary" if a_c == A_c else f"{a_c:g}"
        try:
            net = channel_network(Z, A, FD, a_c, dx, dy)
            nets[key] = net
            fit = fit_chi(net, thetas=np.asarray(thetas), n_boot=n_boot, seed=seed)
            fits[key] = fit.as_dict()
            logger.info("%s A_c=%-8g theta*=%.3f (68%% %s) ks=%.4g (68%% %s) basins=%d pixels=%d", tile, a_c,
                        fit.theta_star, fit.theta_ci68, fit.ks_pooled, fit.ks_ci68, fit.n_basins_used, fit.n_pixels)
        except ValueError as exc:
            fits[key] = {"A_c": a_c, "error": str(exc)}
            logger.warning("%s A_c=%g: %s", tile, a_c, exc)
    out["chi_fits"] = fits
    primary = fits["primary"]
    theta_star = primary.get("theta_star", np.nan)
    ladder_fits = {k: v for k, v in fits.items() if k != "primary"}
    a_plateau, note_plateau = theta_plateau_threshold(ladder_fits)
    out["thresholds"]["A_c_theta_plateau"] = a_plateau
    out["thresholds"]["theta_plateau_note"] = note_plateau
    out["thresholds"]["theta_vs_threshold"] = [
        {"A_c": float(k), "theta_star": v.get("theta_star"), "ks": v.get("ks_pooled"), "n_basins": v.get("n_basins_used"),
         "theta_ci68": v.get("theta_ci68")} for k, v in sorted(ladder_fits.items(), key=lambda kv: float(kv[0]))
    ]
    if a_plateau is not None:
        fp = fits[f"{a_plateau:g}"]
        out["chi_fits_plateau"] = fp
        logger.info("%s: theta plateau from A_c=%g: theta*=%.3f ks=%.4g", tile, a_plateau, fp["theta_star"], fp["ks_pooled"])
    else:
        logger.info("%s: %s", tile, note_plateau)

    # 3. binned slope-area cross-check at A_c*
    try:
        Ks, th, dbg = slope_area_regression_binned(Z, A, FD, dx, min_area=A_c, vertical_interval=10, nbins=24)
        bA, bS, bN = dbg["bin_logA"], dbg["bin_logS"], dbg["bin_counts"].astype(float)
        out["slope_area"] = {"A_c": A_c, "ks": float(Ks), "theta": float(th),
                             "r2": float(weighted_r2(bA, bS, np.log10(Ks) - th * bA, bN)), "n": int(dbg["A"].size),
                             "note": "log-binned median slope-area regression (v1 analysis.py), 10 m vertical interval"}
    except Exception as exc:  # noqa: BLE001
        out["slope_area"] = {"A_c": A_c, "error": str(exc)}

    # 4. hilltop curvature at A_c*
    try:
        basin_of_channel = None
        if "primary" in nets:
            net = nets["primary"]
            # channel pixel labels in basin_labels are assigned in row-major order of the channel mask
            ch_rows, ch_cols = np.nonzero(net.mask)
            lookup = np.zeros(ch_rows.size, dtype=np.int64)
            pos = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(net.rows, net.cols, strict=True))}
            for i, (r, c) in enumerate(zip(ch_rows.tolist(), ch_cols.tolist(), strict=True)):
                lookup[i] = net.basin[pos[(r, c)]]
            basin_of_channel = lookup
        cur = measure_hilltop_curvature(Z, A, FD, A_c, site.U, dx, dy, n_boot=n_boot, seed=seed,
                                        basin_of_channel=basin_of_channel)
        out["curvature"] = cur.as_dict()
        logger.info("%s: C_ht=%.4g 1/m at L*=%.1f m (68%% %s), D0=%.4g m^2/yr, hilltops=%d", tile, cur.C_ht, cur.L_star_m,
                    cur.C_ht_ci68, cur.D0, cur.n_hilltop)
    except ValueError as exc:
        out["curvature"] = {"error": str(exc)}

    # 5. comparisons
    mv = td.cfg.measured_v1
    f = 1.0 / FT_US_IN_M if td.cfg.z_units != "m" else 1.0   # v1 AP values are in feet-based units
    v1 = {"theta": mv["theta"], "ks_m": mv["ks"] / f, "C_ht_m": mv["hilltop_curvature"] / f,
          "D0_m": site.U / (mv["hilltop_curvature"] / f), "ks_stored": mv["ks"], "C_ht_stored": mv["hilltop_curvature"],
          "conversion_factor_applied": 1.0 / f}
    reg = site.regional
    comp: dict[str, Any] = {"v1_hardcoded": v1, "regional": {"theta": reg.get("m"), "ks": (reg["U"] / reg["K"]) if "K" in reg else None,
                                                           "C_ht": reg.get("hilltop_curvature"), "D": reg.get("D"),
                                                           "source": reg.get("source")}}
    if np.isfinite(theta_star):
        ks_v1theta, r2_v1theta = ks_at_theta(nets["primary"], float(mv["theta"]))
        comp["ks_at_v1_theta_primary_threshold"] = {"theta": mv["theta"], "ks": ks_v1theta, "r2": r2_v1theta}
        ks_trap, r2_trap = ks_at_theta(nets["primary"], float(theta_star), rule="trapezoid")
        comp["ks_trapezoid_rule_at_theta_star"] = {"theta": theta_star, "ks": ks_trap, "r2": r2_trap,
                                                   "note": "Perron & Royden trapezoid chi instead of the model's upstream rule"}
        comp["ratios"] = {
            "theta_new_over_v1": theta_star / mv["theta"],
            "ks_new_over_v1_m": primary["ks_pooled"] / v1["ks_m"],
            "C_ht_new_over_v1_m": out["curvature"].get("C_ht", np.nan) / v1["C_ht_m"],
            "D0_new_over_v1_m": out["curvature"].get("D0", np.nan) / v1["D0_m"],
        }
    if run_legacy:
        try:
            t1 = time.perf_counter()
            lc = legacy_chi(Z, A, FD, dx, site.site)
            comp["legacy_v1_method_rerun"] = {"chi": lc.as_dict(), "wall_s": time.perf_counter() - t1}
            comp["legacy_v1_method_rerun"]["curvature_approx"] = legacy_curvature(Z, dx, site.site, site.U)
        except Exception as exc:  # noqa: BLE001
            comp["legacy_v1_method_rerun"] = {"error": repr(exc)}
    out["comparisons"] = comp
    out["wall_s"] = time.perf_counter() - t0
    return out


def run_measure(tiles: Sequence[str], force_metres: bool = True, n_boot: int = 200, run_id: str | None = None,
                results_root=None, run_legacy: bool = True, crop=None) -> RunRecorder:
    rec = RunRecorder("measure", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id,
                      root=results_root)
    results = {}
    curve_rows = []
    snaps = {}
    for tile in tiles:
        res = measure_tile(tile, force_metres=force_metres, n_boot=n_boot, run_legacy=run_legacy, crop=crop)
        results[tile] = res
        snaps[tile] = config_snapshot(load_tile_config(tile))
        for key, fit in res["chi_fits"].items():
            if "thetas" in fit:
                for th, ssr, dis in zip(fit["thetas"], fit["ssr_total"], fit["disorder_total"], strict=True):
                    curve_rows.append({"tile": tile, "curve": "chi_misfit", "threshold": key, "A_c": fit["A_c"], "x": th,
                                       "ssr": ssr, "disorder": dis})
        cur = res.get("curvature", {})
        if "window_lengths_m" in cur:
            for L, c, q16, q84, fc in zip(cur["window_lengths_m"], cur["C_ht_median"], cur["C_ht_q16"], cur["C_ht_q84"],
                                          cur["frac_convex"], strict=True):
                curve_rows.append({"tile": tile, "curve": "curvature_scale", "threshold": "primary", "A_c": cur["A_c"],
                                   "x": L, "C_ht": c, "q16": q16, "q84": q84, "frac_convex": fc})
        roll = res["thresholds"]["rollover"]
        for la, ms, ss, n in zip(roll["log10A_centers"], roll["median_slope"], roll["smoothed_slope"], roll["counts"],
                                 strict=True):
            curve_rows.append({"tile": tile, "curve": "slope_area_rollover", "threshold": "", "A_c": np.nan, "x": la,
                               "median_slope": ms, "smoothed_slope": ss, "count": n})
    rec.add_params("tiles", results)
    rec.add_params("settings", {"force_metres": force_metres, "n_boot": n_boot, "thetas": DEFAULT_THETAS.tolist(),
                                "ladder": list(THRESHOLD_LADDER)})
    rec.write_table(pd.DataFrame(curve_rows), "curves")
    summary = []
    for tile, r in results.items():
        p = r["chi_fits"]["primary"]
        c = r.get("curvature", {})
        fp = r.get("chi_fits_plateau", {})
        summary.append({"tile": tile, "A_c_rollover": r["thresholds"]["A_c_primary"], "theta_rollover": p.get("theta_star"),
                        "theta_ci68_rollover": p.get("theta_ci68"), "ks_rollover": p.get("ks_pooled"),
                        "ks_ci68_rollover": p.get("ks_ci68"),
                        "A_c_theta_plateau": r["thresholds"].get("A_c_theta_plateau"), "theta_plateau": fp.get("theta_star"),
                        "theta_ci68_plateau": fp.get("theta_ci68"), "ks_plateau": fp.get("ks_pooled"),
                        "ks_ci68_plateau": fp.get("ks_ci68"), "n_basins_plateau": fp.get("n_basins_used"),
                        "C_ht": c.get("C_ht"), "C_ht_ci68": c.get("C_ht_ci68"), "L_star_m": c.get("L_star_m"),
                        "D0": c.get("D0"), "D0_ci68": c.get("D0_ci68"),
                        "K0_rollover": r["U"] / p["ks_pooled"] if p.get("ks_pooled") else None,
                        "K0_plateau": r["U"] / fp["ks_pooled"] if fp.get("ks_pooled") else None})
    rec.add_params("summary", summary)
    rec.finalize({"tiles": snaps, "crop": crop})
    return rec
