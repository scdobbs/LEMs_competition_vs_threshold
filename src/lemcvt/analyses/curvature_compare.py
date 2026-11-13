"""Addendum 6, section 4 (the curvature comparison that does not depend on the window choice) and addendum 5, 3.7
(the D-recovery diagnostics), from results only.

For each tile, on the option (iii) pixel set, the median -laplacian C(L) at the nine test windows is computed with the
same estimator from the observed DEM and from three fitted surfaces: the linear joint optimum at the site m, the
nonlinear joint optimum, and the AD optimum. Reported: the ratio C_obs(L)/C_model(L) at every window, the fraction of
those pixels that are channel cells in each fitted model, the model's 5-point Laplacian on those pixels (for the
linear model -U/D wherever they are hillslope cells), and D_fit (linear joint, nonlinear joint) against D0_A and D0_B.
No pass rule.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.config import config_snapshot, load_tile_config
from lemcvt.io.grids import read_geotiff
from lemcvt.measure.diagnostics import _metre_tile
from lemcvt.measure.hilltop_curvature import total_curvature
from lemcvt.measure.phase6_inputs import TEST_HALF_WIDTHS, W_F, option_iii_mask
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)


def _runs_by_stage(tile: str) -> dict[str, tuple[Path, dict]]:
    """Latest Phase 6 run per stage for the tile (metres, D0 variant A, uncropped, K factor 1)."""
    out: dict[str, tuple[Path, dict]] = {}
    for r in find_runs("tile_run", tile):
        try:
            p = json.loads((r / "params.json").read_text())
            rj = json.loads((r / "run.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if p.get("optimizer") != "v2" or (rj.get("config") or {}).get("crop") or p.get("inputs", {}).get("z_units_working") != "m":
            continue
        if p.get("model", {}).get("d0_variant", "A") != "A":
            continue
        v2 = p.get("v2", {})
        if "ad" in v2:
            out["ad"] = (r, p)
        if "partitioned" in v2 and r.name.endswith("partitioned-refined"):
            out["joint"] = (r, p)
        # the addendum 8 merged runs at the fitted S_c (S_c_override) are newer than the Phase 6 runs at S_c = 1.27; found
        # in addendum 10 (no result built before it was affected; reported)
        if ("nonlinear_joint" in v2 and v2["nonlinear_joint"]["setup"].get("k_factor", 1.0) == 1.0 and "split" not in v2["nonlinear_joint"]["setup"]
                and v2["nonlinear_joint"]["setup"].get("S_c_override") is None):
            out["nonlinear_joint"] = (r, p)
        if "nonlinear" in v2:
            out["nonlinear"] = (r, p)
    return out


def _grid(run: Path, suffix: str) -> np.ndarray | None:
    hits = sorted(run.glob(f"*_{suffix}_elevation.tif"))
    return read_geotiff(hits[-1]).data if hits else None


def laplacian5(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The models' 5-point Laplacian (NaN at the edge)."""
    out = np.full(z.shape, np.nan)
    out[1:-1, 1:-1] = (z[1:-1, 2:] + z[1:-1, :-2] - 2 * z[1:-1, 1:-1]) / dx**2 + (z[2:, 1:-1] + z[:-2, 1:-1] - 2 * z[1:-1, 1:-1]) / dy**2
    return out


def run_curvature_compare(tiles: Sequence[str], run_id: str | None = None, results_root=None, half_widths=TEST_HALF_WIDTHS) -> RunRecorder:
    rec = RunRecorder("curvature_compare", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    curv_params: dict[str, Any] = {}
    for r in find_runs("measure"):
        try:
            p = json.loads((r / "params.json").read_text())
            for t, v in p.get("tiles", {}).items():
                if "curvature" in v and "A_c" in v["curvature"]:
                    curv_params[t] = v["curvature"]
        except Exception:  # noqa: BLE001
            continue
    results: dict[str, Any] = {}
    rows = []
    snaps = {}
    for tile in tiles:
        td = _metre_tile(tile)
        cfg = td.cfg
        mp = cfg.model
        Z, A, FD, dx, dy, U = td.Z, td.A, td.FD, td.dx, td.filled.dy, td.site.U
        mask, _m0, _dist, diag = option_iii_mask(Z, A, FD, dx, dy, float(curv_params[tile]["A_c"]), W_F)
        runs = _runs_by_stage(tile)
        surfaces: dict[str, tuple[np.ndarray, dict[str, Any]]] = {"observed": (Z, {})}
        if "joint" in runs:
            r, p = runs["joint"]
            z = _grid(r, "Partitioned-model-a_crit-k-d-opt")
            jo = p["v2"]["partitioned"]["joint_optimum"]
            if z is not None:
                surfaces["linear_joint"] = (z, {"run": r.name, "A_crit": jo["A_crit"], "K": jo["K"], "D": jo["D"], "mse": jo["mse"]})
        if "nonlinear_joint" in runs:
            r, p = runs["nonlinear_joint"]
            z = _grid(r, "Partitioned-model-nonlinear-joint-opt")
            o = p["v2"]["nonlinear_joint"]["optimum"]
            if z is not None:
                surfaces["nonlinear_joint"] = (z, {"run": r.name, "A_crit": o["A_crit"], "K": o["K"], "D": o["D"], "mse": o["mse"]})
        if "ad" in runs:
            r, p = runs["ad"]
            z = _grid(r, "AD-opt")
            o = p["v2"]["ad"]["optimum"]
            if z is not None:
                surfaces["ad"] = (z, {"run": r.name, "K": o["K"], "D": o["D"], "mse": o["mse"], "r": o["r"]})
        out: dict[str, Any] = {"tile": tile, "site": cfg.site, "U": U, "n_pixels": int(mask.sum()), "set": diag, "surfaces": {}, "D0": mp.D0, "D0_B": mp.D0_B}
        Lnom = [int(2 * w + 1) for w in half_widths]
        curves: dict[str, list[float]] = {}
        for name, (z, meta) in surfaces.items():
            c = []
            for w in half_widths:
                C = total_curvature(z, w, dx, dy)
                v = -C[mask]
                v = v[np.isfinite(v)]
                c.append(float(np.median(v)) if v.size else float("nan"))
            curves[name] = c
            lap = laplacian5(z, dx, dy)[mask]
            lap = lap[np.isfinite(lap)]
            srec: dict[str, Any] = dict(meta) | {"C_ht_L": dict(zip([str(L) for L in Lnom], c, strict=True)),
                                                 "median_minus_laplacian5": float(np.median(-lap)) if lap.size else None}
            if "A_crit" in meta:
                ch = np.isfinite(A) & (A >= float(meta["A_crit"]))
                srec["frac_option_iii_pixels_channel"] = float(np.mean(ch[mask])) if mask.any() else None
                hs = mask & ~ch
                lap_h = laplacian5(z, dx, dy)[hs]
                lap_h = lap_h[np.isfinite(lap_h)]
                srec["median_minus_laplacian5_hillslope_pixels"] = float(np.median(-lap_h)) if lap_h.size else None
                srec["U_over_D"] = float(U / meta["D"]) if meta.get("D") else None
            elif name == "ad":
                srec["frac_option_iii_pixels_channel"] = None
            if name != "observed":
                srec["ratio_obs_over_model_L"] = {str(L): float(curves["observed"][i] / c[i]) if c[i] else None for i, L in enumerate(Lnom)}
            out["surfaces"][name] = srec
            for i, L in enumerate(Lnom):
                rows.append({"tile": tile, "surface": name, "L_nominal_m": L, "C_ht": c[i],
                             "ratio_obs_over_model": float(curves["observed"][i] / c[i]) if name != "observed" and c[i] else None})
        # D recovery (addendum 5, 3.7; addendum 6, 5.7)
        drec: dict[str, Any] = {"D0_A": mp.D0, "D0_B": mp.D0_B, "C_ht_obs_at_L_f": curves["observed"][Lnom.index(11)] if 11 in Lnom else None}
        for name in ("linear_joint", "nonlinear_joint", "ad"):
            if name in surfaces:
                D = surfaces[name][1].get("D")
                drec[f"D_fit_{name}"] = D
                drec[f"D_fit_{name}_over_D0_A"] = float(D / mp.D0) if D else None
                drec[f"D_fit_{name}_over_D0_B"] = float(D / mp.D0_B) if (D and mp.D0_B) else None
        out["d_recovery"] = drec
        results[tile] = out
        snaps[tile] = config_snapshot(load_tile_config(tile))
        logger.info("%s: surfaces %s; obs/model at 11 m: %s", tile, list(surfaces), {k: round(v["ratio_obs_over_model_L"]["11"], 3) for k, v in out["surfaces"].items() if "ratio_obs_over_model_L" in v and v["ratio_obs_over_model_L"].get("11")})
    rec.add_params("tiles", results)
    rec.add_params("settings", {"half_widths": list(half_widths), "w_f": W_F, "pixel_set": "option (iii): measure hilltop set of the rollover network, L_f/2 exclusion"})
    rec.write_table(pd.DataFrame(rows), "curvature_curves")
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.finalize({"tiles": snaps})
    return rec
