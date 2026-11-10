"""Phase 5 (addendum 4, section 5): the AP vertical-unit correction.

* :func:`convert_ap_grids` writes the corrected filled-elevation GeoTIFFs (z in metres, exact US
  survey foot 1200/3937 m) next to the originals as ``*_filled_m.tif`` and adds them to
  ``data/manifest.csv`` (kind ``input_metres``). The originals are kept and the tile configs keep
  pointing at them; the loader converts at load time, and the written grid is checked to be
  bit-identical to the loader's in-memory conversion.
* :func:`d8_check` (section 5.2, a check only): recomputes D8 directions from the corrected filled
  tile and compares them cell by cell with the original ``*_fd`` grid, separating the differing
  cells into edge-adjacent cells, ties (two neighbours with the same steepest drop) and anything
  else. No grid is replaced: the area grids were computed on a larger DEM and clipped, and
  recomputing them on the tile would truncate every catchment that crosses the tile edge.
* :func:`run_invariance` (section 5.5): reruns the v2 partitioned profile (stage 2 and joint),
  the AD profile and the exponent profile on a metre tile **at exactly the evaluation points of
  the feet runs** (their A_crit grids, r values and (A_crit, m) grid) and checks that the optima
  and profile shapes are identical, that K and D equal c times the feet values and that the
  MSE equals the feet MSE divided by c^2 (c = 3937/1200).
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.analyses.tile_pipeline import mse
from lemcvt.config import TILE_IDS, config_snapshot, data_root, load_tile, load_tile_config
from lemcvt.hydro.d8 import d8_flow_direction
from lemcvt.io.grids import Grid, read_geotiff, sha256_file, write_geotiff
from lemcvt.io.manifest import MANIFEST_PATH, ManifestEntry, read_manifest, write_manifest
from lemcvt.io.units import FT_US_IN_M, M_IN_FT_US
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.common import extract_grid_data
from lemcvt.optimize.v1 import Trajectory
from lemcvt.optimize.v2 import ADProfiler, PartitionedProfiler, exponent_profile, partitioned_profile
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

AP_TILES = tuple(t for t in TILE_IDS if t.startswith("AP"))
C = M_IN_FT_US                      # 3937/1200 = 3.2808333...: K_m = C K_ft, D_m = C D_ft, MSE_m = MSE_ft / C^2
INV_C2 = FT_US_IN_M ** 2            # 0.09290341...
TOL_REL = 1e-8


# ----------------------------------------------------------------------------- 5.1 corrected grids

def convert_ap_grids(tiles=AP_TILES, run_id: str | None = None, results_root=None) -> RunRecorder:
    """Write ``<site>/tile<n>/<name>_filled_m.tif`` for every AP tile and add it to the manifest."""
    rec = RunRecorder("ap_units_convert", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id,
                      root=results_root)
    root = data_root()
    entries = {e.path: e for e in read_manifest()}
    out: dict[str, Any] = {}
    for tile in tiles:
        cfg = load_tile_config(tile, reproduce_v1_units=False)
        td = load_tile(cfg)                                   # z converted in memory: z_ft * (1200/3937)
        src = root / cfg.grids["filled"]
        feet = read_geotiff(src)
        rel = str(Path(cfg.grids["filled"]).with_name(Path(cfg.grids["filled"]).stem + "_m.tif"))
        dst = root / rel
        g = Grid(data=np.ascontiguousarray(td.Z, dtype=np.float64), transform=feet.transform, crs=feet.crs, nodata=None,
                 name=dst.stem)
        write_geotiff(g, dst, dtype="float64")
        back = read_geotiff(dst).data
        identical = bool(np.array_equal(back, td.Z, equal_nan=True))
        sha = sha256_file(dst)
        entries[rel] = ManifestEntry(rel, sha, dst.stat().st_size, cfg.site, f"tile{cfg.index}", "input_metres", "filled_m")
        out[tile] = {"source": str(src), "source_sha256": sha256_file(src), "written": str(dst), "sha256": sha,
                     "bytes": dst.stat().st_size, "factor_m_per_ftUS": FT_US_IN_M, "bit_identical_to_loader": identical,
                     "z_range_ft": [float(np.nanmin(feet.data)), float(np.nanmax(feet.data))],
                     "z_range_m": [float(np.nanmin(td.Z)), float(np.nanmax(td.Z))], "sanity": td.sanity,
                     "crs": str(feet.crs), "shape": list(td.Z.shape)}
        logger.info("%s: wrote %s (sha256 %s, bit-identical to loader: %s), z %.2f-%.2f m", tile, dst.name, sha[:12], identical,
                    out[tile]["z_range_m"][0], out[tile]["z_range_m"][1])
    ordered = sorted(entries.values(), key=lambda e: (e.site, e.tile, e.kind, e.path))
    write_manifest(ordered, MANIFEST_PATH)
    rec.add_params("tiles", out)
    rec.add_params("manifest", {"path": str(MANIFEST_PATH), "n_entries": len(ordered),
                                "added": [e.path for e in ordered if e.kind == "input_metres"]})
    rec.finalize({"tiles": {t: config_snapshot(load_tile_config(t)) for t in tiles}})
    return rec


# ----------------------------------------------------------------------------- 5.2 D8 check

_OFFS = [(1, 0, 1, 1.0), (2, 1, 1, 1.41), (4, 1, 0, 1.0), (8, 1, -1, 1.41), (16, 0, -1, 1.0), (32, -1, -1, 1.41),
         (64, -1, 0, 1.0), (128, -1, 1, 1.41)]      # code, drow, dcol, divisor (TopoAnalysis uses 1.41, not sqrt 2)


def _drops(Z: np.ndarray) -> dict[int, np.ndarray]:
    """Drop from every cell to each of its 8 neighbours (code -> (drow, dcol)), TopoAnalysis's divisors; -1e22 where
    the neighbour is outside the grid or NaN. ``d[code][r, c] = (Z[r, c] - Z[r + drow, c + dcol]) / div``."""
    ny, nx = Z.shape
    out = {}
    for code, dr, dc, div in _OFFS:
        d = np.full(Z.shape, -1e22)
        rs = slice(max(-dr, 0), ny - max(dr, 0))        # cells whose neighbour (r + dr, c + dc) is inside
        cs = slice(max(-dc, 0), nx - max(dc, 0))
        rn = slice(max(dr, 0), ny - max(-dr, 0))        # the neighbours
        cn = slice(max(dc, 0), nx - max(-dc, 0))
        val = (Z[rs, cs] - Z[rn, cn]) / div
        d[rs, cs] = np.where(np.isfinite(val), val, -1e22)
        out[code] = d
    return out


def d8_check(tile: str, rec: RunRecorder, rel_tol: float = 1e-9, abs_tol: float = 1e-9, n_examples: int = 20) -> dict[str, Any]:
    """Section 5.2. A differing cell is a *tie* when the drop to the original receiver equals the steepest drop within
    ``max(rel_tol |best|, abs_tol)`` (abs_tol in the grid's vertical unit: the priority-flood aggradation increments on
    filled flats are 1e-12 dx, so scaling z by a constant can reorder them in the last bit); otherwise *anything else*."""
    td = load_tile(tile, reproduce_v1_units=False)
    td_ft = load_tile(tile, reproduce_v1_units=True)
    Z_m, Z_ft, fd0 = td.Z, td_ft.Z, td.FD.astype(np.int64)
    ny, nx = Z_m.shape
    fd_m = d8_flow_direction(Z_m).astype(np.int64)
    fd_ft = d8_flow_direction(Z_ft).astype(np.int64)
    rows, cols = np.indices((ny, nx))
    margin = np.minimum(np.minimum(rows, ny - 1 - rows), np.minimum(cols, nx - 1 - cols))
    drops = _drops(Z_m)
    best = np.max(np.stack(list(drops.values())), axis=0)
    drop_orig = np.full(Z_m.shape, np.nan)
    for code, d in drops.items():
        sel = fd0 == code
        drop_orig[sel] = d[sel]
    orig_valid = np.isin(fd0, [c for c, *_ in _OFFS])
    gap = np.abs(drop_orig - best)
    tie = orig_valid & np.isfinite(drop_orig) & (gap <= np.maximum(rel_tol * np.abs(best), abs_tol))
    diff = fd_m != fd0
    edge2 = margin <= 1          # the outer two rows and columns (TopoAnalysis's slab convention and the mosaic clip)
    edge1 = margin == 0
    cls = np.zeros(Z_m.shape, dtype=np.int8)        # 0 same, 1 edge-adjacent, 2 tie, 3 other
    cls[diff & edge2] = 1
    cls[diff & ~edge2 & tie] = 2
    cls[diff & ~edge2 & ~tie] = 3
    other = np.argwhere(cls == 3)
    gap_other = gap[cls == 3]
    gap_other = gap_other[np.isfinite(gap_other)]
    examples = []
    for r, c in other[:n_examples].tolist():
        examples.append({"row": r, "col": c, "fd_orig": int(fd0[r, c]), "fd_new_m": int(fd_m[r, c]), "fd_new_ft": int(fd_ft[r, c]),
                         "z_m": float(Z_m[r, c]), "drops_m": {str(code): float(d[r, c]) for code, d in drops.items()},
                         "drop_to_orig_receiver": float(drop_orig[r, c]) if np.isfinite(drop_orig[r, c]) else None,
                         "best_drop": float(best[r, c]), "neighbours_nan": int(np.sum(~np.isfinite(Z_m[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2])))})
    res = {"tile": tile, "shape": [ny, nx], "n_cells": int(ny * nx),
           "n_diff_new_m_vs_orig": int(diff.sum()), "n_edge_adjacent_margin2": int((cls == 1).sum()),
           "n_edge_adjacent_margin1_only": int((diff & edge1).sum()), "n_tie_flats": int((cls == 2).sum()),
           "n_other": int((cls == 3).sum()), "n_interior_cells": int((~edge2).sum()),
           "n_diff_interior": int((diff & ~edge2).sum()),
           "n_diff_new_m_vs_new_ft": int((fd_m != fd_ft).sum()), "n_diff_new_ft_vs_orig": int((fd_ft != fd0).sum()),
           "n_diff_new_ft_vs_orig_interior": int(((fd_ft != fd0) & ~edge2).sum()),
           "n_orig_code_zero": int((fd0 == 0).sum()), "n_new_code_zero": int((fd_m == 0).sum()),
           "tie_tolerance_rel": rel_tol, "tie_tolerance_abs": abs_tol, "examples_other": examples,
           "gap_other_quantiles": {str(q): float(np.percentile(gap_other, q)) for q in (0, 50, 90, 100)} if gap_other.size else None,
           "n_other_recomputed_feet_equals_orig": int(((cls == 3) & (fd_ft == fd0)).sum()),
           "n_interior_diff_feet_recomputation_vs_orig": int(((fd_ft != fd0) & ~edge2).sum()),
           "note": "the original fd was computed on the larger DEM before clipping; the outer rows/columns of a recomputation "
                   "cannot match it (TopoAnalysis's slab convention gives the last two rows/columns no east/south comparison)"}
    rec.write_grid(td.filled.with_data(cls.astype(np.float64)), f"{tile}_d8_check_classes_0same_1edge_2tie_3other")
    logger.info("%s D8 check: %d differ (edge %d, tie %d, other %d); metres vs feet recomputation differ at %d cells", tile,
                res["n_diff_new_m_vs_orig"], res["n_edge_adjacent_margin2"], res["n_tie_flats"], res["n_other"],
                res["n_diff_new_m_vs_new_ft"])
    return res


def run_d8_check(tiles=AP_TILES, run_id: str | None = None, results_root=None) -> RunRecorder:
    rec = RunRecorder("ap_d8_check", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    out = {t: d8_check(t, rec) for t in tiles}
    rec.add_params("tiles", out)
    rec.finalize({"tiles": {t: config_snapshot(load_tile_config(t)) for t in tiles}})
    return rec


# ----------------------------------------------------------------------------- 5.5 invariance

def _feet_runs(tile: str) -> dict[str, Path]:
    """Latest v2 tile_run per stage whose working z unit was ftUS (the Phase 4 feet runs)."""
    out: dict[str, Path] = {}
    for r in find_runs("tile_run", tile):
        try:
            p = json.loads((r / "params.json").read_text())
            rj = json.loads((r / "run.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if p.get("optimizer") != "v2" or (rj.get("config") or {}).get("crop"):
            continue
        if p.get("inputs", {}).get("z_units_working") != "ftUS":
            continue
        for st in p.get("v2", {}):
            out[st] = r
    return out


def _cmp(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[ok] / b[ok] - 1.0))) if ok.any() else float("nan")


def run_invariance(tile: str, feet_runs: dict[str, str | Path] | None = None, stages=("partitioned", "ad", "m_profile"),
                   run_id: str | None = None, results_root=None) -> RunRecorder:
    runs = {k: Path(v) for k, v in (feet_runs or {}).items()} or _feet_runs(tile)
    missing = [s for s in stages if s not in runs]
    if missing:
        raise FileNotFoundError(f"{tile}: no feet (ftUS) v2 run for stages {missing}")
    td = load_tile(tile, reproduce_v1_units=False)
    mp = td.cfg.model
    if abs(mp.param_scale - C) > 1e-12:
        raise ValueError(f"{tile}: the loader applied param_scale {mp.param_scale!r}, expected c = {C!r}")
    Z0, A, FD, dx, U = td.Z, td.A, td.FD, td.dx, mp.U
    rec = RunRecorder("ap_invariance", tile, run_id=run_id, root=results_root)
    rec.add_params("tile", tile)
    rec.add_params("c", {"c_ft_per_m": C, "inv_c2": INV_C2, "tolerance_rel": TOL_REL})
    rec.add_params("inputs", td.provenance())
    rec.add_params("model_metres", {"K0_partitioned": mp.K0_partitioned, "D0": mp.D0, "K0_ad": mp.K0_ad, "K_bounds": mp.K_bounds,
                                    "D_bounds": mp.D_bounds, "m": mp.m, "param_scale": mp.param_scale})
    rec.add_params("feet_runs", {k: v.name for k, v in runs.items()})
    checks: dict[str, Any] = {}

    def _model_scaled_ok(p_ft: dict) -> dict[str, float]:
        mf = p_ft["model"]
        return {"K0_partitioned": mp.K0_partitioned / (C * mf["K0_partitioned"]) - 1, "D0": mp.D0 / (C * mf["D0"]) - 1,
                "K_lo": mp.K_bounds[0] / (C * mf["K_bounds"][0]) - 1, "K_hi": mp.K_bounds[1] / (C * mf["K_bounds"][1]) - 1,
                "D_lo": mp.D_bounds[0] / (C * mf["D_bounds"][0]) - 1, "D_hi": mp.D_bounds[1] / (C * mf["D_bounds"][1]) - 1}

    prof_p = None
    if "partitioned" in stages or "m_profile" in stages:
        prof_p = PartitionedProfiler(Z0, A, FD, dx, U, m_values=[mp.m])

    # ---------------------------------------------------------------- partitioned: same A_crit grid
    if "partitioned" in stages:
        t0 = time.perf_counter()
        r_ft = runs["partitioned"]
        p_ft = json.loads((r_ft / "params.json").read_text())
        prof_ft = pd.read_parquet(r_ft / "acrit_profile.parquet").sort_values("A_crit").reset_index(drop=True)
        grid = prof_ft.A_crit.to_numpy()
        extras = {}
        cert_ft = p_ft["v2"]["partitioned"].get("certificate", {})
        for name, cft in cert_ft.items():
            extras[name] = (float(cft["A_crit"]), float(cft["K"]) * C, float(cft["D"]) * C)
        traj = Trajectory("partitioned", ("log10_A_crit", "log10_D", "log10_K", "m", "kind"))
        res = partitioned_profile(prof_p, grid, mp.m, mp.K_bounds, mp.D_bounds, mp.K0_partitioned, mp.D0, traj=traj,
                                  extra_points=extras, log_every=200)
        rec.add_trajectory(traj)
        rows = pd.DataFrame(res.pop("rows")).sort_values("A_crit").reset_index(drop=True)
        rec.write_table(rows, "acrit_profile")
        assert np.array_equal(rows.A_crit.to_numpy(), grid)
        cmp = {
            "n_points": int(grid.size), "model_scaling_rel_err": _model_scaled_ok(p_ft),
            "mse_stage2_times_c2_vs_feet": _cmp(rows.mse_stage2 / INV_C2, prof_ft.mse_stage2),
            "mse_joint_times_c2_vs_feet": _cmp(rows.mse_joint / INV_C2, prof_ft.mse_joint),
            "K_opt_vs_c_K_feet": _cmp(rows.K_opt, C * prof_ft.K_opt), "D_opt_vs_c_D_feet": _cmp(rows.D_opt, C * prof_ft.D_opt),
            "n_channel_identical": bool(np.array_equal(rows.n_channel, prof_ft.n_channel)),
            "where_identical": bool((rows.where == prof_ft.where).all()),
            "shape_stage2": _cmp(rows.mse_stage2 / rows.mse_stage2.min(), prof_ft.mse_stage2 / prof_ft.mse_stage2.min()),
            "shape_joint": _cmp(rows.mse_joint / rows.mse_joint.min(), prof_ft.mse_joint / prof_ft.mse_joint.min()),
            "stage2_optimum_A_crit": {"metres": res["stage2_optimum"]["A_crit"], "feet": p_ft["v2"]["partitioned"]["stage2_optimum"]["A_crit"]},
            "joint_optimum_A_crit": {"metres": res["joint_optimum"]["A_crit"], "feet": p_ft["v2"]["partitioned"]["joint_optimum"]["A_crit"]},
            "argmin_stage2_identical": bool(int(rows.mse_stage2.idxmin()) == int(prof_ft.mse_stage2.idxmin())),
            "argmin_joint_identical": bool(int(rows.mse_joint.idxmin()) == int(prof_ft.mse_joint.idxmin())),
            "joint_optimum_K_ratio_to_c_feet": res["joint_optimum"]["K"] / (C * p_ft["v2"]["partitioned"]["joint_optimum"]["K"]),
            "joint_optimum_D_ratio_to_c_feet": res["joint_optimum"]["D"] / (C * p_ft["v2"]["partitioned"]["joint_optimum"]["D"]),
            "joint_optimum_mse_ratio_to_feet": res["joint_optimum"]["mse"] / p_ft["v2"]["partitioned"]["joint_optimum"]["mse"],
            "certificates": {},
            "wall_s": time.perf_counter() - t0,
        }
        for name, cm in res.get("certificate", {}).items():
            cf = cert_ft[name]
            cmp["certificates"][name] = {"mse_times_c2_vs_feet": cm["mse"] / INV_C2 / cf["mse"] - 1,
                                         "gradient_K_rel": cm["gradient_log10"]["K"] / INV_C2 / cf["gradient_log10"]["K"] - 1 if cf["gradient_log10"]["K"] else None,
                                         "gradient_D_rel": cm["gradient_log10"]["D"] / INV_C2 / cf["gradient_log10"]["D"] - 1 if cf["gradient_log10"]["D"] else None,
                                         "best_K_vs_c_feet": cm["best_at_this_A_crit"]["K"] / (C * cf["best_at_this_A_crit"]["K"]) - 1,
                                         "best_D_vs_c_feet": cm["best_at_this_A_crit"]["D"] / (C * cf["best_at_this_A_crit"]["D"]) - 1}
        cmp["pass"] = bool(cmp["argmin_stage2_identical"] and cmp["argmin_joint_identical"] and cmp["n_channel_identical"]
                           and max(cmp["mse_stage2_times_c2_vs_feet"], cmp["mse_joint_times_c2_vs_feet"], cmp["K_opt_vs_c_K_feet"],
                                   cmp["D_opt_vs_c_D_feet"], cmp["shape_stage2"], cmp["shape_joint"]) <= TOL_REL)
        checks["partitioned"] = cmp
        res["feet_run"] = r_ft.name
        rec.add_params("partitioned", res)
        logger.info("%s partitioned invariance: pass=%s (max rel: mse %.2e / %.2e, K %.2e, D %.2e, shape %.2e / %.2e) in %.0f s", tile,
                    cmp["pass"], cmp["mse_stage2_times_c2_vs_feet"], cmp["mse_joint_times_c2_vs_feet"], cmp["K_opt_vs_c_K_feet"],
                    cmp["D_opt_vs_c_D_feet"], cmp["shape_stage2"], cmp["shape_joint"], cmp["wall_s"])

    # ---------------------------------------------------------------- AD: same r values
    if "ad" in stages:
        t0 = time.perf_counter()
        r_ft = runs["ad"]
        p_ft = json.loads((r_ft / "params.json").read_text())
        ad_ft = p_ft["v2"]["ad"]
        rows_ft = [q for q in ad_ft["grid_rows"] if q.get("feasible")]
        ad = advection_diffusion_model(Z0, A, FD, dx, mp.m)
        gd = extract_grid_data(Z0, A, FD, dx)
        prof = ADProfiler(ad, Z0[gd["rev_j"], gd["rev_i"]], U)
        traj = Trajectory("ad", ("log10_K", "log10_D", "log10_r", "kind"))
        out_rows = []
        for q in rows_ft:
            r = float(q["r"])
            red = prof.reduce(r)
            D_lo = max(mp.D_bounds[0], mp.K_bounds[0] / r)
            D_hi = min(mp.D_bounds[1], mp.K_bounds[1] / r)
            y, m_, where = prof.best_y(red, U / D_hi, U / D_lo)
            D = U / y
            out_rows.append({"r": r, "K": r * D, "D": D, "mse": m_, "where": where, "wall_s": red["wall_s"]})
            traj.record({"log10_K": math.log10(r * D), "log10_D": math.log10(D), "log10_r": math.log10(r), "kind": "invariance"}, m_, cached=False)
        rec.add_trajectory(traj)
        df = pd.DataFrame(out_rows)
        df_ft = pd.DataFrame(rows_ft)
        rec.write_table(df, "ad_profile")
        j_m, j_ft = int(df.mse.idxmin()), int(df_ft.mse.idxmin())
        cmp = {"n_points": int(df.shape[0]), "model_scaling_rel_err": _model_scaled_ok(p_ft),
               "mse_times_c2_vs_feet": _cmp(df.mse / INV_C2, df_ft.mse), "K_vs_c_K_feet": _cmp(df.K, C * df_ft.K),
               "D_vs_c_D_feet": _cmp(df.D, C * df_ft.D), "where_identical": bool((df.where == df_ft.where).all()),
               "shape": _cmp(df.mse / df.mse.min(), df_ft.mse / df_ft.mse.min()),
               "argmin_identical": bool(j_m == j_ft), "r_optimum": {"metres": float(df.r[j_m]), "feet": float(df_ft.r[j_ft])},
               "optimum_K_ratio_to_c_feet": float(df.K[j_m] / (C * df_ft.K[j_ft])), "optimum_D_ratio_to_c_feet": float(df.D[j_m] / (C * df_ft.D[j_ft])),
               "optimum_mse_ratio_to_feet": float(df.mse[j_m] / df_ft.mse[j_ft]), "wall_s": time.perf_counter() - t0}
        # the v1 AD point, same code, in metres (K, D x c) against the feet certificate
        cv = ad_ft.get("certificate", {}).get("v1_optimum")
        if cv:
            Zv = ad(U, C * cv["K"], C * cv["D"])
            cmp["v1_point_mse_times_c2_vs_feet"] = mse(Zv, Z0) / INV_C2 / cv["mse_same_code"] - 1
        cmp["pass"] = bool(cmp["argmin_identical"] and max(cmp["mse_times_c2_vs_feet"], cmp["K_vs_c_K_feet"], cmp["D_vs_c_D_feet"], cmp["shape"]) <= TOL_REL)
        checks["ad"] = cmp
        rec.add_params("ad", {"feet_run": r_ft.name, "optimum": out_rows[j_m], "n_evaluations": len(out_rows)})
        logger.info("%s AD invariance: pass=%s (max rel: mse %.2e, K %.2e, D %.2e, shape %.2e) in %.0f s", tile, cmp["pass"],
                    cmp["mse_times_c2_vs_feet"], cmp["K_vs_c_K_feet"], cmp["D_vs_c_D_feet"], cmp["shape"], cmp["wall_s"])

    # ---------------------------------------------------------------- exponent profile: same (A_crit, m) grid
    if "m_profile" in stages:
        t0 = time.perf_counter()
        r_ft = runs["m_profile"]
        p_ft = json.loads((r_ft / "params.json").read_text())
        mp_ft = pd.read_parquet(r_ft / "m_profile.parquet")
        mp_ft["m"] = mp_ft.m.round(2)
        grid = np.sort(mp_ft.A_crit.unique())
        m_values = np.sort(mp_ft.m.unique())
        traj = Trajectory("m_profile", ("log10_A_crit", "log10_D", "log10_K", "m", "kind"))
        res = exponent_profile(prof_p, grid, m_values, mp.K_bounds, mp.D_bounds, mp.K0_partitioned, mp.D0, traj=traj, log_every=50)
        rec.add_trajectory(traj)
        rows = pd.DataFrame(res.pop("rows"))
        rows["m"] = rows.m.round(2)
        rec.write_table(rows, "m_profile")
        mg = rows.merge(mp_ft, on=["A_crit", "m"], suffixes=("_m", "_ft"))
        assert mg.shape[0] == mp_ft.shape[0], (mg.shape, mp_ft.shape)
        o_m, o_ft = res["optimum"], p_ft["v2"]["m_profile"]["optimum"]
        cmp = {"n_points": int(mg.shape[0]), "model_scaling_rel_err": _model_scaled_ok(p_ft),
               "mse_joint_times_c2_vs_feet": _cmp(mg.mse_joint_m / INV_C2, mg.mse_joint_ft),
               "K_opt_vs_c_K_feet": _cmp(mg.K_opt_m, C * mg.K_opt_ft), "D_opt_vs_c_D_feet": _cmp(mg.D_opt_m, C * mg.D_opt_ft),
               "where_identical": bool((mg.where_m == mg.where_ft).all()),
               "shape": _cmp(mg.mse_joint_m / mg.mse_joint_m.min(), mg.mse_joint_ft / mg.mse_joint_ft.min()),
               "optimum": {"metres": {"A_crit": o_m["A_crit"], "m": o_m["m"]}, "feet": {"A_crit": o_ft["A_crit"], "m": o_ft["m"]}},
               "optimum_identical": bool(o_m["A_crit"] == o_ft["A_crit"] and abs(o_m["m"] - o_ft["m"]) < 1e-9),
               "profile_m_shape": _cmp(np.array([q["mse"] for q in res["profile_m"]]) / min(q["mse"] for q in res["profile_m"]),
                                       np.array([q["mse"] for q in p_ft["v2"]["m_profile"]["profile_m"]]) / min(q["mse"] for q in p_ft["v2"]["m_profile"]["profile_m"])),
               "wall_s": time.perf_counter() - t0}
        cmp["pass"] = bool(cmp["optimum_identical"] and max(cmp["mse_joint_times_c2_vs_feet"], cmp["K_opt_vs_c_K_feet"],
                                                            cmp["D_opt_vs_c_D_feet"], cmp["shape"], cmp["profile_m_shape"]) <= TOL_REL)
        checks["m_profile"] = cmp
        res["feet_run"] = r_ft.name
        rec.add_params("m_profile", res)
        logger.info("%s exponent-profile invariance: pass=%s (max rel: mse %.2e, K %.2e, D %.2e, shape %.2e) in %.0f s", tile,
                    cmp["pass"], cmp["mse_joint_times_c2_vs_feet"], cmp["K_opt_vs_c_K_feet"], cmp["D_opt_vs_c_D_feet"], cmp["shape"],
                    cmp["wall_s"])

    checks["all_pass"] = bool(all(v.get("pass") for k, v in checks.items() if isinstance(v, dict)))
    rec.add_params("checks", checks)
    rec.finalize(config_snapshot(td.cfg, {"feet_runs": {k: v.name for k, v in runs.items()}, "stages": list(stages)}))
    return rec
