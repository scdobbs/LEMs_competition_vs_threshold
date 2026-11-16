"""Addendum 9: a fair test of the nonlinear hillslope model (Newton solver, S_c uncertainty, morphology beside the MSE).

Section 2 (solver and tests) lives in :mod:`lemcvt.models.nonlinear_newton`; this module runs it on the tiles.

* :func:`run_pilot` (2.2 tests 1 and 4, 5.1-5.2): on each pilot tile, one A_crit row (the 30-value grid value nearest the
  linear joint A_crit) across the 22-value D grid at the tile's fitted S_c, with Picard (centre form, cap from the config)
  and Newton with continuation in D on both forms; the Phase 6 setting (S_c = 1.27 at the Phase 6 nonlinear joint
  optimum) on the full tile and on a crop, Picard against Newton; wall times; the core-hour estimate of section 3.
* :func:`run_fair_part` (3.1): one interleaved subset of the A_crit grid at one S_c, Newton with continuation in D
  (largest D first, from the linear solution; each next D from the last converged solution), partial rows written
  after every A_crit row, and the best converged surface kept on disk.
* :func:`merge_fair` and :func:`summarize_fair` (3.1-3.2): the optimum over converged evaluations, the outcome counts,
  the contours, D/D0 and the pre-registered rule.
* :func:`run_morphology` (4): crest curvature, the E*-R* ratio along the observed D8 paths and the hillslope gradients
  of every fitted surface against the observed DEM, with the pre-registered factor-1.25 rule.

The model form used for section 3 is ``flux`` (div(D_eff grad z) = -U), at Steve's decision of 2026-11-13 after the
finding that the v1 nonlinear model (``centre``) solves D_eff lap(z) = -U (report section 2).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from lemcvt.analyses.optimizer_v2 import _contours, _latest_refined_partitioned
from lemcvt.analyses.outputs import write_model_outputs
from lemcvt.analyses.tile_pipeline import mse
from lemcvt.config import config_snapshot, load_tile, results_root
from lemcvt.io.grids import read_geotiff
from lemcvt.models.partitioned_nonlinear import partitioned_threshold_model_nonlinear
from lemcvt.optimize.v2 import thin_log_uniform, unique_areas
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

ER_STAR_RUN = "20261113T015030Z_er_star_10tiles"
NETWORK = "primary_1e3"
N_ACRIT = 30
WIDEN = 3.0
D_GRID = (1e-3, 5e-2, 22)
FORM = "flux"
SC_LABELS = ("central", "p16", "p84")
RULE3: dict[str, Any] = {"mse_ratio_max": 1.05, "D_ratio": (0.67, 1.5), "AP_bracket_widen": 1.5, "min_tiles": 4, "untested_not_converged_frac": 0.30}
RULE4: dict[str, Any] = {"factor": 1.25, "min_tiles": 4}
PILOT_TILES = ("AP1", "GM1")
# addendum 10, 1.1: every nonlinear result of v1, Phase 6 and addenda 8-9 carries this label
CENTRE_FORM_LABEL = "centre form, D_eff lap(z) = -U (nonstandard)"
CROP = (300, 600, 300, 600)


# ----------------------------------------------------------------------------- inputs

def d_grid() -> np.ndarray:
    lo, hi, n = D_GRID
    return np.logspace(math.log10(lo), math.log10(hi), n)


def s_c_values(tile: str, er_run: str = ER_STAR_RUN) -> dict[str, float]:
    """The addendum 8 primary-network S_c fit and the 16th and 84th percentiles of its segment bootstrap."""
    p = json.loads((results_root() / er_run / "params.json").read_text())
    sc = p["tiles"][tile]["networks"][NETWORK]["S_c_fit"]
    return {"central": float(sc["S_c"]), "p16": float(sc["S_c_ci68"][0]), "p84": float(sc["S_c_ci68"][1])}


def tile_setup(tile: str, crop: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """Tile data, K and the linear joint optimum of the refined Phase 6 run, and the addendum 6 A_crit grid (30 values)."""
    src = _latest_refined_partitioned(tile)
    p0 = json.loads((src / "params.json").read_text())
    jo = p0["v2"]["partitioned"]["joint_optimum"]
    i5 = p0["v2"]["partitioned"]["joint_identifiability"]["level_0.05"]
    td = load_tile(tile, crop=crop, sanity=crop is None)
    lo_i, hi_i = float(i5["A_crit_min"]), float(i5["A_crit_max"])
    grid_a = thin_log_uniform(unique_areas(td.A, lo_i / WIDEN, hi_i * WIDEN), N_ACRIT)
    return {"td": td, "mp": td.cfg.model, "source_run": src.name, "K": float(jo["K"]), "linear_joint": jo, "grid_a": grid_a,
            "interval_5pct": [lo_i, hi_i]}


def _row(f, Z, Z0, A_crit, D, K, S_c, wall, mask) -> dict[str, Any]:
    out = f.last_outcome
    hs = f.last_hillslope_indices
    ratio = f.compute_slopes(f.last_z_flat, hs) / S_c if hs.size else np.zeros(0)
    nw = f.last_newton or {}
    pic = f.last_picard or {}
    return {"A_crit": float(A_crit), "D": float(D), "K": float(K), "S_c": float(S_c), "solver": out["solver"], "form": out["form"],
            "mse": mse(Z, Z0), "outcome": out["outcome"], "converged": out["converged"], "n_iter": out["n_iter"], "stop_reason": out["stop_reason"],
            "rel_residual": nw.get("rel_residual"), "final_update": (pic["iterations"][-1]["update_max_abs"] if pic.get("iterations") else
                                                                   (nw["iterations"][-1].get("update_max_abs") if nw.get("iterations") else None)),
            "frac_clipped": out["frac_clipped"], "frac_hillslope_gt_0.5Sc": float(np.mean(ratio > 0.5)) if ratio.size else None,
            "frac_hillslope_gt_0.9Sc": float(np.mean(ratio > 0.9)) if ratio.size else None, "max_slope_ratio": float(np.max(ratio)) if ratio.size else None,
            "n_channel": int(mask.sum()), "n_hillslope": int(hs.size), "initial": nw.get("initial", "linear"), "wall_s": float(wall)}


def eval_row(f, Z0: np.ndarray, A_crit: float, D_values: np.ndarray, K: float, S_c: float, solver: str, form: str,
             max_iter: int, tol: float, keep: bool = False, on_eval=None) -> tuple[list[dict[str, Any]], dict[float, np.ndarray]]:
    """One A_crit row. Newton: D descending with continuation (the first D from the linear solution, each next D from
    the most recent converged solution). Picard: every D independently, as in addendum 8. Returns the rows and, with
    ``keep``, the full-grid surface of every converged evaluation keyed by D."""
    rows: list[dict[str, Any]] = []
    kept: dict[float, np.ndarray] = {}
    z_init = None
    order = np.sort(D_values)[::-1] if solver == "newton" else np.sort(D_values)
    for D in order:
        t1 = time.perf_counter()
        Z, mask = f(float(A_crit), float(D), S_c, max_iter, tol, solver=solver, form=form, z_init=z_init)
        r = _row(f, Z, Z0, A_crit, D, K, S_c, time.perf_counter() - t1, mask)
        rows.append(r)
        if on_eval is not None:
            on_eval(r)
        if r["outcome"] == "converged":
            if solver == "newton":
                z_init = f.last_z_flat.copy()
            if keep:
                kept[float(D)] = Z
    return rows, kept


# ----------------------------------------------------------------------------- 2.2 tests 1 and 4, the pilot

def _phase6_nonlinear_joint_optimum(tile: str) -> dict[str, Any] | None:
    """The Phase 6 nonlinear joint optimum at S_c = 1.27 (merged or unsplit run without an S_c override, K factor 1)."""
    best = None
    for r in find_runs("tile_run", tile):
        try:
            p = json.loads((r / "params.json").read_text())
            rj = json.loads((r / "run.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        v = p.get("v2", {}).get("nonlinear_joint")
        if not v or (rj.get("config") or {}).get("crop") or "split" in v["setup"] or v["setup"].get("k_factor", 1.0) != 1.0:
            continue
        if v["setup"].get("S_c_override") is not None or p.get("inputs", {}).get("z_units_working") != "m":
            continue
        best = {"run": r.name, "dir": r, "optimum": v["optimum"], "S_c": float(p["model"]["S_c"]), "K": float(v["setup"]["K"])}
    return best


def _compare(f, Z0, A_crit, D, K, S_c, mp, label: str) -> dict[str, Any]:
    """Picard (cap from the config, tol 1e-6), Picard tightened (tol 1e-10, cap 2000) and Newton, centre form, at one point."""
    out: dict[str, Any] = {"label": label, "A_crit": float(A_crit), "D": float(D), "K": float(K), "S_c": float(S_c)}
    t = time.perf_counter()
    Zp, _ = f(A_crit, D, S_c, mp.nonlinear_max_iter, mp.nonlinear_tol)
    out["picard"] = dict(f.last_outcome) | {"wall_s": time.perf_counter() - t, "final_update": f.last_picard["iterations"][-1]["update_max_abs"] if f.last_picard["iterations"] else None}
    t = time.perf_counter()
    Zt, _ = f(A_crit, D, S_c, 2000, 1e-10)
    out["picard_tight"] = dict(f.last_outcome) | {"wall_s": time.perf_counter() - t}
    t = time.perf_counter()
    Zn, _ = f(A_crit, D, S_c, solver="newton", form="centre")
    out["newton"] = dict(f.last_outcome) | {"wall_s": time.perf_counter() - t, "rel_residual": f.last_newton["rel_residual"]}
    out["max_abs_dz_newton_picard"] = float(np.nanmax(np.abs(Zn - Zp)))
    out["max_abs_dz_newton_picard_tight"] = float(np.nanmax(np.abs(Zn - Zt)))
    out["max_abs_dz_picard_picard_tight"] = float(np.nanmax(np.abs(Zp - Zt)))
    out["pass_1e-6"] = bool(out["max_abs_dz_newton_picard"] <= 1e-6)
    out["mse_newton"] = mse(Zn, Z0)
    out["mse_picard"] = mse(Zp, Z0)
    logger.info("test 1 %s: Picard %s in %d it, Newton %s in %d it; max |dz| Newton-Picard %.2e m (tight Picard %.2e m)", label,
                out["picard"]["outcome"], out["picard"]["n_iter"], out["newton"]["outcome"], out["newton"]["n_iter"],
                out["max_abs_dz_newton_picard"], out["max_abs_dz_newton_picard_tight"])
    return out


def _pilot_row_A(st: dict[str, Any]) -> float:
    """The 30-value A_crit grid value nearest the linear joint A_crit (in log)."""
    return float(st["grid_a"][int(np.argmin(np.abs(np.log(st["grid_a"] / float(st["linear_joint"]["A_crit"])))))])


def flux_pilot_D(st: dict[str, Any]) -> np.ndarray:
    """The flux-form pilot subset of the D grid: the largest D and the grid values nearest the linear joint D and D0_A."""
    Dg = d_grid()
    pick = {float(Dg[-1])}
    for target in (float(st["linear_joint"]["D"]), float(st["mp"].D0)):
        pick.add(float(Dg[int(np.argmin(np.abs(np.log(Dg / target))))]))
    return np.array(sorted(pick))


def run_pilot(tile: str, part: str = "centre", run_id: str | None = None, results_root_=None) -> RunRecorder:
    """One pilot task. ``part="centre"``: 2.2 test 4 (Picard and Newton on the v1 form across the full D grid, one A_crit
    row, the tile's fitted S_c) and test 1 at the Phase 6 setting (full tile and crop). ``part="flux"``: Newton on the flux
    form at three D values of the same row (:func:`flux_pilot_D`), for the outcome classes and the cost per solve."""
    if part not in ("centre", "flux", "test1"):
        raise ValueError(part)
    rec = RunRecorder("a9_pilot", f"{tile}_{part}", run_id=run_id, root=results_root_)
    t0 = time.perf_counter()
    st = tile_setup(tile)
    td, mp, K = st["td"], st["mp"], st["K"]
    sc = s_c_values(tile)["central"]
    a_row = _pilot_row_A(st)
    f = partitioned_threshold_model_nonlinear(td.Z, td.A, td.FD, td.dx, mp.m, mp.U, K)
    rows: list[dict[str, Any]] = []
    runs: tuple = ((("picard", "centre", d_grid()), ("newton", "centre", d_grid())) if part == "centre" else
                   (("newton", "flux", flux_pilot_D(st)),) if part == "flux" else ())
    done: list[dict[str, Any]] = []

    def _partial(x: dict[str, Any]) -> None:          # every evaluation survives a timeout
        done.append(x | {"tile": tile})
        pd.DataFrame(done).to_parquet(rec.dir / "partial_rows.parquet", index=False)
        logger.info("%s %s/%s D %.3e: %s in %d it (%s), clip %.4f, MSE %.3f, %.0f s", tile, x["solver"], x["form"], x["D"], x["outcome"],
                    x["n_iter"], x["stop_reason"], x["frac_clipped"], x["mse"], x["wall_s"])

    for solver, form, Dv in runs:
        r, _ = eval_row(f, td.Z, a_row, Dv, K, sc, solver, form, mp.nonlinear_max_iter, mp.nonlinear_tol, on_eval=_partial)
        rows += [x | {"tile": tile} for x in r]
        logger.info("%s pilot row A_crit %.1f S_c %.3f %s/%s: %s; mean wall %.1f s", tile, a_row, sc, solver, form,
                    pd.Series([x["outcome"] for x in r]).value_counts().to_dict(), float(np.mean([x["wall_s"] for x in r])))
        pd.DataFrame(rows).to_parquet(rec.dir / "partial_rows.parquet", index=False)
    df = pd.DataFrame(rows, columns=None if rows else ["tile", "solver", "form", "D", "outcome", "converged", "wall_s", "n_iter"])
    rec.write_table(df, "pilot_rows")
    out: dict[str, Any] = {"tile": tile, "part": part, "A_crit": a_row, "S_c": sc, "K": K, "linear_joint": st["linear_joint"], "D0": mp.D0, "D0_B": mp.D0_B,
                           "counts": {f"{s_}/{fm}": df[(df.solver == s_) & (df.form == fm)].outcome.value_counts().to_dict() for s_, fm, _ in runs}}
    if part == "centre":
        pic = df[df.solver == "picard"].set_index("D")
        nwc = df[df.solver == "newton"].set_index("D")
        pic_conv = [float(d) for d in pic.index[pic.converged.astype(bool)]]
        nw_out = dict(zip(nwc.index.astype(float), nwc.outcome.astype(str), strict=True))
        out["test4"] = {"picard_converged_D": pic_conv, "newton_outcome_where_picard_converged": {str(d): nw_out[d] for d in pic_conv},
                        "pass": bool(all(nw_out[d] == "converged" for d in pic_conv))}
    if part in ("centre", "test1"):
        test1 = []
        p6 = _phase6_nonlinear_joint_optimum(tile)
        if p6 is not None:
            o = p6["optimum"]
            fK = partitioned_threshold_model_nonlinear(td.Z, td.A, td.FD, td.dx, mp.m, mp.U, p6["K"])
            test1.append(_compare(fK, td.Z, float(o["A_crit"]), float(o["D"]), p6["K"], p6["S_c"], mp, f"{tile} full tile ({p6['run']})"))
            tc = load_tile(tile, crop=CROP, sanity=False)
            fc = partitioned_threshold_model_nonlinear(tc.Z, tc.A, tc.FD, tc.dx, mp.m, mp.U, p6["K"])
            test1.append(_compare(fc, tc.Z, float(o["A_crit"]), float(o["D"]), p6["K"], p6["S_c"], mp, f"{tile} crop {CROP}"))
        out["test1"] = test1
    out["timing"] = {f"{s_}/{fm}": {"n": int(len(g)), "wall_mean_s": float(g.wall_s.mean()), "wall_max_s": float(g.wall_s.max()),
                                    "iter_mean": float(g.n_iter.mean()), "wall_per_iter_s": float(g.wall_s.sum() / max(g.n_iter.sum(), 1))}
                     for (s_, fm), g in df.groupby(["solver", "form"])}
    out["wall_s"] = time.perf_counter() - t0
    rec.add_params("pilot", out)
    rec.finalize(config_snapshot(td.cfg, {"part": part}))
    return rec


def summarize_pilot(tiles: Sequence[str] = PILOT_TILES, run_id: str | None = None, results_root_=None) -> RunRecorder:
    """5.2: the core-hour estimate of section 3 from the flux pilot, and the 2.2 stop rule from test 4 (the fraction of
    the row's evaluations below the linear joint D that are ``no_steady_solution``)."""
    rec = RunRecorder("a9_pilot_summary", "_".join(tiles), run_id=run_id, root=results_root_)
    per: dict[str, Any] = {}
    for tile in tiles:
        for part in ("centre", "flux", "test1"):
            hit = None
            for r in find_runs("a9_pilot", f"{tile}_{part}"):
                hit = (r, json.loads((r / "params.json").read_text())["pilot"])
            if hit:
                per[f"{tile}/{part}"] = hit[1] | {"run": hit[0].name}
    stop: dict[str, Any] = {}
    for tile in tiles:
        c = per.get(f"{tile}/centre")
        if not c:
            continue
        df = pd.read_parquet(results_root() / c["run"] / "pilot_rows.parquet")
        low = df[(df.solver == "newton") & (df.D < float(c["linear_joint"]["D"]))]
        stop[tile] = {"n_low_D": int(len(low)), "n_no_steady": int((low.outcome == "no_steady_solution").sum()),
                      "frac_no_steady": float((low.outcome == "no_steady_solution").mean()) if len(low) else float("nan")}
        stop[tile]["most_no_steady"] = bool(stop[tile]["frac_no_steady"] > 0.5)
    est: dict[str, Any] = {}
    n_eval = len(SC_LABELS) * N_ACRIT * D_GRID[2]
    for tile in tiles:
        fx = per.get(f"{tile}/flux")
        if fx:
            w = fx["timing"]["newton/flux"]["wall_mean_s"]
            est[tile[:2]] = {"wall_per_evaluation_s": w, "core_hours_per_tile": n_eval * w / 3600.0, "core_hours_site": 5 * n_eval * w / 3600.0}
    total = float(sum(v["core_hours_site"] for v in est.values()))
    rec.add_params("per_task", per)
    rec.add_params("stop_rule_2_2", stop | {"applies": bool(stop) and all(v["most_no_steady"] for v in stop.values())})
    rec.add_params("estimate_section3", {"evaluations_per_tile": n_eval, "per_site": est, "total_core_hours": total, "limit_total": 900, "limit_per_submission": 300,
                                         "within_limits": bool(total <= 900)})
    rec.finalize({"tiles": list(tiles)})
    return rec


def run_slope_exceedance(tiles: Sequence[str], run_id: str | None = None, results_root_=None) -> RunRecorder:
    """Diagnostic for section 2 (not in the addendum's list; reported as such): the fraction of hillslope cells (A below
    the linear joint A_crit, tile edge excluded) whose gradient, from the models' slope stencil, exceeds 0.99 S_c (the
    clip) and 0.5 S_c, on the observed DEM and on the linear joint surface, at the three S_c of each tile."""
    rec = RunRecorder("a9_slope_exceedance", f"{len(tiles)}tiles", run_id=run_id, root=results_root_)
    out = []
    for t in tiles:
        st = tile_setup(t)
        td, mp = st["td"], st["mp"]
        scs = s_c_values(t)
        f = partitioned_threshold_model_nonlinear(td.Z, td.A, td.FD, td.dx, mp.m, mp.U, st["K"])
        gd = f.grid_data
        edge = np.zeros(td.A.shape, dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        hs2 = (td.A < float(st["linear_joint"]["A_crit"])) & ~edge
        fixed = ~hs2
        adj = np.zeros_like(fixed)
        adj[1:, :] |= fixed[:-1, :]
        adj[:-1, :] |= fixed[1:, :]
        adj[:, 1:] |= fixed[:, :-1]
        adj[:, :-1] |= fixed[:, 1:]
        hs = gd["idx_map"][hs2]
        zl = _grid_of(_latest_refined_partitioned(t), "Partitioned-model-a_crit-k-d-opt")
        for name, Z in (("observed", td.Z), ("linear_joint", zl)):
            s_ = f.compute_slopes(Z[gd["rev_j"], gd["rev_i"]], hs)
            row: dict[str, Any] = {"tile": t, "surface": name, "n_hillslope": int(hs.size), "slope_median": float(np.median(s_)),
                                   "slope_p90": float(np.percentile(s_, 90))}
            for lab, sc in scs.items():
                r = s_ / sc
                row[f"S_c_{lab}"] = sc
                row[f"frac_gt_0.99Sc_{lab}"] = float(np.mean(r > 0.99))
                row[f"frac_gt_0.5Sc_{lab}"] = float(np.mean(r > 0.5))
            row["frac_of_clipped_adjacent_to_fixed_central"] = float(np.mean(adj[hs2][s_ / scs["central"] > 0.99]))
            out.append(row)
        logger.info("%s: observed > 0.99 S_c %.3f, linear joint %.3f", t, out[-2]["frac_gt_0.99Sc_central"], out[-1]["frac_gt_0.99Sc_central"])
    rec.write_table(pd.DataFrame(out), "slope_exceedance")
    rec.add_params("rows", out)
    rec.finalize({"tiles": list(tiles)})
    return rec


# ----------------------------------------------------------------------------- 2.2 tests 2 and 3, recorded

def strip_profile(form: str, xi, L: float, U: float, D: float, S_c: float):
    """Steady 1-D profile of each form with the channel (z = 0) at distance L from the crest; xi = distance from the crest.
    flux: slope S_c (sqrt(1+u^2) - 1)/u with u = 2 U xi/(D S_c) (Roering et al. 2007); centre: slope S_c tanh(U xi/(D S_c))."""
    if form == "flux":
        c = D * S_c / (2 * U)

        def F(u):
            return np.sqrt(1 + u * u) - np.log(1 + np.sqrt(1 + u * u))

        return S_c * c * (F(L / c) - F(xi / c))
    a = U / (D * S_c)
    return (S_c / a) * (np.log(np.cosh(a * L)) - np.log(np.cosh(a * xi)))


def centre_form_R(E: float) -> float:
    """R*(E*) of the v1 (centre) form in 1-D: (2/E*) ln cosh(E*/2)."""
    return float((2.0 / E) * np.log(np.cosh(E / 2.0)))


def strip_R_star(form: str, E: float, dx: float, L: float = 10.0, U: float = 1e-4, S_c: float = 0.5, y_factor: float = 6.0) -> tuple[float, dict[str, Any]]:
    """R* at the middle row of a ridge strip solved by the 2-D model with Newton: channels (boundary, z = 0) on the two long
    edges at distance L from the crest, the short edges Dirichlet with the analytic profile of the form, ``y_factor`` x the
    ridge width long (so the middle row is 1-D)."""
    D = 2 * U * L / (E * S_c)
    N = int(round(L / dx))
    Nx, Ny = 2 * N + 1, int(round(y_factor * 2 * L / dx)) + 1
    xi = np.abs(np.arange(Nx) * dx - L)
    Z0 = np.zeros((Ny, Nx))
    Z0[0, :] = Z0[-1, :] = strip_profile(form, xi, L, U, D, S_c)
    Z0[:, 0] = Z0[:, -1] = 0.0
    A = np.ones((Ny, Nx))
    A[0, 0] = 10.0                                         # the forced outlet sits on the boundary
    FD = np.zeros((Ny, Nx), dtype=np.uint8)
    f = partitioned_threshold_model_nonlinear(Z0, A, FD, dx, 0.5, U, 1.0)
    Z, mask = f(1e12, D, S_c, solver="newton", form=form)
    if mask.sum() != 1:
        raise RuntimeError("strip: channel cells inside the strip")
    return float(Z[Ny // 2, N] / L / S_c), dict(f.last_outcome) | {"n_cells": int(Nx * Ny), "D": D}


def run_section2_tests(run_id: str | None = None, results_root_=None, dxs=(1.0, 0.5, 0.25), Es=(1.0, 5.0, 20.0)) -> RunRecorder:
    """2.2 tests 2 and 3 with their numbers recorded (the fast test suite asserts the same on coarser grids)."""
    from lemcvt.analyses.er_star import integrate_profile_R
    from lemcvt.models.common import precompute_channel_network
    from lemcvt.models.nonlinear_newton import HillslopeSystem

    rec = RunRecorder("a9_section2_tests", "strip_jacobian", run_id=run_id, root=results_root_)
    rows = []
    for form in ("flux", "centre"):
        for E in Es:
            ref_r, ref_c = integrate_profile_R(E), centre_form_R(E)
            for dx in dxs:
                t = time.perf_counter()
                R, o = strip_R_star(form, E, dx)
                rows.append({"form": form, "E_star": E, "dx": dx, "R_star": R, "R_roering": ref_r, "R_centre_form": ref_c,
                             "rel_err_vs_roering": (R - ref_r) / ref_r, "rel_err_vs_centre_form": (R - ref_c) / ref_c,
                             "outcome": o["outcome"], "frac_clipped": o["frac_clipped"], "n_iter": o["n_iter"], "n_cells": o["n_cells"], "wall_s": time.perf_counter() - t})
                logger.info("strip %s E* %g dx %g: R* %.6f (Roering %.6f, centre-form curve %.6f) %s", form, E, dx, R, ref_r, ref_c, o["outcome"])
    rec.write_table(pd.DataFrame(rows), "strip")
    # test 3 on a synthetic tile (as tests/test_addendum9.py), several finite-difference steps
    import sys as _sys

    from lemcvt.config import REPO_ROOT
    _sys.path.insert(0, str(REPO_ROOT / "tests"))
    from synthetic_tile import hydro_from_padded, synthetic_dem

    z = synthetic_dem(40, 40, 1.0, seed=3)
    zf, fd, area = hydro_from_padded(z, 1.0)
    U_, K_, M_ = 1e-4, 2e-4, 0.4
    A_crit, D, S_c = 200.0, 5e-3, 3.0
    f = partitioned_threshold_model_nonlinear(zf, area, fd, 1.0, M_, U_, K_)
    jac = []
    for form in ("centre", "flux"):
        f(A_crit, D, S_c, 200, 1e-12, form=form)
        gd = f.grid_data
        Zc, _d, k_out, bci, bcv = precompute_channel_network(gd, M_, U_, K_)
        cm = gd["A_flat"] >= A_crit
        cm[k_out] = True
        hm = ~cm
        hm[bci] = False
        hs = np.where(hm)[0]
        h2r = np.full(gd["n"], -1)
        h2r[hs] = np.arange(hs.size)
        sysm = HillslopeSystem(form, gd, hs, h2r, hm, D, S_c, Zc, cm, bcv, U_, f.compute_slopes)
        zz = f.last_z_flat.copy() + np.where(hm, np.random.RandomState(0).normal(0, 0.05, hm.size), 0.0)
        st = sysm.state(zz)
        J = sysm.jacobian(zz, st).toarray()
        big = np.abs(J) > 1e-3 * np.abs(J).max()
        for h in (1e-4, 1e-5, 1e-6):
            Jfd = np.zeros_like(J)
            for k, c in enumerate(hs):
                zp, zm = zz.copy(), zz.copy()
                zp[c] += h
                zm[c] -= h
                Jfd[:, k] = (sysm.state(zp)["F"] - sysm.state(zm)["F"]) / (2 * h)
            e = np.abs(J - Jfd)
            jac.append({"form": form, "h": h, "n_unknowns": int(hs.size), "max_slope_ratio": float(np.max(st["slopes"]) / S_c),
                        "max_err_over_max_J": float(e.max() / np.abs(J).max()), "max_entrywise_rel_err_above_1e-3": float(np.max(e[big] / np.abs(J)[big]))})
    rec.write_table(pd.DataFrame(jac), "jacobian")
    rec.add_params("strip", rows)
    rec.add_params("jacobian", jac)
    rec.finalize({"dxs": list(dxs), "Es": list(Es)})
    return rec


# ----------------------------------------------------------------------------- 3.1 the profile, in parts

def run_fair_part(tile: str, sc_label: str, split: tuple[int, int] | None = None, form: str = FORM, run_id: str | None = None,
                  results_root_=None, crop: tuple[int, int, int, int] | None = None, n_acrit: int | None = None,
                  d_values: np.ndarray | None = None, s_c: float | None = None) -> RunRecorder:
    st = tile_setup(tile, crop=crop)
    td, mp, K = st["td"], st["mp"], st["K"]
    S_c = float(s_c) if s_c is not None else s_c_values(tile)[sc_label]
    grid_full = st["grid_a"] if n_acrit is None else thin_log_uniform(st["grid_a"], n_acrit)
    grid_a = grid_full[split[0]::split[1]] if split else grid_full
    Dg = d_grid() if d_values is None else np.asarray(d_values, float)
    label = f"{tile}_{sc_label}_{form}" + (f"_split{split[0]}of{split[1]}" if split else "")
    rec = RunRecorder("a9_fair", label, run_id=run_id, root=results_root_)
    setup = {"tile": tile, "sc_label": sc_label, "S_c": S_c, "form": form, "solver": "newton", "K": K, "source_run": st["source_run"],
             "linear_joint_optimum": st["linear_joint"], "linear_joint_interval_5pct": st["interval_5pct"], "A_crit_grid_full": [float(a) for a in grid_full],
             "A_crit_grid": [float(a) for a in grid_a], "D_grid": [float(d) for d in Dg], "split": list(split) if split else None,
             "er_star_run": ER_STAR_RUN, "crop": crop, "D0": mp.D0, "D0_B": mp.D0_B, "U": mp.U, "m": mp.m}
    rec.add_params("setup", setup)
    f = partitioned_threshold_model_nonlinear(td.Z, td.A, td.FD, td.dx, mp.m, mp.U, K)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    t0 = time.perf_counter()
    for a_c in grid_a:
        r, kept = eval_row(f, td.Z, float(a_c), Dg, K, S_c, "newton", form, mp.nonlinear_max_iter, mp.nonlinear_tol, keep=True)
        rows += r
        conv = [x for x in r if x["outcome"] == "converged"]
        if conv:
            b = min(conv, key=lambda x: x["mse"])
            if best is None or b["mse"] < best["mse"]:
                best = b
                np.save(rec.dir / "best_surface.npy", kept[b["D"]])
        pd.DataFrame(rows).to_parquet(rec.dir / "partial_rows.parquet", index=False)        # 5.4: a timeout loses at most one row
        logger.info("%s A_crit %.1f: %s; best so far %s [%.0f s]", label, a_c, pd.Series([x["outcome"] for x in r]).value_counts().to_dict(),
                    None if best is None else (round(best["A_crit"], 1), f"{best['D']:.3e}", round(best["mse"], 3)), time.perf_counter() - t0)
    rec.write_table(pd.DataFrame(rows), "fair_rows")
    rec.add_params("best_converged", best)
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.finalize(config_snapshot(td.cfg, {"crop": crop, "sc_label": sc_label, "form": form, "split": split}))
    return rec


def _part_runs(tile: str, sc_label: str, form: str = FORM) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for r in find_runs("a9_fair"):
        try:
            p = json.loads((r / "params.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        s = p["setup"]
        if s["tile"] == tile and s["sc_label"] == sc_label and s["form"] == form and not s.get("crop"):
            out.append((r, p))
    return out


def merge_fair(tile: str, sc_label: str, form: str = FORM, run_id: str | None = None, results_root_=None, save_grid: bool = True) -> RunRecorder:
    """Merge the newest complete set of parts of (tile, S_c label): the optimum over converged evaluations, the contours,
    the outcome counts, D/D0 and the MSE against the linear joint."""
    parts = _part_runs(tile, sc_label, form)
    if not parts:
        raise FileNotFoundError(f"{tile} {sc_label}: no a9_fair parts")
    n = max((p["setup"]["split"] or [0, 1])[1] for _, p in parts)
    group: dict[int, tuple[Path, dict[str, Any]]] = {}
    for r, p in parts:
        sp = p["setup"]["split"] or [0, 1]
        if sp[1] == n:
            group[sp[0]] = (r, p)                       # newest last
    missing = [k for k in range(n) if k not in group]
    if missing:
        raise FileNotFoundError(f"{tile} {sc_label}: parts {missing} of {n} missing")
    frames = [pd.read_parquet(group[k][0] / "fair_rows.parquet") for k in range(n)]
    df = pd.concat(frames, ignore_index=True).sort_values(["A_crit", "D"]).reset_index(drop=True)
    s0 = group[0][1]["setup"]
    rec = RunRecorder("a9_fair_merged", f"{tile}_{sc_label}_{form}", run_id=run_id, root=results_root_)
    rec.write_table(df, "fair_rows")
    counts = df.outcome.value_counts().to_dict()
    n_tot = int(len(df))
    conv = df[df.outcome == "converged"]
    lin = s0["linear_joint_optimum"]
    out: dict[str, Any] = {"tile": tile, "sc_label": sc_label, "S_c": s0["S_c"], "form": form, "K": s0["K"], "n_evaluations": n_tot,
                           "counts": {k: int(counts.get(k, 0)) for k in ("converged", "no_steady_solution", "not_converged")},
                           "frac_not_converged": float(counts.get("not_converged", 0) / n_tot) if n_tot else float("nan"),
                           "linear_joint": lin, "parts": {str(k): group[k][0].name for k in range(n)}, "D0": s0["D0"], "D0_B": s0["D0_B"]}
    out["untested"] = bool(out["frac_not_converged"] > RULE3["untested_not_converged_frac"])
    if len(conv):
        b = conv.iloc[int(np.argmin(conv.mse.to_numpy()))].to_dict()
        out["optimum"] = b
        out["mse_ratio_to_linear_joint"] = float(b["mse"] / float(lin["mse"]))
        out["D_over_D0_A"] = float(b["D"] / s0["D0"])
        out["D_over_D0_B"] = float(b["D"] / s0["D0_B"]) if s0.get("D0_B") else None
        out["contours"] = _contours(cast(list[dict[str, Any]], conv.to_dict("records")))
        grid_a = np.array(s0["A_crit_grid_full"])
        grid_d = np.array(s0["D_grid"])
        out["on_grid_edge"] = {"A_crit": bool(np.isclose(b["A_crit"], grid_a[[0, -1]]).any()), "D": bool(np.isclose(b["D"], grid_d[[0, -1]]).any())}
        if save_grid:
            src = [group[k][0] for k in range(n) if group[k][1].get("best_converged") and np.isclose(group[k][1]["best_converged"]["mse"], b["mse"])]
            if src and (src[0] / "best_surface.npy").exists():
                td = load_tile(tile)
                Zb = np.load(src[0] / "best_surface.npy")
                out["grid_check_mse"] = mse(Zb, td.Z)
                out["files"] = write_model_outputs(rec, td.filled, Zb, f"{td.cfg.site}_1m_best_tile_Partitioned-model-nonlinear-flux-joint-opt", derived=False)
    rec.add_params("merged", out)
    rec.finalize({"tile": tile, "sc_label": sc_label, "form": form, "parts": out["parts"]})
    return rec


# ----------------------------------------------------------------------------- 3.2 the rule

def _latest_merged(tile: str, sc_label: str, form: str = FORM) -> tuple[Path, dict[str, Any]] | None:
    hit = None
    for r in find_runs("a9_fair_merged", f"{tile}_{sc_label}_{form}"):
        hit = (r, json.loads((r / "params.json").read_text())["merged"])
    return hit


def rule3_tile(m: dict[str, Any], site: str) -> dict[str, Any]:
    if m.get("untested") or "optimum" not in m:
        return {"fit": False, "D": False, "pass": False, "untested": bool(m.get("untested")), "reason": "untested" if m.get("untested") else "no converged evaluation"}
    fit = bool(m["mse_ratio_to_linear_joint"] <= RULE3["mse_ratio_max"])
    lo, hi = RULE3["D_ratio"]
    D = float(m["optimum"]["D"])
    if site == "AP":
        a, b_ = sorted([m["D0"], m["D0_B"]])
        dok = bool(a / RULE3["AP_bracket_widen"] <= D <= b_ * RULE3["AP_bracket_widen"])
    else:
        dok = bool(lo <= m["D_over_D0_A"] <= hi)
    return {"fit": fit, "D": dok, "pass": bool(fit and dok), "untested": False}


def summarize_fair(tiles: Sequence[str], run_id: str | None = None, results_root_=None) -> RunRecorder:
    rec = RunRecorder("a9_fair_summary", f"{len(tiles)}tiles", run_id=run_id, root=results_root_)
    table = []
    per: dict[str, Any] = {}
    for tile in tiles:
        per[tile] = {}
        for lab in SC_LABELS:
            hit = _latest_merged(tile, lab)
            if hit is None:
                continue
            r, m = hit
            v = rule3_tile(m, tile[:2])
            per[tile][lab] = m | {"rule": v, "run": r.name}
            o = m.get("optimum") or {}
            c = m.get("contours") or {}
            table.append({"tile": tile, "sc_label": lab, "S_c": m["S_c"], "A_crit": o.get("A_crit"), "D": o.get("D"), "mse": o.get("mse"),
                          "linear_joint_mse": float(m["linear_joint"]["mse"]), "mse_ratio": m.get("mse_ratio_to_linear_joint"),
                          "D_over_D0_A": m.get("D_over_D0_A"), "D_over_D0_B": m.get("D_over_D0_B"), **{f"n_{k}": v_ for k, v_ in m["counts"].items()},
                          "frac_gt_0.5Sc": o.get("frac_hillslope_gt_0.5Sc"), "frac_clipped": o.get("frac_clipped"),
                          "contour1_A": [c.get("contour_0.01", {}).get("A_crit_min"), c.get("contour_0.01", {}).get("A_crit_max")] if c else None,
                          "contour1_D": [c.get("contour_0.01", {}).get("D_min"), c.get("contour_0.01", {}).get("D_max")] if c else None,
                          "contour5_A": [c.get("contour_0.05", {}).get("A_crit_min"), c.get("contour_0.05", {}).get("A_crit_max")] if c else None,
                          "contour5_D": [c.get("contour_0.05", {}).get("D_min"), c.get("contour_0.05", {}).get("D_max")] if c else None,
                          "edge": m.get("on_grid_edge"), "fit": v["fit"], "D_ok": v["D"], "pass": v["pass"], "untested": v["untested"], "run": r.name})
    site_rule: dict[str, Any] = {}
    for site in ("AP", "GM"):
        site_rule[site] = {}
        for lab in SC_LABELS:
            vs = [per[t][lab]["rule"] for t in tiles if t.startswith(site) and lab in per[t]]
            if not vs:
                continue
            n_pass = sum(v["pass"] for v in vs)
            n_d = sum(v["D"] for v in vs)
            verdict = "preferred" if n_pass >= RULE3["min_tiles"] else ("closer D at a cost in fit" if n_d >= RULE3["min_tiles"] else "not preferred")
            site_rule[site][lab] = {"n_tiles": len(vs), "n_pass": n_pass, "n_fit": sum(v["fit"] for v in vs), "n_D": n_d,
                                    "n_untested": sum(v["untested"] for v in vs), "verdict": verdict}
    rec.write_table(pd.DataFrame(table), "fair_summary")
    rec.add_params("per_tile", per)
    rec.add_params("site_rule", site_rule)
    rec.add_params("rule", RULE3)
    rec.finalize({"tiles": list(tiles)})
    return rec


# ----------------------------------------------------------------------------- 4 the morphology score

def _phase6_runs(tile: str) -> dict[str, Path]:
    """The Phase 6 surfaces: linear joint (refined partitioned run), AD optimum, nonlinear joint at S_c = 1.27."""
    out: dict[str, Path] = {"linear_joint": _latest_refined_partitioned(tile)}
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
        if "ad" in p.get("v2", {}) and list(r.glob("*_AD-opt_elevation.tif")):
            out["ad"] = r
    p6 = _phase6_nonlinear_joint_optimum(tile)
    if p6 is not None:
        out["nonlinear_joint_1.27"] = p6["dir"]
    return out


def _grid_of(run: Path, suffix: str) -> np.ndarray:
    hits = sorted(run.glob(f"*_{suffix}_elevation.tif"))
    if not hits:
        raise FileNotFoundError(f"{run}: no *_{suffix}_elevation.tif")
    return read_geotiff(hits[-1]).data.astype(float)


def morphology_tile(tile: str, er_run: str = ER_STAR_RUN, with_a10_flux: bool = True) -> tuple[dict[str, Any], pd.DataFrame]:
    """Section 4.2 on one tile: every surface against the observed DEM on the same option (iii) pixels and D8 paths."""
    from lemcvt.analyses.er_star import trace_to_channel
    from lemcvt.measure.hilltop_curvature import total_curvature
    from lemcvt.measure.phase6_inputs import W_F

    td = load_tile(tile)
    mp = td.cfg.model
    Z0, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    pix = pd.read_parquet(results_root() / er_run / "hilltop_pixels.parquet")
    pix = pix[(pix.tile == tile) & (pix.network == NETWORK)]
    rows, cols = pix.row.to_numpy(), pix.col.to_numpy()
    used = pix.used.to_numpy().astype(bool)
    tr = trace_to_channel(Z0, A, FD, dx, dy, rows, cols, 1e3)
    assert np.allclose(tr["L_H"], pix.L_H.to_numpy())                     # the same D8 paths as addendum 8
    er = json.loads((results_root() / er_run / "params.json").read_text())["tiles"][tile]["networks"][NETWORK]
    S_c_obs = float(er["S_c_fit"]["S_c"])
    runs = _phase6_runs(tile)
    surfaces: dict[str, tuple[np.ndarray, dict[str, Any]]] = {"observed": (Z0, {})}
    jo = json.loads((runs["linear_joint"] / "params.json").read_text())["v2"]["partitioned"]["joint_optimum"]
    surfaces["linear_joint"] = (_grid_of(runs["linear_joint"], "Partitioned-model-a_crit-k-d-opt"), {"run": runs["linear_joint"].name, "A_crit": jo["A_crit"], "D": jo["D"], "K": jo["K"], "mse": jo["mse"]})
    if "ad" in runs:
        o = json.loads((runs["ad"] / "params.json").read_text())["v2"]["ad"]["optimum"]
        surfaces["ad"] = (_grid_of(runs["ad"], "AD-opt"), {"run": runs["ad"].name, "A_crit": jo["A_crit"], "D": o["D"], "K": o["K"], "mse": o["mse"],
                                                           "hillslope_cells": "those of the linear joint optimum (AD has no partition)"})
    if "nonlinear_joint_1.27" in runs:
        p6 = _phase6_nonlinear_joint_optimum(tile)
        assert p6 is not None
        o = p6["optimum"]
        surfaces["nonlinear_joint_1.27"] = (_grid_of(p6["dir"], "Partitioned-model-nonlinear-joint-opt"), {"run": p6["run"], "A_crit": o["A_crit"], "D": o["D"], "K": p6["K"], "mse": o["mse"], "S_c": p6["S_c"],
                                                                                                         "label": CENTRE_FORM_LABEL})
    hit = _latest_merged(tile, "central")
    if hit is not None and hit[1].get("files"):
        r, m = hit
        surfaces["nonlinear_flux_fitted_Sc"] = (_grid_of(r, "Partitioned-model-nonlinear-flux-joint-opt"), {"run": r.name, "A_crit": m["optimum"]["A_crit"], "D": m["optimum"]["D"],
                                                                                                          "K": m["K"], "mse": m["optimum"]["mse"], "S_c": m["S_c"]})
    if with_a10_flux:
        # addendum 10, section 3: the section 2 best flux-form surface of each S_c on GM, where one exists
        from lemcvt.analyses.gm_flux import SC_LABELS as A10_SC
        from lemcvt.analyses.gm_flux import latest_merged as a10_latest

        for lab in A10_SC:
            h10 = a10_latest(tile, lab)
            if h10 is None or not h10[1].get("files"):
                continue
            r10, m10 = h10
            surfaces[f"flux_a10_Sc{lab}"] = (_grid_of(r10, f"Partitioned-model-nonlinear-flux-Sc{lab}-opt"),
                                             {"run": r10.name, "A_crit": m10["optimum"]["A_crit"], "D": m10["optimum"]["D"], "K": m10["K"],
                                              "mse": m10["optimum"]["mse"], "S_c": m10["S_c"], "label": "flux form, div(D_eff grad z) = -U"})
    f = partitioned_threshold_model_nonlinear(Z0, A, FD, dx, mp.m, mp.U, float(jo["K"]))
    gd = f.grid_data
    edge = np.zeros(Z0.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    info: dict[str, Any] = {"tile": tile, "S_c_observed_fit": S_c_obs, "n_pixels": int(rows.size), "n_used": int(used.sum()), "surfaces": {}}
    frames = []
    obs: dict[str, Any] = {}
    for name, (Z, meta) in surfaces.items():
        C = -total_curvature(Z, W_F, dx, dy)[rows, cols]
        R = Z[rows, cols] - Z[tr["end_row"], tr["end_col"]]
        L = tr["L_H"]
        S = R / np.where(L > 0, L, np.nan)
        ok = used & (C > 0) & (R > 0)
        ratio = np.where(ok, 2.0 * S / (C * L), np.nan)
        A_c = float(meta.get("A_crit", jo["A_crit"]))
        hs2d = np.isfinite(A) & (A < A_c) & ~edge
        rec_s: dict[str, Any] = dict(meta)
        rec_s.update({"C_ht_median": float(np.nanmedian(C[used])), "ratio_median": float(np.nanmedian(ratio)), "n_ratio": int(np.isfinite(ratio).sum()),
                      "n_C_nonpositive": int(np.sum(used & ~(C > 0))), "n_R_nonpositive": int(np.sum(used & ~(R > 0))), "hillslope_A_crit": A_c})
        # gradients on this model's hillslope cells, observed and model, with the model's slope stencil
        hs_flat = gd["idx_map"][hs2d]
        g_mod = f.compute_slopes(Z[gd["rev_j"], gd["rev_i"]], hs_flat)
        g_obs = f.compute_slopes(Z0[gd["rev_j"], gd["rev_i"]], hs_flat)
        rec_s.update({"grad_median": float(np.median(g_mod)), "grad_p90": float(np.percentile(g_mod, 90)),
                      "grad_median_obs": float(np.median(g_obs)), "grad_p90_obs": float(np.percentile(g_obs, 90)), "n_hillslope_cells": int(hs_flat.size)})
        if name == "observed":
            obs = rec_s
        else:
            rec_s["mse_check"] = mse(Z, Z0)
            q = {"C_ht": rec_s["C_ht_median"] / obs["C_ht_median"], "ER_ratio": rec_s["ratio_median"] / obs["ratio_median"],
                 "grad_median": rec_s["grad_median"] / rec_s["grad_median_obs"], "grad_p90": rec_s["grad_p90"] / rec_s["grad_p90_obs"]}
            lim = math.log10(RULE4["factor"])
            within = {k: bool(abs(math.log10(v)) <= lim) if v > 0 and np.isfinite(v) else False for k, v in q.items()}
            rec_s["ratios_model_over_observed"] = q
            rec_s["within_1.25"] = within
            rec_s["reproduces_form"] = bool(within["C_ht"] and within["ER_ratio"] and within["grad_median"] and within["grad_p90"])
        info["surfaces"][name] = rec_s
        frames.append(pd.DataFrame({"tile": tile, "surface": name, "row": rows, "col": cols, "used": used, "C_ht": C, "L_H": L, "R": R, "Sbar": S,
                                    "E_star_at_obs_Sc": 2.0 * C * L / S_c_obs, "R_star_at_obs_Sc": S / S_c_obs, "ratio_normalised": ratio}))
    return info, pd.concat(frames, ignore_index=True)


def run_morphology(tiles: Sequence[str], run_id: str | None = None, results_root_=None, with_a10_flux: bool = True) -> RunRecorder:
    rec = RunRecorder("a9_morphology", f"{len(tiles)}tiles" if len(tiles) > 3 else "_".join(tiles), run_id=run_id, root=results_root_)
    t0 = time.perf_counter()
    per: dict[str, Any] = {}
    frames = []
    for tile in tiles:
        info, df = morphology_tile(tile, with_a10_flux=with_a10_flux)
        per[tile] = info
        frames.append(df)
        logger.info("%s morphology: %s", tile, {k: v.get("reproduces_form") for k, v in info["surfaces"].items()})
    rec.write_table(pd.concat(frames, ignore_index=True), "morphology_pixels")
    counts: dict[str, Any] = {}
    for site in ("AP", "GM"):
        counts[site] = {}
        for t in [t for t in tiles if t.startswith(site)]:
            for name, s in per[t]["surfaces"].items():
                if name == "observed":
                    continue
                c = counts[site].setdefault(name, {"n_tiles": 0, "n_pass": 0, "tiles_pass": []})
                c["n_tiles"] += 1
                if s["reproduces_form"]:
                    c["n_pass"] += 1
                    c["tiles_pass"].append(t)
    rec.add_params("per_tile", per)
    rec.add_params("counts", counts)
    rec.add_params("rule", RULE4)
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.add_params("labels", {"nonlinear_joint_1.27": CENTRE_FORM_LABEL})
    rec.finalize({"tiles": list(tiles), "er_star_run": ER_STAR_RUN, "with_a10_flux": with_a10_flux})
    return rec
