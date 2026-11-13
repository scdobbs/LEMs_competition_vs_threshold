"""Addendum 10, section 2: a bounded nonlinear test on Gabilan Mesa (flux form, S_c free).

On GM1-GM5 at the site m: the flux form div(D_eff grad z) = -U solved by Newton with continuation in D (addendum 9, N11),
K fixed at the linear joint optimum, four values of S_c (1.25, 0.9, the addendum 8 primary-network fit and its 84th
percentile), 14 log-spaced D from 3e-3 to 5e-2 m^2/yr and 12 (or 8, the 2.2 fallback) log-spaced unique areas spanning the
linear joint 5 % interval widened x2 on each side.

Outcome classes (addendum 10, 1.4, which replaces the 1 % rule of addendum 9 here): an evaluation is ``converged`` when the
Newton stopping test held and fewer than 1 % of the hillslope cells are clipped (|grad z|/S_c > 0.99), counting only hillslope
cells that are not 4-adjacent to a channel or boundary cell; ``no_steady_solution`` when that fraction is 1 % or more;
``not_converged`` otherwise. The addendum 9 class (every hillslope cell counted) is recorded beside it as ``outcome_a9``.

Stop rule (2.2.4): within one tile and S_c, once more than 30 % of the evaluations so far are ``not_converged`` (counted over
every part of the group), the part stops after its current A_crit row. Partial rows are written after every A_crit row.
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

from lemcvt.analyses.nonlinear_fair import s_c_values, tile_setup
from lemcvt.analyses.optimizer_v2 import _contours
from lemcvt.analyses.outputs import write_model_outputs
from lemcvt.analyses.tile_pipeline import mse
from lemcvt.config import config_snapshot, load_tile, results_root
from lemcvt.models.nonlinear_newton import CLIP
from lemcvt.models.partitioned_nonlinear import partitioned_threshold_model_nonlinear
from lemcvt.optimize.v2 import thin_log_uniform, unique_areas
from lemcvt.results import RunRecorder, dump_json, find_runs

logger = logging.getLogger(__name__)

TILES = ("GM1", "GM2", "GM3", "GM4", "GM5")
SC_FIXED = {"1.25": 1.25, "0.9": 0.9}
SC_LABELS = ("1.25", "0.9", "central", "p84")
D_GRID = (3e-3, 5e-2, 14)
N_ACRIT = 12
N_ACRIT_FALLBACK = 8
WIDEN = 2.0
FORM = "flux"
CLIP_FRAC = 0.01
STOP_NOT_CONVERGED_FRAC = 0.30
RULE: dict[str, Any] = {"mse_ratio_max": 1.05, "D_ratio": (0.67, 1.5), "min_tiles": 4, "S_c_min_preferred": 0.9}
BUDGET: dict[str, Any] = {"total_max": 600.0, "per_submission_max": 300.0}
PILOT_TILES = ("GM1", "GM3")
PILOT_SC = "1.25"


def d_grid() -> np.ndarray:
    lo, hi, n = D_GRID
    return np.logspace(math.log10(lo), math.log10(hi), n)


def s_c_value(tile: str, label: str) -> float:
    if label in SC_FIXED:
        return SC_FIXED[label]
    return s_c_values(tile)[label]


def acrit_grid(st: dict[str, Any], n_acrit: int = N_ACRIT) -> np.ndarray:
    """``n_acrit`` log-spaced unique drainage areas spanning the linear joint 5 % interval widened x2 on each side."""
    lo_i, hi_i = st["interval_5pct"]
    return thin_log_uniform(unique_areas(st["td"].A, lo_i / WIDEN, hi_i * WIDEN), n_acrit)


def pilot_row(st: dict[str, Any], n_acrit: int = N_ACRIT) -> int:
    """The index of the grid value nearest the linear joint A_crit (in log): the pilot's one A_crit row."""
    g = acrit_grid(st, n_acrit)
    return int(np.argmin(np.abs(np.log(g / float(st["linear_joint"]["A_crit"])))))


def adjacency_to_fixed(hillslope_2d: np.ndarray) -> np.ndarray:
    """Hillslope cells with a 4-neighbour that is a channel or boundary cell (every non-hillslope cell of a tile is one)."""
    fixed = ~hillslope_2d
    adj = np.zeros_like(fixed)
    adj[1:, :] |= fixed[:-1, :]
    adj[:-1, :] |= fixed[1:, :]
    adj[:, 1:] |= fixed[:, :-1]
    adj[:, :-1] |= fixed[:, 1:]
    return adj & hillslope_2d


def classify_a10(converged: bool, frac_clipped_interior: float) -> str:
    if frac_clipped_interior >= CLIP_FRAC:
        return "no_steady_solution"
    return "converged" if converged else "not_converged"


def _row(f, Z, Z0, A_crit, D, K, S_c, wall) -> dict[str, Any]:
    out = f.last_outcome
    gd = f.grid_data
    hs = f.last_hillslope_indices
    nw = f.last_newton or {}
    ratio = f.compute_slopes(f.last_z_flat, hs) / S_c if hs.size else np.zeros(0)
    hs2d = np.zeros(Z0.shape, dtype=bool)
    hs2d[gd["rev_j"][hs], gd["rev_i"][hs]] = True
    adj2d = adjacency_to_fixed(hs2d)
    adj = adj2d[gd["rev_j"][hs], gd["rev_i"][hs]]
    clipped = ratio > CLIP
    interior = ~adj
    frac_int = float(np.mean(clipped[interior])) if interior.any() else 0.0
    return {"A_crit": float(A_crit), "D": float(D), "K": float(K), "S_c": float(S_c), "solver": out["solver"], "form": out["form"],
            "mse": mse(Z, Z0), "outcome": classify_a10(bool(out["converged"]), frac_int), "outcome_a9": out["outcome"],
            "converged": bool(out["converged"]), "n_iter": out["n_iter"], "stop_reason": out["stop_reason"], "rel_residual": nw.get("rel_residual"),
            "frac_clipped": out["frac_clipped"], "frac_clipped_interior": frac_int, "n_clipped": int(clipped.sum()),
            "n_clipped_adjacent": int((clipped & adj).sum()), "n_hillslope": int(hs.size), "n_hillslope_adjacent": int(adj.sum()),
            "frac_hillslope_gt_0.5Sc": float(np.mean(ratio > 0.5)) if ratio.size else None,
            "frac_hillslope_gt_0.9Sc": float(np.mean(ratio > 0.9)) if ratio.size else None,
            "max_slope_ratio": float(np.max(ratio)) if ratio.size else None, "initial": nw.get("initial", "linear"), "wall_s": float(wall)}


# ----------------------------------------------------------------------------- 2.1 one part

def _group_dir(group: str) -> Path:
    d = results_root() / "a10_flux_groups" / group
    d.mkdir(parents=True, exist_ok=True)
    return d


def _group_counts(group: str, tile: str, sc_label: str) -> dict[str, int]:
    frames = [pd.read_parquet(p) for p in sorted(_group_dir(group).glob(f"{tile}_{sc_label}_part*.parquet"))]
    if not frames:
        return {"n": 0, "not_converged": 0}
    df = pd.concat(frames, ignore_index=True)
    return {"n": int(len(df)), "not_converged": int((df.outcome == "not_converged").sum())}


def run_flux_part(tile: str, sc_label: str, split: tuple[int, int] | None = None, rows: Sequence[int] | str | None = None,
                  n_acrit: int = N_ACRIT, group: str | None = None, run_id: str | None = None, results_root_=None,
                  crop: tuple[int, int, int, int] | None = None, d_values: np.ndarray | None = None) -> RunRecorder:
    """One part of the (A_crit, D) profile at one S_c: the ``split`` k-th of n interleaved subsets of the A_crit grid, or the
    grid indices ``rows`` (``"pilot"``: the row nearest the linear joint A_crit). Newton on the flux form with continuation in D, largest D first."""
    st = tile_setup(tile, crop=crop)
    td, mp, K = st["td"], st["mp"], st["K"]
    S_c = s_c_value(tile, sc_label)
    grid_full = acrit_grid(st, n_acrit)
    idx = list(range(grid_full.size))
    if rows == "pilot":
        idx = [pilot_row(st, n_acrit)]
    elif rows is not None:
        idx = [int(i) for i in rows]
    elif split:
        idx = idx[split[0]::split[1]]
    grid_a = grid_full[idx]
    Dg = np.sort(d_grid() if d_values is None else np.asarray(d_values, float))[::-1]
    label = f"{tile}_{sc_label}" + (f"_split{split[0]}of{split[1]}" if split else "") + ("_rows" + "-".join(map(str, idx)) if rows is not None else "")
    rec = RunRecorder("a10_flux", label, run_id=run_id, root=results_root_)
    grp = group or rec.dir.name
    setup = {"tile": tile, "sc_label": sc_label, "S_c": S_c, "form": FORM, "solver": "newton", "K": K, "source_run": st["source_run"],
             "linear_joint_optimum": st["linear_joint"], "linear_joint_interval_5pct": st["interval_5pct"], "widen": WIDEN, "n_acrit": n_acrit,
             "A_crit_grid_full": [float(a) for a in grid_full], "A_crit_rows": idx, "A_crit_grid": [float(a) for a in grid_a],
             "D_grid": [float(d) for d in Dg], "split": list(split) if split else None, "group": grp, "crop": crop, "m": mp.m, "U": mp.U,
             "D0": mp.D0, "D0_B": mp.D0_B, "clip_rule": {"clip": CLIP, "frac": CLIP_FRAC, "exclude": "hillslope cells 4-adjacent to a channel or boundary cell"},
             "stop_rule": {"not_converged_frac": STOP_NOT_CONVERGED_FRAC, "scope": "tile and S_c, all parts of the group"}}
    rec.add_params("setup", setup)
    f = partitioned_threshold_model_nonlinear(td.Z, td.A, td.FD, td.dx, mp.m, mp.U, K)
    out_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    stopped: dict[str, Any] | None = None
    gfile = _group_dir(grp) / f"{tile}_{sc_label}_part{split[0] if split else 0}.parquet"
    t0 = time.perf_counter()
    for k, a_c in zip(idx, grid_a, strict=True):
        z_init = None
        kept: dict[float, np.ndarray] = {}
        for D in Dg:
            t1 = time.perf_counter()
            Z, _mask = f(float(a_c), float(D), S_c, solver="newton", form=FORM, z_init=z_init)
            r = _row(f, Z, td.Z, a_c, D, K, S_c, time.perf_counter() - t1) | {"tile": tile, "sc_label": sc_label, "A_crit_index": k}
            out_rows.append(r)
            logger.info("%s S_c %s (%.3f) A_crit %.1f D %.3e: %s (a9 %s) in %d it (%s), clip %.4f / interior %.4f, MSE %.3f, %.0f s", tile, sc_label, S_c,
                        a_c, D, r["outcome"], r["outcome_a9"], r["n_iter"], r["stop_reason"], r["frac_clipped"], r["frac_clipped_interior"], r["mse"], r["wall_s"])
            if r["outcome"] == "converged":
                z_init = f.last_z_flat.copy()
                kept[float(D)] = Z
        row_conv = [x for x in out_rows if x["A_crit_index"] == k and x["outcome"] == "converged"]
        if row_conv:
            b = min(row_conv, key=lambda x: x["mse"])
            if best is None or b["mse"] < best["mse"]:
                best = b
                np.save(rec.dir / "best_surface.npy", kept[b["D"]])
        df = pd.DataFrame(out_rows)
        df.to_parquet(rec.dir / "partial_rows.parquet", index=False)        # 2.2.3: a timeout loses at most one row
        df.to_parquet(gfile, index=False)
        # params.json after every row too, so a part that hits its time limit can still be merged (status "partial")
        dump_json(rec.params | {"best_converged": best, "status": "partial", "rows_done": int(df.A_crit_index.nunique())}, rec.dir / "params.json")
        g = _group_counts(grp, tile, sc_label)
        logger.info("%s S_c %s row %d done: %s; group not_converged %d of %d; best %s [%.0f s]", tile, sc_label, k,
                    df[df.A_crit_index == k].outcome.value_counts().to_dict(), g["not_converged"], g["n"],
                    None if best is None else (round(best["A_crit"], 1), f"{best['D']:.3e}", round(best["mse"], 3)), time.perf_counter() - t0)
        if g["n"] and g["not_converged"] / g["n"] > STOP_NOT_CONVERGED_FRAC:
            stopped = {"after_row": k, "group_counts": g}
            logger.warning("%s S_c %s: stop rule 2.2.4 (%d of %d evaluations not_converged)", tile, sc_label, g["not_converged"], g["n"])
            break
    rec.write_table(pd.DataFrame(out_rows), "flux_rows")
    rec.add_params("best_converged", best)
    rec.add_params("stopped", stopped)
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.finalize(config_snapshot(td.cfg, {"crop": crop, "sc_label": sc_label, "form": FORM, "split": split, "rows": idx, "group": grp}))
    return rec


# ----------------------------------------------------------------------------- 2.2 pilot and estimate

def estimate(pilot_runs: Sequence[str] | None = None, n_acrit: int = N_ACRIT, factor: float = 1.5, n_split: int = 2,
             run_id: str | None = None, results_root_=None) -> RunRecorder:
    """2.2.2: the core-hour estimate from the pilot rows (S_c = 1.25 on GM1 and GM3) and, for the lower S_c, from the cost of
    a Newton iteration in the pilot times the iterations those S_c are expected to need (stated in the output)."""
    rec = RunRecorder("a10_flux_estimate", f"n{n_acrit}", run_id=run_id, root=results_root_)
    frames = []
    for r in find_runs("a10_flux"):
        p = json.loads((r / "params.json").read_text()) if (r / "params.json").exists() else None
        if not p or p["setup"]["tile"] not in PILOT_TILES or p["setup"]["sc_label"] != PILOT_SC or p["setup"]["split"]:
            continue
        if pilot_runs and r.name not in pilot_runs:
            continue
        frames.append(pd.read_parquet(r / "flux_rows.parquet").assign(run=r.name))
    if not frames:
        raise FileNotFoundError("no a10_flux pilot runs")
    pil = pd.concat(frames, ignore_index=True)
    per_eval = float(pil.wall_s.mean())
    per_iter = float(pil.wall_s.sum() / max(pil.n_iter.sum(), 1))
    it_mean = float(pil.n_iter.mean())
    # expected iterations per evaluation at the lower S_c: the addendum 9 flux pilot on GM1 at the fitted S_c (0.544) needed
    # 11-22 iterations from the linear solution; with continuation in D 15 is used for 0.9, central and p84 alike, and the
    # iteration cap (50) for tile/S_c pairs whose observed hillslope cells already exceed the clip widely (GM3, S_c ~ 0.26-0.29)
    n_d = D_GRID[2]
    tasks = []
    for t in TILES:
        for lab in SC_LABELS:
            sc = s_c_value(t, lab)
            its = it_mean if lab == "1.25" else (50.0 if sc < 0.35 else max(15.0, it_mean))
            w = per_iter * its
            ch = n_acrit * n_d * w / 3600.0
            tasks.append({"tile": t, "sc_label": lab, "S_c": sc, "iterations_assumed": its, "wall_per_eval_s": w, "core_hours_expected": ch,
                          "array": f"{lab}_cap" if its >= 50 else lab})
    tdf = pd.DataFrame(tasks)
    # one array per S_c label (tile/S_c pairs expected at the cap in their own array); one time limit per array, so the
    # limits charged are n_tasks x n_split x the longest part of the array
    subs: dict[str, dict[str, Any]] = {}
    for a, g in tdf.groupby("array", sort=False):
        t_h = max(1, math.ceil(factor * float(g.core_hours_expected.max()) / n_split))
        subs[str(a)] = {"tiles": list(g.tile), "sc_label": str(g.sc_label.iloc[0]), "n_tasks": int(len(g) * n_split), "time_limit_h": t_h,
                        "expected": float(g.core_hours_expected.sum()), "limits": float(len(g) * n_split * t_h)}
    out: dict[str, Any] = {"n_acrit": n_acrit, "n_d": n_d, "evaluations_per_tile_and_S_c": n_acrit * n_d, "pilot_wall_per_eval_s": per_eval, "pilot_wall_per_iter_s": per_iter,
           "pilot_iter_mean": it_mean, "time_factor": factor, "parts_per_task": n_split,
           "total_expected": float(tdf.core_hours_expected.sum()), "total_limits": float(sum(v["limits"] for v in subs.values())),
           "submissions": subs, "budget": BUDGET}
    out["within_budget"] = bool(out["total_limits"] <= BUDGET["total_max"] and all(v["limits"] <= BUDGET["per_submission_max"] for v in subs.values()))
    rec.write_table(tdf, "estimate_tasks")
    rec.write_table(pil, "pilot_rows")
    rec.add_params("estimate", out)
    rec.finalize({"n_acrit": n_acrit})
    logger.info("estimate: expected %.0f core-h, limits %.0f core-h, within budget %s", out["total_expected"], out["total_limits"], out["within_budget"])
    return rec


# ----------------------------------------------------------------------------- merge, summary, the rule

def _parts(tile: str, sc_label: str, group: str | None = None) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for r in find_runs("a10_flux"):
        if not (r / "params.json").exists():
            continue
        p = json.loads((r / "params.json").read_text())
        s = p["setup"]
        if s["tile"] == tile and s["sc_label"] == sc_label and not s.get("crop") and s.get("split") and (group is None or s.get("group") == group):
            out.append((r, p))
    return out


def merge_flux(tile: str, sc_label: str, group: str | None = None, run_id: str | None = None, results_root_=None, save_grid: bool = True) -> RunRecorder:
    parts = _parts(tile, sc_label, group)
    if not parts:
        raise FileNotFoundError(f"{tile} {sc_label}: no a10_flux parts")
    n = max(p["setup"]["split"][1] for _, p in parts)
    grp: dict[int, tuple[Path, dict[str, Any]]] = {}
    for r, p in parts:
        if p["setup"]["split"][1] == n:
            grp[p["setup"]["split"][0]] = (r, p)
    missing = [k for k in range(n) if k not in grp]
    if missing:
        raise FileNotFoundError(f"{tile} {sc_label}: parts {missing} of {n} missing")
    frames, partial = [], []
    for k in range(n):
        d = grp[k][0]
        if (d / "flux_rows.parquet").exists():
            frames.append(pd.read_parquet(d / "flux_rows.parquet"))
        else:                                              # the part stopped (time limit) after its last complete row
            frames.append(pd.read_parquet(d / "partial_rows.parquet"))
            partial.append(d.name)
    df = pd.concat(frames, ignore_index=True).sort_values(["A_crit", "D"]).reset_index(drop=True)
    s0 = grp[0][1]["setup"]
    rec = RunRecorder("a10_flux_merged", f"{tile}_{sc_label}", run_id=run_id, root=results_root_)
    rec.write_table(df, "flux_rows")
    n_tot = int(len(df))
    cls = ("converged", "no_steady_solution", "not_converged")
    out: dict[str, Any] = {"tile": tile, "sc_label": sc_label, "S_c": s0["S_c"], "form": FORM, "K": s0["K"], "n_evaluations": n_tot,
                           "n_planned": len(s0["A_crit_grid_full"]) * len(s0["D_grid"]),
                           "counts": {k: int((df.outcome == k).sum()) for k in cls},
                           "counts_a9_rule": {k: int((df.outcome_a9 == k).sum()) for k in cls},
                           "linear_joint": s0["linear_joint_optimum"], "parts": {str(k): grp[k][0].name for k in range(n)},
                           "stopped": {str(k): grp[k][1].get("stopped") for k in range(n)}, "parts_incomplete": partial, "D0": s0["D0"], "D0_B": s0["D0_B"]}
    out["frac_not_converged"] = float(out["counts"]["not_converged"] / n_tot) if n_tot else float("nan")
    out["stop_rule_applied"] = bool(any(v for v in out["stopped"].values()))
    conv = df[df.outcome == "converged"]
    if len(conv):
        b = conv.iloc[int(np.argmin(conv.mse.to_numpy()))].to_dict()
        out["optimum"] = b
        out["mse_ratio_to_linear_joint"] = float(b["mse"] / float(out["linear_joint"]["mse"]))
        out["D_over_D0_A"] = float(b["D"] / s0["D0"])
        out["contours"] = _contours(cast(list[dict[str, Any]], conv.to_dict("records")))
        ga, gd_ = np.array(s0["A_crit_grid_full"]), np.array(s0["D_grid"])
        out["on_grid_edge"] = {"A_crit": bool(np.isclose(b["A_crit"], ga[[0, -1]]).any()), "D": bool(np.isclose(b["D"], gd_[[0, -1]]).any())}
        if save_grid:
            src = [grp[k][0] for k in range(n) if grp[k][1].get("best_converged") and np.isclose(grp[k][1]["best_converged"]["mse"], b["mse"])]
            if src and (src[0] / "best_surface.npy").exists():
                td = load_tile(tile)
                Zb = np.load(src[0] / "best_surface.npy")
                out["grid_check_mse"] = mse(Zb, td.Z)
                out["files"] = write_model_outputs(rec, td.filled, Zb, f"{td.cfg.site}_1m_best_tile_Partitioned-model-nonlinear-flux-Sc{sc_label}-opt", derived=False)
    rec.add_params("merged", out)
    rec.finalize({"tile": tile, "sc_label": sc_label, "parts": out["parts"]})
    return rec


def latest_merged(tile: str, sc_label: str) -> tuple[Path, dict[str, Any]] | None:
    hit = None
    for r in find_runs("a10_flux_merged", f"{tile}_{sc_label}"):
        if (r / "params.json").exists():
            hit = (r, json.loads((r / "params.json").read_text())["merged"])
    return hit


def rule_tile(m: dict[str, Any]) -> dict[str, Any]:
    if "optimum" not in m:
        return {"fit": False, "D": False, "pass": False, "reason": "no converged evaluation"}
    fit = bool(m["mse_ratio_to_linear_joint"] <= RULE["mse_ratio_max"])
    lo, hi = RULE["D_ratio"]
    dok = bool(lo <= m["D_over_D0_A"] <= hi)
    return {"fit": fit, "D": dok, "pass": bool(fit and dok)}


def summarize_flux(tiles: Sequence[str] = TILES, run_id: str | None = None, results_root_=None) -> RunRecorder:
    rec = RunRecorder("a10_flux_summary", f"{len(tiles)}tiles", run_id=run_id, root=results_root_)
    table: list[dict[str, Any]] = []
    per: dict[str, dict[str, Any]] = {}
    for t in tiles:
        per[t] = {}
        for lab in SC_LABELS:
            hit = latest_merged(t, lab)
            if hit is None:
                continue
            r, m = hit
            v = rule_tile(m)
            per[t][lab] = {"rule": v, "run": r.name, "S_c": m["S_c"]}
            o = m.get("optimum") or {}
            c = m.get("contours") or {}
            c1, c5 = c.get("contour_0.01", {}), c.get("contour_0.05", {})
            table.append({"tile": t, "sc_label": lab, "S_c": m["S_c"], "A_crit": o.get("A_crit"), "D": o.get("D"), "mse": o.get("mse"),
                          "linear_joint_mse": float(m["linear_joint"]["mse"]), "linear_joint_A_crit": float(m["linear_joint"]["A_crit"]),
                          "linear_joint_D": float(m["linear_joint"]["D"]), "mse_ratio": m.get("mse_ratio_to_linear_joint"), "D0_A": m["D0"],
                          "D_over_D0_A": m.get("D_over_D0_A"), "n_evaluations": m["n_evaluations"], "n_planned": m["n_planned"],
                          **{f"n_{k}": v_ for k, v_ in m["counts"].items()}, **{f"n_{k}_a9rule": v_ for k, v_ in m["counts_a9_rule"].items()},
                          "frac_not_converged": m["frac_not_converged"], "stop_rule_applied": m["stop_rule_applied"],
                          "frac_gt_0.5Sc": o.get("frac_hillslope_gt_0.5Sc"), "frac_clipped": o.get("frac_clipped"), "frac_clipped_interior": o.get("frac_clipped_interior"),
                          "contour1_A": [c1.get("A_crit_min"), c1.get("A_crit_max")] if c1 else None, "contour1_D": [c1.get("D_min"), c1.get("D_max")] if c1 else None,
                          "contour5_A": [c5.get("A_crit_min"), c5.get("A_crit_max")] if c5 else None, "contour5_D": [c5.get("D_min"), c5.get("D_max")] if c5 else None,
                          "edge": m.get("on_grid_edge"), "fit": v["fit"], "D_ok": v["D"], "pass": v["pass"], "run": r.name})
    tdf = pd.DataFrame(table)
    # 2.3: preferred if, on >= 4 of 5 tiles, at some S_c >= 0.9, fit and D hold at the best (A_crit, D). Read two ways:
    # (a) one S_c shared by the tiles; (b) each tile at either S_c >= 0.9.
    by_sc = {}
    for lab in SC_LABELS:
        vs = [(t, per[t][lab]) for t in tiles if lab in per[t]]
        if not vs:
            continue
        by_sc[lab] = {"n_tiles": len(vs), "n_fit": sum(v["rule"]["fit"] for _, v in vs), "n_D": sum(v["rule"]["D"] for _, v in vs),
                      "n_pass": sum(v["rule"]["pass"] for _, v in vs), "tiles_fit": [t for t, v in vs if v["rule"]["fit"]],
                      "tiles_D": [t for t, v in vs if v["rule"]["D"]], "tiles_pass": [t for t, v in vs if v["rule"]["pass"]],
                      "S_c_values": {t: v["S_c"] for t, v in vs}}
    shared = [lab for lab in SC_LABELS if lab in by_sc and lab in SC_FIXED and by_sc[lab]["n_pass"] >= RULE["min_tiles"]]
    either = [t for t in tiles if any(per[t].get(lab, {}).get("rule", {}).get("pass") for lab in SC_FIXED)]
    # fitted S_c that lie at or above 0.9 count toward "preferred"; below 0.9 they can support "closer D" only
    verdict = {"preferred_shared_S_c": shared, "tiles_pass_at_either_S_c_ge_0.9": either,
               "preferred_reading_a": bool(shared), "preferred_reading_b": bool(len(either) >= RULE["min_tiles"])}
    verdict["preferred"] = bool(verdict["preferred_reading_a"] or verdict["preferred_reading_b"])
    verdict["readings_agree"] = bool(verdict["preferred_reading_a"] == verdict["preferred_reading_b"])
    rec.write_table(tdf, "flux_summary")
    rec.add_params("per_tile", per)
    rec.add_params("by_S_c", by_sc)
    rec.add_params("verdict", verdict)
    rec.add_params("rule", RULE)
    rec.finalize({"tiles": list(tiles)})
    return rec
