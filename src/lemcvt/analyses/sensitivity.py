"""Addendum 12 (v2.1): equal-parameter comparisons (section 2) and per-tile concavity sensitivity (section 1).

Section 2, at the site m (the configs' ``measured.m``), from the exact reductions of :mod:`lemcvt.optimize.v2`:

    partitioned at fixed A_crit   Z = a + x b + y c,   x = U/K, y = U/D
    AD at fixed r = K/D           Z = h(r) + y g(r)

Variants (free parameters in brackets):

    P1  stage 2, K0 and D0 fixed [A_crit]            A1  AD with K = K0 [D]
    P2  D fixed at D0 [K, A_crit]                    A2  AD with D = D0 [K]
    P3  K fixed at K0 [D, A_crit]                    AD  [K, D]
    P4  joint [K, D, A_crit]

On AP every variant that fixes D is evaluated at both ends of the bracket, D0_A and D0_B (suffixes ``_A`` and ``_B``).
At fixed A_crit the MSE is the quadratic ``rr - 2 g.v + v.G.v`` in v = (x, y); a variant with one of x, y fixed is the
one-dimensional minimum of that quadratic, clipped to the bounds, so every partitioned variant is read from the Gram
terms (G, g, rr) of the A_crit row. The Phase 6 refined profiles stored those terms for the whole tile; this module
recomputes them on the whole tile and on each half of a basin split (section 2.2) over the thinned A_crit grid, refines
the staircase around every variant's minimum and fills its 1 % interval.

Held-out (2.2): the tile's basins (outlets on the tile edge, network at 1e3 m^2) are split greedily, largest basin (in
cells on the tile) first, each to the half with fewer cells. Cells in no edge-outlet basin are in neither half. The
model surface is always the whole-tile solution; "fit on a half" means the parameters minimise the MSE over that half's
cells, and the score is the MSE over the other half's cells at those parameters. For the partitioned variants the
fitting half also chooses A_crit; for AD it chooses r.

Section 1: per-tile m_t from the 4b/5 m-profile runs; the site-pooled theta from the addendum 4, 6.4 ladder rerun;
k_s at theta = m_t (the Phase 6 estimator); the m_t runs themselves are the v2 optimizer with a model override (step
``a12_mt``; see :func:`lemcvt.analyses.optimizer_v2.run_tile_v2`).
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from lemcvt.config import apply_d0_variant, config_snapshot, load_tile, load_tile_config
from lemcvt.optimize.v1 import Trajectory
from lemcvt.optimize.v2 import (
    ADProfiler,
    PartitionedProfiler,
    box_quadratic_min,
    identifiability,
    thin_log_uniform,
    unique_areas,
)
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

ACRIT_RANGE = (1e2, 1e5)
THIN_N = 2000
HOLDOUT_A_C = 1e3
FILL_LEVEL = 0.01
FILL_MAX = 1500
AD_N_GRID = 200
AUDIT_RUN = "20261110T160633Z_measure_audit_10tiles"      # addendum 4, 6.4 ladder rerun (theta grid 0.00-0.90)
RULES = {
    "equal_count_min_tiles": 4,          # 2.3: lower than every AD variant of the same count on >= 4 of 5 tiles
    "heldout_min_tiles": 4,              # 2.3: held-out MSE lower on >= 4 of 5 tiles
    "m_immaterial_mse_tiles": 5,         # 1.4: partitioned MSE below AD MSE on 5 of 5 tiles at m_t
    "m_immaterial_median_change": 0.25,  # 1.4: median D/D0_A and median AD K/K0 change by less than 25 %
}


# ============================================================================= the basin split (2.2)

def greedy_basin_halves(Z: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, dy: float,
                        A_c: float = HOLDOUT_A_C) -> tuple[np.ndarray, dict[str, Any]]:
    """Half (1 or 2) of every cell; 0 for cells in no edge-outlet basin of the ``A_c`` network.

    Basins are those of the network's outlets on the tile edge; every cell belongs to the basin of the channel pixel it
    first reaches along D8. Greedy split: basins in descending order of their cell count on the tile (ties by outlet
    index), each to the half with fewer cells so far (half 1 on a tie).
    """
    from lemcvt.measure.chi import channel_network, channel_pixel_basin_lookup
    from lemcvt.measure.hilltop_curvature import basin_labels

    net = channel_network(Z, A, FD, A_c, dx, dy)
    lab = basin_labels(A, FD, net.mask)
    lookup = channel_pixel_basin_lookup(net)
    basin = np.where(lab > 0, lookup[np.maximum(lab - 1, 0)], -1)
    valid = np.isfinite(A)
    basin[~valid] = -1
    outlets = net.outlet_index
    ny, nx = Z.shape
    o_rows, o_cols = net.rows[outlets], net.cols[outlets]
    on_edge = (o_rows == 0) | (o_rows == ny - 1) | (o_cols == 0) | (o_cols == nx - 1)
    ids = np.unique(basin[basin >= 0])
    if not set(ids.tolist()) <= set(outlets.tolist()):
        raise RuntimeError("basin ids are not outlet indices of the network")
    counts = np.bincount(basin[basin >= 0], minlength=int(outlets.max()) + 1 if outlets.size else 1)
    edge_outlets = outlets[on_edge]
    order = sorted(edge_outlets.tolist(), key=lambda o: (-int(counts[o]), int(o)))
    tot = [0, 0]
    half_of: dict[int, int] = {}
    for o in order:
        h = 0 if tot[0] <= tot[1] else 1
        half_of[int(o)] = h + 1
        tot[h] += int(counts[o])
    lut = np.zeros(counts.size, dtype=np.int8)
    for o, h in half_of.items():
        lut[o] = h
    half = np.zeros((ny, nx), dtype=np.int8)
    inb = basin >= 0
    half[inb] = lut[basin[inb]]
    n_valid = int(valid.sum())
    n1, n2 = int((half == 1).sum()), int((half == 2).sum())
    n_interior_basin_cells = int(sum(int(counts[o]) for o in outlets[~on_edge]))
    basins = [{"outlet": int(o), "row": int(net.rows[o]), "col": int(net.cols[o]), "n_cells": int(counts[o]),
               "outlet_area_m2": float(net.area[o]), "half": half_of[int(o)]} for o in order]
    info = {"A_c": A_c, "rule": "edge-outlet basins of the A_c network, greedy by cell count (largest first, each to the lighter half)",
            "n_basins_edge": int(on_edge.sum()), "n_interior_outlets": int((~on_edge).sum()),
            "n_cells_valid": n_valid, "n_cells_half1": n1, "n_cells_half2": n2,
            "share_half1_of_split": n1 / (n1 + n2) if (n1 + n2) else float("nan"),
            "share_half2_of_split": n2 / (n1 + n2) if (n1 + n2) else float("nan"),
            "frac_valid_in_split": (n1 + n2) / n_valid if n_valid else float("nan"),
            "n_cells_interior_outlet_basins": n_interior_basin_cells,
            "n_cells_no_basin": int(n_valid - n1 - n2 - n_interior_basin_cells),
            "n_basins_half1": sum(1 for b in basins if b["half"] == 1), "n_basins_half2": sum(1 for b in basins if b["half"] == 2),
            "largest_basin_share_of_split": (basins[0]["n_cells"] / (n1 + n2)) if basins and (n1 + n2) else float("nan"),
            "basins": basins}
    return half, info


# ============================================================================= quadratic helpers

TERM_KEYS = ("G00", "G01", "G11", "g0", "g1", "rr", "n")


def _terms_masked(red, Z_obs_flat: np.ndarray, keep: np.ndarray | None) -> dict[str, float]:
    r = Z_obs_flat - red.a
    ok = np.isfinite(r) & np.isfinite(red.b) & np.isfinite(red.c)
    if keep is not None:
        ok &= keep
    b, c, r = red.b[ok], red.c[ok], r[ok]
    n = int(ok.sum())
    if n == 0:
        return dict.fromkeys(TERM_KEYS, float("nan")) | {"n": 0}
    return {"G00": float(b @ b) / n, "G01": float(b @ c) / n, "G11": float(c @ c) / n, "g0": float(b @ r) / n,
            "g1": float(c @ r) / n, "rr": float(r @ r) / n, "n": n}


def _q(t: dict[str, float], x: float, y: float) -> float:
    return float(t["rr"] - 2.0 * (t["g0"] * x + t["g1"] * y) + t["G00"] * x * x + 2.0 * t["G01"] * x * y + t["G11"] * y * y)


def _fit(t: dict[str, float], kind: str, lo: np.ndarray, hi: np.ndarray, x0: float, y_fixed: float | None) -> tuple[float, float, str]:
    """(x, y, where) minimising the quadratic for a variant: ``joint`` (both free), ``fix_y`` (y = y_fixed), ``fix_x``
    (x = x0), ``fix_xy`` (x = x0, y = y_fixed)."""
    if kind == "joint":
        v, where = box_quadratic_min(np.array([[t["G00"], t["G01"]], [t["G01"], t["G11"]]]), np.array([t["g0"], t["g1"]]), lo, hi)
        return float(v[0]), float(v[1]), where
    if kind == "fix_xy":
        assert y_fixed is not None
        return x0, y_fixed, "fixed"
    if kind == "fix_y":
        assert y_fixed is not None
        x = (t["g0"] - t["G01"] * y_fixed) / t["G00"] if t["G00"] > 0 else lo[0]
        xc = float(np.clip(x, lo[0], hi[0]))
        return xc, y_fixed, "interior" if xc == x else ("x_lo" if xc == lo[0] else "x_hi")
    if kind == "fix_x":
        y = (t["g1"] - t["G01"] * x0) / t["G11"] if t["G11"] > 0 else lo[1]
        yc = float(np.clip(y, lo[1], hi[1]))
        return x0, yc, "interior" if yc == y else ("y_lo" if yc == lo[1] else "y_hi")
    raise ValueError(kind)


def partitioned_variants(K0: float, D0_A: float, D0_B: float | None) -> dict[str, tuple[str, float | None, int]]:
    """name -> (fit kind, fixed D or None, parameter count)."""
    out: dict[str, tuple[str, float | None, int]] = {"P1_A": ("fix_xy", D0_A, 1), "P2_A": ("fix_y", D0_A, 2), "P3": ("fix_x", None, 2),
                                                     "P4": ("joint", None, 3)}
    if D0_B is not None:
        out["P1_B"] = ("fix_xy", D0_B, 1)
        out["P2_B"] = ("fix_y", D0_B, 2)
    return out


HELDOUT_PARTITIONED = ("P4", "P2_A", "P2_B")


def row_objectives(terms: dict[str, dict[str, float]], U: float, K0: float, D0_A: float, D0_B: float | None,
                   lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    """Every variant's fit on ``all`` (and fit-on-one-half, score-on-the-other for the held-out variants when the half
    terms are present) at one A_crit row."""
    x0 = U / K0
    out: dict[str, Any] = {}
    for name, (kind, Dfix, _cnt) in partitioned_variants(K0, D0_A, D0_B).items():
        yf = U / Dfix if Dfix is not None else None
        for part in ("all", "h1", "h2"):
            if part not in terms:
                continue
            if part != "all" and name not in HELDOUT_PARTITIONED:
                continue
            t = terms[part]
            x, y, where = _fit(t, kind, lo, hi, x0, yf)
            out[f"mse_{name}_{part}"] = _q(t, x, y)
            out[f"K_{name}_{part}"] = U / x
            out[f"D_{name}_{part}"] = U / y
            out[f"where_{name}_{part}"] = where
            if part in ("h1", "h2"):
                other = "h2" if part == "h1" else "h1"
                out[f"mse_{name}_{part}_score_{other}"] = _q(terms[other], x, y)
    return out


def _terms_from_phase6_row(r: dict[str, Any]) -> dict[str, float]:
    return {"G00": r["G00"], "G01": r["G01"], "G11": r["G11"], "g0": r["g0"], "g1": r["g1"], "rr": r["rr"], "n": float("nan")}


# ============================================================================= Phase 6 runs

def phase6_runs(tile: str) -> dict[str, tuple[str, Any]]:
    """The Phase 6 runs of the tile, by the same selection as Table 1 (``ad``, ``partitioned``, ...)."""
    from lemcvt.tables.table1 import _read_records, select_phase6_runs

    return dict(select_phase6_runs(_read_records(tile)))


def _results_dir(name: str) -> Path:
    from lemcvt.config import results_root

    return results_root() / name


# ============================================================================= section 2, partitioned

def run_equal_part(tile: str, thin_n: int = THIN_N, fill_level: float = FILL_LEVEL, fill_max: int = FILL_MAX,
                   run_id: str | None = None, results_root=None, crop=None) -> RunRecorder:
    td = load_tile(tile, crop=crop, sanity=crop is None)
    mp = apply_d0_variant(td.cfg.model, "A")
    Z0, A, FD, dx, dy, U = td.Z, td.A, td.FD, td.dx, td.filled.dy, mp.U
    K0, D0A, D0B = mp.K0_partitioned, mp.D0, mp.D0_B
    rec = RunRecorder("a12_equal_part", tile, run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    rec.add_params("tile", tile)
    rec.add_params("inputs", td.provenance())
    rec.add_params("model", {"m": mp.m, "U": U, "K0": K0, "D0_A": D0A, "D0_B": D0B, "K_bounds": mp.K_bounds, "D_bounds": mp.D_bounds,
                             "z_units_working": td.z_units_working})
    half, hinfo = greedy_basin_halves(Z0, A, FD, dx, dy)
    rec.add_params("split", hinfo)
    logger.info("%s split: %d edge basins, halves %d / %d cells (%.3f / %.3f of the split, %.3f of valid cells in the split)", tile,
                hinfo["n_basins_edge"], hinfo["n_cells_half1"], hinfo["n_cells_half2"], hinfo["share_half1_of_split"],
                hinfo["share_half2_of_split"], hinfo["frac_valid_in_split"])
    prof = PartitionedProfiler(Z0, A, FD, dx, U, m_values=[mp.m])
    gd = prof.gd
    half_flat = half[gd["rev_j"], gd["rev_i"]]
    keep = {"h1": half_flat == 1, "h2": half_flat == 2}
    lo = np.array([U / mp.K_bounds[1], U / mp.D_bounds[1]])
    hi = np.array([U / mp.K_bounds[0], U / mp.D_bounds[0]])
    ua = unique_areas(A, *ACRIT_RANGE)
    grid = thin_log_uniform(ua, thin_n)
    rows: list[dict[str, Any]] = []
    done: set[float] = set()
    traj = Trajectory("a12_equal_part", ("log10_A_crit", "log10_D", "log10_K", "m", "kind"))

    def _eval(a_c: float, kind: str) -> None:
        a_c = float(a_c)
        if a_c in done:
            return
        red = prof.reduce(a_c, mp.m)
        terms = {"all": _terms_masked(red, prof.Z_obs_flat, None), "h1": _terms_masked(red, prof.Z_obs_flat, keep["h1"]),
                 "h2": _terms_masked(red, prof.Z_obs_flat, keep["h2"])}
        row: dict[str, Any] = {"A_crit": a_c, "kind": kind, "n_channel": int(red.channel_mask.sum()), "n_hillslope": red.n_hillslope,
                               "wall_s": red.wall_s}
        for part, t in terms.items():
            row.update({f"{k}_{part}": v for k, v in t.items()})
        row.update(row_objectives(terms, U, K0, D0A, D0B, lo, hi))
        rows.append(row)
        done.add(a_c)
        traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(row["D_P4_all"]), "log10_K": math.log10(row["K_P4_all"]),
                     "m": mp.m, "kind": kind}, row["mse_P4_all"], cached=False)

    for i, a_c in enumerate(grid):
        _eval(a_c, "grid")
        if i % 200 == 0 or i == grid.size - 1:
            logger.info("%s equal-part grid %d/%d A_crit %.1f [%.0f s]", tile, i + 1, grid.size, a_c, time.perf_counter() - t0)
    fit_cols = [c for c in rows[0] if c.startswith("mse_") and "_score_" not in c]
    # staircase refinement around every variant's minimum (every unique area between the neighbouring grid values)
    gs = np.sort(grid)
    extra: set[float] = set()
    for col in fit_cols:
        a_min = min(rows, key=lambda r: r[col])["A_crit"]
        j = int(np.searchsorted(gs, a_min))
        lo_a, hi_a = gs[max(j - 1, 0)], gs[min(j + 1, gs.size - 1)]
        extra |= set(ua[(ua > lo_a) & (ua < hi_a)].tolist())
    extra -= done
    for a_c in sorted(extra):
        _eval(a_c, "refine")
    logger.info("%s equal-part staircase refinement: %d values [%.0f s]", tile, len(extra), time.perf_counter() - t0)
    # fill every variant's fill_level interval (capped)
    targets: set[float] = set()
    fill_info: dict[str, Any] = {"level": fill_level, "max": fill_max, "intervals": {}}
    for col in fit_cols:
        ident = identifiability(rows, col, levels=(fill_level,))
        ivs = ident[f"level_{fill_level:g}"]["intervals"]
        fill_info["intervals"][col] = ivs
        for a_lo, a_hi in ivs:
            targets |= set(ua[(ua >= a_lo) & (ua <= a_hi)].tolist())
    targets -= done
    fill_a = np.array(sorted(targets))
    fill_info["n_targets"] = int(fill_a.size)
    if fill_a.size > fill_max:
        fill_a = thin_log_uniform(fill_a, fill_max)
    fill_info["n_evaluated"] = int(fill_a.size)
    fill_info["capped"] = bool(fill_info["n_targets"] > fill_max)
    for i, a_c in enumerate(fill_a):
        _eval(a_c, "fill")
        if i % 250 == 0:
            logger.info("%s equal-part fill %d/%d [%.0f s]", tile, i + 1, fill_a.size, time.perf_counter() - t0)
    rec.add_trajectory(traj)
    df = pd.DataFrame(rows).sort_values("A_crit").reset_index(drop=True)
    rec.write_table(df, "acrit_profile")
    recs = cast(list[dict[str, Any]], df.to_dict("records"))
    result = summarize_part_rows(recs, tile, mp, U, K0, D0A, D0B, lo, hi, grid_only=[r for r in recs if r["kind"] == "grid"])
    result["fill"] = fill_info
    result["n_staircase"] = len(extra)
    result["n_evaluations"] = len(recs)
    rec.add_params("result", result)
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.finalize(config_snapshot(td.cfg, {"crop": crop, "thin_n": thin_n, "fill_level": fill_level, "fill_max": fill_max}))
    return rec


def _opt(rows: list[dict[str, Any]], name: str, part: str = "all") -> dict[str, Any]:
    col = f"mse_{name}_{part}"
    r = min((q for q in rows if np.isfinite(q.get(col, np.nan))), key=lambda q: q[col])
    out = {"A_crit": float(r["A_crit"]), "mse": float(r[col]), "K": float(r[f"K_{name}_{part}"]), "D": float(r[f"D_{name}_{part}"]),
           "where": r[f"where_{name}_{part}"], "n_channel": int(r["n_channel"])}
    if part in ("h1", "h2"):
        other = "h2" if part == "h1" else "h1"
        out[f"score_{other}"] = float(r[f"mse_{name}_{part}_score_{other}"])
    return out


def summarize_part_rows(recs, tile, mp, U, K0, D0A, D0B, lo, hi, grid_only=None) -> dict[str, Any]:
    names = list(partitioned_variants(K0, D0A, D0B))
    insample = {n: _opt(recs, n) for n in names}
    insample_grid = {n: _opt(grid_only, n) for n in names} if grid_only else {}
    # union with the stored Phase 6 rows (whole-tile terms; the Phase 6 grid with its staircase refinement and 5 % fill)
    p6 = phase6_runs(tile).get("partitioned")
    union: dict[str, Any] = {}
    check: dict[str, Any] = {"available": False}
    if p6 is not None:
        prof6 = pd.read_parquet(_results_dir(p6[0]) / "acrit_profile.parquet")
        rows6 = []
        for r in cast(list[dict[str, Any]], prof6.to_dict("records")):
            obj = row_objectives({"all": _terms_from_phase6_row(r)}, U, K0, D0A, D0B, lo, hi)
            rows6.append({"A_crit": float(r["A_crit"]), "n_channel": int(r["n_channel"]), **obj, "_p6_joint": r["mse_joint"], "_p6_s2": r["mse_stage2"]})
        # Gram-term check against the stored joint / stage-2 columns, and the new rows against the stored rows
        rel_j = max(abs(q["mse_P4_all"] / q["_p6_joint"] - 1) for q in rows6)
        rel_s = max(abs(q["mse_P1_A_all"] / q["_p6_s2"] - 1) for q in rows6)
        by_a6 = {q["A_crit"]: q for q in rows6}
        common = [q for q in recs if q["A_crit"] in by_a6]
        rel_new = max((abs(q["mse_P4_all"] / by_a6[q["A_crit"]]["mse_P4_all"] - 1) for q in common), default=float("nan"))
        rel_new_p2 = max((abs(q["mse_P2_A_all"] / by_a6[q["A_crit"]]["mse_P2_A_all"] - 1) for q in common), default=float("nan"))
        check = {"available": True, "run": p6[0], "n_rows_phase6": len(rows6), "n_common": len(common),
                 "max_rel_stored_terms_vs_stored_joint": rel_j, "max_rel_stored_terms_vs_stored_stage2": rel_s,
                 "max_rel_new_vs_phase6_P4": rel_new, "max_rel_new_vs_phase6_P2_A": rel_new_p2,
                 "phase6_joint_optimum": {k: p6[1]["v2"]["partitioned"]["joint_optimum"][k] for k in ("A_crit", "K", "D", "mse")},
                 "phase6_stage2_optimum": {k: p6[1]["v2"]["partitioned"]["stage2_optimum"][k] for k in ("A_crit", "mse")}}
        allrows = rows6 + [q for q in recs if q["A_crit"] not in by_a6]
        union = {n: _opt(allrows, n) for n in names}
        union["_n_rows"] = len(allrows)
    held: dict[str, Any] = {}
    for n in HELDOUT_PARTITIONED:
        if f"mse_{n}_h1" not in recs[0]:
            continue
        f1, f2 = _opt(recs, n, "h1"), _opt(recs, n, "h2")
        held[n] = {"fit_h1": f1, "fit_h2": f2, "mse_heldout_mean": 0.5 * (f1["score_h2"] + f2["score_h1"])}
    return {"insample": insample, "insample_thinned_grid_only": insample_grid, "insample_union_phase6": union, "phase6_check": check,
            "heldout": held, "param_count": {n: v[2] for n, v in partitioned_variants(K0, D0A, D0B).items()},
            "K0": K0, "D0_A": D0A, "D0_B": D0B}


# ============================================================================= section 2, AD

AD_VARIANTS = {"AD": 2, "A1": 1, "A2_A": 1, "A2_B": 1}


def run_equal_ad(tile: str, n_grid: int = AD_N_GRID, run_id: str | None = None, results_root=None, crop=None) -> RunRecorder:
    from scipy.optimize import minimize_scalar

    from lemcvt.models.ad import advection_diffusion_model
    from lemcvt.models.common import extract_grid_data

    td = load_tile(tile, crop=crop, sanity=crop is None)
    mp = apply_d0_variant(td.cfg.model, "A")
    Z0, A, FD, dx, dy, U = td.Z, td.A, td.FD, td.dx, td.filled.dy, mp.U
    K0, D0A, D0B = mp.K0_ad, mp.D0, mp.D0_B
    (K_lo, K_hi), (D_lo, D_hi) = mp.K_bounds, mp.D_bounds
    rec = RunRecorder("a12_equal_ad", tile, run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    rec.add_params("tile", tile)
    rec.add_params("inputs", td.provenance())
    rec.add_params("model", {"m": mp.m, "U": U, "K0": K0, "D0_A": D0A, "D0_B": D0B, "K_bounds": mp.K_bounds, "D_bounds": mp.D_bounds,
                             "z_units_working": td.z_units_working})
    half, hinfo = greedy_basin_halves(Z0, A, FD, dx, dy)
    rec.add_params("split", hinfo)
    ad = advection_diffusion_model(Z0, A, FD, dx, mp.m)
    gd = extract_grid_data(Z0, A, FD, dx)
    obs = Z0[gd["rev_j"], gd["rev_i"]].astype(float)
    prof = ADProfiler(ad, obs, U)
    half_flat = half[gd["rev_j"], gd["rev_i"]]
    masks = {"all": np.ones(obs.size, dtype=bool), "h1": half_flat == 1, "h2": half_flat == 2}
    cache: dict[float, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    traj = Trajectory("a12_equal_ad", ("log10_K", "log10_D", "log10_r", "kind"))

    def _mse_y(res, g, ok, y):
        e = res[ok] - y * g[ok]
        return float(e @ e / ok.sum())

    def evaluate(r: float, kind: str) -> dict[str, Any]:
        r = float(r)
        if r in cache:
            return cache[r]
        red = prof.reduce(r)
        res = obs - red["h"]
        g = red["g"]
        fin = np.isfinite(res) & np.isfinite(g)
        row: dict[str, Any] = {"r": r, "kind": kind, "wall_s": red["wall_s"]}
        # AD [K, D]: y = U/D free within the D range feasible at this r
        d_lo, d_hi = max(D_lo, K_lo / r), min(D_hi, K_hi / r)
        for part in ("all", "h1", "h2"):
            ok = fin & masks[part]
            if d_lo > d_hi or not ok.any():
                row[f"mse_AD_{part}"] = float("inf")
                continue
            gg = float(g[ok] @ g[ok])
            y = float(g[ok] @ res[ok]) / gg if gg > 0 else U / d_hi
            y_lo, y_hi = U / d_hi, U / d_lo
            where = "interior" if y_lo <= y <= y_hi else ("y_lo" if y < y_lo else "y_hi")
            y = float(np.clip(y, y_lo, y_hi))
            row[f"mse_AD_{part}"] = _mse_y(res, g, ok, y)
            row[f"K_AD_{part}"] = r * U / y
            row[f"D_AD_{part}"] = U / y
            row[f"where_AD_{part}"] = where
            if part != "all":
                other = "h2" if part == "h1" else "h1"
                row[f"mse_AD_{part}_score_{other}"] = _mse_y(res, g, fin & masks[other], y)
        ok = fin
        # A1: K = K0, D = K0 / r
        D1 = K0 / r
        row["mse_A1_all"] = _mse_y(res, g, ok, U / D1) if D_lo <= D1 <= D_hi else float("inf")
        row["K_A1_all"], row["D_A1_all"] = K0, D1
        # A2: D = D0 (both ends), K = r D0
        for suf, D0 in (("A", D0A), ("B", D0B)):
            if D0 is None:
                continue
            K2 = r * D0
            row[f"mse_A2_{suf}_all"] = _mse_y(res, g, ok, U / D0) if K_lo <= K2 <= K_hi else float("inf")
            row[f"K_A2_{suf}_all"], row[f"D_A2_{suf}_all"] = K2, D0
        cache[r] = row
        rows.append(row)
        if np.isfinite(row["mse_AD_all"]):
            traj.record({"log10_K": math.log10(row["K_AD_all"]), "log10_D": math.log10(row["D_AD_all"]), "log10_r": math.log10(r), "kind": kind},
                        row["mse_AD_all"], cached=False)
        return row

    r_lo, r_hi = K_lo / D_hi, K_hi / D_lo
    grid = np.logspace(math.log10(r_lo), math.log10(r_hi), n_grid)
    for i, r in enumerate(grid):
        evaluate(r, "grid")
        if i % 25 == 0 or i == n_grid - 1:
            logger.info("%s equal-ad grid %d/%d r=%.3e [%.0f s]", tile, i + 1, n_grid, r, time.perf_counter() - t0)
    objectives = ["mse_AD_all", "mse_AD_h1", "mse_AD_h2", "mse_A1_all", "mse_A2_A_all"] + (["mse_A2_B_all"] if D0B is not None else [])
    brent: dict[str, Any] = {}
    for col in objectives:
        feas = sorted((q for q in rows if q["kind"] == "grid" and np.isfinite(q[col])), key=lambda q: q["r"])
        if not feas:
            brent[col] = {"feasible": False}
            continue
        jj = int(np.argmin([q[col] for q in feas]))
        lg = np.log10([q["r"] for q in feas])
        a, b = lg[max(jj - 1, 0)], lg[min(jj + 1, len(lg) - 1)]
        info: dict[str, Any] = {"grid_index": jj, "n_feasible_grid": len(feas), "at_feasible_edge": bool(jj in (0, len(feas) - 1))}
        if a < b:
            res_b = minimize_scalar(lambda lr, c=col: evaluate(10 ** lr, f"brent_{c}")[c], bounds=(a, b), method="bounded", options={"xatol": 1e-6})
            info.update({"log10_r": float(res_b.x), "nfev": int(res_b.nfev), "success": bool(res_b.success), "bracket_log10": [float(a), float(b)]})
        brent[col] = info
    rec.add_trajectory(traj)
    df = pd.DataFrame(rows).sort_values("r").reset_index(drop=True)
    rec.write_table(df, "r_profile")
    recs = cast(list[dict[str, Any]], df.to_dict("records"))

    def _best(name: str, part: str) -> dict[str, Any]:
        col = f"mse_{name}_{part}"
        q = min((x for x in recs if np.isfinite(x.get(col, np.inf))), key=lambda x: x[col])
        out = {"r": float(q["r"]), "mse": float(q[col]), "K": float(q[f"K_{name}_{part}"]), "D": float(q[f"D_{name}_{part}"])}
        if f"where_{name}_{part}" in q:
            out["where"] = q[f"where_{name}_{part}"]
        if part in ("h1", "h2"):
            other = "h2" if part == "h1" else "h1"
            out[f"score_{other}"] = float(q[f"mse_{name}_{part}_score_{other}"])
        # on a bound of K or D?
        out["K_at_bound"] = bool(np.isclose(out["K"], K_lo, rtol=1e-9) or np.isclose(out["K"], K_hi, rtol=1e-9))
        out["D_at_bound"] = bool(np.isclose(out["D"], D_lo, rtol=1e-9) or np.isclose(out["D"], D_hi, rtol=1e-9))
        return out

    insample = {"AD": _best("AD", "all"), "A1": _best_named(recs, "A1", K_lo, K_hi, D_lo, D_hi),
                "A2_A": _best_named(recs, "A2_A", K_lo, K_hi, D_lo, D_hi)}
    if D0B is not None:
        insample["A2_B"] = _best_named(recs, "A2_B", K_lo, K_hi, D_lo, D_hi)
    f1, f2 = _best("AD", "h1"), _best("AD", "h2")
    held = {"AD": {"fit_h1": f1, "fit_h2": f2, "mse_heldout_mean": 0.5 * (f1["score_h2"] + f2["score_h1"])}}
    p6 = phase6_runs(tile).get("ad")
    check: dict[str, Any] = {"available": False}
    if p6 is not None:
        o6 = p6[1]["v2"]["ad"]["optimum"]
        check = {"available": True, "run": p6[0], "phase6": {k: o6[k] for k in ("K", "D", "mse")},
                 "rel_diff_mse": insample["AD"]["mse"] / o6["mse"] - 1, "rel_diff_K": insample["AD"]["K"] / o6["K"] - 1,
                 "rel_diff_D": insample["AD"]["D"] / o6["D"] - 1}
    rec.add_params("result", {"insample": insample, "heldout": held, "brent": brent, "phase6_check": check,
                              "param_count": AD_VARIANTS, "n_evaluations": len(recs), "K0": K0, "D0_A": D0A, "D0_B": D0B})
    rec.add_params("wall_s", time.perf_counter() - t0)
    logger.info("%s equal-ad: AD %.4f (Phase 6 %s), A1 %.4f, A2_A %.4f, held-out AD %.4f [%.0f s]", tile, insample["AD"]["mse"],
                check.get("phase6", {}).get("mse"), insample["A1"]["mse"], insample["A2_A"]["mse"], held["AD"]["mse_heldout_mean"],
                time.perf_counter() - t0)
    rec.finalize(config_snapshot(td.cfg, {"crop": crop, "n_grid": n_grid}))
    return rec


def _best_named(recs, name: str, K_lo, K_hi, D_lo, D_hi) -> dict[str, Any]:
    col = f"mse_{name}_all"
    q = min((x for x in recs if np.isfinite(x.get(col, np.inf))), key=lambda x: x[col])
    K, D = float(q[f"K_{name}_all"]), float(q[f"D_{name}_all"])
    return {"r": float(q["r"]), "mse": float(q[col]), "K": K, "D": D,
            "K_at_bound": bool(np.isclose(K, K_lo, rtol=1e-9) or np.isclose(K, K_hi, rtol=1e-9)),
            "D_at_bound": bool(np.isclose(D, D_lo, rtol=1e-9) or np.isclose(D, D_hi, rtol=1e-9))}


# ============================================================================= section 1.1 and 1.2.1: m_t, theta, K0(m_t)

def _m_profile_run(tile: str) -> str:
    cfg = load_tile_config(tile)
    runs = ((cfg.measured.get("provenance") or {}).get("m") or {}).get("m_profile_runs") or {}
    if tile not in runs:
        raise KeyError(f"{tile}: the config's measured.provenance.m names no m-profile run")
    return str(runs[tile])


def tile_m(tile: str) -> dict[str, Any]:
    """m_t = the minimum of the tile's exponent profile MSE*(m) (the 4b/5 m-profile run), its 1 % and 5 % intervals."""
    run = _m_profile_run(tile)
    p = json.loads((_results_dir(run) / "params.json").read_text())
    prof = p["v2"]["m_profile"]["profile_m"]
    ms = np.array([q["m"] for q in prof])
    mse = np.array([q["mse"] for q in prof])
    j = int(np.argmin(mse))
    ident = identifiability([{"m": q["m"], "mse": q["mse"]} for q in prof], "mse", key="m")
    grid_lo, grid_hi = float(ms.min()), float(ms.max())
    out: dict[str, Any] = {"tile": tile, "run": run, "m_t": float(ms[j]), "mse_at_m_t": float(mse[j]), "A_crit_at_m_t": float(prof[j]["A_crit"]),
                           "K_at_m_t": float(prof[j]["K"]), "D_at_m_t": float(prof[j]["D"]), "m_grid": [grid_lo, grid_hi, int(ms.size)],
                           "m_t_on_grid_edge": bool(ms[j] in (grid_lo, grid_hi))}
    for lev in (0.01, 0.05):
        rec = ident[f"level_{lev:g}"]
        out[f"interval_{lev:g}"] = {"m_min": rec["m_min"], "m_max": rec["m_max"], "intervals": rec["intervals"], "n_intervals": rec["n_intervals"],
                                    "touches_grid_edge": bool(rec["m_min"] == grid_lo or rec["m_max"] == grid_hi)}
    return out


def site_theta(audit_run: str = AUDIT_RUN) -> dict[str, Any]:
    """Site-pooled theta at A_c = 1e3 and 1e4 m^2 (pooled and equal-weight chi estimators, 95 % basin-bootstrap band)."""
    p = json.loads((_results_dir(audit_run) / "params.json").read_text())
    out: dict[str, Any] = {"run": audit_run}
    for site in ("AP", "GM"):
        out[site] = {}
        for step in p["pooled"][site]["ladder"]:
            if step["A_c"] not in (1e3, 1e4):
                continue
            out[site][f"{step['A_c']:g}"] = {"n_basins": int(step["n_basins"]),
                                             **{est: {"theta": float(step[est]["theta_star"]), "ci95": [float(v) for v in step[est]["theta_ci95"]],
                                                      "at_edge": bool(step[est].get("theta_at_edge", False))} for est in ("pooled", "equal")}}
    return out


def run_mt_inputs(tiles, n_boot: int | None = None, audit_run: str = AUDIT_RUN, run_id: str | None = None, results_root=None) -> RunRecorder:
    """1.1 (m_t, intervals, grid-edge flags, the measured site theta beside them) and 1.2.1 (k_s and K0 at theta = m_t)."""
    from lemcvt.measure.diagnostics import N_BOOT, _metre_tile, edge_outlet_labels
    from lemcvt.measure.phase6_inputs import ks_at_theta

    n_boot = N_BOOT if n_boot is None else n_boot
    rec = RunRecorder("a12_inputs", f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    theta = site_theta(audit_run)
    rec.add_params("site_theta", theta)
    out: dict[str, Any] = {}
    rows = []
    for tile in tiles:
        cfg = load_tile_config(tile)
        site = cfg.site
        m_site = float(cfg.measured["m"])
        tm = tile_m(tile)
        td = _metre_tile(tile)
        Z, A, FD, dx, dy, U = td.Z, td.A, td.FD, td.dx, td.filled.dy, td.site.U
        n_exp = float(cfg.model.n)
        labels = edge_outlet_labels(A, FD)
        ks: dict[str, Any] = {}
        for a_c in (1e3, 1e4):
            r = ks_at_theta(Z, A, FD, dx, dy, tm["m_t"], a_c, labels, n_boot=n_boot)
            r.pop("_moments", None)
            r["K0"] = float(np.power(U / r["ks"], 1.0 / n_exp)) if r.get("ks") else None
            ks[f"{a_c:g}"] = r
        chk = ks_at_theta(Z, A, FD, dx, dy, m_site, 1e3, labels, n_boot=0)
        ks_check = {"ks_at_site_m": chk["ks"], "config_ks": float(cfg.measured["ks"]), "rel_diff": chk["ks"] / float(cfg.measured["ks"]) - 1}
        inside = {}
        for ac_key in ("1000", "10000"):
            for est in ("pooled", "equal"):
                lo_, hi_ = theta[site][ac_key][est]["ci95"]
                inside[f"{est}_{ac_key}"] = {"m_t_inside": bool(lo_ <= tm["m_t"] <= hi_), "m_site_inside": bool(lo_ <= m_site <= hi_)}
        out[tile] = {**tm, "site": site, "m_site": m_site, "U": U, "ks": ks, "K0_m_t": ks["1000"]["K0"], "K0_site_m": float(cfg.model.K0_partitioned),
                     "ks_check": ks_check, "inside_theta_bands": inside}
        rows.append({"tile": tile, "m_site": m_site, "m_t": tm["m_t"], "m_t_edge": tm["m_t_on_grid_edge"],
                     "i1_lo": tm["interval_0.01"]["m_min"], "i1_hi": tm["interval_0.01"]["m_max"], "i5_lo": tm["interval_0.05"]["m_min"],
                     "i5_hi": tm["interval_0.05"]["m_max"], "i5_edge": tm["interval_0.05"]["touches_grid_edge"], "ks_1e3": ks["1000"]["ks"],
                     "K0_m_t": ks["1000"]["K0"], "n_basins_1e3": ks["1000"]["n_basins"], "n_eff_1e3": ks["1000"].get("n_eff_centred"),
                     "ks_1e4": ks["10000"]["ks"], "ks_check_rel": ks_check["rel_diff"]})
        logger.info("%s: m_t %.2f (site %.2f), k_s(1e3) %.4g -> K0 %.4e; check at site m %.2e", tile, tm["m_t"], m_site, ks["1000"]["ks"] or float("nan"),
                    ks["1000"]["K0"] or float("nan"), ks_check["rel_diff"])
    rec.add_params("tiles", out)
    rec.write_table(pd.DataFrame(rows), "m_t")
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.finalize({"tiles": list(tiles), "n_boot": n_boot, "audit_run": audit_run})
    return rec


def latest_inputs() -> tuple[Path, dict[str, Any]]:
    runs = find_runs("a12_inputs")
    if not runs:
        raise FileNotFoundError("no a12_inputs run: run `lemcvt a12 inputs` first")
    return runs[-1], json.loads((runs[-1] / "params.json").read_text())


# ============================================================================= section 1.2.2-1.2.3: the runs at m_t

MT_STEP = "a12_mt"


def run_mt(tile: str, stage: str, inputs_run: str | Path | None = None, fill_interval: float = 0.05) -> list[RunRecorder]:
    """The v2 optimizer at (m_t, K0(m_t)): ``partitioned`` = thinned profile + staircase refinement + every unique area in
    the 5 % intervals (Phase 6); ``ad`` = the profile over r with Brent refinement. Runs are written with step ``a12_mt``."""
    from lemcvt.analyses.optimizer_v2 import refine_partitioned, run_tile_v2

    if inputs_run is None:
        _r, inp = latest_inputs()
    else:
        rr = Path(inputs_run) if Path(inputs_run).is_absolute() else _results_dir(str(inputs_run))
        inp = json.loads((rr / "params.json").read_text())
    t = inp["tiles"][tile]
    K0 = float(t["K0_m_t"])
    override = {"m": float(t["m_t"]), "K0_ad": K0, "K0_partitioned": K0}
    if stage == "ad":
        return [run_tile_v2(tile, stages=["ad"], model_override=override, step=MT_STEP)]
    if stage == "partitioned":
        rec = run_tile_v2(tile, stages=["partitioned"], acrit_grid="thin", model_override=override, step=MT_STEP, derived_grids=False)
        rec2 = refine_partitioned(tile, rec.dir, fill_interval=fill_interval, step=MT_STEP, derived_grids=False)
        return [rec, rec2]
    raise ValueError(f"stage must be 'ad' or 'partitioned', got {stage!r}")


def refine_mt(tile: str, fill_interval: float = 0.05) -> RunRecorder:
    """Resume: refine the latest thinned m_t partitioned run of the tile (when the refinement did not finish)."""
    from lemcvt.analyses.optimizer_v2 import refine_partitioned

    cands = [r for r in find_runs(MT_STEP, tile) if r.name.endswith("_v2_partitioned")]
    if not cands:
        raise FileNotFoundError(f"no thinned a12_mt partitioned run for {tile}")
    return refine_partitioned(tile, cands[-1], fill_interval=fill_interval, step=MT_STEP, derived_grids=False)


# ============================================================================= 3.1: the estimate

def _count_in(ua: np.ndarray, intervals) -> int:
    s: set[float] = set()
    for a_lo, a_hi in intervals:
        s |= set(ua[(ua >= a_lo) & (ua <= a_hi)].tolist())
    return len(s)


def run_estimate(tiles, run_id: str | None = None, results_root=None) -> RunRecorder:
    """Core-hours per task from the Phase 6 wall times of the same computations (per-evaluation cost x count).

    equal-part: (thinned grid + staircase + the capped fill) x the Phase 6 seconds per partitioned evaluation;
    equal-ad: (200 + 6 Brent refinements x 25) x the Phase 6 seconds per AD evaluation; m_t partitioned: the Phase 6
    thinned wall plus the fill at m_t (joint 5 % interval at m_t counted on the m-profile's 200-value grid, stage-2
    intervals as in Phase 6) x seconds per evaluation; m_t AD: the Phase 6 AD wall.
    """
    from lemcvt.config import data_root
    from lemcvt.io.grids import read_geotiff

    rec = RunRecorder("a12_estimate", f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    rows = []
    for tile in tiles:
        cfg = load_tile_config(tile)
        A = read_geotiff(data_root() / cfg.grids["area"]).data
        ua = unique_areas(A, *ACRIT_RANGE)
        runs = phase6_runs(tile)
        pname, pp = runs["partitioned"]
        part = pp["v2"]["partitioned"]
        src = json.loads((_results_dir(part["refinement"]["source_run"]) / "params.json").read_text())
        n_thin = int(src["v2"]["partitioned"]["n_evaluations"])
        s_eval = float(src["stages_results"]["acrit"]["wall_s"]) / n_thin
        fill = part["interval_fill"]
        n_fill_cols = {c: _count_in(ua, fill[c]) for c in ("mse_stage2", "mse_stage2_B", "mse_joint") if c in fill}
        aname, ap = runs["ad"]
        ad_eval = float(ap["stages_results"]["ad"]["wall_s"]) / float(ap["v2"]["ad"]["n_evaluations"])
        # joint 5 % interval at m_t on the m-profile grid
        tm = tile_m(tile)
        mp_rows = pd.read_parquet(_results_dir(tm["run"]) / "m_profile.parquet")
        sel = mp_rows[np.isclose(mp_rows.m, tm["m_t"])]
        ident = identifiability(cast(list[dict[str, Any]], sel.to_dict("records")), "mse_joint", levels=(0.05,))
        n_joint_mt = _count_in(ua, ident["level_0.05"]["intervals"])
        n_stair = 9 * 2 * max(1, int(round(ua.size / THIN_N)))
        n_fill_eq = min(FILL_MAX, ua.size)
        eq_part_h = (min(THIN_N, ua.size) + n_stair + n_fill_eq) * s_eval / 3600
        eq_ad_h = (AD_N_GRID + 6 * 25) * ad_eval / 3600
        mt_fill = n_joint_mt + n_fill_cols.get("mse_stage2", 0) + n_fill_cols.get("mse_stage2_B", 0)
        mt_part_thin_h = n_thin * s_eval / 3600
        mt_part_refine_h = (mt_fill + 400) * s_eval / 3600
        mt_ad_h = float(ap["stages_results"]["ad"]["wall_s"]) / 3600
        rows.append({"tile": tile, "n_unique": int(ua.size), "s_per_part_eval": s_eval, "s_per_ad_eval": ad_eval,
                     "phase6_fill": int(fill.get("n_unique_evaluated", 0)), "phase6_fill_by_col": n_fill_cols,
                     "n_joint_5pct_at_m_t": n_joint_mt, "m_t": tm["m_t"],
                     "equal_part_h": eq_part_h, "equal_ad_h": eq_ad_h, "mt_part_thin_h": mt_part_thin_h, "mt_part_refine_h": mt_part_refine_h,
                     "mt_part_h": mt_part_thin_h + mt_part_refine_h, "mt_ad_h": mt_ad_h})
    df = pd.DataFrame(rows)
    tot = {k: float(df[k].sum()) for k in ("equal_part_h", "equal_ad_h", "mt_part_h", "mt_ad_h")}
    tot["inputs_h"] = 1.0
    tot["summary_h"] = 0.5
    tot["total_h"] = float(sum(tot.values()))
    rec.add_params("rows", rows)
    rec.add_params("totals_expected_core_hours", tot)
    rec.write_table(df.drop(columns=["phase6_fill_by_col"]), "estimate")
    logger.info("estimate (expected core-hours): %s", {k: round(v, 1) for k, v in tot.items()})
    print(df.drop(columns=["phase6_fill_by_col"]).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print({k: round(v, 1) for k, v in tot.items()})
    rec.finalize({"tiles": list(tiles)})
    return rec


# ============================================================================= 2.3, 1.3, 1.4: the summary and the rules

def _latest(step: str, tile: str, suffix: str | None = None) -> tuple[Path, dict[str, Any]] | None:
    cands = [r for r in find_runs(step, tile) if suffix is None or r.name.endswith(suffix)]
    if not cands:
        return None
    return cands[-1], json.loads((cands[-1] / "params.json").read_text())


def _site_med(vals: list[float]) -> float:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def run_summary(tiles, run_id: str | None = None, results_root=None) -> RunRecorder:
    rec = RunRecorder("a12_summary", f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    used: dict[str, dict[str, str]] = {}
    ins_rows, held_rows, split_rows, side_rows = [], [], [], []
    inp_run, inp = None, None
    try:
        inp_run, inp = latest_inputs()
    except FileNotFoundError:
        pass
    for tile in tiles:
        used[tile] = {}
        site = tile[:2]
        part = _latest("a12_equal_part", tile)
        adr = _latest("a12_equal_ad", tile)
        if part is None or adr is None:
            logger.warning("%s: missing a12_equal_part or a12_equal_ad run", tile)
        if part is not None and adr is not None:
            used[tile] |= {"equal_part": part[0].name, "equal_ad": adr[0].name}
            pr, ar = part[1]["result"], adr[1]["result"]
            union = pr["insample_union_phase6"] or pr["insample"]
            row: dict[str, Any] = {"tile": tile, "site": site}
            for n in ("P1_A", "P1_B", "P2_A", "P2_B", "P3", "P4"):
                if n in union:
                    row[f"{n}_mse"] = union[n]["mse"]
                    row[f"{n}_A_crit"] = union[n]["A_crit"]
                    row[f"{n}_K"] = union[n]["K"]
                    row[f"{n}_D"] = union[n]["D"]
                    row[f"{n}_mse_new_rows"] = pr["insample"][n]["mse"]
            for n in ("AD", "A1", "A2_A", "A2_B"):
                if n in ar["insample"]:
                    row[f"{n}_mse"] = ar["insample"][n]["mse"]
                    row[f"{n}_K"] = ar["insample"][n]["K"]
                    row[f"{n}_D"] = ar["insample"][n]["D"]
                    row[f"{n}_K_at_bound"] = ar["insample"][n]["K_at_bound"]
                    row[f"{n}_D_at_bound"] = ar["insample"][n]["D_at_bound"]
            for end in ("A", "B"):
                if f"P1_{end}_mse" not in row or f"A2_{end}_mse" not in row:
                    continue
                c1 = min(row["A1_mse"], row[f"A2_{end}_mse"])
                row[f"count1_{end}_P1_wins"] = bool(row[f"P1_{end}_mse"] < c1)
                row[f"count2_{end}_P2_wins"] = bool(row[f"P2_{end}_mse"] < row["AD_mse"])
            row["count2_P3_wins"] = bool(row["P3_mse"] < row["AD_mse"])
            row["P4_beats_AD"] = bool(row["P4_mse"] < row["AD_mse"])
            row["phase6_check_part"] = pr["phase6_check"].get("max_rel_new_vs_phase6_P4")
            row["phase6_check_ad"] = ar["phase6_check"].get("rel_diff_mse")
            ins_rows.append(row)
            sp = part[1]["split"]
            split_rows.append({"tile": tile, "n_basins_edge": sp["n_basins_edge"], "n_interior_outlets": sp["n_interior_outlets"],
                               "n_basins_half1": sp["n_basins_half1"], "n_basins_half2": sp["n_basins_half2"],
                               "n_cells_half1": sp["n_cells_half1"], "n_cells_half2": sp["n_cells_half2"],
                               "share_half1": sp["share_half1_of_split"], "share_half2": sp["share_half2_of_split"],
                               "frac_valid_in_split": sp["frac_valid_in_split"], "largest_basin_share": sp["largest_basin_share_of_split"]})
            hrow: dict[str, Any] = {"tile": tile, "site": site, "AD_heldout": ar["heldout"]["AD"]["mse_heldout_mean"],
                                    "AD_h1_fit_score_h2": ar["heldout"]["AD"]["fit_h1"]["score_h2"], "AD_h2_fit_score_h1": ar["heldout"]["AD"]["fit_h2"]["score_h1"]}
            for n in HELDOUT_PARTITIONED:
                if n in pr["heldout"]:
                    h = pr["heldout"][n]
                    hrow[f"{n}_heldout"] = h["mse_heldout_mean"]
                    hrow[f"{n}_h1_fit_score_h2"] = h["fit_h1"]["score_h2"]
                    hrow[f"{n}_h2_fit_score_h1"] = h["fit_h2"]["score_h1"]
                    hrow[f"{n}_A_crit_h1"], hrow[f"{n}_A_crit_h2"] = h["fit_h1"]["A_crit"], h["fit_h2"]["A_crit"]
                    hrow[f"{n}_beats_AD"] = bool(h["mse_heldout_mean"] < hrow["AD_heldout"])
            held_rows.append(hrow)
        # 1.3 side by side
        p6 = phase6_runs(tile)
        mt_p = _latest(MT_STEP, tile, "_v2_partitioned-refined")
        mt_a = _latest(MT_STEP, tile, "_v2_ad")
        if inp is None or mt_p is None or mt_a is None or "partitioned" not in p6 or "ad" not in p6:
            logger.warning("%s: missing m_t runs or inputs", tile)
            continue
        used[tile] |= {"mt_partitioned": mt_p[0].name, "mt_ad": mt_a[0].name, "inputs": inp_run.name if inp_run else ""}
        cfg = load_tile_config(tile)
        D0A, D0B, K0s = float(cfg.model.D0), cfg.model.D0_B, float(cfg.model.K0_partitioned)
        K0t = float(inp["tiles"][tile]["K0_m_t"])
        for label, (pn, pp), (an, aa), K0 in (("site", p6["partitioned"], p6["ad"], K0s), ("m_t", (mt_p[0].name, mt_p[1]), (mt_a[0].name, mt_a[1]), K0t)):
            jo = pp["v2"]["partitioned"]["joint_optimum"]
            ao = aa["v2"]["ad"]["optimum"]
            side_rows.append({"tile": tile, "site": site, "which": label, "m": float(pp["model"]["m"]), "K0": K0, "A_crit": jo["A_crit"],
                              "K_over_K0": jo["K"] / K0, "D_over_D0_A": jo["D"] / D0A, "D_over_D0_B": (jo["D"] / D0B) if D0B else None,
                              "mse_partitioned": jo["mse"], "mse_AD": ao["mse"], "ratio_AD_over_part": ao["mse"] / jo["mse"],
                              "AD_K_over_K0": ao["K"] / K0, "AD_D_over_D0_A": ao["D"] / D0A, "AD_D_over_D0_B": (ao["D"] / D0B) if D0B else None,
                              "joint_where": jo.get("where"), "AD_where": ao.get("where"), "run_partitioned": pn, "run_ad": an,
                              "joint_interval_1pct": pp["v2"]["partitioned"]["joint_identifiability"]["level_0.01"]["intervals"]})
    ins = pd.DataFrame(ins_rows)
    held = pd.DataFrame(held_rows)
    split = pd.DataFrame(split_rows)
    side = pd.DataFrame(side_rows)
    rules: dict[str, Any] = {}
    for site in ("AP", "GM"):
        r: dict[str, Any] = {}
        if not ins.empty:
            s = ins[ins.site == site]
            for col in [c for c in s.columns if c.startswith("count") or c == "P4_beats_AD"]:
                if s[col].notna().any():
                    n_win = int(s[col].fillna(False).astype(bool).sum())
                    r[col] = {"n_tiles": n_win, "of": int(s[col].notna().sum()), "wins": bool(n_win >= RULES["equal_count_min_tiles"])}
        if not held.empty:
            h = held[held.site == site]
            for col in [c for c in h.columns if c.endswith("_beats_AD")]:
                if h[col].notna().any():
                    n_win = int(h[col].fillna(False).astype(bool).sum())
                    r[f"heldout_{col}"] = {"n_tiles": n_win, "of": int(h[col].notna().sum()), "wins": bool(n_win >= RULES["heldout_min_tiles"])}
        if not side.empty:
            a = side[(side.site == site) & (side.which == "site")]
            b = side[(side.site == site) & (side.which == "m_t")]
            if len(b):
                n_mse = int((b.mse_partitioned < b.mse_AD).sum())
                med = {}
                for col in ("D_over_D0_A", "AD_K_over_K0", "K_over_K0", "AD_D_over_D0_A", "D_over_D0_B"):
                    if a[col].notna().any():
                        m0, m1 = _site_med(a[col].tolist()), _site_med(b[col].tolist())
                        med[col] = {"site_m": m0, "m_t": m1, "rel_change": m1 / m0 - 1}
                c1 = n_mse >= RULES["m_immaterial_mse_tiles"]
                c2 = abs(med["D_over_D0_A"]["rel_change"]) < RULES["m_immaterial_median_change"] and abs(med["AD_K_over_K0"]["rel_change"]) < RULES["m_immaterial_median_change"]
                r["m_rule"] = {"n_part_below_AD_at_m_t": n_mse, "of": int(len(b)), "cond_mse": bool(c1), "medians": med, "cond_medians": bool(c2),
                               "immaterial": bool(c1 and c2),
                               "tiles_AD_not_above_part_at_m_t": b.loc[b.mse_partitioned >= b.mse_AD, "tile"].tolist()}
        rules[site] = r
    rec.add_params("rules", RULES)
    rec.add_params("site_rules", rules)
    rec.add_params("runs_used", used)
    if inp is not None:
        rec.add_params("inputs_run", inp_run.name if inp_run else None)
        rec.add_params("m_t", {t: {k: inp["tiles"][t][k] for k in ("m_t", "m_site", "interval_0.01", "interval_0.05", "m_t_on_grid_edge", "inside_theta_bands",
                                                              "K0_m_t", "K0_site_m", "ks_check")} | {"ks_1e3": inp["tiles"][t]["ks"]["1000"].get("ks"),
                                                                                                    "ks_1e3_n_eff": inp["tiles"][t]["ks"]["1000"].get("n_eff_centred"),
                                                                                                    "ks_1e3_ci95": inp["tiles"][t]["ks"]["1000"].get("ks_ci95"),
                                                                                                    "ks_1e4": inp["tiles"][t]["ks"]["10000"].get("ks")}
                                for t in tiles if t in inp["tiles"]})
        rec.add_params("site_theta", inp["site_theta"])
    for name, df in (("insample", ins), ("heldout", held), ("split", split), ("side_by_side", side)):
        if not df.empty:
            rec.write_table(df.astype({c: str for c in df.columns if df[c].dtype == object}), name)
    rec.finalize({"tiles": list(tiles)})
    return rec
