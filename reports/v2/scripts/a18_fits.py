#!/usr/bin/env python3
"""Addendum 18, section 5: the fits of one (tile, group) specification, from the exact reductions of addendum 12.

One partitioned run per spec gives, from the Gram terms of every A_crit row (thin 2000 + the staircase around every variant's
minimum + the 1 % interval of every variant, capped; addendum 12 section 2 / `lemcvt.analyses.sensitivity`):
    P4 [K, D, A_crit], P3 [D, A_crit] (K = K0), and, where the tile has a D0 (decision 4), P2_A/P2_B [K, A_crit] (D = D0_A / D0_B) and
    P1_A/P1_B [A_crit] (K0, D0); the held-out halves (5.3: greedy basin split at the 1e3 m2 network, fit on one half, score on the other)
    for P4, P2_A, P2_B; the 1 % and 5 % A_crit intervals of P4.
One AD run per spec gives AD [K, D] (profile over r = K/D, 200 points, bounded Brent), A1 [D] (K = K0), A2_A/A2_B [K] (D = D0_A/D0_B),
and the AD held-out halves.
Spec options (prep/fit_specs.json): the exponent m and K0 (site m, m_t, the v2.0.0 setting), D0_A/D0_B or none, `union_phase6`
(the stored Phase 6 rows of the v2.0.0 run join the row set and the optimum is checked against the release: 4.1, 5.5), `trunk`
(decision 8: cells with A >= 1 km2 are anchors held at their observed elevation in both models and left out of every MSE).
Model code is unchanged: the trunk anchors are extra boundary cells in the chi integration (`precompute_chi_network` with the same BFS)
and extra Dirichlet rows of the assembled AD matrices.
usage: a18_fits.py --spec TILE:GROUP --stage part|ad [--pilot] [--thin-n 2000 --fill-max 1500 --refine-max 1000 --ad-grid 200]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_common import (  # noqa: E402
    ACRIT_RANGE,
    AD_N_GRID,
    FILL_LEVEL,
    FILL_MAX,
    FIT_STEPS,
    HOLDOUT_A_C,
    REFINE_MAX,
    THIN_N,
    TRUNK_A_M2,
    run_dir,
)

from lemcvt.analyses.sensitivity import (  # noqa: E402
    _fit,
    _q,
    _results_dir,
    _terms_from_phase6_row,
    _terms_masked,
    greedy_basin_halves,
    phase6_runs,
)
from lemcvt.config import config_snapshot, load_tile  # noqa: E402
from lemcvt.models.common import _bfs_integrate, _boundary_anchor, _receivers_vectorised, _upstream_csr  # noqa: E402
from lemcvt.optimize.v1 import Trajectory  # noqa: E402
from lemcvt.optimize.v2 import ADProfiler, PartitionedProfiler, identifiability, thin_log_uniform, unique_areas  # noqa: E402
from lemcvt.results import RunRecorder  # noqa: E402

logger = logging.getLogger("lemcvt.a18")
HELDOUT_PARTITIONED = ("P4", "P2_A", "P2_B")


# ============================================================================= trunk anchors (decision 8)

def chi_network_with_anchors(gd: dict, m: float, extra_anchor_flat: np.ndarray):
    """`precompute_chi_network` with extra anchor cells (chi = 0, z = observed) beside the tile boundary: the same receivers, donor
    lists, step and BFS; the anchors seed the integration, so every tributary of a trunk cell starts from that cell's observed elevation."""
    n = gd["n"]
    downstream_of, dL = _receivers_vectorised(gd)
    bc_inds, bc_vals, k_out = _boundary_anchor(gd)
    extra = np.setdiff1d(np.asarray(extra_anchor_flat, dtype=np.int64), bc_inds)
    bc2 = np.concatenate([np.asarray(bc_inds, dtype=np.int64), extra])
    vals2 = bc_vals.copy()
    obs = gd["Z0"][gd["rev_j"], gd["rev_i"]].astype(float)
    vals2[extra] = obs[extra]
    donors, starts = _upstream_csr(downstream_of, n)
    step = (1.0 / (gd["A_flat"] ** m)) * dL
    chi = np.full(n, np.nan)
    chi[bc2] = 0.0
    chi = _bfs_integrate(bc2, donors, starts, step, chi)
    z_anchor = np.full(n, np.nan)
    z_anchor[bc2] = vals2[bc2]
    z_anchor = _bfs_integrate(bc2, donors, starts, np.where(np.isfinite(step), 0.0, np.nan), z_anchor)
    return chi, z_anchor, downstream_of.astype(int), k_out, bc2, vals2


class TrunkProfiler(PartitionedProfiler):
    """The partitioned reduction with the trunk cells as fixed anchors (they are channel cells at every A_crit of the grid)."""

    def __init__(self, Z0, A, FD, dx, U, trunk_flat: np.ndarray, m_values=()):
        self._trunk_flat = np.asarray(trunk_flat, dtype=np.int64)
        super().__init__(Z0, A, FD, dx, U, m_values=m_values)

    def add_m(self, m: float) -> None:
        m = float(m)
        if m in self.chi:
            return
        chi, z_anchor, _ds, k_out, bc_inds, bc_vals = chi_network_with_anchors(self.gd, m, self._trunk_flat)
        self.chi[m] = chi
        if self.z_anchor is None:
            self.z_anchor, self.k_out, self.bc_inds, self.bc_vals = z_anchor, k_out, bc_inds, bc_vals


def ad_with_anchors(ad, obs: np.ndarray, extra_flat: np.ndarray):
    """The assembled AD matrices with extra Dirichlet rows on ``extra_flat`` (value = observed elevation, no uplift source)."""
    from scipy.sparse import csr_matrix

    M_diff0, M_adv0, I_bc = ad.matrices
    bc_inds, bc_vals, b_U = ad.bc
    n = b_U.size
    extra = np.setdiff1d(np.asarray(extra_flat, dtype=np.int64), bc_inds)
    keep = np.ones(n)
    keep[extra] = 0.0
    ar = np.arange(n)
    R = csr_matrix((keep, (ar, ar)), shape=(n, n))
    M_diff1 = (R @ M_diff0).tocsr()
    M_adv1 = (R @ M_adv0).tocsr()
    M_diff1.eliminate_zeros()
    M_adv1.eliminate_zeros()
    I1 = (I_bc + csr_matrix((np.ones(extra.size), (extra, extra)), shape=(n, n))).tocsr()
    vals = bc_vals.copy()
    vals[extra] = obs[extra]
    bU = b_U.copy()
    bU[extra] = 0.0
    inds = np.concatenate([np.asarray(bc_inds, dtype=np.int64), extra])
    return SimpleNamespace(matrices=(M_diff1, M_adv1, I1), bc=(inds, vals, bU)), extra.size


# ============================================================================= variants

def variants(K0: float, D0A: float | None, D0B: float | None) -> dict[str, tuple[str, float | None, int]]:
    out: dict[str, tuple[str, float | None, int]] = {"P3": ("fix_x", None, 2), "P4": ("joint", None, 3)}
    if D0A is not None:
        out["P1_A"] = ("fix_xy", D0A, 1)
        out["P2_A"] = ("fix_y", D0A, 2)
    if D0B is not None:
        out["P1_B"] = ("fix_xy", D0B, 1)
        out["P2_B"] = ("fix_y", D0B, 2)
    return out


def row_objectives(terms: dict[str, dict[str, float]], U: float, K0: float, D0A, D0B, lo, hi) -> dict[str, Any]:
    x0 = U / K0
    out: dict[str, Any] = {}
    for name, (kind, Dfix, _c) in variants(K0, D0A, D0B).items():
        yf = U / Dfix if Dfix is not None else None
        for part in ("all", "h1", "h2"):
            if part not in terms or (part != "all" and name not in HELDOUT_PARTITIONED):
                continue
            t = terms[part]
            if not np.isfinite(t.get("rr", np.nan)) or t.get("n", 0) == 0:
                out[f"mse_{name}_{part}"] = float("nan")
                continue
            x, y, where = _fit(t, kind, lo, hi, x0, yf)
            out[f"mse_{name}_{part}"] = _q(t, x, y)
            out[f"K_{name}_{part}"] = U / x
            out[f"D_{name}_{part}"] = U / y
            out[f"where_{name}_{part}"] = where
            if part in ("h1", "h2"):
                other = "h2" if part == "h1" else "h1"
                out[f"mse_{name}_{part}_score_{other}"] = _q(terms[other], x, y)
    return out


def _opt(rows: list[dict[str, Any]], name: str, part: str = "all") -> dict[str, Any] | None:
    col = f"mse_{name}_{part}"
    cand = [q for q in rows if np.isfinite(q.get(col, np.nan))]
    if not cand:
        return None
    r = min(cand, key=lambda q: q[col])
    out = {"A_crit": float(r["A_crit"]), "mse": float(r[col]), "K": float(r[f"K_{name}_{part}"]), "D": float(r[f"D_{name}_{part}"]),
           "where": r[f"where_{name}_{part}"], "n_channel": int(r["n_channel"]), "source": r.get("kind", "")}
    if part in ("h1", "h2"):
        other = "h2" if part == "h1" else "h1"
        out[f"score_{other}"] = float(r[f"mse_{name}_{part}_score_{other}"])
    return out


# ============================================================================= the partitioned run

def run_part(spec: dict, a) -> RunRecorder:
    tile, group = spec["tile"], spec["group"]
    td = load_tile(tile)
    mp = td.cfg.model
    Z0, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    U = float(spec["E"])
    assert abs(U - td.site.U) < 1e-15, (U, td.site.U)
    m, K0, D0A, D0B = float(spec["m"]), float(spec["K0"]), spec.get("D0A"), spec.get("D0B")
    rec = RunRecorder(FIT_STEPS["part"], f"{tile}_{group}", run_id=a.run_id)
    t0 = time.perf_counter()
    rec.add_params("tile", tile)
    rec.add_params("group", group)
    rec.add_params("spec", spec)
    rec.add_params("inputs", td.provenance())
    rec.add_params("model", {"m": m, "U": U, "K0": K0, "D0_A": D0A, "D0_B": D0B, "K_bounds": mp.K_bounds, "D_bounds": mp.D_bounds,
                             "z_units_working": td.z_units_working, "variants": list(variants(K0, D0A, D0B))})
    half, hinfo = greedy_basin_halves(Z0, A, FD, dx, dy, A_c=HOLDOUT_A_C)
    rec.add_params("split", hinfo)
    trunk_mask = np.isfinite(A) & (A >= TRUNK_A_M2) if spec.get("trunk") else np.zeros(A.shape, dtype=bool)
    if spec.get("trunk"):
        gd0 = PartitionedProfiler(Z0, A, FD, dx, U).gd
        trunk_flat = np.where(trunk_mask[gd0["rev_j"], gd0["rev_i"]])[0]
        prof: PartitionedProfiler = TrunkProfiler(Z0, A, FD, dx, U, trunk_flat, m_values=[m])
    else:
        prof = PartitionedProfiler(Z0, A, FD, dx, U, m_values=[m])
    gd = prof.gd
    keep_all = ~trunk_mask[gd["rev_j"], gd["rev_i"]]
    half_flat = half[gd["rev_j"], gd["rev_i"]]
    keep = {"all": keep_all, "h1": (half_flat == 1) & keep_all, "h2": (half_flat == 2) & keep_all}
    rec.add_params("trunk", {"applied": bool(spec.get("trunk")), "A_trunk_m2": TRUNK_A_M2, "n_cells": int(trunk_mask.sum()), "frac_valid": float(trunk_mask.sum() / np.isfinite(A).sum()),
                             "max_A_m2": float(np.nanmax(A)), "n_anchor_cells": int(prof.bc_inds.size) if prof.bc_inds is not None else None,
                             "mse_excludes_trunk_cells": True})
    lo = np.array([U / mp.K_bounds[1], U / mp.D_bounds[1]])
    hi = np.array([U / mp.K_bounds[0], U / mp.D_bounds[0]])
    ua = unique_areas(A, *ACRIT_RANGE)
    grid = thin_log_uniform(ua, a.thin_n)
    rows: list[dict[str, Any]] = []
    done: set[float] = set()
    traj = Trajectory(FIT_STEPS["part"], ("log10_A_crit", "log10_D", "log10_K", "m", "kind"))
    walls: list[float] = []

    def _eval(a_c: float, kind: str) -> None:
        a_c = float(a_c)
        if a_c in done:
            return
        t1 = time.perf_counter()
        red = prof.reduce(a_c, m)
        terms = {p: _terms_masked(red, prof.Z_obs_flat, k) for p, k in keep.items()}
        row: dict[str, Any] = {"A_crit": a_c, "kind": kind, "n_channel": int(red.channel_mask.sum()), "n_hillslope": red.n_hillslope, "wall_s": red.wall_s}
        for part, t in terms.items():
            row.update({f"{k}_{part}": v for k, v in t.items()})
        row.update(row_objectives(terms, U, K0, D0A, D0B, lo, hi))
        rows.append(row)
        done.add(a_c)
        walls.append(time.perf_counter() - t1)
        if np.isfinite(row.get("mse_P4_all", np.nan)):
            traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(row["D_P4_all"]), "log10_K": math.log10(row["K_P4_all"]), "m": m, "kind": kind},
                        row["mse_P4_all"], cached=False)

    for i, a_c in enumerate(grid):
        _eval(a_c, "grid")
        if i % 200 == 0 or i == grid.size - 1:
            logger.info("%s:%s grid %d/%d A_crit %.1f [%.0f s]", tile, group, i + 1, grid.size, a_c, time.perf_counter() - t0)
    fit_cols = [c for c in rows[0] if c.startswith("mse_") and "_score_" not in c and np.isfinite(rows[0].get(c, np.nan))]
    n_extra_targets = 0
    extra_a = np.array([])
    fill_info: dict[str, Any] = {"level": a.fill_level, "max": a.fill_max, "intervals": {}, "n_targets": 0, "n_evaluated": 0, "capped": False}
    if not a.pilot:
        gs = np.sort(grid)
        extra: set[float] = set()
        for col in fit_cols:
            cand = [r for r in rows if np.isfinite(r.get(col, np.nan))]
            if not cand:
                continue
            a_min = min(cand, key=lambda r: r[col])["A_crit"]
            j = int(np.searchsorted(gs, a_min))
            lo_a, hi_a = gs[max(j - 1, 0)], gs[min(j + 1, gs.size - 1)]
            extra |= set(ua[(ua > lo_a) & (ua < hi_a)].tolist())
        extra -= done
        extra_a = np.array(sorted(extra))
        n_extra_targets = int(extra_a.size)
        if extra_a.size > a.refine_max:
            extra_a = thin_log_uniform(extra_a, a.refine_max)
        for a_c in extra_a:
            _eval(a_c, "refine")
        logger.info("%s:%s staircase refinement: %d of %d values [%.0f s]", tile, group, extra_a.size, n_extra_targets, time.perf_counter() - t0)
        targets: set[float] = set()
        for col in fit_cols:
            cand = [r for r in rows if np.isfinite(r.get(col, np.nan))]
            ident = identifiability(cand, col, levels=(a.fill_level,))
            ivs = ident[f"level_{a.fill_level:g}"]["intervals"]
            fill_info["intervals"][col] = ivs
            for a_lo, a_hi in ivs:
                targets |= set(ua[(ua >= a_lo) & (ua <= a_hi)].tolist())
        targets -= done
        fill_a = np.array(sorted(targets))
        fill_info["n_targets"] = int(fill_a.size)
        if fill_a.size > a.fill_max:
            fill_a = thin_log_uniform(fill_a, a.fill_max)
        fill_info["n_evaluated"] = int(fill_a.size)
        fill_info["capped"] = bool(fill_info["n_targets"] > a.fill_max)
        for i, a_c in enumerate(fill_a):
            _eval(a_c, "fill")
            if i % 250 == 0:
                logger.info("%s:%s fill %d/%d [%.0f s]", tile, group, i + 1, fill_a.size, time.perf_counter() - t0)
    rec.add_trajectory(traj)
    df = pd.DataFrame(rows).sort_values("A_crit").reset_index(drop=True)
    rec.write_table(df, "acrit_profile")
    recs = cast(list[dict[str, Any]], df.to_dict("records"))
    names = list(variants(K0, D0A, D0B))
    insample = {n: _opt(recs, n) for n in names}
    ident_all = {n: identifiability([r for r in recs if np.isfinite(r.get(f"mse_{n}_all", np.nan))], f"mse_{n}_all") for n in names}
    # union with the stored Phase 6 rows (4.1 gate; 5.5 continuity against v2.0.0): whole-tile terms at the same m, K0, D0
    union: dict[str, Any] = {}
    check: dict[str, Any] = {"requested": bool(spec.get("union_phase6")), "available": False}
    if spec.get("union_phase6") and not a.pilot:
        p6 = phase6_runs(tile).get("partitioned")
        if p6 is not None:
            p6m = p6[1]["model"]
            same = (abs(float(p6m["m"]) - m) < 1e-12 and abs(float(p6m["K0_partitioned"]) / K0 - 1) < 1e-12 and D0A is not None
                    and abs(float(p6m["D0"]) / D0A - 1) < 1e-12 and not spec.get("trunk"))
            check.update({"available": True, "run": p6[0], "same_model": bool(same), "phase6_model": {k: p6m.get(k) for k in ("m", "K0_partitioned", "D0", "D0_B")}})
            if same:
                prof6 = pd.read_parquet(_results_dir(p6[0]) / "acrit_profile.parquet")
                rows6 = []
                for r in cast(list[dict[str, Any]], prof6.to_dict("records")):
                    obj = row_objectives({"all": _terms_from_phase6_row(r)}, U, K0, D0A, D0B, lo, hi)
                    rows6.append({"A_crit": float(r["A_crit"]), "n_channel": int(r["n_channel"]), "kind": "phase6", **obj, "_p6_joint": r["mse_joint"], "_p6_s2": r["mse_stage2"]})
                rel_j = max(abs(q["mse_P4_all"] / q["_p6_joint"] - 1) for q in rows6)
                rel_s = max(abs(q["mse_P1_A_all"] / q["_p6_s2"] - 1) for q in rows6)
                by_a6 = {q["A_crit"]: q for q in rows6}
                common = [q for q in recs if q["A_crit"] in by_a6]
                rel_new = max((abs(q["mse_P4_all"] / by_a6[q["A_crit"]]["mse_P4_all"] - 1) for q in common), default=float("nan"))
                jo6 = p6[1]["v2"]["partitioned"]["joint_optimum"]
                allrows = rows6 + [q for q in recs if q["A_crit"] not in by_a6]
                union = {n: _opt(allrows, n) for n in names}
                union["_n_rows"] = len(allrows)
                union["_ident"] = {n: identifiability([r for r in allrows if np.isfinite(r.get(f"mse_{n}_all", np.nan))], f"mse_{n}_all") for n in names}
                u4 = union["P4"]
                check.update({"n_rows_phase6": len(rows6), "n_common": len(common), "max_rel_stored_terms_vs_stored_joint": rel_j,
                              "max_rel_stored_terms_vs_stored_stage2": rel_s, "max_rel_new_vs_phase6_P4_same_A_crit": rel_new,
                              "phase6_joint_optimum": {k: jo6[k] for k in ("A_crit", "K", "D", "mse")},
                              "phase6_stage2_optimum": {k: p6[1]["v2"]["partitioned"]["stage2_optimum"][k] for k in ("A_crit", "mse")},
                              "union_P4": {k: u4[k] for k in ("A_crit", "K", "D", "mse")},
                              "rel_diff_union_vs_phase6": {k: abs(u4[k] / jo6[k] - 1) for k in ("A_crit", "K", "D", "mse")},
                              "new_rows_only_P4": {k: insample["P4"][k] for k in ("A_crit", "K", "D", "mse")},
                              "rel_diff_new_rows_only_vs_phase6": {k: abs(insample["P4"][k] / jo6[k] - 1) for k in ("A_crit", "K", "D", "mse")}})
    held: dict[str, Any] = {}
    for n in HELDOUT_PARTITIONED:
        if n not in names or f"mse_{n}_h1" not in recs[0]:
            continue
        f1, f2 = _opt(recs, n, "h1"), _opt(recs, n, "h2")
        if f1 is None or f2 is None:
            continue
        held[n] = {"fit_h1": f1, "fit_h2": f2, "mse_heldout_mean": 0.5 * (f1["score_h2"] + f2["score_h1"])}
    result = {"insample": insample, "identifiability": ident_all, "insample_union_phase6": union, "phase6_check": check, "heldout": held,
              "param_count": {n: v[2] for n, v in variants(K0, D0A, D0B).items()}, "K0": K0, "D0_A": D0A, "D0_B": D0B, "m": m,
              "fill": fill_info, "staircase": {"n_targets": n_extra_targets, "n_evaluated": int(extra_a.size), "max": a.refine_max, "capped": bool(n_extra_targets > a.refine_max)},
              "n_evaluations": len(recs), "n_unique_areas": int(ua.size), "thin_n": a.thin_n, "pilot": bool(a.pilot),
              "timing": {"s_per_eval_mean": float(np.mean(walls)), "s_per_eval_median": float(np.median(walls)), "n_timed": len(walls),
                         "wall_total_s": time.perf_counter() - t0}}
    rec.add_params("result", result)
    rec.add_params("wall_s", time.perf_counter() - t0)
    logger.info("%s:%s P4 A_crit %.0f mse %.4f K/K0 %.3f D %.4g; P3 %.4f; %d evaluations, %.2f s each [%.0f s]", tile, group, insample["P4"]["A_crit"],
                insample["P4"]["mse"], insample["P4"]["K"] / K0, insample["P4"]["D"], insample["P3"]["mse"], len(recs), float(np.mean(walls)), time.perf_counter() - t0)
    rec.finalize(config_snapshot(td.cfg, {"thin_n": a.thin_n, "fill_level": a.fill_level, "fill_max": a.fill_max, "refine_max": a.refine_max, "pilot": a.pilot, "spec": spec}))
    return rec


# ============================================================================= the AD run

def run_ad(spec: dict, a) -> RunRecorder:
    from scipy.optimize import minimize_scalar

    from lemcvt.models.ad import advection_diffusion_model
    from lemcvt.models.common import extract_grid_data

    tile, group = spec["tile"], spec["group"]
    td = load_tile(tile)
    mp = td.cfg.model
    Z0, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
    U = float(spec["E"])
    assert abs(U - td.site.U) < 1e-15
    m, K0, D0A, D0B = float(spec["m"]), float(spec["K0"]), spec.get("D0A"), spec.get("D0B")
    (K_lo, K_hi), (D_lo, D_hi) = mp.K_bounds, mp.D_bounds
    rec = RunRecorder(FIT_STEPS["ad"], f"{tile}_{group}", run_id=a.run_id)
    t0 = time.perf_counter()
    rec.add_params("tile", tile)
    rec.add_params("group", group)
    rec.add_params("spec", spec)
    rec.add_params("inputs", td.provenance())
    rec.add_params("model", {"m": m, "U": U, "K0": K0, "D0_A": D0A, "D0_B": D0B, "K_bounds": mp.K_bounds, "D_bounds": mp.D_bounds, "z_units_working": td.z_units_working})
    half, hinfo = greedy_basin_halves(Z0, A, FD, dx, dy, A_c=HOLDOUT_A_C)
    rec.add_params("split", hinfo)
    ad = advection_diffusion_model(Z0, A, FD, dx, m)
    gd = extract_grid_data(Z0, A, FD, dx)
    obs = Z0[gd["rev_j"], gd["rev_i"]].astype(float)
    trunk_mask = np.isfinite(A) & (A >= TRUNK_A_M2) if spec.get("trunk") else np.zeros(A.shape, dtype=bool)
    keep_all = ~trunk_mask[gd["rev_j"], gd["rev_i"]]
    n_extra = 0
    if spec.get("trunk"):
        ad, n_extra = ad_with_anchors(ad, obs, np.where(~keep_all)[0])
    rec.add_params("trunk", {"applied": bool(spec.get("trunk")), "A_trunk_m2": TRUNK_A_M2, "n_cells": int(trunk_mask.sum()), "n_extra_dirichlet": n_extra,
                             "frac_valid": float(trunk_mask.sum() / np.isfinite(A).sum()), "mse_excludes_trunk_cells": True})
    prof = ADProfiler(ad, obs, U)
    half_flat = half[gd["rev_j"], gd["rev_i"]]
    masks = {"all": keep_all, "h1": (half_flat == 1) & keep_all, "h2": (half_flat == 2) & keep_all}
    cache: dict[float, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    walls: list[float] = []
    traj = Trajectory(FIT_STEPS["ad"], ("log10_K", "log10_D", "log10_r", "kind"))

    def _mse_y(res, g, ok, y):
        e = res[ok] - y * g[ok]
        return float(e @ e / ok.sum())

    def evaluate(r: float, kind: str) -> dict[str, Any]:
        r = float(r)
        if r in cache:
            return cache[r]
        t1 = time.perf_counter()
        red = prof.reduce(r)
        res = obs - red["h"]
        g = red["g"]
        fin = np.isfinite(res) & np.isfinite(g)
        row: dict[str, Any] = {"r": r, "kind": kind, "wall_s": red["wall_s"]}
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
        ok = fin & masks["all"]
        D1 = K0 / r
        row["mse_A1_all"] = _mse_y(res, g, ok, U / D1) if D_lo <= D1 <= D_hi else float("inf")
        row["K_A1_all"], row["D_A1_all"] = K0, D1
        for suf, D0 in (("A", D0A), ("B", D0B)):
            if D0 is None:
                continue
            K2 = r * D0
            row[f"mse_A2_{suf}_all"] = _mse_y(res, g, ok, U / D0) if K_lo <= K2 <= K_hi else float("inf")
            row[f"K_A2_{suf}_all"], row[f"D_A2_{suf}_all"] = K2, D0
        cache[r] = row
        rows.append(row)
        walls.append(time.perf_counter() - t1)
        if np.isfinite(row["mse_AD_all"]):
            traj.record({"log10_K": math.log10(row["K_AD_all"]), "log10_D": math.log10(row["D_AD_all"]), "log10_r": math.log10(r), "kind": kind}, row["mse_AD_all"], cached=False)
        return row

    r_lo, r_hi = K_lo / D_hi, K_hi / D_lo
    grid = np.logspace(math.log10(r_lo), math.log10(r_hi), a.ad_grid)
    for i, r in enumerate(grid):
        evaluate(r, "grid")
        if i % 25 == 0 or i == a.ad_grid - 1:
            logger.info("%s:%s AD grid %d/%d r=%.3e [%.0f s]", tile, group, i + 1, a.ad_grid, r, time.perf_counter() - t0)
    objectives = ["mse_AD_all", "mse_AD_h1", "mse_AD_h2", "mse_A1_all"] + [f"mse_A2_{s}_all" for s, D0 in (("A", D0A), ("B", D0B)) if D0 is not None]
    brent: dict[str, Any] = {}
    if not a.pilot:
        for col in objectives:
            feas = sorted((q for q in rows if q["kind"] == "grid" and np.isfinite(q[col])), key=lambda q: q["r"])
            if not feas:
                brent[col] = {"feasible": False}
                continue
            jj = int(np.argmin([q[col] for q in feas]))
            lg = np.log10([q["r"] for q in feas])
            lo_, hi_ = lg[max(jj - 1, 0)], lg[min(jj + 1, len(lg) - 1)]
            info: dict[str, Any] = {"grid_index": jj, "n_feasible_grid": len(feas), "at_feasible_edge": bool(jj in (0, len(feas) - 1))}
            if lo_ < hi_:
                res_b = minimize_scalar(lambda lr, c=col: evaluate(10 ** lr, f"brent_{c}")[c], bounds=(lo_, hi_), method="bounded", options={"xatol": 1e-6})
                info.update({"log10_r": float(res_b.x), "nfev": int(res_b.nfev), "success": bool(res_b.success), "bracket_log10": [float(lo_), float(hi_)]})
            brent[col] = info
    rec.add_trajectory(traj)
    df = pd.DataFrame(rows).sort_values("r").reset_index(drop=True)
    rec.write_table(df, "r_profile")
    recs = cast(list[dict[str, Any]], df.to_dict("records"))

    def _best(name: str, part: str) -> dict[str, Any] | None:
        col = f"mse_{name}_{part}"
        cand = [x for x in recs if np.isfinite(x.get(col, np.inf))]
        if not cand:
            return None
        q = min(cand, key=lambda x: x[col])
        K, D = float(q[f"K_{name}_{part}"]), float(q[f"D_{name}_{part}"])
        out = {"r": float(q["r"]), "mse": float(q[col]), "K": K, "D": D, "where": q.get(f"where_{name}_{part}"),
               "K_at_bound": bool(np.isclose(K, K_lo, rtol=1e-9) or np.isclose(K, K_hi, rtol=1e-9)), "D_at_bound": bool(np.isclose(D, D_lo, rtol=1e-9) or np.isclose(D, D_hi, rtol=1e-9))}
        if part in ("h1", "h2"):
            other = "h2" if part == "h1" else "h1"
            out[f"score_{other}"] = float(q[f"mse_{name}_{part}_score_{other}"])
        return out

    insample = {"AD": _best("AD", "all"), "A1": _best("A1", "all")}
    for suf, D0 in (("A", D0A), ("B", D0B)):
        if D0 is not None:
            insample[f"A2_{suf}"] = _best(f"A2_{suf}", "all")
    f1, f2 = _best("AD", "h1"), _best("AD", "h2")
    held = {"AD": {"fit_h1": f1, "fit_h2": f2, "mse_heldout_mean": 0.5 * (f1["score_h2"] + f2["score_h1"])}} if f1 and f2 else {}
    grid_rows = [{"r": q["r"], "mse": q["mse_AD_all"]} for q in recs if np.isfinite(q["mse_AD_all"])]
    ident = identifiability(grid_rows, "mse", key="r")
    check: dict[str, Any] = {"requested": bool(spec.get("union_phase6")), "available": False}
    if spec.get("union_phase6") and not a.pilot:
        p6 = phase6_runs(tile).get("ad")
        if p6 is not None:
            o6 = p6[1]["v2"]["ad"]["optimum"]
            p6m = p6[1]["model"]
            check = {"requested": True, "available": True, "run": p6[0], "same_m": bool(abs(float(p6m["m"]) - m) < 1e-12), "phase6": {k: o6[k] for k in ("K", "D", "mse", "r")},
                     "new": {k: insample["AD"][k] for k in ("K", "D", "mse", "r")},
                     "rel_diff": {k: abs(insample["AD"][k] / o6[k] - 1) for k in ("K", "D", "mse", "r")}}
    result = {"insample": insample, "heldout": held, "brent": brent, "identifiability_r": ident, "phase6_check": check, "K0": K0, "D0_A": D0A, "D0_B": D0B, "m": m,
              "n_evaluations": len(recs), "pilot": bool(a.pilot),
              "timing": {"s_per_eval_mean": float(np.mean(walls)), "s_per_eval_median": float(np.median(walls)), "n_timed": len(walls), "wall_total_s": time.perf_counter() - t0}}
    rec.add_params("result", result)
    rec.add_params("wall_s", time.perf_counter() - t0)
    logger.info("%s:%s AD mse %.4f K/K0 %.3f D %.4g; A1 %.4f; %d evaluations, %.1f s each [%.0f s]", tile, group, insample["AD"]["mse"], insample["AD"]["K"] / K0,
                insample["AD"]["D"], insample["A1"]["mse"] if insample["A1"] else float("nan"), len(recs), float(np.mean(walls)), time.perf_counter() - t0)
    rec.finalize(config_snapshot(td.cfg, {"n_grid": a.ad_grid, "pilot": a.pilot, "spec": spec}))
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="TILE:GROUP key of prep/fit_specs.json")
    p.add_argument("--stage", required=True, choices=["part", "ad"])
    p.add_argument("--pilot", action="store_true", help="timing only: the thinned grid without refinement or fill; no Brent")
    p.add_argument("--thin-n", type=int, default=THIN_N)
    p.add_argument("--fill-level", type=float, default=FILL_LEVEL)
    p.add_argument("--fill-max", type=int, default=FILL_MAX)
    p.add_argument("--refine-max", type=int, default=REFINE_MAX)
    p.add_argument("--ad-grid", type=int, default=AD_N_GRID)
    p.add_argument("--run-id", default=None)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rd = run_dir()
    specs = json.load(open(os.path.join(rd, "prep", "fit_specs.json")))["specs"]
    spec = specs[a.spec]
    if a.run_id is None and a.pilot:
        from lemcvt.results import new_run_id
        a.run_id = new_run_id(FIT_STEPS[a.stage] + "_pilot", f"{spec['tile']}_{spec['group']}")
    rec = run_part(spec, a) if a.stage == "part" else run_ad(spec, a)
    print(f"results: {rec.dir}", flush=True)


if __name__ == "__main__":
    main()
