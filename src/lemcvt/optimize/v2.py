"""Optimizer v2 (directive Phase 4, addendum 3 section 5): exact reductions and profiles.

Linear partitioned model, fixed A_crit (and fixed m)
    Channel cells carry ``z_anchor + (U/K) chi`` (Phase 3 step 4). Hillslope cells solve the
    reduced Laplacian ``L z_h = -(U/D) 1 - B z_fixed`` (``L`` is the hillslope Laplacian with
    D = 1, ``B`` the coupling to the fixed channel and boundary cells), so with ``x = U/K`` and
    ``y = U/D``

        Z(K, D) = a + x b + y c,

    where on the hillslope cells ``a = -L^-1 B z_fixed0`` (anchor and boundary values),
    ``b = -L^-1 B chi`` and ``c = -L^-1 1``, and on the fixed cells ``a = z_fixed0``,
    ``b = chi`` (0 on the boundary) and ``c = 0``. One sparse LU of ``L`` and three solves give the
    whole (K, D) plane at that A_crit; ``L``, ``B``, ``a`` and ``c`` do not depend on m, so the
    exponent profile of addendum 3 section 5.2 costs one extra solve per m. The MSE is a convex
    quadratic in (x, y); its minimum under the box ``K, D in bounds`` is found exactly
    (:func:`box_quadratic_min`). The joint problem is a one-dimensional profile over A_crit,
    sampled at the unique drainage areas where the staircase changes.

AD model, fixed r = K/D
    ``(M_diff0 + r M_adv0 + I_bc) z = y b_U + bc`` with ``y = U/D``, so ``Z = h(r) + y g(r)`` from
    one LU and two solves; the optimal ``y`` is closed-form (clipped to the bounds). The AD
    optimisation is a profile over log10 r, refined with bounded Brent.

Nonlinear model
    No reduction: the same A_crit grid evaluated with the Picard solver at D0
    (:mod:`lemcvt.analyses.optimizer_v2`).

Every evaluation is recorded in a :class:`lemcvt.optimize.v1.Trajectory`.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from lemcvt.models.common import LAPLACIAN_OFFSETS, assemble_hillslope_system, extract_grid_data, precompute_chi_network
from lemcvt.optimize.v1 import Trajectory

logger = logging.getLogger(__name__)

LN10 = math.log(10.0)


# ----------------------------------------------------------------------------- A_crit grids

def unique_areas(A: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Every unique drainage-area value in ``[lo, hi]``: the staircase changes exactly there."""
    a = A[np.isfinite(A)]
    a = a[(a >= lo) & (a <= hi)]
    return np.unique(a)


def thin_log_uniform(values: np.ndarray, n: int) -> np.ndarray:
    """``n`` unique values nearest a uniform log grid between min and max of ``values``.

    Each grid point takes its nearest unique value; where several grid points snap to the same
    value (the dense low end of the staircase) the count is topped up by inserting, repeatedly,
    the unselected value nearest the midpoint of the largest remaining log gap, so that exactly
    ``min(n, len(values))`` values are returned.
    """
    values = np.unique(np.asarray(values, dtype=float))
    if values.size <= n:
        return values
    lv = np.log10(values)
    targets = np.linspace(lv[0], lv[-1], n)
    pos = np.searchsorted(lv, targets).clip(1, lv.size - 1)
    nearer = np.where(np.abs(lv[pos] - targets) <= np.abs(lv[pos - 1] - targets), pos, pos - 1)
    chosen = np.zeros(values.size, dtype=bool)
    chosen[np.unique(nearer)] = True
    while chosen.sum() < n:
        idx = np.where(chosen)[0]
        gaps = lv[idx[1:]] - lv[idx[:-1]]
        order = np.argsort(-gaps)
        inserted = False
        for k in order:
            lo, hi = idx[k], idx[k + 1]
            if hi - lo > 1:
                mid = 0.5 * (lv[lo] + lv[hi])
                cand = lo + 1 + int(np.argmin(np.abs(lv[lo + 1:hi] - mid)))
                chosen[cand] = True
                inserted = True
                break
        if not inserted:
            break
    return values[chosen]


# ----------------------------------------------------------------------------- box quadratic

def box_quadratic_min(G: np.ndarray, g: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, str]:
    """Minimise ``v^T G v - 2 g^T v`` (G symmetric positive semi-definite, 2 x 2) over ``lo <= v <= hi``.

    Exact for a convex quadratic: the unconstrained minimiser if it is feasible, otherwise the
    best of the four edges, each a one-dimensional convex problem solved by clipping. Returns
    ``(v, where)`` with ``where`` in ``interior``, ``x_lo``, ``x_hi``, ``y_lo``, ``y_hi``, ``corner``.
    """
    G = np.asarray(G, float)
    g = np.asarray(g, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)

    def f(v):
        return float(v @ G @ v - 2.0 * g @ v)

    try:
        v0 = np.linalg.solve(G, g)
        if np.all(v0 >= lo) and np.all(v0 <= hi):
            return v0, "interior"
    except np.linalg.LinAlgError:
        pass
    cands: list[tuple[float, np.ndarray, str]] = []
    for i in range(2):
        j = 1 - i
        for bound, name in ((lo[i], "lo"), (hi[i], "hi")):
            vj = (g[j] - G[i, j] * bound) / G[j, j] if G[j, j] > 0 else lo[j]
            vj = float(np.clip(vj, lo[j], hi[j]))
            v = np.empty(2)
            v[i], v[j] = bound, vj
            cands.append((f(v), v, f"{'xy'[i]}_{name}"))
    for vx in (lo[0], hi[0]):
        for vy in (lo[1], hi[1]):
            v = np.array([vx, vy])
            cands.append((f(v), v, "corner"))
    best = min(cands, key=lambda t: t[0])
    return best[1], best[2]


# ----------------------------------------------------------------------------- partitioned reduction

@dataclass
class PartitionedReduction:
    """Z(K, D) = a + (U/K) b + (U/D) c on the valid cells, at one A_crit and one m."""

    A_crit: float
    m: float
    U: float
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    channel_mask: np.ndarray
    n_hillslope: int
    wall_s: float
    terms: dict[str, Any] = field(default_factory=dict, repr=False)

    def Z_flat(self, K: float, D: float) -> np.ndarray:
        return self.a + (self.U / K) * self.b + (self.U / D) * self.c

    def mse_terms(self, Z_obs_flat: np.ndarray) -> dict[str, Any]:
        """Gram matrix and vector of the quadratic MSE(x, y) over the valid cells (cached)."""
        if self.terms:
            return self.terms
        r = Z_obs_flat - self.a
        ok = np.isfinite(r) & np.isfinite(self.b) & np.isfinite(self.c)
        b, c, r = self.b[ok], self.c[ok], r[ok]
        n = int(ok.sum())
        self.terms = {"G": np.array([[b @ b, b @ c], [b @ c, c @ c]]) / n, "g": np.array([b @ r, c @ r]) / n,
                      "rr": float(r @ r) / n, "n": n}
        return self.terms

    @staticmethod
    def mse_from_terms(t: dict[str, Any], x: float, y: float) -> float:
        v = np.array([x, y])
        return float(t["rr"] - 2.0 * t["g"] @ v + v @ t["G"] @ v)

    @staticmethod
    def gradient_log10(t: dict[str, Any], x: float, y: float) -> dict[str, float]:
        """dMSE/dlog10 K and dMSE/dlog10 D at ``x = U/K``, ``y = U/D`` (channel mask fixed)."""
        v = np.array([x, y])
        dv = 2.0 * (t["G"] @ v - t["g"])
        return {"K": float(-LN10 * x * dv[0]), "D": float(-LN10 * y * dv[1])}


@dataclass
class Factored:
    A_crit: float
    channel: np.ndarray
    hs: np.ndarray
    lu: Any
    B: Any                   # (n_hs, n) coupling to fixed cells: sum of D-free weights
    a: np.ndarray            # m-independent parts
    c: np.ndarray
    wall_s: float


class PartitionedProfiler:
    """Reductions of the linear partitioned model for one tile at any A_crit and m."""

    def __init__(self, Z0: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, U: float, m_values=()):
        self.gd = extract_grid_data(Z0, A, FD, dx)
        self.U = U
        self.n = self.gd["n"]
        self.A_flat = self.gd["A_flat"]
        self.Z_obs_flat = Z0[self.gd["rev_j"], self.gd["rev_i"]].astype(float)
        self.chi: dict[float, np.ndarray] = {}
        self.z_anchor = self.k_out = self.bc_inds = self.bc_vals = None
        for m in list(m_values):
            self.add_m(m)

    def add_m(self, m: float) -> None:
        m = float(m)
        if m in self.chi:
            return
        t0 = time.perf_counter()
        chi, z_anchor, _ds, k_out, bc_inds, bc_vals = precompute_chi_network(self.gd, m)
        self.chi[m] = chi
        if self.z_anchor is None:
            self.z_anchor, self.k_out, self.bc_inds, self.bc_vals = z_anchor, k_out, bc_inds, bc_vals
        logger.debug("chi(m=%.3f) in %.1f s", m, time.perf_counter() - t0)

    def masks(self, A_crit: float):
        channel = self.A_flat >= A_crit
        if not channel[self.k_out]:
            channel[self.k_out] = True
        fixed = channel.copy()
        fixed[self.bc_inds] = True
        hs_mask = ~fixed
        hs = np.where(hs_mask)[0]
        to_red = np.full(self.n, -1, dtype=int)
        to_red[hs] = np.arange(hs.size)
        return channel, hs_mask, hs, to_red

    def _coupling(self, hs: np.ndarray, hs_mask: np.ndarray):
        """B: (n_hs, n) with B[i, k] = weight of fixed neighbour k of hillslope cell i (inv_dx2 or inv_dy2)."""
        gd = self.gd
        Ny, Nx, idx_map = gd["Ny"], gd["Nx"], gd["idx_map"]
        i, j = gd["rev_i"][hs], gd["rev_j"][hs]
        inv = {"x": gd["inv_dx2"], "y": gd["inv_dy2"]}
        rows, cols, vals = [], [], []
        ar = np.arange(hs.size)
        for di, dj, axis in LAPLACIAN_OFFSETS:
            ni, nj = i + di, j + dj
            inside = (ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny)
            nk = np.full(hs.size, -1, dtype=np.int64)
            nk[inside] = idx_map[nj[inside], ni[inside]]
            fixed = (nk >= 0) & ~hs_mask[np.where(nk >= 0, nk, 0)]
            rows.append(ar[fixed])
            cols.append(nk[fixed])
            vals.append(np.full(int(fixed.sum()), inv[axis]))
        return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(hs.size, self.n)).tocsr()

    def factor(self, A_crit: float) -> Factored:
        """LU of the hillslope Laplacian (D = 1), the coupling matrix and the m-independent a, c."""
        t0 = time.perf_counter()
        channel, hs_mask, hs, to_red = self.masks(A_crit)
        n = self.n
        zeros = np.zeros(n)
        a = np.full(n, np.nan)
        c = np.full(n, np.nan)
        a[self.bc_inds] = self.bc_vals[self.bc_inds]
        a[channel] = self.z_anchor[channel]
        c[self.bc_inds] = 0.0
        c[channel] = 0.0
        lu = B = None
        if hs.size:
            L, rhs_c = assemble_hillslope_system(self.gd, hs, to_red, hs_mask, 1.0, zeros, channel, zeros, 1.0)  # rhs_c = -1
            lu = splu(L.tocsc())
            B = self._coupling(hs, hs_mask)
            zfix0 = np.where(channel, self.z_anchor, self.bc_vals)
            a[hs] = lu.solve(-(B @ zfix0))
            c[hs] = lu.solve(rhs_c)
        return Factored(float(A_crit), channel, hs, lu, B, a, c, time.perf_counter() - t0)

    def reduce(self, A_crit: float, m: float, factored: Factored | None = None) -> PartitionedReduction:
        t0 = time.perf_counter()
        m = float(m)
        self.add_m(m)
        chi = self.chi[m]
        F = factored if factored is not None and factored.A_crit == float(A_crit) else self.factor(A_crit)
        b = np.full(self.n, np.nan)
        b[self.bc_inds] = 0.0
        b[F.channel] = chi[F.channel]
        if F.hs.size:
            b[F.hs] = F.lu.solve(-(F.B @ np.where(F.channel, chi, 0.0)))
        wall = (F.wall_s if factored is None else 0.0) + (time.perf_counter() - t0)
        return PartitionedReduction(float(A_crit), m, self.U, F.a, b, F.c, F.channel, int(F.hs.size), wall)

    def to_grid(self, flat: np.ndarray) -> np.ndarray:
        Z = np.full((self.gd["Ny"], self.gd["Nx"]), np.nan)
        Z[self.gd["rev_j"], self.gd["rev_i"]] = flat
        return Z


def _row(red: PartitionedReduction, t: dict[str, Any], U, K0, D0, lo, hi, D0_B: float | None = None) -> dict[str, Any]:
    mse_stage2 = red.mse_from_terms(t, U / K0, U / D0)
    v, where = box_quadratic_min(t["G"], t["g"], lo, hi)
    out = {"A_crit": red.A_crit, "m": red.m, "n_channel": int(red.channel_mask.sum()), "n_hillslope": red.n_hillslope,
           "mse_stage2": mse_stage2, "mse_joint": red.mse_from_terms(t, v[0], v[1]), "K_opt": float(U / v[0]),
           "D_opt": float(U / v[1]), "where": where, "wall_s": red.wall_s,
           "G00": float(t["G"][0, 0]), "G01": float(t["G"][0, 1]), "G11": float(t["G"][1, 1]),
           "g0": float(t["g"][0]), "g1": float(t["g"][1]), "rr": float(t["rr"])}
    if D0_B is not None:
        out["mse_stage2_B"] = red.mse_from_terms(t, U / K0, U / D0_B)     # addendum 6, 5.2: stage 2 at the other end of the D0 bracket
    return out


def partitioned_profile(prof: PartitionedProfiler, A_crits: np.ndarray, m: float, K_bounds, D_bounds, K0: float, D0: float,
                        traj: Trajectory | None = None, extra_points: dict[str, tuple[float, float, float]] | None = None,
                        log_every: int = 100, D0_B: float | None = None) -> dict[str, Any]:
    """MSE(A_crit) profile of the linear partitioned model at fixed m.

    At every A_crit: the MSE at (K0, D0) (the v1 stage-2 objective) and the exact box-constrained
    minimum over (U/K, U/D) (the joint objective) with its K, D and position in the box.
    ``extra_points`` = {name: (A_crit, K, D)} are evaluated with the same code (the optimality
    certificate), with the MSE gradient in log10 K and log10 D there and the best (K, D) for that
    channel mask.
    """
    U = prof.U
    lo = np.array([U / K_bounds[1], U / D_bounds[1]])
    hi = np.array([U / K_bounds[0], U / D_bounds[0]])
    rows = []
    t_start = time.perf_counter()
    A_crits = np.asarray(A_crits, float)
    for i, a_c in enumerate(A_crits):
        red = prof.reduce(a_c, m)
        t = red.mse_terms(prof.Z_obs_flat)
        row = _row(red, t, U, K0, D0, lo, hi, D0_B=D0_B)
        rows.append(row)
        if traj is not None:
            traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(D0), "log10_K": math.log10(K0), "m": m,
                         "kind": "stage2"}, row["mse_stage2"], cached=False)
            if D0_B is not None:
                traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(D0_B), "log10_K": math.log10(K0), "m": m,
                             "kind": "stage2_B"}, row["mse_stage2_B"], cached=True)
            traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(row["D_opt"]), "log10_K": math.log10(row["K_opt"]),
                         "m": m, "kind": "joint"}, row["mse_joint"], cached=True)
        if log_every and (i % log_every == 0 or i == len(A_crits) - 1):
            logger.info("A_crit profile m=%.3f: %d/%d A_crit=%.2f stage2 %.4f joint %.4f (K %.3e D %.3e %s) [%.0f s]", m, i + 1,
                        len(A_crits), a_c, row["mse_stage2"], row["mse_joint"], row["K_opt"], row["D_opt"], row["where"],
                        time.perf_counter() - t_start)
    out: dict[str, Any] = {"m": m, "rows": rows, "n_evaluations": len(rows), "wall_s": time.perf_counter() - t_start}
    if rows:
        j2 = int(np.argmin([r["mse_stage2"] for r in rows]))
        jj = int(np.argmin([r["mse_joint"] for r in rows]))
        out["stage2_optimum"] = {"A_crit": rows[j2]["A_crit"], "mse": rows[j2]["mse_stage2"], "K": K0, "D": D0, "index": j2,
                                 "n_channel": rows[j2]["n_channel"]}
        out["joint_optimum"] = {"A_crit": rows[jj]["A_crit"], "mse": rows[jj]["mse_joint"], "K": rows[jj]["K_opt"],
                                "D": rows[jj]["D_opt"], "where": rows[jj]["where"], "index": jj, "n_channel": rows[jj]["n_channel"]}
        out["stage2_identifiability"] = identifiability(rows, "mse_stage2")
        out["joint_identifiability"] = identifiability(rows, "mse_joint")
        if D0_B is not None:
            jb = int(np.argmin([r["mse_stage2_B"] for r in rows]))
            out["stage2_B_optimum"] = {"A_crit": rows[jb]["A_crit"], "mse": rows[jb]["mse_stage2_B"], "K": K0, "D": D0_B, "index": jb,
                                       "n_channel": rows[jb]["n_channel"]}
            out["stage2_B_identifiability"] = identifiability(rows, "mse_stage2_B")
    if extra_points:
        cert = {}
        for name, (a_c, K, D) in extra_points.items():
            red = prof.reduce(a_c, m)
            t = red.mse_terms(prof.Z_obs_flat)
            x, y = U / K, U / D
            v, where = box_quadratic_min(t["G"], t["g"], lo, hi)
            cert[name] = {"A_crit": float(a_c), "K": float(K), "D": float(D), "mse": red.mse_from_terms(t, x, y),
                          "gradient_log10": red.gradient_log10(t, x, y), "n_channel": int(red.channel_mask.sum()),
                          "best_at_this_A_crit": {"K": float(U / v[0]), "D": float(U / v[1]),
                                                  "mse": red.mse_from_terms(t, v[0], v[1]), "where": where}}
            if traj is not None:
                traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(D), "log10_K": math.log10(K), "m": m,
                             "kind": f"certificate_{name}"}, cert[name]["mse"], cached=False)
        out["certificate"] = cert
    return out


def exponent_profile(prof: PartitionedProfiler, A_crits: np.ndarray, m_values: np.ndarray, K_bounds, D_bounds, K0: float, D0: float,
                     traj: Trajectory | None = None, log_every: int = 20) -> dict[str, Any]:
    """MSE(A_crit, m) with the exact (U/K, U/D) minimum at every point (addendum 3, section 5.2).

    One LU per A_crit; one solve per m. Returns all rows, the joint optimum (A_crit, m, K, D), the
    profile MSE(m) minimised over A_crit (and K, D), and the 1 % and 5 % contours as point sets.
    """
    U = prof.U
    lo = np.array([U / K_bounds[1], U / D_bounds[1]])
    hi = np.array([U / K_bounds[0], U / D_bounds[0]])
    for m in m_values:
        prof.add_m(float(m))
    rows = []
    t_start = time.perf_counter()
    A_crits = np.asarray(A_crits, float)
    for i, a_c in enumerate(A_crits):
        F = prof.factor(a_c)
        for m in m_values:
            red = prof.reduce(a_c, float(m), factored=F)
            t = red.mse_terms(prof.Z_obs_flat)
            row = _row(red, t, U, K0, D0, lo, hi)
            rows.append(row)
            if traj is not None:
                traj.record({"log10_A_crit": math.log10(a_c), "log10_D": math.log10(row["D_opt"]), "log10_K": math.log10(row["K_opt"]),
                             "m": float(m), "kind": "m_profile"}, row["mse_joint"], cached=False)
        if log_every and (i % log_every == 0 or i == len(A_crits) - 1):
            best = min((r for r in rows if r["A_crit"] == a_c), key=lambda r: r["mse_joint"])
            logger.info("m profile: %d/%d A_crit=%.2f best m %.2f mse %.4f [%.0f s]", i + 1, len(A_crits), a_c, best["m"],
                        best["mse_joint"], time.perf_counter() - t_start)
    mse = np.array([r["mse_joint"] for r in rows])
    j = int(np.argmin(mse))
    out: dict[str, Any] = {"rows": rows, "n_evaluations": len(rows), "wall_s": time.perf_counter() - t_start,
                           "optimum": {k: rows[j][k] for k in ("A_crit", "m", "mse_joint", "K_opt", "D_opt", "where", "n_channel")}}
    prof_m = []
    for m in m_values:
        sel = [r for r in rows if r["m"] == float(m)]
        b = min(sel, key=lambda r: r["mse_joint"])
        prof_m.append({"m": float(m), "mse": b["mse_joint"], "A_crit": b["A_crit"], "K": b["K_opt"], "D": b["D_opt"]})
    out["profile_m"] = prof_m
    fmin = float(mse[j])
    for lev in (0.01, 0.05):
        pts = [r for r in rows if r["mse_joint"] <= fmin * (1 + lev)]
        out[f"contour_{lev:g}"] = {"n_points": len(pts), "A_crit_min": min(r["A_crit"] for r in pts), "A_crit_max": max(r["A_crit"] for r in pts),
                                   "m_min": min(r["m"] for r in pts), "m_max": max(r["m"] for r in pts),
                                   "K_min": min(r["K_opt"] for r in pts), "K_max": max(r["K_opt"] for r in pts),
                                   "D_min": min(r["D_opt"] for r in pts), "D_max": max(r["D_opt"] for r in pts),
                                   "points": [[r["A_crit"], r["m"]] for r in pts]}
    return out


def identifiability(rows: list[dict[str, Any]], col: str, levels=(0.01, 0.05), key: str = "A_crit") -> dict[str, Any]:
    """Interval(s) of ``key`` within which the profile MSE is no more than ``level`` above its minimum, and the K, D range there."""
    a = np.array([r[key] for r in rows])
    f = np.array([r[col] for r in rows])
    order = np.argsort(a)
    a, f = a[order], f[order]
    rows_s = [rows[i] for i in order]
    fmin = float(np.nanmin(f))
    out: dict[str, Any] = {"mse_min": fmin}
    for lev in levels:
        idx = np.where(f <= fmin * (1 + lev))[0]
        runs = []
        if idx.size:
            start = prev = idx[0]
            for i in idx[1:]:
                if i != prev + 1:
                    runs.append((start, prev))
                    start = i
                prev = i
            runs.append((start, prev))
        rec: dict[str, Any] = {"n_grid_points": int(idx.size), f"{key}_min": float(a[idx].min()) if idx.size else None,
                               f"{key}_max": float(a[idx].max()) if idx.size else None,
                               "intervals": [[float(a[s]), float(a[e])] for s, e in runs], "n_intervals": len(runs)}
        if idx.size and "K_opt" in rows_s[0]:
            Ks = np.array([rows_s[i]["K_opt"] for i in idx])
            Ds = np.array([rows_s[i]["D_opt"] for i in idx])
            rec.update({"K_min": float(Ks.min()), "K_max": float(Ks.max()), "D_min": float(Ds.min()), "D_max": float(Ds.max()),
                        "log10_K_range": float(np.log10(Ks.max() / Ks.min())), "log10_D_range": float(np.log10(Ds.max() / Ds.min()))})
        out[f"level_{lev:g}"] = rec
    return out


# ----------------------------------------------------------------------------- AD reduction

class ADProfiler:
    """Z = h(r) + (U/D) g(r) for the advection-diffusion model at fixed r = K/D."""

    def __init__(self, ad_model, Z_obs_flat: np.ndarray, U: float):
        self.M_diff0, self.M_adv0, self.I_bc = ad_model.matrices
        self.bc_inds, self.bc_vals, self.b_U = ad_model.bc
        self.U = U
        self.obs = np.asarray(Z_obs_flat, float)

    def reduce(self, r: float) -> dict[str, Any]:
        t0 = time.perf_counter()
        lu = splu((self.M_diff0 + r * self.M_adv0 + self.I_bc).tocsc())
        rhs_h = np.zeros(self.b_U.size)
        rhs_h[self.bc_inds] = self.bc_vals[self.bc_inds]
        return {"r": float(r), "h": lu.solve(rhs_h), "g": lu.solve(self.b_U.copy()), "wall_s": time.perf_counter() - t0}

    def best_y(self, red: dict[str, Any], y_lo: float, y_hi: float) -> tuple[float, float, str]:
        r = self.obs - red["h"]
        g = red["g"]
        ok = np.isfinite(r) & np.isfinite(g)
        gg = float(g[ok] @ g[ok])
        y = float(g[ok] @ r[ok]) / gg if gg > 0 else y_lo
        where = "interior"
        if y < y_lo:
            y, where = y_lo, "y_lo"
        elif y > y_hi:
            y, where = y_hi, "y_hi"
        res = r[ok] - y * g[ok]
        return float(y), float(res @ res / ok.sum()), where

    def mse_at(self, red: dict[str, Any], y: float) -> float:
        r = self.obs - red["h"] - y * red["g"]
        ok = np.isfinite(r)
        return float(r[ok] @ r[ok] / ok.sum())


def ad_profile(prof: ADProfiler, K_bounds, D_bounds, n_grid: int = 200, traj: Trajectory | None = None,
               refine: bool = True, log_every: int = 25) -> dict[str, Any]:
    """Profile MSE*(r) over log10 r on ``n_grid`` points spanning the bounds' ratio range, then bounded Brent."""
    U = prof.U
    r_lo, r_hi = K_bounds[0] / D_bounds[1], K_bounds[1] / D_bounds[0]
    grid = np.logspace(math.log10(r_lo), math.log10(r_hi), n_grid)
    rows: list[dict[str, Any]] = []
    cache: dict[float, dict[str, Any]] = {}

    def evaluate(r: float) -> dict[str, Any]:
        r = float(r)
        if r in cache:
            return cache[r]
        red = prof.reduce(r)
        D_lo = max(D_bounds[0], K_bounds[0] / r)
        D_hi = min(D_bounds[1], K_bounds[1] / r)
        if D_lo > D_hi:
            rec = {"r": r, "feasible": False, "mse": float("inf"), "wall_s": red["wall_s"]}
        else:
            y, m, where = prof.best_y(red, U / D_hi, U / D_lo)
            D = U / y
            rec = {"r": r, "feasible": True, "D": float(D), "K": float(r * D), "mse": m, "where": where, "wall_s": red["wall_s"]}
            if traj is not None:
                traj.record({"log10_K": math.log10(rec["K"]), "log10_D": math.log10(rec["D"]), "log10_r": math.log10(r), "kind": "profile"},
                            m, cached=False)
        cache[r] = rec
        rows.append(rec)
        return rec

    t_start = time.perf_counter()
    for i, r in enumerate(grid):
        rec = evaluate(r)
        if log_every and (i % log_every == 0 or i == n_grid - 1):
            logger.info("AD profile %d/%d r=%.3e mse=%s [%.0f s]", i + 1, n_grid, r,
                        f"{rec['mse']:.4f}" if rec["feasible"] else "infeasible", time.perf_counter() - t_start)
    feas = sorted((r for r in rows if r["feasible"]), key=lambda q: q["r"])
    if not feas:
        raise ValueError("no feasible r: the K and D bounds do not intersect")
    jj = int(np.argmin([r["mse"] for r in feas]))
    best = feas[jj]
    out: dict[str, Any] = {"grid_rows": [dict(r) for r in rows], "grid_optimum": dict(best), "n_grid": n_grid, "r_range": [r_lo, r_hi]}
    if refine:
        from scipy.optimize import minimize_scalar

        lg = np.log10([r["r"] for r in feas])
        a, b = lg[max(jj - 1, 0)], lg[min(jj + 1, len(lg) - 1)]
        if a < b:
            res = minimize_scalar(lambda lr: evaluate(10 ** lr)["mse"], bounds=(a, b), method="bounded", options={"xatol": 1e-6})
            fin = evaluate(10 ** res.x)
            out["brent"] = {"log10_r": float(res.x), "nfev": int(res.nfev), "success": bool(res.success), "bracket_log10": [float(a), float(b)]}
            if fin["mse"] <= best["mse"]:
                best = fin
    out["optimum"] = dict(best)
    out["n_evaluations"] = len(rows)
    out["wall_s"] = time.perf_counter() - t_start
    grid_rows = [{"r": q["r"], "mse": q["mse"], "K_opt": q["K"], "D_opt": q["D"]} for q in feas]
    out["identifiability"] = identifiability(grid_rows, "mse", key="r")
    return out
