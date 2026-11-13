"""Newton solver for the steady hillslope equations of the nonlinear partitioned model (addendum 9, section 2).

Two discrete forms, selected with ``form``:

``centre`` (v1, the default of every existing command)
    ``D_eff,i * sum_n inv_n (z_n - z_i) + U = 0``: the v1 equation, D_eff evaluated at the cell centre and multiplying
    the 5-point Laplacian (``assemble_hillslope_system``). This is not the divergence form: in 1-D its steady slope is
    ``S_c tanh(U x / (D S_c))``, not the Roering et al. (2007) profile, and R* = (2/E*) ln cosh(E*/2).
``flux`` (addendum 9, added at Steve's decision of 2026-11-13)
    ``sum_n inv_n Dface_in (z_n - z_i) + U = 0`` with ``Dface = (D_eff,i + D_eff,n)/2`` between two hillslope cells and
    ``D_eff,i`` on a face to a fixed (channel or boundary) cell, i.e. div(D_eff grad z) = -U. Invalid neighbours carry no
    flux (the v1 ghost value 0 of the centre form is kept there; neither occurs on the ten tiles, every cell is valid).

Both use the same cell-centre slope stencil (``_compute_slopes`` of the model) and the same
``D_eff = D / (1 - clip(|grad z|/S_c, -0.99, 0.99)^2)``, so a Picard fixed point of a form is a root of its residual
``F(z) = M(D_eff(z)) z - b(D_eff(z))``.

Newton: analytic sparse Jacobian ``J = M(D_eff) + G diag(dD_eff/ds) dS/dz``, where ``G = dF/dD_eff`` (diagonal for the
centre form, the face stencil for the flux form) and ``dS/dz`` is the derivative of the slope stencil; the clipped cells
have dD_eff/ds = 0. The Newton step solves ``J delta = -F`` directly (SuperLU) by default. ``linear_solver="gmres"``
solves it by GMRES on the exact Jacobian preconditioned with the LU of its 5-point part, to the inexact-Newton tolerance
``min(1e-2, max(1e-12, ||F||/||b||))``; on a 300 x 300 AP1 crop at the fitted S_c it reached the 600-iteration limit
without meeting that tolerance at every Newton step, so it is not used for any reported number. Backtracking (Armijo) line search on ||F||_2. Converged when ||F||_2/||b||_2 <= ``rtol`` (the relative
residual of the Picard audit, ``_residual`` of the model) and the max update is <= ``utol`` metres.

Outcome of every evaluation (addendum 9, 2.1), checked in this order:
``no_steady_solution`` the clip is active (|grad z|/S_c > 0.99) on more than 1 % of hillslope cells at the last iterate,
                       whether or not the clipped system converged;
``converged``          the stopping test held;
``not_converged``      anything else (iteration cap, failed line search, singular or non-finite step).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, gmres, splu, spsolve

from lemcvt.models.common import LAPLACIAN_OFFSETS, assemble_hillslope_system

logger = logging.getLogger(__name__)

FORMS = ("centre", "flux")
SOLVERS = ("picard", "newton")
CLIP = 0.99
NO_STEADY_CLIP_FRAC = 0.01
NEWTON_MAX_ITER = 50
NEWTON_RTOL = 1e-10
NEWTON_UTOL = 1e-6
ARMIJO_C = 1e-4
LS_MIN_STEP = 2.0 ** -12
GMRES_RESTART = 60
GMRES_MAXITER = 10          # restarts
ETA_MAX = 1e-2              # inexact-Newton forcing term: inner relative tolerance min(ETA_MAX, max(ETA_MIN, ||F||/||b||))
ETA_MIN = 1e-12


def d_eff(slopes: np.ndarray, D: float, S_c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(D_eff, dD_eff/ds, clipped)``, D_eff with the Picard iteration's exact expressions."""
    raw = slopes / S_c
    ratio = np.clip(raw, -CLIP, CLIP)
    De = D / (1 - ratio**2)
    clipped = raw > CLIP
    dD = np.where(clipped, 0.0, D * 2.0 * ratio / (S_c * (1.0 - ratio**2) ** 2))
    return De, dD, clipped


def classify(converged: bool, frac_clipped: float) -> str:
    if frac_clipped > NO_STEADY_CLIP_FRAC:
        return "no_steady_solution"
    return "converged" if converged else "not_converged"


def _neighbours(gd: dict, hs: np.ndarray, di: int, dj: int) -> np.ndarray:
    """Flat valid index of the (di, dj) neighbour of each hillslope cell, -1 where outside the grid or invalid."""
    Ny, Nx, idx_map = gd["Ny"], gd["Nx"], gd["idx_map"]
    i = gd["rev_i"][hs] + di
    j = gd["rev_j"][hs] + dj
    inside = (i >= 0) & (i < Nx) & (j >= 0) & (j < Ny)
    nk = np.full(hs.size, -1, dtype=np.int64)
    nk[inside] = idx_map[j[inside], i[inside]]
    return nk


def assemble_flux_system(gd: dict, hillslope_indices: np.ndarray, hillslope_to_reduced: np.ndarray, hillslope_mask: np.ndarray,
                         D_eff, Z_channel_flat: np.ndarray, channel_mask_flat: np.ndarray, bc_boundary_vals: np.ndarray,
                         U: float) -> tuple[csr_matrix, np.ndarray]:
    """Reduced system ``M z = b`` of the flux form (module docstring). With a constant D it equals the centre form's
    matrix and right-hand side bit for bit wherever every neighbour is valid (``0.5 * (D + D) == D``)."""
    inv = {"x": gd["inv_dx2"], "y": gd["inv_dy2"]}
    hs = np.asarray(hillslope_indices)
    n_h = hs.size
    D = np.broadcast_to(np.asarray(D_eff, dtype=float), (n_h,)) if np.isscalar(D_eff) else np.asarray(D_eff, dtype=float)
    diag = np.zeros(n_h)
    b = np.full(n_h, -U, dtype=float)
    rows, cols, vals = [], [], []
    ar = np.arange(n_h)
    for di, dj, axis in LAPLACIAN_OFFSETS:
        nk = _neighbours(gd, hs, di, dj)
        valid_n = nk >= 0
        nk_safe = np.where(valid_n, nk, 0)
        is_hs = valid_n & hillslope_mask[nk_safe]
        fixed = valid_n & ~is_hs
        r_n = np.where(is_hs, hillslope_to_reduced[nk_safe], 0)
        Dn = np.where(is_hs, D[r_n], D)
        w = (0.5 * (D + Dn)) * inv[axis]
        w = np.where(valid_n, w, 0.0)
        diag -= w
        rows.append(ar[is_hs])
        cols.append(r_n[is_hs])
        vals.append(w[is_hs])
        if fixed.any():
            nkf = nk[fixed]
            val = np.where(channel_mask_flat[nkf], Z_channel_flat[nkf], bc_boundary_vals[nkf])
            b[fixed] -= w[fixed] * val
    rows.append(ar)
    cols.append(ar)
    vals.append(diag)
    M = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_h, n_h)).tocsr()
    return M, b


def assemble(form: str, *args) -> tuple[csr_matrix, np.ndarray]:
    if form == "centre":
        return assemble_hillslope_system(*args)
    if form == "flux":
        return assemble_flux_system(*args)
    raise ValueError(f"form must be one of {FORMS}, not {form!r}")


def slope_operators(gd: dict, hs: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
    """``(Gx, Gy)``, (n_h x n) matrices over every valid cell with ``gx = Gx @ z_flat`` and ``gy = Gy @ z_flat``: the model's
    slope stencil (one-sided where one neighbour is missing, centred where both exist, zero where neither)."""
    n_h = hs.size
    ar = np.arange(n_h)
    out = []
    for (dlo, dhi, h) in (((-1, 0), (1, 0), gd["dx"]), ((0, -1), (0, 1), gd["dy"])):
        lo = _neighbours(gd, hs, *dlo)
        hi = _neighbours(gd, hs, *dhi)
        has_lo, has_hi = lo >= 0, hi >= 0
        cnt = has_lo.astype(float) + has_hi.astype(float)
        c = np.where(cnt > 0, 1.0 / (np.where(cnt > 0, cnt, 1.0) * h), 0.0)
        rows, cols, vals = [], [], []
        # g = c * (z_hi - z_c) [if hi] - c * (z_lo - z_c) [if lo]; the z_c terms cancel when both exist
        for nk, has, sign in ((hi, has_hi, 1.0), (lo, has_lo, -1.0)):
            rows += [ar[has], ar[has]]
            cols += [nk[has], hs[has]]
            vals += [sign * c[has], -sign * c[has]]
        out.append(coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_h, gd["n"])).tocsr())
    return out[0], out[1]


def _G_matrix(form: str, gd: dict, hs: np.ndarray, hillslope_to_reduced: np.ndarray, hillslope_mask: np.ndarray,
              z_flat: np.ndarray) -> csr_matrix:
    """``dF/dD_eff`` at fixed z (n_h x n_h); ``z_flat`` holds the fixed (channel and boundary) values."""
    inv = {"x": gd["inv_dx2"], "y": gd["inv_dy2"]}
    n_h = hs.size
    ar = np.arange(n_h)
    zc = z_flat[hs]
    diag = np.zeros(n_h)
    rows, cols, vals = [], [], []
    for di, dj, axis in LAPLACIAN_OFFSETS:
        nk = _neighbours(gd, hs, di, dj)
        valid_n = nk >= 0
        nk_safe = np.where(valid_n, nk, 0)
        is_hs = valid_n & hillslope_mask[nk_safe]
        dz = np.where(valid_n, z_flat[nk_safe] - zc, 0.0)
        if form == "centre":
            diag += inv[axis] * np.where(valid_n, dz, -zc)      # invalid neighbour: ghost value 0 (v1)
        else:
            diag += inv[axis] * np.where(is_hs, 0.5, 1.0) * dz
            rows.append(ar[is_hs])
            cols.append(hillslope_to_reduced[nk_safe[is_hs]])
            vals.append(0.5 * inv[axis] * dz[is_hs])
    rows.append(ar)
    cols.append(ar)
    vals.append(diag)
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_h, n_h)).tocsr()


class HillslopeSystem:
    """The residual and Jacobian of one form on one hillslope set (fixed channel and boundary values)."""

    def __init__(self, form: str, gd: dict, hillslope_indices: np.ndarray, hillslope_to_reduced: np.ndarray, hillslope_mask: np.ndarray,
                 D: float, S_c: float, Z_channel_flat: np.ndarray, channel_mask_flat: np.ndarray, bc_boundary_vals: np.ndarray, U: float,
                 compute_slopes):
        if form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}, not {form!r}")
        self.form, self.gd, self.hs, self.h2r, self.hmask = form, gd, np.asarray(hillslope_indices), hillslope_to_reduced, hillslope_mask
        self.D, self.S_c, self.Zc, self.cmask, self.bc_vals, self.U = float(D), float(S_c), Z_channel_flat, channel_mask_flat, bc_boundary_vals, float(U)
        self.compute_slopes = compute_slopes
        self.Gx_full, self.Gy_full = slope_operators(gd, self.hs)
        self.Gx = self.Gx_full.tocsc()[:, self.hs].tocsr()          # derivative with respect to the hillslope unknowns
        self.Gy = self.Gy_full.tocsc()[:, self.hs].tocsr()

    def state(self, z_flat: np.ndarray) -> dict[str, Any]:
        s = self.compute_slopes(z_flat, self.hs)
        De, dD, clipped = d_eff(s, self.D, self.S_c)
        M, b = assemble(self.form, self.gd, self.hs, self.h2r, self.hmask, De, self.Zc, self.cmask, self.bc_vals, self.U)
        F = M @ z_flat[self.hs] - b
        nb = float(np.linalg.norm(b))
        nF = float(np.linalg.norm(F))
        return {"slopes": s, "D_eff": De, "dD": dD, "clipped": clipped, "M": M, "b": b, "F": F, "normF": nF,
                "rel": nF / nb if nb > 0 else float("nan"), "max_abs_over_U": float(np.max(np.abs(F)) / abs(self.U)) if self.U else float("nan"),
                "frac_clipped": float(np.mean(clipped)) if clipped.size else 0.0}

    def residual(self, z_h: np.ndarray, z_flat_template: np.ndarray) -> np.ndarray:
        z = z_flat_template.copy()
        z[self.hs] = z_h
        return self.state(z)["F"]

    def jacobian(self, z_flat: np.ndarray, st: dict[str, Any]) -> csr_matrix:
        gx = self.Gx_full @ z_flat
        gy = self.Gy_full @ z_flat
        s = np.sqrt(gx * gx + gy * gy)
        with np.errstate(invalid="ignore", divide="ignore"):
            ux = np.where(s > 0, gx / s, 0.0)
            uy = np.where(s > 0, gy / s, 0.0)
        dS = diags(ux) @ self.Gx + diags(uy) @ self.Gy
        G = _G_matrix(self.form, self.gd, self.hs, self.h2r, self.hmask, z_flat)
        return (st["M"] + G @ (diags(st["dD"]) @ dS)).tocsr()


def _step(J: csr_matrix, M: csr_matrix, F: np.ndarray, rel: float, method: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve ``J delta = -F``; ``method`` "direct" (SuperLU) or "gmres" (preconditioned with the LU of J on M's pattern)."""
    if method == "direct":
        return spsolve(J.tocsc(), -F), {"linear": "direct"}
    pattern = M.copy()
    pattern.data[:] = 1.0
    J5 = J.multiply(pattern).tocsc()
    lu = splu(J5)
    P = LinearOperator(J.shape, matvec=lu.solve, dtype=float)
    eta = min(ETA_MAX, max(ETA_MIN, rel))
    count = [0]

    def _cb(_x):
        count[0] += 1

    delta, info = gmres(J, -F, rtol=eta, atol=0.0, restart=GMRES_RESTART, maxiter=GMRES_MAXITER, M=P, callback=_cb, callback_type="pr_norm")
    res = float(np.linalg.norm(J @ delta + F) / max(np.linalg.norm(F), 1e-300))
    return delta, {"linear": "gmres", "eta": eta, "gmres_info": int(info), "gmres_iterations": count[0], "linear_rel_residual": res}


def newton_solve(system: HillslopeSystem, z0_flat: np.ndarray, max_iter: int = NEWTON_MAX_ITER, rtol: float = NEWTON_RTOL,
                 utol: float = NEWTON_UTOL, linear_solver: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Newton with Armijo backtracking from ``z0_flat`` (fixed entries must hold their values). Returns ``(z_flat, info)``."""
    method = linear_solver or "direct"
    t0 = time.perf_counter()
    hs = system.hs
    z = z0_flat.copy()
    st = system.state(z)
    hist: list[dict[str, Any]] = [{"iter": 0, "rel_residual": st["rel"], "max_abs_over_U": st["max_abs_over_U"], "frac_clipped": st["frac_clipped"]}]
    converged = False
    stop = "max_iter"
    n_iter = 0
    for it in range(1, max_iter + 1):
        ta = time.perf_counter()
        J = system.jacobian(z, st)
        tb = time.perf_counter()
        try:
            delta, lin = _step(J, st["M"], st["F"], st["rel"], method)
        except Exception:  # noqa: BLE001
            stop = "singular_jacobian"
            break
        if not np.all(np.isfinite(delta)):
            stop = "nonfinite_step"
            break
        tc = time.perf_counter()
        t = 1.0
        accepted = False
        while t >= LS_MIN_STEP:
            z_try = z.copy()
            z_try[hs] += t * delta
            st_try = system.state(z_try)
            if np.isfinite(st_try["normF"]) and st_try["normF"] <= (1.0 - ARMIJO_C * t) * st["normF"]:
                accepted = True
                break
            t *= 0.5
        n_iter = it
        if not accepted:
            hist.append({"iter": it, "step": t, "line_search": "failed", "rel_residual": st["rel"]})
            stop = "line_search_failed"
            break
        upd = float(t * np.max(np.abs(delta)))
        z, st = z_try, st_try
        hist.append({"iter": it, "step": t, "update_max_abs": upd, "rel_residual": st["rel"], "max_abs_over_U": st["max_abs_over_U"],
                     "frac_clipped": st["frac_clipped"], "t_jacobian": tb - ta, "t_solve": tc - tb, "t_line_search": time.perf_counter() - tc} | lin)
        if st["rel"] <= rtol and upd <= utol:
            converged = True
            stop = "tolerance"
            break
    info = {"solver": "newton", "form": system.form, "linear_solver": method, "n_iter": n_iter, "max_iter": max_iter, "rtol": rtol, "utol": utol, "converged": converged,
            "stop_reason": stop, "rel_residual": st["rel"], "max_abs_over_U": st["max_abs_over_U"], "frac_clipped": st["frac_clipped"],
            "outcome": classify(converged, st["frac_clipped"]), "n_hillslope": int(hs.size), "iterations": hist,
            "wall_s": time.perf_counter() - t0}
    return z, info
