"""The rebuilt D-infinity partitioned model (Phase 4b, addendum 3 section 6 / addendum 4 section 4).

The v1 model (``lemcvt.models.dinf_models.partitioned_threshold_model_dinf_v1``) integrated channel
elevations upstream from the boundary with an explicit two-receiver update, which builds channel
elevations through hillslope cells and mixes different flow paths (``02d`` 2.6). This module replaces
that integration by **one sparse linear system over all valid cells**, as proposed in ``02d`` 2.6.6:

* boundary cells (valid edge cells)  ``z_k = z_obs,k``;
* channel cells (``A_dinf >= A_crit``, plus the outlet)  the D-infinity upwind advection row of the
  AD D-infinity model with D = 0, divided by ``K A_dinf^m``::

      [cos r (z_k - z_1) + sin r (z_1 - z_2)] / dx = (U/K) A_dinf^-m

  (``z_1`` the cardinal receiver, ``z_2`` the diagonal one, ``r`` the facet angle in ``[0, pi/4]``;
  with only one valid receiver the AD D-infinity model's single-receiver stencils are used);
* hillslope cells (everything else)  the 5-point Laplacian row of the partitioned model with K = 0::

      lap z = -(U/D)

The matrix does not depend on K or D, so ``Z(K, D) = a + (U/K) b + (U/D) c`` with ``a = M^-1 rhs_bc``,
``b = M^-1 rhs_ch`` (``A_dinf^-m`` on channel rows) and ``c = M^-1 rhs_hs`` (``-1`` on hillslope rows):
the Phase 4 reduction holds exactly and the v2 optimizer applies unchanged. When every facet angle is
cardinal or diagonal the channel row is the D8 integration step (``z_k = z_down + (U/K) A^-m dL``,
``dL = dx`` or ``dx / cos(pi/4) = dx sqrt 2``) and the model reduces to the D8 partitioned model
(``tests/test_dinf_rebuild.py``). A channel cell with no usable receiver (no facet, or a zero-weight
single receiver) gets a hillslope row and is counted in ``info``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from lemcvt.hydro.dinf import FACETS
from lemcvt.models.common import LAPLACIAN_OFFSETS, _boundary_anchor, extract_grid_data

logger = logging.getLogger(__name__)

_E1_DI = np.array([f[0] for f in FACETS], dtype=np.int64)
_E1_DJ = np.array([f[1] for f in FACETS], dtype=np.int64)
_E2_DI = np.array([f[2] for f in FACETS], dtype=np.int64)
_E2_DJ = np.array([f[3] for f in FACETS], dtype=np.int64)


@dataclass
class DinfSystem:
    """The coupled system at one A_crit: ``M z = rhs_bc + (U/K) rhs_ch + (U/D) rhs_hs``."""

    A_crit: float
    M: Any
    rhs_bc: np.ndarray
    rhs_ch: np.ndarray
    rhs_hs: np.ndarray
    channel: np.ndarray          # bool (n): channel rows (A_dinf >= A_crit or outlet, not boundary, with a receiver)
    boundary: np.ndarray         # bool (n)
    hillslope: np.ndarray        # bool (n)
    k1: np.ndarray               # cardinal receiver per cell (-1 none), weight w1 on z_1 in the channel row
    k2: np.ndarray               # diagonal receiver per cell (-1 none), weight w2 on z_2
    w1: np.ndarray
    w2: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    lu: Any = None
    wall_s: float = 0.0

    def factor(self):
        if self.lu is None:
            t0 = time.perf_counter()
            self.lu = splu(self.M.tocsc())
            self.wall_s += time.perf_counter() - t0
        return self.lu

    def solve(self, U: float, K: float, D: float) -> np.ndarray:
        return self.factor().solve(self.rhs_bc + (U / K) * self.rhs_ch + (U / D) * self.rhs_hs)

    def reduce(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(a, b, c)`` with ``Z = a + (U/K) b + (U/D) c``."""
        lu = self.factor()
        return lu.solve(self.rhs_bc), lu.solve(self.rhs_ch), lu.solve(self.rhs_hs)


def dinf_receivers(gd: dict, r_facet: np.ndarray, facet_idx: np.ndarray):
    """Flat receiver indices (cardinal ``k1``, diagonal ``k2``; -1 where outside or invalid) and cos r, sin r per cell."""
    Ny, Nx, n = gd["Ny"], gd["Nx"], gd["n"]
    idx_map, rev_i, rev_j = gd["idx_map"], gd["rev_i"], gd["rev_j"]
    fac = facet_idx[rev_j, rev_i].astype(np.int64)
    r = r_facet[rev_j, rev_i].astype(float)
    has = fac >= 0
    fac_s = np.where(has, fac, 0)
    k1 = np.full(n, -1, dtype=np.int64)
    k2 = np.full(n, -1, dtype=np.int64)
    for k_arr, di_t, dj_t in ((k1, _E1_DI, _E1_DJ), (k2, _E2_DI, _E2_DJ)):
        ni = rev_i + di_t[fac_s]
        nj = rev_j + dj_t[fac_s]
        inside = has & (ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny)
        k_arr[inside] = idx_map[nj[inside], ni[inside]]
    r = np.where(has, r, 0.0)
    return k1, k2, np.cos(r), np.sin(r), has


def assemble_dinf_partitioned(gd: dict, m: float, r_facet: np.ndarray, facet_idx: np.ndarray, A_dinf: np.ndarray,
                              A_crit: float, bc=None, receivers=None) -> DinfSystem:
    """Assemble the coupled system at ``A_crit`` (see the module docstring)."""
    t0 = time.perf_counter()
    n, dx = gd["n"], gd["dx"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    idx_map, rev_i, rev_j = gd["idx_map"], gd["rev_i"], gd["rev_j"]
    inv = {"x": gd["inv_dx2"], "y": gd["inv_dy2"]}
    bc_inds, bc_vals, k_out = bc if bc is not None else _boundary_anchor(gd)
    k1, k2, cos_r, sin_r, has = receivers if receivers is not None else dinf_receivers(gd, r_facet, facet_idx)
    A_flat = A_dinf[rev_j, rev_i].astype(float)

    boundary = np.zeros(n, dtype=bool)
    boundary[bc_inds] = True
    want_channel = (A_flat >= A_crit)
    want_channel[k_out] = True
    want_channel &= ~boundary
    ok1 = k1 >= 0
    ok2 = k2 >= 0
    both = want_channel & ok1 & ok2
    only1 = want_channel & ok1 & ~ok2 & (cos_r > 0)
    only2 = want_channel & ~ok1 & ok2 & (sin_r > 0)
    channel = both | only1 | only2
    no_receiver = want_channel & ~channel
    hillslope = ~boundary & ~channel
    ar = np.arange(n)
    rows, cols, vals = [], [], []
    w1 = np.zeros(n)
    w2 = np.zeros(n)
    # channel rows: [cos r (z_k - z1) + sin r (z1 - z2)] / dx = (U/K) A^-m
    kb = ar[both]
    rows += [kb, kb, kb]
    cols += [kb, k1[kb], k2[kb]]
    vals += [cos_r[kb] / dx, (sin_r[kb] - cos_r[kb]) / dx, -sin_r[kb] / dx]
    w1[kb] = cos_r[kb] - sin_r[kb]
    w2[kb] = sin_r[kb]
    k_1 = ar[only1]
    rows += [k_1, k_1]
    cols += [k_1, k1[k_1]]
    vals += [cos_r[k_1] / dx, -cos_r[k_1] / dx]
    w1[k_1] = cos_r[k_1]
    k_2 = ar[only2]
    rows += [k_2, k_2]
    cols += [k_2, k2[k_2]]
    vals += [sin_r[k_2] / dx, -sin_r[k_2] / dx]
    w2[k_2] = sin_r[k_2]
    rhs_ch = np.zeros(n)
    rhs_ch[channel] = A_flat[channel] ** (-m)
    # hillslope rows: v1's Laplacian convention (an outside/invalid neighbour still subtracts inv from the diagonal)
    hs = ar[hillslope]
    i, j = rev_i[hs], rev_j[hs]
    diag = np.zeros(hs.size)
    for di, dj, axis in LAPLACIAN_OFFSETS:
        ni, nj = i + di, j + dj
        inside = (ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny)
        nk = np.full(hs.size, -1, dtype=np.int64)
        nk[inside] = idx_map[nj[inside], ni[inside]]
        okn = nk >= 0
        rows.append(hs[okn])
        cols.append(nk[okn])
        vals.append(np.full(int(okn.sum()), inv[axis]))
        diag -= inv[axis]
    rows.append(hs)
    cols.append(hs)
    vals.append(diag)
    rhs_hs = np.zeros(n)
    rhs_hs[hillslope] = -1.0
    # boundary rows
    rows.append(ar[boundary])
    cols.append(ar[boundary])
    vals.append(np.ones(int(boundary.sum())))
    rhs_bc = np.zeros(n)
    rhs_bc[boundary] = bc_vals[boundary]
    M = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    M.sum_duplicates()
    M.eliminate_zeros()
    info = {"n": int(n), "n_boundary": int(boundary.sum()), "n_channel": int(channel.sum()), "n_hillslope": int(hillslope.sum()),
            "n_channel_both_receivers": int(both.sum()), "n_channel_cardinal_only": int(only1.sum()),
            "n_channel_diagonal_only": int(only2.sum()), "n_channel_without_receiver_as_hillslope": int(no_receiver.sum()),
            "n_channel_wanted": int(want_channel.sum()), "k_out": int(k_out), "outlet_is_boundary": bool(boundary[k_out])}
    return DinfSystem(float(A_crit), M, rhs_bc, rhs_ch, rhs_hs, channel, boundary, hillslope, k1, k2, w1, w2, info,
                      wall_s=time.perf_counter() - t0)


def channel_monotonicity(z_flat: np.ndarray, sysm: DinfSystem, tol: float = 0.0) -> dict[str, Any]:
    """Channel cells lower than their receivers of nonzero weight (addendum 3, section 6, test 4).

    ``n_lower_than_every_receiver``: channel cells below every receiver that carries a nonzero weight
    (the test's quantity, expected 0); ``n_lower_than_any_receiver``: below at least one such receiver.
    """
    ch = np.nonzero(sysm.channel)[0]
    below_all = np.ones(ch.size, dtype=bool)        # lower than each receiver of nonzero weight
    below_any = np.zeros(ch.size, dtype=bool)       # lower than at least one
    any_used = np.zeros(ch.size, dtype=bool)
    drop = np.full(ch.size, np.inf)                 # smallest drop to a used receiver
    for k_arr, w in ((sysm.k1, sysm.w1), (sysm.k2, sysm.w2)):
        use = (k_arr[ch] >= 0) & (np.abs(w[ch]) > 1e-15)
        zr = z_flat[np.where(use, k_arr[ch], 0)]
        lower = use & (z_flat[ch] < zr - tol)
        below_any |= lower
        below_all &= ~use | lower
        any_used |= use
        drop = np.where(use, np.minimum(drop, z_flat[ch] - zr), drop)
    below_all &= any_used
    drop = np.where(any_used, drop, np.nan)
    return {"n_channel": int(ch.size), "n_channel_with_receiver": int(any_used.sum()),
            "n_lower_than_every_receiver": int(below_all.sum()), "n_lower_than_any_receiver": int(below_any.sum()),
            "min_drop_to_receiver": float(np.nanmin(drop)) if np.isfinite(drop).any() else None,
            "median_drop_to_receiver": float(np.nanmedian(drop)) if np.isfinite(drop).any() else None}


def partitioned_threshold_model_dinf(
    Z0: np.ndarray, A_d8: np.ndarray, FD: np.ndarray, dx: float, m: float, U: float, K: float,
    angles: np.ndarray, r_facet_arr: np.ndarray, facet_idx_arr: np.ndarray, A_dinf: np.ndarray,
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Build ``f(A_crit, D) -> (Z, channel_mask)`` for the rebuilt D-infinity partitioned model.

    Same signature as the v1 factory. ``f.system(A_crit)`` returns the :class:`DinfSystem`,
    ``f.solve(A_crit, D, K)`` solves for another K, ``f.reduce(A_crit)`` returns ``(a, b, c)`` on the
    valid cells (flat), ``f.monotonicity(A_crit, D)`` the channel monotonicity record, and
    ``f.to_grid(flat)`` maps a flat vector back to the grid. ``angles`` is accepted for API
    compatibility and not used (``r_facet`` and ``facet_idx`` define the receivers).
    """
    gd = extract_grid_data(Z0, A_d8, FD, dx)
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    bc = _boundary_anchor(gd)
    receivers = dinf_receivers(gd, r_facet_arr, facet_idx_arr)
    cache: dict[float, DinfSystem] = {}

    def system(A_crit: float) -> DinfSystem:
        a = float(A_crit)
        if a not in cache:
            cache.clear()
            cache[a] = assemble_dinf_partitioned(gd, m, r_facet_arr, facet_idx_arr, A_dinf, a, bc=bc, receivers=receivers)
        return cache[a]

    def to_grid(flat: np.ndarray) -> np.ndarray:
        Z = np.full((Ny, Nx), np.nan)
        Z[rev_j, rev_i] = flat
        return Z

    def output_function(A_crit: float, D: float, K_: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        s = system(A_crit)
        z = s.solve(U, K if K_ is None else K_, D)
        output_function.last_system = s  # type: ignore[attr-defined]
        output_function.last_z_flat = z  # type: ignore[attr-defined]
        mask = np.zeros((Ny, Nx), dtype=bool)
        ch = s.channel | (s.boundary & ((A_dinf[rev_j, rev_i] >= A_crit) | (np.arange(n) == s.info["k_out"])))
        mask[rev_j, rev_i] = ch
        return to_grid(z), mask

    def solve(A_crit: float, D: float, K_: float) -> tuple[np.ndarray, np.ndarray]:
        return output_function(A_crit, D, K_)

    def reduce(A_crit: float):
        return system(A_crit).reduce()

    def monotonicity(A_crit: float, D: float, K_: float | None = None) -> dict[str, Any]:
        s = system(A_crit)
        z = s.solve(U, K if K_ is None else K_, D)
        return channel_monotonicity(z, s)

    output_function.system = system  # type: ignore[attr-defined]
    output_function.solve = solve  # type: ignore[attr-defined]
    output_function.reduce = reduce  # type: ignore[attr-defined]
    output_function.monotonicity = monotonicity  # type: ignore[attr-defined]
    output_function.to_grid = to_grid  # type: ignore[attr-defined]
    output_function.grid_data = gd  # type: ignore[attr-defined]
    output_function.n_valid = n  # type: ignore[attr-defined]
    return output_function
