"""Threshold-partitioned model with nonlinear (Roering) hillslope diffusion. Exact port of v1.

D_eff = D / (1 - (|grad z|/S_c)^2), solved by Picard iteration (max 50, tol 1e-6 on
the maximum hillslope elevation change), with slopes from one-sided/centred
differences over valid neighbours exactly as in v1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lemcvt.models.common import extract_grid_data, precompute_channel_network, solve_hillslope_diffusion


def partitioned_threshold_model_nonlinear(
    Z0: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, m: float, U: float, K: float,
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Build ``f(A_crit, D_linear, S_c, max_iter=50, tol=1e-6) -> (Z, channel_mask)``."""
    gd = extract_grid_data(Z0, A, FD, dx)
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    valid = gd["valid"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    dx, dy = gd["dx"], gd["dy"]
    A_flat = gd["A_flat"]

    Z_channel_all, _downstream_of, k_out, bc_boundary_inds, bc_boundary_vals = (
        precompute_channel_network(gd, m, U, K)
    )

    def _compute_slopes(z_flat: np.ndarray, hillslope_indices: np.ndarray) -> np.ndarray:
        """Vectorised slope magnitude with v1's operation order (Phase 3, step 2).

        v1: ``dzdi = 0; dzdi -= (z_left - z_c)/dx; dzdi += (z_right - z_c)/dx; dzdi /= count`` with
        each term present only when that neighbour exists and is valid; same for j. The order
        ``(0 - left) + right`` then ``/ count`` is kept so the result is bit-identical.
        """
        hs = np.asarray(hillslope_indices)
        i = rev_i[hs]
        j = rev_j[hs]
        zc = z_flat[hs]

        def _term(ii, jj, ok, d):
            out = np.zeros(hs.size)
            kk = np.full(hs.size, -1, dtype=np.int64)
            kk[ok] = idx_map[jj[ok], ii[ok]]
            use = kk >= 0
            out[use] = (z_flat[kk[use]] - zc[use]) / d
            return out, use

        left, ul = _term(i - 1, j, i > 0, dx)
        right, ur = _term(i + 1, j, i < Nx - 1, dx)
        cnt = ul.astype(float) + ur.astype(float)
        dzdi = (0.0 - left) + right
        dzdi = np.where(cnt > 0, dzdi / np.where(cnt > 0, cnt, 1.0), 0.0)
        down, ud = _term(i, j - 1, j > 0, dy)
        up, uu = _term(i, j + 1, j < Ny - 1, dy)
        cnt = ud.astype(float) + uu.astype(float)
        dzdj = (0.0 - down) + up
        dzdj = np.where(cnt > 0, dzdj / np.where(cnt > 0, cnt, 1.0), 0.0)
        return np.sqrt(dzdi**2 + dzdj**2)

    def _compute_slopes_v1(z_flat: np.ndarray, hillslope_indices: np.ndarray) -> np.ndarray:
        """v1 per-cell loop, kept as the reference for the vectorised version."""
        slopes = np.zeros(len(hillslope_indices))
        for idx_hs, k in enumerate(hillslope_indices):
            i, j = rev_i[k], rev_j[k]
            z_center = z_flat[k]

            dzdi = 0.0
            count = 0
            if i > 0 and valid[j, i - 1]:
                dzdi -= (z_flat[idx_map[j, i - 1]] - z_center) / dx
                count += 1
            if i < Nx - 1 and valid[j, i + 1]:
                dzdi += (z_flat[idx_map[j, i + 1]] - z_center) / dx
                count += 1
            if count > 0:
                dzdi /= count

            dzdj = 0.0
            count = 0
            if j > 0 and valid[j - 1, i]:
                dzdj -= (z_flat[idx_map[j - 1, i]] - z_center) / dy
                count += 1
            if j < Ny - 1 and valid[j + 1, i]:
                dzdj += (z_flat[idx_map[j + 1, i]] - z_center) / dy
                count += 1
            if count > 0:
                dzdj /= count

            slopes[idx_hs] = np.sqrt(dzdi**2 + dzdj**2)
        return slopes

    def output_function(
        A_crit: float, D_linear: float, S_c: float, max_iter: int = 50, tol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        channel_mask_flat = A_flat >= A_crit
        if not channel_mask_flat[k_out]:
            channel_mask_flat[k_out] = True

        Z_channel_flat = Z_channel_all.copy()

        is_channel_or_boundary = channel_mask_flat.copy()
        is_channel_or_boundary[bc_boundary_inds] = True
        hillslope_mask = ~is_channel_or_boundary

        hillslope_indices = np.where(hillslope_mask)[0]
        n_hillslope = len(hillslope_indices)

        if n_hillslope == 0:
            z_flat = np.full(n, np.nan)
            z_flat[bc_boundary_inds] = bc_boundary_vals[bc_boundary_inds]
            z_flat[channel_mask_flat] = Z_channel_flat[channel_mask_flat]
            n_iter = 0
        else:
            hillslope_to_reduced = np.full(n, -1, dtype=int)
            hillslope_to_reduced[hillslope_indices] = np.arange(n_hillslope)

            z_flat_prev = solve_hillslope_diffusion(
                gd, hillslope_indices, hillslope_to_reduced, hillslope_mask,
                D_linear, Z_channel_flat, channel_mask_flat, bc_boundary_inds, bc_boundary_vals, U,
            )

            z_flat_new = z_flat_prev
            n_iter = 0
            for _iteration in range(max_iter):
                n_iter += 1
                slopes = _compute_slopes(z_flat_prev, hillslope_indices)
                slope_ratio = np.clip(slopes / S_c, -0.99, 0.99)
                D_eff = D_linear / (1 - slope_ratio**2)

                z_flat_new = solve_hillslope_diffusion(
                    gd, hillslope_indices, hillslope_to_reduced, hillslope_mask,
                    D_eff, Z_channel_flat, channel_mask_flat, bc_boundary_inds, bc_boundary_vals, U,
                )

                hillslope_change = np.max(np.abs(z_flat_new[hillslope_indices] - z_flat_prev[hillslope_indices]))
                if hillslope_change < tol:
                    break
                z_flat_prev = z_flat_new.copy()

            z_flat = z_flat_new

        output_function.last_n_iter = n_iter  # type: ignore[attr-defined]

        Z_ss = np.full((Ny, Nx), np.nan)
        Z_ss[rev_j, rev_i] = z_flat

        channel_mask_2d = np.zeros((Ny, Nx), dtype=bool)
        channel_mask_2d[rev_j, rev_i] = channel_mask_flat
        return Z_ss, channel_mask_2d

    output_function.last_n_iter = 0  # type: ignore[attr-defined]
    output_function.compute_slopes = _compute_slopes  # type: ignore[attr-defined]
    output_function.compute_slopes_v1 = _compute_slopes_v1  # type: ignore[attr-defined]
    return output_function
