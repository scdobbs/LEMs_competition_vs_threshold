"""Threshold-partitioned model with linear hillslope diffusion. Exact port of v1."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from lemcvt.models.common import extract_grid_data, precompute_channel_network, solve_hillslope_diffusion

logger = logging.getLogger(__name__)


def partitioned_threshold_model(
    Z0: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, m: float, U: float, K: float,
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Build ``f(A_crit, D) -> (Z, channel_mask)``.

    Channels (``A >= A_crit``, plus the outlet) carry the pre-integrated
    stream-power profile for this ``K``; hillslopes solve ``0 = U + D lap(z)``
    with Dirichlet values from channels and the tile boundary.
    """
    gd = extract_grid_data(Z0, A, FD, dx)
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    A_flat = gd["A_flat"]

    Z_channel_all, _downstream_of, k_out, bc_boundary_inds, bc_boundary_vals = (
        precompute_channel_network(gd, m, U, K)
    )

    def output_function(A_crit: float, D: float, debug: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Grid %d x %d, valid=%d, A_crit=%g, D=%g", Ny, Nx, n, A_crit, D)

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
        else:
            hillslope_to_reduced = np.full(n, -1, dtype=int)
            hillslope_to_reduced[hillslope_indices] = np.arange(n_hillslope)
            z_flat = solve_hillslope_diffusion(
                gd, hillslope_indices, hillslope_to_reduced, hillslope_mask,
                D, Z_channel_flat, channel_mask_flat, bc_boundary_inds, bc_boundary_vals, U,
            )

        Z_ss = np.full((Ny, Nx), np.nan)
        Z_ss[rev_j, rev_i] = z_flat

        channel_mask_2d = np.zeros((Ny, Nx), dtype=bool)
        channel_mask_2d[rev_j, rev_i] = channel_mask_flat
        return Z_ss, channel_mask_2d

    output_function.Z_channel_all = Z_channel_all  # type: ignore[attr-defined]
    output_function.grid_data = gd  # type: ignore[attr-defined]
    return output_function
