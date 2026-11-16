"""
D-infinity landscape evolution models for steady-state topography.

Mirrors the three model factories in :mod:`models` but uses D-infinity
(Tarboton 1997) flow routing instead of D8, distributing flow to two
receivers proportionally based on the continuous flow angle.

Factories
---------
- :func:`advection_diffusion_model_dinf` — coupled AD PDE with D-inf stencil
- :func:`partitioned_threshold_model_dinf` — area-threshold + linear diffusion
- :func:`partitioned_threshold_model_nonlinear_dinf` — same with nonlinear diffusion
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from dinf import FACETS, get_receivers
from models import (
    D8_OFFSETS,
    _extract_grid_data,
    _make_boundary_mask,
    _find_boundary_value,
    _solve_hillslope_diffusion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Advection-diffusion model with D-inf stencil
# ---------------------------------------------------------------------------

def advection_diffusion_model_dinf(
    dem: Any,
    area_d8: Any,
    flow_direction: Any,
    m: float,
    angles: np.ndarray,
    r_facet: np.ndarray,
    facet_idx: np.ndarray,
    A_dinf: np.ndarray,
) -> Callable[[float, float, float], np.ndarray]:
    """Build a solver for the coupled AD PDE using D-infinity flow routing.

    The advection stencil distributes the incision term to two receivers
    proportionally, based on the D-inf flow angle. The diffusion stencil
    (5-point Laplacian) is identical to D8.

    Parameters
    ----------
    dem : Elevation-like object
    area_d8 : Area-like object
        D8 drainage area (used only for grid extraction).
    flow_direction : FlowDirectionD8-like object
        D8 flow direction (used only for grid extraction).
    m : float
        Area exponent in stream-power law.
    angles, r_facet, facet_idx : np.ndarray
        D-inf flow routing arrays from :func:`dinf.compute_dinf_flow_direction`.
    A_dinf : np.ndarray
        D-inf drainage area from :func:`dinf.compute_dinf_area`.

    Returns
    -------
    Callable[[float, float, float], np.ndarray]
        ``output_function(U, K, D)`` returning 2-D steady-state elevation.
    """
    gd = _extract_grid_data(dem, area_d8, flow_direction)
    Z0 = gd["Z0"]
    dx = gd["dx"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    valid = gd["valid"]
    n = gd["n"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    inv_dx2, inv_dy2 = gd["inv_dx2"], gd["inv_dy2"]

    # ── 1. Build diffusion and advection stencils ─────────────────────────
    M_diff = lil_matrix((n, n), dtype=float)
    M_adv = lil_matrix((n, n), dtype=float)

    b_U = np.full(n, -1.0, dtype=float)
    b_elev = np.zeros(n, dtype=float)

    laplacian_offsets = [
        (-1, 0, inv_dx2), (1, 0, inv_dx2),
        (0, -1, inv_dy2), (0, 1, inv_dy2),
    ]

    for k in range(n):
        i, j = rev_i[k], rev_j[k]

        # Diffusion stencil (identical to D8)
        for di, dj, inv in laplacian_offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                M_diff[k, idx_map[nj, ni]] += inv
                M_diff[k, k] -= inv
            else:
                M_diff[k, k] -= inv

        # D-inf advection stencil
        fac = facet_idx[j, i]
        if fac < 0:
            continue

        r = r_facet[j, i]
        j1, i1, j2, i2, p1, p2 = get_receivers(j, i, fac, r)

        # Check both receivers are in bounds and valid
        k1_valid = (0 <= i1 < Nx and 0 <= j1 < Ny and valid[j1, i1])
        k2_valid = (0 <= i2 < Nx and 0 <= j2 < Ny and valid[j2, i2])

        if not (k1_valid or k2_valid):
            continue

        A_here = A_dinf[j, i]
        coeff_Am = A_here ** m

        # D-inf stencil:
        #   M_adv[k, k]   += -(A^m) * cos(r) / d1
        #   M_adv[k, k1]  += (A^m) * (cos(r)/d1 - sin(r)/d2)
        #   M_adv[k, k2]  += (A^m) * sin(r) / d2
        # where d1 = dx (cardinal), d2 = dx (perpendicular within facet)
        cos_r = np.cos(r)
        sin_r = np.sin(r)

        if k1_valid and k2_valid:
            k1 = idx_map[j1, i1]
            k2 = idx_map[j2, i2]
            M_adv[k, k] += -coeff_Am * cos_r / dx
            M_adv[k, k1] += coeff_Am * (cos_r / dx - sin_r / dx)
            M_adv[k, k2] += coeff_Am * sin_r / dx
        elif k1_valid:
            # Only cardinal receiver valid — degrade to single-receiver
            k1 = idx_map[j1, i1]
            M_adv[k, k] += -coeff_Am * cos_r / dx
            M_adv[k, k1] += coeff_Am * cos_r / dx
        elif k2_valid:
            # Only diagonal receiver valid
            k2 = idx_map[j2, i2]
            M_adv[k, k] += -coeff_Am * sin_r / dx
            M_adv[k, k2] += coeff_Am * sin_r / dx

    # ── 2. Boundary conditions ────────────────────────────────────────────
    boundary_mask = _make_boundary_mask(valid)
    bc_inds: list[int] = []
    bc_vals = np.zeros(n, dtype=float)

    for j_b, i_b in zip(*np.where(boundary_mask)):
        k = idx_map[j_b, i_b]
        bc_inds.append(k)
        bc_vals[k] = Z0[j_b, i_b]
        b_U[k] = 0.0

    # Outlet handling (same as D8)
    A = gd["A"]
    j_out, i_out = np.unravel_index(np.nanargmax(A), A.shape)
    if not (
        (j_out == 0 or j_out == Ny - 1 or i_out == 0 or i_out == Nx - 1)
        and valid[j_out, i_out]
    ):
        found = False
        for ii, jj in [(i_out, 0), (i_out, Ny - 1), (0, j_out), (Nx - 1, j_out)]:
            if valid[jj, ii]:
                i_out, j_out = ii, jj
                found = True
                break
        if not found:
            raise RuntimeError("Cannot force the outlet onto a valid perimeter cell.")

    kout = idx_map[j_out, i_out]
    if kout not in bc_inds:
        bc_inds.append(kout)
        bc_vals[kout] = Z0[j_out, i_out]
        b_U[kout] = 0.0

    # Convert to efficient formats
    M_diff_csr = M_diff.tocsr()
    M_adv_csr = M_adv.tocsr()
    bc_inds_arr = np.array(bc_inds)

    # ── 3. Solver closure ─────────────────────────────────────────────────
    def output_function(U: float, K: float, D: float) -> np.ndarray:
        """Solve for steady-state elevation with D-inf advection.

        Parameters
        ----------
        U : float
            Uplift rate [m/yr].
        K : float
            Erodibility coefficient.
        D : float
            Hillslope diffusivity [m²/yr].

        Returns
        -------
        np.ndarray
            2-D array (Ny × Nx) of steady-state elevations.
        """
        M = D * M_diff_csr + K * M_adv_csr
        b = U * b_U + b_elev

        M = M.tolil()
        M[bc_inds_arr, :] = 0.0
        M[bc_inds_arr, bc_inds_arr] = 1.0
        b[bc_inds_arr] = bc_vals[bc_inds_arr]

        z_flat = spsolve(M.tocsr(), b)

        Z = np.full_like(Z0, np.nan)
        Z[rev_j, rev_i] = z_flat
        return Z

    return output_function


# ---------------------------------------------------------------------------
# Channel integration helpers for D-inf
# ---------------------------------------------------------------------------

def _precompute_channel_network_dinf(
    gd: dict,
    m: float,
    U: float,
    K: float,
    angles: np.ndarray,
    r_facet: np.ndarray,
    facet_idx: np.ndarray,
    A_dinf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Topological-sort channel elevation integration using D-inf routing.

    Integrates steady-state advective elevations upstream from boundary
    cells using S = (U/K) / A^m in the D-inf steepest direction.

    Processing order: ascending observed elevation (lowest first).

    Parameters
    ----------
    gd : dict
        Grid-data dictionary from :func:`models._extract_grid_data`.
    m, U, K : float
        Stream-power exponent, uplift rate, erodibility.
    angles, r_facet, facet_idx : np.ndarray
        D-inf flow routing arrays.
    A_dinf : np.ndarray
        D-inf drainage area.

    Returns
    -------
    Z_channel_all : np.ndarray
        1-D array (length *n*) of channel elevations.
    downstream_of : np.ndarray
        Not used in D-inf (kept for API compat); all -1.
    k_out : int
        Flat index of outlet cell.
    bc_boundary_inds : np.ndarray
        Flat indices of boundary cells.
    bc_boundary_vals : np.ndarray
        Boundary-cell elevations.
    """
    Z0 = gd["Z0"]
    dx = gd["dx"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    n = gd["n"]
    valid = gd["valid"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]

    # Build flat arrays of D-inf area
    A_dinf_flat = A_dinf[rev_j, rev_i]
    facet_idx_flat = facet_idx[rev_j, rev_i]
    r_facet_flat = r_facet[rev_j, rev_i]

    # Steady-state theoretical slopes
    S0_flat = (U / K) / (A_dinf_flat ** m)

    # Downstream topology: for each cell, find where it flows to
    # In D-inf, each cell has two downstream neighbors
    downstream_of = np.full(n, -1, dtype=int)  # primary downstream (not used directly)

    # Boundary mask and elevations
    boundary_mask = _make_boundary_mask(valid)
    Z_channel_all = np.full(n, np.nan)

    bc_boundary_inds_list: list[int] = []
    bc_boundary_vals = np.zeros(n, dtype=float)

    for j_b, i_b in zip(*np.where(boundary_mask)):
        k = idx_map[j_b, i_b]
        elev = Z0[j_b, i_b] if np.isfinite(Z0[j_b, i_b]) else _find_boundary_value(Z0, i_b, j_b, Nx, Ny)
        Z_channel_all[k] = elev
        bc_boundary_inds_list.append(k)
        bc_boundary_vals[k] = elev

    bc_boundary_inds = np.array(bc_boundary_inds_list)

    # Outlet
    j_out, i_out = np.unravel_index(np.nanargmax(gd["A"]), (Ny, Nx))
    k_out = idx_map[j_out, i_out]

    # Build upstream neighbor lists based on D-inf routing
    upstream_neighbors: list[list[int]] = [[] for _ in range(n)]
    for k in range(n):
        i_c, j_c = rev_i[k], rev_j[k]
        fac = facet_idx[j_c, i_c]
        if fac < 0:
            continue

        r = r_facet[j_c, i_c]
        j1, i1, j2, i2, p1, p2 = get_receivers(j_c, i_c, fac, r)

        # Cardinal receiver
        if 0 <= i1 < Nx and 0 <= j1 < Ny and valid[j1, i1]:
            k1 = idx_map[j1, i1]
            upstream_neighbors[k1].append(k)
            downstream_of[k] = k1  # primary downstream for compat

        # Diagonal receiver
        if 0 <= i2 < Nx and 0 <= j2 < Ny and valid[j2, i2]:
            k2 = idx_map[j2, i2]
            upstream_neighbors[k2].append(k)

    # BFS from boundary cells upstream
    # Process by ascending observed elevation (lowest first ensures both
    # downstream neighbors are processed before any upstream cell)
    queue = deque(k for k in range(n) if not np.isnan(Z_channel_all[k]))
    processed_count = 0

    while queue:
        k_current = queue.popleft()
        for k_upstream in upstream_neighbors[k_current]:
            if not np.isnan(Z_channel_all[k_upstream]):
                continue

            i_u, j_u = rev_i[k_upstream], rev_j[k_upstream]
            fac = facet_idx[j_u, i_u]
            if fac < 0:
                continue

            r = r_facet[j_u, i_u]
            j1, i1, j2, i2, p1, p2 = get_receivers(j_u, i_u, fac, r)

            cos_r = np.cos(r)
            sin_r = np.sin(r)

            # Check if both receivers have known elevations
            k1_valid = (0 <= i1 < Nx and 0 <= j1 < Ny and valid[j1, i1])
            k2_valid = (0 <= i2 < Nx and 0 <= j2 < Ny and valid[j2, i2])

            z1_known = False
            z2_known = False
            z1_val = np.nan
            z2_val = np.nan

            if k1_valid:
                k1 = idx_map[j1, i1]
                if not np.isnan(Z_channel_all[k1]):
                    z1_known = True
                    z1_val = Z_channel_all[k1]

            if k2_valid:
                k2 = idx_map[j2, i2]
                if not np.isnan(Z_channel_all[k2]):
                    z2_known = True
                    z2_val = Z_channel_all[k2]

            s_here = S0_flat[k_upstream]
            if not np.isfinite(s_here):
                continue

            if z1_known and z2_known and cos_r > 1e-12:
                # Full D-inf integration:
                # S = (z0 - z1)*cos(r)/dx + (z1 - z2)*sin(r)/dx
                # z0 = z1 + [S*dx - (z1 - z2)*sin(r)] / cos(r)
                z_here = z1_val + (s_here * dx - (z1_val - z2_val) * sin_r) / cos_r
            elif z1_known:
                # Single-receiver fallback (cardinal)
                dL = dx
                z_here = z1_val + s_here * dL
            elif z2_known:
                # Single-receiver fallback (diagonal)
                dL = dx * np.sqrt(2.0)
                z_here = z2_val + s_here * dL
            else:
                continue

            if not np.isfinite(z_here):
                continue

            Z_channel_all[k_upstream] = z_here
            queue.append(k_upstream)
            processed_count += 1

    logger.debug(
        "D-inf channel propagation: %d cells processed, %d NaNs remaining out of %d",
        processed_count, int(np.sum(np.isnan(Z_channel_all))), n,
    )

    return Z_channel_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals


# ---------------------------------------------------------------------------
# Partitioned threshold model with D-inf
# ---------------------------------------------------------------------------

def partitioned_threshold_model_dinf(
    dem: Any,
    area_d8: Any,
    flow_direction: Any,
    m: float,
    U: float,
    K: float,
    angles: np.ndarray,
    r_facet_arr: np.ndarray,
    facet_idx_arr: np.ndarray,
    A_dinf: np.ndarray,
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Build a partitioned threshold model with D-inf channel integration.

    Parameters
    ----------
    dem : Elevation-like object
    area_d8 : Area-like object
    flow_direction : FlowDirectionD8-like object
    m : float
        Area exponent.
    U, K : float
        Uplift rate and erodibility.
    angles, r_facet_arr, facet_idx_arr : np.ndarray
        D-inf routing arrays.
    A_dinf : np.ndarray
        D-inf drainage area.

    Returns
    -------
    Callable
        ``f(A_crit, D) → (Z_ss, channel_mask)``
    """
    gd = _extract_grid_data(dem, area_d8, flow_direction)
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]

    # Use D-inf area for channel thresholding
    A_dinf_flat = A_dinf[rev_j, rev_i]

    Z_channel_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals = (
        _precompute_channel_network_dinf(
            gd, m, U, K, angles, r_facet_arr, facet_idx_arr, A_dinf,
        )
    )

    def output_function(
        A_crit: float,
        D: float,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply area threshold and solve diffusion with D-inf channels.

        Parameters
        ----------
        A_crit : float
            Critical drainage area threshold [m²].
        D : float
            Hillslope diffusivity [m²/yr].
        debug : bool, optional
            If ``True``, emit diagnostic log messages.

        Returns
        -------
        Z_ss : np.ndarray
            2-D steady-state elevation (Ny × Nx).
        channel_mask : np.ndarray
            2-D boolean mask of channel cells (Ny × Nx).
        """
        if debug:
            logger.setLevel(logging.DEBUG)

        channel_mask_flat = A_dinf_flat >= A_crit
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

            z_flat = _solve_hillslope_diffusion(
                gd, hillslope_indices, hillslope_to_reduced, hillslope_mask,
                D, Z_channel_flat, channel_mask_flat,
                bc_boundary_inds, bc_boundary_vals, U,
            )

        Z_ss = np.full((Ny, Nx), np.nan)
        Z_ss[rev_j, rev_i] = z_flat

        channel_mask_2d = np.zeros((Ny, Nx), dtype=bool)
        channel_mask_2d[rev_j, rev_i] = channel_mask_flat

        return Z_ss, channel_mask_2d

    return output_function


# ---------------------------------------------------------------------------
# Nonlinear partitioned threshold model with D-inf
# ---------------------------------------------------------------------------

def partitioned_threshold_model_nonlinear_dinf(
    dem: Any,
    area_d8: Any,
    flow_direction: Any,
    m: float,
    U: float,
    K: float,
    angles: np.ndarray,
    r_facet_arr: np.ndarray,
    facet_idx_arr: np.ndarray,
    A_dinf: np.ndarray,
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Build a partitioned threshold model with nonlinear diffusion and D-inf.

    Parameters
    ----------
    dem : Elevation-like object
    area_d8 : Area-like object
    flow_direction : FlowDirectionD8-like object
    m : float
        Area exponent.
    U, K : float
        Uplift rate and erodibility.
    angles, r_facet_arr, facet_idx_arr : np.ndarray
        D-inf routing arrays.
    A_dinf : np.ndarray
        D-inf drainage area.

    Returns
    -------
    Callable
        ``f(A_crit, D_linear, S_c, max_iter=50, tol=1e-6) → (Z_ss, channel_mask)``
    """
    gd = _extract_grid_data(dem, area_d8, flow_direction)
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    valid = gd["valid"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    dx, dy = gd["dx"], gd["dy"]

    A_dinf_flat = A_dinf[rev_j, rev_i]

    Z_channel_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals = (
        _precompute_channel_network_dinf(
            gd, m, U, K, angles, r_facet_arr, facet_idx_arr, A_dinf,
        )
    )

    def _compute_slopes(
        z_flat: np.ndarray,
        hillslope_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute slope magnitude for each hillslope cell."""
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

            slopes[idx_hs] = np.sqrt(dzdi ** 2 + dzdj ** 2)
        return slopes

    def output_function(
        A_crit: float,
        D_linear: float,
        S_c: float,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply area threshold and solve nonlinear hillslope diffusion.

        Parameters
        ----------
        A_crit : float
            Critical drainage area threshold [m²].
        D_linear : float
            Linear diffusivity [m²/yr].
        S_c : float
            Critical slope for nonlinear diffusion.
        max_iter : int, optional
            Maximum Picard iterations (default 50).
        tol : float, optional
            Convergence tolerance (default 1e-6).

        Returns
        -------
        Z_ss : np.ndarray
            2-D steady-state elevation (Ny × Nx).
        channel_mask : np.ndarray
            2-D boolean mask of channel cells (Ny × Nx).
        """
        channel_mask_flat = A_dinf_flat >= A_crit
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

            # Initialize with linear diffusion
            z_flat_prev = _solve_hillslope_diffusion(
                gd, hillslope_indices, hillslope_to_reduced, hillslope_mask,
                D_linear, Z_channel_flat, channel_mask_flat,
                bc_boundary_inds, bc_boundary_vals, U,
            )

            # Picard iteration
            for iteration in range(max_iter):
                slopes = _compute_slopes(z_flat_prev, hillslope_indices)
                slope_ratio = np.clip(slopes / S_c, -0.99, 0.99)
                D_eff = D_linear / (1 - slope_ratio ** 2)

                z_flat_new = _solve_hillslope_diffusion(
                    gd, hillslope_indices, hillslope_to_reduced, hillslope_mask,
                    D_eff, Z_channel_flat, channel_mask_flat,
                    bc_boundary_inds, bc_boundary_vals, U,
                )

                hillslope_change = np.max(
                    np.abs(z_flat_new[hillslope_indices] - z_flat_prev[hillslope_indices])
                )
                if hillslope_change < tol:
                    break
                z_flat_prev = z_flat_new.copy()

            z_flat = z_flat_new

        Z_ss = np.full((Ny, Nx), np.nan)
        Z_ss[rev_j, rev_i] = z_flat

        channel_mask_2d = np.zeros((Ny, Nx), dtype=bool)
        channel_mask_2d[rev_j, rev_i] = channel_mask_flat

        return Z_ss, channel_mask_2d

    return output_function
