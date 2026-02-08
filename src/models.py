"""
Landscape evolution models for steady-state topography.

Provides three model factories that build and return solver closures:

- :func:`advection_diffusion_model` — coupled advection-diffusion PDE
- :func:`partitioned_threshold_model` — area-threshold partitioning with
  linear hillslope diffusion
- :func:`partitioned_threshold_model_nonlinear` — same partitioning with
  slope-dependent (nonlinear) hillslope diffusion

Each factory pre-computes sparse matrices and topological data, then returns
a lightweight function that solves for steady-state elevation given physical
parameters (U, K, D, etc.).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

D8_OFFSETS: dict[int, tuple[int, int]] = {
    1: (1, 0),
    2: (1, 1),
    4: (0, 1),
    8: (-1, 1),
    16: (-1, 0),
    32: (-1, -1),
    64: (0, -1),
    128: (1, -1),
}
"""D8 flow-direction code → (di, dj) offset lookup."""


# ---------------------------------------------------------------------------
# Shared private helpers
# ---------------------------------------------------------------------------

def _make_boundary_mask(valid: np.ndarray) -> np.ndarray:
    """Return a boolean mask of edge cells that are also valid.

    Parameters
    ----------
    valid : np.ndarray
        2-D boolean array where ``True`` marks cells with finite data.

    Returns
    -------
    np.ndarray
        Boolean mask (same shape as *valid*) that is ``True`` for valid
        cells on the first/last row or first/last column.
    """
    edges = np.zeros_like(valid, dtype=bool)
    edges[0, :] = edges[-1, :] = edges[:, 0] = edges[:, -1] = True
    return edges & valid


def _extract_grid_data(
    dem: Any,
    area: Any,
    flow_direction: Any,
) -> dict:
    """Extract arrays and build flat indexing from raster objects.

    Parameters
    ----------
    dem : Elevation-like object
        Must expose ``._griddata`` (2-D elevation array) and
        ``._georef_info.dx`` (cell size).
    area : Area-like object
        Must expose ``._griddata`` (2-D drainage-area array).
    flow_direction : FlowDirectionD8-like object
        Must expose ``._griddata`` (2-D D8 code array).

    Returns
    -------
    dict
        Keys: ``Z0``, ``A``, ``FD``, ``dx``, ``dy``, ``Ny``, ``Nx``,
        ``valid``, ``n``, ``idx_map``, ``rev_i``, ``rev_j``,
        ``A_flat``, ``FD_flat``, ``inv_dx2``, ``inv_dy2``.
    """
    Z0 = dem._griddata
    A = area._griddata
    FD = flow_direction._griddata

    dx = dem._georef_info.dx
    dy = dx
    Ny, Nx = A.shape

    valid = np.isfinite(A)
    valid_indices = np.where(valid)
    n = len(valid_indices[0])

    idx_map = np.full((Ny, Nx), -1, dtype=int)
    idx_map[valid_indices] = np.arange(n)
    rev_i = valid_indices[1]
    rev_j = valid_indices[0]

    A_flat = A[rev_j, rev_i]
    FD_flat = FD[rev_j, rev_i].astype(int)

    inv_dx2 = 1.0 / dx ** 2
    inv_dy2 = 1.0 / dy ** 2

    return dict(
        Z0=Z0, A=A, FD=FD, dx=dx, dy=dy, Ny=Ny, Nx=Nx,
        valid=valid, n=n, idx_map=idx_map, rev_i=rev_i, rev_j=rev_j,
        A_flat=A_flat, FD_flat=FD_flat, inv_dx2=inv_dx2, inv_dy2=inv_dy2,
    )


def _precompute_channel_network(
    gd: dict,
    m: float,
    U: float,
    K: float,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Topological-sort channel elevation integration.

    Integrates steady-state advective elevations upstream from boundary
    cells using the relation S₀ = (U / K) / A^m.

    Parameters
    ----------
    gd : dict
        Grid-data dictionary returned by :func:`_extract_grid_data`.
    m, U, K : float
        Stream-power exponent, uplift rate, erodibility.

    Returns
    -------
    Z_channel_all : np.ndarray
        1-D array (length *n*) of pre-computed channel elevations.
    downstream_of : np.ndarray
        1-D array (length *n*) giving the flat index of each cell's
        downstream neighbour (−1 if none).
    k_out : int
        Flat index of the outlet cell (largest drainage area).
    bc_boundary_inds : np.ndarray
        1-D integer array of flat indices for boundary cells.
    bc_boundary_vals : np.ndarray
        1-D float array (length *n*) of boundary-cell elevations.
    """
    Z0 = gd["Z0"]
    dx = gd["dx"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    n = gd["n"]
    valid = gd["valid"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    A_flat, FD_flat = gd["A_flat"], gd["FD_flat"]

    # Steady-state theoretical slopes
    S0_flat = (U / K) / (A_flat ** m)

    # Build downstream topology
    downstream_of = np.full(n, -1, dtype=int)
    for k in range(n):
        i, j = rev_i[k], rev_j[k]
        code = int(FD_flat[k])
        if code in D8_OFFSETS:
            di, dj = D8_OFFSETS[code]
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                downstream_of[k] = idx_map[nj, ni]

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

    # Build upstream neighbour lists
    upstream_neighbors: list[list[int]] = [[] for _ in range(n)]
    for k in range(n):
        k_down = downstream_of[k]
        if k_down >= 0:
            upstream_neighbors[k_down].append(k)

    # BFS from boundary cells upstream
    queue = deque(k for k in range(n) if not np.isnan(Z_channel_all[k]))
    processed_count = 0

    while queue:
        k_current = queue.popleft()
        for k_upstream in upstream_neighbors[k_current]:
            if not np.isnan(Z_channel_all[k_upstream]):
                continue

            z_down = Z_channel_all[k_current]
            if np.isnan(z_down):
                logger.debug(
                    "NaN downstream at k=%d while propagating to k=%d",
                    k_current, k_upstream,
                )
                continue

            code = int(FD_flat[k_upstream])
            if code not in D8_OFFSETS:
                logger.debug(
                    "Invalid FD code %d at k=%d", code, k_upstream,
                )
                continue

            di, dj = D8_OFFSETS[code]
            dL = dx * (np.sqrt(2.0) if (di != 0 and dj != 0) else 1.0)
            s_here = S0_flat[k_upstream]

            if not np.isfinite(s_here):
                logger.debug(
                    "Non-finite slope at k=%d (A=%.1f)", k_upstream, A_flat[k_upstream],
                )
                continue

            z_here = z_down + s_here * dL
            if not np.isfinite(z_here):
                logger.debug(
                    "Non-finite elevation at k=%d: z_down=%.2f, S=%.4e, dL=%.2f",
                    k_upstream, z_down, s_here, dL,
                )
                continue

            Z_channel_all[k_upstream] = z_here
            queue.append(k_upstream)
            processed_count += 1

    logger.debug(
        "Channel propagation: %d cells processed, %d NaNs remaining out of %d",
        processed_count, int(np.sum(np.isnan(Z_channel_all))), n,
    )

    return Z_channel_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals


def _build_diffusion_stencil(gd: dict) -> lil_matrix:
    """Build the Laplacian diffusion stencil as a sparse LIL matrix.

    Parameters
    ----------
    gd : dict
        Grid-data dictionary from :func:`_extract_grid_data`.

    Returns
    -------
    lil_matrix
        (n × n) sparse matrix representing the discrete Laplacian.
    """
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    valid = gd["valid"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    inv_dx2, inv_dy2 = gd["inv_dx2"], gd["inv_dy2"]

    laplacian_offsets = [
        (-1, 0, inv_dx2), (1, 0, inv_dx2),
        (0, -1, inv_dy2), (0, 1, inv_dy2),
    ]

    M = lil_matrix((n, n), dtype=float)
    for k in range(n):
        i, j = rev_i[k], rev_j[k]
        for di, dj, inv in laplacian_offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                M[k, idx_map[nj, ni]] += inv
                M[k, k] -= inv
            else:
                M[k, k] -= inv
    return M


def _solve_hillslope_diffusion(
    gd: dict,
    hillslope_indices: np.ndarray,
    hillslope_to_reduced: np.ndarray,
    hillslope_mask: np.ndarray,
    D_or_D_eff: float | np.ndarray,
    Z_channel_flat: np.ndarray,
    channel_mask_flat: np.ndarray,
    bc_boundary_inds: np.ndarray,
    bc_boundary_vals: np.ndarray,
    U: float,
) -> np.ndarray:
    """Assemble and solve the reduced diffusion system for hillslope cells.

    Parameters
    ----------
    gd : dict
        Grid-data dictionary.
    hillslope_indices : np.ndarray
        Flat indices of hillslope cells in the full 1-D system.
    hillslope_to_reduced : np.ndarray
        Map from full flat index → reduced matrix row (−1 if not hillslope).
    hillslope_mask : np.ndarray
        Boolean, length *n*, ``True`` for hillslope cells.
    D_or_D_eff : float or np.ndarray
        Scalar (uniform) or per-hillslope-cell diffusivity array.
    Z_channel_flat : np.ndarray
        Pre-computed channel elevations (length *n*).
    channel_mask_flat : np.ndarray
        Boolean channel mask (length *n*).
    bc_boundary_inds, bc_boundary_vals : np.ndarray
        Boundary condition indices and values.
    U : float
        Uplift rate.

    Returns
    -------
    np.ndarray
        1-D array (length *n*) of elevations for all cells.
    """
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    valid = gd["valid"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    inv_dx2, inv_dy2 = gd["inv_dx2"], gd["inv_dy2"]

    n_hillslope = len(hillslope_indices)
    M_reduced = lil_matrix((n_hillslope, n_hillslope), dtype=float)
    b_reduced = np.full(n_hillslope, -U, dtype=float)

    scalar_D = np.isscalar(D_or_D_eff)

    laplacian_offsets = [
        (-1, 0, inv_dx2), (1, 0, inv_dx2),
        (0, -1, inv_dy2), (0, 1, inv_dy2),
    ]

    for idx, k in enumerate(hillslope_indices):
        i, j = rev_i[k], rev_j[k]
        D_here = D_or_D_eff if scalar_D else D_or_D_eff[idx]

        for di, dj, inv in laplacian_offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                neighbor_k = idx_map[nj, ni]
                if hillslope_mask[neighbor_k]:
                    neighbor_idx = hillslope_to_reduced[neighbor_k]
                    M_reduced[idx, neighbor_idx] += D_here * inv
                    M_reduced[idx, idx] -= D_here * inv
                else:
                    if channel_mask_flat[neighbor_k]:
                        neighbor_val = Z_channel_flat[neighbor_k]
                    else:
                        neighbor_val = bc_boundary_vals[neighbor_k]
                    b_reduced[idx] -= D_here * inv * neighbor_val
                    M_reduced[idx, idx] -= D_here * inv
            else:
                M_reduced[idx, idx] -= D_here * inv

    try:
        z_hillslope = spsolve(M_reduced.tocsr(), b_reduced)
    except Exception:
        logger.warning("Hillslope diffusion solve failed; returning NaN", exc_info=True)
        z_hillslope = np.full(n_hillslope, np.nan)

    z_flat = np.full(n, np.nan)
    z_flat[hillslope_indices] = z_hillslope
    z_flat[bc_boundary_inds] = bc_boundary_vals[bc_boundary_inds]
    z_flat[channel_mask_flat] = Z_channel_flat[channel_mask_flat]

    return z_flat


# ---------------------------------------------------------------------------
# Public model factories
# ---------------------------------------------------------------------------

def advection_diffusion_model(
    dem: Any,
    area: Any,
    flow_direction: Any,
    m: float,
) -> Callable[[float, float, float], np.ndarray]:
    """Build a solver for the coupled advection-diffusion steady-state PDE.

    Assembles sparse matrices for the equation

    .. math::

        0 = U - K A^m \\frac{\\partial z}{\\partial x} + D \\nabla^2 z

    and returns a closure ``F(U, K, D) → Z`` that solves for the
    steady-state elevation grid.

    Parameters
    ----------
    dem : Elevation-like object
        Raster with ``._griddata`` (2-D array) and ``._georef_info.dx``.
    area : Area-like object
        Raster with ``._griddata`` (drainage area).
    flow_direction : FlowDirectionD8-like object
        Raster with ``._griddata`` (D8 codes).
    m : float
        Area exponent in the stream-power incision term.

    Returns
    -------
    Callable[[float, float, float], np.ndarray]
        ``output_function(U, K, D)`` returning 2-D steady-state elevation.
    """
    gd = _extract_grid_data(dem, area, flow_direction)
    Z0 = gd["Z0"]
    A = gd["A"]
    FD = gd["FD"]
    dx = gd["dx"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    valid = gd["valid"]
    n = gd["n"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    inv_dx2, inv_dy2 = gd["inv_dx2"], gd["inv_dy2"]
    sqrt2 = np.sqrt(2.0)

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

        # Diffusion stencil
        for di, dj, inv in laplacian_offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                M_diff[k, idx_map[nj, ni]] += inv
                M_diff[k, k] -= inv
            else:
                M_diff[k, k] -= inv

        # Advection stencil
        code = FD[j, i]
        if code in D8_OFFSETS:
            di, dj = D8_OFFSETS[code]
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                dL = dx * (sqrt2 if (di != 0 and dj != 0) else 1.0)
                coeff = -(A[j, i] ** m) / dL
                M_adv[k, k] += coeff
                M_adv[k, idx_map[nj, ni]] -= coeff

    # ── 2. Boundary conditions ────────────────────────────────────────────
    boundary_mask = _make_boundary_mask(valid)
    bc_inds: list[int] = []
    bc_vals = np.zeros(n, dtype=float)

    for j_b, i_b in zip(*np.where(boundary_mask)):
        k = idx_map[j_b, i_b]
        bc_inds.append(k)
        bc_vals[k] = Z0[j_b, i_b]
        b_U[k] = 0.0

    # Outlet handling
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
        """Solve for steady-state elevation.

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


def partitioned_threshold_model(
    dem: Any,
    area: Any,
    flow_direction: Any,
    m: float,
    U: float,
    K: float,
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Build a partitioned threshold model with linear hillslope diffusion.

    Pre-computes the advective channel network, then returns a function
    that applies an area threshold and solves hillslope diffusion.

    The channel cells (A >= A_crit) use the detachment-limited steady-state
    elevation S = (U/K) / A^m integrated from boundaries. Hillslope cells
    are solved with a Laplacian diffusion equation.

    Parameters
    ----------
    dem : Elevation-like object
    area : Area-like object
    flow_direction : FlowDirectionD8-like object
    m : float
        Area exponent in stream-power law.
    U : float
        Uplift rate [m/yr].
    K : float
        Erodibility coefficient.

    Returns
    -------
    Callable
        ``f(A_crit, D, debug=False) → (Z_ss, channel_mask)``
    """
    gd = _extract_grid_data(dem, area, flow_direction)
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    A_flat = gd["A_flat"]

    Z_channel_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals = (
        _precompute_channel_network(gd, m, U, K)
    )

    # Pre-compute diffusion matrix for the output_function's debug path
    inv_dx2, inv_dy2 = gd["inv_dx2"], gd["inv_dy2"]

    def output_function(
        A_crit: float,
        D: float,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply area threshold and solve diffusion.

        Parameters
        ----------
        A_crit : float
            Critical drainage area threshold [m²].
        D : float
            Hillslope diffusivity [m²/yr].
        debug : bool, optional
            If ``True``, emit diagnostic log messages (via ``logging``).

        Returns
        -------
        Z_ss : np.ndarray
            2-D steady-state elevation (Ny × Nx).
        channel_mask : np.ndarray
            2-D boolean mask of channel cells (Ny × Nx).
        """
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("=== PARTITIONED THRESHOLD MODEL DIAGNOSTICS ===")
            logger.debug("Grid %d × %d, valid=%d, A_crit=%g, D=%g", Ny, Nx, n, A_crit, D)

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


def partitioned_threshold_model_nonlinear(
    dem: Any,
    area: Any,
    flow_direction: Any,
    m: float,
    U: float,
    K: float,
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Build a partitioned threshold model with nonlinear hillslope diffusion.

    Same channel treatment as :func:`partitioned_threshold_model`, but the
    hillslope diffusivity is slope-dependent:

    .. math::

        D_{\\text{eff}} = \\frac{D}{1 - (|\\nabla z| / S_c)^2}

    The solver iterates (Picard iteration) until convergence.

    Parameters
    ----------
    dem : Elevation-like object
    area : Area-like object
    flow_direction : FlowDirectionD8-like object
    m : float
        Area exponent in stream-power law.
    U : float
        Uplift rate [m/yr].
    K : float
        Erodibility coefficient.

    Returns
    -------
    Callable
        ``f(A_crit, D_linear, S_c, max_iter=50, tol=1e-6) → (Z_ss, channel_mask)``
    """
    gd = _extract_grid_data(dem, area, flow_direction)
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    valid = gd["valid"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    dx, dy = gd["dx"], gd["dy"]
    A_flat = gd["A_flat"]

    Z_channel_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals = (
        _precompute_channel_network(gd, m, U, K)
    )

    # ── Nested helpers for nonlinear iteration ────────────────────────────

    def _compute_slopes(
        z_flat: np.ndarray,
        hillslope_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute slope magnitude for each hillslope cell via finite differences.

        Parameters
        ----------
        z_flat : np.ndarray
            Current elevation solution (length *n*).
        hillslope_indices : np.ndarray
            Flat indices of hillslope cells.

        Returns
        -------
        np.ndarray
            Slope magnitudes (one per hillslope cell).
        """
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

    # ── Output function ───────────────────────────────────────────────────

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
            Linear (reference) diffusivity [m²/yr].
        S_c : float
            Critical slope for nonlinear diffusion.
        max_iter : int, optional
            Maximum Picard iterations (default 50).
        tol : float, optional
            Convergence tolerance on max elevation change (default 1e-6).

        Returns
        -------
        Z_ss : np.ndarray
            2-D steady-state elevation (Ny × Nx).
        channel_mask : np.ndarray
            2-D boolean mask of channel cells (Ny × Nx).
        """
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

            # Initialize with linear diffusion
            z_flat_prev = _solve_hillslope_diffusion(
                gd, hillslope_indices, hillslope_to_reduced, hillslope_mask,
                D_linear, Z_channel_flat, channel_mask_flat,
                bc_boundary_inds, bc_boundary_vals, U,
            )

            # Picard iteration for nonlinear diffusivity
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


def _find_boundary_value(
    Z0: np.ndarray,
    i: int,
    j: int,
    Nx: int,
    Ny: int,
) -> float:
    """Find a valid elevation on the grid boundary for a given row/column.

    Searches the left, right, top, and bottom edges (in that order) for a
    finite elevation value at the same row or column as ``(i, j)``.

    Parameters
    ----------
    Z0 : np.ndarray
        2-D elevation array.
    i, j : int
        Column and row indices of the query cell.
    Nx, Ny : int
        Grid dimensions.

    Returns
    -------
    float
        The first finite boundary elevation found, or ``np.nan`` if all
        edges are NaN.
    """
    if np.isfinite(Z0[j, 0]):
        return float(Z0[j, 0])
    if np.isfinite(Z0[j, Nx - 1]):
        return float(Z0[j, Nx - 1])
    if np.isfinite(Z0[0, i]):
        return float(Z0[0, i])
    if np.isfinite(Z0[Ny - 1, i]):
        return float(Z0[Ny - 1, i])

    logger.warning(
        "All boundary edges NaN for cell (%d, %d); returning NaN", i, j,
    )
    return np.nan
