"""Shared pieces of the D8 models: grid flattening, boundary mask, channel-network
integration and the reduced hillslope diffusion solve.

Exact port of the private helpers in v1 ``src/models.py``. The array convention
is v1's: ``Z[j, i]`` with ``i`` the column (x) and ``j`` the row (y), and
``D8_OFFSETS`` maps a code to ``(di, dj)`` = (column step, row step).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)

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
"""D8 code -> (di_col, dj_row)."""


def make_boundary_mask(valid: np.ndarray) -> np.ndarray:
    """Edge cells that are also valid."""
    edges = np.zeros_like(valid, dtype=bool)
    edges[0, :] = edges[-1, :] = edges[:, 0] = edges[:, -1] = True
    return edges & valid


def extract_grid_data(Z0: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float) -> dict:
    """Flatten the valid cells (``isfinite(A)``) and build index maps (v1 ``_extract_grid_data``)."""
    Z0 = np.asarray(Z0)
    A = np.asarray(A)
    FD = np.asarray(FD)
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

    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2

    return dict(
        Z0=Z0, A=A, FD=FD, dx=dx, dy=dy, Ny=Ny, Nx=Nx,
        valid=valid, n=n, idx_map=idx_map, rev_i=rev_i, rev_j=rev_j,
        A_flat=A_flat, FD_flat=FD_flat, inv_dx2=inv_dx2, inv_dy2=inv_dy2,
    )


def find_boundary_value(Z0: np.ndarray, i: int, j: int, Nx: int, Ny: int) -> float:
    """First finite elevation on the left, right, top or bottom edge of row j / column i."""
    if np.isfinite(Z0[j, 0]):
        return float(Z0[j, 0])
    if np.isfinite(Z0[j, Nx - 1]):
        return float(Z0[j, Nx - 1])
    if np.isfinite(Z0[0, i]):
        return float(Z0[0, i])
    if np.isfinite(Z0[Ny - 1, i]):
        return float(Z0[Ny - 1, i])
    logger.warning("All boundary edges NaN for cell (%d, %d); returning NaN", i, j)
    return np.nan


def precompute_channel_network(gd: dict, m: float, U: float, K: float):
    """Integrate S0 = (U/K)/A^m upstream from the boundary along D8 (v1 BFS).

    Returns ``(Z_channel_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals)``.
    """
    Z0 = gd["Z0"]
    dx = gd["dx"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    n = gd["n"]
    valid = gd["valid"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    A_flat, FD_flat = gd["A_flat"], gd["FD_flat"]

    S0_flat = (U / K) / (A_flat**m)

    downstream_of = np.full(n, -1, dtype=int)
    for k in range(n):
        i, j = rev_i[k], rev_j[k]
        code = int(FD_flat[k])
        if code in D8_OFFSETS:
            di, dj = D8_OFFSETS[code]
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                downstream_of[k] = idx_map[nj, ni]

    boundary_mask = make_boundary_mask(valid)
    Z_channel_all = np.full(n, np.nan)

    bc_boundary_inds_list: list[int] = []
    bc_boundary_vals = np.zeros(n, dtype=float)

    for j_b, i_b in zip(*np.where(boundary_mask), strict=True):
        k = idx_map[j_b, i_b]
        elev = Z0[j_b, i_b] if np.isfinite(Z0[j_b, i_b]) else find_boundary_value(Z0, i_b, j_b, Nx, Ny)
        Z_channel_all[k] = elev
        bc_boundary_inds_list.append(k)
        bc_boundary_vals[k] = elev

    bc_boundary_inds = np.array(bc_boundary_inds_list)

    j_out, i_out = np.unravel_index(np.nanargmax(gd["A"]), (Ny, Nx))
    k_out = idx_map[j_out, i_out]

    upstream_neighbors: list[list[int]] = [[] for _ in range(n)]
    for k in range(n):
        k_down = downstream_of[k]
        if k_down >= 0:
            upstream_neighbors[k_down].append(k)

    queue = deque(k for k in range(n) if not np.isnan(Z_channel_all[k]))
    processed_count = 0

    while queue:
        k_current = queue.popleft()
        for k_upstream in upstream_neighbors[k_current]:
            if not np.isnan(Z_channel_all[k_upstream]):
                continue
            z_down = Z_channel_all[k_current]
            if np.isnan(z_down):
                continue
            code = int(FD_flat[k_upstream])
            if code not in D8_OFFSETS:
                continue
            di, dj = D8_OFFSETS[code]
            dL = dx * (np.sqrt(2.0) if (di != 0 and dj != 0) else 1.0)
            s_here = S0_flat[k_upstream]
            if not np.isfinite(s_here):
                continue
            z_here = z_down + s_here * dL
            if not np.isfinite(z_here):
                continue
            Z_channel_all[k_upstream] = z_here
            queue.append(k_upstream)
            processed_count += 1

    logger.debug("Channel propagation: %d cells processed, %d NaNs remaining out of %d",
                 processed_count, int(np.sum(np.isnan(Z_channel_all))), n)

    return Z_channel_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals


def build_diffusion_stencil(gd: dict) -> lil_matrix:
    """Full-grid Laplacian as LIL (v1 ``_build_diffusion_stencil``; kept for completeness)."""
    n = gd["n"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    valid = gd["valid"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    inv_dx2, inv_dy2 = gd["inv_dx2"], gd["inv_dy2"]

    laplacian_offsets = [(-1, 0, inv_dx2), (1, 0, inv_dx2), (0, -1, inv_dy2), (0, 1, inv_dy2)]

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


def solve_hillslope_diffusion(
    gd: dict,
    hillslope_indices: np.ndarray,
    hillslope_to_reduced: np.ndarray,
    hillslope_mask: np.ndarray,
    D_or_D_eff,
    Z_channel_flat: np.ndarray,
    channel_mask_flat: np.ndarray,
    bc_boundary_inds: np.ndarray,
    bc_boundary_vals: np.ndarray,
    U: float,
) -> np.ndarray:
    """Assemble and solve the reduced diffusion system for hillslope cells (v1 exact)."""
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

    laplacian_offsets = [(-1, 0, inv_dx2), (1, 0, inv_dx2), (0, -1, inv_dy2), (0, 1, inv_dy2)]

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
    except Exception:  # noqa: BLE001
        logger.warning("Hillslope diffusion solve failed; returning NaN", exc_info=True)
        z_hillslope = np.full(n_hillslope, np.nan)

    z_flat = np.full(n, np.nan)
    z_flat[hillslope_indices] = z_hillslope
    z_flat[bc_boundary_inds] = bc_boundary_vals[bc_boundary_inds]
    z_flat[channel_mask_flat] = Z_channel_flat[channel_mask_flat]
    return z_flat
