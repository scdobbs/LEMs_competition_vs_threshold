"""Advection-diffusion model: 0 = U - K A^m dz/dx + D laplacian(z). Exact port of v1."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from lemcvt.models.common import D8_OFFSETS, extract_grid_data, make_boundary_mask


def advection_diffusion_model(
    Z0: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, m: float,
) -> Callable[[float, float, float], np.ndarray]:
    """Build ``F(U, K, D) -> Z`` for the coupled steady-state AD equation.

    Dirichlet boundary conditions take the observed elevation on every valid
    edge cell and at the outlet (largest-area cell, forced onto the perimeter).
    Each call assembles ``D*M_diff + K*M_adv``, imposes the boundary rows by a
    CSR->LIL->CSR round trip and solves with SuperLU, exactly as v1 did.
    """
    gd = extract_grid_data(Z0, A, FD, dx)
    Z0 = gd["Z0"]
    A = gd["A"]
    FD = gd["FD"]
    Ny, Nx = gd["Ny"], gd["Nx"]
    valid = gd["valid"]
    n = gd["n"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    inv_dx2, inv_dy2 = gd["inv_dx2"], gd["inv_dy2"]
    sqrt2 = np.sqrt(2.0)

    M_diff = lil_matrix((n, n), dtype=float)
    M_adv = lil_matrix((n, n), dtype=float)

    b_U = np.full(n, -1.0, dtype=float)
    b_elev = np.zeros(n, dtype=float)

    laplacian_offsets = [(-1, 0, inv_dx2), (1, 0, inv_dx2), (0, -1, inv_dy2), (0, 1, inv_dy2)]

    for k in range(n):
        i, j = rev_i[k], rev_j[k]

        for di, dj, inv in laplacian_offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                M_diff[k, idx_map[nj, ni]] += inv
                M_diff[k, k] -= inv
            else:
                M_diff[k, k] -= inv

        code = FD[j, i]
        if code in D8_OFFSETS:
            di, dj = D8_OFFSETS[code]
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                dL = dx * (sqrt2 if (di != 0 and dj != 0) else 1.0)
                coeff = -(A[j, i] ** m) / dL
                M_adv[k, k] += coeff
                M_adv[k, idx_map[nj, ni]] -= coeff

    boundary_mask = make_boundary_mask(valid)
    bc_inds: list[int] = []
    bc_vals = np.zeros(n, dtype=float)

    for j_b, i_b in zip(*np.where(boundary_mask), strict=True):
        k = idx_map[j_b, i_b]
        bc_inds.append(k)
        bc_vals[k] = Z0[j_b, i_b]
        b_U[k] = 0.0

    j_out, i_out = np.unravel_index(np.nanargmax(A), A.shape)
    if not ((j_out == 0 or j_out == Ny - 1 or i_out == 0 or i_out == Nx - 1) and valid[j_out, i_out]):
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

    M_diff_csr = M_diff.tocsr()
    M_adv_csr = M_adv.tocsr()
    bc_inds_arr = np.array(bc_inds)

    def output_function(U: float, K: float, D: float) -> np.ndarray:
        M = D * M_diff_csr + K * M_adv_csr
        b = U * b_U + b_elev

        M = M.tolil()
        M[bc_inds_arr, :] = 0.0
        M[bc_inds_arr, bc_inds_arr] = 1.0
        b[bc_inds_arr] = bc_vals[bc_inds_arr]

        z_flat = spsolve(M.tocsr(), b)

        Z = np.full_like(Z0, np.nan, dtype=float)
        Z[rev_j, rev_i] = z_flat
        return Z

    output_function.n_valid = n  # type: ignore[attr-defined]
    return output_function
