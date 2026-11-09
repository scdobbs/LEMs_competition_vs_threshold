"""Advection-diffusion model: 0 = U - K A^m dz/dx + D laplacian(z).

``advection_diffusion_model`` (Phase 3, step 3) assembles the diffusion and advection
stencils from index arrays and imposes the Dirichlet rows once, at build time, by
removing those rows from the stencils and adding an identity on them; each evaluation
is then ``D*M_diff0 + K*M_adv0 + I_bc`` and one SuperLU solve, with no CSR -> LIL -> CSR
round trip. The matrix entries, their pattern and the right-hand side are identical to
the v1 construction (``advection_diffusion_model_v1``, kept for the bit-identity test in
``tests/test_models.py``): the diagonal accumulates ``-inv`` once per neighbour in v1's
offset order, every off-diagonal is a single product, and the boundary rows hold a single
1.0 on the diagonal.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from lemcvt.models.common import D8_OFFSETS, LAPLACIAN_OFFSETS, extract_grid_data, make_boundary_mask


def _outlet_on_perimeter(A: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    Ny, Nx = A.shape
    j_out, i_out = np.unravel_index(np.nanargmax(A), A.shape)
    if not ((j_out == 0 or j_out == Ny - 1 or i_out == 0 or i_out == Nx - 1) and valid[j_out, i_out]):
        for ii, jj in [(i_out, 0), (i_out, Ny - 1), (0, j_out), (Nx - 1, j_out)]:
            if valid[jj, ii]:
                return int(ii), int(jj)
        raise RuntimeError("Cannot force the outlet onto a valid perimeter cell.")
    return int(i_out), int(j_out)


def advection_diffusion_model(
    Z0: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, m: float,
) -> Callable[[float, float, float], np.ndarray]:
    """Build ``F(U, K, D) -> Z`` for the coupled steady-state AD equation (vectorised, Phase 3 step 3).

    Same equations, boundary conditions and solver as v1; see the module docstring.
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
    inv = {"x": gd["inv_dx2"], "y": gd["inv_dy2"]}
    sqrt2 = np.sqrt(2.0)
    ar = np.arange(n)

    # diffusion stencil: entries exactly as the v1 loop produced them
    diag = np.zeros(n)
    rows, cols, vals = [], [], []
    for di, dj, axis in LAPLACIAN_OFFSETS:
        w = inv[axis]
        ni, nj = rev_i + di, rev_j + dj
        inside = (ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny)
        nk = np.full(n, -1, dtype=np.int64)
        nk[inside] = idx_map[nj[inside], ni[inside]]
        ok = nk >= 0
        rows.append(ar[ok])
        cols.append(nk[ok])
        vals.append(np.full(int(ok.sum()), w))
        diag -= w
    rows.append(ar)
    cols.append(ar)
    vals.append(diag)
    M_diff = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()

    # advection stencil: coeff = -(A^m)/dL on the diagonal, -coeff on the D8 receiver
    codes = FD[rev_j, rev_i].astype(int)
    di_arr = np.zeros(n, dtype=np.int64)
    dj_arr = np.zeros(n, dtype=np.int64)
    has = np.zeros(n, dtype=bool)
    for code, (di, dj) in D8_OFFSETS.items():
        sel = codes == code
        di_arr[sel], dj_arr[sel], has[sel] = di, dj, True
    ni, nj = rev_i + di_arr, rev_j + dj_arr
    inside = has & (ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny)
    nk = np.full(n, -1, dtype=np.int64)
    nk[inside] = idx_map[nj[inside], ni[inside]]
    ok = nk >= 0
    dL = np.where((di_arr != 0) & (dj_arr != 0), dx * sqrt2, dx)
    Am = A[rev_j, rev_i] ** m
    coeff = -(Am[ok]) / dL[ok]
    M_adv = coo_matrix((np.concatenate([coeff, -coeff]), (np.concatenate([ar[ok], ar[ok]]), np.concatenate([ar[ok], nk[ok]]))),
                       shape=(n, n)).tocsr()

    # Dirichlet rows: valid edge cells and the outlet (forced onto the perimeter), imposed once
    b_U = np.full(n, -1.0)
    bc_vals = np.zeros(n)
    boundary_mask = make_boundary_mask(valid)
    jb, ib = np.where(boundary_mask)
    bc_inds = idx_map[jb, ib]
    bc_vals[bc_inds] = Z0[jb, ib]
    i_out, j_out = _outlet_on_perimeter(A, valid)
    kout = idx_map[j_out, i_out]
    if kout not in set(bc_inds.tolist()):
        bc_inds = np.append(bc_inds, kout)
        bc_vals[kout] = Z0[j_out, i_out]
    b_U[bc_inds] = 0.0
    keep = np.ones(n)
    keep[bc_inds] = 0.0
    R = csr_matrix((keep, (ar, ar)), shape=(n, n))            # zero the Dirichlet rows of the stencils
    M_diff0 = (R @ M_diff).tocsr()
    M_adv0 = (R @ M_adv).tocsr()
    M_diff0.eliminate_zeros()
    M_adv0.eliminate_zeros()
    I_bc = csr_matrix((np.ones(bc_inds.size), (bc_inds, bc_inds)), shape=(n, n))
    bc_inds_arr = np.asarray(bc_inds)

    def output_function(U: float, K: float, D: float) -> np.ndarray:
        M = (D * M_diff0 + K * M_adv0 + I_bc).tocsr()
        b = U * b_U
        b[bc_inds_arr] = bc_vals[bc_inds_arr]
        z_flat = spsolve(M, b)
        Z = np.full_like(Z0, np.nan, dtype=float)
        Z[rev_j, rev_i] = z_flat
        return Z

    output_function.n_valid = n  # type: ignore[attr-defined]
    output_function.matrices = (M_diff0, M_adv0, I_bc)  # type: ignore[attr-defined]
    output_function.bc = (bc_inds_arr, bc_vals, b_U)  # type: ignore[attr-defined]
    return output_function


def advection_diffusion_model_v1(
    Z0: np.ndarray, A: np.ndarray, FD: np.ndarray, dx: float, m: float,
) -> Callable[[float, float, float], np.ndarray]:
    """v1 construction (per-cell LIL loops; CSR -> LIL -> CSR Dirichlet rows per solve). Reference only.

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
