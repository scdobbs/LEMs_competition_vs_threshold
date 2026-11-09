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
from scipy.sparse import coo_matrix, lil_matrix
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


def _receivers_vectorised(gd: dict) -> tuple[np.ndarray, np.ndarray]:
    """``downstream_of`` (flat valid index of the D8 receiver, -1 if none) and ``dL`` per cell."""
    Ny, Nx, n = gd["Ny"], gd["Nx"], gd["n"]
    idx_map, rev_i, rev_j, dx = gd["idx_map"], gd["rev_i"], gd["rev_j"], gd["dx"]
    codes = gd["FD_flat"].astype(int)
    di_arr = np.zeros(n, dtype=np.int64)
    dj_arr = np.zeros(n, dtype=np.int64)
    has = np.zeros(n, dtype=bool)
    for code, (di, dj) in D8_OFFSETS.items():
        sel = codes == code
        di_arr[sel], dj_arr[sel], has[sel] = di, dj, True
    ni, nj = rev_i + di_arr, rev_j + dj_arr
    inside = has & (ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny)
    downstream_of = np.full(n, -1, dtype=np.int64)
    downstream_of[inside] = idx_map[nj[inside], ni[inside]]
    dL = np.where((di_arr != 0) & (dj_arr != 0), dx * np.sqrt(2.0), dx)
    return downstream_of, dL


def _upstream_csr(downstream_of: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Donors of every cell in CSR form: ``donors[starts[k]:starts[k+1]]`` in ascending donor index (v1's list order)."""
    order = np.argsort(downstream_of, kind="stable")
    donors = order[downstream_of[order] >= 0]
    recv = downstream_of[donors]
    starts = np.searchsorted(recv, np.arange(n + 1))
    return donors, starts


def _bfs_integrate_py(seeds, donors, starts, step, z):
    """Breadth-first upstream integration z[u] = z[c] + step[u] (reference Python loop; v1 queue order)."""
    queue = deque(seeds.tolist())
    while queue:
        kc = queue.popleft()
        for ku in donors[starts[kc]:starts[kc + 1]].tolist():
            if not np.isnan(z[ku]):
                continue
            st = step[ku]
            if not np.isfinite(st):
                continue
            z_here = z[kc] + st
            if not np.isfinite(z_here):
                continue
            z[ku] = z_here
            queue.append(ku)
    return z


try:  # Phase 3 step 5: the same loop compiled with numba when available
    import numba as _nb

    @_nb.njit(cache=True)
    def _bfs_integrate_nb(seeds, donors, starts, step, z):  # pragma: no cover (compiled)
        n = z.shape[0]
        queue = np.empty(n, dtype=np.int64)
        head = 0
        tail = 0
        for s_ in seeds:
            queue[tail] = s_
            tail += 1
        while head < tail:
            kc = queue[head]
            head += 1
            for p_ in range(starts[kc], starts[kc + 1]):
                ku = donors[p_]
                if not np.isnan(z[ku]):
                    continue
                st = step[ku]
                if not np.isfinite(st):
                    continue
                z_here = z[kc] + st
                if not np.isfinite(z_here):
                    continue
                z[ku] = z_here
                queue[tail] = ku
                tail += 1
        return z

    _bfs_integrate = _bfs_integrate_nb
    HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    _bfs_integrate = _bfs_integrate_py
    HAVE_NUMBA = False


def _boundary_anchor(gd: dict):
    Z0, Ny, Nx = gd["Z0"], gd["Ny"], gd["Nx"]
    valid, idx_map = gd["valid"], gd["idx_map"]
    boundary_mask = make_boundary_mask(valid)
    jb, ib = np.where(boundary_mask)
    bc_boundary_inds = idx_map[jb, ib]
    vals = Z0[jb, ib].astype(float)
    bad = ~np.isfinite(vals)
    for t in np.nonzero(bad)[0].tolist():
        vals[t] = find_boundary_value(Z0, int(ib[t]), int(jb[t]), Nx, Ny)
    bc_boundary_vals = np.zeros(gd["n"], dtype=float)
    bc_boundary_vals[bc_boundary_inds] = vals
    j_out, i_out = np.unravel_index(np.nanargmax(gd["A"]), (Ny, Nx))
    k_out = int(idx_map[j_out, i_out])
    return bc_boundary_inds, bc_boundary_vals, k_out


def precompute_channel_network(gd: dict, m: float, U: float, K: float):
    """Integrate S0 = (U/K)/A^m upstream from the boundary along D8 (v1 BFS; Phase 3 step 5 implementation).

    Same arithmetic as v1 (``z_up = z_down + S0_up * dL_up`` in breadth-first order from the
    boundary cells) with the receiver table, the donor lists and the queue seeds built with
    numpy and the loop compiled with numba when available (``_bfs_integrate_py`` is the
    reference). Returns ``(Z_channel_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals)``.
    """
    n = gd["n"]
    S0_flat = (U / K) / (gd["A_flat"]**m)
    downstream_of, dL = _receivers_vectorised(gd)
    bc_boundary_inds, bc_boundary_vals, k_out = _boundary_anchor(gd)
    Z_channel_all = np.full(n, np.nan)
    Z_channel_all[bc_boundary_inds] = bc_boundary_vals[bc_boundary_inds]
    donors, starts = _upstream_csr(downstream_of, n)
    step = S0_flat * dL
    Z_channel_all = _bfs_integrate(np.asarray(bc_boundary_inds, dtype=np.int64), donors, starts, step, Z_channel_all)
    logger.debug("Channel propagation: %d NaNs remaining out of %d", int(np.sum(np.isnan(Z_channel_all))), n)
    return Z_channel_all, downstream_of.astype(int), k_out, np.asarray(bc_boundary_inds), bc_boundary_vals


def precompute_channel_network_v1(gd: dict, m: float, U: float, K: float):
    """v1 per-cell Python construction and BFS, kept as the reference for the tests."""
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



def precompute_chi_network(gd: dict, m: float):
    """chi_D8 = sum A^-m dL and the anchor elevation of every cell's D8 path (Phase 3, step 4).

    The same BFS as :func:`precompute_channel_network` with the factor U/K taken out:
    ``Z_channel(K) = z_anchor + (U/K) * chi``. Rounding differs from the v1 integration at
    the last bit (the sum is scaled once instead of term by term; ``docs/numerics_changes.md``
    N6). Returns ``(chi_all, z_anchor_all, downstream_of, k_out, bc_boundary_inds, bc_boundary_vals)``.
    """
    n = gd["n"]
    downstream_of, dL = _receivers_vectorised(gd)
    bc_boundary_inds, bc_boundary_vals, k_out = _boundary_anchor(gd)
    donors, starts = _upstream_csr(downstream_of, n)
    step = (1.0 / (gd["A_flat"]**m)) * dL
    chi = np.full(n, np.nan)
    chi[bc_boundary_inds] = 0.0
    chi = _bfs_integrate(np.asarray(bc_boundary_inds, dtype=np.int64), donors, starts, step, chi)
    # anchor elevation: propagate the boundary value along the same tree (zero step)
    z_anchor = np.full(n, np.nan)
    z_anchor[bc_boundary_inds] = bc_boundary_vals[bc_boundary_inds]
    z_anchor = _bfs_integrate(np.asarray(bc_boundary_inds, dtype=np.int64), donors, starts, np.where(np.isfinite(step), 0.0, np.nan), z_anchor)
    return chi, z_anchor, downstream_of.astype(int), k_out, np.asarray(bc_boundary_inds), bc_boundary_vals


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


LAPLACIAN_OFFSETS = ((-1, 0, "x"), (1, 0, "x"), (0, -1, "y"), (0, 1, "y"))
"""Neighbour offsets ``(di_col, dj_row, axis)`` in the order v1 visited them."""


def assemble_hillslope_system(
    gd: dict,
    hillslope_indices: np.ndarray,
    hillslope_to_reduced: np.ndarray,
    hillslope_mask: np.ndarray,
    D_or_D_eff,
    Z_channel_flat: np.ndarray,
    channel_mask_flat: np.ndarray,
    bc_boundary_vals: np.ndarray,
    U: float,
):
    """Vectorised assembly of the reduced hillslope system ``M z = b`` (Phase 3, step 2).

    Produces the same matrix entries and right-hand side as the v1 per-cell loop
    (:func:`solve_hillslope_diffusion_v1`), including the order of the floating-point
    operations: the diagonal accumulates ``-D_i * inv`` once per neighbour in v1's offset
    order, off-diagonals are single products ``D_i * inv``, and fixed neighbours subtract
    ``(D_i * inv) * value`` from ``b`` in the same order. Returns ``(M_csr, b)``.
    """
    Ny, Nx = gd["Ny"], gd["Nx"]
    idx_map = gd["idx_map"]
    rev_i, rev_j = gd["rev_i"], gd["rev_j"]
    inv = {"x": gd["inv_dx2"], "y": gd["inv_dy2"]}
    hs = np.asarray(hillslope_indices)
    n_h = hs.size
    i = rev_i[hs]
    j = rev_j[hs]
    D = np.broadcast_to(np.asarray(D_or_D_eff, dtype=float), (n_h,)) if np.isscalar(D_or_D_eff) \
        else np.asarray(D_or_D_eff, dtype=float)
    diag = np.zeros(n_h, dtype=float)
    b = np.full(n_h, -U, dtype=float)
    rows, cols, vals = [], [], []
    ar = np.arange(n_h)
    for di, dj, axis in LAPLACIAN_OFFSETS:
        w = D * inv[axis]                                  # D_here * inv, as v1 multiplies it
        ni = i + di
        nj = j + dj
        inside = (ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny)
        nk = np.full(n_h, -1, dtype=np.int64)
        nk[inside] = idx_map[nj[inside], ni[inside]]       # -1 where the neighbour is not a valid cell
        valid_n = nk >= 0
        diag -= w                                          # every case subtracts D_here * inv from the diagonal
        nk_safe = np.where(valid_n, nk, 0)
        is_hs = valid_n & hillslope_mask[nk_safe]
        rows.append(ar[is_hs])
        cols.append(hillslope_to_reduced[nk[is_hs]])
        vals.append(w[is_hs])
        fixed = valid_n & ~is_hs
        if fixed.any():
            nkf = nk[fixed]
            val = np.where(channel_mask_flat[nkf], Z_channel_flat[nkf], bc_boundary_vals[nkf])
            b[fixed] -= w[fixed] * val
    rows.append(ar)
    cols.append(ar)
    vals.append(diag)
    M = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_h, n_h)).tocsr()
    return M, b


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
    """Assemble (vectorised, Phase 3 step 2) and solve the reduced diffusion system for hillslope cells.

    Same matrix, right-hand side and solver as :func:`solve_hillslope_diffusion_v1`; the only
    change is how the matrix is built (``tests/test_models.py`` checks bit-identity).
    """
    n = gd["n"]
    M_reduced, b_reduced = assemble_hillslope_system(gd, hillslope_indices, hillslope_to_reduced, hillslope_mask,
                                                     D_or_D_eff, Z_channel_flat, channel_mask_flat, bc_boundary_vals, U)
    try:
        z_hillslope = spsolve(M_reduced, b_reduced)
    except Exception:  # noqa: BLE001
        logger.warning("Hillslope diffusion solve failed; returning NaN", exc_info=True)
        z_hillslope = np.full(len(hillslope_indices), np.nan)

    z_flat = np.full(n, np.nan)
    z_flat[hillslope_indices] = z_hillslope
    z_flat[bc_boundary_inds] = bc_boundary_vals[bc_boundary_inds]
    z_flat[channel_mask_flat] = Z_channel_flat[channel_mask_flat]
    return z_flat


def solve_hillslope_diffusion_v1(
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
    """v1 per-cell LIL assembly of the reduced diffusion system. Kept as the reference for
    ``solve_hillslope_diffusion`` (Phase 3, step 2), which must reproduce it bit for bit."""
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
