"""D8 flow direction and drainage area, reproducing TopoAnalysis bit for bit.

Code convention (ArcGIS style, TopoAnalysis ``FlowDirectionD8``)::

    |32  64 128|      1 = east (i, j+1)      2 = south-east (i+1, j+1)
    |16   x   1|      4 = south (i+1, j)     8 = south-west (i+1, j-1)
    | 8   4   2|     16 = west (i, j-1)     32 = north-west (i-1, j-1)
                     64 = north (i-1, j)   128 = north-east (i-1, j+1)
    0 = no direction (NaN elevation or no finite drop computed)

where ``i`` is the row and ``j`` the column. The models in :mod:`lemcvt.models`
index arrays as ``Z[j, i]`` with ``i`` the column, so their ``D8_OFFSETS`` are
expressed as ``(di_col, dj_row)``; :data:`D8_OFFSETS` here is in (row, col).

Quirks of the TopoAnalysis implementation that are reproduced deliberately
(Phase 1: numerics unchanged):

* diagonal drops are divided by 1.41, not sqrt(2);
* the slab assignments stop two cells short of the right/bottom edges, so the
  last row and column never see an east/south comparison;
* on ties the highest code wins (later assignments overwrite);
* a cell whose steepest drop is zero or negative still receives a code (the
  least-uphill neighbour), which is what the priority-flood aggradation slope
  normally prevents.
"""

from __future__ import annotations

import numpy as np

D8_OFFSETS: dict[int, tuple[int, int]] = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}
"""D8 code -> (drow, dcol)."""

_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int64)
_DROW = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_DCOL = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)


def d8_flow_direction(z_filled: np.ndarray) -> np.ndarray:
    """Exact port of ``FlowDirectionD8._create_from_flooded_dem``.

    Parameters
    ----------
    z_filled
        2-D filled elevation (float). NaN cells get code 0.

    Returns
    -------
    np.ndarray of uint8
        D8 codes.
    """
    Z = np.asarray(z_filled, dtype=np.float64)
    big = -np.ones_like(Z) * 1e22

    ijp1 = big.copy()
    ijp1[:, 0:-2] = Z[:, 0:-2] - Z[:, 1:-1]
    ip1jp1 = big.copy()
    ip1jp1[0:-2, 0:-2] = (Z[0:-2, 0:-2] - Z[1:-1, 1:-1]) / 1.41
    ip1j = big.copy()
    ip1j[0:-2, :] = Z[0:-2, :] - Z[1:-1, :]
    ip1jm1 = big.copy()
    ip1jm1[0:-2, 1:-1] = (Z[0:-2, 1:-1] - Z[1:-1, 0:-2]) / 1.41
    ijm1 = big.copy()
    ijm1[:, 1:-1] = Z[:, 1:-1] - Z[:, 0:-2]
    im1jm1 = big.copy()
    im1jm1[1:-1, 1:-1] = (Z[1:-1, 1:-1] - Z[0:-2, 0:-2]) / 1.41
    im1j = big.copy()
    im1j[1:-1, :] = Z[1:-1, :] - Z[0:-2, :]
    im1jp1 = big.copy()
    im1jp1[1:-1, 0:-2] = (Z[1:-1, 0:-2] - Z[0:-2, 1:-1]) / 1.41

    min_sl = np.fmax(
        np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(ijp1, ip1jp1), ip1j), ip1jm1), ijm1), im1jm1), im1j),
        im1jp1,
    )

    fd = np.zeros_like(Z, dtype=np.uint8)
    fd[ijp1 == min_sl] = 1
    fd[ip1jp1 == min_sl] = 2
    fd[ip1j == min_sl] = 4
    fd[ip1jm1 == min_sl] = 8
    fd[ijm1 == min_sl] = 16
    fd[im1jm1 == min_sl] = 32
    fd[im1j == min_sl] = 64
    fd[im1jp1 == min_sl] = 128
    fd[np.isnan(Z)] = 0
    return fd


def d8_receiver_arrays(fd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Receiver (row, col) of every cell and a validity mask.

    Equivalent to TopoAnalysis ``get_flow_to_cell``: a receiver is valid only
    if the code is one of the eight directions and the target is inside the grid.
    """
    fd = np.asarray(fd)
    ny, nx = fd.shape
    rows, cols = np.indices((ny, nx))
    r_to = np.full((ny, nx), -1, dtype=np.int64)
    c_to = np.full((ny, nx), -1, dtype=np.int64)
    for code, dr, dc in zip(_CODES, _DROW, _DCOL, strict=True):
        m = fd == code
        r_to[m] = rows[m] + dr
        c_to[m] = cols[m] + dc
    good = (r_to >= 0) & (r_to < ny) & (c_to >= 0) & (c_to < nx) & (fd != 0)
    r_to[~good] = -1
    c_to[~good] = -1
    return r_to, c_to, good


def d8_area(fd: np.ndarray, z_filled: np.ndarray, dx: float) -> np.ndarray:
    """Exact port of TopoAnalysis ``Area(flow_direction=...)``.

    Cells are processed from highest to lowest *filled elevation* (the order
    ``FlowDirectionD8`` stored when it was built: ``argsort(axis=None)`` of the
    filled grid, reversed) and each passes its accumulated area to its receiver.
    Every cell, NaN ones included, starts with ``dx**2``; NaN cells have code 0
    and therefore never pass anything on.

    Tie-break (``docs/numerics_changes.md``, N1): cells of equal filled elevation are
    processed in ascending flat index. TopoAnalysis left the tie order to numpy's
    unstable default sort, which depends on the numpy version.
    """
    Z = np.asarray(z_filled, dtype=np.float64)
    fd = np.asarray(fd)
    ny, nx = fd.shape
    area = dx**2 * np.ones((ny, nx), dtype=np.float64)
    Zf = Z.ravel()
    order = np.lexsort((np.arange(Zf.size), -Zf))   # Z descending, ties by ascending flat index
    r_to, c_to, good = d8_receiver_arrays(fd)
    r_to_f = r_to.ravel()
    c_to_f = c_to.ravel()
    good_f = good.ravel()
    area_f = area.ravel()  # view
    for k in order:
        if good_f[k]:
            area_f[r_to_f[k] * nx + c_to_f[k]] += area_f[k]
    return area


def d8_divides(fd: np.ndarray) -> np.ndarray:
    """Exact port of TopoAnalysis ``FlowDirectionD8.divides``.

    A divide cell is one that receives flow from none of its eight neighbours.
    As in TopoAnalysis the outer two rows/columns are never divides.
    """
    fd = np.asarray(fd)
    div = np.zeros_like(fd, dtype=np.float64)
    ip1j = fd[2:-1, 1:-2]
    im1j = fd[0:-3, 1:-2]
    ijp1 = fd[1:-2, 2:-1]
    ijm1 = fd[1:-2, 0:-3]
    ip1jp1 = fd[2:-1, 2:-1]
    ip1jm1 = fd[2:-1, 0:-3]
    im1jp1 = fd[0:-3, 2:-1]
    im1jm1 = fd[0:-3, 0:-3]
    div[1:-2, 1:-2] = (
        (ijm1 != 1) & (im1jm1 != 2) & (im1j != 4) & (im1jp1 != 8)
        & (ijp1 != 16) & (ip1jp1 != 32) & (ip1j != 64) & (ip1jm1 != 128)
    )
    return div


def d8_upstream_lists(fd: np.ndarray, valid: np.ndarray | None = None) -> list[list[tuple[int, int]]]:
    """For every cell (flat index), the (row, col) of cells draining into it."""
    fd = np.asarray(fd)
    ny, nx = fd.shape
    r_to, c_to, good = d8_receiver_arrays(fd)
    ups: list[list[tuple[int, int]]] = [[] for _ in range(ny * nx)]
    rows, cols = np.nonzero(good if valid is None else (good & valid))
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        ups[r_to[r, c] * nx + c_to[r, c]].append((r, c))
    return ups
