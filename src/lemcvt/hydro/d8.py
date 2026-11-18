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

Border guard (addendum 16, section 5). Because of the second quirk, a grid whose
valid cells reach the grid border is not routed correctly at that border: the last
two rows and columns never see an east/south comparison, so valid cells there
receive a least-uphill code and can form two-cell cycles (found on the GM 1 m full
grid, addendum 15 section 3: 134 cycles and an unclosed area budget). The models
in v1/v2 ran their 1 km tiles with valid cells on every border under exactly this
treatment (their stored flow directions were clipped from larger grids; the derived
grids of model surfaces were not). :func:`d8_flow_direction` therefore refuses a
grid with valid border cells unless the caller designates how the border is
treated (``border="legacy"`` keeps the TopoAnalysis behaviour, ``border="outlets"``
makes every data-edge cell an outlet, or an explicit ``outlets`` mask covering the
border); a grid padded with NaN so that no valid cell touches the border needs no
designation. The D8 algorithm itself is unchanged.
"""

from __future__ import annotations

from typing import Any

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


BORDER_TREATMENTS = ("legacy", "outlets")


class D8BorderError(ValueError):
    """A grid with valid cells on its border was given to D8 without a border treatment."""


def valid_border_cells(z: np.ndarray) -> np.ndarray:
    """Boolean mask of finite cells in the first/last row and column of ``z``."""
    Z = np.asarray(z)
    ring = np.zeros(Z.shape, dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return ring & np.isfinite(Z)


def data_edge_cells(z: np.ndarray) -> np.ndarray:
    """Finite cells with a NaN neighbour (8-connected) or on the grid border: the priority flood's edge set."""
    Z = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(Z)
    ny, nx = valid.shape
    pad = np.pad(valid, 1, constant_values=False)
    adj = np.zeros_like(valid)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            adj |= ~pad[1 + dr:ny + 1 + dr, 1 + dc:nx + 1 + dc]
    return adj & valid


def check_border(z_filled: np.ndarray, border: str | None, outlets: np.ndarray | None, name: str = "grid") -> int:
    """The guard: return the number of valid border cells; raise :class:`D8BorderError` if there are any and
    neither a ``border`` treatment nor an ``outlets`` mask covering them was designated."""
    if border is not None and border not in BORDER_TREATMENTS:
        raise ValueError(f"border must be one of {BORDER_TREATMENTS} or None, got {border!r}")
    vb = valid_border_cells(z_filled)
    n = int(vb.sum())
    if n == 0 or border is not None:
        return n
    if outlets is not None:
        uncovered = int((vb & ~np.asarray(outlets, dtype=bool)).sum())
        if uncovered == 0:
            return n
        detail = f"the outlets mask leaves {uncovered} of them uncovered"
    else:
        detail = "no border treatment was designated"
    ny, nx = np.asarray(z_filled).shape
    raise D8BorderError(
        f"D8 flow direction: {name!r} ({ny} x {nx}) has {n} valid cells on its grid border and {detail}. "
        "The TopoAnalysis port never compares the last two rows and columns eastward/southward, so border cells "
        "get a least-uphill code and can form two-cell cycles (addendum 15, section 3). Fix: pad the grid with at "
        "least one row/column of NaN on every side so that no valid cell touches the border, or pass "
        "border='outlets' (every data-edge cell, border included, becomes an outlet with code 0; the second-to-last "
        "row/column keep the quirk unless the grid is padded), or pass outlets=<mask> covering every valid border "
        "cell, or border='legacy' to keep the TopoAnalysis behaviour knowingly (the v1/v2 model-surface treatment).")


def d8_flow_direction(z_filled: np.ndarray, border: str | None = None, outlets: np.ndarray | None = None,
                      name: str = "grid") -> np.ndarray:
    """Exact port of ``FlowDirectionD8._create_from_flooded_dem``, with the border guard.

    Parameters
    ----------
    z_filled
        2-D filled elevation (float). NaN cells get code 0.
    border
        Treatment of a grid whose valid cells touch the grid border (see the module docstring):
        ``None`` (default) refuses such a grid with :class:`D8BorderError`; ``"legacy"`` keeps the
        TopoAnalysis codes everywhere (v1/v2 behaviour); ``"outlets"`` sets code 0 on every data-edge cell
        (finite cells bordering NaN or the grid edge), the convention of the full-grid routing of addenda 14-15.
    outlets
        Optional boolean mask of designated outlet cells (code 0). Counts as the border designation when it
        covers every valid border cell.
    name
        Name of the grid, used in the error message.

    Returns
    -------
    np.ndarray of uint8
        D8 codes.
    """
    check_border(z_filled, border, outlets, name)
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
    if border == "outlets":
        fd[data_edge_cells(Z)] = 0
    if outlets is not None:
        fd[np.asarray(outlets, dtype=bool)] = 0
    return fd


def d8_area_budget(fd: np.ndarray, area: np.ndarray, z: np.ndarray, dx: float) -> dict[str, float | int | bool]:
    """Closed-budget check: the area summed over the root cells (valid cells with no valid receiver) must equal
    the number of valid cells times ``dx**2``; also counts receiver pairs that form two-cell cycles."""
    valid = np.isfinite(np.asarray(z, dtype=np.float64))
    ny, nx = valid.shape
    r_to, c_to, good = d8_receiver_arrays(fd)
    good = good & valid
    good[good] &= valid[r_to[good], c_to[good]]
    roots = valid & ~good
    flat = np.arange(ny * nx).reshape(ny, nx)
    tgt = np.full((ny, nx), -1, dtype=np.int64)
    tgt[good] = flat[r_to[good], c_to[good]]
    back = np.full(ny * nx, -1, dtype=np.int64)
    back[good.ravel()] = tgt.ravel()[tgt.ravel()[good.ravel()]]
    two_cycles = int(np.sum(back == flat.ravel()) // 2)
    total = float(np.sum(np.asarray(area, dtype=np.float64)[roots]))
    expected = float(valid.sum()) * float(dx) ** 2
    return {"n_valid": int(valid.sum()), "n_roots": int(roots.sum()), "root_area_sum": total, "expected": expected,
            "closed": bool(abs(total - expected) <= 1e-6 * max(expected, 1.0)), "two_cell_cycles": two_cycles}


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
    target = np.where(good.ravel(), r_to.ravel() * nx + c_to.ravel(), -1)
    _accumulate(order, target, area.ravel())           # in place on the view
    return area


def _accumulate_py(order, target, area_f):
    """Reference loop: pass accumulated area to the receiver, highest cell first (Phase 3 step 5 keeps this order)."""
    for k in order:
        t = target[k]
        if t >= 0:
            area_f[t] += area_f[k]
    return area_f


try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _accumulate_nb(order, target, area_f):  # pragma: no cover (compiled)
        for k in order:
            t = target[k]
            if t >= 0:
                area_f[t] += area_f[k]
        return area_f

    _accumulate: Any = _accumulate_nb
except Exception:  # noqa: BLE001
    _accumulate = _accumulate_py


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
