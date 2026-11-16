"""Priority-flood depression filling, an exact port of TopoAnalysis ``_flood``.

Barnes et al. (2014) priority flood with a tiny aggradation slope
(``1e-12 * dx * distance_multiplier``) so that filled flats drain. The edge set,
the neighbour visiting order and the stable (counter-tied) heap are those of
TopoAnalysis, so results are identical cell by cell.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

AGGRADATION_SLOPE = 1e-12

# TopoAnalysis _getNeighborIndices order and unit distances
_ROW_KERNEL = np.array([1, 1, 1, 0, 0, -1, -1, -1])
_COL_KERNEL = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
_RT2 = float(np.sqrt(2))
_DX_MULTS = np.array([_RT2, 1.0, _RT2, 1.0, 1.0, _RT2, 1.0, _RT2])


def dem_edge_cells(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``Elevation.findDEMedge``: cells bordering NaN/zero cells or the grid edge."""
    original = np.zeros(z.shape, dtype=np.float64)
    original[np.isnan(z) | (z == 0)] = 1
    structure = generate_binary_structure(2, 2)
    dilated = binary_dilation(original, structure=structure, border_value=1).astype(int)
    edges = dilated - original
    rows, cols = np.nonzero(edges)
    return rows, cols


def priority_flood_fill(z: np.ndarray, dx: float) -> np.ndarray:
    """Return a filled copy of ``z`` (TopoAnalysis ``FilledElevation(elevation=...)``).

    Parameters
    ----------
    z
        2-D elevation; NaN cells are left untouched and never filled.
    dx
        Pixel size; enters only through the aggradation increment.
    """
    Z = np.array(z, dtype=np.float64, copy=True)
    ny, nx = Z.shape
    closed = np.zeros_like(Z, dtype=bool)
    closed[np.isnan(Z)] = True

    edge_rows, edge_cols = dem_edge_cells(Z)

    heap: list[tuple[float, int, int, int]] = []
    counter = 0
    for r, c in zip(edge_rows.tolist(), edge_cols.tolist(), strict=True):
        closed[r, c] = True
        counter += 1
        heapq.heappush(heap, (float(Z[r, c]), counter, int(r), int(c)))

    while heap:
        _priority, _cnt, row, col = heapq.heappop(heap)
        elevation = Z[row, col]
        out_rows = _ROW_KERNEL + row
        out_cols = _COL_KERNEL + col
        in_bounds = (out_rows >= 0) & (out_rows < ny) & (out_cols >= 0) & (out_cols < nx)
        for nr, nc, mult in zip(out_rows[in_bounds].tolist(), out_cols[in_bounds].tolist(),
                                _DX_MULTS[in_bounds].tolist(), strict=True):
            if not closed[nr, nc]:
                if Z[nr, nc] <= elevation:
                    Z[nr, nc] = elevation + AGGRADATION_SLOPE * dx * mult
                closed[nr, nc] = True
                counter += 1
                heapq.heappush(heap, (float(Z[nr, nc]), counter, int(nr), int(nc)))
    return Z
