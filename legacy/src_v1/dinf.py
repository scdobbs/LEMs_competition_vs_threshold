"""
D-infinity (Tarboton 1997) flow routing.

Computes continuous flow angles from 8 triangular facets and distributes
drainage area to two receivers proportionally, eliminating D8's 45-degree
quantization.

Functions
---------
- :func:`compute_dinf_flow_direction` — flow angle, within-facet angle, facet index
- :func:`get_receivers` — receiver coordinates and proportional weights
- :func:`compute_dinf_area` — proportional area accumulation
- :func:`compute_dinf_from_dem` — convenience wrapper for raster objects
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Facet definitions (Tarboton 1997, counterclockwise from east)
# ---------------------------------------------------------------------------
# Each facet is (e1_di, e1_dj, e2_di, e2_dj, base_angle)
# where e1 is the cardinal neighbor and e2 is the diagonal neighbor.
# Offsets are (di_col, dj_row) matching the models.py convention (Z[j, i]).

FACETS = [
    # facet 0: E → NE
    (1, 0, 1, -1, 0.0),
    # facet 1: N → NE
    (0, -1, 1, -1, np.pi / 4),
    # facet 2: N → NW
    (0, -1, -1, -1, np.pi / 2),
    # facet 3: W → NW
    (-1, 0, -1, -1, 3 * np.pi / 4),
    # facet 4: W → SW
    (-1, 0, -1, 1, np.pi),
    # facet 5: S → SW
    (0, 1, -1, 1, 5 * np.pi / 4),
    # facet 6: S → SE
    (0, 1, 1, 1, 3 * np.pi / 2),
    # facet 7: E → SE
    (1, 0, 1, 1, 7 * np.pi / 4),
]

# ---------------------------------------------------------------------------
# D8 code → D-inf fallback mapping
# ---------------------------------------------------------------------------
# For flat cells where no facet has positive slope, we fall back to the D8
# direction. Each D8 code maps to (angle, facet_idx, r_facet).

D8_CODE_TO_DINF: dict[int, tuple[float, int, float]] = {
    1:   (0.0,             0, 0.0),         # E  → facet 0, r=0
    128: (np.pi / 4,       0, np.pi / 4),   # NE → facet 0, r=π/4
    64:  (np.pi / 2,       2, 0.0),         # N  → facet 2, r=0
    32:  (3 * np.pi / 4,   2, np.pi / 4),   # NW → facet 2, r=π/4
    16:  (np.pi,           4, 0.0),         # W  → facet 4, r=0
    8:   (5 * np.pi / 4,   4, np.pi / 4),   # SW → facet 4, r=π/4
    4:   (3 * np.pi / 2,   6, 0.0),         # S  → facet 6, r=0
    2:   (7 * np.pi / 4,   6, np.pi / 4),   # SE → facet 6, r=π/4
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_dinf_flow_direction(
    Z: np.ndarray,
    dx: float,
    FD_d8: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute D-infinity flow direction from an elevation grid.

    For each interior cell, evaluates all 8 triangular facets and selects
    the one with maximum downhill slope. The within-facet angle *r* determines
    the proportional split between two receivers.

    Parameters
    ----------
    Z : np.ndarray
        2-D elevation array (Ny × Nx). NaN marks invalid cells.
    dx : float
        Cell spacing (assumed square: dx = dy).
    FD_d8 : np.ndarray or None
        Optional D8 flow-direction grid (same shape as *Z*) used as fallback
        for flat cells where no facet has positive slope. When provided,
        guarantees no NaN angles for valid cells in filled DEMs.

    Returns
    -------
    angles : np.ndarray
        2-D array of absolute flow angles in [0, 2π). NaN for invalid cells.
    r_facet : np.ndarray
        2-D array of within-facet angles in [0, π/4]. NaN for invalid cells.
    facet_idx : np.ndarray
        2-D integer array of selected facet indices (0–7). -1 for invalid.
    """
    Ny, Nx = Z.shape
    angles = np.full((Ny, Nx), np.nan)
    r_facet = np.full((Ny, Nx), np.nan)
    facet_idx = np.full((Ny, Nx), -1, dtype=int)

    pi_over_4 = np.pi / 4.0

    for j in range(Ny):
        for i in range(Nx):
            z0 = Z[j, i]
            if not np.isfinite(z0):
                continue

            best_slope = 0.0
            best_r = np.nan
            best_facet = -1

            for f, (e1_di, e1_dj, e2_di, e2_dj, base) in enumerate(FACETS):
                # e1 neighbor (cardinal)
                i1, j1 = i + e1_di, j + e1_dj
                if not (0 <= i1 < Nx and 0 <= j1 < Ny):
                    continue
                z1 = Z[j1, i1]
                if not np.isfinite(z1):
                    continue

                # e2 neighbor (diagonal)
                i2, j2 = i + e2_di, j + e2_dj
                if not (0 <= i2 < Nx and 0 <= j2 < Ny):
                    continue
                z2 = Z[j2, i2]
                if not np.isfinite(z2):
                    continue

                # Slopes within the facet
                # d1 = dx (cardinal distance), d2 = dx (diagonal cross-distance)
                s1 = (z0 - z1) / dx
                s2 = (z1 - z2) / dx

                # Within-facet angle
                r = np.arctan2(s2, s1)

                # Clamp to [0, π/4]
                if r < 0.0:
                    r = 0.0
                elif r > pi_over_4:
                    r = pi_over_4

                # Slope magnitude after clamping
                slope_mag = np.sqrt(
                    ((z0 - z1) / dx * np.cos(r) + (z1 - z2) / dx * np.sin(r)) ** 2
                    + ((z1 - z2) / dx * np.cos(r) - (z0 - z1) / dx * np.sin(r)) ** 2
                )
                # Simpler: slope in the steepest direction within the facet
                # S = s1*cos(r) + s2*sin(r) after clamping
                slope_dir = s1 * np.cos(r) + s2 * np.sin(r)

                if slope_dir > best_slope:
                    best_slope = slope_dir
                    best_r = r
                    best_facet = f

            if best_facet >= 0:
                angles[j, i] = FACETS[best_facet][4] + best_r
                r_facet[j, i] = best_r
                facet_idx[j, i] = best_facet
            elif FD_d8 is not None:
                # Flat cell fallback: use D8 direction
                code = int(FD_d8[j, i])
                if code in D8_CODE_TO_DINF:
                    fallback_angle, fallback_facet, fallback_r = D8_CODE_TO_DINF[code]
                    angles[j, i] = fallback_angle
                    r_facet[j, i] = fallback_r
                    facet_idx[j, i] = fallback_facet

    return angles, r_facet, facet_idx


def get_receivers(
    j: int,
    i: int,
    fac_idx: int,
    r: float,
) -> tuple[int, int, int, int, float, float]:
    """Get the two receiver cells and proportional weights for a D-inf cell.

    Parameters
    ----------
    j, i : int
        Row and column of the source cell.
    fac_idx : int
        Facet index (0–7) from :func:`compute_dinf_flow_direction`.
    r : float
        Within-facet angle in [0, π/4].

    Returns
    -------
    j1, i1 : int
        Row, column of the cardinal (e1) receiver.
    j2, i2 : int
        Row, column of the diagonal (e2) receiver.
    p1, p2 : float
        Proportional weights (p1 + p2 = 1).
    """
    e1_di, e1_dj, e2_di, e2_dj, _ = FACETS[fac_idx]
    j1 = j + e1_dj
    i1 = i + e1_di
    j2 = j + e2_dj
    i2 = i + e2_di

    pi_over_4 = np.pi / 4.0
    p1 = (pi_over_4 - r) / pi_over_4
    p2 = r / pi_over_4

    return j1, i1, j2, i2, p1, p2


def compute_dinf_area(
    Z: np.ndarray,
    dx: float,
    angles: np.ndarray,
    r_facet: np.ndarray,
    facet_idx: np.ndarray,
) -> np.ndarray:
    """Compute D-infinity drainage area by proportional distribution.

    Processes cells from highest to lowest elevation, distributing each
    cell's accumulated area to its two receivers proportionally.

    Parameters
    ----------
    Z : np.ndarray
        2-D elevation array.
    dx : float
        Cell spacing.
    angles : np.ndarray
        Flow angles from :func:`compute_dinf_flow_direction`.
    r_facet : np.ndarray
        Within-facet angles from :func:`compute_dinf_flow_direction`.
    facet_idx : np.ndarray
        Facet indices from :func:`compute_dinf_flow_direction`.

    Returns
    -------
    A : np.ndarray
        2-D drainage area array (same shape as *Z*).
    """
    Ny, Nx = Z.shape
    cell_area = dx * dx

    A = np.full((Ny, Nx), np.nan)

    # Initialize valid cells with one cell's worth of area
    valid = np.isfinite(Z) & np.isfinite(angles)
    A[valid] = cell_area

    # Sort valid cells by descending elevation
    valid_coords = np.argwhere(valid)  # (N, 2) array of [j, i]
    elevations = Z[valid_coords[:, 0], valid_coords[:, 1]]
    sort_order = np.argsort(-elevations)
    sorted_coords = valid_coords[sort_order]

    for idx in range(len(sorted_coords)):
        j, i = sorted_coords[idx]
        fac = facet_idx[j, i]
        if fac < 0:
            continue

        r = r_facet[j, i]
        j1, i1, j2, i2, p1, p2 = get_receivers(j, i, fac, r)

        area_here = A[j, i]

        # Distribute to cardinal receiver
        if 0 <= j1 < Ny and 0 <= i1 < Nx and np.isfinite(A[j1, i1]):
            A[j1, i1] += p1 * area_here

        # Distribute to diagonal receiver
        if 0 <= j2 < Ny and 0 <= i2 < Nx and np.isfinite(A[j2, i2]):
            A[j2, i2] += p2 * area_here

    return A


def compute_dinf_from_dem(
    dem_obj,
    fd_obj=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: compute D-inf routing from raster objects.

    Parameters
    ----------
    dem_obj : Elevation-like object
        Must expose ``._griddata`` and ``._georef_info.dx``.
    fd_obj : FlowDirectionD8-like object or None
        If provided, its ``._griddata`` is used as D8 fallback for flat cells.

    Returns
    -------
    angles, r_facet, facet_idx, A_dinf : np.ndarray
        Flow angles, within-facet angles, facet indices, and drainage area.
    """
    Z = dem_obj._griddata
    dx = dem_obj._georef_info.dx
    FD_d8 = fd_obj._griddata if fd_obj is not None else None

    angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx, FD_d8=FD_d8)
    A_dinf = compute_dinf_area(Z, dx, angles, r_facet, facet_idx)

    return angles, r_facet, facet_idx, A_dinf
