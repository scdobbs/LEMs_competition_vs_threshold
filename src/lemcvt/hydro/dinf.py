"""D-infinity (Tarboton 1997) flow routing. Verbatim port of v1 ``src/dinf.py``.

Computes continuous flow angles from 8 triangular facets and distributes
drainage area to two receivers proportionally, eliminating D8's 45-degree
quantization.
"""

from __future__ import annotations

import numpy as np

# Each facet is (e1_di, e1_dj, e2_di, e2_dj, base_angle) where e1 is the cardinal
# neighbour and e2 the diagonal. Offsets are (di_col, dj_row), i.e. Z[j, i].
FACETS = [
    (1, 0, 1, -1, 0.0),            # facet 0: E -> NE
    (0, -1, 1, -1, np.pi / 4),     # facet 1: N -> NE
    (0, -1, -1, -1, np.pi / 2),    # facet 2: N -> NW
    (-1, 0, -1, -1, 3 * np.pi / 4),  # facet 3: W -> NW
    (-1, 0, -1, 1, np.pi),         # facet 4: W -> SW
    (0, 1, -1, 1, 5 * np.pi / 4),  # facet 5: S -> SW
    (0, 1, 1, 1, 3 * np.pi / 2),   # facet 6: S -> SE
    (1, 0, 1, 1, 7 * np.pi / 4),   # facet 7: E -> SE
]

D8_CODE_TO_DINF: dict[int, tuple[float, int, float]] = {
    1: (0.0, 0, 0.0),
    128: (np.pi / 4, 0, np.pi / 4),
    64: (np.pi / 2, 2, 0.0),
    32: (3 * np.pi / 4, 2, np.pi / 4),
    16: (np.pi, 4, 0.0),
    8: (5 * np.pi / 4, 4, np.pi / 4),
    4: (3 * np.pi / 2, 6, 0.0),
    2: (7 * np.pi / 4, 6, np.pi / 4),
}


def compute_dinf_flow_direction(
    Z: np.ndarray,
    dx: float,
    FD_d8: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute D-infinity flow direction from an elevation grid (v1 algorithm).

    Returns ``(angles, r_facet, facet_idx)``; NaN / -1 for invalid cells. With
    ``FD_d8`` given, flat cells fall back to the D8 direction.
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

            for f, (e1_di, e1_dj, e2_di, e2_dj, _base) in enumerate(FACETS):
                i1, j1 = i + e1_di, j + e1_dj
                if not (0 <= i1 < Nx and 0 <= j1 < Ny):
                    continue
                z1 = Z[j1, i1]
                if not np.isfinite(z1):
                    continue

                i2, j2 = i + e2_di, j + e2_dj
                if not (0 <= i2 < Nx and 0 <= j2 < Ny):
                    continue
                z2 = Z[j2, i2]
                if not np.isfinite(z2):
                    continue

                s1 = (z0 - z1) / dx
                s2 = (z1 - z2) / dx

                r = np.arctan2(s2, s1)
                if r < 0.0:
                    r = 0.0
                elif r > pi_over_4:
                    r = pi_over_4

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
                code = int(FD_d8[j, i])
                if code in D8_CODE_TO_DINF:
                    fallback_angle, fallback_facet, fallback_r = D8_CODE_TO_DINF[code]
                    angles[j, i] = fallback_angle
                    r_facet[j, i] = fallback_r
                    facet_idx[j, i] = fallback_facet

    return angles, r_facet, facet_idx


def get_receivers(j: int, i: int, fac_idx: int, r: float) -> tuple[int, int, int, int, float, float]:
    """Two receiver cells and proportional weights for a D-inf cell."""
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
    """D-infinity drainage area by proportional distribution, highest cell first."""
    Ny, Nx = Z.shape
    cell_area = dx * dx

    A = np.full((Ny, Nx), np.nan)
    valid = np.isfinite(Z) & np.isfinite(angles)
    A[valid] = cell_area

    valid_coords = np.argwhere(valid)
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
        if 0 <= j1 < Ny and 0 <= i1 < Nx and np.isfinite(A[j1, i1]):
            A[j1, i1] += p1 * area_here
        if 0 <= j2 < Ny and 0 <= i2 < Nx and np.isfinite(A[j2, i2]):
            A[j2, i2] += p2 * area_here

    return A


def compute_dinf(Z: np.ndarray, dx: float, FD_d8: np.ndarray | None = None):
    """Convenience wrapper: ``(angles, r_facet, facet_idx, A_dinf)`` from arrays."""
    angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx, FD_d8=FD_d8)
    A_dinf = compute_dinf_area(Z, dx, angles, r_facet, facet_idx)
    return angles, r_facet, facet_idx, A_dinf
