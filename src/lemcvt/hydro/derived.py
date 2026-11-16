"""Derived byte products saved alongside model output (hillshade, log-area).

These reproduce TopoAnalysis ``Hillshade`` (Esri formula with the
``assignBCs`` padding, including its corner quirk) and ``LogArea``. v1 wrote them
through GDAL as Byte, which rounds half-up and clamps to [0, 255]
(``GDALCopyWords``); :func:`to_byte_like_gdal` applies the same conversion.
"""

from __future__ import annotations

import numpy as np


def assign_bcs(grid: np.ndarray) -> np.ndarray:
    """Pad by one cell replicating the margins (TopoAnalysis ``assignBCs``, corner quirk kept)."""
    ny, nx = grid.shape
    Zbc = np.zeros((ny + 2, nx + 2))
    Zbc[1:-1, 1:-1] = grid
    Zbc[0, 1:-1] = grid[0, :]
    Zbc[-1, 1:-1] = grid[-1, :]
    Zbc[1:-1, 0] = grid[:, 0]
    Zbc[1:-1, -1] = grid[:, -1]
    Zbc[0, 0] = grid[0, 0]
    Zbc[0, -1] = grid[0, -1]
    Zbc[-1, 0] = grid[-1, 0]
    Zbc[-1, -1] = grid[-1, 0]  # sic (TopoAnalysis)
    return Zbc


def finite_slopes(grid: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Centered-difference Sx, Sy with replicated boundaries (``_calcFiniteSlopes``)."""
    Zbc = assign_bcs(grid)
    Sx = (Zbc[1:-1, 2:] - Zbc[1:-1, :-2]) / (2 * dx)
    Sy = (Zbc[2:, 1:-1] - Zbc[:-2, 1:-1]) / (2 * dx)
    return Sx, Sy


def hillshade(z: np.ndarray, dx: float, azimuth: float = 315.0, inclination: float = 20.0) -> np.ndarray:
    """Esri hillshade in float (0..255 scale), as TopoAnalysis ``calcHillshade``."""
    az_rad = (360 - azimuth + 90) * np.pi / 180
    elev_rad = (90 - inclination) * np.pi / 180
    Sx, Sy = finite_slopes(z, dx)
    aspect = np.arctan2(Sy, Sx)
    smag = np.arctan(np.sqrt(Sx**2 + Sy**2))
    return 255.0 * ((np.cos(elev_rad) * np.cos(smag)) + (np.sin(elev_rad) * np.sin(smag) * np.cos(az_rad - aspect)))


def log_area(area: np.ndarray) -> np.ndarray:
    return np.log10(area)


def to_byte_like_gdal(a: np.ndarray) -> np.ndarray:
    """Float -> uint8 as GDAL's Byte writer does it: round half up, clamp to [0, 255], NaN -> 0."""
    out = np.floor(np.nan_to_num(a, nan=0.0) + 0.5)
    return np.clip(out, 0, 255).astype(np.uint8)
