"""Small utilities ported from v1 ``src/utils.py`` (slope magnitude, NaN-aware resampling)."""

from __future__ import annotations

import numpy as np
from rasterio.transform import Affine
from scipy.ndimage import zoom

from lemcvt.io.grids import Grid


def compute_slope_magnitude(Z: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """``sqrt(gx^2 + gy^2)`` from ``np.gradient`` (unit pixel spacing, as in v1)."""
    gy, gx = np.gradient(Z)
    S = np.sqrt(gx**2 + gy**2)
    if mask is not None:
        S[~mask] = np.nan
    return S


def nanaware_zoom(grid: np.ndarray, zoom_yx: tuple[float, float], order: int = 1) -> np.ndarray:
    """Resample with NaN tracking (v1 ``_nanaware_zoom``)."""
    grid = np.asarray(grid, dtype=float)
    w = np.isfinite(grid).astype(float)
    v = np.where(np.isfinite(grid), grid, 0.0)

    v2 = zoom(v, zoom=zoom_yx, order=order, mode="nearest", prefilter=(order > 1))
    w2 = zoom(w, zoom=zoom_yx, order=1, mode="nearest")

    out = v2 / np.where(w2 > 0, w2, np.nan)
    out[w2 < 0.5] = np.nan
    return out


def resample_from_1m(grid: Grid, dx_target: float, order: int = 1) -> Grid:
    """Downsample a grid to ``dx_target`` (v1 ``resample_from_1m``; zoom = dx/dx_target)."""
    old_dx = grid.dx if np.isfinite(grid.dx) and grid.dx > 0 else 1.0
    old_dy = grid.dy if np.isfinite(grid.dy) and grid.dy > 0 else old_dx
    zx = old_dx / float(dx_target)
    zy = old_dy / float(dx_target)
    new_data = nanaware_zoom(grid.data, (zy, zx), order=order)
    t = grid.transform
    new_t = Affine(float(dx_target), t.b, t.c, t.d, -float(dx_target), t.f)
    return Grid(data=new_data, transform=new_t, crs=grid.crs, nodata=grid.nodata,
                name=f"{grid.name}_dx{dx_target:g}", meta=dict(grid.meta, resampled_from_dx=old_dx))
