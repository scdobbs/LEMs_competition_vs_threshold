"""
General-purpose utilities for the landscape evolution modelling workflow.

Provides functions for data I/O, raster cropping, slope computation, and
portable project-root discovery.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def find_project_root(
    start: str | Path | None = None,
    marker_dir: str = "data",
) -> Path:
    """Walk upward to find the repository root containing *marker_dir*.

    Parameters
    ----------
    start : str or Path or None
        Starting directory.  Defaults to the current working directory.
    marker_dir : str
        Directory name whose presence signals the project root
        (default ``"data"``).

    Returns
    -------
    Path
        Absolute path to the project root.

    Raises
    ------
    FileNotFoundError
        If *marker_dir* cannot be found in any ancestor directory.
    """
    start_path = Path(start or Path.cwd()).resolve()
    for parent in [start_path, *start_path.parents]:
        if (parent / marker_dir).is_dir():
            return parent
    raise FileNotFoundError(
        f"Could not find '{marker_dir}/' by searching upward from:\n"
        f"  {start_path}\n"
        f"Make sure this notebook is somewhere inside the repository folder tree."
    )


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------

def load_tile_data(
    base_path: str | Path,
    region: str,
) -> tuple[Any, Any, Any]:
    """Load DEM, area, and flow-direction rasters from a standard tile path.

    Expects filenames of the form ``<region>_1m_best_tile_{filled,area,fd}``
    inside *base_path*.

    Parameters
    ----------
    base_path : str or Path
        Directory containing the tile rasters (e.g. ``data/AP/tile1/``).
    region : str
        Region code (``"AP"`` or ``"GM"``).

    Returns
    -------
    dem, area, flow_direction
        Loaded TopoAnalysis objects.
    """
    import TopoAnalysis.dem as d

    base = Path(base_path)
    prefix = f"{region}_1m_best_tile"

    dem = d.Elevation.load(str(base / f"{prefix}_filled"))
    area_obj = d.Area.load(str(base / f"{prefix}_area"))
    fd = d.FlowDirectionD8.load(str(base / f"{prefix}_fd"))

    return dem, area_obj, fd


def save_model_outputs(
    dem: Any,
    Z_model: np.ndarray,
    save_dir: str | Path,
    prefix: str,
    inclination: float = 20.0,
    azimuth: float = 315.0,
) -> str:
    """Save modelled elevation and derived grids.

    Writes elevation, filled DEM, flow direction, drainage area, log-area,
    and hillshade rasters to *save_dir* using filenames starting with
    *prefix*.

    Parameters
    ----------
    dem : Elevation-like object
        Template DEM whose metadata (georef, nodata) is reused.
    Z_model : np.ndarray
        2-D modelled elevation array.
    save_dir : str or Path
        Output directory.
    prefix : str
        Filename prefix for all output rasters.
    inclination : float
        Hillshade inclination angle (default 20).
    azimuth : float
        Hillshade azimuth angle (default 315).

    Returns
    -------
    str
        The *prefix* string (for downstream logging).
    """
    import TopoAnalysis.dem as d

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    full_prefix = str(save_dir / prefix)

    dem_opt = copy.copy(dem)
    dem_opt._griddata = Z_model.astype(float)
    dem_opt.save(full_prefix + "_elevation")

    filled = d.FilledElevation(elevation=dem_opt)
    filled.save(full_prefix + "_filled")

    fd = d.FlowDirectionD8(flooded_dem=filled)
    fd.save(full_prefix + "_flow_direction")

    area_opt = d.Area(flow_direction=fd)
    area_opt.save(full_prefix + "_area")

    logarea = d.LogArea(area=area_opt)
    logarea.save(full_prefix + "_logarea")

    hillshade = d.Hillshade(
        elevation=dem_opt, inclination=inclination, azimuth=azimuth,
    )
    hillshade.save(full_prefix + "_hillshade")

    logger.info("Saved model outputs with prefix '%s'", full_prefix)
    return prefix


# ---------------------------------------------------------------------------
# Raster processing
# ---------------------------------------------------------------------------

def crop(
    dem: Any,
    area: Any,
    flow_direction: Any,
    extent_indexes: tuple[int, int, int, int] = (200, 800, 200, 800),
) -> tuple[Any, Any, Any]:
    """Crop DEM, area, and flow-direction rasters to a specified extent.

    Parameters
    ----------
    dem : Elevation-like object
        Must expose ``._rowscols_to_xy`` and ``.clip_to_extent``.
    area : Area-like object
    flow_direction : FlowDirectionD8-like object
    extent_indexes : tuple of int
        ``(row_min, row_max, col_min, col_max)`` pixel indices
        (default ``(200, 800, 200, 800)``).

    Returns
    -------
    dem_clip, area_clip, fd_clip
        Clipped raster objects.
    """
    rowscols_minmin = [extent_indexes[0], extent_indexes[2]]
    rowscols_maxmax = [extent_indexes[1], extent_indexes[3]]

    extent_minmin = dem._rowscols_to_xy((rowscols_minmin,))[0]
    extent_maxmax = dem._rowscols_to_xy((rowscols_maxmax,))[0]

    extent = (
        extent_minmin[0], extent_maxmax[0],
        extent_maxmax[1], extent_minmin[1],
    )

    dem_clip = dem.clip_to_extent(extent)
    area_clip = area.clip_to_extent(extent)
    fd_clip = flow_direction.clip_to_extent(extent)

    return dem_clip, area_clip, fd_clip


# ---------------------------------------------------------------------------
# Slope computation
# ---------------------------------------------------------------------------

def compute_slope_magnitude(
    Z: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute slope magnitude from an elevation grid.

    Uses ``np.gradient`` (central differences) to compute the gradient,
    then returns ``sqrt(dz/dx² + dz/dy²)``.

    Parameters
    ----------
    Z : np.ndarray
        2-D elevation grid.
    mask : np.ndarray or None, optional
        If given, sets slope to NaN outside the mask.

    Returns
    -------
    np.ndarray
        2-D slope magnitude array (same shape as *Z*).
    """
    gy, gx = np.gradient(Z)
    S = np.sqrt(gx ** 2 + gy ** 2)

    if mask is not None:
        S[~mask] = np.nan

    return S
