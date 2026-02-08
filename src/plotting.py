"""
Plotting utilities for landscape evolution model output.

Provides comparison plots (observed vs. modelled elevation, slope-area
relationships, channel masks) and a Nature-style matplotlib configuration
helper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Lazy import: TopoAnalysis may not be available in test environments.
# Functions that need it import it at call time.
import TopoAnalysis.dem as d


# ---------------------------------------------------------------------------
# Style helper
# ---------------------------------------------------------------------------

def set_nature_style() -> None:
    """Apply Nature-style matplotlib defaults (300 dpi, Helvetica, 8 pt).

    Configures ``plt.rcParams`` for publication-quality figures following
    Nature's formatting guidelines.  Safe to call multiple times.
    """
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# ---------------------------------------------------------------------------
# Shared figure setup
# ---------------------------------------------------------------------------

def _setup_elevation_panels(
    dem_griddata: np.ndarray,
    Z_model: np.ndarray,
    n_right_panels: int = 1,
    figsize: tuple[float, float] = (18, 5),
    right_ratio: float = 4.0,
) -> tuple[plt.Figure, list[plt.Axes], plt.cm.ScalarMappable]:
    """Create a figure with two elevation panels, a colorbar, and extra panels.

    Parameters
    ----------
    dem_griddata : np.ndarray
        Observed elevation grid.
    Z_model : np.ndarray
        Modelled elevation grid.
    n_right_panels : int
        Number of panels to the right of the colorbar (default 1).
    figsize : tuple[float, float]
        Figure size in inches.
    right_ratio : float
        Width ratio for right-side panels relative to the elevation panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of Axes
        ``[ax_orig, ax_model, ax_cbar, *ax_right]``
    im : ScalarMappable
        The image object from the modelled elevation panel (for colorbar).
    """
    ratios = [3, 3, 0.25] + [right_ratio] * n_right_panels
    fig, axes = plt.subplots(
        1, 3 + n_right_panels, figsize=figsize,
        gridspec_kw={"width_ratios": ratios},
    )

    vmin = np.nanmin(dem_griddata)
    vmax = np.nanmax(dem_griddata)

    axes[0].imshow(dem_griddata, cmap="terrain", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Elevation")
    axes[0].axis("off")

    im = axes[1].imshow(Z_model, cmap="terrain", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Modeled Elevation")
    axes[1].axis("off")

    fig.colorbar(im, cax=axes[2], orientation="vertical")

    return fig, list(axes), im


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_maps_area_slope(
    dem: Any,
    area: Any,
    flow_direction: Any,
    Z_model: np.ndarray,
    channel_mask: np.ndarray | None = None,
    vertical_interval: int = 10,
    min_area: float = 2000,
    figsize: tuple[float, float] = (18, 5),
) -> None:
    """Plot observed and modelled elevation with a slope-area comparison.

    Creates a 4-panel figure:

    1. Original observed elevation
    2. Modelled elevation (same colour scale)
    3. Shared colour bar
    4. Log-log slope vs. area with best-fit lines

    Parameters
    ----------
    dem : Elevation-like object
        Observed DEM with ``._griddata`` and ``._georef_info``.
    area : Area-like object
        Drainage area raster.
    flow_direction : FlowDirectionD8-like object
        D8 flow direction raster.
    Z_model : np.ndarray
        2-D modelled elevation array (same shape as ``dem._griddata``).
    channel_mask : np.ndarray or None, optional
        If given, restricts slope-area points to these cells.
    vertical_interval : int, optional
        Vertical interval for slope smoothing (default 10 m).
    min_area : float, optional
        Minimum drainage area for slope calculation (default 2000 m²).
    figsize : tuple, optional
        Figure size (default ``(18, 5)``).
    """
    # Build a model Elevation object for slope computation
    Z_filled = Z_model.copy()
    elev_model = d.Elevation()
    elev_model._griddata = Z_filled
    elev_model._georef_info = dem._georef_info
    elev_model._nodata_value = np.nan

    slope_orig_obj = d.ChannelSlopeWithSmoothing(
        elevation=dem,
        area=area,
        flow_direction=flow_direction,
        vertical_interval=vertical_interval,
        min_area=min_area,
    )
    slope_model_obj = d.ChannelSlopeWithSmoothing(
        elevation=elev_model,
        area=area,
        flow_direction=flow_direction,
        vertical_interval=vertical_interval,
        min_area=min_area,
    )

    S_obs = slope_orig_obj._griddata
    S_mod = slope_model_obj._griddata
    A_orig = area._griddata

    if channel_mask is not None:
        mask = channel_mask & np.isfinite(S_obs) & np.isfinite(S_mod) & np.isfinite(A_orig)
    else:
        mask = np.isfinite(S_obs) & np.isfinite(S_mod) & np.isfinite(A_orig)

    A_masked = A_orig[mask]
    S1 = S_obs[mask]
    S2 = S_mod[mask]

    logA = np.log10(A_masked)
    logS1 = np.log10(S1)
    logS2 = np.log10(S2)

    slope1, intercept1, *_ = linregress(logA, logS1)
    slope2, intercept2, *_ = linregress(logA, logS2)
    ks1 = 10 ** intercept1
    ks2 = 10 ** intercept2

    fig, axes, im = _setup_elevation_panels(
        dem._griddata, Z_model, n_right_panels=1, figsize=figsize, right_ratio=4.0,
    )

    A_fit = np.logspace(np.log10(A_masked.min()), np.log10(A_masked.max()), 200)
    S_fit1 = ks1 * A_fit ** slope1
    S_fit2 = ks2 * A_fit ** slope2

    ax = axes[3]
    ax.loglog(A_masked, S1, ".", alpha=0.3, label="Original")
    ax.loglog(A_masked, S2, ".", alpha=0.3, label="Modeled")
    ax.loglog(A_fit, S_fit1, "k--", lw=2,
              label=f"Original fit: k_s={ks1:.2e}, theta={slope1:.2f}")
    ax.loglog(A_fit, S_fit2, "r--", lw=2,
              label=f"Modeled fit: k_s={ks2:.2e}, theta={slope2:.2f}")
    ax.set_xlabel("Drainage Area A (m\u00b2)")
    ax.set_ylabel("Slope S")
    ax.set_title("Slope\u2013Area Comparison")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()


def plot_maps_and_channel_mask(
    dem: Any,
    Z_model: np.ndarray,
    mask: np.ndarray,
    figsize: tuple[float, float] = (18, 5),
) -> None:
    """Plot observed/modelled elevation and a channel mask.

    Creates a 4-panel figure:

    1. Original observed elevation
    2. Modelled elevation (same colour scale)
    3. Shared colour bar
    4. Channel mask (binary, seismic colour map)

    Parameters
    ----------
    dem : Elevation-like object
        Observed DEM with ``._griddata``.
    Z_model : np.ndarray
        2-D modelled elevation array.
    mask : np.ndarray
        2-D boolean channel mask.
    figsize : tuple, optional
        Figure size (default ``(18, 5)``).
    """
    fig, axes, im = _setup_elevation_panels(
        dem._griddata, Z_model, n_right_panels=1, figsize=figsize, right_ratio=3.0,
    )

    axes[3].imshow(mask, origin="lower", vmin=0, vmax=1, cmap="seismic")
    axes[3].set_title("Channel Mask")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
