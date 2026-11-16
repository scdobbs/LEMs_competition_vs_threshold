"""Vertical and horizontal units of the tile grids.

The Allegheny Plateau tiles store z in US survey feet with x, y in metres
(``reports/directives/ap_vertical_units_fix_directive.md``). Every tile config
declares ``z_units`` and ``xy_units``; the loader converts z to metres unless the
``reproduce_v1_units`` flag is set, in which case it logs a loud warning and
leaves z as stored (Phase 1/2 regression only).

The US survey foot is exactly 1200/3937 m = 0.304800609601 m. The directive
quotes 0.3048 (international foot) and c = 3.2808; the two differ by 2 ppm,
far below any tolerance used in this project. ``FT_US_IN_M`` is the single
constant used everywhere so that scaled comparisons (Phase 5.4) are exact.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

FT_US_IN_M: float = 1200.0 / 3937.0
"""Metres per US survey foot (exact)."""

M_IN_FT_US: float = 1.0 / FT_US_IN_M
"""US survey feet per metre (3.28083333...)."""

Z_UNIT_FACTORS: dict[str, float] = {"m": 1.0, "ftUS": FT_US_IN_M, "ft": 0.3048}
XY_UNITS_ALLOWED = ("m",)


class UnitError(ValueError):
    """Raised when declared units fail validation or sanity checks."""


def z_scale_factor(z_units: str) -> float:
    """Multiplicative factor taking stored z to metres."""
    try:
        return Z_UNIT_FACTORS[z_units]
    except KeyError as exc:
        raise UnitError(f"unknown z_units {z_units!r}; allowed: {sorted(Z_UNIT_FACTORS)}") from exc


def z_to_metres(z: np.ndarray, z_units: str, reproduce_v1_units: bool = False) -> tuple[np.ndarray, float]:
    """Return (z in metres, factor applied).

    With ``reproduce_v1_units=True`` the data are returned unchanged (factor 1.0)
    and a warning is logged, because v1 ran the AP tiles in feet.
    """
    factor = z_scale_factor(z_units)
    if reproduce_v1_units:
        if factor != 1.0:
            logger.warning(
                "reproduce_v1_units=True: elevations declared as %s are NOT converted to metres "
                "(factor %.9f skipped). Results are in the stored vertical unit, as in v1.",
                z_units, factor,
            )
        return z, 1.0
    if factor == 1.0:
        return z, 1.0
    return z * factor, factor


def check_xy_units(xy_units: str) -> None:
    if xy_units not in XY_UNITS_ALLOWED:
        raise UnitError(f"xy_units must be one of {XY_UNITS_ALLOWED}, got {xy_units!r}")


def median_pixel_slope(z: np.ndarray, dx: float, dy: float | None = None) -> float:
    """Median |grad z| by central differences (NaN-aware), for the sanity check."""
    dy = dx if dy is None else dy
    gy, gx = np.gradient(z, dy, dx)
    s = np.sqrt(gx * gx + gy * gy)
    return float(np.nanmedian(s))


def sanity_check_elevation(
    z_m: np.ndarray,
    dx: float,
    site_z_range_m: tuple[float, float],
    max_median_slope: float,
    label: str = "",
) -> dict[str, float]:
    """Refuse grids whose metre elevations or slopes are implausible for the site.

    Raises :class:`UnitError` if the finite elevation range falls outside
    ``site_z_range_m`` or the median pixel slope exceeds ``max_median_slope``
    (0.6 for soil-mantled sites per the directive). Returns the diagnostics.
    """
    finite = np.isfinite(z_m)
    zmin, zmax = float(z_m[finite].min()), float(z_m[finite].max())
    med_slope = median_pixel_slope(z_m, dx)
    diag = {"z_min_m": zmin, "z_max_m": zmax, "median_slope": med_slope}
    lo, hi = site_z_range_m
    problems = []
    if zmin < lo or zmax > hi:
        problems.append(f"elevation range [{zmin:.1f}, {zmax:.1f}] m outside site range [{lo}, {hi}] m")
    if med_slope > max_median_slope:
        problems.append(f"median pixel slope {med_slope:.3f} > {max_median_slope}")
    if problems:
        raise UnitError(f"{label}: declared units disagree with the data: " + "; ".join(problems))
    return diag
