"""Measured parameters: theta, k_s, hilltop curvature; slope-area utilities."""

from lemcvt.measure.slope_area import (
    compute_iqr_errors,
    fit_slope_area,
    ks_from_loglog_fixed_theta,
    ks_local_from_SA,
    ks_mod_from_modeled,
    ks_obs_from_observed,
    slope_area_arrays_from_elevation,
    slope_area_regression_binned,
    weighted_r2,
)

__all__ = [
    "compute_iqr_errors",
    "fit_slope_area",
    "ks_from_loglog_fixed_theta",
    "ks_local_from_SA",
    "ks_mod_from_modeled",
    "ks_obs_from_observed",
    "slope_area_arrays_from_elevation",
    "slope_area_regression_binned",
    "weighted_r2",
]
