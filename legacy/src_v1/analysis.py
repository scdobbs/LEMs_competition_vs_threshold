"""
Slope-area regression and channel steepness analysis.

Provides log-binned regression, IQR error bars, and weighted R² for
quantifying channel steepness (k_s) and concavity (theta) from
elevation, drainage area, and flow direction data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress


def slope_area_regression_binned(
    elev_obj,
    area_obj,
    fd_obj,
    min_area: float,
    vertical_interval: int = 10,
    nbins: int = 24,
    min_per_bin: int = 10,
    agg: str = "median",
    require_weighted: bool = True,
    relax_if_sparse: bool = True,
) -> tuple[float, float, dict]:
    r"""Log-binned slope-area regression for channel steepness and concavity.

    Estimates :math:`k_s` and :math:`\theta` from the power-law relation

    .. math::

        S = k_s \, A^{-\theta}

    by binning in :math:`\log_{10}(A)` and fitting a weighted least-squares
    line to the bin aggregates, avoiding overweighting of the many small-area
    pixels.

    Parameters
    ----------
    elev_obj : Elevation-like object
    area_obj : Area-like object
    fd_obj : FlowDirectionD8-like object
    min_area : float
        Minimum drainage area threshold for channel pixels.
    vertical_interval : int
        Vertical interval for ``ChannelSlopeWithSmoothing`` (default 10).
    nbins : int
        Number of equal-width bins in log10(A) (default 24).
    min_per_bin : int
        Minimum raw points per bin (default 10).
    agg : ``{"median", "mean"}``
        Aggregate statistic within each bin.
    require_weighted : bool
        Use weighted least squares with bin counts (default ``True``).
    relax_if_sparse : bool
        Progressively relax *min_per_bin* if fewer than 3 bins survive
        (default ``True``).

    Returns
    -------
    ks : float
        Channel steepness index.
    theta : float
        Concavity (positive).
    debug : dict
        Diagnostic arrays: ``A``, ``S``, ``logA``, ``logS``,
        ``bin_edges``, ``bin_centers``, ``bin_logA``, ``bin_logS``,
        ``bin_counts``.

    Raises
    ------
    ValueError
        If insufficient data or too few bins survive filtering.
    """
    import TopoAnalysis.dem as d

    slope_obj = d.ChannelSlopeWithSmoothing(
        elevation=elev_obj,
        area=area_obj,
        flow_direction=fd_obj,
        vertical_interval=vertical_interval,
        min_area=min_area,
    )

    S = slope_obj._griddata.flatten()
    A = area_obj._griddata.flatten()

    valid = (A >= float(min_area)) & (S > 0) & np.isfinite(A) & np.isfinite(S)
    if np.count_nonzero(valid) < 30:
        raise ValueError(
            f"Insufficient channel pixels after masking: {np.count_nonzero(valid)} points"
        )

    logA = np.log10(A[valid])
    logS = np.log10(S[valid])

    a_min, a_max = logA.min(), logA.max()
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_min == a_max:
        raise ValueError("Degenerate drainage area range for binning.")

    bin_edges = np.linspace(a_min, a_max, nbins + 1)

    def _bin_aggregate(min_count: int):
        bA, bS, bN = [], [], []
        for i in range(nbins):
            left, right = bin_edges[i], bin_edges[i + 1]
            if i < nbins - 1:
                idx = (logA >= left) & (logA < right)
            else:
                idx = (logA >= left) & (logA <= right)
            n_in = int(np.count_nonzero(idx))
            if n_in < min_count:
                continue
            if agg == "median":
                bA.append(float(np.median(logA[idx])))
                bS.append(float(np.median(logS[idx])))
            elif agg == "mean":
                bA.append(float(np.mean(logA[idx])))
                bS.append(float(np.mean(logS[idx])))
            else:
                raise ValueError("agg must be 'median' or 'mean'.")
            bN.append(n_in)
        return bA, bS, bN

    bA, bS, bN = _bin_aggregate(min_per_bin)

    if relax_if_sparse and len(bA) < 3:
        for target in [max(3, min_per_bin // 2), 3]:
            bA, bS, bN = _bin_aggregate(target)
            if len(bA) >= 3:
                break

    if len(bA) < 3:
        raise ValueError(
            f"Too few populated bins for regression (got {len(bA)}). "
            "Consider lowering min_area, reducing nbins, or relaxing min_per_bin."
        )

    bin_logA = np.asarray(bA)
    bin_logS = np.asarray(bS)
    w = np.asarray(bN, dtype=float)

    if require_weighted:
        wx = np.sum(w * bin_logA) / np.sum(w)
        wy = np.sum(w * bin_logS) / np.sum(w)
        num = np.sum(w * (bin_logA - wx) * (bin_logS - wy))
        den = np.sum(w * (bin_logA - wx) ** 2)
        if den == 0:
            raise RuntimeError("Zero variance in binned log10(A); cannot fit.")
        b1 = num / den
        b0 = wy - b1 * wx
    else:
        b1, b0 = np.polyfit(bin_logA, bin_logS, 1)

    ks = 10 ** b0
    theta = -b1

    debug = dict(
        A=A[valid],
        S=S[valid],
        logA=logA,
        logS=logS,
        bin_edges=bin_edges,
        bin_centers=0.5 * (bin_edges[:-1] + bin_edges[1:]),
        bin_logA=bin_logA,
        bin_logS=bin_logS,
        bin_counts=w,
    )
    return ks, theta, debug


def compute_iqr_errors(
    dbg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 25-75% IQR error bars for each bin from a regression debug dict.

    Parameters
    ----------
    dbg : dict
        Debug dictionary from :func:`slope_area_regression_binned`.

    Returns
    -------
    lower_err : np.ndarray
        ``median - q25`` for each bin.
    upper_err : np.ndarray
        ``q75 - median`` for each bin.
    """
    A = dbg["A"]
    S = dbg["S"]
    bA = dbg["bin_logA"]
    nb = len(bA)

    # Build edges in log10-space from bin centres
    centers_log = bA
    edges_log = np.empty(nb + 1, dtype=float)
    edges_log[1:-1] = 0.5 * (centers_log[:-1] + centers_log[1:])
    edges_log[0] = centers_log[0] - (edges_log[1] - centers_log[0])
    edges_log[-1] = centers_log[-1] + (centers_log[-1] - edges_log[-2])
    edges = 10 ** edges_log

    lower_err: list[float] = []
    upper_err: list[float] = []

    for i in range(nb):
        mask = (A >= edges[i]) & (A < edges[i + 1])
        svals = S[mask]

        if np.sum(mask) < 3:
            lower_err.append(0.0)
            upper_err.append(0.0)
            continue

        q25 = float(np.percentile(svals, 25))
        q75 = float(np.percentile(svals, 75))
        med = float(np.median(svals))

        lower_err.append(med - q25)
        upper_err.append(q75 - med)

    return np.array(lower_err), np.array(upper_err)


def weighted_r2(
    x: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    w: np.ndarray,
) -> float:
    """Weighted coefficient of determination.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (unused, kept for API compatibility).
    y : np.ndarray
        Observed values.
    yhat : np.ndarray
        Predicted values.
    w : np.ndarray
        Weights.

    Returns
    -------
    float
        Weighted R².
    """
    ybar = np.sum(w * y) / np.sum(w)
    SS_res = np.sum(w * (y - yhat) ** 2)
    SS_tot = np.sum(w * (y - ybar) ** 2)
    if SS_tot == 0:
        return float("nan")
    return float(1 - SS_res / SS_tot)


# ---------------------------------------------------------------------------
# Slope-area regression (fixed or free theta)
# ---------------------------------------------------------------------------

def fit_slope_area(
    A_1d: np.ndarray,
    S_1d: np.ndarray,
    theta_fixed: float | None = None,
    min_points: int = 5,
) -> dict:
    r"""Log-log slope-area regression with fixed or free concavity.

    Parameters
    ----------
    A_1d, S_1d : np.ndarray
        1-D arrays of drainage area and slope.
    theta_fixed : float or None
        If given, fix concavity to this value; otherwise fit freely.
    min_points : int
        Minimum valid points required for regression.

    Returns
    -------
    dict
        Keys: ``ks``, ``theta``, ``r2``, ``n_good``, ``logA``, ``logS``,
        ``logS_pred``, ``intercept``, ``slope``, ``used_mask``.
    """
    used_mask = (
        np.isfinite(A_1d) & np.isfinite(S_1d) & (A_1d > 0) & (S_1d > 0)
    )
    n_good = int(used_mask.sum())

    out: dict = {
        "ks": np.nan, "theta": np.nan, "r2": np.nan, "n_good": n_good,
        "logA": None, "logS": None, "logS_pred": None,
        "intercept": np.nan, "slope": np.nan,
        "used_mask": used_mask,
    }

    if n_good < min_points:
        return out

    logA = np.log10(A_1d[used_mask])
    logS = np.log10(S_1d[used_mask])

    if theta_fixed is not None:
        c = -float(theta_fixed)
        b = float(np.mean(logS - c * logA))
        logS_pred = b + c * logA

        ss_res = float(np.sum((logS - logS_pred) ** 2))
        ss_tot = float(np.sum((logS - np.mean(logS)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        out.update({
            "ks": float(10 ** b),
            "theta": float(theta_fixed),
            "r2": float(r2),
            "logA": logA, "logS": logS, "logS_pred": logS_pred,
            "intercept": float(b), "slope": float(c),
        })
        return out

    # free-theta regression
    slope_val, intercept_val, r_value, _, _ = linregress(logA, logS)
    logS_pred = intercept_val + slope_val * logA

    out.update({
        "ks": float(10 ** intercept_val),
        "theta": float(-slope_val),
        "r2": float(r_value ** 2),
        "logA": logA, "logS": logS, "logS_pred": logS_pred,
        "intercept": float(intercept_val), "slope": float(slope_val),
    })
    return out


def ks_local_from_SA(
    A_1d: np.ndarray,
    S_1d: np.ndarray,
    m: float,
    min_points: int = 5,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    r"""Local steepness proxy :math:`k_s = S \, A^m` along a channel.

    Parameters
    ----------
    A_1d, S_1d : np.ndarray
        1-D arrays of drainage area and slope.
    m : float
        Area exponent (= fixed concavity).
    min_points : int
        Minimum valid points for summary statistics.

    Returns
    -------
    ks_local : np.ndarray
        Per-pixel steepness array.
    mask : np.ndarray
        Boolean mask of valid entries.
    n_good : int
        Number of valid points.
    median, q25, q75 : float
        Summary statistics (NaN if fewer than *min_points*).
    """
    ks_local = S_1d * (A_1d ** m)

    mask = (
        np.isfinite(ks_local) & np.isfinite(A_1d) & np.isfinite(S_1d)
        & (A_1d > 0) & (S_1d > 0) & (ks_local > 0)
    )
    n_good = int(mask.sum())

    if n_good < min_points:
        return ks_local, mask, n_good, np.nan, np.nan, np.nan

    vals = ks_local[mask]
    med = float(np.nanmedian(vals))
    q25 = float(np.nanpercentile(vals, 25))
    q75 = float(np.nanpercentile(vals, 75))
    return ks_local, mask, n_good, med, q25, q75


# ---------------------------------------------------------------------------
# Ks estimation with fixed theta (log-space intercept)
# ---------------------------------------------------------------------------

def ks_from_loglog_fixed_theta(
    A: np.ndarray,
    S: np.ndarray,
    theta_fixed: float,
) -> tuple[float, float, float]:
    r"""Estimate :math:`K_s` with fixed concavity via mean log-space intercept.

    Parameters
    ----------
    A, S : np.ndarray
        Drainage area and slope arrays (1-D, positive, finite).
    theta_fixed : float
        Fixed concavity index.

    Returns
    -------
    Ks : float
        Channel steepness.
    theta : float
        Echo of *theta_fixed*.
    r2 : float
        Coefficient of determination.
    """
    logA = np.log10(A)
    logS = np.log10(S)

    theta = float(theta_fixed)
    y = logS + theta * logA
    logKs = float(np.nanmean(y))
    Ks = 10.0 ** logKs

    yhat = logKs - theta * logA
    ss_res = float(np.sum((logS - yhat) ** 2))
    ss_tot = float(np.sum((logS - np.mean(logS)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(Ks), float(theta), float(r2)


def slope_area_arrays_from_elevation(
    elev_obj,
    area_obj,
    fd_obj,
    min_area: float = 1e5,
    vertical_interval: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract paired (A, S) arrays from TopoAnalysis raster objects.

    Parameters
    ----------
    elev_obj, area_obj, fd_obj : raster-like
        Elevation, area, and flow-direction objects.
    min_area : float
        Minimum drainage area for channel pixels.
    vertical_interval : int
        Vertical interval for slope smoothing.

    Returns
    -------
    A, S : np.ndarray
        1-D arrays of area and slope for valid channel pixels.
    """
    import TopoAnalysis.dem as d

    slope_obj = d.ChannelSlopeWithSmoothing(
        elevation=elev_obj,
        area=area_obj,
        flow_direction=fd_obj,
        vertical_interval=vertical_interval,
        min_area=min_area,
    )
    S = slope_obj._griddata
    A = area_obj._griddata
    m = np.isfinite(S) & np.isfinite(A) & (S > 0) & (A >= float(min_area))
    return A[m].astype(float), S[m].astype(float)


def ks_obs_from_observed(
    filled_obs,
    area,
    fd,
    theta: float,
    min_area: float = 1e5,
    vertical_interval: int = 10,
) -> tuple[float, float, int, float]:
    """Compute Ks from observed elevation at a given resolution.

    Parameters
    ----------
    filled_obs : elevation-like
        Observed (filled) elevation object.
    area, fd : raster-like
        Area and flow direction objects.
    theta : float
        Fixed concavity.
    min_area : float
        Minimum drainage area.
    vertical_interval : int
        Vertical interval for slope smoothing.

    Returns
    -------
    Ks, r2 : float
        Steepness and R².
    n_pts : int
        Number of valid points.
    A_max : float
        Maximum drainage area of valid points.
    """
    A_obs, S_obs = slope_area_arrays_from_elevation(
        filled_obs, area, fd, min_area=min_area,
        vertical_interval=vertical_interval,
    )
    if A_obs.size == 0:
        return np.nan, np.nan, 0, np.nan
    Ks, _, r2 = ks_from_loglog_fixed_theta(A_obs, S_obs, theta_fixed=theta)
    return float(Ks), float(r2), int(A_obs.size), float(np.nanmax(A_obs))


def ks_mod_from_modeled(
    Z_model: np.ndarray,
    georef_template_dem,
    area,
    fd,
    theta: float,
    min_area: float = 1e5,
    vertical_interval: int = 10,
) -> tuple[float, float, int, float]:
    """Compute Ks from a modelled elevation array.

    Parameters
    ----------
    Z_model : np.ndarray
        2-D modelled elevation.
    georef_template_dem : raster-like
        Template DEM providing ``._georef_info``.
    area, fd : raster-like
        Area and flow direction objects.
    theta : float
        Fixed concavity.
    min_area : float
        Minimum drainage area.
    vertical_interval : int
        Vertical interval for slope smoothing.

    Returns
    -------
    Ks, r2 : float
    n_pts : int
    A_max : float
    """
    import TopoAnalysis.dem as d

    dem_model = d.Elevation()
    dem_model._griddata = np.asarray(Z_model, dtype=float)
    dem_model._georef_info = georef_template_dem._georef_info
    dem_model._nodata_value = np.nan

    A_mod, S_mod = slope_area_arrays_from_elevation(
        dem_model, area, fd, min_area=min_area,
        vertical_interval=vertical_interval,
    )
    if A_mod.size == 0:
        return np.nan, np.nan, 0, np.nan
    Ks, _, r2 = ks_from_loglog_fixed_theta(A_mod, S_mod, theta_fixed=theta)
    return float(Ks), float(r2), int(A_mod.size), float(np.nanmax(A_mod))
