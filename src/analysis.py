"""
Slope-area regression and channel steepness analysis.

Provides log-binned regression, IQR error bars, and weighted R² for
quantifying channel steepness (k_s) and concavity (theta) from
elevation, drainage area, and flow direction data.
"""

from __future__ import annotations

import numpy as np


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
