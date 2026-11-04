"""Across-tile fits of log10 A_crit on log10(median head slope).

Model variants: one line (OLS and reduced major axis), common slope with
separate site intercepts, each site alone, and the same without AP tile 5.
Uncertainty is a two-level bootstrap: resample tiles, then resample heads
within each resampled tile to recompute its median slope.

The exponent ``b`` is the slope of log10 A_crit against log10 S, i.e.
A_crit is proportional to S**b; Montgomery & Dietrich's A S^2 = const is b = -2.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# point estimates
# ---------------------------------------------------------------------------

def fit_ols(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    n = len(x)
    s = float(np.sqrt(np.sum(res**2) / (n - 2))) if n > 2 else np.nan
    r = float(np.corrcoef(x, y)[0, 1]) if n > 2 else np.nan
    return dict(method="OLS", a=float(a), b=float(b), n=n, sigma=s, r2=r**2, residuals=res.tolist())


def fit_rma(x: np.ndarray, y: np.ndarray) -> dict:
    """Reduced major axis: b = sign(r) * sd(y) / sd(x) (both variables carry error)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    b = float(np.sign(r) * np.std(y, ddof=1) / np.std(x, ddof=1))
    a = float(np.mean(y) - b * np.mean(x))
    res = y - (a + b * x)
    n = len(x)
    s = float(np.sqrt(np.sum(res**2) / (n - 2))) if n > 2 else np.nan
    return dict(method="RMA", a=a, b=b, n=n, sigma=s, r2=r**2, residuals=res.tolist())


def fit_common_slope(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """y = a_group + b x by least squares (parallel lines, one per site)."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    levels = sorted(set(g.tolist()))
    X = np.column_stack([x] + [(g == lv).astype(float) for lv in levels])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    b = float(coef[0])
    intercepts = {lv: float(c) for lv, c in zip(levels, coef[1:])}
    pred = X @ coef
    res = y - pred
    n, p = len(x), X.shape[1]
    s = float(np.sqrt(np.sum(res**2) / (n - p))) if n > p else np.nan
    sst = float(np.sum((y - y.mean())**2))
    return dict(method="common_slope", b=b, intercepts=intercepts, n=n, sigma=s,
                r2=1 - float(np.sum(res**2)) / sst if sst > 0 else np.nan, residuals=res.tolist())


# ---------------------------------------------------------------------------
# two-level bootstrap
# ---------------------------------------------------------------------------

def _resampled_median(rng: np.random.Generator, slopes: np.ndarray) -> float:
    return float(np.median(rng.choice(slopes, len(slopes), replace=True)))


def two_level_bootstrap(keys, logA: dict, site: dict, head_slopes: dict, method: str,
                        n_boot: int, rng: np.random.Generator, resample_tiles: bool = True) -> dict:
    """Bootstrap distribution of the exponent (and intercepts) for one variant.

    keys: tile keys in the fit. logA[key]: log10 A_crit. site[key]: site label.
    head_slopes[key]: array of head slopes (positive). method: OLS | RMA | common_slope.
    Each iteration resamples the tiles with replacement (unless resample_tiles
    is False, which keeps the tile set fixed and only resamples heads), then
    resamples heads within each tile to recompute its median slope.
    Returns dict(b=array, a=array or None, intercepts=dict of arrays, n_ok, n_skipped).
    """
    keys = list(keys)
    n = len(keys)
    bs, as_, ints = [], [], {}
    n_skip = 0
    for _ in range(n_boot):
        sel = rng.integers(0, n, n) if resample_tiles else np.arange(n)
        xs, ys, gs = [], [], []
        for k in (keys[i] for i in sel):
            xs.append(np.log10(_resampled_median(rng, head_slopes[k])))
            ys.append(logA[k])
            gs.append(site[k])
        xs, ys, gs = np.array(xs), np.array(ys), np.array(gs)
        if np.std(xs) == 0 or len(set(gs.tolist())) < (2 if method == "common_slope" else 1):
            n_skip += 1
            continue
        if method == "OLS":
            f = fit_ols(xs, ys)
            as_.append(f["a"])
        elif method == "RMA":
            f = fit_rma(xs, ys)
            as_.append(f["a"])
        else:
            f = fit_common_slope(xs, ys, gs)
            for lv, v in f["intercepts"].items():
                ints.setdefault(lv, []).append(v)
        bs.append(f["b"])
    return dict(b=np.array(bs), a=np.array(as_) if as_ else None,
                intercepts={k: np.array(v) for k, v in ints.items()}, n_ok=len(bs), n_skipped=n_skip)


def ci(values: np.ndarray, pcts) -> list[float]:
    if values is None or len(values) == 0:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(values, pcts)
    return [float(lo), float(hi)]


# ---------------------------------------------------------------------------
# prediction for a held-out tile
# ---------------------------------------------------------------------------

def ols_prediction_interval(x: np.ndarray, y: np.ndarray, x0: float, alpha: float = 0.05) -> dict:
    """Classical OLS prediction interval for a new observation at x0."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = fit_ols(x, y)
    n = len(x)
    sxx = float(np.sum((x - x.mean())**2))
    se = f["sigma"] * np.sqrt(1 + 1 / n + (x0 - x.mean())**2 / sxx)
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    yhat = f["a"] + f["b"] * x0
    return dict(predicted=float(yhat), lo=float(yhat - t * se), hi=float(yhat + t * se), se=float(se), t=float(t))


def bootstrap_prediction_interval(boot: dict, residuals, x0_samples: np.ndarray, rng: np.random.Generator,
                                  pcts, intercept_key=None) -> dict:
    """Prediction interval from bootstrap (a, b) pairs plus a residual draw plus x0 uncertainty.

    For each bootstrap fit: y = a_b + b_b * x0_b + e, with x0_b a resampled
    median slope of the held-out tile and e drawn from the full-fit residuals.
    """
    res = np.asarray(residuals, float)
    b = boot["b"]
    a = boot["a"] if intercept_key is None else boot["intercepts"][intercept_key]
    m = min(len(a), len(b))
    if m == 0:
        return dict(lo=np.nan, hi=np.nan)
    x0 = rng.choice(x0_samples, m, replace=True)
    e = rng.choice(res, m, replace=True)
    pred = a[:m] + b[:m] * x0 + e
    lo, hi = np.percentile(pred, pcts)
    return dict(lo=float(lo), hi=float(hi), median=float(np.median(a[:m] + b[:m] * x0)))
