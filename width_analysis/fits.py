"""Censored (Tobit) power-law fits, bootstrap intervals, OLS comparison and
the logistic channelized-fraction fit.

All fits are in log10: log10 W = a + b log10 A, c = 10**a.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Tobit (left-censored) maximum likelihood
# ---------------------------------------------------------------------------

def tobit_nll(theta: np.ndarray, x: np.ndarray, y: np.ndarray, cens: np.ndarray, ybound: np.ndarray) -> float:
    """Negative log-likelihood of y = a + b x + N(0, sigma^2), with y <= ybound
    observed only as the bound where ``cens`` is True."""
    a, b, log_sigma = theta
    sigma = np.exp(log_sigma)
    mu = a + b * x
    ll_obs = norm.logpdf(y[~cens], loc=mu[~cens], scale=sigma)
    ll_cen = log_ndtr((ybound[cens] - mu[cens]) / sigma)
    return -(ll_obs.sum() + ll_cen.sum())


def tobit_fit(logA: np.ndarray, logW: np.ndarray, censored: np.ndarray, log_bound: np.ndarray) -> dict:
    """Censored MLE of log10 W = a + b log10 A.

    ``logW`` of censored points is ignored; ``log_bound`` is used there.
    Returns dict(a, b, sigma, c, converged, n, n_censored, nll).
    """
    x = np.asarray(logA, float)
    y = np.asarray(logW, float)
    cens = np.asarray(censored, bool)
    yb = np.asarray(log_bound, float)
    y = np.where(cens, yb, y)
    out = dict(a=np.nan, b=np.nan, sigma=np.nan, c=np.nan, converged=False, n=int(len(x)), n_censored=int(cens.sum()), nll=np.nan)
    if len(x) < 3 or (~cens).sum() < 2:
        return out
    # start from OLS on everything (censored at their bounds)
    b0, a0 = np.polyfit(x, y, 1)
    s0 = max(np.std(y - (a0 + b0 * x)), 0.05)
    res = minimize(tobit_nll, np.array([a0, b0, np.log(s0)]), args=(x, y, cens, yb), method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-9, maxiter=20000))
    a, b, ls = res.x
    out.update(a=float(a), b=float(b), sigma=float(np.exp(ls)), c=float(10 ** a), converged=bool(res.success), nll=float(res.fun))
    return out


def bootstrap_tobit(logA, logW, censored, log_bound, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Pairs-bootstrap of the Tobit fit. Returns an (n_ok, 2) array of (a, b)."""
    n = len(logA)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        f = tobit_fit(logA[idx], logW[idx], censored[idx], log_bound[idx])
        if np.isfinite(f["a"]) and np.isfinite(f["b"]):
            out.append((f["a"], f["b"]))
    return np.array(out).reshape(-1, 2)


def ols_uncensored(logA, logW, censored) -> dict:
    """OLS on uncensored points only (biased: drops the narrow channels)."""
    m = ~np.asarray(censored, bool) & np.isfinite(logW)
    out = dict(a=np.nan, b=np.nan, c=np.nan, n=int(m.sum()), label="OLS on uncensored points only (biased)")
    if m.sum() >= 3:
        b, a = np.polyfit(np.asarray(logA)[m], np.asarray(logW)[m], 1)
        out.update(a=float(a), b=float(b), c=float(10 ** a))
    return out


def predict_band(boot_ab: np.ndarray, logA_grid: np.ndarray, pcts=(2.5, 97.5)) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise bootstrap percentiles of log10 W over a log10 A grid."""
    if len(boot_ab) == 0:
        return np.full_like(logA_grid, np.nan), np.full_like(logA_grid, np.nan)
    pred = boot_ab[:, [0]] + boot_ab[:, [1]] * logA_grid[None, :]
    lo, hi = np.percentile(pred, pcts, axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# logistic fit of the channelized fraction
# ---------------------------------------------------------------------------

def _logit_nll(theta, x, y):
    eta = theta[0] + theta[1] * x
    return -np.sum(y * eta - np.logaddexp(0.0, eta))


def logistic_fit(logA: np.ndarray, y: np.ndarray) -> dict:
    """P(channelized) = logistic(b0 + b1 log10 A); A50 = 10**(-b0/b1).

    If the response is all 0 or all 1 the fit is undefined (NaN).
    """
    x = np.asarray(logA, float)
    yy = np.asarray(y, float)
    out = dict(b0=np.nan, b1=np.nan, logA50=np.nan, A50=np.nan, n=int(len(x)), n_pos=int(yy.sum()))
    if len(x) < 4 or yy.sum() == 0 or yy.sum() == len(yy):
        return out
    res = minimize(_logit_nll, np.array([0.0, 1.0]), args=(x - x.mean(), yy), method="BFGS")
    b1 = res.x[1]
    b0 = res.x[0] - b1 * x.mean()
    out.update(b0=float(b0), b1=float(b1))
    if b1 > 0:
        out["logA50"] = float(-b0 / b1)
        out["A50"] = float(10 ** out["logA50"])
    return out


def bootstrap_logistic(logA, y, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of log10 A50 (finite values only)."""
    n = len(logA)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        f = logistic_fit(logA[idx], y[idx])
        if np.isfinite(f["logA50"]):
            vals.append(f["logA50"])
    return np.array(vals)


# ---------------------------------------------------------------------------
# site-level driver
# ---------------------------------------------------------------------------

def fit_site_widths(df, cfg: dict, rng: np.random.Generator, label: str) -> dict:
    """Censored MLE + bootstrap + OLS on the channelized points of one subset.

    ``df`` must have columns area_m2, W_inner_bankbreak, W_inner_bankbreak_censored,
    censor_bound and channelized. Returns a JSON-serialisable dict.
    """
    sub = df[df["channelized"]].copy()
    logA = np.log10(sub["area_m2"].to_numpy(float))
    W = sub["W_inner_bankbreak"].to_numpy(float)
    cens = sub["W_inner_bankbreak_censored"].to_numpy(bool)
    lb = np.log10(sub["censor_bound"].to_numpy(float))
    logW = np.log10(np.where(W > 0, W, np.nan))
    res = dict(label=label, n_channelized=int(len(sub)), n_censored=int(cens.sum()),
               width_measure="W_inner_bankbreak", censor_rule="W < censor_multiple x pixel size")
    fit = tobit_fit(logA, logW, cens, lb)
    res["tobit"] = fit
    boot = bootstrap_tobit(logA, logW, cens, lb, cfg["bootstrap_n"], rng) if len(sub) >= 3 else np.zeros((0, 2))
    res["bootstrap_n_ok"] = int(len(boot))
    if len(boot):
        p = cfg["ci_percentiles"]
        res["c_ci"] = [float(10 ** v) for v in np.percentile(boot[:, 0], p)]
        res["b_ci"] = [float(v) for v in np.percentile(boot[:, 1], p)]
    else:
        res["c_ci"] = res["b_ci"] = [np.nan, np.nan]
    res["ols_uncensored"] = ols_uncensored(logA, logW, cens)
    res["_boot_ab"] = boot          # stripped before JSON
    return res


def fit_site_channelized(df, cfg: dict, rng: np.random.Generator) -> dict:
    """Logistic fit of channelized vs log10 A for one site with a bootstrap CI on A50."""
    logA = np.log10(df["area_m2"].to_numpy(float))
    y = df["channelized"].to_numpy(bool).astype(float)
    f = logistic_fit(logA, y)
    boot = bootstrap_logistic(logA, y, cfg["bootstrap_n"], rng)
    f["bootstrap_n_ok"] = int(len(boot))
    if len(boot):
        lo, hi = np.percentile(boot, cfg["ci_percentiles"])
        f["A50_ci"] = [float(10 ** lo), float(10 ** hi)]
    else:
        f["A50_ci"] = [np.nan, np.nan]
    f["_boot_logA50"] = boot
    return f
