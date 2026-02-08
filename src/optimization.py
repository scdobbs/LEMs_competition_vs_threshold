"""
Optimization wrappers for landscape evolution model calibration.

Each function wraps :func:`scipy.optimize.minimize` with log-space
parameterisation, LRU caching of forward solves, and MSE misfit
computation against observed elevations.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def optimize_advection_diffusion(
    solver: Callable[[float, float, float], np.ndarray],
    dem_griddata: np.ndarray,
    U: float,
    K0: float,
    D0: float,
    K_bounds: tuple[float, float] = (1e-12, 2e-3),
    D_bounds: tuple[float, float] = (4e-10, 1e-2),
    method: str = "L-BFGS-B",
) -> tuple[float, float, float]:
    """Optimise K and D for the advection-diffusion model.

    Parameters
    ----------
    solver : callable
        ``output_function(U, K, D) → Z`` returned by
        :func:`models.advection_diffusion_model`.
    dem_griddata : np.ndarray
        Observed elevation grid (used for MSE computation).
    U : float
        Uplift rate (held fixed during optimisation).
    K0, D0 : float
        Initial guesses for erodibility and diffusivity.
    K_bounds, D_bounds : tuple[float, float]
        Lower and upper bounds for K and D.
    method : str
        ``scipy.optimize.minimize`` method (default ``"L-BFGS-B"``).

    Returns
    -------
    K_star : float
        Optimised erodibility.
    D_star : float
        Optimised diffusivity.
    mse_star : float
        Final MSE at the optimum.
    """

    @lru_cache(maxsize=None)
    def _solve(log10_K: float, log10_D: float) -> np.ndarray:
        return solver(U, 10 ** log10_K, 10 ** log10_D)

    def _misfit(x: np.ndarray) -> float:
        Z_mod = _solve(float(x[0]), float(x[1]))
        return float(np.nanmean((Z_mod - dem_griddata) ** 2))

    x0 = np.log10([K0, D0])
    bounds = [tuple(np.log10(K_bounds)), tuple(np.log10(D_bounds))]

    res = minimize(_misfit, x0, method=method, bounds=bounds)
    K_star = 10 ** res.x[0]
    D_star = 10 ** res.x[1]
    mse_star = float(res.fun)

    logger.info(
        "AD optimisation: K*=%.3e, D*=%.3e, MSE=%.4e", K_star, D_star, mse_star,
    )
    return K_star, D_star, mse_star


def optimize_threshold_acrit(
    partitioned_model: Callable[..., tuple[np.ndarray, np.ndarray]],
    dem_griddata: np.ndarray,
    D0: float,
    A_crit_0: float = 5000.0,
    A_crit_bounds: tuple[float, float] = (1e2, 1e5),
    method: str = "Nelder-Mead",
) -> tuple[float, float]:
    """Optimise A_crit for the partitioned threshold model (linear diffusion).

    Parameters
    ----------
    partitioned_model : callable
        ``f(A_crit, D) → (Z_ss, channel_mask)`` returned by
        :func:`models.partitioned_threshold_model`.
    dem_griddata : np.ndarray
        Observed elevation grid.
    D0 : float
        Fixed diffusivity.
    A_crit_0 : float
        Initial guess for critical area (default 5000).
    A_crit_bounds : tuple[float, float]
        Bounds for A_crit.
    method : str
        Optimisation method (default ``"Nelder-Mead"``).

    Returns
    -------
    A_crit_star : float
        Optimised critical area.
    mse_star : float
        Final MSE at the optimum.
    """

    @lru_cache(maxsize=None)
    def _solve(log10_A_crit: float) -> np.ndarray:
        Z, _ = partitioned_model(10 ** log10_A_crit, D0)
        return Z

    def _misfit(x: np.ndarray) -> float:
        Z_mod = _solve(float(x[0]))
        return float(np.nanmean((Z_mod - dem_griddata) ** 2))

    x0 = np.array([np.log10(A_crit_0)])
    bounds = [tuple(np.log10(A_crit_bounds))]

    res = minimize(_misfit, x0, method=method, bounds=bounds)
    A_crit_star = 10 ** res.x[0]
    mse_star = float(res.fun)

    logger.info("A_crit optimisation: A_crit*=%.1f, MSE=%.4e", A_crit_star, mse_star)
    return A_crit_star, mse_star


def optimize_threshold_acrit_nonlinear(
    partitioned_model_nl: Callable[..., tuple[np.ndarray, np.ndarray]],
    dem_griddata: np.ndarray,
    D0: float,
    Sc: float,
    A_crit_0: float = 5000.0,
    A_crit_bounds: tuple[float, float] = (1e2, 1e5),
    method: str = "Nelder-Mead",
) -> tuple[float, float]:
    """Optimise A_crit for the nonlinear partitioned threshold model.

    Parameters
    ----------
    partitioned_model_nl : callable
        ``f(A_crit, D_linear, S_c) → (Z_ss, channel_mask)`` returned by
        :func:`models.partitioned_threshold_model_nonlinear`.
    dem_griddata : np.ndarray
        Observed elevation grid.
    D0 : float
        Fixed linear diffusivity.
    Sc : float
        Critical slope for nonlinear diffusion.
    A_crit_0 : float
        Initial guess for critical area.
    A_crit_bounds : tuple[float, float]
        Bounds for A_crit.
    method : str
        Optimisation method.

    Returns
    -------
    A_crit_star : float
        Optimised critical area.
    mse_star : float
        Final MSE at the optimum.
    """

    @lru_cache(maxsize=None)
    def _solve(log10_A_crit: float) -> np.ndarray:
        Z, _ = partitioned_model_nl(10 ** log10_A_crit, D0, Sc)
        return Z

    def _misfit(x: np.ndarray) -> float:
        Z_mod = _solve(float(x[0]))
        return float(np.nanmean((Z_mod - dem_griddata) ** 2))

    x0 = np.array([np.log10(A_crit_0)])
    bounds = [tuple(np.log10(A_crit_bounds))]

    res = minimize(_misfit, x0, method=method, bounds=bounds)
    A_crit_star = 10 ** res.x[0]
    mse_star = float(res.fun)

    logger.info("NL A_crit optimisation: A_crit*=%.1f, MSE=%.4e", A_crit_star, mse_star)
    return A_crit_star, mse_star


def optimize_threshold_full(
    dem: Any,
    area: Any,
    flow_direction: Any,
    m: float,
    U: float,
    dem_griddata: np.ndarray,
    K0: float,
    D0: float,
    A_crit_0: float = 5000.0,
    K_bounds: tuple[float, float] = (1e-12, 2e-3),
    D_bounds: tuple[float, float] = (4e-10, 1e-2),
    A_crit_bounds: tuple[float, float] = (1e2, 1e5),
    method: str = "Nelder-Mead",
) -> tuple[float, float, float, float]:
    """Jointly optimise A_crit, D, and K for the partitioned model.

    Rebuilds the partitioned model for each candidate K, so this is
    slower than single-parameter optimisation.

    Parameters
    ----------
    dem, area, flow_direction : raster-like objects
        Input grids (passed to :func:`models.partitioned_threshold_model`).
    m : float
        Area exponent.
    U : float
        Uplift rate.
    dem_griddata : np.ndarray
        Observed elevation for MSE.
    K0, D0, A_crit_0 : float
        Initial guesses.
    K_bounds, D_bounds, A_crit_bounds : tuple[float, float]
        Parameter bounds.
    method : str
        Optimisation method.

    Returns
    -------
    A_crit_star, D_star, K_star : float
        Optimised parameters.
    mse_star : float
        Final MSE.
    """
    from models import partitioned_threshold_model

    @lru_cache(maxsize=None)
    def _solve(
        log10_A_crit: float,
        log10_D: float,
        log10_K: float,
    ) -> np.ndarray:
        K = 10 ** log10_K
        D = 10 ** log10_D
        A_crit = 10 ** log10_A_crit
        model_fn = partitioned_threshold_model(dem, area, flow_direction, m, U, K)
        Z, _ = model_fn(A_crit, D)
        return Z

    def _misfit(x: np.ndarray) -> float:
        Z_mod = _solve(float(x[0]), float(x[1]), float(x[2]))
        return float(np.nanmean((Z_mod - dem_griddata) ** 2))

    x0 = np.log10([A_crit_0, D0, K0])
    bounds = [
        tuple(np.log10(A_crit_bounds)),
        tuple(np.log10(D_bounds)),
        tuple(np.log10(K_bounds)),
    ]

    res = minimize(_misfit, x0, method=method, bounds=bounds)
    A_crit_star = 10 ** res.x[0]
    D_star = 10 ** res.x[1]
    K_star = 10 ** res.x[2]
    mse_star = float(res.fun)

    logger.info(
        "Full optimisation: A_crit*=%.1f, D*=%.3e, K*=%.3e, MSE=%.4e",
        A_crit_star, D_star, K_star, mse_star,
    )
    return A_crit_star, D_star, K_star, mse_star
