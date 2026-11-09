"""v1 optimizers: exact port of ``src/optimization.py``.

Each function wraps ``scipy.optimize.minimize`` with log10 parameters, an
``lru_cache`` on the forward solve and the nan-aware MSE misfit. The only
addition is a :class:`Trajectory` recorder that logs every misfit evaluation
(parameters, MSE, wall time, cache hit) without altering the sequence of
evaluations or any arithmetic.

Stage names used across the package: ``ad``, ``acrit``, ``nonlinear``, ``joint``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.optimize import minimize

from lemcvt.models.partitioned import partitioned_threshold_model, partitioned_threshold_model_chi

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Every misfit evaluation of one optimizer run."""

    stage: str
    columns: tuple[str, ...]
    rows: list[dict[str, Any]] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def record(self, params: dict[str, float], mse: float, cached: bool) -> None:
        self.rows.append({
            "eval": len(self.rows),
            "t_s": time.perf_counter() - self.t0,
            "cached": cached,
            **params,
            "mse": mse,
        })

    @property
    def n_solves(self) -> int:
        return sum(1 for r in self.rows if not r["cached"])

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows)


@dataclass
class OptResult:
    params: dict[str, float]
    mse: float
    n_iter: int
    n_fev: int
    n_solves: int
    wall_s: float
    success: bool
    message: str
    trajectory: Trajectory
    scipy_x: list[float]


def _mse(Z_mod: np.ndarray, dem_griddata: np.ndarray) -> float:
    return float(np.nanmean((Z_mod - dem_griddata) ** 2))


def optimize_advection_diffusion(
    solver: Callable[[float, float, float], np.ndarray],
    dem_griddata: np.ndarray,
    U: float,
    K0: float,
    D0: float,
    K_bounds: tuple[float, float] = (1e-12, 2e-3),
    D_bounds: tuple[float, float] = (4e-10, 1e-2),
    method: str = "L-BFGS-B",
    return_result: bool = False,
):
    """Optimise K and D for the AD model (L-BFGS-B in log10 space).

    Returns ``(K_star, D_star, mse_star)`` as v1 did, or an :class:`OptResult`
    with the trajectory when ``return_result=True``.
    """
    traj = Trajectory("ad", ("log10_K", "log10_D"))
    t0 = time.perf_counter()

    @lru_cache(maxsize=None)
    def _solve(log10_K: float, log10_D: float) -> np.ndarray:
        _solve.calls += 1  # type: ignore[attr-defined]
        return solver(U, 10**log10_K, 10**log10_D)

    _solve.calls = 0  # type: ignore[attr-defined]

    def _misfit(x: np.ndarray) -> float:
        before = _solve.calls  # type: ignore[attr-defined]
        Z_mod = _solve(float(x[0]), float(x[1]))
        mse = _mse(Z_mod, dem_griddata)
        traj.record({"log10_K": float(x[0]), "log10_D": float(x[1])}, mse, cached=_solve.calls == before)  # type: ignore[attr-defined]
        return mse

    x0 = np.log10([K0, D0])
    bounds = [tuple(np.log10(K_bounds)), tuple(np.log10(D_bounds))]

    res = minimize(_misfit, x0, method=method, bounds=bounds)
    K_star = 10 ** res.x[0]
    D_star = 10 ** res.x[1]
    mse_star = float(res.fun)

    logger.info("AD optimisation: K*=%.3e, D*=%.3e, MSE=%.4e", K_star, D_star, mse_star)
    if not return_result:
        return K_star, D_star, mse_star
    return OptResult({"K": float(K_star), "D": float(D_star)}, mse_star, int(getattr(res, "nit", -1)),
                     int(res.nfev), traj.n_solves, time.perf_counter() - t0, bool(res.success),
                     str(res.message), traj, [float(v) for v in res.x])


def optimize_threshold_acrit(
    partitioned_model: Callable[..., tuple[np.ndarray, np.ndarray]],
    dem_griddata: np.ndarray,
    D0: float,
    A_crit_0: float = 5000.0,
    A_crit_bounds: tuple[float, float] = (1e2, 1e5),
    method: str = "Nelder-Mead",
    return_result: bool = False,
    stage: str = "acrit",
):
    """Optimise A_crit for the linear partitioned model at fixed D0 (Nelder-Mead in log10)."""
    traj = Trajectory(stage, ("log10_A_crit",))
    t0 = time.perf_counter()

    @lru_cache(maxsize=None)
    def _solve(log10_A_crit: float) -> np.ndarray:
        _solve.calls += 1  # type: ignore[attr-defined]
        Z, _ = partitioned_model(10**log10_A_crit, D0)
        return Z

    _solve.calls = 0  # type: ignore[attr-defined]

    def _misfit(x: np.ndarray) -> float:
        before = _solve.calls  # type: ignore[attr-defined]
        Z_mod = _solve(float(x[0]))
        mse = _mse(Z_mod, dem_griddata)
        traj.record({"log10_A_crit": float(x[0])}, mse, cached=_solve.calls == before)  # type: ignore[attr-defined]
        return mse

    x0 = np.array([np.log10(A_crit_0)])
    bounds = [tuple(np.log10(A_crit_bounds))]

    res = minimize(_misfit, x0, method=method, bounds=bounds)
    A_crit_star = 10 ** res.x[0]
    mse_star = float(res.fun)

    logger.info("A_crit optimisation: A_crit*=%.1f, MSE=%.4e", A_crit_star, mse_star)
    if not return_result:
        return A_crit_star, mse_star
    return OptResult({"A_crit": float(A_crit_star)}, mse_star, int(getattr(res, "nit", -1)), int(res.nfev),
                     traj.n_solves, time.perf_counter() - t0, bool(res.success), str(res.message), traj,
                     [float(v) for v in res.x])


def optimize_threshold_acrit_nonlinear(
    partitioned_model_nl: Callable[..., tuple[np.ndarray, np.ndarray]],
    dem_griddata: np.ndarray,
    D0: float,
    Sc: float,
    A_crit_0: float = 5000.0,
    A_crit_bounds: tuple[float, float] = (1e2, 1e5),
    method: str = "Nelder-Mead",
    return_result: bool = False,
):
    """Optimise A_crit for the nonlinear partitioned model at fixed D0 and S_c."""
    traj = Trajectory("nonlinear", ("log10_A_crit",))
    t0 = time.perf_counter()

    @lru_cache(maxsize=None)
    def _solve(log10_A_crit: float) -> np.ndarray:
        _solve.calls += 1  # type: ignore[attr-defined]
        Z, _ = partitioned_model_nl(10**log10_A_crit, D0, Sc)
        return Z

    _solve.calls = 0  # type: ignore[attr-defined]

    def _misfit(x: np.ndarray) -> float:
        before = _solve.calls  # type: ignore[attr-defined]
        Z_mod = _solve(float(x[0]))
        mse = _mse(Z_mod, dem_griddata)
        traj.record({"log10_A_crit": float(x[0])}, mse, cached=_solve.calls == before)  # type: ignore[attr-defined]
        return mse

    x0 = np.array([np.log10(A_crit_0)])
    bounds = [tuple(np.log10(A_crit_bounds))]

    res = minimize(_misfit, x0, method=method, bounds=bounds)
    A_crit_star = 10 ** res.x[0]
    mse_star = float(res.fun)

    logger.info("NL A_crit optimisation: A_crit*=%.1f, MSE=%.4e", A_crit_star, mse_star)
    if not return_result:
        return A_crit_star, mse_star
    return OptResult({"A_crit": float(A_crit_star)}, mse_star, int(getattr(res, "nit", -1)), int(res.nfev),
                     traj.n_solves, time.perf_counter() - t0, bool(res.success), str(res.message), traj,
                     [float(v) for v in res.x])


def optimize_threshold_full(
    Z0: np.ndarray,
    A: np.ndarray,
    FD: np.ndarray,
    dx: float,
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
    return_result: bool = False,
    rebuild_per_K: bool = False,
):
    """Jointly optimise (A_crit, D, K).

    v1 rebuilt the partitioned model (channel BFS) for every candidate K; since Phase 3 step 4
    the channel network is integrated once and ``Z_channel = z_anchor + (U/K) chi``
    (``rebuild_per_K=True`` restores the v1 path for comparison).
    """
    traj = Trajectory("joint", ("log10_A_crit", "log10_D", "log10_K"))
    t0 = time.perf_counter()
    chi_model = None if rebuild_per_K else partitioned_threshold_model_chi(Z0, A, FD, dx, m, U)

    @lru_cache(maxsize=None)
    def _solve(log10_A_crit: float, log10_D: float, log10_K: float) -> np.ndarray:
        _solve.calls += 1  # type: ignore[attr-defined]
        K = 10**log10_K
        D = 10**log10_D
        A_crit = 10**log10_A_crit
        if chi_model is not None:
            Z, _ = chi_model(A_crit, D, K)
            return Z
        model_fn = partitioned_threshold_model(Z0, A, FD, dx, m, U, K)
        Z, _ = model_fn(A_crit, D)
        return Z

    _solve.calls = 0  # type: ignore[attr-defined]

    def _misfit(x: np.ndarray) -> float:
        before = _solve.calls  # type: ignore[attr-defined]
        Z_mod = _solve(float(x[0]), float(x[1]), float(x[2]))
        mse = _mse(Z_mod, dem_griddata)
        traj.record({"log10_A_crit": float(x[0]), "log10_D": float(x[1]), "log10_K": float(x[2])}, mse,
                    cached=_solve.calls == before)  # type: ignore[attr-defined]
        return mse

    x0 = np.log10([A_crit_0, D0, K0])
    bounds = [tuple(np.log10(A_crit_bounds)), tuple(np.log10(D_bounds)), tuple(np.log10(K_bounds))]

    res = minimize(_misfit, x0, method=method, bounds=bounds)
    A_crit_star = 10 ** res.x[0]
    D_star = 10 ** res.x[1]
    K_star = 10 ** res.x[2]
    mse_star = float(res.fun)

    logger.info("Full optimisation: A_crit*=%.1f, D*=%.3e, K*=%.3e, MSE=%.4e", A_crit_star, D_star, K_star, mse_star)
    if not return_result:
        return A_crit_star, D_star, K_star, mse_star
    return OptResult({"A_crit": float(A_crit_star), "D": float(D_star), "K": float(K_star)}, mse_star,
                     int(getattr(res, "nit", -1)), int(res.nfev), traj.n_solves, time.perf_counter() - t0,
                     bool(res.success), str(res.message), traj, [float(v) for v in res.x])
