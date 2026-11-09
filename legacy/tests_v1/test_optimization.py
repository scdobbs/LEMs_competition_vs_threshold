"""Tests for src/optimization.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from optimization import (
    optimize_advection_diffusion,
    optimize_threshold_acrit,
    optimize_threshold_acrit_nonlinear,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny model stubs that return simple elevations
# ---------------------------------------------------------------------------

@pytest.fixture
def observed_dem():
    """20×20 'observed' elevation grid."""
    np.random.seed(99)
    j, i = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    return 500.0 - 3.0 * j + 0.5 * np.random.randn(20, 20)


@pytest.fixture
def ad_solver(observed_dem):
    """A dummy solver that returns elevation close to observed for reasonable params."""

    def solver(U, K, D):
        # Returns a tilted plane perturbed by K and D
        j, i = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        return 500.0 - 3.0 * j + 0.05 * np.log10(max(K, 1e-15)) + 0.01 * np.log10(max(D, 1e-15))

    return solver


@pytest.fixture
def threshold_solver(observed_dem):
    """A dummy partitioned solver."""

    def solver(A_crit, D):
        j, i = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        Z = 500.0 - 3.0 * j + 0.1 * np.log10(max(A_crit, 1.0))
        mask = np.ones((20, 20), dtype=bool)
        return Z, mask

    return solver


@pytest.fixture
def threshold_nl_solver(observed_dem):
    """A dummy nonlinear partitioned solver."""

    def solver(A_crit, D, Sc):
        j, i = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        Z = 500.0 - 3.0 * j + 0.1 * np.log10(max(A_crit, 1.0))
        mask = np.ones((20, 20), dtype=bool)
        return Z, mask

    return solver


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOptimizeAdvectionDiffusion:

    def test_returns_correct_types(self, ad_solver, observed_dem):
        K_star, D_star, mse = optimize_advection_diffusion(
            ad_solver, observed_dem, U=1e-4, K0=1e-5, D0=1e-2,
        )
        assert isinstance(K_star, float)
        assert isinstance(D_star, float)
        assert isinstance(mse, float)
        assert K_star > 0
        assert D_star > 0
        assert mse >= 0

    def test_mse_is_finite(self, ad_solver, observed_dem):
        _, _, mse = optimize_advection_diffusion(
            ad_solver, observed_dem, U=1e-4, K0=1e-5, D0=1e-2,
        )
        assert np.isfinite(mse)


class TestOptimizeThresholdAcrit:

    def test_returns_correct_types(self, threshold_solver, observed_dem):
        A_crit_star, mse = optimize_threshold_acrit(
            threshold_solver, observed_dem, D0=1e-2,
        )
        assert isinstance(A_crit_star, float)
        assert isinstance(mse, float)
        assert A_crit_star > 0
        assert mse >= 0

    def test_mse_is_finite(self, threshold_solver, observed_dem):
        _, mse = optimize_threshold_acrit(
            threshold_solver, observed_dem, D0=1e-2,
        )
        assert np.isfinite(mse)


class TestOptimizeThresholdNonlinear:

    def test_returns_correct_types(self, threshold_nl_solver, observed_dem):
        A_crit_star, mse = optimize_threshold_acrit_nonlinear(
            threshold_nl_solver, observed_dem, D0=1e-2, Sc=1.2,
        )
        assert isinstance(A_crit_star, float)
        assert isinstance(mse, float)
        assert A_crit_star > 0
        assert mse >= 0
