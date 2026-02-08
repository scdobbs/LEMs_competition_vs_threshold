"""Tests for src/analysis.py — regression on synthetic power-law data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import (
    compute_iqr_errors,
    weighted_r2,
    fit_slope_area,
    ks_local_from_SA,
    ks_from_loglog_fixed_theta,
)


# ---------------------------------------------------------------------------
# Tests: weighted_r2
# ---------------------------------------------------------------------------

class TestWeightedR2:

    def test_perfect_fit(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        yhat = y.copy()
        w = np.ones_like(x)
        r2 = weighted_r2(x, y, yhat, w)
        assert abs(r2 - 1.0) < 1e-10

    def test_zero_fit(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        yhat = np.full_like(y, np.mean(y))  # predicting mean → R²=0
        w = np.ones_like(x)
        r2 = weighted_r2(x, y, yhat, w)
        assert abs(r2) < 1e-10

    def test_weighted_gives_different_result(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 100.0])
        yhat = np.array([1.0, 2.0, 3.0, 4.0])
        w_uniform = np.ones(4)
        w_heavy = np.array([1.0, 1.0, 1.0, 100.0])
        r2_uniform = weighted_r2(x, y, yhat, w_uniform)
        r2_heavy = weighted_r2(x, y, yhat, w_heavy)
        # With heavy weight on the outlier, R² should be lower
        assert r2_heavy < r2_uniform


# ---------------------------------------------------------------------------
# Tests: compute_iqr_errors
# ---------------------------------------------------------------------------

class TestComputeIqrErrors:

    def test_returns_arrays(self):
        dbg = {
            "A": np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float),
            "S": np.array([0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
            "bin_logA": np.array([np.log10(300), np.log10(700)]),
            "bin_logS": np.array([np.log10(0.08), np.log10(0.04)]),
            "bin_counts": np.array([5, 5], dtype=float),
        }
        lo, hi = compute_iqr_errors(dbg)
        assert isinstance(lo, np.ndarray)
        assert isinstance(hi, np.ndarray)
        assert len(lo) == 2
        assert len(hi) == 2

    def test_errors_non_negative(self):
        np.random.seed(42)
        A = np.random.uniform(100, 10000, 200)
        S = 0.01 * (A / 1000) ** -0.5 + np.random.uniform(0, 0.005, 200)
        logA = np.log10(A)
        bin_centers = np.array([2.5, 3.0, 3.5])
        dbg = {
            "A": A,
            "S": S,
            "bin_logA": bin_centers,
            "bin_logS": np.array([-2.0, -2.1, -2.2]),
            "bin_counts": np.array([50, 80, 70], dtype=float),
        }
        lo, hi = compute_iqr_errors(dbg)
        assert np.all(lo >= 0)
        assert np.all(hi >= 0)


# ---------------------------------------------------------------------------
# Tests: fit_slope_area
# ---------------------------------------------------------------------------

class TestFitSlopeArea:

    def test_fixed_theta_returns_correct_keys(self):
        A = np.logspace(3, 6, 50)
        S = 0.05 * A ** -0.4
        result = fit_slope_area(A, S, theta_fixed=0.4)
        assert "ks" in result
        assert "theta" in result
        assert "r2" in result
        assert np.isfinite(result["ks"])
        assert result["ks"] > 0

    def test_free_theta_recovers_known(self):
        theta_true = 0.45
        ks_true = 100.0
        A = np.logspace(3, 7, 100)
        S = ks_true * A ** -theta_true
        result = fit_slope_area(A, S, theta_fixed=None)
        np.testing.assert_allclose(result["theta"], theta_true, atol=0.01)
        np.testing.assert_allclose(result["ks"], ks_true, rtol=0.05)

    def test_insufficient_data(self):
        A = np.array([100.0, 200.0])
        S = np.array([0.1, 0.05])
        result = fit_slope_area(A, S, min_points=5)
        assert np.isnan(result["ks"])
        assert result["n_good"] == 2


# ---------------------------------------------------------------------------
# Tests: ks_local_from_SA
# ---------------------------------------------------------------------------

class TestKsLocalFromSA:

    def test_returns_correct_structure(self):
        A = np.logspace(3, 6, 20)
        S = 0.1 * A ** -0.4
        ks_local, mask, n_good, med, q25, q75 = ks_local_from_SA(A, S, m=0.4)
        assert len(ks_local) == len(A)
        assert isinstance(n_good, int)
        assert n_good > 0
        assert np.isfinite(med)
        assert q25 <= med <= q75


# ---------------------------------------------------------------------------
# Tests: ks_from_loglog_fixed_theta
# ---------------------------------------------------------------------------

class TestKsFromLoglogFixedTheta:

    def test_recovers_known_ks(self):
        ks_true = 50.0
        theta = 0.45
        A = np.logspace(4, 7, 200)
        S = ks_true * A ** -theta
        Ks_est, theta_out, r2 = ks_from_loglog_fixed_theta(A, S, theta_fixed=theta)
        np.testing.assert_allclose(Ks_est, ks_true, rtol=0.01)
        assert abs(theta_out - theta) < 1e-10
        assert r2 > 0.99
