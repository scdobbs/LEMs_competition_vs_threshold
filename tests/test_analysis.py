"""Tests for src/analysis.py — regression on synthetic power-law data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import compute_iqr_errors, weighted_r2


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
