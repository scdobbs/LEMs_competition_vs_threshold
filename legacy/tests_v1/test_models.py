"""Tests for src/models.py — locks down current behaviour before refactoring."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models import (
    advection_diffusion_model,
    partitioned_threshold_model,
    partitioned_threshold_model_nonlinear,
    _find_boundary_value,
)


# ── advection_diffusion_model ───────────────────────────────────────────────

class TestAdvectionDiffusionModel:

    def test_returns_correct_shape(self, synthetic_dem, synthetic_area,
                                    synthetic_flow_direction, model_params):
        solver = advection_diffusion_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"],
        )
        Z = solver(model_params["U"], model_params["K"], model_params["D"])
        assert Z.shape == synthetic_dem._griddata.shape

    def test_boundary_conditions(self, synthetic_dem, synthetic_area,
                                  synthetic_flow_direction, model_params):
        """Boundary cells should match the input DEM."""
        solver = advection_diffusion_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"],
        )
        Z = solver(model_params["U"], model_params["K"], model_params["D"])
        Z0 = synthetic_dem._griddata
        Ny, Nx = Z0.shape

        # Check edges — boundary cells with valid data should be preserved
        for j in range(Ny):
            for i in range(Nx):
                if j == 0 or j == Ny - 1 or i == 0 or i == Nx - 1:
                    if np.isfinite(Z0[j, i]) and np.isfinite(Z[j, i]):
                        np.testing.assert_allclose(Z[j, i], Z0[j, i], rtol=1e-10)

    def test_deterministic(self, synthetic_dem, synthetic_area,
                            synthetic_flow_direction, model_params):
        """Same inputs must produce the same output."""
        solver = advection_diffusion_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"],
        )
        Z1 = solver(model_params["U"], model_params["K"], model_params["D"])
        Z2 = solver(model_params["U"], model_params["K"], model_params["D"])
        np.testing.assert_array_equal(Z1, Z2)

    def test_positive_elevation(self, synthetic_dem, synthetic_area,
                                 synthetic_flow_direction, model_params):
        """Elevation should remain physically reasonable (no huge negatives)."""
        solver = advection_diffusion_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"],
        )
        Z = solver(model_params["U"], model_params["K"], model_params["D"])
        # All finite values should be > -1000 (we start around 900-1000)
        finite = Z[np.isfinite(Z)]
        assert np.all(finite > -1000), f"Min elevation: {finite.min()}"


# ── partitioned_threshold_model ─────────────────────────────────────────────

class TestPartitionedThresholdModel:

    def test_returns_tuple(self, synthetic_dem, synthetic_area,
                            synthetic_flow_direction, model_params):
        f = partitioned_threshold_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        result = f(model_params["A_crit"], model_params["D"])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mask_is_boolean(self, synthetic_dem, synthetic_area,
                              synthetic_flow_direction, model_params):
        f = partitioned_threshold_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        Z, mask = f(model_params["A_crit"], model_params["D"])
        assert mask.dtype == bool

    def test_mask_shape(self, synthetic_dem, synthetic_area,
                         synthetic_flow_direction, model_params):
        f = partitioned_threshold_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        Z, mask = f(model_params["A_crit"], model_params["D"])
        assert mask.shape == synthetic_dem._griddata.shape

    def test_elevation_shape(self, synthetic_dem, synthetic_area,
                              synthetic_flow_direction, model_params):
        f = partitioned_threshold_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        Z, mask = f(model_params["A_crit"], model_params["D"])
        assert Z.shape == synthetic_dem._griddata.shape


# ── partitioned_threshold_model_nonlinear ───────────────────────────────────

class TestPartitionedNonlinearModel:

    def test_returns_tuple(self, synthetic_dem, synthetic_area,
                            synthetic_flow_direction, model_params):
        f = partitioned_threshold_model_nonlinear(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        result = f(model_params["A_crit"], model_params["D"], model_params["S_c"])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_converges(self, synthetic_dem, synthetic_area,
                        synthetic_flow_direction, model_params):
        """Should complete without raising for reasonable inputs."""
        f = partitioned_threshold_model_nonlinear(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        Z, mask = f(model_params["A_crit"], model_params["D"],
                     model_params["S_c"], max_iter=10, tol=1e-4)
        assert Z.shape == synthetic_dem._griddata.shape
        assert mask.dtype == bool


# ── _find_boundary_value ────────────────────────────────────────────────────

class TestFindBoundaryValue:

    def test_valid_boundary(self):
        """Should return the boundary elevation for a valid edge."""
        Ny, Nx = 10, 10
        Z0 = np.full((Ny, Nx), np.nan)
        Z0[0, :] = 100.0   # top edge valid
        Z0[-1, :] = 50.0   # bottom edge valid
        Z0[:, 0] = 75.0    # left edge valid
        Z0[:, -1] = 80.0   # right edge valid

        # Interior cell — should find left edge value for that row
        val = _find_boundary_value(Z0, 5, 5, Nx, Ny)
        assert np.isfinite(val)

    def test_nan_boundary(self):
        """When all edges are NaN, should return NaN gracefully."""
        Ny, Nx = 10, 10
        Z0 = np.full((Ny, Nx), np.nan)
        val = _find_boundary_value(Z0, 5, 5, Nx, Ny)
        assert np.isnan(val)
