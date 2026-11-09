"""
Tests for D-infinity model factories (src/models_dinf.py).

Uses conftest.py fixtures (MockGrid, synthetic grids).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from conftest import MockGrid, _build_d8_flow, _build_drainage_area
from dinf import compute_dinf_flow_direction, compute_dinf_area
from models_dinf import (
    advection_diffusion_model_dinf,
    partitioned_threshold_model_dinf,
    partitioned_threshold_model_nonlinear_dinf,
)
from models import advection_diffusion_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_setup():
    """Build a complete D8 + D-inf test grid (20×20 tilted plane)."""
    Ny, Nx, dx = 20, 20, 10.0
    np.random.seed(42)
    j_coords, i_coords = np.meshgrid(
        np.arange(Ny), np.arange(Nx), indexing="ij"
    )
    Z = 1000.0 - 5.0 * j_coords + 0.1 * np.random.randn(Ny, Nx)

    fd_grid = _build_d8_flow(Ny, Nx)
    area_grid = _build_drainage_area(Ny, Nx, dx)

    dem = MockGrid(Z, dx=dx)
    area = MockGrid(area_grid, dx=dx)
    fd = MockGrid(fd_grid, dx=dx)

    # Compute D-inf routing
    angles, r_facet, facet_idx = compute_dinf_flow_direction(
        Z, dx, FD_d8=fd_grid,
    )
    A_dinf = compute_dinf_area(Z, dx, angles, r_facet, facet_idx)

    return {
        "dem": dem, "area": area, "fd": fd,
        "angles": angles, "r_facet": r_facet, "facet_idx": facet_idx,
        "A_dinf": A_dinf,
        "Ny": Ny, "Nx": Nx, "dx": dx,
        "Z": Z,
    }


# ---------------------------------------------------------------------------
# AD model tests
# ---------------------------------------------------------------------------

class TestAdvectionDiffusionDinf:
    """Tests for the D-inf advection-diffusion model."""

    def test_output_shape(self, grid_setup):
        gs = grid_setup
        solver = advection_diffusion_model_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        Z_out = solver(1e-4, 1e-5, 1e-2)
        assert Z_out.shape == (gs["Ny"], gs["Nx"])

    def test_output_type(self, grid_setup):
        gs = grid_setup
        solver = advection_diffusion_model_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        Z_out = solver(1e-4, 1e-5, 1e-2)
        assert isinstance(Z_out, np.ndarray)
        assert Z_out.dtype == np.float64 or np.issubdtype(Z_out.dtype, np.floating)

    def test_boundary_conditions(self, grid_setup):
        """Edge cells should match observed DEM."""
        gs = grid_setup
        solver = advection_diffusion_model_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        Z_out = solver(1e-4, 1e-5, 1e-2)
        Z_obs = gs["Z"]

        # Check edges
        for j in range(gs["Ny"]):
            for i in range(gs["Nx"]):
                if j == 0 or j == gs["Ny"] - 1 or i == 0 or i == gs["Nx"] - 1:
                    if np.isfinite(Z_obs[j, i]) and np.isfinite(Z_out[j, i]):
                        np.testing.assert_allclose(
                            Z_out[j, i], Z_obs[j, i], atol=1e-6,
                            err_msg=f"Boundary cell ({j},{i}) should match observed",
                        )

    def test_determinism(self, grid_setup):
        """Same inputs should produce same output."""
        gs = grid_setup
        solver = advection_diffusion_model_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        Z1 = solver(1e-4, 1e-5, 1e-2)
        Z2 = solver(1e-4, 1e-5, 1e-2)
        np.testing.assert_array_equal(Z1, Z2)

    def test_finite_output(self, grid_setup):
        """Output should be finite for valid cells."""
        gs = grid_setup
        solver = advection_diffusion_model_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        Z_out = solver(1e-4, 1e-5, 1e-2)
        valid = np.isfinite(gs["Z"])
        assert np.all(np.isfinite(Z_out[valid])), \
            "All valid cells should have finite elevations"


# ---------------------------------------------------------------------------
# Cardinal-grid equivalence: D-inf ≈ D8 when all flow is cardinal
# ---------------------------------------------------------------------------

class TestCardinalEquivalence:
    """On a grid where all D-inf flow is exactly cardinal (r=0),
    D-inf result should approximate D8 result."""

    def test_south_tilted_plane(self):
        """Plane tilting strictly south → all r=0, facet=6 → matches D8."""
        Ny, Nx, dx = 20, 20, 10.0
        j_coords = np.arange(Ny).reshape(-1, 1) * np.ones((1, Nx))
        Z = 1000.0 - 5.0 * j_coords

        fd_grid = np.full((Ny, Nx), 4.0)  # all flow south (D8 code 4)
        area_grid = _build_drainage_area(Ny, Nx, dx)

        dem = MockGrid(Z, dx=dx)
        area = MockGrid(area_grid, dx=dx)
        fd = MockGrid(fd_grid, dx=dx)

        # D8 model
        solver_d8 = advection_diffusion_model(dem, area, fd, 0.5)
        Z_d8 = solver_d8(1e-4, 1e-5, 1e-2)

        # D-inf model
        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx, FD_d8=fd_grid)
        A_dinf = compute_dinf_area(Z, dx, angles, r_facet, facet_idx)

        solver_dinf = advection_diffusion_model_dinf(
            dem, area, fd, 0.5, angles, r_facet, facet_idx, A_dinf,
        )
        Z_dinf = solver_dinf(1e-4, 1e-5, 1e-2)

        # Interior cells should be similar (not exact due to area differences)
        valid = np.isfinite(Z_d8) & np.isfinite(Z_dinf)
        diff = np.abs(Z_d8[valid] - Z_dinf[valid])
        # Allow some tolerance for area distribution differences
        assert np.median(diff) < 5.0, \
            f"Median difference too large: {np.median(diff):.2f}"


# ---------------------------------------------------------------------------
# Partitioned threshold model tests
# ---------------------------------------------------------------------------

class TestPartitionedThresholdDinf:
    """Tests for the D-inf partitioned threshold model."""

    def test_output_shape(self, grid_setup):
        gs = grid_setup
        model = partitioned_threshold_model_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5, 1e-4, 1e-5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        Z_out, mask = model(500.0, 1e-2)
        assert Z_out.shape == (gs["Ny"], gs["Nx"])
        assert mask.shape == (gs["Ny"], gs["Nx"])
        assert mask.dtype == bool

    def test_determinism(self, grid_setup):
        gs = grid_setup
        model = partitioned_threshold_model_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5, 1e-4, 1e-5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        Z1, m1 = model(500.0, 1e-2)
        Z2, m2 = model(500.0, 1e-2)
        np.testing.assert_array_equal(Z1, Z2)
        np.testing.assert_array_equal(m1, m2)

    def test_channel_mask_nonempty(self, grid_setup):
        gs = grid_setup
        model = partitioned_threshold_model_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5, 1e-4, 1e-5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        _, mask = model(100.0, 1e-2)  # low threshold → many channels
        assert np.any(mask), "Channel mask should not be empty"


# ---------------------------------------------------------------------------
# Nonlinear threshold model tests
# ---------------------------------------------------------------------------

class TestPartitionedThresholdNonlinearDinf:
    """Tests for the D-inf nonlinear partitioned threshold model."""

    def test_output_shape(self, grid_setup):
        gs = grid_setup
        model = partitioned_threshold_model_nonlinear_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5, 1e-4, 1e-5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        Z_out, mask = model(500.0, 1e-2, 1.2)
        assert Z_out.shape == (gs["Ny"], gs["Nx"])
        assert mask.shape == (gs["Ny"], gs["Nx"])

    def test_determinism(self, grid_setup):
        gs = grid_setup
        model = partitioned_threshold_model_nonlinear_dinf(
            gs["dem"], gs["area"], gs["fd"], 0.5, 1e-4, 1e-5,
            gs["angles"], gs["r_facet"], gs["facet_idx"], gs["A_dinf"],
        )
        Z1, _ = model(500.0, 1e-2, 1.2)
        Z2, _ = model(500.0, 1e-2, 1.2)
        np.testing.assert_array_equal(Z1, Z2)
