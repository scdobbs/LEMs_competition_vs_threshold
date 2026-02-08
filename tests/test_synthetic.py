"""Tests for src/synthetic.py — planform grid and solver verification."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synthetic import (
    compute_area_planform,
    downstream_link,
    solve_advection_diffusion_planform,
    inlet_area_from_Pe,
    channel_slope_centerline,
    A_top_for_target_inlet_area,
    D_from_Pe,
    Pe_from_D_Ain,
)


# ---------------------------------------------------------------------------
# Tests: compute_area_planform
# ---------------------------------------------------------------------------

class TestComputeAreaPlanform:

    def test_shape(self):
        A = compute_area_planform(11, 7, 10.0, 10.0)
        assert A.shape == (7, 11)

    def test_requires_odd_Nx(self):
        with pytest.raises(ValueError, match="odd"):
            compute_area_planform(10, 7, 10.0, 10.0)

    def test_positive_area(self):
        A = compute_area_planform(11, 7, 10.0, 10.0)
        assert np.all(A >= 0)

    def test_centre_column_increases_downstream(self):
        Nx, Ny = 11, 7
        A = compute_area_planform(Nx, Ny, 10.0, 10.0)
        ci = (Nx - 1) // 2
        # Centre column area should increase with j
        for j in range(1, Ny):
            assert A[j, ci] >= A[j - 1, ci]

    def test_area_with_A_top(self):
        Nx, Ny = 11, 7
        A0 = compute_area_planform(Nx, Ny, 10.0, 10.0, A_top=0.0)
        A1 = compute_area_planform(Nx, Ny, 10.0, 10.0, A_top=1000.0)
        ci = (Nx - 1) // 2
        # With A_top, centre column should have more area
        assert A1[0, ci] > A0[0, ci]


# ---------------------------------------------------------------------------
# Tests: downstream_link
# ---------------------------------------------------------------------------

class TestDownstreamLink:

    def test_shape(self):
        i_dn, j_dn, is_outlet = downstream_link(11, 7)
        assert i_dn.shape == (7, 11)
        assert j_dn.shape == (7, 11)
        assert is_outlet.shape == (7, 11)

    def test_single_outlet(self):
        _, _, is_outlet = downstream_link(11, 7)
        assert np.sum(is_outlet) == 1

    def test_outlet_at_bottom_centre(self):
        Nx, Ny = 11, 7
        _, _, is_outlet = downstream_link(Nx, Ny)
        ci = (Nx - 1) // 2
        assert is_outlet[Ny - 1, ci]


# ---------------------------------------------------------------------------
# Tests: solve_advection_diffusion_planform
# ---------------------------------------------------------------------------

class TestSolveAdvecDiffPlanform:

    def test_runs_without_error(self):
        Z, A, dx, dy = solve_advection_diffusion_planform(
            U=1e-4, K=1e-4, m=0.35, D=1e-3,
            Nx=11, Ny=7, Lx=100.0, Ly=60.0,
        )
        assert Z.shape == (7, 11)
        assert A.shape == (7, 11)

    def test_outlet_elevation_zero(self):
        Z, A, dx, dy = solve_advection_diffusion_planform(
            U=1e-4, K=1e-4, m=0.35, D=1e-3,
            Nx=11, Ny=7, Lx=100.0, Ly=60.0,
        )
        ci = (11 - 1) // 2
        assert abs(Z[6, ci]) < 1e-10

    def test_requires_odd_Nx(self):
        with pytest.raises(ValueError):
            solve_advection_diffusion_planform(
                U=1e-4, K=1e-4, m=0.35, D=1e-3,
                Nx=10, Ny=7, Lx=100.0, Ly=60.0,
            )

    def test_requires_min_size(self):
        with pytest.raises(ValueError):
            solve_advection_diffusion_planform(
                U=1e-4, K=1e-4, m=0.35, D=1e-3,
                Nx=3, Ny=3, Lx=20.0, Ly=20.0,
            )

    def test_positive_interior_elevation(self):
        Z, _, _, _ = solve_advection_diffusion_planform(
            U=1e-4, K=1e-4, m=0.35, D=1e-3,
            Nx=11, Ny=7, Lx=100.0, Ly=60.0,
        )
        # Interior should have positive elevation (uplifted above outlet)
        assert np.all(Z[:-1, :] > -1e-6)


# ---------------------------------------------------------------------------
# Tests: Peclet-number utilities
# ---------------------------------------------------------------------------

class TestPecletUtilities:

    def test_inlet_area_from_Pe(self):
        A = inlet_area_from_Pe(Pe=100.0, D=1e-3, K=1e-4, L_PE=50.0, m=0.35)
        assert isinstance(A, float)
        assert A > 0

    def test_D_from_Pe(self):
        D = D_from_Pe(Pe=100.0, A_inlet=1000.0, K=1e-4, L_PE=50.0, m=0.35)
        assert isinstance(D, float)
        assert D > 0

    def test_Pe_round_trip(self):
        """Pe → A_in → Pe should round-trip."""
        D, K, L, m = 1e-3, 1e-4, 50.0, 0.35
        Pe_orig = 100.0
        A_in = inlet_area_from_Pe(Pe_orig, D, K, L, m)
        Pe_back = Pe_from_D_Ain(D, A_in, K, L, m)
        np.testing.assert_allclose(Pe_back, Pe_orig, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: channel_slope_centerline
# ---------------------------------------------------------------------------

class TestChannelSlopeCenterline:

    def test_length(self):
        Z = np.random.RandomState(0).rand(7, 11)
        S = channel_slope_centerline(Z, 11, 7, 10.0, 10.0)
        assert len(S) == 7
        assert np.isnan(S[-1])

    def test_tilted_plane(self):
        Nx, Ny = 11, 7
        j, i = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
        dy = 10.0
        Z = 100.0 - 2.0 * j * dy  # slope = 2.0 in y direction
        S = channel_slope_centerline(Z, Nx, Ny, 10.0, dy)
        # Expected slope = 2.0 for all but the last row
        np.testing.assert_allclose(S[:-1], 2.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: A_top_for_target_inlet_area
# ---------------------------------------------------------------------------

class TestAtopForTargetInletArea:

    def test_non_negative(self):
        A_top = A_top_for_target_inlet_area(11, 7, 100.0, 60.0, A_inlet_target=5000.0)
        assert A_top >= 0

    def test_zero_when_target_small(self):
        # If target is smaller than left+right contrib, A_top should be 0
        A_top = A_top_for_target_inlet_area(11, 7, 100.0, 60.0, A_inlet_target=0.0)
        assert A_top == 0.0
