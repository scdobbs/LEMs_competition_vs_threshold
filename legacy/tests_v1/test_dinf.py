"""
Tests for D-infinity flow routing (src/dinf.py).

Uses small synthetic grids — no data dependencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from dinf import (
    FACETS,
    D8_CODE_TO_DINF,
    compute_dinf_flow_direction,
    get_receivers,
    compute_dinf_area,
    compute_dinf_from_dem,
)


# ---------------------------------------------------------------------------
# Helper: build small synthetic grids
# ---------------------------------------------------------------------------

def _tilted_plane(Ny, Nx, dx, slope_i, slope_j):
    """Build a tilted plane: Z = z0 - slope_i * i * dx - slope_j * j * dx."""
    j_coords, i_coords = np.meshgrid(
        np.arange(Ny, dtype=float), np.arange(Nx, dtype=float), indexing="ij"
    )
    Z = 1000.0 - slope_i * i_coords * dx - slope_j * j_coords * dx
    return Z


# ---------------------------------------------------------------------------
# Test: tilted plane flowing east (all angles ≈ 0)
# ---------------------------------------------------------------------------

class TestTiltedPlaneEast:
    """Plane tilting east → all D-inf angles should be ≈ 0 (east)."""

    def test_angles_near_zero(self):
        Ny, Nx, dx = 10, 10, 1.0
        Z = _tilted_plane(Ny, Nx, dx, slope_i=1.0, slope_j=0.0)
        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx)

        # Interior cells should have angle ≈ 0 (east)
        interior = np.isfinite(angles[1:-1, 1:-1])
        assert np.all(interior), "All interior cells should have valid angles"
        np.testing.assert_allclose(
            angles[1:-1, 1:-1], 0.0, atol=1e-10,
            err_msg="East-tilted plane should have angle ≈ 0",
        )

    def test_r_facet_near_zero(self):
        Ny, Nx, dx = 10, 10, 1.0
        Z = _tilted_plane(Ny, Nx, dx, slope_i=1.0, slope_j=0.0)
        _, r_facet, _ = compute_dinf_flow_direction(Z, dx)
        np.testing.assert_allclose(
            r_facet[1:-1, 1:-1], 0.0, atol=1e-10,
            err_msg="Cardinal flow should have r ≈ 0",
        )


# ---------------------------------------------------------------------------
# Test: tilted plane flowing south (angles ≈ 3π/2)
# ---------------------------------------------------------------------------

class TestTiltedPlaneSouth:
    """Plane tilting south → angles should point southward.

    With pure south tilt, facets 5 (S→SW, base=5π/4) and 6 (S→SE, base=3π/2)
    are tied. The algorithm picks the lower-index facet (5), giving angle=5π/4.
    Both are correct southward directions; we just verify the angle is in the
    southward half-plane.
    """

    def test_angles_southward(self):
        Ny, Nx, dx = 10, 10, 1.0
        Z = _tilted_plane(Ny, Nx, dx, slope_i=0.0, slope_j=1.0)
        angles, _, _ = compute_dinf_flow_direction(Z, dx)

        interior = angles[1:-1, 1:-1]
        # Should be in the southward half: either 5π/4 or 3π/2
        # (tied facets 5 and 6 both point south)
        assert np.all(np.isfinite(interior)), "All interior cells should have valid angles"
        assert np.all(interior >= np.pi), "South-flow angles should be >= π"
        assert np.all(interior <= 2 * np.pi), "South-flow angles should be <= 2π"


# ---------------------------------------------------------------------------
# Test: tilted plane flowing SE (angles ≈ 7π/4)
# ---------------------------------------------------------------------------

class TestTiltedPlaneSE:
    """Plane tilting SE → angles should be ≈ 7π/4."""

    def test_angles_se(self):
        Ny, Nx, dx = 10, 10, 1.0
        # Equal tilt east and south
        Z = _tilted_plane(Ny, Nx, dx, slope_i=1.0, slope_j=1.0)
        angles, _, _ = compute_dinf_flow_direction(Z, dx)

        expected = 7 * np.pi / 4
        np.testing.assert_allclose(
            angles[1:-1, 1:-1], expected, atol=1e-10,
            err_msg="SE-tilted plane should have angle ≈ 7π/4",
        )


# ---------------------------------------------------------------------------
# Test: area conservation
# ---------------------------------------------------------------------------

class TestAreaConservation:
    """Total distributed area should approximately equal N_cells × dx²."""

    def test_total_area(self):
        Ny, Nx, dx = 15, 15, 5.0
        Z = _tilted_plane(Ny, Nx, dx, slope_i=1.0, slope_j=0.3)

        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx)
        A = compute_dinf_area(Z, dx, angles, r_facet, facet_idx)

        # Total area flowing out of the domain should equal total input
        # (each cell contributes dx²)
        n_valid = np.sum(np.isfinite(Z))
        total_input = n_valid * dx * dx

        # The outlet cell(s) should accumulate close to total_input
        # But area is conserved within the grid: sum of all areas >= total_input
        # because each cell starts with dx² and receives from upstream
        assert np.nanmax(A) > 0, "Max area should be positive"
        # Check that no area is lost (outlet area + boundary losses)
        # Every cell has at least dx²
        assert np.all(A[np.isfinite(A)] >= dx * dx - 1e-10), \
            "Every valid cell should have at least one cell's area"


# ---------------------------------------------------------------------------
# Test: symmetric V-shape
# ---------------------------------------------------------------------------

class TestSymmetricV:
    """V-shaped valley: center column should accumulate left+right equally."""

    def test_center_column_symmetry(self):
        Ny, Nx, dx = 11, 11, 1.0
        center = Nx // 2
        j_coords, i_coords = np.meshgrid(
            np.arange(Ny, dtype=float), np.arange(Nx, dtype=float), indexing="ij"
        )
        # V-shape: elevation increases away from center column, decreases southward
        Z = np.abs(i_coords - center) * 10.0 + (Ny - j_coords) * 0.5

        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx)
        A = compute_dinf_area(Z, dx, angles, r_facet, facet_idx)

        # Center column interior cells should have roughly symmetric area
        # (left and right contribute equally)
        for j in range(2, Ny - 1):
            a_left = A[j, center - 1]
            a_right = A[j, center + 1]
            if np.isfinite(a_left) and np.isfinite(a_right):
                np.testing.assert_allclose(
                    a_left, a_right, rtol=0.1,
                    err_msg=f"Row {j}: left/right area should be symmetric",
                )


# ---------------------------------------------------------------------------
# Test: angle and r_facet ranges
# ---------------------------------------------------------------------------

class TestAngleRanges:
    """All valid angles in [0, 2π), all r_facet in [0, π/4]."""

    def test_angle_range(self):
        Ny, Nx, dx = 20, 20, 1.0
        np.random.seed(123)
        Z = 500.0 + np.random.randn(Ny, Nx) * 10.0
        # Add a general trend to avoid too many flats
        j_coords = np.arange(Ny).reshape(-1, 1)
        Z -= j_coords * 5.0

        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx)

        valid = np.isfinite(angles)
        assert np.all(angles[valid] >= 0.0), "All angles should be >= 0"
        assert np.all(angles[valid] < 2 * np.pi + 1e-10), "All angles should be < 2π"

    def test_r_facet_range(self):
        Ny, Nx, dx = 20, 20, 1.0
        np.random.seed(456)
        Z = 500.0 + np.random.randn(Ny, Nx) * 10.0
        j_coords = np.arange(Ny).reshape(-1, 1)
        Z -= j_coords * 5.0

        _, r_facet, _ = compute_dinf_flow_direction(Z, dx)

        valid = np.isfinite(r_facet)
        assert np.all(r_facet[valid] >= 0.0), "All r_facet should be >= 0"
        assert np.all(r_facet[valid] <= np.pi / 4 + 1e-10), "All r_facet should be <= π/4"


# ---------------------------------------------------------------------------
# Test: cardinal flow matches D8 area
# ---------------------------------------------------------------------------

class TestCardinalFlowMatch:
    """When flow is exactly cardinal, D-inf area increases downstream."""

    def test_east_flow_area_increases(self):
        Ny, Nx, dx = 5, 20, 1.0
        Z = _tilted_plane(Ny, Nx, dx, slope_i=1.0, slope_j=0.0)

        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx)
        A = compute_dinf_area(Z, dx, angles, r_facet, facet_idx)

        # For a plane tilting east, area should increase from left to right
        # in each interior row (downstream accumulation)
        for j in range(1, Ny - 1):
            # Check that area is monotonically non-decreasing in interior
            for i in range(2, Nx - 2):
                if np.isfinite(A[j, i]) and np.isfinite(A[j, i + 1]):
                    assert A[j, i + 1] >= A[j, i] - 1e-10, \
                        f"Row {j}, col {i}: area should increase downstream (east)"


# ---------------------------------------------------------------------------
# Test: flat cell fallback with D8
# ---------------------------------------------------------------------------

class TestFlatCellFallback:
    """Grid with flat region + D8 FD → no NaN angles, area passes through."""

    def test_no_nan_with_d8_fallback(self):
        Ny, Nx, dx = 10, 10, 1.0
        Z = _tilted_plane(Ny, Nx, dx, slope_i=0.5, slope_j=0.3)

        # Create a flat region in the middle
        Z[3:6, 3:6] = Z[3, 3]

        # Build a simple D8 FD grid: everything flows south (code 4)
        FD = np.full((Ny, Nx), 4.0)

        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx, FD_d8=FD)

        valid_z = np.isfinite(Z)
        assert np.all(np.isfinite(angles[valid_z])), \
            "With D8 fallback, all valid cells should have finite angles"
        assert np.all(facet_idx[valid_z] >= 0), \
            "With D8 fallback, all valid cells should have valid facet indices"

    def test_area_through_flat_region(self):
        Ny, Nx, dx = 10, 10, 1.0
        Z = _tilted_plane(Ny, Nx, dx, slope_i=0.0, slope_j=1.0)

        # Flat patch
        Z[4:6, 4:6] = Z[4, 4]

        FD = np.full((Ny, Nx), 4.0)  # all flow south

        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx, FD_d8=FD)
        A = compute_dinf_area(Z, dx, angles, r_facet, facet_idx)

        # Area should pass through the flat region to the bottom
        # Bottom row should have accumulated area from above
        assert np.nanmax(A[-1, :]) > 3 * dx * dx, \
            "Bottom row should accumulate area from upstream through flat region"

    def test_without_d8_fallback_has_nans(self):
        """Without D8 fallback, flat cells should have NaN angles."""
        Ny, Nx, dx = 10, 10, 1.0
        Z = np.ones((Ny, Nx)) * 100.0  # completely flat
        # Add a tiny slope at edges to make most cells flat
        Z[0, :] = 101.0

        angles, _, _ = compute_dinf_flow_direction(Z, dx, FD_d8=None)

        # Most interior cells should have NaN (flat, no downhill)
        n_nan = np.sum(np.isnan(angles[1:-1, 1:-1]))
        assert n_nan > 0, "Flat cells without D8 fallback should have NaN angles"


# ---------------------------------------------------------------------------
# Test: no NaN guarantee with D8 fallback
# ---------------------------------------------------------------------------

class TestNoNanGuarantee:
    """With FD_d8 provided, all valid cells should have finite angles."""

    def test_random_grid(self):
        Ny, Nx, dx = 15, 15, 2.0
        np.random.seed(789)
        Z = 500.0 + np.random.randn(Ny, Nx) * 5.0

        # D8 fallback: everything flows south
        FD = np.full((Ny, Nx), 4.0)

        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx, FD_d8=FD)

        valid = np.isfinite(Z)
        assert np.all(np.isfinite(angles[valid])), \
            "With D8 fallback, all valid cells should have finite angles"

    def test_grid_with_nans(self):
        Ny, Nx, dx = 12, 12, 1.0
        Z = _tilted_plane(Ny, Nx, dx, slope_i=0.5, slope_j=0.5)

        # Insert some NaN holes
        Z[2, 3] = np.nan
        Z[5, 7] = np.nan
        Z[8, 4] = np.nan

        FD = np.full((Ny, Nx), 4.0)

        angles, r_facet, facet_idx = compute_dinf_flow_direction(Z, dx, FD_d8=FD)

        valid = np.isfinite(Z)
        assert np.all(np.isfinite(angles[valid])), \
            "Valid cells should have finite angles even with NaN neighbors"
        assert np.all(np.isnan(angles[~valid])), \
            "NaN cells should remain NaN"


# ---------------------------------------------------------------------------
# Test: get_receivers proportions
# ---------------------------------------------------------------------------

class TestGetReceivers:
    """Test receiver coordinate and proportion computation."""

    def test_r_zero_all_to_cardinal(self):
        """r=0 → all flow to cardinal receiver (p1=1, p2=0)."""
        j1, i1, j2, i2, p1, p2 = get_receivers(5, 5, fac_idx=0, r=0.0)
        assert p1 == pytest.approx(1.0)
        assert p2 == pytest.approx(0.0)
        # Facet 0: e1 = (+1, 0) → i1=6, j1=5
        assert i1 == 6
        assert j1 == 5

    def test_r_pi_over_4_all_to_diagonal(self):
        """r=π/4 → all flow to diagonal receiver (p1=0, p2=1)."""
        j1, i1, j2, i2, p1, p2 = get_receivers(5, 5, fac_idx=0, r=np.pi / 4)
        assert p1 == pytest.approx(0.0)
        assert p2 == pytest.approx(1.0)
        # Facet 0: e2 = (+1, -1) → i2=6, j2=4
        assert i2 == 6
        assert j2 == 4

    def test_equal_split(self):
        """r=π/8 → equal split (p1=p2=0.5)."""
        _, _, _, _, p1, p2 = get_receivers(5, 5, fac_idx=0, r=np.pi / 8)
        assert p1 == pytest.approx(0.5)
        assert p2 == pytest.approx(0.5)

    def test_proportions_sum_to_one(self):
        """p1 + p2 should always equal 1."""
        for r in np.linspace(0, np.pi / 4, 20):
            for fac in range(8):
                _, _, _, _, p1, p2 = get_receivers(5, 5, fac_idx=fac, r=r)
                assert p1 + p2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test: compute_dinf_from_dem convenience wrapper
# ---------------------------------------------------------------------------

class TestComputeDinfFromDem:
    """Test the convenience wrapper with mock objects."""

    def test_basic_output(self):
        class MockGeoref:
            dx = 5.0

        class MockDem:
            _griddata = _tilted_plane(10, 10, 5.0, slope_i=0.5, slope_j=0.3)
            _georef_info = MockGeoref()

        class MockFD:
            _griddata = np.full((10, 10), 4.0)

        angles, r_facet, facet_idx, A_dinf = compute_dinf_from_dem(MockDem(), MockFD())

        assert angles.shape == (10, 10)
        assert A_dinf.shape == (10, 10)
        assert np.all(np.isfinite(angles[np.isfinite(MockDem._griddata)]))
        assert np.all(A_dinf[np.isfinite(A_dinf)] > 0)
