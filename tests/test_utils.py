"""Tests for lemcvt.utils (ported from v1 tests/test_utils.py; crop/resample now on Grid)."""

import numpy as np
import pytest
from rasterio.transform import Affine

from lemcvt.io.grids import Grid, crop_grid
from lemcvt.utils import compute_slope_magnitude, nanaware_zoom, resample_from_1m


def _grid(data, dx=1.0):
    return Grid(data=data, transform=Affine(dx, 0, 0, 0, -dx, data.shape[0] * dx), crs=None)


class TestCrop:

    def test_shape_and_transform(self):
        g = _grid(np.random.RandomState(0).rand(1000, 1000) * 100)
        c = crop_grid(g, 100, 300, 100, 300)
        assert c.shape == (200, 200)
        assert c.transform.c == 100.0
        assert c.transform.f == 1000.0 - 100.0

    def test_default_like_v1_extent(self):
        g = _grid(np.random.RandomState(0).rand(1000, 1000))
        c = crop_grid(g, 200, 800, 200, 800)
        assert c.shape == (600, 600)
        np.testing.assert_array_equal(c.data, g.data[200:800, 200:800])


class TestComputeSlopeMagnitude:

    def test_flat_surface_zero_slope(self):
        Z = np.ones((20, 20)) * 100.0
        S = compute_slope_magnitude(Z)
        assert S.shape == (20, 20)
        np.testing.assert_allclose(S, 0.0, atol=1e-12)

    def test_tilted_plane_uniform_slope(self):
        j, i = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        Z = 100.0 + 0.1 * j
        S = compute_slope_magnitude(Z)
        np.testing.assert_allclose(S[1:-1, 1:-1], 0.1, atol=1e-10)

    def test_mask_applied(self):
        Z = np.random.RandomState(0).rand(20, 20)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        S = compute_slope_magnitude(Z, mask=mask)
        assert np.all(np.isnan(S[~mask]))
        assert np.all(np.isfinite(S[mask]))

    def test_shape_preserved(self):
        Z = np.random.RandomState(1).rand(30, 40)
        S = compute_slope_magnitude(Z)
        assert S.shape == (30, 40)


class TestNanawareZoom:

    def test_shape_changes_with_zoom(self):
        grid = np.random.RandomState(0).rand(100, 100)
        out = nanaware_zoom(grid, (0.5, 0.5), order=1)
        assert out.shape == (50, 50)

    def test_nan_handling(self):
        grid = np.ones((20, 20))
        grid[5:10, 5:10] = np.nan
        out = nanaware_zoom(grid, (0.5, 0.5), order=1)
        assert np.any(np.isnan(out))
        assert np.any(np.isfinite(out))


class TestResampleFrom1m:

    def test_shape_changes(self):
        g = _grid(np.random.RandomState(0).rand(100, 100), dx=1.0)
        r = resample_from_1m(g, dx_target=5.0, order=1)
        assert r.shape == (20, 20)
        assert r.dx == 5.0
        assert r.dy == 5.0

    def test_origin_preserved(self):
        g = _grid(np.random.RandomState(0).rand(100, 100), dx=1.0)
        r = resample_from_1m(g, dx_target=4.0)
        assert r.transform.c == g.transform.c
        assert r.transform.f == g.transform.f

    def test_v1_zoom_factor(self):
        """v1 used zoom = dx_old / dx_target with scipy.ndimage.zoom; same array here."""
        from scipy.ndimage import zoom

        data = np.random.RandomState(2).rand(60, 60)
        g = _grid(data, dx=1.0)
        r = resample_from_1m(g, dx_target=3.0, order=1)
        ref = zoom(data, zoom=(1 / 3, 1 / 3), order=1, mode="nearest", prefilter=False)
        np.testing.assert_allclose(r.data, ref)


@pytest.mark.parametrize("order", [0, 1])
def test_zoom_order_runs(order):
    out = nanaware_zoom(np.random.RandomState(3).rand(30, 30), (0.5, 0.5), order=order)
    assert out.shape == (15, 15)
