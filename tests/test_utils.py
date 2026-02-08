"""Tests for src/utils.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Mock objects for crop()
# ---------------------------------------------------------------------------

class _MockGeorefInfo:
    def __init__(self, dx=1.0):
        self.dx = dx


class _MockRaster:
    """Mimics a TopoAnalysis raster object for testing crop()."""

    def __init__(self, data: np.ndarray, dx: float = 1.0):
        self._griddata = data.copy()
        self._georef_info = _MockGeorefInfo(dx)
        self._nodata_value = np.nan
        self._dx = dx
        self._shape = data.shape

    def _rowscols_to_xy(self, rowscols_list):
        results = []
        for rc in rowscols_list:
            row, col = rc
            x = col * self._dx
            y = (self._shape[0] - row) * self._dx
            results.append((x, y))
        return results

    def clip_to_extent(self, extent):
        xmin, xmax, ymin, ymax = extent
        dx = self._dx
        Ny = self._shape[0]
        col_min = max(0, int(round(xmin / dx)))
        col_max = min(self._shape[1], int(round(xmax / dx)))
        row_max = min(Ny, int(round((Ny * dx - ymin) / dx)))
        row_min = max(0, int(round((Ny * dx - ymax) / dx)))
        clipped_data = self._griddata[row_min:row_max, col_min:col_max]
        return _MockRaster(clipped_data, dx=dx)


def _make_mock_rasters(Ny=1000, Nx=1000, dx=1.0):
    dem = _MockRaster(np.random.RandomState(0).rand(Ny, Nx) * 100, dx=dx)
    area = _MockRaster(np.random.RandomState(1).rand(Ny, Nx) * 1e6, dx=dx)
    fd = _MockRaster(np.full((Ny, Nx), 4.0), dx=dx)
    return dem, area, fd


# ---------------------------------------------------------------------------
# Tests: crop()
# ---------------------------------------------------------------------------

class TestCrop:

    def test_returns_tuple_of_three(self):
        from utils import crop
        dem, area, fd = _make_mock_rasters()
        result = crop(dem, area, fd)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_default_extent(self):
        from utils import crop
        dem, area, fd = _make_mock_rasters()
        dem_c, area_c, fd_c = crop(dem, area, fd)
        assert dem_c._griddata.shape[0] <= 1000
        assert dem_c._griddata.shape[1] <= 1000

    def test_custom_extent(self):
        from utils import crop
        dem, area, fd = _make_mock_rasters()
        dem_c, area_c, fd_c = crop(dem, area, fd, extent_indexes=(100, 300, 100, 300))
        assert dem_c._griddata.shape[0] <= 1000
        assert dem_c._griddata.shape[1] <= 1000


# ---------------------------------------------------------------------------
# Tests: compute_slope_magnitude()
# ---------------------------------------------------------------------------

class TestComputeSlopeMagnitude:

    def test_flat_surface_zero_slope(self):
        from utils import compute_slope_magnitude
        Z = np.ones((20, 20)) * 100.0
        S = compute_slope_magnitude(Z)
        assert S.shape == (20, 20)
        np.testing.assert_allclose(S, 0.0, atol=1e-12)

    def test_tilted_plane_uniform_slope(self):
        from utils import compute_slope_magnitude
        j, i = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        Z = 100.0 + 0.1 * j  # gradient of 0.1 in j direction
        S = compute_slope_magnitude(Z)
        # Interior cells should have slope ~0.1
        np.testing.assert_allclose(S[1:-1, 1:-1], 0.1, atol=1e-10)

    def test_mask_applied(self):
        from utils import compute_slope_magnitude
        Z = np.random.RandomState(0).rand(20, 20)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        S = compute_slope_magnitude(Z, mask=mask)
        assert np.all(np.isnan(S[~mask]))
        assert np.all(np.isfinite(S[mask]))

    def test_shape_preserved(self):
        from utils import compute_slope_magnitude
        Z = np.random.RandomState(1).rand(30, 40)
        S = compute_slope_magnitude(Z)
        assert S.shape == (30, 40)


# ---------------------------------------------------------------------------
# Tests: find_project_root()
# ---------------------------------------------------------------------------

class TestFindProjectRoot:

    def test_finds_repo_root(self, tmp_path):
        from utils import find_project_root
        # Create a mock directory tree: tmp_path/data/
        (tmp_path / "data").mkdir()
        (tmp_path / "notebooks").mkdir()
        start = tmp_path / "notebooks"
        root = find_project_root(start=str(start))
        assert root == tmp_path

    def test_raises_when_not_found(self, tmp_path):
        from utils import find_project_root
        with pytest.raises(FileNotFoundError):
            find_project_root(start=str(tmp_path), marker_dir="nonexistent_dir")

    def test_custom_marker(self, tmp_path):
        from utils import find_project_root
        (tmp_path / "my_marker").mkdir()
        root = find_project_root(start=str(tmp_path), marker_dir="my_marker")
        assert root == tmp_path
