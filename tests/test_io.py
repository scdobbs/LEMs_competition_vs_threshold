"""GeoTIFF round trip, units and sanity checks (directive Phase 1, requirement 6)."""

import numpy as np
import pytest
from rasterio.transform import Affine

from lemcvt.io.grids import Grid, read_geotiff, sha256_file, write_geotiff
from lemcvt.io.units import (
    FT_US_IN_M,
    UnitError,
    check_xy_units,
    sanity_check_elevation,
    z_scale_factor,
    z_to_metres,
)


def _grid(data, dx=1.0, crs="EPSG:32610"):
    from rasterio.crs import CRS

    return Grid(data=data, transform=Affine(dx, 0, 700991.0, 0, -dx, 3981451.0), crs=CRS.from_string(crs))


class TestRoundTrip:

    def test_float64_bit_identical(self, tmp_path):
        rs = np.random.RandomState(0)
        g = _grid(rs.rand(37, 41) * 300 + 100)
        p = write_geotiff(g, tmp_path / "z.tif")
        r = read_geotiff(p)
        assert r.data.dtype == np.float64
        np.testing.assert_array_equal(r.data, g.data)
        assert r.transform == g.transform
        assert r.crs == g.crs
        assert r.dx == 1.0 and r.dy == 1.0

    def test_uint8_round_trip(self, tmp_path):
        fd = np.random.RandomState(1).choice([1, 2, 4, 8, 16, 32, 64, 128], size=(20, 20)).astype(np.uint8)
        g = _grid(fd)
        p = write_geotiff(g, tmp_path / "fd.tif", dtype="uint8")
        r = read_geotiff(p)
        assert r.data.dtype == np.uint8
        np.testing.assert_array_equal(r.data, fd)

    def test_nodata_becomes_nan(self, tmp_path):
        data = np.ones((10, 10)) * 5.0
        data[0, 0] = -9999.0
        g = _grid(data)
        p = write_geotiff(g, tmp_path / "nd.tif", nodata=-9999.0)
        r = read_geotiff(p)
        assert np.isnan(r.data[0, 0]) and r.data[1, 1] == 5.0

    def test_write_is_deterministic(self, tmp_path):
        g = _grid(np.random.RandomState(2).rand(16, 16))
        p1 = write_geotiff(g, tmp_path / "a.tif")
        p2 = write_geotiff(g, tmp_path / "b.tif")
        assert sha256_file(p1) == sha256_file(p2)

    def test_anisotropic_pixels_recorded(self, tmp_path):
        g = Grid(data=np.zeros((5, 6)), transform=Affine(0.9751997, 0, 0, 0, -0.9751305, 0))
        p = write_geotiff(g, tmp_path / "ap.tif")
        r = read_geotiff(p)
        assert r.dx == pytest.approx(0.9751997)
        assert r.dy == pytest.approx(0.9751305)
        assert r.shape == (5, 6)


class TestUnits:

    def test_constants(self):
        assert FT_US_IN_M == pytest.approx(0.3048006096, rel=1e-9)
        assert abs(FT_US_IN_M - 0.3048) / 0.3048 < 3e-6

    def test_factor_lookup(self):
        assert z_scale_factor("m") == 1.0
        assert z_scale_factor("ftUS") == FT_US_IN_M
        with pytest.raises(UnitError):
            z_scale_factor("furlongs")

    def test_conversion(self):
        z = np.array([1000.0, 1400.0])
        zm, f = z_to_metres(z, "ftUS")
        assert f == FT_US_IN_M
        np.testing.assert_allclose(zm, z * FT_US_IN_M)

    def test_reproduce_v1_disables_conversion_with_warning(self, caplog):
        z = np.array([1000.0, 1400.0])
        with caplog.at_level("WARNING", logger="lemcvt.io.units"):
            zm, f = z_to_metres(z, "ftUS", reproduce_v1_units=True)
        assert f == 1.0
        assert zm is z
        assert any("NOT converted" in r.message for r in caplog.records)

    def test_xy_units(self):
        check_xy_units("m")
        with pytest.raises(UnitError):
            check_xy_units("ft")


class TestSanity:

    def _plane(self, slope):
        j, i = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
        return 300.0 + slope * j

    def test_passes_in_range(self):
        d = sanity_check_elevation(self._plane(0.2), 1.0, (200.0, 600.0), 0.6, label="t")
        assert d["median_slope"] == pytest.approx(0.2)

    def test_refuses_feet_range(self):
        z = self._plane(0.2) * 3.2808  # as if feet were declared metres
        with pytest.raises(UnitError, match="elevation range"):
            sanity_check_elevation(z, 1.0, (200.0, 600.0), 0.6, label="t")

    def test_refuses_steep_median_slope(self):
        with pytest.raises(UnitError, match="median pixel slope"):
            sanity_check_elevation(self._plane(0.9), 1.0, (200.0, 600.0), 0.6, label="t")
