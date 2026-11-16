"""D8, fill, area, channel slope: synthetic properties, and bit-level equivalence with
TopoAnalysis when it is importable (Sherlock: PYTHONPATH=$GROUP_HOME/opt)."""

import importlib.util

import numpy as np
import pytest

from lemcvt.hydro.channel_slope import channel_slope_with_smoothing
from lemcvt.hydro.d8 import d8_area, d8_divides, d8_flow_direction, d8_receiver_arrays
from lemcvt.hydro.derived import hillshade, to_byte_like_gdal
from lemcvt.hydro.fill import priority_flood_fill
from synthetic_tile import synthetic_dem

HAS_TA = importlib.util.find_spec("TopoAnalysis") is not None


def _south_plane(ny=12, nx=10, dx=2.0):
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return 100.0 - 1.0 * j + 0.0 * i


class TestD8:

    def test_south_plane_codes(self):
        fd = d8_flow_direction(_south_plane())
        # interior rows (not the last two, see TopoAnalysis slicing) flow south
        assert np.all(fd[0:-2, 1:-2] == 4)
        assert fd.dtype == np.uint8

    def test_nan_cells_zero(self):
        z = _south_plane()
        z[3, 3] = np.nan
        fd = d8_flow_direction(z)
        assert fd[3, 3] == 0

    def test_receivers_consistent(self):
        fd = d8_flow_direction(_south_plane())
        r_to, c_to, good = d8_receiver_arrays(fd)
        assert good[0, 1] and r_to[0, 1] == 1 and c_to[0, 1] == 1

    def test_area_bounds(self):
        """Area starts at one cell everywhere and never exceeds the grid; exact agreement with
        TopoAnalysis is checked in TestTopoAnalysisEquivalence (its D8 can route to a cell with no
        positive drop, so a simple donor/receiver monotonicity property does not hold)."""
        z = synthetic_dem(40, 40, 1.0)
        zf = priority_flood_fill(z, 1.0)
        fd = d8_flow_direction(zf)
        area = d8_area(fd, zf, 1.0)
        assert area.min() == 1.0
        assert area.max() <= 40 * 40
        r_to, c_to, good = d8_receiver_arrays(fd)
        rr, cc = np.nonzero(good)
        downhill = zf[r_to[rr, cc], c_to[rr, cc]] < zf[rr, cc]
        assert downhill.mean() > 0.95

    def test_divides_shape(self):
        fd = d8_flow_direction(_south_plane())
        d = d8_divides(fd)
        assert d.shape == fd.shape


class TestFill:

    def test_fills_pit(self):
        z = _south_plane()
        z[5, 5] = 0.0  # pit
        zf = priority_flood_fill(z, 1.0)
        assert zf[5, 5] > 90.0
        assert np.all(zf >= z - 1e-12)

    def test_leaves_draining_surface(self):
        z = _south_plane()
        zf = priority_flood_fill(z, 1.0)
        np.testing.assert_array_equal(zf, z)

    def test_nan_untouched(self):
        z = _south_plane()
        z[2, 2] = np.nan
        zf = priority_flood_fill(z, 1.0)
        assert np.isnan(zf[2, 2])


class TestChannelSlope:

    def test_slope_on_plane(self):
        z = synthetic_dem(60, 60, 1.0)
        zf = priority_flood_fill(z, 1.0)
        fd = d8_flow_direction(zf)
        area = d8_area(fd, zf, 1.0)
        S = channel_slope_with_smoothing(zf, area, fd, 1.0, vertical_interval=2.0, min_area=30.0)
        assert S.shape == zf.shape
        assert np.all(np.isnan(S[area <= 30.0]))
        vals = S[np.isfinite(S)]
        assert vals.size > 0
        assert np.all(np.isfinite(vals))


class TestDerived:

    def test_hillshade_range_and_byte(self):
        z = synthetic_dem(30, 30, 1.0)
        hs = hillshade(z, 1.0)
        b = to_byte_like_gdal(hs)
        assert b.dtype == np.uint8
        assert b.min() >= 0 and b.max() <= 255

    def test_byte_rounding_half_up(self):
        np.testing.assert_array_equal(to_byte_like_gdal(np.array([0.4, 0.5, 254.5, 300.0, -3.0, np.nan])),
                                      np.array([0, 1, 255, 255, 0, 0], dtype=np.uint8))


def _tied_receivers(area, fd):
    """Cells with two or more donors of exactly equal (maximal) drainage area."""
    from lemcvt.hydro.d8 import d8_receiver_arrays

    r_to, c_to, good = d8_receiver_arrays(fd)
    donors = {}
    for i, j in zip(*np.nonzero(good), strict=True):
        donors.setdefault((int(r_to[i, j]), int(c_to[i, j])), []).append(float(area[i, j]))
    return {k for k, v in donors.items() if len(v) > 1 and sorted(v)[-1] == sorted(v)[-2]}


def _path_touches(up_i, up_j, dn_i, dn_j, i, j, tied, max_steps=200):
    """True if the main-stem path through (i, j) (both directions) visits a tied receiver."""
    for arr_i, arr_j in ((up_i, up_j), (dn_i, dn_j)):
        ci, cj = i, j
        for _ in range(max_steps):
            if (ci, cj) in tied:
                return True
            ni, nj = arr_i[ci, cj], arr_j[ci, cj]
            if ni < 0:
                break
            ci, cj = int(ni), int(nj)
    return (i, j) in tied


@pytest.mark.skipif(not HAS_TA, reason="TopoAnalysis not importable")
class TestTopoAnalysisEquivalence:
    """Bit-level agreement with the TopoAnalysis routines on a synthetic grid."""

    def _ta_objects(self, z, dx):
        import TopoAnalysis.dem as d

        e = d.Elevation()
        e._griddata = z.copy()
        e._georef_info.dx = dx
        e._georef_info.nx = z.shape[1]
        e._georef_info.ny = z.shape[0]
        e._georef_info.geoTransform = (0.0, dx, 0.0, z.shape[0] * dx, 0.0, -dx)
        e._georef_info.xllcenter = dx / 2
        e._georef_info.yllcenter = dx / 2
        return d, e

    def test_fill_fd_area(self):
        z = synthetic_dem(50, 50, 1.0, seed=3)
        z[10, 10] -= 5.0  # a pit
        d, e = self._ta_objects(z, 1.0)
        filled = d.FilledElevation(elevation=e)
        fd = d.FlowDirectionD8(flooded_dem=filled)
        area = d.Area(flow_direction=fd)
        zf = priority_flood_fill(z, 1.0)
        np.testing.assert_array_equal(zf, filled._griddata)
        np.testing.assert_array_equal(d8_flow_direction(zf), fd._griddata)
        np.testing.assert_array_equal(d8_area(d8_flow_direction(zf), zf, 1.0), area._griddata)

    def test_channel_slope(self):
        z = synthetic_dem(50, 50, 1.0, seed=4)
        d, e = self._ta_objects(z, 1.0)
        filled = d.FilledElevation(elevation=e)
        fd = d.FlowDirectionD8(flooded_dem=filled)
        area = d.Area(flow_direction=fd)
        s_ta = d.ChannelSlopeWithSmoothing(elevation=filled, area=area, flow_direction=fd,
                                           vertical_interval=2.0, min_area=20.0)._griddata
        s = channel_slope_with_smoothing(filled._griddata, area._griddata, fd._griddata, 1.0,
                                         vertical_interval=2.0, min_area=20.0)
        # N3 (docs/numerics_changes.md): lemcvt breaks ties between equal-area tributaries explicitly
        # (highest flat index wins), TopoAnalysis leaves them to numpy's unstable argsort. The two
        # therefore agree exactly except on smoothing paths that pass a receiver with tied donors.
        from lemcvt.hydro.channel_slope import flow_graph_main_stem

        up_i, up_j, dn_i, dn_j = flow_graph_main_stem(area._griddata, fd._griddata)
        tied = _tied_receivers(area._griddata, fd._griddata)
        differ = np.isfinite(s) & np.isfinite(s_ta) & (s != s_ta)
        differ |= np.isnan(s) != np.isnan(s_ta)
        for i, j in zip(*np.nonzero(differ), strict=True):
            assert _path_touches(up_i, up_j, dn_i, dn_j, int(i), int(j), tied), (i, j)
        same = ~differ & np.isfinite(s)
        np.testing.assert_array_equal(s[same], s_ta[same])
        assert same.sum() > 0.8 * np.isfinite(s).sum()

    def test_main_stem_tie_rule(self):
        """Among equal-area donors the one with the highest flat (row-major) index is the main stem (N3)."""
        from lemcvt.hydro.channel_slope import flow_graph_main_stem

        # 3 x 3: two donors of equal area (row 0, cols 0 and 2) flow into (1, 1) diagonally; (1, 1) flows south.
        fd = np.zeros((3, 3), dtype=np.uint8)
        fd[0, 0] = 2      # SE -> (1, 1)
        fd[0, 2] = 8      # SW -> (1, 1)
        fd[1, 1] = 4      # S  -> (2, 1)
        area = np.ones((3, 3))
        area[1, 1] = 3.0
        area[2, 1] = 4.0
        up_i, up_j, _, _ = flow_graph_main_stem(area, fd)
        assert (up_i[1, 1], up_j[1, 1]) == (0, 2)

    def test_hillshade(self):
        z = synthetic_dem(30, 30, 1.0, seed=5)
        d, e = self._ta_objects(z, 1.0)
        hs = d.Hillshade(elevation=e, azimuth=315.0, inclination=20.0)
        np.testing.assert_allclose(hillshade(z, 1.0), hs._griddata, rtol=0, atol=1e-10)
