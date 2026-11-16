"""Tests for src/plotting.py — smoke tests that ensure plots don't crash."""

import sys
from pathlib import Path

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# We need to mock TopoAnalysis.dem since it may not be installed in test env.
# plotting.py imports it as `import TopoAnalysis.dem as d`.


class _MockGeorefInfo:
    def __init__(self, dx=10.0):
        self.dx = dx


class _MockElevation:
    """Mock d.Elevation that plotting.py constructs internally."""
    def __init__(self, griddata=None, georef_info=None):
        self._griddata = griddata
        self._georef_info = georef_info or _MockGeorefInfo()
        self._nodata_value = np.nan


class _MockSlopeObj:
    """Mock d.ChannelSlopeWithSmoothing — returns a simple slope grid."""
    def __init__(self, **kwargs):
        elev = kwargs.get("elevation")
        if elev is not None and elev._griddata is not None:
            shape = elev._griddata.shape
            self._griddata = np.abs(np.random.RandomState(0).randn(*shape) * 0.01) + 0.001
        else:
            self._griddata = np.ones((20, 20)) * 0.01


class _MockDemModule:
    """Stand-in for `TopoAnalysis.dem` module used by plotting.py."""
    Elevation = _MockElevation
    ChannelSlopeWithSmoothing = _MockSlopeObj


@pytest.fixture(autouse=True)
def _patch_topoanalysis(monkeypatch):
    """Replace TopoAnalysis.dem with our mock before importing plotting."""
    import importlib
    mock_mod = _MockDemModule()
    monkeypatch.setitem(sys.modules, "TopoAnalysis", type(sys)("TopoAnalysis"))
    monkeypatch.setitem(sys.modules, "TopoAnalysis.dem", mock_mod)

    # Force re-import so plotting.py picks up the mock
    if "plotting" in sys.modules:
        importlib.reload(sys.modules["plotting"])


@pytest.fixture
def dem_grid():
    Ny, Nx = 20, 20
    Z = np.random.RandomState(1).rand(Ny, Nx) * 100 + 500
    obj = _MockElevation()
    obj._griddata = Z
    obj._georef_info = _MockGeorefInfo(10.0)
    return obj


@pytest.fixture
def area_grid():
    Ny, Nx = 20, 20
    A = np.random.RandomState(2).rand(Ny, Nx) * 1e5 + 100
    obj = _MockElevation()
    obj._griddata = A
    obj._georef_info = _MockGeorefInfo(10.0)
    return obj


@pytest.fixture
def fd_grid():
    Ny, Nx = 20, 20
    fd = np.full((Ny, Nx), 4.0)
    obj = _MockElevation()
    obj._griddata = fd
    return obj


class TestPlotMapsAreaSlope:

    def test_runs_without_crash(self, dem_grid, area_grid, fd_grid):
        from plotting import plot_maps_area_slope
        Z_model = dem_grid._griddata + np.random.RandomState(3).randn(20, 20)
        # Should not raise
        plot_maps_area_slope(dem_grid, area_grid, fd_grid, Z_model)
        plt.close("all")

    def test_creates_figure(self, dem_grid, area_grid, fd_grid):
        from plotting import plot_maps_area_slope
        Z_model = dem_grid._griddata + np.random.RandomState(4).randn(20, 20)
        plot_maps_area_slope(dem_grid, area_grid, fd_grid, Z_model)
        figs = plt.get_fignums()
        assert len(figs) > 0
        plt.close("all")


class TestPlotMapsAndChannelMask:

    def test_runs_without_crash(self, dem_grid):
        from plotting import plot_maps_and_channel_mask
        Z_model = dem_grid._griddata + np.random.RandomState(5).randn(20, 20)
        mask = np.random.RandomState(6).rand(20, 20) > 0.5
        plot_maps_and_channel_mask(dem_grid, Z_model, mask)
        plt.close("all")

    def test_creates_figure(self, dem_grid):
        from plotting import plot_maps_and_channel_mask
        Z_model = dem_grid._griddata + np.random.RandomState(7).randn(20, 20)
        mask = np.random.RandomState(8).rand(20, 20) > 0.5
        plot_maps_and_channel_mask(dem_grid, Z_model, mask)
        figs = plt.get_fignums()
        assert len(figs) > 0
        plt.close("all")
