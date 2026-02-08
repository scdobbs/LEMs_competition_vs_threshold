"""
Shared test fixtures for the Dobbs & Hilley landscape evolution models.

Provides small synthetic grids (20×20) that mimic the DEM/area/flow_direction
objects used by the models. Uses mock objects with _griddata, _georef_info,
_nodata_value attributes — no real data files or TopoAnalysis needed.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lightweight mock classes that replicate the TopoAnalysis interface
# ---------------------------------------------------------------------------

class MockGeorefInfo:
    """Mimics dem._georef_info with a dx attribute."""
    def __init__(self, dx: float = 10.0):
        self.dx = dx


class MockGrid:
    """Mimics a TopoAnalysis raster object (Elevation, Area, FlowDirectionD8)."""
    def __init__(self, griddata: np.ndarray, dx: float = 10.0, nodata=np.nan):
        self._griddata = griddata.astype(float)
        self._georef_info = MockGeorefInfo(dx)
        self._nodata_value = nodata


# ---------------------------------------------------------------------------
# Helper: build a consistent D8 flow-direction grid for a tilted plane
# ---------------------------------------------------------------------------

def _build_d8_flow(Ny: int, Nx: int) -> np.ndarray:
    """Create a D8 flow direction grid where everything flows to the bottom row.

    Convention:
        1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
    Interior cells flow south (code 4). Bottom row is the outlet row.
    Left-column interior flows SE (2), right-column interior flows SW (8).
    Bottom row cells flow toward center bottom (outlet).
    """
    fd = np.full((Ny, Nx), 4, dtype=float)  # default: flow south

    # Left column flows SE
    fd[:-1, 0] = 2
    # Right column flows SW
    fd[:-1, -1] = 8

    # Bottom row: flow toward center
    center = Nx // 2
    for i in range(Nx):
        if i < center:
            fd[-1, i] = 1   # flow east
        elif i > center:
            fd[-1, i] = 16  # flow west
        else:
            fd[-1, i] = 4   # outlet, code doesn't matter much

    return fd


def _build_drainage_area(Ny: int, Nx: int, dx: float) -> np.ndarray:
    """Build a simple drainage area grid that increases downstream (toward bottom).

    Each cell has area proportional to (Ny - row), so bottom rows have the
    largest area, mimicking a real basin.
    """
    cell_area = dx * dx
    area = np.zeros((Ny, Nx), dtype=float)
    for j in range(Ny):
        # Rows closer to the bottom accumulate more area
        area[j, :] = cell_area * (Ny - j)

    # Make the center of the bottom row the outlet with max area
    center = Nx // 2
    area[-1, center] = cell_area * Ny * Nx
    return area


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_shape():
    """Standard test grid dimensions."""
    return (20, 20)  # Ny, Nx


@pytest.fixture
def dx():
    """Grid spacing in metres."""
    return 10.0


@pytest.fixture
def synthetic_dem(grid_shape, dx):
    """20×20 grid with a tilted plane + small noise (elevation decreasing southward)."""
    Ny, Nx = grid_shape
    np.random.seed(42)
    j_coords, i_coords = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
    # Elevation: high in the north (j=0), low in the south (j=Ny-1)
    Z = 1000.0 - 5.0 * j_coords + 0.1 * np.random.randn(Ny, Nx)
    return MockGrid(Z, dx=dx)


@pytest.fixture
def synthetic_area(grid_shape, dx):
    """Drainage area grid (increasing downstream)."""
    Ny, Nx = grid_shape
    A = _build_drainage_area(Ny, Nx, dx)
    return MockGrid(A, dx=dx)


@pytest.fixture
def synthetic_flow_direction(grid_shape):
    """D8 flow direction grid pointing consistently downslope."""
    Ny, Nx = grid_shape
    fd = _build_d8_flow(Ny, Nx)
    return MockGrid(fd, dx=10.0)


@pytest.fixture
def synthetic_georef_info(dx):
    """Mock georef_info with dx."""
    return MockGeorefInfo(dx)


@pytest.fixture
def model_params():
    """Reasonable physical parameter set for tests."""
    return {
        "m": 0.5,
        "U": 1e-4,
        "K": 1e-5,
        "D": 1e-2,
        "A_crit": 500.0,
        "S_c": 1.2,
    }
