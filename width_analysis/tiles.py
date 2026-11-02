"""Read the repo's tile rasters (filled DEM, D8 area, D8 flow direction)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Tile:
    site: str
    tile: int
    z: np.ndarray          # filled elevation, float64, (ny, nx)
    area: np.ndarray       # D8 drainage area, m^2
    fd: np.ndarray         # D8 codes: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE
    dx: float              # pixel width (m)
    dy: float              # pixel height (m), positive
    x0: float              # x of the left edge of column 0
    y0: float              # y of the top edge of row 0
    crs: str

    @property
    def shape(self):
        return self.z.shape

    @property
    def pixel_size(self) -> float:
        """Mean pixel dimension (the repo's _mean_pixel_dimension analogue)."""
        return 0.5 * (self.dx + self.dy)

    def rc_to_xy(self, row, col):
        """Pixel-centre map coordinates for (fractional) row/col."""
        x = self.x0 + (np.asarray(col) + 0.5) * self.dx
        y = self.y0 - (np.asarray(row) + 0.5) * self.dy
        return x, y


def tile_paths(site: str, tile: int, root: Path = REPO_ROOT) -> dict[str, Path]:
    base = root / "data" / site / f"tile{tile}"
    prefix = f"{site}_1m_best_tile"
    return {k: base / f"{prefix}_{k}" for k in ("filled", "area", "fd")}


def load_tile(site: str, tile: int, root: Path = REPO_ROOT) -> Tile:
    p = tile_paths(site, tile, root)
    with rasterio.open(p["filled"]) as ds:
        z = ds.read(1).astype(np.float64)
        t = ds.transform
        dx, dy = float(t.a), float(-t.e)
        x0, y0 = float(t.c), float(t.f)
        crs = ds.crs.to_string() if ds.crs else ""
        if ds.nodata is not None:
            z[z == ds.nodata] = np.nan
    with rasterio.open(p["area"]) as ds:
        area = ds.read(1).astype(np.float64)
    with rasterio.open(p["fd"]) as ds:
        fd = ds.read(1).astype(np.uint8)
    if z.shape != area.shape or z.shape != fd.shape:
        raise ValueError(f"shape mismatch in {site} tile {tile}")
    return Tile(site, tile, z, area, fd, dx, dy, x0, y0, crs)
