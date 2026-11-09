"""GeoTIFF grids.

The v1 grids written by TopoAnalysis are plain GeoTIFFs (GDAL GTiff driver,
LZW, no nodata tag, no projection on in-memory outputs); see
``reference/v1_grids_manifest.csv``. Reading them here with rasterio reproduces
TopoAnalysis's ``BaseSpatialGrid.load`` exactly:

* data are read as ``float64`` (``uint8`` for flow directions and the derived
  byte grids) and a nodata value, if tagged, is replaced by NaN for float grids;
* ``dx`` is the geotransform's x pixel size (``transform.a``) and, as in v1,
  ``dy`` is *not* used by the models (``dy = dx``). The true y pixel size is kept
  on the :class:`Grid` so the AP anisotropy (0.9752 x 0.97513 m) is visible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class Grid:
    """A single-band raster with its georeference.

    Attributes
    ----------
    data : np.ndarray
        2-D array (rows = y from north, cols = x).
    transform : Affine
        GDAL-style affine transform of the upper-left pixel corner.
    crs : Any
        rasterio CRS or ``None`` (the v1 output grids carry none).
    nodata : float | None
        Nodata tag of the file; NaN is used in memory for float grids.
    name : str
        Free text (file stem) for logging.
    """

    data: np.ndarray
    transform: Affine
    crs: Any = None
    nodata: float | None = None
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- geometry -------------------------------------------------------------------
    @property
    def dx(self) -> float:
        """x pixel size; this is what v1 (TopoAnalysis ``Georef_info.dx``) used everywhere."""
        return float(self.transform.a)

    @property
    def dy(self) -> float:
        """|y pixel size| (not used by the v1 numerics; recorded for provenance)."""
        return float(abs(self.transform.e))

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def ny(self) -> int:
        return int(self.data.shape[0])

    @property
    def nx(self) -> int:
        return int(self.data.shape[1])

    @property
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL geotransform tuple ``(x0, dx, rx, y0, ry, dy)``."""
        t = self.transform
        return (t.c, t.a, t.b, t.f, t.d, t.e)

    def with_data(self, data: np.ndarray, name: str | None = None) -> Grid:
        """Copy of this grid's georeference carrying new data of the same shape."""
        if data.shape != self.data.shape:
            raise ValueError(f"shape {data.shape} differs from grid shape {self.data.shape}")
        return replace(self, data=data, name=self.name if name is None else name)

    def stats(self) -> dict[str, float]:
        a = self.data.astype(np.float64, copy=False)
        finite = np.isfinite(a)
        return {
            "min": float(a[finite].min()) if finite.any() else float("nan"),
            "max": float(a[finite].max()) if finite.any() else float("nan"),
            "mean": float(a[finite].mean()) if finite.any() else float("nan"),
            "n_finite": int(finite.sum()),
            "n_nan": int(np.isnan(a).sum()),
        }


def read_geotiff(path: str | Path, dtype: str | None = None) -> Grid:
    """Read band 1 of a GeoTIFF exactly as TopoAnalysis loads it.

    Parameters
    ----------
    path
        File path (with or without ``.tif`` extension).
    dtype
        ``"float64"`` (default for float sources), ``"uint8"`` or any numpy dtype
        string. With a float dtype the file's nodata value becomes NaN.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        src_dtype = src.dtypes[0]
    if dtype is None:
        dtype = "uint8" if src_dtype == "uint8" else "float64"
    out = arr.astype(dtype)
    if nodata is not None and np.dtype(dtype).kind == "f":
        out[arr == nodata] = np.nan
    return Grid(data=out, transform=transform, crs=crs, nodata=nodata, name=path.stem,
                meta={"source": str(path), "source_dtype": src_dtype})


def write_geotiff(grid: Grid, path: str | Path, dtype: str | None = None,
                  compress: str = "LZW", nodata: float | None = None) -> Path:
    """Write a :class:`Grid` as a single-band GeoTIFF.

    Defaults match TopoAnalysis's ``save`` (GTiff, LZW, no nodata tag) except that
    the CRS is written when the grid has one. ``dtype`` defaults to the array's
    dtype; TopoAnalysis's byte products (flow direction, hillshade, log-area) are
    written as ``uint8`` by passing ``dtype="uint8"``, which truncates like
    ``astype(uint8)`` did in v1.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = grid.data if dtype is None else grid.data.astype(dtype)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": str(data.dtype),
        "transform": grid.transform,
        "crs": grid.crs,
        "compress": compress,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def crop_grid(grid: Grid, row0: int, row1: int, col0: int, col1: int) -> Grid:
    """Crop by pixel indices (half-open slices) and shift the transform accordingly."""
    data = grid.data[row0:row1, col0:col1]
    t = grid.transform
    new_t = Affine(t.a, t.b, t.c + col0 * t.a + row0 * t.b, t.d, t.e, t.f + col0 * t.d + row0 * t.e)
    return Grid(data=data, transform=new_t, crs=grid.crs, nodata=grid.nodata,
                name=f"{grid.name}_crop", meta=dict(grid.meta, crop=(row0, row1, col0, col1)))


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    """sha256 of a file's bytes (the manifest convention)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    """sha256 of an array's bytes (C order, native dtype); for in-memory provenance."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
