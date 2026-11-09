"""A self-contained synthetic tile (grids + configs) for loader, CLI and pipeline tests.

Builds a 100 x 100 metre-unit landscape with a dendritic-ish drainage pattern,
fills it, computes D8 directions and area with :mod:`lemcvt.hydro`, writes the
three GeoTIFFs under ``<tmp>/data/GM/tile1/`` and writes site/tile/slurm/data
configs under ``<tmp>/configs``. Returns the paths and the values needed to set
``LEMCVT_DATA`` / ``LEMCVT_CONFIGS`` / ``LEMCVT_RESULTS``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml
from rasterio.crs import CRS
from rasterio.transform import Affine

from lemcvt.hydro.d8 import d8_area, d8_flow_direction
from lemcvt.hydro.fill import priority_flood_fill
from lemcvt.io.grids import Grid, sha256_file, write_geotiff


def synthetic_dem(ny: int = 100, nx: int = 100, dx: float = 1.0, seed: int = 0,
                  z_units: str = "m") -> np.ndarray:
    """Tilted surface draining south with two incised valleys and gentle hilltops."""
    rs = np.random.RandomState(seed)
    y, x = np.meshgrid(np.arange(ny) * dx, np.arange(nx) * dx, indexing="ij")
    z = 320.0 + 0.25 * (ny * dx - y)                      # regional slope to the south (outlet row = last)
    for xc in (0.3 * nx * dx, 0.72 * nx * dx):          # two valleys
        z -= 6.0 * np.exp(-((x - xc) ** 2) / (2 * (0.08 * nx * dx) ** 2)) * (y / (ny * dx))
    z += 2.0 * np.sin(2 * np.pi * x / (0.5 * nx * dx)) * np.cos(2 * np.pi * y / (0.9 * ny * dx))
    z += 0.05 * rs.randn(ny, nx)
    if z_units == "ftUS":
        z = z / (1200.0 / 3937.0)
    return z


def hydro_from_padded(z: np.ndarray, dx: float, pad: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill, D8 and area computed on a padded copy and cropped back.

    The v1 tiles were clipped from mosaics after routing, so their edge rows carry
    proper flow directions. Computing D8 on the bare tile instead triggers
    TopoAnalysis's edge convention (the last two rows/columns get no south/east
    comparisons), which routes flow along the edges and creates 2-cycles. Padding by
    mirrored, slightly lowered margins and cropping reproduces the clipped-mosaic case.
    """
    ny, nx = z.shape
    zp = np.pad(z, pad, mode="reflect")
    # tilt the margins down so that flow leaves through them
    yy, xx = np.indices(zp.shape)
    dist = np.maximum.reduce([np.maximum(pad - yy, 0), np.maximum(yy - (ny + pad - 1), 0),
                              np.maximum(pad - xx, 0), np.maximum(xx - (nx + pad - 1), 0)])
    zp = zp - 2.0 * dist * dx
    zf = priority_flood_fill(zp, dx)
    fd = d8_flow_direction(zf)
    area = d8_area(fd, zf, dx)
    sl = (slice(pad, pad + ny), slice(pad, pad + nx))
    return zf[sl].copy(), fd[sl].copy(), area[sl].copy()


def make_synthetic_tile(root: Path, tile: str = "GM1", site: str = "GM", ny: int = 100, nx: int = 100,
                        dx: float = 1.0, z_units: str = "m", reproduce_v1_units: bool = False,
                        seed: int = 0) -> dict:
    root = Path(root)
    data = root / "data"
    cfgs = root / "configs"
    results = root / "results"
    for d in (data, cfgs / "sites", cfgs / "tiles", cfgs / "slurm", results):
        d.mkdir(parents=True, exist_ok=True)

    z = synthetic_dem(ny, nx, dx, seed, z_units)
    zf, fd, area = hydro_from_padded(z, dx)
    transform = Affine(dx, 0, 700000.0, 0, -dx, 3980000.0)
    crs = CRS.from_epsg(32610 if site == "GM" else 26917)
    idx = int(tile[2:])
    tdir = data / site / f"tile{idx}"
    base = f"{site}_1m_best_tile"
    paths = {
        "filled": write_geotiff(Grid(zf, transform, crs), tdir / f"{base}_filled.tif"),
        "area": write_geotiff(Grid(area, transform, crs), tdir / f"{base}_area.tif"),
        "fd": write_geotiff(Grid(fd, transform, crs), tdir / f"{base}_fd.tif", dtype="uint8"),
    }
    U = 1.47e-4 if site == "GM" else 5.5e-5
    site_cfg = {
        "site": site, "name": f"synthetic {site}", "U": U, "n": 1.0, "S_c": 1.27,
        "z_range_m": [200.0, 600.0], "max_median_slope": 0.6, "crs_epsg": crs.to_epsg(),
        "regional": {"K": 1e-4, "D": 1e-2, "m": 0.35, "A_crit": 1e4, "U": U},
    }
    (cfgs / "sites" / f"{site}.yaml").write_text(yaml.safe_dump(site_cfg, sort_keys=False))
    theta, ks, cht = 0.35, 5.0, 1.5e-2
    tile_cfg = {
        "tile": tile, "site": site,
        "grids": {
            "filled": f"{site}/tile{idx}/{base}_filled.tif", "area": f"{site}/tile{idx}/{base}_area.tif",
            "fd": f"{site}/tile{idx}/{base}_fd.tif",
            "sha256": {k: sha256_file(p) for k, p in paths.items()},
            "shape": [ny, nx], "dx": dx, "dy": dx, "crs_epsg": crs.to_epsg(),
        },
        "units": {"z_units": z_units, "xy_units": "m", "reproduce_v1_units": reproduce_v1_units},
        "measured_v1": {"theta": theta, "ks": ks, "hilltop_curvature": cht},
        "model": {
            "m": theta, "S_c": 1.27, "K0_ad": "U/ks", "K0_partitioned": "U/ks", "D0": "U/C_ht",
            "K_bounds": [7e-7, 2e-2], "D_bounds": [4e-5, 5e-2], "A_crit_0": 500.0, "A_crit_bounds": [1e2, 1e5],
            "nonlinear_max_iter": 5, "nonlinear_tol": 1e-4,
        },
        "v1_reference": {"stages": {}},
    }
    (cfgs / "tiles" / f"{tile}.yaml").write_text(yaml.safe_dump(tile_cfg, sort_keys=False))
    slurm_cfg = {"partition": None, "account": None, "qos": None, "micromamba": "micromamba",
                 "env_prefix": "env", "log_dir": str(root / "jobs"), "defaults": {"time": "01:00:00", "cpus": 2, "mem": "8G"},
                 "steps": {"tile_run": {"time": "02:00:00", "cpus": 2, "mem": "8G"}}}
    (cfgs / "slurm" / "sherlock.yaml").write_text(yaml.safe_dump(slurm_cfg))
    (cfgs / "data.yaml").write_text(yaml.safe_dump({"source": str(data)}))
    return {"root": root, "data": data, "configs": cfgs, "results": results, "paths": paths, "tile": tile,
            "z": z, "zf": zf, "fd": fd, "area": area, "dx": dx, "tile_cfg": tile_cfg, "site_cfg": site_cfg}


def set_env(info: dict) -> dict[str, str | None]:
    """Point lemcvt at the synthetic tile; returns previous values for restoration."""
    prev = {k: os.environ.get(k) for k in ("LEMCVT_DATA", "LEMCVT_CONFIGS", "LEMCVT_RESULTS")}
    os.environ["LEMCVT_DATA"] = str(info["data"])
    os.environ["LEMCVT_CONFIGS"] = str(info["configs"])
    os.environ["LEMCVT_RESULTS"] = str(info["results"])
    return prev


def restore_env(prev: dict[str, str | None]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
