"""Configuration: sites, tiles, Slurm, data locations.

Everything a run needs is explicit in YAML under ``configs/`` (directive Phase 1,
requirement 2). Paths to grids are relative to ``$LEMCVT_DATA``; results go
under ``$LEMCVT_RESULTS``. No hard-coded paths anywhere else in the package.

Unit handling lives in :func:`load_tile`: the tile config declares ``z_units``
and ``xy_units``; z is converted to metres at load time unless
``reproduce_v1_units`` is true (loud warning), and the metre grid must pass the
site sanity checks (elevation range, median slope) or the loader refuses.

Parameters that carry the vertical unit (Phase 5, addendum 4 section 3): in the linear
models ``z -> z / c`` corresponds to ``K -> c K`` and ``D -> c D`` (and MSE -> MSE / c^2).
The v1 AP values of K0, D0 and the K and D bounds were obtained with z in US survey feet;
the tile config declares this with ``model.param_z_units`` (default: the grid's
``z_units``) and :func:`load_tile_config` multiplies K0_ad, K0_partitioned, D0, K_bounds and
D_bounds by ``c = 3937/1200`` when the working vertical unit is metres and the declared
parameter unit is ftUS. The factor applied is recorded in ``ModelParams.param_scale``
(and therefore in every run's config snapshot). With ``reproduce_v1_units: true``, or
with the ``reproduce_v1_units`` override of :func:`load_tile_config` / :func:`load_tile`,
the parameters are left as declared and the grid stays in its stored unit, so v1 can
still be reproduced in feet for the record.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lemcvt.io.grids import Grid, read_geotiff, sha256_file
from lemcvt.io.units import UnitError, check_xy_units, sanity_check_elevation, z_to_metres

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_root() -> Path:
    """``$LEMCVT_CONFIGS`` or ``<repo>/configs`` (evaluated at call time so tests can redirect it)."""
    return Path(os.environ.get("LEMCVT_CONFIGS", REPO_ROOT / "configs"))


TILE_IDS = tuple(f"{s}{i}" for s in ("AP", "GM") for i in range(1, 6))
STAGES = ("ad", "acrit", "nonlinear", "joint")


class ConfigError(ValueError):
    """Invalid or inconsistent configuration."""


def data_root() -> Path:
    """``$LEMCVT_DATA`` (required for any grid access)."""
    v = os.environ.get("LEMCVT_DATA")
    if not v:
        raise ConfigError("LEMCVT_DATA is not set; point it at the directory holding <site>/tile<n>/*.tif")
    return Path(v)


def results_root() -> Path:
    """``$LEMCVT_RESULTS`` (default ``<repo>/results``)."""
    return Path(os.environ.get("LEMCVT_RESULTS", REPO_ROOT / "results"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _require(d: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError(f"{where}: missing keys {missing}")


def _pair(v: Any, name: str, where: str) -> tuple[float, float]:
    if not (isinstance(v, list | tuple) and len(v) == 2):
        raise ConfigError(f"{where}: {name} must be a 2-element list")
    lo, hi = float(v[0]), float(v[1])
    if not (lo > 0 and hi > lo):
        raise ConfigError(f"{where}: {name} must satisfy 0 < lo < hi, got {v}")
    return lo, hi


# --------------------------------------------------------------------------------------
# Site
# --------------------------------------------------------------------------------------

@dataclass
class SiteConfig:
    site: str
    name: str
    U: float
    n: float
    S_c: float
    z_range_m: tuple[float, float]
    max_median_slope: float
    crs_epsg: int | None
    regional: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def load_site(site: str) -> SiteConfig:
    path = config_root() / "sites" / f"{site}.yaml"
    d = _read_yaml(path)
    _require(d, ("site", "name", "U", "n", "S_c", "z_range_m", "max_median_slope"), str(path))
    if d["site"] != site:
        raise ConfigError(f"{path}: site field {d['site']!r} != {site!r}")
    zr = d["z_range_m"]
    if not (isinstance(zr, list) and len(zr) == 2 and zr[0] < zr[1]):
        raise ConfigError(f"{path}: z_range_m must be [lo, hi]")
    return SiteConfig(
        site=site, name=str(d["name"]), U=float(d["U"]), n=float(d["n"]), S_c=float(d["S_c"]),
        z_range_m=(float(zr[0]), float(zr[1])), max_median_slope=float(d["max_median_slope"]),
        crs_epsg=d.get("crs_epsg"), regional=dict(d.get("regional", {})), raw=d,
    )


# --------------------------------------------------------------------------------------
# Tile
# --------------------------------------------------------------------------------------

@dataclass
class ModelParams:
    m: float
    U: float
    n: float
    S_c: float
    K0_ad: float
    K0_partitioned: float
    D0: float
    K_bounds: tuple[float, float]
    D_bounds: tuple[float, float]
    A_crit_0: float
    A_crit_bounds: tuple[float, float]
    nonlinear_max_iter: int = 50
    nonlinear_tol: float = 1e-6
    methods: dict[str, str] = field(default_factory=lambda: {
        "ad": "L-BFGS-B", "acrit": "Nelder-Mead", "nonlinear": "Nelder-Mead", "joint": "Nelder-Mead",
    })
    param_z_units: str = "m"
    """Vertical unit the declared K0, D0 and bounds refer to (``ftUS`` on the AP tiles, v1 values)."""
    param_scale: float = 1.0
    """Factor applied to K0_ad, K0_partitioned, D0, K_bounds and D_bounds (c = 3937/1200 for ftUS -> m)."""
    z_units_working: str = "m"


@dataclass
class TileConfig:
    tile: str
    site: str
    grids: dict[str, str]
    sha256: dict[str, str]
    shape: tuple[int, int]
    dx: float
    dy: float
    z_units: str
    xy_units: str
    reproduce_v1_units: bool
    measured_v1: dict[str, Any]
    model: ModelParams
    v1_reference: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    path: Path | None = None

    @property
    def index(self) -> int:
        return int(self.tile[2:])


def _resolve_expr(v: Any, ctx: dict[str, float], where: str) -> float:
    """Numeric value, or one of the sentinels ``"U/ks"`` / ``"U/C_ht"`` evaluated from ctx.

    Using the expression keeps v1's exact float (it computed ``U/steepness`` at runtime).
    """
    if isinstance(v, int | float):
        return float(v)
    if v == "U/ks":
        return float(np.power(ctx["U"] / ctx["ks"], 1.0 / ctx["n"]))
    if v == "U/C_ht":
        return ctx["U"] / ctx["C_ht"]
    raise ConfigError(f"{where}: cannot interpret value {v!r} (number, 'U/ks' or 'U/C_ht')")


def load_tile_config(tile: str, reproduce_v1_units: bool | None = None) -> TileConfig:
    """Read ``configs/tiles/<tile>.yaml``.

    ``reproduce_v1_units`` overrides the file's flag (``True``: keep the stored vertical unit
    and the declared parameters, as v1 ran; ``False``: metres, parameters scaled). The
    override exists so that feet runs (Phase 2 regression, Phase 4b) remain possible after
    Phase 5 switched the AP configs to metres; it changes nothing on disk.
    """
    path = config_root() / "tiles" / f"{tile}.yaml"
    d = _read_yaml(path)
    _require(d, ("tile", "site", "grids", "units", "measured_v1", "model"), str(path))
    if d["tile"] != tile:
        raise ConfigError(f"{path}: tile field {d['tile']!r} != {tile!r}")
    site = load_site(d["site"])

    g = d["grids"]
    _require(g, ("filled", "area", "fd", "sha256", "shape", "dx", "dy"), f"{path}: grids")
    units = d["units"]
    _require(units, ("z_units", "xy_units"), f"{path}: units")
    check_xy_units(units["xy_units"])
    repro = bool(units.get("reproduce_v1_units", False)) if reproduce_v1_units is None else bool(reproduce_v1_units)
    mv = d["measured_v1"]
    _require(mv, ("theta", "ks", "hilltop_curvature"), f"{path}: measured_v1")

    mdl = d["model"]
    _require(mdl, ("m", "K0_ad", "K0_partitioned", "D0", "K_bounds", "D_bounds", "A_crit_0", "A_crit_bounds"),
             f"{path}: model")
    ctx = {"U": site.U, "n": site.n, "ks": float(mv["ks"]), "C_ht": float(mv["hilltop_curvature"])}
    model = ModelParams(
        m=float(mdl["m"]), U=site.U, n=site.n, S_c=float(mdl.get("S_c", site.S_c)),
        K0_ad=_resolve_expr(mdl["K0_ad"], ctx, f"{path}: K0_ad"),
        K0_partitioned=_resolve_expr(mdl["K0_partitioned"], ctx, f"{path}: K0_partitioned"),
        D0=_resolve_expr(mdl["D0"], ctx, f"{path}: D0"),
        K_bounds=_pair(mdl["K_bounds"], "K_bounds", str(path)),
        D_bounds=_pair(mdl["D_bounds"], "D_bounds", str(path)),
        A_crit_0=float(mdl["A_crit_0"]),
        A_crit_bounds=_pair(mdl["A_crit_bounds"], "A_crit_bounds", str(path)),
        nonlinear_max_iter=int(mdl.get("nonlinear_max_iter", 50)),
        nonlinear_tol=float(mdl.get("nonlinear_tol", 1e-6)),
    )
    if "methods" in mdl:
        model.methods.update({k: str(v) for k, v in mdl["methods"].items()})
    # Phase 5 (addendum 4, sections 3 and 5.3): K- and D-type parameters declared for z in `param_z_units`
    # are scaled to the working vertical unit; z_m = z_ft / c <=> K_m = c K_ft, D_m = c D_ft
    from lemcvt.io.units import z_scale_factor

    model.param_z_units = str(mdl.get("param_z_units", units["z_units"]))
    model.z_units_working = str(units["z_units"]) if repro else "m"
    if not repro and model.param_z_units != "m":
        scale = 1.0 / z_scale_factor(model.param_z_units)      # c = 3937/1200 for ftUS
        model.param_scale = scale
        model.K0_ad *= scale
        model.K0_partitioned *= scale
        model.D0 *= scale
        model.K_bounds = (model.K_bounds[0] * scale, model.K_bounds[1] * scale)
        model.D_bounds = (model.D_bounds[0] * scale, model.D_bounds[1] * scale)
        logger.info("%s: K0, D0 and the K, D bounds declared for z in %s multiplied by %.10f for the metre grid",
                    path.name, model.param_z_units, scale)
    for name, (lo, hi) in (("K", model.K_bounds), ("D", model.D_bounds), ("A_crit", model.A_crit_bounds)):
        x0 = {"K": model.K0_ad, "D": model.D0, "A_crit": model.A_crit_0}[name]
        if not (lo <= x0 <= hi):
            raise ConfigError(f"{path}: initial {name} = {x0:g} outside bounds [{lo:g}, {hi:g}]")
    if model.m != float(mv["theta"]):
        logger.warning("%s: model.m (%g) differs from measured_v1.theta (%g)", path, model.m, mv["theta"])

    shape = tuple(int(s) for s in g["shape"])
    if len(shape) != 2:
        raise ConfigError(f"{path}: grids.shape must have two entries")
    return TileConfig(
        tile=tile, site=d["site"], grids={k: str(g[k]) for k in ("filled", "area", "fd")},
        sha256=dict(g["sha256"]), shape=shape, dx=float(g["dx"]), dy=float(g["dy"]),
        z_units=str(units["z_units"]), xy_units=str(units["xy_units"]),
        reproduce_v1_units=repro,
        measured_v1=dict(mv), model=model, v1_reference=dict(d.get("v1_reference", {})), raw=d, path=path,
    )


# --------------------------------------------------------------------------------------
# Tile data
# --------------------------------------------------------------------------------------

@dataclass
class TileData:
    """Grids of one tile as the models consume them."""

    cfg: TileConfig
    site: SiteConfig
    filled: Grid      # z in the working vertical unit (metres unless reproduce_v1_units)
    area: Grid
    fd: Grid
    z_factor: float   # factor applied to the stored z (1.0 when reproduce_v1_units)
    z_units_working: str
    sanity: dict[str, float]

    @property
    def Z(self) -> np.ndarray:
        return self.filled.data

    @property
    def A(self) -> np.ndarray:
        return self.area.data

    @property
    def FD(self) -> np.ndarray:
        return self.fd.data

    @property
    def dx(self) -> float:
        """What v1 used: the x pixel size of the geotransform."""
        return self.filled.dx

    def provenance(self) -> dict[str, Any]:
        return {
            "tile": self.cfg.tile,
            "grids": {k: str(data_root() / v) for k, v in self.cfg.grids.items()},
            "sha256": self.cfg.sha256,
            "z_units_stored": self.cfg.z_units,
            "z_units_working": self.z_units_working,
            "z_factor": self.z_factor,
            "reproduce_v1_units": self.cfg.reproduce_v1_units,
            "dx": self.dx,
            "dy": self.filled.dy,
            "shape": list(self.filled.shape),
            "sanity": self.sanity,
        }


def load_tile(tile: str | TileConfig, verify_sha: bool = True, crop: tuple[int, int, int, int] | None = None,
              sanity: bool = True, reproduce_v1_units: bool | None = None) -> TileData:
    """Read a tile's three grids, verify them, and apply the unit policy.

    Parameters
    ----------
    tile
        Tile id (e.g. ``"GM1"``) or a loaded :class:`TileConfig`.
    verify_sha
        Check each file's sha256 against the tile config (manifest) before use.
    crop
        Optional ``(row0, row1, col0, col1)`` pixel crop applied after loading (tests, smoke runs).
    sanity
        Run the site sanity checks on the metre grid (skipped for crops of tiny size).
    reproduce_v1_units
        Override of the config flag (see :func:`load_tile_config`); the config is reloaded with it.
    """
    if isinstance(tile, TileConfig) and reproduce_v1_units is not None and reproduce_v1_units != tile.reproduce_v1_units:
        tile = tile.tile
    cfg = tile if isinstance(tile, TileConfig) else load_tile_config(tile, reproduce_v1_units=reproduce_v1_units)
    site = load_site(cfg.site)
    root = data_root()

    grids: dict[str, Grid] = {}
    for key in ("filled", "area", "fd"):
        p = root / cfg.grids[key]
        if not p.exists():
            raise ConfigError(f"{cfg.tile}: grid file missing: {p} (run `lemcvt data fetch`)")
        if verify_sha:
            h = sha256_file(p)
            if h != cfg.sha256[key]:
                raise ConfigError(f"{cfg.tile}: sha256 mismatch for {p}: {h} != manifest {cfg.sha256[key]}")
        grids[key] = read_geotiff(p, dtype="uint8" if key == "fd" else "float64")

    filled, area, fd = grids["filled"], grids["area"], grids["fd"]
    for g in (area, fd):
        if g.shape != filled.shape:
            raise ConfigError(f"{cfg.tile}: grid shapes differ: {filled.shape} vs {g.shape} ({g.name})")
    if filled.shape != cfg.shape:
        raise ConfigError(f"{cfg.tile}: shape {filled.shape} != config {cfg.shape}")
    if not np.isclose(filled.dx, cfg.dx, rtol=0, atol=1e-12):
        raise ConfigError(f"{cfg.tile}: dx {filled.dx!r} != config {cfg.dx!r}")

    z_m, factor = z_to_metres(filled.data, cfg.z_units, reproduce_v1_units=cfg.reproduce_v1_units)
    working_units = cfg.z_units if cfg.reproduce_v1_units else "m"
    filled = filled.with_data(np.ascontiguousarray(z_m))

    diag: dict[str, float] = {}
    if sanity and not cfg.reproduce_v1_units:
        diag = sanity_check_elevation(filled.data, filled.dx, site.z_range_m, site.max_median_slope, label=cfg.tile)
    elif sanity:
        # In reproduce mode the stored unit may be feet; check against the site range converted back.
        from lemcvt.io.units import z_scale_factor

        f = z_scale_factor(cfg.z_units)
        lo, hi = site.z_range_m
        try:
            diag = sanity_check_elevation(filled.data, filled.dx, (lo / f, hi / f), site.max_median_slope / f,
                                          label=f"{cfg.tile} (stored units {cfg.z_units})")
        except UnitError as exc:
            raise ConfigError(str(exc)) from exc

    if crop is not None:
        from lemcvt.io.grids import crop_grid

        r0, r1, c0, c1 = crop
        filled, area, fd = (crop_grid(g, r0, r1, c0, c1) for g in (filled, area, fd))

    return TileData(cfg=cfg, site=site, filled=filled, area=area, fd=fd, z_factor=factor,
                    z_units_working=working_units, sanity=diag)


# --------------------------------------------------------------------------------------
# Slurm and data configs
# --------------------------------------------------------------------------------------

def load_slurm_config() -> dict[str, Any]:
    d = _read_yaml(config_root() / "slurm" / "sherlock.yaml")
    _require(d, ("micromamba", "env_prefix", "defaults", "steps"), "slurm/sherlock.yaml")
    return d


def load_data_config() -> dict[str, Any]:
    return _read_yaml(config_root() / "data.yaml")


def config_snapshot(cfg: TileConfig | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """JSON-serialisable snapshot of the configuration a run used."""
    snap: dict[str, Any] = {"configs_root": str(config_root())}
    if cfg is not None:
        snap["tile"] = cfg.raw
        snap["site"] = load_site(cfg.site).raw
        snap["model_resolved"] = asdict(cfg.model)
    if extra:
        snap.update(extra)
    return snap
