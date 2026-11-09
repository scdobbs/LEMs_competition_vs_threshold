"""lemcvt: calibration of steady-state landscape evolution models against 1 m lidar tiles.

v2 of the Dobbs & Hilley "competition vs threshold" code base. Phase 1 of
``reports/directives/refactor_v2_directive.md`` ports the v1 notebooks and
``src/`` modules to this package without changing any numerical operation; later
phases add performance work and the v2 optimizer behind flags.

Sub-packages
------------
io        GeoTIFF read/write, unit handling, data manifest
hydro     D8 flow routing, priority-flood fill, D-infinity, channel slope smoothing
models    advection-diffusion, partitioned (linear/nonlinear), synthetic planform
optimize  v1 (exact port of the notebook optimizers); v2 (Phase 4)
measure   theta, k_s, hilltop curvature; slope-area utilities
analyses  tile pipeline, resolution sweep, synthetic Peclet, D8 vs D-inf, Perron forward
figures   manuscript / SI figures (read results only)
tables    Table 1 (LaTeX + CSV)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lemcvt")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "2.0.0a0"

__all__ = ["__version__"]
