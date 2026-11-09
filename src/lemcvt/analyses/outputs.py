"""Model output grids, written as v1's ``save_model_outputs`` did (now GeoTIFF with CRS).

For a modelled elevation ``Z`` v1 saved six grids with one prefix:
``_elevation`` (float64), ``_filled`` (priority-flood filled, float64),
``_flow_direction`` (uint8), ``_area`` (float64), ``_logarea`` (uint8) and
``_hillshade`` (uint8, azimuth 315, inclination 20). The same set is produced here
from the exact TopoAnalysis ports in :mod:`lemcvt.hydro`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lemcvt.hydro.d8 import d8_area, d8_flow_direction
from lemcvt.hydro.derived import hillshade, log_area, to_byte_like_gdal
from lemcvt.hydro.fill import priority_flood_fill
from lemcvt.io.grids import Grid

logger = logging.getLogger(__name__)


@dataclass
class ModelOutputs:
    elevation: Grid
    filled: Grid
    flow_direction: Grid
    area: Grid
    logarea: Grid
    hillshade: Grid

    def as_dict(self) -> dict[str, tuple[Grid, str | None]]:
        return {
            "elevation": (self.elevation, None),
            "filled": (self.filled, None),
            "flow_direction": (self.flow_direction, "uint8"),
            "area": (self.area, None),
            "logarea": (self.logarea, "uint8"),
            "hillshade": (self.hillshade, "uint8"),
        }


def derive_model_outputs(template: Grid, Z_model: np.ndarray, derived: bool = True) -> ModelOutputs:
    """Build the six v1 output grids from a modelled elevation (``derived=False`` -> elevation only)."""
    z = np.asarray(Z_model, dtype=float)
    elev = template.with_data(z, name="elevation")
    if not derived:
        empty = template.with_data(np.zeros_like(z), name="unused")
        return ModelOutputs(elev, empty, empty, empty, empty, empty)
    dx = template.dx
    zf = priority_flood_fill(z, dx)
    fd = d8_flow_direction(zf)
    area = d8_area(fd, zf, dx)
    return ModelOutputs(
        elevation=elev,
        filled=template.with_data(zf, name="filled"),
        flow_direction=template.with_data(fd, name="flow_direction"),
        area=template.with_data(area, name="area"),
        logarea=template.with_data(to_byte_like_gdal(log_area(area)), name="logarea"),
        hillshade=template.with_data(to_byte_like_gdal(hillshade(z, dx)), name="hillshade"),
    )


def write_model_outputs(recorder, template: Grid, Z_model: np.ndarray, prefix: str, derived: bool = True) -> list[str]:
    """Write the output set through a :class:`lemcvt.results.RunRecorder`; returns file names."""
    outs = derive_model_outputs(template, Z_model, derived=derived)
    written = []
    items = outs.as_dict() if derived else {"elevation": (outs.elevation, None)}
    for key, (grid, dtype) in items.items():
        p = recorder.write_grid(grid, f"{prefix}_{key}", dtype=dtype)
        written.append(p.name)
    return written
