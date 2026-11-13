"""Tile rasters for the width analysis, read through the lemcvt loader (addendum 8, 1.4: the Phase 7 port).

The feature branch read the GeoTIFFs directly, so AP elevations were in US survey feet while x, y were metres; here
every grid comes from :func:`lemcvt.config.load_tile`, which converts z to metres, verifies the manifest and applies the
site sanity checks, so the width measures on AP are in one unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lemcvt.config import REPO_ROOT  # noqa: F401  (re-exported for the drivers' git-hash helper)
from lemcvt.config import load_tile as _load_tile


@dataclass
class Tile:
    site: str
    tile: int
    z: np.ndarray          # filled elevation in the working unit (metres), float64, (ny, nx)
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


def load_tile(site: str, tile: int) -> Tile:
    """The tile ``<site><tile>`` through the lemcvt loader (metres, manifest-verified)."""
    td = _load_tile(f"{site}{int(tile)}")
    g = td.filled
    t = g.transform
    crs = ""
    if g.crs is not None:
        try:
            crs = g.crs.to_string()
        except AttributeError:
            crs = str(g.crs)
    return Tile(site, int(tile), np.asarray(td.Z, dtype=np.float64), np.asarray(td.A, dtype=np.float64), np.asarray(td.FD).astype(np.uint8),
                float(t.a), float(-t.e), float(t.c), float(t.f), crs)
