"""D8 flow graph on a tile: receivers, main-stem donors, confluences, path walks,
local channel slope with vertical-interval smoothing.

Everything here is derived from the repo's existing ``_fd`` and ``_area``
grids; no routing is recomputed.
"""

from __future__ import annotations

import numpy as np

from .tiles import Tile

# TopoAnalysis FlowDirectionD8 codes -> (d_row, d_col)
D8_OFFSETS = {
    1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
    16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
}


def receiver_grids(fd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row/col of each pixel's D8 receiver; -1 where there is none or it leaves the grid."""
    ny, nx = fd.shape
    di = np.zeros(fd.shape, dtype=int)
    dj = np.zeros(fd.shape, dtype=int)
    has = np.zeros(fd.shape, dtype=bool)
    for code, (a, b) in D8_OFFSETS.items():
        m = fd == code
        di[m] = a
        dj[m] = b
        has[m] = True
    ii, jj = np.indices(fd.shape)
    ri = ii + di
    rj = jj + dj
    inside = has & (ri >= 0) & (ri < ny) & (rj >= 0) & (rj < nx)
    return np.where(inside, ri, -1), np.where(inside, rj, -1)


class FlowGraph:
    """Receiver / main-stem-donor links plus the channel-confluence mask for one tile."""

    def __init__(self, tile: Tile, a_min: float, confluence_min_fraction: float = 0.0):
        self.tile = tile
        self.a_min = float(a_min)
        self.de = tile.pixel_size
        self.ri, self.rj = receiver_grids(tile.fd)
        ny, nx = tile.shape
        ii, jj = np.indices(tile.shape)
        ok = self.ri >= 0
        src_i, src_j = ii[ok], jj[ok]
        dst_i, dst_j = self.ri[ok], self.rj[ok]
        a_src = tile.area[ok]

        # main-stem donor = donor with the largest drainage area
        max_donor_area = np.zeros(tile.shape)
        np.maximum.at(max_donor_area, (dst_i, dst_j), a_src)
        is_main = a_src >= max_donor_area[dst_i, dst_j]
        self.ui = -np.ones(tile.shape, dtype=int)
        self.uj = -np.ones(tile.shape, dtype=int)
        self.ui[dst_i[is_main], dst_j[is_main]] = src_i[is_main]
        self.uj[dst_i[is_main], dst_j[is_main]] = src_j[is_main]

        # number of donors that are themselves channel pixels (A >= a_min)
        n_chan = np.zeros(tile.shape, dtype=int)
        np.add.at(n_chan, (dst_i, dst_j), (a_src >= self.a_min).astype(int))
        self.n_channel_donors = n_chan
        # confluence: the second-largest donor is a channel AND a significant
        # fraction of the receiver's area (a jump in A along the main stem)
        second = np.zeros(tile.shape)
        nm = ~is_main
        np.maximum.at(second, (dst_i[nm], dst_j[nm]), a_src[nm])
        self.second_donor_area = second
        thresh = np.maximum(self.a_min, float(confluence_min_fraction) * tile.area)
        self.confluence = (n_chan >= 2) & (second >= thresh)
        self.channel = tile.area >= self.a_min

    # -- geometry helpers ---------------------------------------------------
    def step_length(self, i, j, i2, j2) -> float:
        return self.de * (np.sqrt(2.0) if (i != i2 and j != j2) else 1.0)

    def down(self, i, j):
        return int(self.ri[i, j]), int(self.rj[i, j])

    def up(self, i, j):
        return int(self.ui[i, j]), int(self.uj[i, j])

    def walk(self, i, j, max_dist: float, direction: str) -> list[tuple[int, int, float]]:
        """Pixels along the main stem from (i, j), with cumulative path distance (m).

        ``direction`` is ``"down"`` (receivers) or ``"up"`` (largest-area donors).
        The start pixel is included at distance 0.
        """
        nxt = self.down if direction == "down" else self.up
        out = [(i, j, 0.0)]
        dist = 0.0
        while True:
            i2, j2 = nxt(i, j)
            if i2 < 0:
                break
            dist += self.step_length(i, j, i2, j2)
            if dist > max_dist:
                break
            out.append((i2, j2, dist))
            i, j = i2, j2
        return out

    def edge_distance(self, i, j) -> float:
        ny, nx = self.tile.shape
        return float(min(i, j, ny - 1 - i, nx - 1 - j)) * self.de

    def distance_to_confluence(self, i, j, max_dist: float) -> float:
        """Shortest along-path distance (up the main stem or downstream) to a confluence pixel."""
        best = np.inf
        for direction in ("down", "up"):
            for (a, b, d) in self.walk(i, j, max_dist, direction):
                if self.confluence[a, b]:
                    best = min(best, d)
                    break
        return best

    # -- slope ---------------------------------------------------------------
    def local_slope(self, i, j, vertical_interval: float) -> float:
        """Channel slope by vertical-interval smoothing.

        Re-implements the path construction of TopoAnalysis
        ``ChannelSlopeWithSmoothing`` (vertical_interval mode): starting from the
        pixel, extend one step upstream (largest-area donor) and one step
        downstream alternately until the elevation difference between the two
        ends is at least ``vertical_interval``; slope is that difference divided
        by the geometric path length between the ends. NaN when the path cannot
        be built (edge of grid, no donor, or the two ends meet).
        """
        z = self.tile.z
        ui, uj = self.up(i, j)
        di, dj = self.down(i, j)
        if ui < 0 or di < 0:
            return np.nan
        length = self.step_length(ui, uj, i, j) + self.step_length(i, j, di, dj)
        delta = z[ui, uj] - z[di, dj]
        while delta < vertical_interval:
            ui2, uj2 = self.up(ui, uj)
            di2, dj2 = self.down(di, dj)
            if ui2 < 0 or di2 < 0 or (ui2 == di2 and uj2 == dj2):
                return np.nan
            length += self.step_length(ui2, uj2, ui, uj) + self.step_length(di, dj, di2, dj2)
            ui, uj, di, dj = ui2, uj2, di2, dj2
            delta = z[ui, uj] - z[di, dj]
        return float(delta / length)

    def local_slope_horizontal(self, i, j, half_length: float) -> float:
        """Slope over the main-stem path +/- half_length around the pixel (fallback for low-gradient trunks)."""
        up = self.walk(i, j, half_length, "up")
        down = self.walk(i, j, half_length, "down")
        if len(up) < 2 or len(down) < 2:
            return np.nan
        (ui, uj, du) = up[-1]
        (di, dj, dd) = down[-1]
        return float((self.tile.z[ui, uj] - self.tile.z[di, dj]) / (du + dd))

    def slope_with_fallback(self, i, j, vertical_interval: float, horizontal_fallback: float) -> tuple[float, str]:
        """Vertical-interval slope, or the horizontal-interval slope where the drop cannot be reached."""
        s = self.local_slope(i, j, vertical_interval)
        if np.isfinite(s):
            return s, "vertical_interval"
        return self.local_slope_horizontal(i, j, horizontal_fallback), "horizontal_fallback"

    def reach_median_slope(self, i, j, half_length: float, vertical_interval: float,
                           horizontal_fallback: float = 0.0) -> tuple[float, int]:
        """Median local slope over channel pixels within +/- half_length along the path (excluding (i, j))."""
        vals = []
        for direction in ("down", "up"):
            for (a, b, _d) in self.walk(i, j, half_length, direction)[1:]:
                if self.channel[a, b]:
                    s = self.local_slope(a, b, vertical_interval)
                    if not np.isfinite(s) and horizontal_fallback > 0:
                        s = self.local_slope_horizontal(a, b, horizontal_fallback)
                    if np.isfinite(s):
                        vals.append(s)
        if not vals:
            return np.nan, 0
        return float(np.median(vals)), len(vals)
