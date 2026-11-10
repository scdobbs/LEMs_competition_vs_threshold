"""Channel network and chi (Perron & Royden 2013) on the tile's own D8 area grid.

Definitions used throughout :mod:`lemcvt.measure`:

* **channel pixels**: ``A >= A_c`` (the same D8 drainage-area grid the models use);
* **outlets**: channel pixels on the tile edge (first/last row or column) and channel
  pixels whose D8 receiver is not a channel pixel. Edge pixels are the models'
  Dirichlet boundary, and the partitioned model integrates channel elevations
  upstream from every boundary cell, so chi starts at zero at the same places;
* **basins**: the set of channel pixels draining to one outlet (labelled by walking
  the D8 receivers downstream);
* **chi**: integrated upstream from each outlet with ``chi = 0`` there, along D8
  links with the reference area ``A0 = 1 m^2``. The default rule (``"upstream"``)
  is the one the partitioned model uses to integrate channel elevations,
  ``z_up = z_down + (U/K) A_up^(-m) dL`` (``lemcvt.models.common.precompute_channel_network``),
  so ``z - z_outlet = k_s * chi`` has ``k_s = U/K`` exactly for ``m = theta``, ``n = 1``.
  The trapezoid rule of Perron & Royden (2013) is available as ``rule="trapezoid"``
  and reported as a sensitivity; on a synthetic steady state it biases k_s by ~1 %;
* link lengths are ``dx`` (east/west), ``dy`` (north/south) and ``sqrt(dx^2 + dy^2)``
  (diagonal); on the AP tiles ``dy`` differs from ``dx`` by 7e-5 relative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lemcvt.hydro.d8 import d8_receiver_arrays


@dataclass
class ChannelNetwork:
    A_c: float
    mask: np.ndarray            # bool grid of channel pixels
    rows: np.ndarray            # channel pixel rows (BFS order from the outlets)
    cols: np.ndarray
    area: np.ndarray            # area of each channel pixel (same order)
    z: np.ndarray               # elevation of each channel pixel
    receiver: np.ndarray        # index into the channel arrays of the receiver, -1 for outlets
    link_length: np.ndarray     # length of the link to the receiver (0 for outlets)
    basin: np.ndarray           # basin id per channel pixel (outlet id)
    outlet_index: np.ndarray    # indices (into channel arrays) of the outlets
    n_edge_outlets: int

    @property
    def n_pixels(self) -> int:
        return int(self.rows.size)

    @property
    def n_basins(self) -> int:
        return int(self.outlet_index.size)


def channel_network(z: np.ndarray, area: np.ndarray, fd: np.ndarray, A_c: float, dx: float, dy: float | None = None,
                    min_basin_pixels: int = 1) -> ChannelNetwork:
    """Extract the channel network at threshold ``A_c`` with receivers, link lengths and basins.

    Pixels are stored in breadth-first order from the outlets upstream, so a single pass
    over the arrays visits every receiver before its donors (this does not rely on area
    ordering, which TopoAnalysis's D8 quirks can violate). Channel pixels that cannot be
    reached from an outlet (a D8 cycle between tied cells) are dropped and counted.
    """
    dy = dx if dy is None else dy
    ny, nx = area.shape
    mask0 = np.isfinite(area) & (area >= A_c) & np.isfinite(z)
    rows0, cols0 = np.nonzero(mask0)
    n0 = rows0.size
    index_grid = np.full((ny, nx), -1, dtype=np.int64)
    index_grid[rows0, cols0] = np.arange(n0)

    r_to, c_to, good = d8_receiver_arrays(fd)
    rr, cc = r_to[rows0, cols0], c_to[rows0, cols0]
    ok = good[rows0, cols0]
    receiver0 = np.full(n0, -1, dtype=np.int64)
    receiver0[ok] = index_grid[rr[ok], cc[ok]]       # -1 if the receiver is not a channel pixel
    receiver0[receiver0 == np.arange(n0)] = -1        # self-receiver (should not happen)
    on_edge = (rows0 == 0) | (rows0 == ny - 1) | (cols0 == 0) | (cols0 == nx - 1)
    receiver0[on_edge] = -1                           # edge channel pixels are roots (model boundary)

    # upstream lists within the channel network, BFS from outlets
    ups: list[list[int]] = [[] for _ in range(n0)]
    for i in range(n0):
        r = receiver0[i]
        if r >= 0:
            ups[r].append(i)
    outlets0 = np.nonzero(receiver0 < 0)[0]
    order: list[int] = []
    basin0 = np.full(n0, -1, dtype=np.int64)
    for o in outlets0.tolist():
        basin0[o] = o
        queue = [o]
        while queue:
            k = queue.pop(0)
            order.append(k)
            for u in ups[k]:
                if basin0[u] < 0:
                    basin0[u] = o
                    queue.append(u)
    order_a = np.asarray(order, dtype=np.int64)
    n_unreached = n0 - order_a.size

    # re-index in BFS order
    new_index = np.full(n0, -1, dtype=np.int64)
    new_index[order_a] = np.arange(order_a.size)
    rows, cols = rows0[order_a], cols0[order_a]
    receiver = np.where(receiver0[order_a] >= 0, new_index[np.maximum(receiver0[order_a], 0)], -1)
    basin = new_index[basin0[order_a]]
    rr, cc = r_to[rows, cols], c_to[rows, cols]
    drow = np.where(receiver >= 0, rr - rows, 0)
    dcol = np.where(receiver >= 0, cc - cols, 0)
    link = np.where((drow != 0) & (dcol != 0), np.hypot(dx, dy), np.where(drow != 0, dy, dx))
    link = np.where(receiver >= 0, link, 0.0)
    outlet_index = np.nonzero(receiver < 0)[0]
    mask = np.zeros_like(mask0)
    mask[rows, cols] = True

    if min_basin_pixels > 1:
        counts = np.bincount(basin, minlength=rows.size)
        keep = counts[basin] >= min_basin_pixels
        sub_mask = np.zeros_like(mask)
        sub_mask[rows[keep], cols[keep]] = True
        return channel_network(z, np.where(sub_mask, area, np.nan), fd, A_c, dx, dy, 1)
    edge = (rows[outlet_index] == 0) | (rows[outlet_index] == ny - 1) | (cols[outlet_index] == 0) | (cols[outlet_index] == nx - 1)
    net = ChannelNetwork(A_c=A_c, mask=mask, rows=rows, cols=cols, area=area[rows, cols].astype(float),
                         z=z[rows, cols].astype(float), receiver=receiver, link_length=link, basin=basin,
                         outlet_index=outlet_index, n_edge_outlets=int(edge.sum()))
    net.n_unreached = n_unreached  # type: ignore[attr-defined]
    return net


def chi_for_thetas(net: ChannelNetwork, thetas: np.ndarray, A0: float = 1.0, rule: str = "upstream") -> np.ndarray:
    """chi of every channel pixel for every theta: array (n_pixels, n_thetas); 0 at outlets.

    ``rule="upstream"`` (default, the partitioned model's discretisation):
        chi_i = chi_r + (A0/A_i)^theta * L_ir
    ``rule="trapezoid"`` (Perron & Royden 2013):
        chi_i = chi_r + 0.5 * ((A0/A_i)^theta + (A0/A_r)^theta) * L_ir
    Pixels are stored in BFS order from the outlets, so a single pass accumulates along the tree.
    """
    if rule not in ("upstream", "trapezoid"):
        raise ValueError("rule must be 'upstream' or 'trapezoid'")
    thetas = np.asarray(thetas, dtype=float)
    a_pow = np.power((A0 / net.area)[:, None], thetas[None, :])          # (n, k)
    chi = np.zeros_like(a_pow)
    rec = net.receiver
    L = net.link_length
    if rule == "upstream":
        for i in range(net.n_pixels):
            r = rec[i]
            if r >= 0:
                chi[i] = chi[r] + a_pow[i] * L[i]
    else:
        for i in range(net.n_pixels):
            r = rec[i]
            if r >= 0:
                chi[i] = chi[r] + 0.5 * (a_pow[i] + a_pow[r]) * L[i]
    return chi


def relative_elevation(net: ChannelNetwork) -> np.ndarray:
    """z minus the elevation of the pixel's own outlet."""
    z_out = net.z[net.outlet_index]
    outlet_of_basin = {int(o): k for k, o in enumerate(net.outlet_index)}
    idx = np.fromiter((outlet_of_basin[int(b)] for b in net.basin), dtype=np.int64, count=net.n_pixels)
    return net.z - z_out[idx]


def channel_pixel_basin_lookup(net: ChannelNetwork) -> np.ndarray:
    """Outlet-basin id of every channel pixel in the row-major order of ``net.mask``.

    ``lemcvt.measure.hilltop_curvature.basin_labels`` numbers channel pixels in row-major order of the
    mask; ``hilltop_mask(..., basin_of_channel=lookup)`` maps those labels to the network's outlet
    basins so that the divide is the one between catchments of the network, which is how ``lemcvt
    measure`` defines hilltops. The same lookup is used by the diagnostics and audits (addendum 4, 6.4.1).
    """
    ch_rows, ch_cols = np.nonzero(net.mask)
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(net.rows.tolist(), net.cols.tolist(), strict=True))}
    lookup = np.zeros(ch_rows.size, dtype=np.int64)
    for i, (r, c) in enumerate(zip(ch_rows.tolist(), ch_cols.tolist(), strict=True)):
        lookup[i] = net.basin[pos[(r, c)]]
    return lookup
