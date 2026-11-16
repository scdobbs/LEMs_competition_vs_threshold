"""Channel slope smoothed over a vertical interval: exact port of TopoAnalysis
``ChannelSlopeWithSmoothing`` (and the ``AlongFlowSmoothing`` path finder).

For every channel pixel (``A > min_area``, strict) the routine walks one step
downstream and one upstream along the D8 network, extending both ends until the
elevation difference between the two ends reaches ``vertical_interval``, and
returns ``dz / along-path length``. The "upstream" neighbour of a cell is the
one with the largest drainage area among the cells draining into it (the last
assignment in ascending-area order), i.e. the main stem.

Quirks reproduced on purpose: the diagonal-length adjustment of 0.414 is applied
to the step *after* each detected diagonal transition and never to the first
step; a path whose upstream end is -1 or coincides with the downstream end is
discarded (NaN).
"""

from __future__ import annotations

import numpy as np

from lemcvt.hydro.d8 import d8_receiver_arrays


def flow_graph_main_stem(area: np.ndarray, fd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(upstream_i, upstream_j, downstream_i, downstream_j)`` as TopoAnalysis builds them.

    Cells are visited in ascending area order (``argsort(axis=None)``); each sets
    its receiver's upstream pointer, so the largest-area tributary wins. Among
    tributaries of equal area the one with the highest flat (row-major) index wins
    (visited last); see ``docs/numerics_changes.md``, N3. TopoAnalysis left this to
    numpy's unstable default sort, which is what produced the two drift rows of the
    Phase 2 resolution sweep.
    """
    ny, nx = area.shape
    upstream_i = np.full((ny, nx), -1, dtype=np.int64)
    upstream_j = np.full((ny, nx), -1, dtype=np.int64)
    downstream_i = np.full((ny, nx), -1, dtype=np.int64)
    downstream_j = np.full((ny, nx), -1, dtype=np.int64)
    area_f = np.asarray(area, dtype=np.float64).ravel()
    order = np.lexsort((np.arange(area_f.size), area_f))   # area ascending, ties by ascending flat index
    r_to, c_to, good = d8_receiver_arrays(fd)
    i_s, j_s = np.unravel_index(order, (ny, nx))
    for i, j in zip(i_s.tolist(), j_s.tolist(), strict=True):
        if good[i, j]:
            ds_i, ds_j = r_to[i, j], c_to[i, j]
            downstream_i[i, j] = ds_i
            downstream_j[i, j] = ds_j
            upstream_i[ds_i, ds_j] = i
            upstream_j[ds_i, ds_j] = j
    return upstream_i, upstream_j, downstream_i, downstream_j


def _points_vertical(z, up_i, up_j, dn_i, dn_j, this_i, this_j, vertical_interval):
    ret = [(this_i, this_j)]
    ups_i, ups_j = up_i[this_i, this_j], up_j[this_i, this_j]
    ds_i, ds_j = dn_i[this_i, this_j], dn_j[this_i, this_j]
    if (ups_i < 0) or (ds_i < 0):
        return None
    ret = [(ds_i, ds_j)] + ret + [(ups_i, ups_j)]
    delta_e = z[ups_i, ups_j] - z[ds_i, ds_j]
    while (delta_e < vertical_interval) & (ups_i >= 0):
        ups_i, ups_j = up_i[ups_i, ups_j], up_j[ups_i, ups_j]
        ds_i, ds_j = dn_i[ds_i, ds_j], dn_j[ds_i, ds_j]
        if (ups_i < 0) or (ds_i < 0) or ((ups_i == ds_i) and (ups_j == ds_j)):
            return None
        ret = [(ds_i, ds_j)] + ret + [(ups_i, ups_j)]
        delta_e = z[ups_i, ups_j] - z[ds_i, ds_j]
    return ret


def _points_horizontal(z, de, up_i, up_j, dn_i, dn_j, this_i, this_j, horizontal_interval):
    horizontal_distance = 0.0
    ret = [(this_i, this_j)]
    ups_i, ups_j = up_i[this_i, this_j], up_j[this_i, this_j]
    ds_i, ds_j = dn_i[this_i, this_j], dn_j[this_i, this_j]
    if (ups_i < 0) or (ds_i < 0):
        return None
    ret = [(ds_i, ds_j)] + ret + [(ups_i, ups_j)]
    horizontal_distance += ((1.0 if ((ds_i == this_i) or (ds_j == this_j)) else 1.414)
                            + (1.0 if ((ups_i == this_i) or (ups_j == this_j)) else 1.414)) * de[this_i, this_j]
    while (horizontal_distance < horizontal_interval) & (ups_i >= 0):
        ups_i, ups_j = up_i[ups_i, ups_j], up_j[ups_i, ups_j]
        ds_i, ds_j = dn_i[ds_i, ds_j], dn_j[ds_i, ds_j]
        if (ups_i < 0) or (ds_i < 0) or ((ups_i == ds_i) and (ups_j == ds_j)):
            return None
        horizontal_distance += (1.0 if ((ds_i == ret[0][0]) or (ds_j == ret[0][1])) else 1.414) * de[ds_i, ds_j] \
            + (1.0 if ((ups_i == ret[-1][0]) or (ups_j == ret[-1][1])) else 1.414) * de[ups_i, ups_j]
        ret = [(ds_i, ds_j)] + ret + [(ups_i, ups_j)]
    return ret


def _slope_from_points(points, z, de):
    pts = list(zip(*points, strict=True))
    arr = np.array(pts).astype(int)
    adjustment = np.ones(len(arr[0]))
    ind = np.where((arr[0, 1:-1] != arr[0, 2:]) & (arr[1, 1:-1] != arr[1, 2:]))
    adjustment[ind[0] + 1] += 0.414
    elevation_profile = z[arr[0], arr[1]]
    de_profile = de[arr[0], arr[1]]
    dxs = np.sum(de_profile * adjustment)
    dy = elevation_profile[-1] - elevation_profile[0]
    return dy / dxs


def channel_slope_with_smoothing(
    z: np.ndarray,
    area: np.ndarray,
    fd: np.ndarray,
    dx: float,
    vertical_interval: float | None = 10.0,
    horizontal_interval: float | None = None,
    min_area: float = 1e6,
) -> np.ndarray:
    """Slope grid (NaN off-channel), identical to ``ChannelSlopeWithSmoothing._griddata``.

    Parameters
    ----------
    z, area, fd
        Elevation, drainage area and D8 codes (same shape).
    dx
        Pixel size (``_mean_pixel_dimension`` in TopoAnalysis).
    vertical_interval
        Elevation drop (same unit as ``z``) the path must span; default 10 as in v1.
    horizontal_interval
        Alternative along-path length criterion (used if ``vertical_interval`` is None).
    min_area
        Channel pixels satisfy ``area > min_area`` (strict).
    """
    z = np.asarray(z, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    de = dx * np.ones_like(z, dtype=np.float64)
    up_i, up_j, dn_i, dn_j = flow_graph_main_stem(area, fd)

    S = np.full(z.shape, np.nan)
    sel = (area != 0) & ~np.isnan(area) & ~np.isnan(z) & (area > min_area)
    ii, jj = np.where(sel)
    for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
        if vertical_interval is not None:
            pts = _points_vertical(z, up_i, up_j, dn_i, dn_j, i, j, vertical_interval)
        else:
            if horizontal_interval is None:
                raise ValueError("give vertical_interval or horizontal_interval")
            pts = _points_horizontal(z, de, up_i, up_j, dn_i, dn_j, i, j, horizontal_interval)
        if pts is None or len(pts) == 0:
            continue
        S[i, j] = _slope_from_points(pts, z, de)
    return S
