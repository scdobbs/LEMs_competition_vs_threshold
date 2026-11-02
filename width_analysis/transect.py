"""Local flow line and perpendicular cross-section through a channel pixel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .network import FlowGraph
from .tiles import Tile


@dataclass
class Transect:
    s: np.ndarray        # distance along the transect (m); 0 at the channel pixel centre
    z: np.ndarray        # bilinear elevation (m)
    x: np.ndarray        # map x of each sample
    y: np.ndarray        # map y of each sample
    direction: np.ndarray   # unit vector of the local flow line (downstream)
    normal: np.ndarray      # unit vector of the transect (left bank looking downstream is +s)
    trace_xy: np.ndarray    # (n, 2) channel-trace coordinates used for the line fit
    trace_rc: list          # [(row, col), ...] of the trace, upstream first
    trace_length_m: float   # path length of the trace
    azimuth_deg: float      # azimuth of the flow line, degrees clockwise from north


def channel_trace(graph: FlowGraph, row: int, col: int, half_length: float):
    """Main-stem trace through the pixel: ``half_length`` m upstream and downstream, ordered downstream.

    Returns (xy array (n, 2), list of (row, col), path length in m).
    """
    up = graph.walk(row, col, half_length, "up")
    down = graph.walk(row, col, half_length, "down")
    pts = [(i, j) for (i, j, _) in reversed(up[1:])] + [(i, j) for (i, j, _) in down]
    length = (up[-1][2] if up else 0.0) + (down[-1][2] if down else 0.0)
    rr = np.array([p[0] for p in pts])
    cc = np.array([p[1] for p in pts])
    x, y = graph.tile.rc_to_xy(rr, cc)
    return np.column_stack([x, y]), pts, float(length)


def flow_line_direction(trace_xy: np.ndarray) -> np.ndarray:
    """Unit direction of the best-fit line (total least squares / PCA), oriented downstream.

    D8 directions are quantized to 45 degrees; fitting a line through +/- 10 m
    of trace gives a continuous local orientation.
    """
    if len(trace_xy) < 2:
        raise ValueError("trace needs at least two points")
    c = trace_xy - trace_xy.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    chord = trace_xy[-1] - trace_xy[0]
    if np.dot(d, chord) < 0:
        d = -d
    return d / np.linalg.norm(d)


def sample_bilinear(tile: Tile, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of the DEM at map coordinates (pixel-centre registration)."""
    col = (np.asarray(x) - tile.x0) / tile.dx - 0.5
    row = (tile.y0 - np.asarray(y)) / tile.dy - 0.5
    return map_coordinates(tile.z, [row, col], order=1, mode="nearest")


def make_transect(tile: Tile, graph: FlowGraph, row: int, col: int, cfg: dict) -> Transect:
    trace, trace_rc, trace_len = channel_trace(graph, row, col, cfg["trace_half_length_m"])
    d = flow_line_direction(trace)
    n = np.array([-d[1], d[0]])             # +90 deg: left bank when looking downstream
    half = cfg["transect_half_length_m"]
    step = cfg["transect_step_m"]
    s = np.arange(-half, half + 0.5 * step, step)
    xc, yc = tile.rc_to_xy(row, col)
    x = xc + s * n[0]
    y = yc + s * n[1]
    z = sample_bilinear(tile, x, y)
    az = float(np.degrees(np.arctan2(d[0], d[1])) % 360.0)
    return Transect(s=s, z=z, x=x, y=y, direction=d, normal=n, trace_xy=trace, trace_rc=trace_rc,
                    trace_length_m=trace_len, azimuth_deg=az)
