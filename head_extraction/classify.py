"""Per-head measurements, the drop rules, and the QC classes of directive section 4.

Measurements reuse the Test 1 helpers: largest-donor upslope path and
``S_up_L`` (``acrit_slope.heads.path_slopes``), the 10 m block plan-curvature
grid, and the trunk-distance idea of ``acrit_slope.qc_diagnostics``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from acrit_slope.heads import path_slopes, walk_path
from width_analysis.network import FlowGraph
from width_analysis.tiles import Tile

CLASSES = ["valley_floor", "bench_plateau", "rill", "road_ditch", "hollow", "unclassified"]
DROP_REASONS = ["tile_edge", "path_leaves_tile"]


def flow_azimuth(graph: FlowGraph, i: int, j: int, length_m: float) -> float:
    """Azimuth (degrees clockwise from north) of the vector from the head to the point
    ``length_m`` down the D8 receiver path; NaN if the path is shorter."""
    d, _, rc = walk_path(graph, i, j, "down", length_m)
    if d[-1] < length_m:
        return np.nan
    x0, y0 = graph.tile.rc_to_xy(i, j)
    x1, y1 = graph.tile.rc_to_xy(rc[-1][0], rc[-1][1])
    return float(np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360.0)


def downslope_slope(graph: FlowGraph, i: int, j: int, length_m: float) -> float:
    d, z, _ = walk_path(graph, i, j, "down", length_m)
    if d[-1] < length_m:
        return np.nan
    zl = float(np.interp(length_m, d, z))
    return (float(graph.tile.z[i, j]) - zl) / length_m


def measure(tile: Tile, graph: FlowGraph, rows, cols, ext: dict, qc: dict, curv) -> pd.DataFrame:
    """One record per head pixel: position, A_D8, slopes, azimuth, curvature, trunk distance, drop verdict."""
    cgrid, nblk, _ = curv
    d_trunk = distance_transform_edt(~(tile.area >= qc["trunk_area_m2"]), sampling=(tile.dy, tile.dx)) \
        if (tile.area >= qc["trunk_area_m2"]).any() else np.full(tile.shape, np.inf)
    ny, nx = tile.shape
    L = [float(v) for v in ext["L_values_m"]]
    recs = []
    for i, j in zip(np.asarray(rows, int), np.asarray(cols, int)):
        x, y = tile.rc_to_xy(i, j)
        edge = float(min(i, j, ny - 1 - i, nx - 1 - j)) * tile.pixel_size
        s = path_slopes(graph, i, j, L)
        bi, bj = i // nblk, j // nblk
        pc = float(cgrid[bi, bj]) if (bi < cgrid.shape[0] and bj < cgrid.shape[1]) else np.nan
        rec = dict(row=int(i), col=int(j), x=float(x), y=float(y), area_D8_m2=float(tile.area[i, j]),
                   z_m=float(tile.z[i, j]), edge_distance_m=edge, plan_curvature_10m=pc,
                   convergent_10m=bool(np.isfinite(pc) and pc > 0), dist_to_trunk_m=float(d_trunk[i, j]),
                   flow_azimuth_deg=flow_azimuth(graph, i, j, qc["flow_azimuth_length_m"]),
                   S_down_10=downslope_slope(graph, i, j, 10.0))
        rec.update(s)
        rec["flag_tile_edge"] = edge < ext["edge_buffer_m"]
        # the upslope path "leaves the tile" if it reaches the grid edge before min_upslope_path_m;
        # a path that ends at an interior divide (no donor) within that distance is kept and flagged short
        du, _, rc_up = walk_path(graph, i, j, "up", ext["min_upslope_path_m"])
        ti, tj = rc_up[-1]
        at_edge = ti == 0 or tj == 0 or ti == ny - 1 or tj == nx - 1
        rec["flag_path_leaves_tile"] = bool(s["upslope_path_m"] < ext["min_upslope_path_m"] and at_edge)
        rec["flag_short_upslope_path"] = bool(s["upslope_path_m"] < ext["min_upslope_path_m"] and not at_edge)
        rec["drop_reason"] = next((r for r in DROP_REASONS if rec[f"flag_{r}"]), "")
        rec["retained"] = rec["drop_reason"] == ""
        recs.append(rec)
    return pd.DataFrame(recs)


def _angle_diff(a, b):
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d


def _line_fit_residual(xy: np.ndarray):
    """PCA line through points; returns (RMS perpendicular residual, axial direction unit vector)."""
    c = xy - xy.mean(axis=0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    res = c @ n
    return float(np.sqrt(np.mean(res**2))), d


def classify(df: pd.DataFrame, qc: dict, Ld: str = "S_up_10", road_mask=None) -> pd.DataFrame:
    """Assign one class per retained head, rules tested in the directive's order."""
    h = df.copy()
    h["head_class"] = ""
    kept = h[h.retained]
    if len(kept) == 0:
        return h
    xy = kept[["x", "y"]].to_numpy()
    az = kept["flow_azimuth_deg"].to_numpy()
    idx = kept.index.to_numpy()
    classes, detail = [], []
    for n, (k, r) in enumerate(kept.iterrows()):
        dist = np.hypot(xy[:, 0] - r.x, xy[:, 1] - r.y)
        dist[n] = np.inf
        # valley_floor
        if r.dist_to_trunk_m <= qc["trunk_buffer_m"] and r[Ld] < qc["floor_slope_max"]:
            classes.append("valley_floor"); detail.append(""); continue
        # bench_plateau
        flat20 = np.isfinite(r.S_up_20) and r.S_up_20 < qc["plateau_slope_max"]
        flat10_convex = (np.isfinite(r.S_up_10) and r.S_up_10 < qc["plateau_slope_max"]
                         and np.isfinite(r.S_down_10) and r.S_down_10 > qc["plateau_drop_slope_min"])
        if flat20 or flat10_convex:
            classes.append("bench_plateau"); detail.append("S_up_20<thr" if flat20 else "flat10+convex"); continue
        # rill
        near = dist <= qc["rill_search_m"]
        if r[Ld] > qc["rill_slope_min"] and np.isfinite(r.flow_azimuth_deg):
            par = near & np.isfinite(az) & (_angle_diff(az, r.flow_azimuth_deg) <= qc["rill_azimuth_tol_deg"])
            if par.sum() >= qc["rill_min_neighbours"]:
                classes.append("rill"); detail.append(f"{int(par.sum())} parallel neighbours"); continue
        # road_ditch
        if road_mask is not None and road_mask[int(r.row), int(r.col)]:
            classes.append("road_ditch"); detail.append("road buffer"); continue
        near60 = dist <= qc["road_search_m"]
        if near60.sum() >= qc["road_min_heads"] and np.isfinite(r.flow_azimuth_deg):
            pts = np.vstack([xy[near60], [[r.x, r.y]]])
            rms, d = _line_fit_residual(pts)
            fall = np.array([np.sin(np.radians(r.flow_azimuth_deg)), np.cos(np.radians(r.flow_azimuth_deg))])
            ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(d, fall)), 0, 1))))
            if rms < qc["road_max_residual_m"] and ang > qc["road_min_angle_to_fall_line_deg"]:
                classes.append("road_ditch"); detail.append(f"{int(near60.sum())} collinear, rms {rms:.1f} m, {ang:.0f} deg to fall line"); continue
        # hollow
        if r.convergent_10m:
            classes.append("hollow"); detail.append(""); continue
        classes.append("unclassified"); detail.append("not convergent at 10 m")
    h.loc[idx, "head_class"] = classes
    h.loc[idx, "class_detail"] = detail
    return h
