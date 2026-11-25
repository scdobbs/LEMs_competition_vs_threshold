#!/usr/bin/env python3
"""Addendum 23 (Tennessee Valley on one clipped domain, the road check, a 1 m head finder, a fair measure of channelization):
shared constants and helpers for the a23_* scripts.

Directive `reports/directives/2026-11-24T0915PST_v2_addendum_23_ttv_domain_roads_heads_channelization.md`. Decisions of its
section 2 and the readings of its section 7 are constants here, fixed before any result.

Package change (allowed by directive 1 "Code"): `extract_grid_data(..., outlets=, noflux_outside=)` carried by the AD and
partitioned models and by `PartitionedProfiler` (docs/numerics_changes.md N14; tests/test_domain.py). Everything else is here.

Choices made in applying the directive (every one is listed in the report's state check):
  - The domain window: the M&D polygon's bounding box plus WINDOW_MARGIN_M on every side, read from the full-site 1 m grids
    (filled, area_cells, fd; EPSG:6339; origin 529495 / 4198505). The window is checked to hold every domain basin (no domain cell
    on the window edge) and the final arrays are cut to the domain's bounding box plus one NaN cell on every side.
  - Outlets (2.1): polygon cells with drainage area >= 1e4 m2 whose D8 receiver is outside the polygon (or absent). The domain is
    the union of the cells draining to those outlets (upstream labelling along D8). An outlet whose receiver is itself a domain
    cell (its path leaves the polygon and re-enters) is "nested": it is a domain cell, not a fixed cell; the fixed cells are the
    outlets whose receiver is outside the domain. FD at the fixed outlets is set to 0; everywhere else the full-grid D8 is kept
    (every domain cell's donors are domain cells by construction, so the full-grid area is the domain area).
  - The polygon on the lattice: `study_area` rasterised with ALL_TOUCHED=FALSE (cell centres), as in addenda 19-22.
  - Roads (2.2): TIGER/Line 2024 ROADS for Marin County (every MTFCC class the file holds), reprojected to EPSG:6339, rasterised with
    ALL_TOUCHED=TRUE on the domain lattice; distances by Euclidean distance transform. "Within 15 m of a road" is distance <= 15 m.
    The file holds no trail-class feature (S1500, S1710, S1820, S1830) inside or near the study area; this is stated in the report.
  - The observed 12 m criterion networks (the split models' channels when no 1 m finder is accepted) come from the addendum 21
    full-site 12 m statistics (grid12_arrays.npz: M, Mdon at alpha = 2 and 0): a 12 m head cell at C seeds the largest-area 1 m cell
    of its block (full-grid area) and the 1 m channel set is every domain cell at or downstream of a seed, on the domain D8.
  - The mapped-head network: the addendum 21 pick seeds (largest-area 1 m cell of the pick's 12 m block) traced downstream likewise.
  - K0 = U / k_s on the domain: `ks_at_theta` (pooled through-origin chi-z at theta = m above A_c = 1e3 m2, the addendum 17 Part B
    estimator) with the basins labelled by their fixed outlet; crest curvature: the addendum 19/20 hilltop set on the 1e3 m2 network
    (edge margin 2, slope <= 0.4, L_f/2 from the network, 10 m road buffer, no segment filter: the addendum 20 D0_N set), median
    -laplacian at L_f = 11 m with the segment bootstrap; relief D: those crest pixels inside the polygon traced to the mapped-head
    network, D = U L_H^2 / (2 R), segment-bootstrap median (addendum 21 section 2.2).
  - The 1 m head finder (2.4): T = A S^2 with S the steepest-descent slope over 10 m downstream (`valley_slope_reach` on the D8 of
    the surface used), the upstream maximum M of T along that D8, a channel where M >= C and a head where no donor has M >= C
    (`criterion_sets`). For the multiple-flow-direction candidate A is the Freeman (1991) area (exponent 1.1) on the filled 1 m
    surface and the D8 is the surface's own; for the smoothed candidates the surface is Gaussian-smoothed (NaN-aware, sigma 3 and
    6 m), filled and D8-routed, and A, S and the D8 are the smoothed surface's. The reference is the addendum 22 12 m method.
  - The statistic at a pick: the largest value of T (and of A) over the 1 m cells within PICK_RADIUS_M of the pick (1 m cells of a
    hollow carry parallel threads; addendum 20 section 12); the 12 m reference uses the largest-area cell of the 3 x 3 block as in
    addendum 21. Calibration = the median of the picks' T on the calibration half; the F1-optimal threshold on the calibration half
    is reported beside it as a sensitivity (it is not the reading).
  - Halves (2.4): "TTV2 ground" = the picks inside the TTV2 tile footprint (34); "TTV1 + TTV3 ground" = inside TTV1 or TTV3 (27).
    Heads are counted on the same ground (the test half's footprints intersected with the domain and the polygon).
  - Power spectrum (2.5): the largest axis-aligned square of valid domain cells (largest inscribed square by the distance-transform
    method); plane detrend by least squares; 2-D Hann window (product of two 1-D Hann windows); periodogram |F|^2 normalised by the
    window power; binned by radial wavelength in N_WAVELENGTH_BINS logarithmic bins from 4 to 500 m (mean power per bin); the ratio
    is the model's binned power over the lidar's. The roll-off wavelength: a power law is fitted to the lidar's binned spectrum over
    wavelengths >= ROLLOFF_FIT_MIN_M, and the roll-off is the longest wavelength below which the lidar power stays under half the
    extrapolated power law (reported, not a reading).
  - Valley share (2.5): contour curvature of the Gaussian-smoothed surface (Perron et al. 2012 eq. 1, positive = convergent) after
    NaN cells are filled by nearest neighbour; the threshold is the lidar's mean + 2 sd over domain cells, applied to every surface;
    shares are over domain cells.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("A21_RUN", "20261123T184712Z_a21_ttv_channel_head_test")
os.environ.setdefault("A22_RUN", "20261124T000138Z_a22_ttv_cleanup_head_metrics")
from a22_common import (  # noqa: E402,F401  # noqa: E402
    A_F,
    ALPHA_PRIMARY,
    BIG,
    BLOCK_M,
    C_F,
    COLOUR,
    D_PHYS,
    D_RELIEF_RANGE,
    DCOL,
    DROW,
    FITTED,
    K_PHYS_RANGE,
    M_SITE,
    MARKER,
    MD_GPKG,
    PLAIN,
    RESCALE,
    RESULTS_ROOT,
    SCRATCH,
    T_PRIMARY_M,
    TOLERANCES_M,
    U_NEW,
    WHAT,
    a21_path,
    a22_run_dir,
    block_mean,
    channel_heads,
    classify_extra,
    coarse_grid,
    coarse_products,
    criterion_sets,
    donor_slope,
    dump,
    fit_terms,
    head_statistic_coarse,
    heads_from_products,
    hillshade,
    json_ready,
    load_json,
    mask_from_entries,
    match_heads,
    metre_tile,
    mse_at,
    partitioned_surface,
    receiver_flat,
    reduce_mask,
    slurm_record,
    terms_of,
)
from a22_common import out_path as a22_out_path

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DIRECTIVE = "reports/directives/2026-11-24T0915PST_v2_addendum_23_ttv_domain_roads_heads_channelization.md"
A22_REPORT = "reports/v2/20261124T0830PST_ttv_cleanup_head_metrics.md"
FIG_DIR = os.path.join(REPO, "reports", "v2", "figures", "a23")
JOB_DIR = os.path.join(SCRATCH, "lemcvt_jobs", "a23")
A15 = os.path.join(RESULTS_ROOT, "20261118T160326Z_a15_stage_b")
TTV_FULL = {k: os.path.join(A15, "grid_TTV", f"TTV_1m_{k}.tif") for k in ("filled", "fd", "area_cells")}
FULL_ORIGIN = (529495.0, 4198505.0)         # x0, y1 of the full-site grid (EPSG:6339), 1 m cells, 19010 x 19010
TIGER_MARIN = os.path.join(os.environ.get("GROUP_HOME", "/home/groups/hilley"), "lemcvt_data", "source", "TIGER2024_roads", "tl_2024_06041_roads.zip")
HEADS_CSV = os.path.join(REPO, "data", "ttv", "ttv_heads_manual_steve_20261119.csv")
EPSG = 6339

# ----------------------------------------------------------------------------- decision 2.1 (the domain)
OUTLET_A_MIN_M2 = 1.0e4
WINDOW_MARGIN_M = 1500.0
DOMAIN_CHECK_TOL = 1e-8
U = U_NEW                                    # 6.5e-5 m/yr
M = M_SITE                                   # 0.33
TILES = ["TTV1", "TTV2", "TTV3"]
KS_A_C = 1e3                                 # m2, the chi-z network (addendum 17 Part B estimator)
HILLTOP_A_C = 1e3                            # the addendum 19 fixed network for the hilltop set
W_F = 5                                      # L_f = 11 m
ROAD_BUFFER_HILLTOP_M = 10.0
N_BOOT_SEG = 2000
SEED = 20261124

# ----------------------------------------------------------------------------- decision 2.2 (roads)
ROAD_HEAD_BUFFER_M = 15.0
ROAD_TRUNK_BUFFER_M = 5.0
ROAD_BUFFER_SENS_M = (5.0, 15.0, 25.0)
TRUNK_A_M2 = 1.0e5
ROAD_HALF = 0.5                              # reading: at least half

# ----------------------------------------------------------------------------- decision 2.4 (the 1 m head finder)
MFD_EXPONENT = 1.1
SMOOTH_SIGMAS_M = (3.0, 6.0)
REACH_M = 10.0
PICK_RADIUS_M = 3.0
PICK_RADIUS_SENS_M = (1.0, 5.0)
CANDIDATES = ["mfd", "d8s3", "d8s6", "ref12"]
CANDIDATE_PLAIN = {"mfd": "multiple-flow-direction area on the 1 m surface", "d8s3": "D8 on the 1 m surface smoothed 3 m",
                   "d8s6": "D8 on the 1 m surface smoothed 6 m", "ref12": "the 12 m method of Addendum 22 (reference)"}
HALVES = {"TTV2": ["TTV2"], "TTV13": ["TTV1", "TTV3"]}
HALF_PLAIN = {"TTV2": "TTV2 ground", "TTV13": "TTV1 + TTV3 ground"}
HA_SHARE_FOUND_MIN = 0.6
HA_SHARE_REAL_MIN = 0.5

# ----------------------------------------------------------------------------- decision 2.5 (channelization)
WAVELENGTH_RANGE_M = (4.0, 500.0)
N_WAVELENGTH_BINS = 14
READING_BAND_M = (20.0, 200.0)
RATIO_BAND = (0.5, 2.0)
ROLLOFF_FIT_MIN_M = 100.0
VALLEY_SIGMAS_M = (10.0, 5.0)
VALLEY_K_SD = 2.0
VALLEY_SHARE_RANGE = (0.67, 1.5)

# ----------------------------------------------------------------------------- decision 2.6 (the five models) and plain names
MODELS = ["obs", "AD", "PF_C", "PF_A", "F2_relief", "SJ_C"]
FITTED_MODELS = ["AD", "PF_C", "PF_A", "SJ_C"]
MAP_ORDER = ["obs", "AD", "PF_C", "PF_A", "F2_relief", "SJ_C"]
SJC_THIN = 400
SJC_REFINE_MAX = 400
SJC_A_RANGE = (1e3, 1e5)
AD_N_GRID = 200


def a23_run_dir() -> str:
    rid = os.environ.get("A23_RUN")
    if rid is None or rid == "A23_RUN_PLACEHOLDER":
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a23_ttv_domain_roads_heads_channelization"
        os.environ["A23_RUN"] = rid
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def out_path(*parts: str) -> str:
    p = os.path.join(a23_run_dir(), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def a22_path(*parts: str) -> str:
    return os.path.join(a22_run_dir(), *parts)


# ----------------------------------------------------------------------------- rasters and vectors

def read_window(path: str, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as ds:
        return ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))


def full_grid_meta() -> dict[str, Any]:
    import rasterio
    with rasterio.open(TTV_FULL["filled"]) as ds:
        return {"shape": [ds.height, ds.width], "transform": list(ds.transform)[:6], "crs": ds.crs.to_string(), "nodata": ds.nodata}


def rasterize_layer(gpkg_or_path: str, layer: str | int, shape: tuple[int, int], x0: float, y1: float, all_touched: bool, epsg: int = EPSG) -> np.ndarray:
    """Rasterise a vector layer (reprojected to ``epsg`` if needed) on a 1 m lattice with upper-left corner (x0, y1)."""
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
    ny, nx = shape
    ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((x0, 1.0, 0.0, y1, 0.0, -1.0))
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(epsg)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetProjection(dst.ExportToWkt())
    src = ogr.Open(gpkg_or_path)
    lyr = src.GetLayerByName(layer) if isinstance(layer, str) else src.GetLayer(layer)
    srs = lyr.GetSpatialRef()
    if srs is not None and not srs.IsSame(dst):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(srs, dst)
        mem = ogr.GetDriverByName("Memory").CreateDataSource("")
        ml = mem.CreateLayer("l", dst, lyr.GetGeomType())
        for f in lyr:
            g = f.GetGeometryRef()
            if g is None:
                continue
            g = g.Clone()
            g.Transform(ct)
            nf = ogr.Feature(ml.GetLayerDefn())
            nf.SetGeometry(g)
            ml.CreateFeature(nf)
        lyr = ml
    gdal.RasterizeLayer(ds, [1], lyr, burn_values=[1], options=[f"ALL_TOUCHED={'TRUE' if all_touched else 'FALSE'}"])
    return ds.GetRasterBand(1).ReadAsArray().astype(bool)


def tiger_roads_in_box(x0: float, y1: float, nx: int, ny: int) -> dict[str, Any]:
    """TIGER Marin roads (zip shapefile) reprojected to EPSG:6339: the rasterised line cells on the lattice and the feature list."""
    from osgeo import ogr, osr
    ogr.UseExceptions()
    osr.UseExceptions()
    path = f"/vsizip/{TIGER_MARIN}/tl_2024_06041_roads.shp"
    src = ogr.Open(path)
    lyr = src.GetLayer(0)
    srs = lyr.GetSpatialRef()
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(EPSG)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(srs, dst)
    inv = osr.CoordinateTransformation(dst, srs)
    pad = 50.0
    x1, y0 = x0 + nx, y1 - ny
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for X, Y in ((x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad), (x0 - pad, y1 + pad), (x0 - pad, y0 - pad)):
        ring.AddPoint(X, Y)
    box = ogr.Geometry(ogr.wkbPolygon)
    box.AddGeometry(ring)
    box.Transform(inv)
    lyr.SetSpatialFilter(box)
    hit = np.zeros((ny, nx), dtype=bool)
    feats = []
    for f in lyr:
        g = f.GetGeometryRef().Clone()
        g.Transform(ct)
        g.Segmentize(0.5)
        lines = [g] if g.GetGeometryType() in (ogr.wkbLineString, ogr.wkbLineString25D) else [g.GetGeometryRef(i) for i in range(g.GetGeometryCount())]
        n_in = 0
        for ln in lines:
            pts = np.array(ln.GetPoints())
            if pts.size == 0:
                continue
            c = np.floor(pts[:, 0] - x0).astype(int)
            r = np.floor(y1 - pts[:, 1]).astype(int)
            ok = (c >= 0) & (c < nx) & (r >= 0) & (r < ny)
            hit[r[ok], c[ok]] = True
            n_in += int(ok.sum())
        if n_in:
            feats.append({"LINEARID": f.GetField("LINEARID"), "FULLNAME": f.GetField("FULLNAME"), "MTFCC": f.GetField("MTFCC"), "n_cells": n_in})
    lyr.SetSpatialFilter(None)
    return {"cells": hit, "features": feats, "n_features": len(feats), "mtfcc": {k: sum(1 for q in feats if q["MTFCC"] == k) for k in sorted({q["MTFCC"] for q in feats})},
            "source": path}


def md_layer_mask(layer: str, shape: tuple[int, int], x0: float, y1: float, all_touched: bool) -> np.ndarray:
    return rasterize_layer(MD_GPKG, layer, shape, x0, y1, all_touched)


def md_polygon_area_and_length(domain_mask: np.ndarray, x0: float, y1: float) -> dict[str, float]:
    """The M&D traced channel length (gpkg `channels`) inside the study polygon, and the polygon area (OGR geometry)."""
    from osgeo import ogr
    ogr.UseExceptions()
    src = ogr.Open(MD_GPKG)
    poly = None
    for f in src.GetLayerByName("study_area"):
        g = f.GetGeometryRef().Clone()
        g.FlattenTo2D()
        poly = g if poly is None else poly.Union(g)
    total = 0.0
    for f in src.GetLayerByName("channels"):
        g = f.GetGeometryRef().Clone()
        g.FlattenTo2D()
        gi = g.Intersection(poly)
        if gi is not None and not gi.IsEmpty():
            total += gi.Length()
    return {"md_traced_length_in_polygon_m": float(total), "polygon_area_m2": float(poly.GetArea())}


# ----------------------------------------------------------------------------- D8 helpers on any grid (numba where it matters)

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def _donor_csr(recv: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(recv, kind="stable")
    donors = order[recv[order] >= 0]
    starts = np.searchsorted(recv[donors], np.arange(n + 1))
    return donors.astype(np.int64), starts.astype(np.int64)


def _label_upstream_py(seeds, donors, starts, lab):
    from collections import deque
    q = deque(seeds.tolist())
    while q:
        k = q.popleft()
        for u in donors[starts[k]:starts[k + 1]].tolist():
            if lab[u] < 0:
                lab[u] = lab[k]
                q.append(u)
    return lab


if njit is not None:
    @njit(cache=True)
    def _label_upstream_nb(seeds, donors, starts, lab):  # pragma: no cover (compiled)
        n = lab.size
        queue = np.empty(n, dtype=np.int64)
        head = 0
        tail = 0
        for s in seeds:
            queue[tail] = s
            tail += 1
        while head < tail:
            k = queue[head]
            head += 1
            for j in range(starts[k], starts[k + 1]):
                u = donors[j]
                if lab[u] < 0:
                    lab[u] = lab[k]
                    queue[tail] = u
                    tail += 1
        return lab
    _label_upstream = _label_upstream_nb
else:  # pragma: no cover
    _label_upstream = _label_upstream_py


def label_upstream(FD: np.ndarray, seed_flat: np.ndarray, seed_labels: np.ndarray | None = None, stop_at_seeds: bool = True) -> np.ndarray:
    """Label every cell that drains to a seed with the seed's label (breadth-first upstream along D8); -1 elsewhere.
    With ``stop_at_seeds`` a seed keeps its own label (its upstream cells take it): seeds nested in another seed's basin start
    their own basin. Without it, every seed is labelled first and only unlabelled cells are claimed, which is the same thing."""
    ny, nx = FD.shape
    recv = receiver_flat(FD)
    lab = np.full(ny * nx, -1, dtype=np.int64)
    seed_flat = np.asarray(seed_flat, dtype=np.int64)
    lab[seed_flat] = np.arange(seed_flat.size) if seed_labels is None else np.asarray(seed_labels, dtype=np.int64)
    donors, starts = _donor_csr(recv, ny * nx)
    lab = _label_upstream(seed_flat, donors, starts, lab)
    return lab.reshape(ny, nx)


def receiver_rc(FD: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = FD.shape
    rows, cols = np.indices((ny, nx))
    nr, nc = rows + DROW[FD], cols + DCOL[FD]
    good = (FD != 0) & (nr >= 0) & (nr < ny) & (nc >= 0) & (nc < nx)
    return nr, nc, good


# ----------------------------------------------------------------------------- the multiple-flow-direction area (Freeman 1991)

_NB_DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
_NB_DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)


def _mfd_py(order, Zf, valid, recv_r, recv_c, has_recv, dx, p, area):  # reference loop
    ny, nx = Zf.shape
    for k in order.tolist():
        r, c = k // nx, k % nx
        if not valid[r, c]:
            continue
        z = Zf[r, c]
        w = np.zeros(8)
        tot = 0.0
        for i in range(8):
            rr, cc = r + _NB_DR[i], c + _NB_DC[i]
            if 0 <= rr < ny and 0 <= cc < nx and valid[rr, cc]:
                dist = dx * (math.sqrt(2.0) if (_NB_DR[i] != 0 and _NB_DC[i] != 0) else 1.0)
                s = (z - Zf[rr, cc]) / dist
                if s > 0:
                    w[i] = s ** p
                    tot += w[i]
        if tot > 0:
            for i in range(8):
                if w[i] > 0:
                    area[r + _NB_DR[i], c + _NB_DC[i]] += area[r, c] * w[i] / tot
        elif has_recv[r, c]:
            area[recv_r[r, c], recv_c[r, c]] += area[r, c]
    return area


if njit is not None:
    @njit(cache=True)
    def _mfd_nb(order, Zf, valid, recv_r, recv_c, has_recv, dx, p, area):  # pragma: no cover (compiled)
        ny, nx = Zf.shape
        dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
        dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
        w = np.zeros(8)
        for idx in range(order.size):
            k = order[idx]
            r = k // nx
            c = k % nx
            if not valid[r, c]:
                continue
            z = Zf[r, c]
            tot = 0.0
            for i in range(8):
                w[i] = 0.0
                rr = r + dr[i]
                cc = c + dc[i]
                if rr >= 0 and rr < ny and cc >= 0 and cc < nx and valid[rr, cc]:
                    dist = dx
                    if dr[i] != 0 and dc[i] != 0:
                        dist = dx * 1.4142135623730951
                    s = (z - Zf[rr, cc]) / dist
                    if s > 0:
                        w[i] = s ** p
                        tot += w[i]
            if tot > 0:
                for i in range(8):
                    if w[i] > 0:
                        area[r + dr[i], c + dc[i]] += area[r, c] * w[i] / tot
            elif has_recv[r, c]:
                area[recv_r[r, c], recv_c[r, c]] += area[r, c]
        return area
    _mfd = _mfd_nb
else:  # pragma: no cover
    _mfd = _mfd_py


def mfd_area(Zf: np.ndarray, FD: np.ndarray, dx: float, p: float = MFD_EXPONENT) -> np.ndarray:
    """Freeman (1991) multiple-flow-direction drainage area (m2) on a filled surface: each cell's area goes to its lower
    neighbours in proportion to slope^p; a cell with no lower neighbour (a filled flat) passes everything to its D8 receiver.
    Cells are visited in descending elevation (ties by flat index), so every donor is visited before its receivers."""
    Zf = np.asarray(Zf, dtype=np.float64)
    valid = np.isfinite(Zf)
    ny, nx = Zf.shape
    recv_r, recv_c, good = receiver_rc(FD)
    z = np.where(valid, Zf, -np.inf).ravel()
    order = np.lexsort((np.arange(z.size), -z))
    order = order[valid.ravel()[order]]
    area = np.where(valid, dx * dx, 0.0)
    area = _mfd(order.astype(np.int64), Zf, valid, recv_r.astype(np.int64), recv_c.astype(np.int64), good, float(dx), float(p), area)
    area[~valid] = np.nan
    return area


# ----------------------------------------------------------------------------- smoothing, filling, routing on a NaN-embedded domain

def gaussian_smooth_nan(Z: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Normalised Gaussian convolution: NaN cells carry no weight; the result is NaN where Z is."""
    from scipy.ndimage import gaussian_filter
    valid = np.isfinite(Z)
    num = gaussian_filter(np.where(valid, Z, 0.0), sigma_cells, mode="constant", cval=0.0)
    den = gaussian_filter(valid.astype(float), sigma_cells, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def fill_and_route(Z: np.ndarray, dx: float) -> dict[str, np.ndarray]:
    """Pit-fill and D8-route a surface (NaN outside the domain; data-edge cells are outlets): Zf, FD, A (m2)."""
    from lemcvt.hydro.d8 import d8_area, d8_flow_direction
    from lemcvt.hydro.fill import priority_flood_fill
    valid = np.isfinite(Z)
    Zf = priority_flood_fill(np.where(valid, Z, np.nan), dx)
    FD = d8_flow_direction(Zf, border="outlets")
    A = d8_area(FD, Zf, dx)
    A[~valid] = np.nan
    return {"Zf": Zf, "FD": FD, "A": A, "valid": valid}


def head_statistic_1m(A: np.ndarray, Zf: np.ndarray, FD: np.ndarray, dx: float, reach_m: float = REACH_M) -> dict[str, np.ndarray]:
    """Decision 2.4 on a 1 m grid: S = steepest-descent slope over ``reach_m`` downstream (valley_slope_reach), T = A S^2,
    M = upstream maximum of T along the D8, Mdon = the donors' largest M."""
    from lemcvt.analyses.slope_threshold import valley_slope_reach
    S, L, _left = valley_slope_reach(Zf, FD, dx, dx, reach_m)
    hs = head_statistic_coarse(A, S, FD, ALPHA_PRIMARY)
    return {"S": S, "L": L, "T": hs["T"], "M": hs["M"], "Mdon": hs["Mdon"]}


def value_near_picks(T: np.ndarray, A: np.ndarray, rows: np.ndarray, cols: np.ndarray, radius_m: float, dx: float = 1.0) -> dict[str, np.ndarray]:
    """For each pick: the largest T (and the A at that cell, and the largest A) over the cells within ``radius_m``."""
    r = int(math.ceil(radius_m / dx))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) * dx * dx <= radius_m ** 2 + 1e-9
    ny, nx = T.shape
    tmax = np.full(rows.size, np.nan)
    a_at = np.full(rows.size, np.nan)
    amax = np.full(rows.size, np.nan)
    cell = np.full((rows.size, 2), -1, dtype=np.int64)
    for i, (pr, pc) in enumerate(zip(rows.tolist(), cols.tolist(), strict=True)):
        r0, r1, c0, c1 = max(pr - r, 0), min(pr + r + 1, ny), max(pc - r, 0), min(pc + r + 1, nx)
        sub = T[r0:r1, c0:c1].copy()
        dsk = disk[r0 - (pr - r):r1 - (pr - r), c0 - (pc - r):c1 - (pc - r)]
        sub[~dsk] = np.nan
        if np.isfinite(sub).any():
            k = int(np.nanargmax(sub))
            rr, cc = r0 + k // sub.shape[1], c0 + k % sub.shape[1]
            tmax[i], a_at[i], cell[i] = sub.ravel()[k], A[rr, cc], (rr, cc)
        suba = A[r0:r1, c0:c1].copy()
        suba[~dsk] = np.nan
        if np.isfinite(suba).any():
            amax[i] = np.nanmax(suba)
    return {"T_max": tmax, "A_at_T_max": a_at, "A_max": amax, "cell": cell}


def f1_optimal_threshold(T_picks: np.ndarray, M: np.ndarray, Mdon: np.ndarray, valid_heads: np.ndarray, mapped_xy: np.ndarray, t: float,
                         grid: np.ndarray) -> dict[str, Any]:
    """Sensitivity only: the threshold on ``grid`` maximising the F1 of share found and share real on the calibration ground."""
    best = None
    for C in grid:
        _ch, hd = criterion_sets(M, Mdon, float(C))
        hr, hc = np.nonzero(hd & valid_heads)
        mt = match_heads(mapped_xy, np.stack([hr, hc], axis=1).astype(float), t)
        sf, sr = mt["share_found"] or 0.0, mt["share_real"] or 0.0
        f1 = 2 * sf * sr / (sf + sr) if (sf + sr) > 0 else 0.0
        if best is None or f1 > best["f1"]:
            best = {"C": float(C), "f1": f1, "share_found": sf, "share_real": sr, "n_heads": int(hr.size)}
    return best or {}


# ----------------------------------------------------------------------------- the power spectrum (Perron et al. 2008) and the valley share

def largest_inscribed_square(mask: np.ndarray) -> tuple[int, int, int]:
    """(row0, col0, side) of the largest axis-aligned square of True cells (dynamic programme)."""
    m = mask.astype(np.int32)
    ny, nx = m.shape
    dp = np.zeros((ny, nx), dtype=np.int32)
    best = (0, 0, 0)
    for r in range(ny):
        row = m[r]
        if r == 0:
            dp[r] = row
        else:
            up = dp[r - 1]
            dp[r, 0] = row[0]
            # dp[r, c] = min(dp[r-1, c], dp[r, c-1], dp[r-1, c-1]) + 1 where m == 1; sequential in c
            prev = dp[r - 1]
            cur = dp[r]
            for c in range(1, nx):
                if row[c]:
                    cur[c] = min(prev[c], cur[c - 1], prev[c - 1]) + 1
                else:
                    cur[c] = 0
            _ = up
        k = int(dp[r].max())
        if k > best[2]:
            c = int(np.argmax(dp[r]))
            best = (r - k + 1, c - k + 1, k)
    return best


if njit is not None:
    @njit(cache=True)
    def _lis_nb(m):  # pragma: no cover (compiled)
        ny, nx = m.shape
        dp = np.zeros((ny, nx), dtype=np.int32)
        br, bc, bk = 0, 0, 0
        for r in range(ny):
            for c in range(nx):
                if m[r, c]:
                    if r == 0 or c == 0:
                        dp[r, c] = 1
                    else:
                        v = dp[r - 1, c]
                        if dp[r, c - 1] < v:
                            v = dp[r, c - 1]
                        if dp[r - 1, c - 1] < v:
                            v = dp[r - 1, c - 1]
                        dp[r, c] = v + 1
                    if dp[r, c] > bk:
                        bk = dp[r, c]
                        br = r - bk + 1
                        bc = c - bk + 1
        return br, bc, bk

    def largest_inscribed_square(mask: np.ndarray) -> tuple[int, int, int]:  # noqa: F811
        br, bc, bk = _lis_nb(np.ascontiguousarray(mask.astype(np.uint8)))
        return int(br), int(bc), int(bk)


def detrend_plane(Z: np.ndarray) -> np.ndarray:
    ny, nx = Z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    G = np.column_stack([xx.ravel(), yy.ravel(), np.ones(Z.size)])
    coef, *_ = np.linalg.lstsq(G, Z.ravel(), rcond=None)
    return Z - (G @ coef).reshape(ny, nx)


def power_spectrum_binned(Z: np.ndarray, dx: float, bins_m: np.ndarray) -> dict[str, Any]:
    """Perron et al. (2008): plane detrend, 2-D Hann window, periodogram, binned by radial wavelength (mean power per bin).
    Power is normalised by the window's mean square so that the ratio of two surfaces processed alike is window-free."""
    Zd = detrend_plane(np.asarray(Z, dtype=float))
    ny, nx = Zd.shape
    wy, wx = np.hanning(ny), np.hanning(nx)
    W = wy[:, None] * wx[None, :]
    F = np.fft.fftshift(np.fft.fft2(Zd * W))
    P = (np.abs(F) ** 2) / (np.sum(W ** 2))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fr = np.hypot(fy[:, None], fx[None, :])
    with np.errstate(divide="ignore"):
        lam = np.where(fr > 0, 1.0 / fr, np.inf)
    idx = np.digitize(lam.ravel(), bins_m) - 1
    ok = (idx >= 0) & (idx < bins_m.size - 1) & np.isfinite(lam.ravel())
    sums = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=bins_m.size - 1)
    counts = np.bincount(idx[ok], minlength=bins_m.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
    centres = np.sqrt(bins_m[:-1] * bins_m[1:])
    return {"wavelength_m": centres, "bin_edges_m": bins_m, "power": mean, "n": counts, "side_m": ny * dx, "variance_detrended": float(np.var(Zd))}


def wavelength_bins() -> np.ndarray:
    return np.logspace(math.log10(WAVELENGTH_RANGE_M[0]), math.log10(WAVELENGTH_RANGE_M[1]), N_WAVELENGTH_BINS + 1)


def rolloff_wavelength(lam: np.ndarray, power: np.ndarray, fit_min_m: float = ROLLOFF_FIT_MIN_M) -> dict[str, Any]:
    """Perron et al. (2008): a background power law fitted to every bin (log-log least squares) and the normalised spectrum
    power / background. Reported: the background exponent; the characteristic wavelength (the peak of the normalised spectrum, the
    scale at which the landscape holds most power above its background: the valley spacing in Perron et al.); and the roll-off, the
    shortest wavelength above the peak at which the normalised spectrum falls back below 1 (None if it does not within the range).
    ``fit_min_m`` is kept for the record of the long-wavelength exponent (bins at or above it)."""
    ok = np.isfinite(power) & (power > 0)
    if ok.sum() < 3:
        return {"rolloff_m": None, "characteristic_wavelength_m": None, "fit_n": int(ok.sum())}
    b, a = np.polyfit(np.log10(lam[ok]), np.log10(power[ok]), 1)
    norm = np.where(ok, power / (10 ** (a + b * np.log10(lam))), np.nan)
    order = np.argsort(lam)
    cand = [j for j in order if lam[j] >= READING_BAND_M[0] and np.isfinite(norm[j])]     # the peak is sought above the lidar's noise scales (>= 20 m)
    j_peak = int(cand[int(np.nanargmax(norm[cand]))])
    roll = None
    for j in order:
        if lam[j] > lam[j_peak] and np.isfinite(norm[j]) and norm[j] < 1.0:
            roll = float(lam[j])
            break
    okl = ok & (lam >= fit_min_m)
    b_long = float(np.polyfit(np.log10(lam[okl]), np.log10(power[okl]), 1)[0]) if okl.sum() >= 2 else None
    return {"rolloff_m": roll, "characteristic_wavelength_m": float(lam[j_peak]), "normalised_peak": float(norm[j_peak]), "background_exponent": float(b),
            "normalised_spectrum": norm.tolist(), "fit_n": int(ok.sum()), "long_wavelength_exponent_at_or_above_fit_min": b_long, "fit_min_m": fit_min_m}


def contour_curvature(Z: np.ndarray, dx: float, sigma_m: float) -> np.ndarray:
    """Perron et al. (2012) eq. (1) on the Gaussian-smoothed surface (NaN filled by nearest neighbour first); positive = convergent."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    Z = np.asarray(Z, dtype=float)
    nan = ~np.isfinite(Z)
    Zf = Z
    if nan.any():
        idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
        Zf = Z[tuple(idx)]
    Zs = gaussian_filter(Zf, sigma_m / dx, mode="nearest")
    zy, zx = np.gradient(Zs, dx)
    zxy, zxx = np.gradient(zx, dx)
    zyy, _ = np.gradient(zy, dx)
    g2 = zx ** 2 + zy ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        kc = (zy ** 2 * zxx - 2 * zx * zy * zxy + zx ** 2 * zyy) / g2 ** 1.5
    kc[nan] = np.nan
    return kc


# ----------------------------------------------------------------------------- the domain record

def load_domain() -> dict[str, Any]:
    """The domain arrays and record written by a23_domain.py."""
    D = dict(np.load(out_path("domain", "domain.npz")))
    rec = load_json(out_path("domain", "domain.json"))
    shape = tuple(D["Z"].shape)
    out: dict[str, Any] = {"rec": rec, "shape": shape, "Z": D["Z"].astype(float), "A": D["A"].astype(float), "FD": D["FD"].astype(np.uint8),
                           "domain": np.unpackbits(D["domain_bits"])[: shape[0] * shape[1]].reshape(shape).astype(bool),
                           "poly": np.unpackbits(D["poly_bits"])[: shape[0] * shape[1]].reshape(shape).astype(bool),
                           "outlets": np.unpackbits(D["outlets_bits"])[: shape[0] * shape[1]].reshape(shape).astype(bool),
                           "road_dist": D["road_dist"].astype(float), "x0": float(rec["box"]["x0"]), "y1": float(rec["box"]["y1"]),
                           "picks": {k: D[f"pick_{k}"] for k in ("id", "row", "col", "in_domain", "in_poly", "tile")}}
    for key in ("net_C", "net_A", "net_mapped"):
        out[key] = np.unpackbits(D[f"{key}_bits"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    out["tile_boxes"] = rec["tile_boxes"]
    return out


def tile_footprint(dom: dict[str, Any], tiles: list[str]) -> np.ndarray:
    """Cells of the domain lattice inside any of the tiles' footprints."""
    m = np.zeros(dom["shape"], dtype=bool)
    for t in tiles:
        b = dom["tile_boxes"][t]
        r0, r1, c0, c1 = (int(v) for v in (b["r0"], b["r1"], b["c0"], b["c1"]))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, m.shape[0]), min(c1, m.shape[1])
        if r1 > r0 and c1 > c0:
            m[r0:r1, c0:c1] = True
    return m


def domain_profiler(dom: dict[str, Any], m_values=(M,), channel_stat=None):
    from lemcvt.optimize.v2 import PartitionedProfiler
    return PartitionedProfiler(dom["Z"], dom["A"], dom["FD"], 1.0, U, m_values=list(m_values), channel_stat=channel_stat, outlets=dom["outlets"], noflux_outside=True)


def domain_ad_model(dom: dict[str, Any], m: float = M):
    from lemcvt.models.ad import advection_diffusion_model
    return advection_diffusion_model(dom["Z"], dom["A"], dom["FD"], 1.0, m, outlets=dom["outlets"], noflux_outside=True)


def bounds_box(rec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    """The addendum 18 box in (U/K, U/D) at the new U (the TTV2 config bounds x 65/77, as addenda 21-22)."""
    Kb = tuple(rec["bounds_rescaled"]["K"])
    Db = tuple(rec["bounds_rescaled"]["D"])
    lo = np.array([U / Kb[1], U / Db[1]])
    hi = np.array([U / Kb[0], U / Db[0]])
    return lo, hi, Kb, Db


def reading_ha(share_found: float | None, share_real: float | None) -> bool:
    return bool(share_found is not None and share_real is not None and share_found >= HA_SHARE_FOUND_MIN and share_real >= HA_SHARE_REAL_MIN)


_ = (json, A_F, BIG, C_F, COLOUR, D_PHYS, D_RELIEF_RANGE, FITTED, K_PHYS_RANGE, MARKER, PLAIN, RESCALE, T_PRIMARY_M, TOLERANCES_M, WHAT, a21_path, block_mean,
     channel_heads, classify_extra, coarse_grid, coarse_products, criterion_sets, donor_slope, dump, fit_terms, heads_from_products, hillshade, json_ready, load_json,
     mask_from_entries, match_heads, metre_tile, mse_at, partitioned_surface, reduce_mask, slurm_record, terms_of, a22_out_path)
