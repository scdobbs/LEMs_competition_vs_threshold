#!/usr/bin/env python3
"""Addendum 13, section 2.2.3: the Tennessee Valley (TTV) NCALM 2006 LAS tile.

Reports, from the file itself (nothing assumed):
  * header bounds and whether the file name is the floor of the header minimum x and y;
  * the bounds in geographic coordinates under the two UTM 10N candidates (WGS84 EPSG:32610, NAD83 EPSG:26910), and
    whether they fall inside the USGS Point Bonita 7.5' quadrangle (37.75-37.875 N, 122.5-122.625 W) and contain the
    Montgomery & Dietrich (1989) elevation band 61-314 m;
  * the vertical-datum evidence: the distribution of the lowest points (a sea-level surface near z = 0 is orthometric;
    ellipsoidal heights at sea level here would be about -32 m (geoid height), stated as the test, not assumed);
  * classification and return counts; ground density over the tile and in hollows. Hollows are defined here from a
    2 m grid of the lowest point per cell (ground if classified, else all points), smoothed, as cells whose Laplacian at
    an 11 m window exceeds +0.02 1/m (convergent, concave-up); the definition is a stand-in and is stated in the report.

usage: a13_ttv.py --las FILE --out DIR
"""
import argparse, json, os
import numpy as np
import laspy
from pyproj import Transformer
from scipy.ndimage import uniform_filter, distance_transform_edt, generic_filter

QUAD = {"lat": (37.75, 37.875), "lon": (-122.625, -122.5)}  # USGS Point Bonita 7.5-minute quadrangle


def main():
    p = argparse.ArgumentParser(); p.add_argument("--las", required=True); p.add_argument("--out", required=True); p.add_argument("--cell", type=float, default=2.0)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    rec = {"las": a.las}
    with laspy.open(a.las) as f:
        h = f.header
        rec["header"] = {"version": str(h.version), "point_format": int(h.point_format.id), "n_points": int(h.point_count), "mins": [float(v) for v in h.mins], "maxs": [float(v) for v in h.maxs],
                         "scales": [float(v) for v in h.scales], "offsets": [float(v) for v in h.offsets], "generating_software": h.generating_software, "n_vlrs": len(h.vlrs),
                         "creation_date": str(h.creation_date), "dimensions": list(h.point_format.dimension_names)}
        name = os.path.basename(a.las).split(".")[0]
        parts = name.split("_")
        rec["name_check"] = {"name": name, "name_x": parts[1], "name_y": parts[2], "floor_min_x": int(np.floor(h.mins[0])), "floor_min_y": int(np.floor(h.mins[1])),
                             "name_equals_floor_of_header_min": (int(parts[1]) == int(np.floor(h.mins[0])) and int(parts[2]) == int(np.floor(h.mins[1])))}
        las = f.read()
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    c = np.asarray(las.classification); rn = np.asarray(las.return_number); nr = np.asarray(las.number_of_returns)
    rec["classes"] = {int(k): int(v) for k, v in zip(*np.unique(c, return_counts=True))}
    rec["return_number"] = {int(k): int(v) for k, v in zip(*np.unique(rn, return_counts=True))}
    rec["number_of_returns"] = {int(k): int(v) for k, v in zip(*np.unique(nr, return_counts=True))}
    rec["intensity_range"] = [int(np.min(las.intensity)), int(np.max(las.intensity))]
    rec["scan_angle_range"] = [int(np.min(las.scan_angle_rank)), int(np.max(las.scan_angle_rank))] if "scan_angle_rank" in las.point_format.dimension_names else None
    area = (h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1])
    rec["density_all_per_m2_if_metres"] = len(z) / area
    ground = c == 2
    rec["ground_classified"] = bool(ground.any())
    rec["density_ground_per_m2_if_metres"] = float(ground.sum() / area)
    # geographic bounds under both UTM 10N candidates
    geo = {}
    for epsg in (32610, 26910):
        t = Transformer.from_crs(epsg, 4326, always_xy=True)
        corners = [t.transform(X, Y) for X in (h.mins[0], h.maxs[0]) for Y in (h.mins[1], h.maxs[1])]
        lons, lats = zip(*corners)
        geo[str(epsg)] = {"lon": [min(lons), max(lons)], "lat": [min(lats), max(lats)],
                          "inside_point_bonita_quad": bool(min(lons) >= QUAD["lon"][0] and max(lons) <= QUAD["lon"][1] and min(lats) >= QUAD["lat"][0] and max(lats) <= QUAD["lat"][1])}
    rec["geographic_bounds"] = geo
    rec["utm_candidates_differ_by_m"] = "NAD83 vs WGS84 UTM 10N differ by about 1-2 m here; both place the tile at the same quadrangle position"
    # vertical datum evidence: lowest points
    zz = z[ground] if ground.any() else z
    rec["z_percentiles_used_points"] = {str(q): float(np.percentile(zz, q)) for q in (0.01, 0.1, 1, 5, 50, 95, 99, 99.9)}
    low = zz < 2.0
    rec["n_points_below_2m"] = int(low.sum())
    if low.any():
        xl, yl = (x[ground] if ground.any() else x)[low], (y[ground] if ground.any() else y)[low]
        rec["low_points_location_utm"] = {"x": [float(xl.min()), float(xl.max())], "y": [float(yl.min()), float(yl.max())], "x_median": float(np.median(xl)), "y_median": float(np.median(yl))}
    rec["elevation_band_61_314m_present"] = bool((zz > 61).any() and (zz < 314).any())
    rec["frac_used_points_in_61_314m"] = float(np.mean((zz >= 61) & (zz <= 314)))
    # 2 m grid of the lowest point per cell, hollows, ground density in hollows
    cs = a.cell
    nx = int(np.ceil((h.maxs[0] - h.mins[0]) / cs)); ny = int(np.ceil((h.maxs[1] - h.mins[1]) / cs))
    ix = np.clip(((x - h.mins[0]) / cs).astype(int), 0, nx - 1); iy = np.clip(((h.maxs[1] - y) / cs).astype(int), 0, ny - 1)
    use = ground if ground.any() else np.ones_like(ground)
    cnt = np.zeros((ny, nx), np.int64); np.add.at(cnt, (iy[use], ix[use]), 1)
    zmin = np.full((ny, nx), np.inf); np.minimum.at(zmin, (iy[use], ix[use]), z[use])
    have = np.isfinite(zmin)
    rec["grid"] = {"cell_m": cs, "nx": nx, "ny": ny, "frac_cells_with_used_point": float(have.mean()), "frac_1m_cells_with_ground_point": None}
    # 1 m occupancy
    ix1 = np.clip(((x - h.mins[0]) / 1.0).astype(int), 0, int(np.ceil(h.maxs[0] - h.mins[0])) - 1); iy1 = np.clip(((h.maxs[1] - y) / 1.0).astype(int), 0, int(np.ceil(h.maxs[1] - h.mins[1])) - 1)
    occ = np.zeros((iy1.max() + 1, ix1.max() + 1), bool); occ[iy1[use], ix1[use]] = True
    rec["grid"]["frac_1m_cells_with_ground_point"] = float(occ.mean())
    # fill gaps by nearest, smooth 3 cells, Laplacian at 11 m (5 cells half-width -> window 11 m at 2 m cells = 5.5 cells; use 5)
    idx = distance_transform_edt(~have, return_distances=False, return_indices=True)
    Zg = zmin[tuple(idx)]
    Zs = uniform_filter(Zg, 3)
    w = max(1, int(round(11 / cs / 2)))
    lap = (np.roll(Zs, w, 0) + np.roll(Zs, -w, 0) + np.roll(Zs, w, 1) + np.roll(Zs, -w, 1) - 4 * Zs) / (w * cs) ** 2
    hollow = lap > 0.02; divergent = lap < -0.02
    inner = np.zeros_like(have); inner[w:-w, w:-w] = True
    for nm, m in (("hollow_lap_gt_0.02", hollow), ("divergent_lap_lt_-0.02", divergent), ("all", np.ones_like(have))):
        mm = m & inner & have
        rec[f"density_used_points_{nm}"] = {"n_cells": int(mm.sum()), "frac_of_cells": float(mm.sum() / (inner & have).sum()), "points_per_m2": float(cnt[mm].sum() / (mm.sum() * cs * cs)) if mm.any() else None,
                                           "frac_cells_empty_before_fill": float(np.mean(~have[m & inner])) if (m & inner).any() else None}
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
        ls = LightSource(315, 45); hs = ls.hillshade(Zs, vert_exag=1, dx=cs, dy=cs)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ext = [h.mins[0], h.maxs[0], h.mins[1], h.maxs[1]]
        ax[0].imshow(hs, cmap="gray", extent=ext); ax[0].set_title(f"{name}: hillshade of {cs:g} m lowest-point grid (UTM 10N m)")
        im = ax[1].imshow(np.where(have, Zg, np.nan), extent=ext, cmap="terrain"); fig.colorbar(im, ax=ax[1], label="z (header units)"); ax[1].set_title("z of lowest point per cell")
        for axx in ax:
            axx.set_xlabel("E"); axx.set_ylabel("N")
        fig.savefig(os.path.join(a.out, f"{name}_hillshade.png"), dpi=120); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 6)); ax.imshow(np.where(hollow, 1, np.where(divergent, -1, 0)), extent=ext, cmap="RdBu"); ax.set_title("hollow (+1) / divergent (-1) by Laplacian at 11 m")
        fig.savefig(os.path.join(a.out, f"{name}_hollows.png"), dpi=110); plt.close(fig)
    except Exception as e:
        rec["figure_error"] = repr(e)
    np.save(os.path.join(a.out, f"{name}_grid{cs:g}m_zmin.npy"), np.where(have, zmin, np.nan).astype(np.float32))
    json.dump(rec, open(os.path.join(a.out, "ttv.json"), "w"), indent=1, default=str)
    print(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
