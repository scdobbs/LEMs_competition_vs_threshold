#!/usr/bin/env python3
"""Addendum 16, section 3.2: the GM extension gate on the 10 m NED.

For each survey that reaches the GM landscape (a16_common.GM_GATE_SURVEYS) and for the NCALM 2003 footprint itself (the
baseline pool): the NED crop (a16_common.GM_GATE_CROP_UTM, from gm_footprints/GM_NED_10m_utm10n.tif) is masked to the
survey's footprint and routed with the stage A conventions (a14_pools.route_full_grid: priority flood, D8 with data-edge
outlets, area, edge-touched flag), candidate 1 km windows are laid on the four stage A lattice offsets (every cell valid in
the masked grid and inside the GM landscape, a16_common.GM_LANDSCAPE_POLY_UTM within GM_LANDSCAPE_RADIUS_M of the tiles'
centroid), and the addendum 15 rules are applied: threshold set B screens (trunk <= 5 km2, single inflow <= 2 km2, chi-z
R2 >= 0.90 at theta_ref, dBIC <= 20, NLCD 11/22-24/31 <= 10 %, upstream area fully in the survey's valid data), k_s and |C_ht|
within +-25 % (and +-50 %) of the legacy GM reference. The gap rule needs the 1 m grid and cannot be applied here (stated).
Gate (decision 5, written now): the survey passes if its eligible windows that (a) do not overlap GM1-GM7 and (b) are not
eligible in the NCALM baseline pool on the same lattice ("added") number at least 10 at the tolerance the widening rule
selects on the combined count (baseline + survey at +-25 %; +-50 % if fewer than 25). Counts are also given for the disc
and the polygon alone, and for the windows that lie beyond the NCALM footprint.
Outputs (run dir gm_gate/): <pool>_windows.csv, gm_gate.json, GM_landscape_10m.tif, GM16_nlcd2021.tif, gm_gate_map.png.
usage: a16_gm_gate.py --out DIR --footprints DIR [--no-network]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import LEGACY_REF  # noqa: E402
from a14_pools import A_C, basin_stats_window, crop, fetch_nlcd, route_full_grid, window_lattice, window_metrics  # noqa: E402
from a16_common import (  # noqa: E402
    GM_GATE_CROP_UTM,
    GM_GATE_MIN_ELIGIBLE,
    GM_GATE_SURVEYS,
    GM_LANDSCAPE_POLY_UTM,
    GM_LANDSCAPE_RADIUS_M,
    MIN_ELIGIBLE,
    THRESHOLDS,
    TOL_REL,
    TOL_REL_WIDE,
    dump,
    slurm_record,
    write_tif,
)
from a16_gm_footprints import GM_TILES  # noqa: E402
from make_a14_tables import screen_masks  # noqa: E402

TILE = 1000


def overlaps(f1, f2):
    return f1[0] < f2[0] + TILE and f2[0] < f1[0] + TILE and f1[1] < f2[1] + TILE and f2[1] < f1[1] + TILE


def read_crop(path, box):
    with rasterio.open(path) as ds:
        b = ds.bounds
        assert b.left <= box[0] and b.bottom <= box[1] and box[2] <= b.right and box[3] <= b.top, (path, box, tuple(b))   # the crop must lie inside the raster
        w = from_bounds(box[0], box[1], box[2], box[3], ds.transform)
        a = ds.read(1, window=w)
        tr = ds.window_transform(w)
        return a, tr, ds.crs, ds.nodata


def pool(name, z, mask, land, tr, dx, nlcd, ref, T, W, offsets, ncalm_full, out=None):
    """Candidate windows and their screen statistics on z masked to `mask` (one survey or the NCALM footprint).
    With `out`, each offset's rows are checkpointed to <out>/<name>_offset<k>_partial.csv and reused on a rerun (resume)."""
    zs = np.where(mask, z, np.nan)
    R = route_full_grid(zs, dx)
    area, fd, tgt, et = R["area"], R["fd"], R["target"], R["edge_touched"]
    disturbed = np.isin(nlcd, list(T["disturbance_nlcd_classes"])) if nlcd is not None else None   # set B classes (11, 22-24, 31)
    rows = []
    t0 = time.time()
    for oi, (orr, occ) in enumerate(offsets):
        part = os.path.join(out, f"{name}_offset{oi}_partial.csv") if out else None
        if part and os.path.exists(part):
            prev = pd.read_csv(part)
            rows.extend(prev.to_dict("records"))
            print(name, "offset", oi, "resumed from", part, len(prev), "windows", flush=True)
            continue
        n_before = len(rows)
        _wid, wins = window_lattice(z.shape, W, orr, occ)
        for k, (r0, c0) in enumerate(wins):
            zc = crop(zs, r0, c0, W)
            if not (np.isfinite(zc).all() and crop(land, r0, c0, W).all()):
                continue
            ac, fdc = crop(area, r0, c0, W), crop(fd, r0, c0, W)
            x0 = tr.c + c0 * tr.a
            y0 = tr.f + (r0 + W) * tr.e
            row = {"pool": name, "offset": oi, "window": int(k), "row0": int(r0), "col0": int(c0), "x0": x0, "y0": y0, "fp_x0": int(np.floor(x0)), "fp_y0": int(np.floor(y0)),
                   "in_ncalm_full": bool(crop(ncalm_full, r0, c0, W).all()), "A_max_inside_m2": float(np.nanmax(ac))}
            cells = np.zeros(z.shape, bool)
            cells[r0:r0 + W, c0:c0 + W] = True
            cf = cells.ravel()
            donors_out = (~cf) & (tgt >= 0) & cf[np.maximum(tgt, 0)]
            row["A_inflow_edge_m2"] = float(np.nansum(area.ravel()[donors_out]))
            row["A_inflow_max_single_m2"] = float(np.nanmax(area.ravel()[donors_out])) if donors_out.any() else 0.0
            row["upstream_fully_valid"] = bool(~crop(et, r0, c0, W).any())
            row["relief_m"] = float(np.nanmax(zc) - np.nanmin(zc))
            bsw = basin_stats_window(zc, ac, fdc, dx, ref["theta"], A_C["GM"])
            for key in ("n_basins", "chi_r2_area_weighted", "sa_dbic_area_weighted"):
                row[key] = bsw.get(key, np.nan)
            row["nlcd_disturbed_frac"] = float(crop(disturbed, r0, c0, W).mean()) if disturbed is not None else np.nan
            row.update(window_metrics("GM", zc, ac, fdc, dx, ref["theta"], A_C["GM"]))
            rows.append(row)
        print(name, "offset", oi, "windows", len([r for r in rows if r["offset"] == oi]), round(time.time() - t0), "s", flush=True)
        if part:
            pd.DataFrame(rows[n_before:]).to_csv(part, index=False)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, R
    masks = screen_masks(df, T, "GM", None)
    df["screens_pass"] = np.logical_and.reduce([m.values for m in masks.values()])
    for nme, m in masks.items():
        df["screen: " + nme] = m.values
    ks = pd.to_numeric(df["ks_ref_median"], errors="coerce")
    ch = pd.to_numeric(df["C_ht"], errors="coerce")
    df["ks_rel_dev"] = ks / ref["ks"] - 1
    df["C_ht_rel_dev"] = ch / ref["C_ht"] - 1
    df["tol25"] = (df["ks_rel_dev"].abs() <= TOL_REL) & (df["C_ht_rel_dev"].abs() <= TOL_REL)
    df["tol50"] = (df["ks_rel_dev"].abs() <= TOL_REL_WIDE) & (df["C_ht_rel_dev"].abs() <= TOL_REL_WIDE)
    df["eligible25"] = df["screens_pass"] & df["tol25"]
    df["eligible50"] = df["screens_pass"] & df["tol50"]
    df["overlaps_GM1_7"] = [any(overlaps((int(r.fp_x0), int(r.fp_y0)), fp) for fp in GM_TILES.values()) for r in df.itertuples()]
    return df, R


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--footprints", required=True)
    p.add_argument("--no-network", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    box = GM_GATE_CROP_UTM
    z, tr, crs, nod = read_crop(os.path.join(a.footprints, "GM_NED_10m_utm10n.tif"), box)
    z = z.astype(np.float64)
    z[z == nod] = np.nan
    dx = float(tr.a)
    ny, nx = z.shape
    frac, _, _, _ = read_crop(os.path.join(a.footprints, "GM_ncalm_valid_frac_10m.tif"), box)
    ncalm_full = frac >= 0.999
    ncalm_any = frac > 0
    # landscape mask: polygon and disc
    from osgeo import gdal, ogr, osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32610)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in GM_LANDSCAPE_POLY_UTM:
        ring.AddPoint(float(x), float(y))
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    mem = ogr.GetDriverByName("Memory").CreateDataSource("m")
    lyr = mem.CreateLayer("p", srs, ogr.wkbPolygon)
    f = ogr.Feature(lyr.GetLayerDefn())
    f.SetGeometry(poly)
    lyr.CreateFeature(f)
    tgt_ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    tgt_ds.SetGeoTransform((tr.c, tr.a, 0.0, tr.f, 0.0, tr.e))
    tgt_ds.SetProjection(srs.ExportToWkt())
    gdal.RasterizeLayer(tgt_ds, [1], lyr, burn_values=[1], options=["ALL_TOUCHED=FALSE"])
    in_poly = tgt_ds.ReadAsArray().astype(bool)
    cx = np.mean([v[0] + 500 for v in GM_TILES.values()])
    cy = np.mean([v[1] + 500 for v in GM_TILES.values()])
    xs = tr.c + (np.arange(nx) + 0.5) * tr.a
    ys = tr.f + (np.arange(ny) + 0.5) * tr.e
    in_disc = np.hypot(xs[None, :] - cx, ys[:, None] - cy) <= GM_LANDSCAPE_RADIUS_M
    landscapes = {"polygon_and_disc": in_poly & in_disc, "disc": in_disc, "polygon": in_poly}
    write_tif(os.path.join(a.out, "GM_landscape_10m.tif"), (in_poly & in_disc).astype(np.uint8), tr, crs)
    rec = {"crop": box, "shape": [ny, nx], "dx": dx, "n_valid": int(np.isfinite(z).sum()), "tiles_centroid": [float(cx), float(cy)], "radius_m": GM_LANDSCAPE_RADIUS_M,
           "polygon_utm": GM_LANDSCAPE_POLY_UTM, "landscape_km2": {k: float(v.sum() * dx * dx / 1e6) for k, v in landscapes.items()},
           "landscape_beyond_ncalm_km2": {k: float((v & ~ncalm_any).sum() * dx * dx / 1e6) for k, v in landscapes.items()},
           "reference": dict(LEGACY_REF["GM"]), "thresholds": THRESHOLDS, "gate_min_eligible": GM_GATE_MIN_ELIGIBLE, "gap_rule": "not applicable on the 10 m NED (needs the 1 m grid); stated"}
    T = json.load(open(THRESHOLDS))
    nlcd, nlcd_info = (None, {"skipped": True}) if a.no_network else fetch_nlcd(a.out, tr, z.shape, 32610, "GM16")
    rec["nlcd"] = nlcd_info
    W = int(round(1000.0 / dx))
    H = W // 2
    offsets = [[0, 0], [0, H], [H, 0], [H, H]]
    ref = LEGACY_REF["GM"]
    land = landscapes["polygon_and_disc"]
    # pools: the NCALM baseline (windows fully inside the NCALM footprint) and each survey
    masks = {"NCALM2003": ncalm_full}
    for s in GM_GATE_SURVEYS:
        m, _, _, _ = read_crop(os.path.join(a.footprints, f"GM_survey_{s}_10m.tif"), box)
        masks[s] = m.astype(bool)
    dfs = {}
    for name, m in masks.items():
        df, R = pool(name, z, m, landscapes["disc"] | landscapes["polygon"], tr, dx, nlcd, ref, T, W, offsets, ncalm_full, out=a.out)
        df.to_csv(os.path.join(a.out, f"{name}_windows.csv"), index=False)
        dfs[name] = df
        print(name, "candidates", len(df), "screens", int(df["screens_pass"].sum()) if len(df) else 0, "eligible25", int(df["eligible25"].sum()) if len(df) else 0,
              "eligible50", int(df["eligible50"].sum()) if len(df) else 0, flush=True)
    base = dfs["NCALM2003"]
    gate = {}
    for lname, lmask in landscapes.items():
        def in_land(df):
            return np.array([crop(lmask, int(r.row0), int(r.col0), W).all() for r in df.itertuples()], bool) if len(df) else np.zeros(0, bool)
        bl = in_land(base)
        for tol in ("25", "50"):
            base_keys = set((int(r.offset), int(r.window)) for r in base[bl & base[f"eligible{tol}"].values].itertuples())
            for s in GM_GATE_SURVEYS:
                df = dfs[s]
                if len(df) == 0:
                    gate[(lname, tol, s)] = {"n_candidates": 0}
                    continue
                il = in_land(df)
                el = df[f"eligible{tol}"].values & il
                keys = [(int(r.offset), int(r.window)) for r in df.itertuples()]
                added = np.array([k not in base_keys for k in keys], bool)
                nov = ~df["overlaps_GM1_7"].values
                beyond = ~df["in_ncalm_full"].values
                gate[(lname, tol, s)] = {"n_candidates": int(il.sum()), "n_screens_pass": int((df["screens_pass"].values & il).sum()), "n_eligible": int(el.sum()),
                                         "n_eligible_not_overlapping_GM1_7": int((el & nov).sum()), "n_eligible_added_not_overlapping": int((el & nov & added).sum()),
                                         "n_eligible_beyond_ncalm_not_overlapping": int((el & nov & beyond).sum()), "n_baseline_eligible": len(base_keys),
                                         "n_combined_eligible": len(base_keys | set(k for k, e in zip(keys, el, strict=True) if e))}
    # the widening rule on the combined count (landscape = polygon_and_disc)
    rec["gate"] = {}
    for s in GM_GATE_SURVEYS:
        g25 = gate.get(("polygon_and_disc", "25", s), {})
        g50 = gate.get(("polygon_and_disc", "50", s), {})
        combined25 = g25.get("n_combined_eligible", 0)
        tol_used = "25" if combined25 >= MIN_ELIGIBLE["GM"] else "50"
        g = g25 if tol_used == "25" else g50
        rec["gate"][s] = {"tolerance_used": tol_used, "combined_eligible_at_25": combined25, "count_for_gate": g.get("n_eligible_added_not_overlapping", 0),
                          "pass": g.get("n_eligible_added_not_overlapping", 0) >= GM_GATE_MIN_ELIGIBLE,
                          "by_landscape_and_tolerance": {f"{ln}|{tl}": gate[(ln, tl, s)] for (ln, tl, ss) in gate if ss == s}}
        print("GATE", s, json.dumps({k: v for k, v in rec["gate"][s].items() if k != "by_landscape_and_tolerance"}), flush=True)
    rec["baseline"] = {"n_candidates": int(len(base)), "n_eligible25": int(base["eligible25"].sum()) if len(base) else 0, "n_eligible50": int(base["eligible50"].sum()) if len(base) else 0}
    rec["seconds"] = time.time() - t0
    dump(rec, os.path.join(a.out, "gm_gate.json"))
    # map
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)), dx, dx)
        sl = np.arctan(np.hypot(gx, gy))
        asp = np.arctan2(-gx, gy)
        hs = np.sin(np.radians(45)) * np.cos(sl) + np.cos(np.radians(45)) * np.sin(sl) * np.cos(np.radians(315) - asp)
        fig, ax = plt.subplots(figsize=(14, 14 * ny / nx + 1), dpi=110)
        ext = [tr.c, tr.c + nx * tr.a, tr.f + ny * tr.e, tr.f]
        ax.imshow(hs, cmap="gray", extent=ext, vmin=-0.3, vmax=1.3)
        X, Y = np.meshgrid(xs, ys)
        ax.contour(X, Y, ncalm_any.astype(float), levels=[0.5], colors="lime", linewidths=1.5)
        ax.contour(X, Y, land.astype(float), levels=[0.5], colors="yellow", linewidths=1.5)
        for s, col in zip(GM_GATE_SURVEYS, ("red", "orange"), strict=False):
            ax.contour(X, Y, masks[s].astype(float), levels=[0.5], colors=col, linewidths=1.0)
            df = dfs[s]
            for r in df[df["screens_pass"]].itertuples():
                ax.add_patch(Rectangle((r.fp_x0, r.fp_y0), TILE, TILE, fill=False, ec="0.7", lw=0.3))
            for r in df[df["eligible50"] & ~df["overlaps_GM1_7"]].itertuples():
                ax.add_patch(Rectangle((r.fp_x0, r.fp_y0), TILE, TILE, fill=False, ec="deepskyblue" if r.eligible25 else "dodgerblue", lw=0.9 if r.eligible25 else 0.5, ls="-" if r.eligible25 else "--"))
        for tx, ty in GM_TILES.values():
            ax.add_patch(Rectangle((tx, ty), TILE, TILE, fill=False, ec="lime", lw=1.2))
        ax.set_title("GM gate: landscape (yellow), NCALM (lime), FEMA 2018 (red), ARRA 2010 (orange); windows passing the screens grey, eligible at 25 % blue solid / 50 % dashed")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        fig.savefig(os.path.join(a.out, "gm_gate_map.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("map failed", repr(e), flush=True)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
