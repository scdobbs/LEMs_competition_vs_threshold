#!/usr/bin/env python3
"""Addendum 17 Part B, go-ahead section 3: the two screens, applied before the primary set is fixed.

--part bedrock (3.1): the Digital Geologic Map of Monterey County (Rosenberg; PURL cm427jp1187, copy of record GEOLOGY_DIR) is
intersected with every GM tile footprint (EPSG:6339 squares reprojected to the map's NAD83 / California zone 4 ftUS). The reference
unit is the non-Quaternary general unit (GEN_LAB) with the largest combined area under GM1_fema-GM7_fema; a tile passes if the
fraction of its footprint mapped as the reference unit plus the counted Quaternary units (a17b_common.QUATERNARY_COUNTED_GEN_LAB;
landslide deposits never counted) is >= 90 %. Table for all 30 GM tiles, a map panel, and a stop flag if any of GM1_fema-GM7_fema fails.
--part gapfill (3.2): for every expected primary tile, the fraction of nearest-neighbour-filled or gap-filled cells (nnfill_mask == 1 or
gapfill_mask == 1) on (a) the channel pixels the k_s chi-z fit uses (network A >= 1e3 m2, the kept basins of the pre-registered filters)
and (b) the option (iii) hilltop pixels at L_f = 11 m (rollover A_c from the tile's latest `measure` run); > 20 % on either leaves the
primary set. --part final: combines both and writes primary_set.json (the set is fixed here, before any m-profile run).
usage: a17b_screens.py --out DIR --part bedrock|gapfill|final [--data $LEMCVT_DATA]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17b_common import (  # noqa: E402
    A_C_KS,
    BEDROCK_PASS_FRAC,
    GAPFILL_MAX_FRAC,
    GEOLOGY_PURL,
    GEOLOGY_SHP,
    PRIMARY_EXPECTED,
    QUATERNARY_COUNTED_GEN_LAB,
    QUATERNARY_LANDSLIDE_GEN_LAB,
    TIER1,
    TIER2,
    dump,
    sha256_file,
    slurm_record,
    tile_registry,
)

GM_CTRL = [f"GM{n}_fema" for n in range(1, 8)]
GM_ALL = GM_CTRL + [f"GM{n}" for n in range(8, 31)]


# ------------------------------------------------------------------------------------------------ bedrock

def bedrock(a, out):
    from osgeo import ogr, osr
    ogr.UseExceptions()
    reg = tile_registry()
    ds = ogr.Open(GEOLOGY_SHP)
    lyr = ds.GetLayer()
    map_srs = lyr.GetSpatialRef()
    map_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src = osr.SpatialReference()
    src.ImportFromEPSG(6339)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, map_srs)
    FT = 1200.0 / 3937.0
    import rasterio
    per_tile = {}
    polys_for_map = {}
    for t in GM_ALL:
        with rasterio.open(os.path.join(a.data, reg[t]["grids"]["filled"])) as r:
            b = r.bounds
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in [(b.left, b.bottom), (b.right, b.bottom), (b.right, b.top), (b.left, b.top), (b.left, b.bottom)]:
            ring.AddPoint(float(x), float(y))
        sq = ogr.Geometry(ogr.wkbPolygon)
        sq.AddGeometry(ring)
        sq.Transform(ct)
        sq_area = sq.GetArea()
        lyr.SetSpatialFilter(sq)
        acc = {}
        for f in lyr:
            g = f.GetGeometryRef()
            inter = g.Intersection(sq)
            if inter is None or inter.IsEmpty():
                continue
            key = (f.GetField("GEN_LAB"), f.GetField("DET_LAB"), f.GetField("GEN_NAME"), f.GetField("PERIOD"), f.GetField("EPOCH"))
            acc[key] = acc.get(key, 0.0) + inter.GetArea() / sq_area
            polys_for_map.setdefault(f.GetFID(), (f.GetField("GEN_LAB"), f.GetField("PERIOD"), g.Clone()))
        lyr.SetSpatialFilter(None)
        per_tile[t] = {"fp_6339": [b.left, b.bottom, b.right, b.top], "square_area_km2": sq_area * FT * FT / 1e6, "units": [
            {"GEN_LAB": k[0], "DET_LAB": k[1], "GEN_NAME": k[2], "PERIOD": k[3], "EPOCH": k[4], "fraction": v} for k, v in sorted(acc.items(), key=lambda kv: -kv[1])],
            "coverage": float(sum(acc.values()))}
    # the reference unit: largest non-Quaternary general unit under the seven control footprints
    comb = {}
    for t in GM_CTRL:
        for u in per_tile[t]["units"]:
            if u["PERIOD"] != "Quaternary":
                comb[u["GEN_LAB"]] = comb.get(u["GEN_LAB"], 0.0) + u["fraction"]
    ref = max(comb.items(), key=lambda kv: kv[1])[0]
    ref_name = next(u["GEN_NAME"] for t in GM_CTRL for u in per_tile[t]["units"] if u["GEN_LAB"] == ref)
    rows = []
    for t in GM_ALL:
        u = per_tile[t]["units"]
        f_ref = sum(x["fraction"] for x in u if x["GEN_LAB"] == ref)
        f_q = sum(x["fraction"] for x in u if x["PERIOD"] == "Quaternary" and x["GEN_LAB"] in QUATERNARY_COUNTED_GEN_LAB)
        f_ls = sum(x["fraction"] for x in u if x["GEN_LAB"] in QUATERNARY_LANDSLIDE_GEN_LAB)
        f_other_bedrock = sum(x["fraction"] for x in u if x["PERIOD"] != "Quaternary" and x["GEN_LAB"] != ref)
        f_other_q = sum(x["fraction"] for x in u if x["PERIOD"] == "Quaternary" and x["GEN_LAB"] not in QUATERNARY_COUNTED_GEN_LAB and x["GEN_LAB"] not in QUATERNARY_LANDSLIDE_GEN_LAB)
        frac = f_ref + f_q
        other_units = "; ".join(f"{x['GEN_LAB']} ({x['GEN_NAME']}) {100 * x['fraction']:.1f} %" for x in u if x["GEN_LAB"] != ref)
        rows.append({"tile": t, "kind": "control" if t in GM_CTRL else "drawn", "frac_reference": f_ref, "frac_quaternary_counted": f_q, "frac_rule": frac,
                     "frac_landslide": f_ls, "frac_other_bedrock": f_other_bedrock, "frac_other_quaternary": f_other_q, "coverage": per_tile[t]["coverage"],
                     "pass": bool(frac >= BEDROCK_PASS_FRAC), "tier1": t in TIER1, "other_units": other_units})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "bedrock_screen.csv"), index=False)
    ctrl_fail = [t for t in GM_CTRL if not df.set_index("tile").loc[t, "pass"]]
    rec = {"map": {"path": GEOLOGY_SHP, "purl": GEOLOGY_PURL, "sha256_shp": sha256_file(GEOLOGY_SHP), "sha256_dbf": sha256_file(GEOLOGY_SHP[:-4] + ".dbf"),
                   "crs": map_srs.GetName(), "n_features": lyr.GetFeatureCount()},
           "reference_unit": {"GEN_LAB": ref, "GEN_NAME": ref_name, "combined_fraction_over_controls": comb[ref], "candidates": comb},
           "rule": {"pass_fraction": BEDROCK_PASS_FRAC, "quaternary_counted": sorted(QUATERNARY_COUNTED_GEN_LAB), "landslide_not_counted": sorted(QUATERNARY_LANDSLIDE_GEN_LAB),
                    "note": "fraction = reference unit + counted Quaternary units (by GEN_LAB; queried labels count with their general label); PERIOD == 'Quaternary' defines surficial units, so QT units are bedrock"},
           "tiles": per_tile, "table": rows, "fail": [r["tile"] for r in rows if not r["pass"]], "control_fail": ctrl_fail,
           "stop_before_m_profiles": bool(ctrl_fail)}
    dump(rec, os.path.join(out, "bedrock_screen.json"))
    print(df.to_string(), flush=True)
    print("reference unit:", ref, ref_name, "fail:", rec["fail"], "control fail:", ctrl_fail, flush=True)
    # one-panel map
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon as MplPoly
        from matplotlib.patches import Rectangle
        inv = osr.CoordinateTransformation(map_srs, src)
        pad = 3000.0
        xs = [v["fp_6339"][0] for v in per_tile.values()] + [v["fp_6339"][2] for v in per_tile.values()]
        ys = [v["fp_6339"][1] for v in per_tile.values()] + [v["fp_6339"][3] for v in per_tile.values()]
        box = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]), (box[0], box[1])]:
            ring.AddPoint(x, y)
        bq = ogr.Geometry(ogr.wkbPolygon)
        bq.AddGeometry(ring)
        bq.Transform(ct)
        lyr.SetSpatialFilter(bq)
        labs = {}
        patches, cols = [], []
        import colorsys
        for f in lyr:
            g = f.GetGeometryRef().Clone()
            g.Transform(inv)
            gl = f.GetField("GEN_LAB")
            per = f.GetField("PERIOD")
            if gl not in labs:
                labs[gl] = (len(labs), per, f.GetField("GEN_NAME"))
            for gi in (range(g.GetGeometryCount()) if g.GetGeometryName() == "MULTIPOLYGON" else [None]):
                poly = g.GetGeometryRef(gi) if gi is not None else g
                r0 = poly.GetGeometryRef(0)
                pts = [(r0.GetX(i), r0.GetY(i)) for i in range(r0.GetPointCount())]
                patches.append(MplPoly(pts, closed=True))
                cols.append(gl)
        lyr.SetSpatialFilter(None)
        keys = sorted(labs, key=lambda k: labs[k][0])
        def colour(gl):
            k, per, _ = labs[gl]
            if gl == ref:
                return (0.95, 0.85, 0.55)
            if gl in QUATERNARY_LANDSLIDE_GEN_LAB:
                return (0.85, 0.45, 0.45)
            if per == "Quaternary" and gl in QUATERNARY_COUNTED_GEN_LAB:
                return (0.80, 0.90, 0.75)
            h = (0.55 + 0.37 * (k % 7)) % 1.0
            return colorsys.hls_to_rgb(h, 0.65 if per == "Quaternary" else 0.5, 0.45)
        fig, ax = plt.subplots(figsize=(11, 11 * (box[3] - box[1]) / (box[2] - box[0]) + 1.5), dpi=200)
        pc = PatchCollection(patches, facecolors=[colour(c) for c in cols], edgecolors="k", linewidths=0.15, alpha=0.9)
        ax.add_collection(pc)
        for r in rows:
            fp = per_tile[r["tile"]]["fp_6339"]
            ec = "#1f77b4" if r["pass"] else "#d62728"
            ax.add_patch(Rectangle((fp[0], fp[1]), fp[2] - fp[0], fp[3] - fp[1], fill=False, ec=ec, lw=1.6, ls="-" if r["kind"] == "drawn" else "--", zorder=5))
            ax.text((fp[0] + fp[2]) / 2, (fp[1] + fp[3]) / 2, f"{r['tile']}\n{100 * r['frac_rule']:.0f} %", ha="center", va="center", fontsize=5, fontweight="bold", zorder=6,
                    bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 0.5})
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Patch(fc=colour(k), ec="k", lw=0.3, label=f"{k}: {labs[k][2]} ({labs[k][1]})") for k in keys]
        handles += [Line2D([], [], color="#1f77b4", lw=1.6, label=f"tile passes (>= {100 * BEDROCK_PASS_FRAC:.0f} %)"), Line2D([], [], color="#d62728", lw=1.6, label="tile fails (off-mesa)"),
                    Line2D([], [], color="k", lw=1.6, ls="--", label="control tile GM<n>_fema")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.5, frameon=False)
        ax.set_xlim(box[0], box[2])
        ax.set_ylim(box[1], box[3])
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
        ax.set_xlabel("Easting (km, EPSG:6339)")
        ax.set_ylabel("Northing (km, EPSG:6339)")
        ax.set_title(f"GM bedrock screen: Digital Geologic Map of Monterey County (Rosenberg, PURL cm427jp1187) over the 30 GM tiles; reference unit {ref} ({ref_name}); "
                     f"label = fraction of the footprint in the reference unit + counted Quaternary units", fontsize=7, wrap=True)
        fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "a17b")
        os.makedirs(fdir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(fdir, f"GM_bedrock_screen.{ext}"), dpi=200, bbox_inches="tight")
            fig.savefig(os.path.join(out, f"GM_bedrock_screen.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("map written", os.path.getsize(os.path.join(fdir, "GM_bedrock_screen.png")), "bytes", flush=True)
    except Exception as e:  # pragma: no cover
        print("map failed", repr(e), flush=True)
    return rec


# ------------------------------------------------------------------------------------------------ gap fill

def gapfill(a, out):
    import rasterio

    from lemcvt.measure.audits import MIN_CHI_RANGE_M, MIN_PIXELS_PER_BASIN, MIN_RELIEF_M, THETA_REF_CHI
    from lemcvt.measure.chi import channel_network, chi_for_thetas, relative_elevation
    from lemcvt.measure.phase6_inputs import W_F, _measure_curvature_params, option_iii_mask
    from lemcvt.measure.pipeline import _metre_tile
    reg = tile_registry()
    curv = _measure_curvature_params()
    rows = []
    for site, tiles in PRIMARY_EXPECTED.items():
        for t in tiles:
            t0 = time.time()
            td = _metre_tile(t)
            Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
            masks = reg[t]["masks"]
            fill = np.zeros(Z.shape, bool)
            for k in ("nnfill_mask", "gapfill_mask"):
                if k in masks:
                    with rasterio.open(os.path.join(a.data, masks[k])) as ds:
                        fill |= ds.read(1) == 1
            # (a) the channel pixels of the k_s fit: network at A_c = 1e3, basins kept by the pre-registered filters
            net = channel_network(Z, A, FD, A_C_KS, dx, dy)
            chi_ref = chi_for_thetas(net, np.array([THETA_REF_CHI]))[:, 0]
            zrel = relative_elevation(net)
            keep = np.zeros(net.rows.size, bool)
            n_b = n_k = 0
            for o in net.outlet_index:
                idx = np.nonzero(net.basin == o)[0]
                idx = idx[idx != o]
                n_b += 1
                if idx.size >= MIN_PIXELS_PER_BASIN and float(zrel[idx].max()) >= MIN_RELIEF_M and float(chi_ref[idx].max()) >= MIN_CHI_RANGE_M:
                    keep[idx] = True
                    n_k += 1
            ch_fill = fill[net.rows[keep], net.cols[keep]]
            # (b) the option (iii) hilltop pixels at L_f
            cp = curv.get(t)
            if not cp:
                raise FileNotFoundError(f"{t}: no measure run with a rollover A_c")
            mask, mask0, dist, diag = option_iii_mask(Z, A, FD, dx, dy, float(cp["A_c"]), W_F)
            ht_fill = fill[mask]
            r = {"tile": t, "site": site, "tier2": t in TIER2, "n_channel_px_kept": int(keep.sum()), "n_basins_kept": n_k, "n_basins": n_b,
                 "frac_fill_channel": float(ch_fill.mean()) if ch_fill.size else None, "n_hilltop_px": int(mask.sum()), "frac_fill_hilltop": float(ht_fill.mean()) if ht_fill.size else None,
                 "frac_fill_tile": float(fill.mean()), "A_c_rollover": float(cp["A_c"]), "measure_run": cp.get("_run"), "has_masks": bool(masks), "seconds": time.time() - t0}
            r["pass"] = bool((r["frac_fill_channel"] or 0.0) <= GAPFILL_MAX_FRAC and (r["frac_fill_hilltop"] or 0.0) <= GAPFILL_MAX_FRAC)
            rows.append(r)
            print(json.dumps(r), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "gapfill_screen.csv"), index=False)
    rec = {"rule": {"max_fraction": GAPFILL_MAX_FRAC, "a": f"channel pixels of the k_s fit (A >= {A_C_KS:g} m2, kept basins)", "b": "option (iii) hilltop pixels at L_f = 11 m"},
           "table": rows, "fail": [r["tile"] for r in rows if not r["pass"]]}
    dump(rec, os.path.join(out, "gapfill_screen.json"))
    print("gap-fill fail:", rec["fail"], flush=True)
    return rec


def final(a, out):
    b = json.load(open(os.path.join(out, "bedrock_screen.json")))
    g = json.load(open(os.path.join(out, "gapfill_screen.json")))
    primary, removed = {}, {}
    for site, tiles in PRIMARY_EXPECTED.items():
        keep, rem = [], []
        for t in tiles:
            why = []
            if site == "GM" and t in b["fail"]:
                why.append("off-mesa (bedrock screen)")
            if t in g["fail"]:
                why.append("gap-filled (> 20 % of the measurement pixels)")
            (rem if why else keep).append({"tile": t, "reason": "; ".join(why)} if why else t)
        primary[site] = keep
        removed[site] = rem
    rec = {"fixed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "primary": primary, "counts": {s: len(v) for s, v in primary.items()},
           "removed_by_screens": removed, "tier1_excluded": TIER1, "tier2": TIER2, "clean_subset": {s: [t for t in v if t not in TIER2] for s, v in primary.items()},
           "expected_before_screens": {s: len(v) for s, v in PRIMARY_EXPECTED.items()}, "bedrock_reference_unit": b["reference_unit"]["GEN_LAB"],
           "stop_before_m_profiles": b["stop_before_m_profiles"]}
    dump(rec, os.path.join(out, "primary_set.json"))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("tier1_excluded", "tier2")}, indent=1), flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--part", required=True, choices=["bedrock", "gapfill", "final"])
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    rec = {"bedrock": bedrock, "gapfill": gapfill, "final": final}[a.part](a, a.out)
    if a.part == "bedrock" and rec["stop_before_m_profiles"]:
        print("STOP: a control tile fails the bedrock screen", flush=True)
        sys.exit(3)
    if a.part == "final":
        print("primary set fixed", rec["counts"], flush=True)


if __name__ == "__main__":
    main()
