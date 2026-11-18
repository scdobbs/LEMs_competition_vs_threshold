#!/usr/bin/env python3
"""Addendum 14, 4.1.2: is the USGS 3DEP copy of PA_WesternPA_1_2019 the same point set as PASDA's LiDAR2020 PA_South files?

The AWS Entwine copy (usgs-lidar-public) is a re-tiled, EPSG:3857 octree and cannot equal PASDA's coordinates; its source
list names the original LAZ tiles (s3://usgs-lidar/Projects/PA_WesternPA_2019_D20/PA_WesternPA_1_2019/LAZ/, requester
pays), which USGS also stages on rockyweb in the delivered parameters. This script, for ONE PASDA quadrant already on
Sherlock:
  1. selects the USGS 1 km LAZ tiles whose EPSG:3857 bounds (Entwine source list) intersect the quadrant;
  2. downloads them from rockyweb (resumable curl), timing the transfer, and reads their headers (CRS, units, counts);
  3. reprojects their class-2 points into the PASDA CRS (EPSG:6565, ftUS) with PDAL, crops to the quadrant square, and
     compares with PASDA's class-2 points in the quadrant: counts, sha256 of the sorted coordinates rounded to 0.01 ft,
     and the fraction of PASDA points with a USGS point within 0.01 ft / 0.1 ft (nearest neighbour on a 200k sample);
  4. times an 8 MB range request from PASDA for the same quadrant from this compute node (rate comparison).

usage: a14_usgs_vs_pasda.py --quadrant 19001270PAS_NE --out DIR --ept-sources JSON --listing HTML --work DIR
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a13_pasda_check import quadrant_bounds  # noqa: E402
from a14_common import SOURCE, dump, slurm_record  # noqa: E402

ROCKY = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/PA_WesternPA_2019_D20/PA_WesternPA_1_2019/LAZ/"
PASDA = "https://pasda.psu.edu/download/usgs/LiDAR2020/LAS/PA_South/"
UA = "Mozilla/5.0 (research; scdobbs@stanford.edu)"


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout, r.stderr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quadrant", default="19001270PAS_NE")
    p.add_argument("--out", required=True)
    p.add_argument("--ept-sources", required=True)
    p.add_argument("--listing", required=True)
    p.add_argument("--work", default=os.environ.get("L_SCRATCH_JOB", "/tmp"))
    p.add_argument("--skip-rockyweb", action="store_true", help="only the EPT (AWS) route")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    slurm_record(a.out)
    from pyproj import Transformer
    rec = {"quadrant": a.quadrant}
    E0, N0, E1, N1 = quadrant_bounds(a.quadrant + "_LAS.zip")
    rec["quadrant_bounds_ftus"] = [E0, N0, E1, N1]
    t = Transformer.from_crs(6565, 3857, always_xy=True)
    xs, ys = t.transform([E0, E1, E1, E0], [N0, N0, N1, N1])
    qb = [min(xs), min(ys), max(xs), max(ys)]
    srcs = json.load(open(a.ept_sources))
    sizes = {n: int(s) for n, _d, s in re.findall(r'<a href="([^"]+\.laz)">[^<]*</a>\s*(\d{1,2}-\w{3}-\d{4} \d\d:\d\d)\s+(\d+)', open(a.listing).read())}
    tiles = []
    for s in srcs:
        b = s["bounds"]
        if b[0] < qb[2] and b[3] > qb[0] and b[1] < qb[3] and b[4] > qb[1]:
            name = os.path.basename(s["path"])
            tiles.append({"file": name, "points_ept": s.get("points"), "bytes_listed": sizes.get(name)})
    if a.skip_rockyweb:
        tiles = []
    rec["usgs_tiles"] = tiles
    print("USGS tiles intersecting the quadrant:", [(x["file"], x["bytes_listed"]) for x in tiles], flush=True)
    # 2. download
    dl = []
    procs = []
    t0 = time.time()
    def valid_laz(path, size):
        if not os.path.exists(path):
            return False
        if size and os.path.getsize(path) != size:
            return False
        with open(path, "rb") as fh:
            return fh.read(4) == b"LASF"

    def start(x):
        dst = os.path.join(a.work, x["file"])
        x["local"] = dst
        if valid_laz(dst, x["bytes_listed"]):
            return None
        if os.path.exists(dst) and (os.path.getsize(dst) < 1000 or open(dst, "rb").read(4) != b"LASF"):
            os.remove(dst)          # a 429 body or a corrupt partial: start again
        return subprocess.Popen(["curl", "-s", "-L", "-C", "-", "-A", UA, "-m", "9000", "-o", dst, "-w", "%{http_code} %{size_download} %{speed_download}",
                                 ROCKY + x["file"]], stdout=subprocess.PIPE, text=True)
    # two connections at a time (rockyweb answers 429 to bursts of new connections; each connection runs at ~75-120 KB/s)
    pending = list(tiles)
    running = []
    while pending or running:
        while pending and len(running) < 2:
            x = pending.pop(0)
            pr = start(x)
            if pr is None:
                procs.append((x, None))
            else:
                running.append((x, pr))
                time.sleep(3)
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                procs.append(item)
        time.sleep(2)
    for x, pr in procs:
        so = pr.communicate()[0] if pr is not None else "cached"
        if so.startswith("429"):
            time.sleep(30)
            pr2 = start(x)
            so = pr2.communicate()[0] if pr2 is not None else so
        dl.append({"file": x["file"], "curl": so, "seconds": time.time() - t0, "bytes": os.path.getsize(x["local"]) if os.path.exists(x["local"]) else 0,
                   "bytes_listed": x["bytes_listed"]})
        print("downloaded", dl[-1], flush=True)
    rec["download"] = dl
    rec["download_wall_seconds"] = time.time() - t0
    # 2b. the AWS Entwine copy (EPSG:3857): read the quadrant with readers.ept, time it, reproject to EPSG:6565
    poly = f"POLYGON(({E0} {N0},{E1} {N0},{E1} {N1},{E0} {N1},{E0} {N0}))"
    ept = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/PA_WesternPA_1_2019/ept.json"
    ept_out = os.path.join(a.work, f"ept_{a.quadrant}_6565.laz")
    t0 = time.time()
    bounds = f"([{qb[0] - 50},{qb[2] + 50}],[{qb[1] - 50},{qb[3] + 50}])"
    pipe = {"pipeline": [{"type": "readers.ept", "filename": ept, "bounds": bounds, "resolution": 0},
                         {"type": "filters.range", "limits": "Classification[2:2]"},
                         {"type": "filters.reprojection", "in_srs": "EPSG:3857", "out_srs": "EPSG:6565"},
                         {"type": "filters.crop", "polygon": poly},
                         {"type": "writers.las", "filename": ept_out, "compression": "laszip", "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001, "offset_x": "auto", "offset_y": "auto", "offset_z": "auto", "a_srs": "EPSG:6565"}]}
    pp = os.path.join(a.work, "ept_pipeline.json")
    json.dump(pipe, open(pp, "w"))
    code, so, se = run(["pdal", "pipeline", pp, "--metadata", os.path.join(a.work, "ept_meta.json")])
    rec["ept"] = {"url": ept, "bounds_3857": bounds, "exit": code, "seconds": time.time() - t0, "stderr_tail": se[-600:] if code else "",
                  "laz_bytes": os.path.getsize(ept_out) if os.path.exists(ept_out) else 0}
    try:
        em = json.load(open(os.path.join(a.work, "ept_meta.json")))
        md = em.get("metadata", em.get("stages", em))
        rd = md.get("readers.ept", {})
        if isinstance(rd, list):
            rd = rd[0] if rd else {}
        wl = md.get("writers.las", {})
        if isinstance(wl, list):
            wl = wl[0] if wl else {}
        rec["ept"]["reader_metadata"] = {k: rd.get(k) for k in ("count", "srs") if k in rd}
        rec["ept"]["points_read_class2_in_quadrant"] = wl.get("count")
        rec["ept"]["metadata_keys"] = list(md.keys())[:10]
    except Exception as e:
        rec["ept"]["meta_error"] = repr(e)
    print("EPT read", {k: v for k, v in rec["ept"].items() if k != "reader_metadata"}, flush=True)
    # PASDA rate from this node (8 MB range)
    t0 = time.time()
    code, so, se = run(["curl", "-s", "-L", "-r", "0-8388607", "-o", os.devnull, "-m", "300", "-w", "%{http_code} %{size_download} %{speed_download}", PASDA + a.quadrant + "_LAS.zip"])
    rec["pasda_range_test"] = {"curl": so, "seconds": time.time() - t0}
    print("PASDA range test", rec["pasda_range_test"], flush=True)
    # headers
    hdr = []
    for x in tiles:
        code, so, se = run(["pdal", "info", "--metadata", x["local"]])
        if code == 0:
            md = json.loads(so)["metadata"]
            srs = md.get("srs", {})
            hdr.append({"file": x["file"], "count": md.get("count"), "version": f"{md.get('major_version')}.{md.get('minor_version')}", "point_format": md.get("dataformat_id"),
                        "software": md.get("software_id"), "system": md.get("system_id"), "scale": [md.get("scale_x"), md.get("scale_y"), md.get("scale_z")],
                        "minx": md.get("minx"), "maxx": md.get("maxx"), "miny": md.get("miny"), "maxy": md.get("maxy"), "minz": md.get("minz"), "maxz": md.get("maxz"),
                        "srs_horizontal": (srs.get("horizontal") or "")[:200], "srs_vertical": (srs.get("vertical") or "")[:200], "srs_units": srs.get("units"),
                        "compound_wkt_head": (srs.get("compoundwkt") or srs.get("wkt") or "")[:160]})
        else:
            hdr.append({"file": x["file"], "error": se[-500:]})
    rec["usgs_headers"] = hdr
    print("headers", json.dumps(hdr, indent=1)[:1500], flush=True)
    # 3. class-2 points of both, in EPSG:6565, cropped to the quadrant
    pasda_las = os.path.join(SOURCE, "AP-3DEP", "las", a.quadrant + "_LAS.las")
    def class2_xyz(path, reproject):
        out = os.path.join(a.work, os.path.basename(path) + ".c2.txt")
        stages = [{"type": "readers.las", "filename": path}, {"type": "filters.range", "limits": "Classification[2:2]"}]
        if reproject:
            stages.append({"type": "filters.reprojection", "out_srs": "EPSG:6565"})
        stages += [{"type": "filters.crop", "polygon": poly}, {"type": "writers.text", "filename": out, "order": "X,Y,Z", "keep_unspecified": False, "write_header": False, "precision": 4}]
        pipe = os.path.join(a.work, os.path.basename(path) + ".json")
        json.dump({"pipeline": stages}, open(pipe, "w"))
        code, so, se = run(["pdal", "pipeline", pipe])
        if code != 0:
            raise RuntimeError(se[-800:])
        arr = np.loadtxt(out, delimiter=",", ndmin=2)
        return arr
    t0 = time.time()
    P = class2_xyz(pasda_las, reproject=False)
    rec["pasda_class2_in_quadrant"] = int(P.shape[0])
    parts = []
    for x in tiles:
        if not valid_laz(x["local"], x["bytes_listed"]):
            x["skipped"] = "not a complete LAZ after download"
            continue
        try:
            parts.append(class2_xyz(x["local"], reproject=True))
        except RuntimeError as e:
            x["error"] = str(e)[-300:]
    U = np.vstack(parts) if parts else np.zeros((0, 3))
    rec["usgs_tiles_used"] = len(parts)
    rec["usgs_class2_in_quadrant"] = int(U.shape[0])
    rec["count_equal"] = bool(P.shape[0] == U.shape[0])
    E = class2_xyz(ept_out, reproject=False) if os.path.exists(ept_out) and os.path.getsize(ept_out) > 1000 else np.zeros((0, 3))
    rec["ept_class2_in_quadrant"] = int(E.shape[0])
    rec["ept_count_equal_pasda"] = bool(P.shape[0] == E.shape[0])
    def checksum(arr, dec):
        r = np.round(arr, dec)
        order = np.lexsort((r[:, 2], r[:, 1], r[:, 0]))
        return hashlib.sha256(np.ascontiguousarray(r[order]).tobytes()).hexdigest()
    rec["sha256_sorted_xyz_0.01ft"] = {"pasda": checksum(P, 2), "usgs": checksum(U, 2)} if P.size and U.size else None
    rec["sha256_sorted_xyz_0.1ft"] = {"pasda": checksum(P, 1), "usgs": checksum(U, 1)} if P.size and U.size else None
    if P.size and U.size:
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(0)
        samp = P[rng.choice(P.shape[0], size=min(200000, P.shape[0]), replace=False)]
        tree = cKDTree(U[:, :2])
        d, i = tree.query(samp[:, :2], k=1)
        dz = np.abs(U[i, 2] - samp[:, 2])
        rec["nn_match"] = {"n_sample": int(samp.shape[0]), "frac_xy_within_0.01ft": float(np.mean(d <= 0.01)), "frac_xy_within_0.1ft": float(np.mean(d <= 0.1)),
                           "frac_xy_within_1ft": float(np.mean(d <= 1.0)), "median_xy_dist_ft": float(np.median(d)), "p95_xy_dist_ft": float(np.percentile(d, 95)),
                           "frac_z_within_0.01ft_given_xy_0.1": float(np.mean(dz[d <= 0.1] <= 0.01)) if (d <= 0.1).any() else None,
                           "median_abs_dz_ft_xy_matched": float(np.median(dz[d <= 0.1])) if (d <= 0.1).any() else None}
        rec["bounds_ftus"] = {"pasda": [float(v) for v in np.r_[P[:, :2].min(0), P[:, :2].max(0)]], "usgs": [float(v) for v in np.r_[U[:, :2].min(0), U[:, :2].max(0)]]}
    if P.size and E.size:
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(1)
        samp = P[rng.choice(P.shape[0], size=min(200000, P.shape[0]), replace=False)]
        tree = cKDTree(E[:, :2])
        d, i = tree.query(samp[:, :2], k=1)
        dz = np.abs(E[i, 2] - samp[:, 2])
        rec["ept_nn_match"] = {"n_sample": int(samp.shape[0]), "frac_xy_within_0.01ft": float(np.mean(d <= 0.01)), "frac_xy_within_0.05ft": float(np.mean(d <= 0.05)),
                               "frac_xy_within_0.1ft": float(np.mean(d <= 0.1)), "frac_xy_within_1ft": float(np.mean(d <= 1.0)), "median_xy_dist_ft": float(np.median(d)),
                               "p95_xy_dist_ft": float(np.percentile(d, 95)), "median_abs_dz_ft_xy_matched_0.1": float(np.median(dz[d <= 0.1])) if (d <= 0.1).any() else None,
                               "frac_z_equal_0.01ft_given_xy_0.1": float(np.mean(dz[d <= 0.1] <= 0.01)) if (d <= 0.1).any() else None,
                               "z_ratio_median_ept_over_pasda": float(np.median(E[i, 2] / samp[:, 2]))}
        rec.setdefault("sha256_sorted_xyz_0.01ft", {}) or rec.__setitem__("sha256_sorted_xyz_0.01ft", {})
        rec["sha256_sorted_xyz_0.01ft"]["ept"] = checksum(E, 2)
        rec["sha256_sorted_xyz_0.01ft"]["pasda"] = checksum(P, 2)
        rec.setdefault("z_range_ftus", {}) or rec.__setitem__("z_range_ftus", {})
        rec["z_range_ftus"]["ept"] = [float(E[:, 2].min()), float(E[:, 2].max())]
        rec["z_range_ftus"]["pasda"] = [float(P[:, 2].min()), float(P[:, 2].max())]
        rec["z_range_ftus"] = {"pasda": [float(P[:, 2].min()), float(P[:, 2].max())] if P.size else None, "usgs": [float(U[:, 2].min()), float(U[:, 2].max())] if U.size else None}
    rec["seconds_compare"] = time.time() - t0
    print(json.dumps({k: v for k, v in rec.items() if k not in ("usgs_tiles", "download", "usgs_headers")}, indent=1), flush=True)
    dump(rec, os.path.join(a.out, f"usgs_vs_pasda_{a.quadrant}{a.tag}.json"))
    rec["laz_kept_at"] = a.work


if __name__ == "__main__":
    main()
