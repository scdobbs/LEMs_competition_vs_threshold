#!/usr/bin/env python3
"""Addendum 15, section 2: georeference Montgomery & Dietrich (1989) Fig. 3 and digitise it (EPSG:6339 GeoPackage).

Stages (``--stage``):
  extract   Fig. 3 from the PDF (page 3, 300 dpi CCITT scan, 2455 x 3379 px; poppler pdfimages), cropped to the map;
            masks: black pixels; thick lines (7 x 7 disk opening = channels, boundary dashes, symbols); channel network =
            thick components > 600 px excluding the north arrow and scale bar; thin lines = contours and outlines.
  locate    coarse placement: normalised cross-correlation of the figure's channel network (blurred, at the scale-bar
            scale 500 m = 398 px) against the D8 channel network (A >= 2e4 m^2) of the stage A TTV 1 m grid over
            538-546 km E, 4185-4194 km N, at 5 m, scales 0.9-1.1 and rotations +-3 deg. Then local refinement:
            150 m patches of the figure's network matched within +-110 m, and a robust affine from the patch matches.
  gcp       ground control: channel confluences of the figure (junctions of the thinned channel skeleton) paired with
            D8 confluences of the 1 m grid (two donors >= 2e3 m^2 meeting at >= 4e3 m^2) within 20 m of the patch
            affine with no other confluence within 40 m; the reviewed set (REVIEWED, chips inspected by eye, plus the
            man-made pond) is fitted by least squares; residuals; rule 2.1 (RMS <= 30 m, every point <= 60 m).
  digitize  study-area boundary (dash-dot line: thick components that are not the channel network, closed by a
            25 px dilation, thinned and traced); channels (thinned network, traced node to node); channel heads
            (upstream endpoints; see head_rule); colluvial deposits (stipple-dot density); GeoPackage in EPSG:6339.
  check     rule 2.2: polygon area 2.1 km^2 +-15 %; elevation range on the 1 m grid 61-314 m, +-15 m at each end.
usage: a15_md_georef.py --out DIR [--stage all]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15_common import STAGE_A, dump, slurm_record  # noqa: E402

PDF = ("papers/Montgomery_and_Dietrich_1989_Water Resources Research - August 1989 - Montgomery - Source areas  drainage density  "
       "and channel initiation.pdf")
POPPLER = "/share/software/user/open/poppler/0.47.0"
CROP = (330, 250, 2120, 1960)          # (left, top, right, bottom) of Fig. 3 on page 3 (page pixels)
SCALE_BAR_ROWS = (1570, 1650)          # in the crop: the 0-500 m bar
NORTH_ARROW = (0, 330, 0, 450)         # rows 0-330, cols 0-450 of the crop: the N arrow
TTV_GRID = os.path.join(STAGE_A, "grid_TTV")
WIN = (538700, 541500, 4189200, 4192000)   # x0, x1, y0, y1 of the 1 m working window (EPSG:6339)
FT = 0.3048
# Reviewed ground-control features (figure pixel row, col of the feature on the crop; DEM feature from the gcp stage).
# Confluences: identified on the chips image (figure junction and D8 confluence side by side); the pond: the solid black
# symbol at crop (825, 815) and the flat-floored basin with a dam on the 1 m hillshade at 539,998 E, 4,190,672 N.
REVIEWED = {"confluences_fig_rc": [[1305, 1154], [823, 1589], [1057, 910], [817, 791], [988, 622], [1018, 513], [708, 613], [656, 533],
                                   [978, 384], [270, 1072], [408, 1227], [1223, 870], [715, 1224]],
            "pond": {"fig_rc": [825, 815], "utm": [539998.0, 4190672.0]}}
LETTERS = {"A": [800, 1030], "B": [1080, 1270], "C": [1340, 1000], "D": [300, 1120]}   # approximate crop (row, col) of the labels
CULVERT_TRIANGLE = [778, 528]
# Head review (by eye on head_candidates_<quadrant>.png): candidate ids to drop (end points that are not mapped heads:
# contour-merge spurs, dash ends, lower ends of discontinuous reaches) and mapped heads to add as crop (row, col).
HEAD_REVIEW = {"drop": [36, 51, 53, 54, 61, 67, 79, 81, 82], "add": []}
# dropped: 36 spur on the main east-west channel (merged contour); 51, 53 downstream ends of discontinuous reaches on the
# north-facing slope (the 1 m grid put them above their upper ends); 54, 61, 67, 81, 82 second tips of forks a few pixels
# from a kept head; 79 downstream end of the reach below head 76. Candidate ids refer to head_candidates.csv of the
# 2026-11-18 09:19 PST run (job 52822212); the ids are stable for the same masks.
# The dash-dot study-area boundary, digitised by hand from gridded views of the crop (50 px grid; 25 px on the south-east
# section), clockwise from the west side; (row, col). On the south-east the boundary follows the solid tributary channel
# between the last dash of the north-east arc (1150, 1425) and the dashes at row 1425-1432.
BOUNDARY_RC = [(870, 205), (840, 215), (790, 235), (735, 250), (690, 265), (650, 280), (610, 300), (575, 320), (545, 335), (520, 350), (500, 360), (470, 375),
               (455, 400), (440, 430), (435, 460), (432, 500), (430, 540), (425, 580), (420, 620), (415, 660), (410, 700), (400, 740), (385, 770), (360, 800),
               (335, 815), (310, 825), (290, 835), (265, 850), (240, 865), (215, 880), (190, 895), (170, 905), (145, 915), (125, 925), (112, 945), (108, 965),
               (108, 990), (110, 1010), (120, 1040), (135, 1080), (145, 1110), (165, 1150), (185, 1190), (205, 1230), (225, 1270), (245, 1300), (265, 1340),
               (280, 1380), (300, 1420), (320, 1450), (350, 1480), (380, 1500), (410, 1520), (440, 1540), (470, 1555), (500, 1570), (530, 1585), (555, 1600),
               (580, 1620), (605, 1640), (635, 1660), (665, 1680), (690, 1695), (740, 1660), (790, 1630), (840, 1600), (885, 1580), (950, 1560), (975, 1545),
               (1010, 1530), (1040, 1515), (1070, 1500), (1100, 1480), (1125, 1455), (1150, 1425), (1165, 1400), (1200, 1360), (1240, 1330), (1280, 1305),
               (1320, 1297), (1370, 1297), (1400, 1300), (1425, 1290), (1432, 1250), (1428, 1200), (1425, 1150), (1445, 1100), (1480, 1050), (1500, 1000),
               (1500, 960), (1492, 920), (1480, 880), (1465, 870), (1445, 840), (1420, 810), (1395, 780), (1360, 750), (1330, 720), (1300, 690), (1270, 650),
               (1250, 610), (1235, 570), (1225, 530), (1215, 490), (1200, 450), (1195, 410), (1190, 370), (1180, 330), (1165, 290), (1145, 250), (1120, 215),
               (1090, 195), (1050, 190), (1000, 190), (950, 195), (900, 200)]


def run_pdfimages(out):
    env = dict(os.environ, LD_LIBRARY_PATH=POPPLER + "/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
    subprocess.run([POPPLER + "/bin/pdfimages", "-png", "-f", "3", "-l", "3", PDF, os.path.join(out, "page3")], check=True, env=env)
    pngs = sorted(f for f in os.listdir(out) if f.startswith("page3") and f.endswith(".png"))
    return os.path.join(out, pngs[0])


def stage_extract(out):
    page = run_pdfimages(out)
    im = Image.open(page)
    rec = {"pdf": PDF, "page": 3, "page_pixels": list(im.size), "dpi": 300, "mode": im.mode, "crop_box_page_px": list(CROP)}
    crop = im.convert("L").crop(CROP)
    crop.save(os.path.join(out, "fig3_crop.png"))
    arr = np.array(crop) < 128
    sub = arr[SCALE_BAR_ROWS[0]:SCALE_BAR_ROWS[1], 150:620]
    r = SCALE_BAR_ROWS[0] + int(np.argmax(sub.sum(1)))
    cols = np.nonzero(arr[r - 2:r + 3, 100:700].any(0))[0] + 100
    bar_px = int(cols.max() - cols.min() + 1)
    rec.update({"crop_pixels": list(arr.shape[::-1]), "scale_bar_px": bar_px, "scale_bar_m": 500.0, "m_per_px": 500.0 / bar_px,
                "map_scale_1_to": 500.0 / bar_px * 300 / 0.0254})
    yy, xx = np.mgrid[-3:4, -3:4]
    disk = (xx ** 2 + yy ** 2) <= 2.9 ** 2        # 6-px disk: contour lines (4-5 px) and stipple dots (3-5 px) drop out, channels (7-9 px) and dashes (8-10 px) stay
    thick = ndimage.binary_opening(arr, structure=disk)
    thick[SCALE_BAR_ROWS[0] - 40:, :] = False
    thick[NORTH_ARROW[0]:NORTH_ARROW[1], NORTH_ARROW[2]:NORTH_ARROW[3]] = False
    lab, n = ndimage.label(thick, structure=np.ones((3, 3)))
    sizes = ndimage.sum(thick, lab, range(1, n + 1))
    net = np.isin(lab, 1 + np.nonzero(sizes > 120)[0])
    thin = arr & ~ndimage.binary_dilation(thick, iterations=2)
    thin[SCALE_BAR_ROWS[0] - 40:, :] = False
    for name, m in (("mask_black", arr), ("mask_thick", thick), ("mask_network", net), ("mask_thin", thin)):
        Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(out, name + ".png"))
    rec.update({"n_thick_components": int(n), "n_network_components": int((sizes > 120).sum()), "network_px": int(net.sum())})
    dump(rec, os.path.join(out, "extract.json"))
    print("extract", rec, flush=True)
    return rec


def load_mask(out, name):
    return np.array(Image.open(os.path.join(out, name + ".png"))) > 0


def ncc(img, t):
    t = t - t.mean()
    n = np.sqrt((t ** 2).sum())
    if n == 0:
        return None
    t = t / n
    i = img - img.mean()
    num = fftconvolve(i, t[::-1, ::-1], mode="valid")
    ones = np.ones_like(t)
    s1 = fftconvolve(i, ones, mode="valid")
    s2 = fftconvolve(i ** 2, ones, mode="valid")
    return num / np.sqrt(np.maximum(s2 - s1 ** 2 / ones.size, 1e-9))


def read_window(name, bounds, out_shape=None):
    with rasterio.open(os.path.join(TTV_GRID, name)) as ds:
        w = from_bounds(bounds[0], bounds[2], bounds[1], bounds[3], ds.transform)
        a = ds.read(1, window=w, out_shape=out_shape)
        nod = ds.nodata
    return a, nod


def stage_locate(out):
    ex = json.load(open(os.path.join(out, "extract.json")))
    mpp = ex["m_per_px"]
    net = load_mask(out, "mask_network")
    step = 5.0
    f = step / mpp
    ny, nx = int(net.shape[0] / f), int(net.shape[1] / f)
    tmpl0 = np.array(Image.fromarray(net.astype(np.uint8) * 255).resize((nx, ny), Image.BOX)).astype(float) / 255.0
    x0, x1, y0, y1 = 538000, 546000, 4185000, 4194000
    a, _ = read_window("TTV_1m_area_cells.tif", (x0, x1, y0, y1))
    s = int(step)
    a = a[:a.shape[0] // s * s, :a.shape[1] // s * s].reshape(a.shape[0] // s, s, a.shape[1] // s, s).max(axis=(1, 3))
    img = ndimage.gaussian_filter((a >= 20000).astype(float), 2.0)
    res = []
    for sc in [0.9, 0.95, 1.0, 1.05, 1.1]:
        for rot in [-3, -1.5, 0, 1.5, 3]:
            t = ndimage.gaussian_filter(ndimage.zoom(ndimage.rotate(tmpl0, rot, reshape=False, order=1), sc, order=1), 2.0)
            c = ncc(img, t)
            k = np.unravel_index(np.argmax(c), c.shape)
            c2 = c.copy()
            c2[max(0, k[0] - 40):k[0] + 40, max(0, k[1] - 40):k[1] + 40] = -1
            k2 = np.unravel_index(np.argmax(c2), c2.shape)
            res.append({"ncc": float(c.max()), "scale": sc, "rot_deg": rot, "top_left_utm": [x0 + k[1] * step, y1 - k[0] * step],
                        "second_ncc": float(c2.max()), "second_top_left_utm": [x0 + k2[1] * step, y1 - k2[0] * step]})
    res.sort(key=lambda r: -r["ncc"])
    best = res[0]
    X0, Y1 = best["top_left_utm"]
    # local refinement: 150 m patches of the network at 1 m
    H, W = int(net.shape[0] * mpp), int(net.shape[1] * mpp)
    raw = np.array(Image.fromarray(net.astype(np.uint8) * 255).resize((W, H), Image.BOX)).astype(float) / 255.0
    fig1 = ndimage.gaussian_filter(raw, 3.0)
    wx0, wx1, wy0, wy1 = WIN
    a1, _ = read_window("TTV_1m_area_cells.tif", WIN)
    dem = ndimage.gaussian_filter((a1.astype(float) >= 3000).astype(float), 3.0)
    rad, srch = 75, 110
    matches = []
    for fr in range(rad, H - rad, 100):
        for fc in range(rad, W - rad, 100):
            t_raw = raw[fr - rad:fr + rad, fc - rad:fc + rad]
            if t_raw.mean() < 0.04:
                continue
            ux, uy = X0 + fc, Y1 - fr
            dr, dc = int(wy1 - uy), int(ux - wx0)
            r0, c0 = dr - rad - srch, dc - rad - srch
            if r0 < 0 or c0 < 0 or dr + rad + srch > dem.shape[0] or dc + rad + srch > dem.shape[1]:
                continue
            c = ncc(dem[r0:dr + rad + srch, c0:dc + rad + srch], fig1[fr - rad:fr + rad, fc - rad:fc + rad])
            pk = np.unravel_index(np.argmax(c), c.shape)
            c2 = c.copy()
            c2[max(0, pk[0] - 20):pk[0] + 20, max(0, pk[1] - 20):pk[1] + 20] = -1
            mx, my = wx0 + c0 + pk[1] + rad, wy1 - (r0 + pk[0] + rad)
            matches.append({"fig_row_px": fr / mpp, "fig_col_px": fc / mpp, "utm_match": [float(mx), float(my)], "shift_m": [float(mx - ux), float(my - uy)],
                            "ncc": float(c.max()), "ncc2": float(c2.max()), "fig_frac": float(t_raw.mean())})
    good = [m for m in matches if m["ncc"] > 0.5 and m["ncc"] > 1.2 * m["ncc2"] and abs(m["shift_m"][0]) < 60 and abs(m["shift_m"][1]) < 60]
    pts = good
    for _it in range(6):
        A, d = fit_affine(pts)
        rms = float(np.sqrt((d ** 2).mean()))
        keep = d < max(2.5 * rms, 15)
        if keep.all():
            break
        pts = [o for o, k in zip(pts, keep, strict=True) if k]
    rec = {"coarse": res[:5], "coarse_best": best, "n_patches": len(matches), "n_good": len(good), "n_used": len(pts), "patch_affine": A.tolist(),
           "patch_affine_rms_m": rms, "patch_affine_max_m": float(d.max()), "pixel_size_m": [float(np.hypot(A[0, 0], A[0, 1])), float(np.hypot(A[1, 0], A[1, 1]))],
           "rotation_deg": [float(np.degrees(np.arctan2(A[0, 1], A[0, 0]))), float(np.degrees(np.arctan2(-A[1, 0], -A[1, 1])))]}
    dump({"rec": rec, "matches": matches}, os.path.join(out, "locate.json"))
    print("locate", json.dumps({k: v for k, v in rec.items() if k != "coarse"}, indent=1), flush=True)
    return rec


def fit_affine(pts):
    P = np.array([[o["fig_col_px"], o["fig_row_px"], 1.0] for o in pts])
    U = np.array([o["utm_match"] for o in pts])
    A, *_ = np.linalg.lstsq(P, U, rcond=None)
    res = U - P @ A
    return A, np.hypot(res[:, 0], res[:, 1])


def apply_affine(A, rc):
    rc = np.atleast_2d(np.asarray(rc, float))
    P = np.c_[rc[:, 1], rc[:, 0], np.ones(len(rc))]
    return P @ np.asarray(A)


def zhang_suen(img):
    img = img.copy().astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4, p5 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:]
            p6, p7, p8, p9 = P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            Aen = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0) if step == 0 else (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            m = (img == 1) & (B >= 2) & (B <= 6) & (Aen == 1) & c
            if m.any():
                img[m] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def neighbours(sk, r, c):
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and 0 <= r + dr < sk.shape[0] and 0 <= c + dc < sk.shape[1] and sk[r + dr, c + dc]:
                out.append((r + dr, c + dc))
    return out


def prune(sk, L=15, passes=3):
    sk = sk.copy()
    K = np.ones((3, 3), int)
    for _ in range(passes):
        nb = ndimage.convolve(sk.astype(int), K, mode="constant") - sk.astype(int)
        for r, c in np.argwhere(sk & (nb == 1)):
            path = [(r, c)]
            cur, prev = (r, c), None
            hit_junction = False
            for _k in range(L):
                nbrs = [p for p in neighbours(sk, *cur) if p != prev and p not in path]
                if len(nbrs) != 1:
                    hit_junction = len(nbrs) > 1
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
                if len(neighbours(sk, *cur)) >= 3:
                    hit_junction = True
                    break
            if hit_junction and len(path) < L:
                for p in path[:-1]:
                    sk[p] = False
    return sk


def branch_filter(sk, dt, dt_min=3.0, L_min=20):
    """One pass: drop terminal branches (node to end point) whose mean half-width (distance transform of the black mask)
    is below dt_min or whose length is below L_min px; then drop components shorter than 30 px."""
    sk = sk.copy()
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk.astype(int)
    node = sk & (nb != 2)
    n_rm = 0
    for ln in trace_lines(sk):
        a, b = ln[0], ln[-1]
        if not (nb[a] == 1 or nb[b] == 1):
            continue
        d = np.array([dt[q] for q in ln])
        if nb[a] == 1:
            d = d[::-1]                       # junction side first
        core = d[: max(5, int(0.7 * len(d)))]  # mapped channel lines taper towards the head: judge the width on the junction-side 70 %
        if core.mean() < dt_min or len(ln) < L_min:
            for q in ln:
                if not (node[q] and nb[q] != 1):
                    sk[q] = False
            n_rm += 1
    lab, n = ndimage.label(sk, structure=np.ones((3, 3)))
    sizes = ndimage.sum(sk, lab, range(1, n + 1))
    sk &= np.isin(lab, 1 + np.nonzero(sizes >= 30)[0])
    return sk, n_rm


def trace_lines(sk):
    """Polylines (lists of (row, col)) between nodes (junctions and endpoints) of a thinned mask."""
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk.astype(int)
    node = sk & (nb != 2)
    visited = np.zeros(sk.shape, bool)
    lines = []
    for r, c in np.argwhere(node):
        for n0 in neighbours(sk, r, c):
            if visited[n0] and not node[n0]:
                continue
            path = [(r, c), n0]
            prev, cur = (r, c), n0
            while not node[cur]:
                visited[cur] = True
                nxt = [p for p in neighbours(sk, *cur) if p != prev and p not in path[-3:]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            key = (path[0], path[-1])
            if key[0] > key[1]:
                continue    # each node-to-node line once (from the lower node id); loops are kept once
            if len(path) >= 2:
                lines.append(path)
    return lines


def stage_gcp(out):
    ex = json.load(open(os.path.join(out, "extract.json")))
    loc = json.load(open(os.path.join(out, "locate.json")))["rec"]
    A0 = np.array(loc["patch_affine"])
    net = load_mask(out, "mask_network")
    sk = prune(zhang_suen(net))
    Image.fromarray((sk * 255).astype(np.uint8)).save(os.path.join(out, "mask_skeleton.png"))
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk.astype(int)
    junc = sk & (nb >= 3)
    jl, nj = ndimage.label(junc, structure=np.ones((3, 3)))
    jc = np.array(ndimage.center_of_mass(junc, jl, range(1, nj + 1)))
    wx0, wx1, wy0, wy1 = WIN
    a, _ = read_window("TTV_1m_area_cells.tif", WIN)
    a = a.astype(float)
    fd, _ = read_window("TTV_1m_fd.tif", WIN)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
    from lemcvt.hydro.d8 import d8_receiver_arrays
    r_to, c_to, good = d8_receiver_arrays(fd)
    cnt = np.zeros(a.shape, int)
    rr, cc = np.nonzero((a >= 2000) & good)
    np.add.at(cnt, (r_to[rr, cc], c_to[rr, cc]), 1)
    dj = (a >= 4000) & (cnt >= 2)
    djl, ndj = ndimage.label(dj, structure=np.ones((3, 3)))
    djc = np.array(ndimage.center_of_mass(dj, djl, range(1, ndj + 1)))
    dju = np.c_[wx0 + djc[:, 1] + 0.5, wy1 - djc[:, 0] - 0.5]
    fu = apply_affine(A0, jc)
    pairs = []
    for i, (fx, fy) in enumerate(fu):
        d = np.hypot(dju[:, 0] - fx, dju[:, 1] - fy)
        j = int(np.argmin(d))
        d2 = np.sort(d)[1] if len(d) > 1 else 1e9
        if d[j] < 20 and d2 > 40:
            pairs.append({"fig_rc": jc[i].tolist(), "fig_utm_patch_affine": [float(fx), float(fy)], "dem_utm": dju[j].tolist(), "dist_m": float(d[j]),
                          "dem_area_m2": float(a[int(djc[j, 0]), int(djc[j, 1])])})
    # the reviewed set: nearest pair to each reviewed figure junction (within 6 px), plus the pond
    gcps = []
    for k, (r, c) in enumerate(REVIEWED["confluences_fig_rc"]):
        d = [np.hypot(p["fig_rc"][0] - r, p["fig_rc"][1] - c) for p in pairs]
        j = int(np.argmin(d))
        if d[j] > 6:
            print("reviewed confluence", (r, c), f"has no pair within 6 px (nearest {d[j]:.1f} px)", flush=True)
            continue
        p = pairs[j]
        gcps.append({"id": f"J{k + 1}", "feature": "channel confluence (figure junction; D8 confluence on the 1 m grid)", "fig_rc": p["fig_rc"], "utm": p["dem_utm"],
                     "dem_area_m2": p["dem_area_m2"]})
    gcps.append({"id": "P", "feature": "man-made pond (solid black symbol; flat-floored basin with a dam on the 1 m hillshade)", "fig_rc": REVIEWED["pond"]["fig_rc"],
                 "utm": REVIEWED["pond"]["utm"]})
    pts = [{"fig_row_px": g["fig_rc"][0], "fig_col_px": g["fig_rc"][1], "utm_match": g["utm"]} for g in gcps]
    A, d = fit_affine(pts)
    rms = float(np.sqrt((d ** 2).mean()))
    for g, di in zip(gcps, d, strict=True):
        g["residual_m"] = float(di)
        g["fitted_utm"] = apply_affine(A, [g["fig_rc"]])[0].tolist()
    rule = {"rms_m": rms, "max_m": float(d.max()), "n": len(gcps), "pass": bool(rms <= 30 and d.max() <= 60), "criterion": "RMS <= 30 m and every point <= 60 m"}
    # spread: bounding box of the GCPs as a fraction of the map
    rc = np.array([g["fig_rc"] for g in gcps])
    rule["spread_rows_px"] = [float(rc[:, 0].min()), float(rc[:, 0].max())]
    rule["spread_cols_px"] = [float(rc[:, 1].min()), float(rc[:, 1].max())]
    rec = {"n_figure_junctions": int(nj), "n_dem_confluences": int(ndj), "n_unambiguous_pairs": len(pairs), "gcps": gcps, "affine_col_row_to_xy": A.tolist(),
           "pixel_size_m": [float(np.hypot(A[0, 0], A[0, 1])), float(np.hypot(A[1, 0], A[1, 1]))], "scale_bar_m_per_px": ex["m_per_px"],
           "rotation_deg": [float(np.degrees(np.arctan2(A[0, 1], A[0, 0]))), float(np.degrees(np.arctan2(-A[1, 0], -A[1, 1])))], "rule_2_1": rule,
           "second_order": None}
    if not rule["pass"]:
        rec["second_order"] = "affine failed rule 2.1; a second-order polynomial was not fitted (stop and report)"
    dump({"rec": rec, "pairs": pairs}, os.path.join(out, "gcp.json"))
    print("gcp", json.dumps({k: v for k, v in rec.items() if k != "gcps"}, indent=1), flush=True)
    for g in gcps:
        print(f"  {g['id']} fig({g['fig_rc'][0]:.0f},{g['fig_rc'][1]:.0f}) -> {g['utm'][0]:.1f} {g['utm'][1]:.1f}  residual {g['residual_m']:.1f} m", flush=True)
    return rec


def stage_digitize(out):
    json.load(open(os.path.join(out, "extract.json")))  # extract.json must exist
    g = json.load(open(os.path.join(out, "gcp.json")))["rec"]
    A = np.array(g["affine_col_row_to_xy"])
    black = load_mask(out, "mask_black")
    thick = load_mask(out, "mask_thick")
    net = load_mask(out, "mask_network")
    sk = load_mask(out, "mask_skeleton")
    from osgeo import ogr, osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(6339)
    drv = ogr.GetDriverByName("GPKG")
    path = os.path.join(out, "md1989_fig3.gpkg")
    if os.path.exists(path):
        drv.DeleteDataSource(path)
    ds = drv.CreateDataSource(path)
    rec = {}
    # letter candidates (A-D): solid components 150-500 px with a 18-45 px box, outside the arrow and scale bar
    labl, nl = ndimage.label(black, structure=np.ones((3, 3)))
    letter_cands = []
    for k, sl in enumerate(ndimage.find_objects(labl), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 18 <= h <= 45 and 14 <= w <= 45 and sl[0].start > NORTH_ARROW[1] and sl[0].stop < SCALE_BAR_ROWS[0] - 40:
            size = int((labl[sl] == k).sum())
            if 120 <= size <= 600:
                letter_cands.append([int(sl[0].start + h / 2), int(sl[1].start + w / 2), size, h, w])
    rec["letter_candidates_rc_size_h_w"] = letter_cands
    z1m, nod = read_window("TTV_1m_mean.tif", WIN)
    z1m = z1m.astype(float)
    z1m[z1m == nod] = np.nan
    wx0, wx1, wy0, wy1 = WIN

    def z_at(xy):
        c = int(xy[0] - wx0)
        r = int(wy1 - xy[1])
        return float(z1m[r, c]) if 0 <= r < z1m.shape[0] and 0 <= c < z1m.shape[1] else np.nan

    # ---- boundary: the hand-digitised dash-dot line (BOUNDARY_RC); each vertex snapped to the nearest pixel of a thick
    # non-network component (a dash or dot of the line) within 25 px, which removes the reading error of the hand digitising
    ring_rc = np.array(BOUNDARY_RC, float)
    lab_t, n_t = ndimage.label(thick, structure=np.ones((3, 3)))
    sizes_t = ndimage.sum(thick, lab_t, range(1, n_t + 1))
    dash_px = thick & np.isin(lab_t, 1 + np.nonzero((sizes_t >= 40) & (sizes_t <= 2500))[0])
    dash_px[SCALE_BAR_ROWS[0] - 40:, :] = False
    dash_px[NORTH_ARROW[0]:NORTH_ARROW[1], NORTH_ARROW[2]:NORTH_ARROW[3]] = False
    dist, (ir, ic) = ndimage.distance_transform_edt(~dash_px, return_distances=True, return_indices=True)
    n_snapped = 0
    for k, (r, c) in enumerate(ring_rc):
        r_, c_ = int(round(r)), int(round(c))
        if dist[r_, c_] <= 25:
            ring_rc[k] = [ir[r_, c_], ic[r_, c_]]
            n_snapped += 1
    ring_xy = apply_affine(A, ring_rc)
    poly = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ring_xy:
        poly.AddPoint(float(x), float(y))
    poly.CloseRings()
    pg = ogr.Geometry(ogr.wkbPolygon)
    pg.AddGeometry(poly)
    pg = pg.Buffer(0)
    lyr = ds.CreateLayer("study_area", srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("area_km2", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("source", ogr.OFTString))
    f = ogr.Feature(lyr.GetLayerDefn())
    f.SetGeometry(pg)
    f.SetField("area_km2", pg.GetArea() / 1e6)
    f.SetField("source", "M&D 1989 Fig. 3 dash-dot boundary, digitised by hand from gridded views of the scan (BOUNDARY_RC in a15_md_georef.py)")
    lyr.CreateFeature(f)
    rec["study_area"] = {"area_km2": pg.GetArea() / 1e6, "n_vertices": len(BOUNDARY_RC), "n_vertices_snapped_to_dashes": n_snapped, "source": "hand-digitised, snapped"}
    # the network is cut at the boundary corridor (+-12 px): dashes merged with exiting channels drop out, exits become end points
    from matplotlib.path import Path
    pth = Path(ring_rc[:, ::-1])
    yy, xx = np.mgrid[0:thick.shape[0], 0:thick.shape[1]]
    interior = pth.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(thick.shape)
    corridor = ndimage.binary_dilation(interior, iterations=8) & ~ndimage.binary_erosion(interior, iterations=8)
    net = net & ~corridor & ndimage.binary_dilation(interior, iterations=8)
    sk = prune(zhang_suen(net), L=12)
    dt_black = ndimage.distance_transform_edt(black)
    sk, n_branches_removed = branch_filter(sk, dt_black, 2.9, 15)
    sk = prune(sk, L=12)
    rec["skeleton_terminal_branches_removed"] = int(n_branches_removed)
    Image.fromarray((sk * 255).astype(np.uint8)).save(os.path.join(out, "mask_skeleton_clipped.png"))
    near_ring = ndimage.binary_dilation(corridor, iterations=12)
    # interior mask on the figure grid (for the colluvium clip and the raster mask): rasterise the ring in pixel space
    interior = np.zeros(thick.shape, bool)
    from matplotlib.path import Path
    pth = Path(ring_rc[:, ::-1])
    yy, xx = np.mgrid[0:thick.shape[0], 0:thick.shape[1]]
    interior = pth.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(thick.shape)
    # interior mask in UTM for later use (10 m NED reference): written as a 1 m raster over WIN
    yy, xx = np.mgrid[0:z1m.shape[0], 0:z1m.shape[1]]
    Ainv = np.linalg.inv(np.vstack([A.T, [0, 0, 1]]))
    X = wx0 + xx + 0.5
    Y = wy1 - yy - 0.5
    col = Ainv[0, 0] * X + Ainv[0, 1] * Y + Ainv[0, 2]
    row = Ainv[1, 0] * X + Ainv[1, 1] * Y + Ainv[1, 2]
    inside = ndimage.map_coordinates(interior.astype(np.uint8), [row, col], order=0, mode="constant", cval=0).astype(bool)
    from a14_common import write_tif
    from rasterio.transform import from_origin
    write_tif(os.path.join(out, "md_study_area_1m.tif"), inside.astype(np.uint8), from_origin(wx0, wy1, 1.0, 1.0), rasterio.crs.CRS.from_epsg(6339))
    zin = z1m[inside]
    rec["study_area"].update({"raster_area_km2": float(inside.sum() / 1e6), "z_min_1m": float(np.nanmin(zin)), "z_max_1m": float(np.nanmax(zin)),
                              "z_p01_1m": float(np.nanpercentile(zin, 1)), "z_p99_1m": float(np.nanpercentile(zin, 99))})
    # ---- channels
    lines = trace_lines(sk)
    lyr = ds.CreateLayer("channels", srs, ogr.wkbLineString)
    lyr.CreateField(ogr.FieldDefn("length_m", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("component", ogr.OFTInteger))
    comp_lab, ncomp = ndimage.label(sk, structure=np.ones((3, 3)))
    total = 0.0
    for ln in lines:
        xy = apply_affine(A, np.array(ln, float))
        geom = ogr.Geometry(ogr.wkbLineString)
        for x, y in xy[::2] if len(xy) > 4 else xy:
            geom.AddPoint(float(x), float(y))
        if geom.GetPointCount() < 2:
            continue
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(geom)
        f.SetField("length_m", geom.Length())
        f.SetField("component", int(comp_lab[ln[0]]))
        lyr.CreateFeature(f)
        total += geom.Length()
    rec["channels"] = {"n_lines": len(lines), "total_length_km": total / 1e3, "n_components": int(ncomp), "paper_total_km": 9.1}
    # ---- channel heads: endpoints of the thinned network; per component the lowest endpoint (1 m grid) is the outlet, the rest are heads
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk.astype(int)
    ends = np.argwhere(sk & (nb == 1))
    end_xy = apply_affine(A, ends.astype(float))
    end_z = np.array([z_at(p) for p in end_xy])
    end_comp = comp_lab[ends[:, 0], ends[:, 1]]
    is_head = np.ones(len(ends), bool)
    ring_close = ndimage.binary_dilation(interior & ~ndimage.binary_erosion(interior, iterations=1), iterations=10)   # within 10 px of the boundary line
    is_head[ring_close[ends[:, 0], ends[:, 1]]] = False
    is_exit = np.zeros(len(ends), bool)
    for cidx in np.unique(end_comp):
        sel = np.nonzero(end_comp == cidx)[0]
        if len(sel) >= 2:
            o = sel[np.nanargmin(end_z[sel])]
            is_head[o] = False      # the component's outlet (lowest end point on the 1 m grid)
            is_exit[o] = bool(near_ring[ends[o, 0], ends[o, 1]])
    # continuation rule: a component whose outlet lies within 50 m of (and above) another component's channel pixels is a discontinuous
    # reach of the same channel; its upper endpoint(s) are headcuts below the mapped head, not heads (the paper counts one head per hollow)
    sk_xy = apply_affine(A, np.argwhere(sk).astype(float))
    sk_comp = comp_lab[sk]
    cont = np.zeros(len(ends), bool)
    for cidx in np.unique(end_comp):
        sel = np.nonzero(end_comp == cidx)[0]
        if len(sel) < 2:
            continue
        o = sel[np.nanargmin(end_z[sel])]
        d = np.hypot(sk_xy[:, 0] - end_xy[o, 0], sk_xy[:, 1] - end_xy[o, 1])
        other = (sk_comp != cidx) & (d < 50)
        if other.any():
            zo = np.array([z_at(p) for p in sk_xy[other][:: max(1, other.sum() // 50)]])
            if np.nanmedian(zo) <= end_z[o] + 5:
                cont[sel] = True
    heads = [i for i in range(len(ends)) if is_head[i]]
    # review hooks: candidates table and numbered quadrant images; HEAD_REVIEW drops false candidates and adds missed heads
    cand_rows = [{"cand_id": k, "fig_row": int(ends[i, 0]), "fig_col": int(ends[i, 1]), "x": float(end_xy[i, 0]), "y": float(end_xy[i, 1]), "z_1m": float(end_z[i]), "continuation": bool(cont[i])} for k, i in enumerate(heads)]
    with open(os.path.join(out, "head_candidates.csv"), "w") as fh:
        fh.write("cand_id,fig_row,fig_col,x,y,z_1m,continuation\n")
        for c in cand_rows:
            fh.write(f"{c['cand_id']},{c['fig_row']},{c['fig_col']},{c['x']:.1f},{c['y']:.1f},{c['z_1m']:.1f},{int(c['continuation'])}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, (r0, c0) in {"NW": (0, 100), "NE": (0, 900), "SW": (800, 100), "SE": (800, 900)}.items():
            fig = plt.figure(figsize=(19, 19), dpi=100)
            ax = fig.add_axes([0.03, 0.03, 0.95, 0.95])
            rgb = np.ones(black.shape + (3,), float)
            rgb[black] = [0.6, 0.6, 0.6]
            rgb[sk] = [0.0, 0.0, 1.0]
            ax.imshow(rgb[r0:r0 + 900, c0:c0 + 900], interpolation="nearest", extent=[c0, c0 + 900, r0 + 900, r0])
            for c in cand_rows:
                if r0 <= c["fig_row"] < r0 + 900 and c0 <= c["fig_col"] < c0 + 900:
                    ax.plot(c["fig_col"], c["fig_row"], "o", mfc="none", mec="lime", ms=11, mew=1.5)
                    ax.text(c["fig_col"] + 6, c["fig_row"] - 6, str(c["cand_id"]), color="green", fontsize=9, fontweight="bold")
            ax.set_xticks(range(c0, c0 + 901, 50))
            ax.set_yticks(range(r0, r0 + 901, 50))
            ax.grid(color="cyan", lw=0.5)
            ax.tick_params(labelsize=7, labelrotation=90)
            fig.savefig(os.path.join(out, f"head_candidates_{name}.png"))
            plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("candidate render failed", repr(e), flush=True)
    drop = set(HEAD_REVIEW.get("drop", []))
    heads = [i for k, i in enumerate(heads) if k not in drop]
    for (ar, ac) in HEAD_REVIEW.get("add", []):
        ends = np.vstack([ends, [[ar, ac]]])
        xy = apply_affine(A, [[ar, ac]])[0]
        end_xy = np.vstack([end_xy, xy])
        end_z = np.append(end_z, z_at(xy))
        end_comp = np.append(end_comp, -1)
        cont = np.append(cont, False)
        is_head = np.append(is_head, True)
        heads.append(len(ends) - 1)
    rec_review = {"n_candidates": len(cand_rows), "n_dropped": len(drop), "n_added": len(HEAD_REVIEW.get("add", []))}
    lyr = ds.CreateLayer("channel_heads", srs, ogr.wkbPoint)
    for name, typ in (("head_id", ogr.OFTInteger), ("label", ogr.OFTString), ("z_1m", ogr.OFTReal), ("component", ogr.OFTInteger), ("continuation", ogr.OFTInteger),
                      ("fig_row", ogr.OFTReal), ("fig_col", ogr.OFTReal)):
        lyr.CreateField(ogr.FieldDefn(name, typ))
    # labels A-D: nearest head within 70 px of the letter; the culvert: nearest head within 40 px of the triangle
    labels = {}
    for L, (lr, lc) in LETTERS.items():
        d = np.hypot(ends[heads, 0] - lr, ends[heads, 1] - lc)
        if d.min() < 90:
            labels[heads[int(np.argmin(d))]] = L
        rec.setdefault("letter_nearest_head_px", {})[L] = float(d.min())
    d = np.hypot(ends[heads, 0] - CULVERT_TRIANGLE[0], ends[heads, 1] - CULVERT_TRIANGLE[1])
    culvert_xy = apply_affine(A, [CULVERT_TRIANGLE])[0]
    if d.min() < 40:
        labels[heads[int(np.argmin(d))]] = "culvert"
    k = 0
    for i in heads:
        k += 1
        f = ogr.Feature(lyr.GetLayerDefn())
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint(float(end_xy[i, 0]), float(end_xy[i, 1]))
        f.SetGeometry(pt)
        f.SetField("head_id", k)
        f.SetField("label", labels.get(i, ""))
        f.SetField("z_1m", float(end_z[i]) if np.isfinite(end_z[i]) else None)
        f.SetField("component", int(end_comp[i]))
        f.SetField("continuation", int(cont[i]))
        f.SetField("fig_row", float(ends[i, 0]))
        f.SetField("fig_col", float(ends[i, 1]))
        lyr.CreateFeature(f)
    rec["heads"] = {"review": rec_review, "n_endpoints": int(len(ends)), "n_exits_at_boundary": int(is_exit.sum()), "n_heads_all_upstream_endpoints": int(len(heads)), "n_heads_excluding_continuations": int(sum(1 for i in heads if not cont[i])),
                    "n_components": int(len(np.unique(end_comp))), "labels": {v: [float(end_xy[i, 0]), float(end_xy[i, 1])] for i, v in labels.items()},
                    "culvert_triangle_utm": culvert_xy.tolist(), "paper": "68 channelled hollows of 90; 5 excluded for trails; 1 culvert head"}
    # ---- colluvial deposits: stipple dots (black components <= 12 px) density
    lab, n = ndimage.label(black, structure=np.ones((3, 3)))
    sizes = ndimage.sum(black, lab, range(1, n + 1))
    dots = np.isin(lab, 1 + np.nonzero((sizes >= 2) & (sizes <= 30))[0])
    dots[SCALE_BAR_ROWS[0] - 40:, :] = False
    dens = ndimage.uniform_filter(dots.astype(float), 21)
    coll = (dens > 0.03) & ndimage.binary_dilation(interior, iterations=10)
    coll = ndimage.binary_closing(coll, structure=np.ones((5, 5)), iterations=2)
    coll = ndimage.binary_opening(coll, structure=np.ones((5, 5)))
    lab, n = ndimage.label(coll, structure=np.ones((3, 3)))
    sizes = ndimage.sum(coll, lab, range(1, n + 1))
    coll = np.isin(lab, 1 + np.nonzero(sizes >= 150)[0])
    Image.fromarray((coll * 255).astype(np.uint8)).save(os.path.join(out, "mask_colluvium.png"))
    coll_utm = ndimage.map_coordinates(coll.astype(np.uint8), [row, col], order=0, mode="constant", cval=0).astype(bool)
    from rasterio import features
    lyr = ds.CreateLayer("colluvial_deposits", srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("area_m2", ogr.OFTReal))
    nc, tot = 0, 0.0
    for geom, _val in features.shapes(coll_utm.astype(np.uint8), mask=coll_utm, transform=from_origin(wx0, wy1, 1.0, 1.0)):
        og = ogr.CreateGeometryFromJson(json.dumps(geom)).Simplify(1.5)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(og)
        f.SetField("area_m2", og.GetArea())
        lyr.CreateFeature(f)
        nc += 1
        tot += og.GetArea()
    rec["colluvium"] = {"n_polygons": nc, "total_area_km2": tot / 1e6, "dot_density_threshold": 0.03, "dot_px_max": 30, "density_window_px": 21}
    # GCPs as a layer
    lyr = ds.CreateLayer("ground_control", srs, ogr.wkbPoint)
    for name, typ in (("id", ogr.OFTString), ("feature", ogr.OFTString), ("fig_row", ogr.OFTReal), ("fig_col", ogr.OFTReal), ("residual_m", ogr.OFTReal)):
        lyr.CreateField(ogr.FieldDefn(name, typ))
    for gc in g["gcps"]:
        f = ogr.Feature(lyr.GetLayerDefn())
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint(float(gc["utm"][0]), float(gc["utm"][1]))
        f.SetGeometry(pt)
        f.SetField("id", gc["id"])
        f.SetField("feature", gc["feature"])
        f.SetField("fig_row", float(gc["fig_rc"][0]))
        f.SetField("fig_col", float(gc["fig_rc"][1]))
        f.SetField("residual_m", float(gc["residual_m"]))
        lyr.CreateFeature(f)
    ds = None
    rec["gpkg"] = path
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(22, 21), dpi=100)
        ax = fig.add_axes([0.03, 0.03, 0.95, 0.95])
        rgb = np.ones(black.shape + (3,), float)
        rgb[black] = [0.6, 0.6, 0.6]
        rgb[coll] = [1.0, 0.75, 0.9]
        rgb[black & coll] = [0.5, 0.3, 0.4]
        rgb[sk] = [0.0, 0.0, 1.0]
        ax.imshow(rgb, interpolation="nearest")
        ax.plot(np.r_[ring_rc[:, 1], ring_rc[0, 1]], np.r_[ring_rc[:, 0], ring_rc[0, 0]], "-", color="orange", lw=2)
        hh = np.array([ends[i] for i in heads])
        ax.plot(hh[:, 1], hh[:, 0], "o", mfc="none", mec="lime", ms=9, mew=1.5)
        cc_ = np.array([ends[i] for i in heads if cont[i]])
        if len(cc_):
            ax.plot(cc_[:, 1], cc_[:, 0], "x", color="cyan", ms=9, mew=2)
        ex_ = ends[is_exit]
        ax.plot(ex_[:, 1], ex_[:, 0], "s", mfc="none", mec="red", ms=8)
        for i, L in labels.items():
            ax.text(ends[i][1] + 10, ends[i][0] - 10, L, color="green", fontsize=14, fontweight="bold")
        for gc in g["gcps"]:
            ax.plot(gc["fig_rc"][1], gc["fig_rc"][0], "s", mfc="yellow", mec="k", ms=7)
        ax.set_xlim(0, black.shape[1])
        ax.set_ylim(black.shape[0], 0)
        ax.set_title("figure space: skeleton (blue), heads (green o; cyan x = continuation), exits (red), boundary (orange), colluvium (pink), GCPs (yellow)")
        fig.savefig(os.path.join(out, "digitized_on_figure.png"))
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("figure-space render failed", repr(e), flush=True)
    dump(rec, os.path.join(out, "digitize.json"))
    print("digitize", json.dumps(rec, indent=1), flush=True)
    return rec


def walk_ring(outline):
    pts = np.argwhere(outline)
    S = set(map(tuple, pts))
    start = min(S)
    order = [start]
    prev, cur = None, start
    while True:
        nbrs = [(cur[0] + dr, cur[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (cur[0] + dr, cur[1] + dc) in S]
        nbrs = [p for p in nbrs if p != prev and p not in order[-3:]]
        if not nbrs:
            break
        # prefer 4-neighbours
        nbrs.sort(key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
        prev, cur = cur, nbrs[0]
        if cur == start or len(order) > len(S):
            break
        order.append(cur)
    return order


def stage_check(out):
    d = json.load(open(os.path.join(out, "digitize.json")))
    sa = d["study_area"]
    area_ok = 2.1 * 0.85 <= sa["area_km2"] <= 2.1 * 1.15
    zmin_ok = 61 - 15 <= sa["z_min_1m"] <= 61 + 15
    zmax_ok = 314 - 15 <= sa["z_max_1m"] <= 314 + 15
    rec = {"area_km2": sa["area_km2"], "area_window": [2.1 * 0.85, 2.1 * 1.15], "area_pass": area_ok, "z_min_1m": sa["z_min_1m"], "z_max_1m": sa["z_max_1m"],
           "z_min_window": [46, 76], "z_max_window": [299, 329], "z_min_pass": zmin_ok, "z_max_pass": zmax_ok, "pass": bool(area_ok and zmin_ok and zmax_ok)}
    dump(rec, os.path.join(out, "rule_2_2.json"))
    print("rule 2.2", json.dumps(rec, indent=1), flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--stage", default="all", choices=["extract", "locate", "gcp", "digitize", "check", "all"])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    if a.stage in ("extract", "all"):
        stage_extract(a.out)
    if a.stage in ("locate", "all"):
        stage_locate(a.out)
    if a.stage in ("gcp", "all"):
        stage_gcp(a.out)
    if a.stage in ("digitize", "all"):
        stage_digitize(a.out)
    if a.stage in ("check", "all"):
        stage_check(a.out)
    print("done in", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
