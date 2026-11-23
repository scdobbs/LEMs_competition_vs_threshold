#!/usr/bin/env python3
"""Addendum 21, a diagnostic of the Perron (2012) valley-head sensitivity on one tile's observed surface: cluster, network, skeleton and
end-point counts at each stage, the loop count of the skeleton, and a figure. usage: a21_perron_diag.py --tile TTV2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import (  # noqa: E402
    FIG_DIR,
    PERRON_CLOSE_R_M,
    PERRON_K_SD,
    PERRON_MIN_CELLS,
    PERRON_PRUNE_M,
    _neighbours8,
    contour_curvature,
    dump,
    hillshade,
    json_ready,
    mask_from_entries,
    metre_tile,
    out_path,
    picks_on_tile,
    prune_spurs,
    thin_zhang_suen,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile", default="TTV2")
    a = p.parse_args()
    from scipy.ndimage import binary_closing, label
    t0 = time.time()
    td = metre_tile(a.tile)
    Z, A, FD, dx = td.Z, td.A, td.FD, td.dx
    kc = contour_curvature(Z, dx)
    v = kc[np.isfinite(kc)]
    thr = float(v.mean() + PERRON_K_SD * v.std())
    valley = np.isfinite(kc) & (kc > thr)
    lab, n = label(valley, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= PERRON_MIN_CELLS
    keep[0] = False
    clusters = keep[lab]
    ids = np.nonzero(keep)[0]
    Af = np.where(np.isfinite(A), A, -1.0)
    rows, cols = [], []
    for k in ids:
        rr, cc = np.nonzero(lab == k)
        j = int(np.argmax(Af[rr, cc]))
        rows.append(rr[j])
        cols.append(cc[j])
    paths = mask_from_entries(np.array(rows), np.array(cols), A, FD)
    net = clusters | paths
    r = int(round(PERRON_CLOSE_R_M / dx))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r * r
    closed = binary_closing(net, structure=disk, border_value=0) | net
    sk0 = thin_zhang_suen(closed)
    nb0 = _neighbours8(sk0)
    sk = prune_spurs(sk0, int(round(PERRON_PRUNE_M / dx)))
    nb = _neighbours8(sk)
    _, n_comp = label(sk, structure=np.ones((3, 3)))
    _, n_holes = label(~sk & ~np.pad(np.zeros(sk.shape, dtype=bool), 0), structure=np.ones((3, 3)))     # background components (holes + outside)
    rec = {"tile": a.tile, "kc_threshold": thr, "kc_mean": float(v.mean()), "kc_sd": float(v.std()), "n_valley_cells": int(valley.sum()), "n_clusters_all": int(n), "n_clusters_kept": int(keep.sum()),
           "n_cluster_cells": int(clusters.sum()), "cluster_size_pct": {q: float(np.percentile(sizes[keep], q)) for q in (5, 50, 95)} if keep.any() else None, "n_path_cells": int(paths.sum()),
           "n_net_cells": int(net.sum()), "n_closed_cells": int(closed.sum()), "n_skeleton_before_prune": int(sk0.sum()), "n_endpoints_before_prune": int((sk0 & (nb0 == 1)).sum()),
           "n_junctions_before_prune": int((sk0 & (nb0 >= 3)).sum()), "n_skeleton_after_prune": int(sk.sum()), "n_endpoints_after_prune": int((sk & (nb == 1)).sum()),
           "n_junctions_after_prune": int((sk & (nb >= 3)).sum()), "n_components": int(n_comp), "n_background_components": int(n_holes), "wall_s": time.time() - t0}
    edge = np.zeros_like(sk)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    rec["n_heads_off_edge"] = int((sk & (nb == 1) & ~edge).sum())
    rec["n_cluster_tips_isolated_components"] = int(n_comp)
    print(rec, flush=True)
    dump(json_ready(rec), out_path("diag", f"perron_{a.tile}.json"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr, pc = picks_on_tile(td)[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    hs = np.nan_to_num(hillshade(Z, dx), nan=1.0)
    for ax, (m, title) in zip(axes, ((clusters, f"clusters (kc > {thr:.4f}), {int(keep.sum())}"), (closed, "clusters + paths, closed"), (sk, "skeleton, pruned")), strict=True):
        ax.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(np.where(m, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#0072B2"]), alpha=0.8, interpolation="nearest")
        if pr.size:
            ax.plot(pc, pr, "^", color="#D55E00", ms=5, mec="k", mew=0.4)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    hr, hc = np.nonzero(sk & (nb == 1) & ~edge)
    axes[2].plot(hc, hr, "o", mfc="none", mec="#E69F00", ms=6)
    os.makedirs(os.path.join(FIG_DIR, "diag"), exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "diag", f"perron_{a.tile}.png"), dpi=110, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
