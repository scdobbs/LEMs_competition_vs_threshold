#!/usr/bin/env python3
"""Addendum 20, decision 6d and rule R8: model channel heads against the snapped heads (the A_P10 cell of each pick) on TTV1-TTV3.

Models: P4 (addendum 18), PF at A_F (median A_P10), 7,400, 2,848 and 611 m2, SF at alpha = 0, 1.65, 2 (C_F), SJ at each alpha run (the
a20 slope-threshold runs). Per model and tile: the channel cells (A >= A_crit, or M >= C with M the upstream maximum of A S^alpha), the heads
(channel cells with no channel donor), the distance from each snapped head on the tile to the nearest model channel cell, the count of model
heads and of snapped heads inside the M&D polygon. R8 on TTV1-TTV3 together: median distance <= 25 m and head ratio within 0.67-1.5;
otherwise "too few channels" (ratio < 0.67) or "too many". Figures as addendum 19 decision 6.
usage: a20_modelheads.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_hilltops import md_polygon_mask  # noqa: E402
from a20_common import (  # noqa: E402
    A18_SUMMARY,
    FIG_DIR,
    MD_TILES,
    R8_MEDIAN_DIST_M,
    R8_RATIO_RANGE,
    channel_heads,
    dump,
    hillshade,
    load_json,
    metre_tile,
    out_path,
    slurm_record,
)

from lemcvt.analyses.slope_threshold import head_statistic  # noqa: E402
from lemcvt.results import find_runs  # noqa: E402

PF_LABELS = {"A_F": "PF_AF", "A_F_table1": "PF_7400", "A_F_A10": "PF_2848", "A_F_A3": "PF_611"}


def sj_optima(tile: str) -> dict[str, dict]:
    out = {}
    for r in find_runs("slope_threshold"):
        if "_a20_sj_" not in r.name or f"_{tile}_a" not in r.name:
            continue
        try:
            p = json.loads((r / "params.json").read_text())
        except (OSError, ValueError):
            continue
        if p.get("tile") != tile or "result" not in p:
            continue
        o = p["result"]["joint_optimum"]
        out[f"{float(p['alpha']):g}"] = {"run": r.name, "C": o["C"], "K": o["K"], "D": o["D"], "mse": o["mse"], "where": o["where"], "n_channel": o["n_channel"]}
    return out


def reading(med, ratio):
    if med is not None and med <= R8_MEDIAN_DIST_M and ratio is not None and R8_RATIO_RANGE[0] <= ratio <= R8_RATIO_RANGE[1]:
        return True, "agree with the mapped heads"
    if ratio is not None and ratio < R8_RATIO_RANGE[0]:
        return False, "too few channels" + (" (distances large)" if med is not None and med > R8_MEDIAN_DIST_M else "")
    if ratio is not None and ratio > R8_RATIO_RANGE[1]:
        return False, "too many channels"
    return False, "count in range, distances large" if (med is not None and med > R8_MEDIAN_DIST_M) else "undetermined"


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("modelheads", "modelheads.json")))
    fig_dir = os.path.join(FIG_DIR, "model_heads")
    os.makedirs(fig_dir, exist_ok=True)
    heads = load_json(out_path("heads", "heads.json"))
    fits18 = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    snap = np.load(out_path("heads", "snapped_network.npz"))
    full_tr = snap["transform"]
    hx = full_tr[2] + (snap["head_cols"].astype(float) + 0.5) * full_tr[0]
    hy = full_tr[5] + (snap["head_rows"].astype(float) + 0.5) * full_tr[4]
    out: dict = {"tiles": {}, "rule_R8": {}, "figures": {}, "heads_used": "A_P10 cell of each pick (decision 5)"}
    pooled: dict[str, dict] = {}
    for tile in MD_TILES:
        td = metre_tile(tile)
        Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
        tr = td.filled.transform
        md = md_polygon_mask(td)
        ny, nx = Z.shape
        cc = np.floor((hx - tr.c) / tr.a).astype(int)
        rr = np.floor((hy - tr.f) / tr.e).astype(int)
        on = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
        pr, pc = rr[on], cc[on]
        ids = [q["pick_id"] for q, o in zip(heads["picks"], on, strict=True) if o]
        in_poly = md[pr, pc]
        n_picks_poly = int(in_poly.sum())
        fixed_p = out_path("fixed", f"{tile}.json")
        fixed = load_json(fixed_p) if os.path.exists(fixed_p) else None
        models: dict[str, tuple] = {}
        p4 = fits18[f"{tile}:primary"]["P4"]
        models["P4"] = ("area", p4["A_crit"], {"K": p4["K"], "D": p4["D"], "mse": p4["mse"]})
        if fixed:
            for key, nm in PF_LABELS.items():
                f = fixed["PF"][key]
                models[nm] = ("area", f["A_crit"], {"K": f["free_KD"]["K"], "D": f["free_KD"]["D"], "mse": f["free_KD"]["mse"]})
            for al in fixed["field_values"]["alphas_SF"]:
                f = fixed["SF"][f"alpha_{al:g}:C_F"]
                models[f"SF_a{al:g}"] = ("slope", f["C"], {"alpha": al, "K": f["free_KD"]["K"], "D": f["free_KD"]["D"], "mse": f["free_KD"]["mse"]})
        for al, s in sj_optima(tile).items():
            models[f"SJ_a{al}"] = ("slope", s["C"], {"alpha": float(al), "K": s["K"], "D": s["D"], "mse": s["mse"], "run": s["run"]})
        stats, chans, Mc = {}, {}, {}
        for name, (kind, thr, info) in models.items():
            if kind == "area" or info.get("alpha") == 0:
                ch = np.isfinite(A) & (A >= thr)
            else:
                al = float(info["alpha"])
                if al not in Mc:
                    Mc[al] = head_statistic(Z, A, FD, dx, dy, al)["M"]
                ch = np.isfinite(Mc[al]) & (Mc[al] >= thr)
            hd = channel_heads(ch, FD)
            dist = distance_transform_edt(~ch, sampling=(dy, dx)) if ch.any() else np.full(Z.shape, np.inf)
            d = dist[pr, pc]
            nh = int((hd & md).sum())
            stats[name] = {"kind": kind, "threshold": float(thr), "n_channel": int(ch.sum()), "n_heads_tile": int(hd.sum()), "n_heads_in_polygon": nh, "n_snapped_on_tile": int(pr.size),
                           "n_snapped_in_polygon": n_picks_poly, "ratio_heads_to_picks": nh / n_picks_poly if n_picks_poly else None,
                           "dist_m": {"median": float(np.median(d)) if d.size else None, "q25": float(np.percentile(d, 25)) if d.size else None, "q75": float(np.percentile(d, 75)) if d.size else None,
                                      "frac_within_25m": float(np.mean(d <= R8_MEDIAN_DIST_M)) if d.size else None, "values": dict(zip(map(str, ids), d.tolist(), strict=True))}} | {k: v for k, v in info.items()}
            chans[name] = (ch, hd)
            pl = pooled.setdefault(name, {"d": [], "heads": 0, "picks": 0, "tiles": []})
            pl["d"] += d.tolist()
            pl["heads"] += nh
            pl["picks"] += n_picks_poly
            pl["tiles"].append(tile)
        out["tiles"][tile] = {"n_snapped_on_tile": int(pr.size), "n_snapped_in_polygon": n_picks_poly, "polygon_px": int(md.sum()), "models": stats}
        names = list(models)
        ncols = 4
        nrows = int(np.ceil(len(names) / ncols))
        hs = hillshade(Z, dx)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.8 * nrows))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, name in zip(axes.ravel(), names, strict=False):
            ax.axis("on")
            ch, hd = chans[name]
            ax.imshow(np.nan_to_num(hs, nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.imshow(np.where(ch, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#0072B2"]), alpha=0.8, interpolation="nearest")
            ax.contour(md.astype(float), levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
            h_r, h_c = np.nonzero(hd)
            ax.plot(h_c, h_r, ".", color="#E69F00", ms=2.5)
            ax.plot(pc, pr, "^", color="#D55E00", ms=5, mec="k", mew=0.4)
            s = stats[name]
            thr_txt = f"A_crit {s['threshold']:,.0f} m²" if s["kind"] == "area" else f"α {s['alpha']:g}, C {s['threshold']:.4g}"
            ax.set_title(f"{name}: {thr_txt}\nmedian dist {s['dist_m']['median'] if s['dist_m']['median'] is None else round(s['dist_m']['median'], 1)} m; heads/picks "
                         f"{s['ratio_heads_to_picks'] if s['ratio_heads_to_picks'] is None else round(s['ratio_heads_to_picks'], 2)}; D {s['D']:.3g}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{tile}: model channels (blue), model heads (orange), snapped heads (triangles, A_P10 cells); dashed = M&D study area", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(fig_dir, f"{tile}_model_heads.png")
        for sc in (1.0, 0.8, 0.6, 0.45):
            fig.savefig(path, dpi=int(170 * sc), bbox_inches="tight", facecolor="white")
            if os.path.getsize(path) <= 5_000_000:
                break
        plt.close(fig)
        out["figures"][tile] = {"path": os.path.relpath(path, os.path.dirname(os.path.dirname(FIG_DIR))), "bytes": os.path.getsize(path)}
        print(tile, {k: (v["dist_m"]["median"] and round(v["dist_m"]["median"], 1), v["n_heads_in_polygon"], n_picks_poly) for k, v in stats.items()}, f"[{time.time() - t0:.0f} s]", flush=True)
    for name, p in pooled.items():
        d = np.array(p["d"])
        med = float(np.median(d)) if d.size else None
        ratio = p["heads"] / p["picks"] if p["picks"] else None
        agree, rd = reading(med, ratio)
        out["rule_R8"][name] = {"median_dist_m": med, "n_snapped": int(d.size), "n_heads_in_polygon": p["heads"], "n_snapped_in_polygon": p["picks"], "ratio": ratio, "agree": agree, "reading": rd,
                                "tiles": p["tiles"], "complete": len(p["tiles"]) == len(MD_TILES)}
    out["wall_s"] = time.time() - t0
    dump(out, out_path("modelheads", "modelheads.json"))
    print("R8:", {k: (v["median_dist_m"], v["ratio"], v["reading"]) for k, v in out["rule_R8"].items()}, flush=True)


if __name__ == "__main__":
    main()
