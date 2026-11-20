#!/usr/bin/env python3
"""Addendum 19, decision 6 and rule R8: model channel heads against the picked heads on TTV1-TTV3 (the tiles inside the M&D map).

For P4 (addendum 18), PF (A_F, 7,400 m2, and the A_10 sensitivity), SF and SJ at each alpha: the model's channel cells (A >= A_crit, or
M >= C with M the upstream maximum of A S^alpha), its heads (channel cells with no channel donor), the distance from each pick on the tile to
the nearest model channel cell, and the count of model heads against the count of picks inside the M&D polygon on the tile. R8 on TTV1-TTV3
together: median pick-to-channel distance <= 25 m and head/pick ratio within 0.67-1.5. Figures: hillshade, channel cells, heads, picks.
usage: a19_modelheads.py
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_common import (  # noqa: E402
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


def sj_optima(tile: str) -> dict[str, dict]:
    """The latest a19 slope-threshold run per alpha for the tile: C, K, D, mse."""
    import json
    out = {}
    for r in find_runs("slope_threshold"):
        if "a19_sj" not in r.name or f"_{tile}_a" not in r.name:
            continue
        try:
            p = json.loads((r / "params.json").read_text())
        except (OSError, ValueError):
            continue
        if p.get("tile") != tile or "result" not in p or "pilot" in r.name:
            continue
        al = f"{float(p['alpha']):g}"
        out[al] = {"run": r.name, "C": p["result"]["joint_optimum"]["C"], "K": p["result"]["joint_optimum"]["K"], "D": p["result"]["joint_optimum"]["D"], "mse": p["result"]["joint_optimum"]["mse"],
                   "n_channel": p["result"]["joint_optimum"]["n_channel"]}
    return out


def main() -> None:
    t0 = time.time()
    out_dir = os.path.dirname(out_path("modelheads", "modelheads.json"))
    slurm_record(out_dir)
    fig_dir = os.path.join(FIG_DIR, "heads")
    os.makedirs(fig_dir, exist_ok=True)
    heads = load_json(out_path("heads", "heads.json"))
    fits18 = load_json(os.path.join(A18_SUMMARY, "per_tile_fits.json"))
    fv = heads["field_values"]
    out: dict = {"tiles": {}, "models": {}, "rule_R8": {}, "figures": {}}
    pooled: dict[str, dict] = {}
    for tile in MD_TILES:
        td = metre_tile(tile)
        Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
        tr = td.filled.transform
        z = np.load(out_path("hilltops", f"{tile}_masks.npz"))
        md = z["md_polygon"]
        pk = [q for q in heads["picks"] if q["tile"] == tile]
        pr = np.array([int(np.floor((q["y"] - tr.f) / tr.e)) for q in pk])
        pc = np.array([int(np.floor((q["x"] - tr.c) / tr.a)) for q in pk])
        n_picks_in_poly = int(sum(1 for q, r, c in zip(pk, pr, pc, strict=True) if md[r, c]))
        fixed = load_json(out_path("fixed", f"{tile}.json")) if os.path.exists(out_path("fixed", f"{tile}.json")) else None
        models: dict[str, tuple[str, float, dict]] = {}
        p4 = fits18[f"{tile}:primary"]["P4"]
        models["P4"] = ("area", p4["A_crit"], {"K": p4["K"], "D": p4["D"], "mse": p4["mse"]})
        if fixed:
            for name, key in (("PF_A_F", "A_F"), ("PF_7400", "A_F_table1"), ("PF_A10", "A_F_A10_sensitivity")):
                f = fixed["PF"][key]
                models[name] = ("area", f["A_crit"], {"K": f["free_KD"]["K"], "D": f["free_KD"]["D"], "mse": f["free_KD"]["mse"]})
            for al in fv["alphas"]:
                f = fixed["SF"][f"alpha_{al:g}:C_F"]
                models[f"SF_a{al:g}"] = ("slope", f["C"], {"alpha": al, "K": f["free_KD"]["K"], "D": f["free_KD"]["D"], "mse": f["free_KD"]["mse"]})
        for al, s in sj_optima(tile).items():
            models[f"SJ_a{al}"] = ("slope", s["C"], {"alpha": float(al), "K": s["K"], "D": s["D"], "mse": s["mse"], "run": s["run"]})
        stats = {}
        chans = {}
        M_cache: dict[float, np.ndarray] = {}
        for name, (kind, thr, info) in models.items():
            if kind == "area":
                ch = np.isfinite(A) & (A >= thr)
            else:
                al = float(info["alpha"])
                if al not in M_cache:
                    M_cache[al] = head_statistic(Z, A, FD, dx, dy, al)["M"]
                ch = np.isfinite(M_cache[al]) & (M_cache[al] >= thr)
            hd = channel_heads(ch, FD)
            dist = distance_transform_edt(~ch, sampling=(dy, dx)) if ch.any() else np.full(Z.shape, np.inf)
            d_picks = dist[pr, pc] if pr.size else np.array([])
            n_heads_poly = int((hd & md).sum())
            stats[name] = {"kind": kind, "threshold": float(thr), "n_channel": int(ch.sum()), "n_heads_tile": int(hd.sum()), "n_heads_in_polygon": n_heads_poly, "n_picks_on_tile": int(pr.size),
                           "n_picks_in_polygon": n_picks_in_poly, "ratio_heads_to_picks": (n_heads_poly / n_picks_in_poly) if n_picks_in_poly else None,
                           "pick_to_channel_m": {"median": float(np.median(d_picks)) if d_picks.size else None, "q25": float(np.percentile(d_picks, 25)) if d_picks.size else None,
                                                 "q75": float(np.percentile(d_picks, 75)) if d_picks.size else None, "max": float(np.max(d_picks)) if d_picks.size else None,
                                                 "frac_within_25m": float(np.mean(d_picks <= R8_MEDIAN_DIST_M)) if d_picks.size else None, "values": d_picks.tolist()}} | {k: v for k, v in info.items() if k != "alpha"} | ({"alpha": info["alpha"]} if "alpha" in info else {})
            chans[name] = (ch, hd)
            pooled.setdefault(name, {"d": [], "heads": 0, "picks": 0})
            pooled[name]["d"] += d_picks.tolist()
            pooled[name]["heads"] += n_heads_poly
            pooled[name]["picks"] += n_picks_in_poly
        out["tiles"][tile] = {"n_picks_on_tile": int(pr.size), "n_picks_in_polygon": n_picks_in_poly, "polygon_px": int(md.sum()), "models": stats}
        # figure: 3 x 3 panels
        names = list(models)
        ncols = 3
        nrows = int(np.ceil(len(names) / ncols))
        hs = hillshade(Z, dx)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, name in zip(axes.ravel(), names, strict=False):
            ax.axis("on")
            ch, hd = chans[name]
            ax.imshow(np.nan_to_num(hs, nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ov = np.where(ch, 1.0, np.nan)
            ax.imshow(ov, cmap=matplotlib.colors.ListedColormap(["#0072B2"]), alpha=0.8, interpolation="nearest")
            if md.any():
                ax.contour(md.astype(float), levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
            hr, hc = np.nonzero(hd)
            ax.plot(hc, hr, ".", color="#E69F00", ms=3, label=f"model heads ({int(hd.sum())}; {stats[name]['n_heads_in_polygon']} in polygon)")
            ax.plot(pc, pr, "^", color="#D55E00", ms=6, mec="k", mew=0.5, label=f"picks ({pr.size}; {n_picks_in_poly} in polygon)")
            s = stats[name]
            kind_txt = f"A_crit = {s['threshold']:,.0f} m²" if s["kind"] == "area" else f"α = {s['alpha']:g}, C = {s['threshold']:.4g}"
            ax.set_title(f"{name}: {kind_txt}; channel cells {s['n_channel']:,}\nmedian pick→channel {s['pick_to_channel_m']['median'] if s['pick_to_channel_m']['median'] is None else round(s['pick_to_channel_m']['median'], 1)} m; "
                         f"heads/picks {s['ratio_heads_to_picks'] if s['ratio_heads_to_picks'] is None else round(s['ratio_heads_to_picks'], 2)}; D {s['D']:.3g}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=6, loc="lower right")
        fig.suptitle(f"{tile}: model channel cells (blue), model heads (orange dots) and Steve's picks (triangles); dashed = M&D study-area polygon", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(fig_dir, f"{tile}_model_heads.png")
        for sc in (1.0, 0.8, 0.65):
            fig.savefig(path, dpi=int(200 * sc), bbox_inches="tight", facecolor="white")
            if os.path.getsize(path) <= 5_000_000:
                break
        plt.close(fig)
        out["figures"][tile] = {"path": os.path.relpath(path, os.path.dirname(os.path.dirname(FIG_DIR))), "bytes": os.path.getsize(path)}
        print(tile, {k: (round(v["pick_to_channel_m"]["median"], 1) if v["pick_to_channel_m"]["median"] is not None else None, v["n_heads_in_polygon"]) for k, v in stats.items()}, f"[{time.time() - t0:.0f} s]", flush=True)
    # R8 on TTV1-TTV3 together
    for name, p in pooled.items():
        d = np.array(p["d"])
        med = float(np.median(d)) if d.size else None
        ratio = p["heads"] / p["picks"] if p["picks"] else None
        agree = bool(med is not None and med <= R8_MEDIAN_DIST_M and ratio is not None and R8_RATIO_RANGE[0] <= ratio <= R8_RATIO_RANGE[1])
        if agree:
            reading = "agree with the mapped heads"
        elif ratio is not None and ratio < R8_RATIO_RANGE[0]:
            reading = "too few channels" + (" (distances large)" if med is not None and med > R8_MEDIAN_DIST_M else "")
        elif ratio is not None and ratio > R8_RATIO_RANGE[1]:
            reading = "too many channels"
        else:
            reading = "count in range but distances large" if med is not None and med > R8_MEDIAN_DIST_M else "undetermined"
        out["rule_R8"][name] = {"median_pick_to_channel_m": med, "n_picks": int(d.size), "n_heads_in_polygon": p["heads"], "n_picks_in_polygon": p["picks"], "ratio": ratio, "agree": agree, "reading": reading,
                                "thresholds": {"median_m": R8_MEDIAN_DIST_M, "ratio_range": list(R8_RATIO_RANGE)}}
    out["wall_s"] = time.time() - t0
    dump(out, out_path("modelheads", "modelheads.json"))
    print("R8:", {k: (v["median_pick_to_channel_m"], v["ratio"], v["reading"]) for k, v in out["rule_R8"].items()}, flush=True)


if __name__ == "__main__":
    main()
