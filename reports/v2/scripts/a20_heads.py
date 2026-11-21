#!/usr/bin/env python3
"""Addendum 20, decision 5 (section 3.3): Steve's 64 picks snapped along their own D8 flow paths on the stage B full-site TTV grid.

Per pick: the D8 path downstream from the pick cell; A at path distances 0, 5, 10, 15, 20, 30 m (the first path cell whose path length from
the pick is >= the distance; A_P10 primary); the axis entry (the first path cell with A >= 10 x A at the pick cell: its distance and area);
from the A_P10 cell S_up (largest-area donor chain, 10 m; 5 and 20 m sensitivity) and S_down (10 m of D8 path below it); the same slopes from
the A_P5 and A_P20 cells for their fits. Fits (ii) A_P10/S_up, (iii) A_P10/S_down, (iv) (ii) without pick 49, and A_P5, A_P20 (each with
S_up from its own cell), OLS on log10 A with the 10,000-draw bootstrap (seed 20261120); (i) Table 1 and rule R4 carried from addendum 19
(recomputed here as a check); R5 with A_P10 and S_up; A S^2 distributions; field values A_F = median A_P10, C_F(alpha) = median
A_P10 S_down^alpha at alpha = 0, 1.65, 2 and b_hat (rounded to 0.05). The snapped-head network (union of D8 paths below the A_P10 cells) for
the D_R of decision 3. QC pages: 40 m x 40 m per pick, 16 to a page.
usage: a20_heads.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_heads import TTV_TILES_ALL, in_polygon, read_full, receiver_of, slope_down, slope_up, step_len, tile_of  # noqa: E402
from a20_common import (  # noqa: E402
    ALPHA_BHAT_MIN_SEP,
    ALPHAS,
    AXIS_ENTRY_FACTOR,
    FIG_DIR,
    HEADS_CSV,
    HEADS_SHA256,
    MD_EQ10,
    MD_TABLE1_MEDIAN_M2,
    PATH_DISTANCES_M,
    QC_A_MIN_M2,
    QC_BOX_M,
    QC_PER_PAGE,
    R5_RATIO_RANGE,
    TTV_PRIMARY,
    a19_path,
    dump,
    hillshade,
    load_json,
    md_table1,
    ols_loglog_bootstrap,
    out_path,
    pct,
    sha256_file,
    slurm_record,
    tile_bounds,
)

S_UP_REACHES = (10.0, 5.0, 20.0)


def picks() -> list[dict]:
    import csv
    assert sha256_file(HEADS_CSV) == HEADS_SHA256, "heads CSV checksum differs from the copy of record"
    with open(HEADS_CSV) as f:
        rows = list(csv.DictReader(f))
    assert [r["pick_id"] for r in rows] == [str(i) for i in range(1, 65)]
    return [{"pick_id": int(r["pick_id"]), "x": float(r["x_epsg6339"]), "y": float(r["y_epsg6339"])} for r in rows]


def trace_path(FD, Z, r, c, dx, dy, max_len_m=200.0):
    """D8 path from (r, c): cells and cumulative path length, until max_len_m, a missing receiver or the grid edge."""
    ny, nx = Z.shape
    cells, L = [(r, c)], [0.0]
    cur = (r, c)
    while L[-1] < max_len_m:
        nxt = receiver_of(FD, *cur)
        if nxt is None or not (0 <= nxt[0] < ny and 0 <= nxt[1] < nx) or not np.isfinite(Z[nxt]):
            break
        L.append(L[-1] + step_len(nxt[0] - cur[0], nxt[1] - cur[1], dx, dy))
        cells.append(nxt)
        cur = nxt
    return cells, np.array(L)


def downstream_union(FD, Z, starts):
    ny, nx = FD.shape
    visited = np.zeros((ny, nx), dtype=bool)
    for r, c in starts:
        cur = (r, c)
        while not visited[cur]:
            visited[cur] = True
            nxt = receiver_of(FD, *cur)
            if nxt is None or not (0 <= nxt[0] < ny and 0 <= nxt[1] < nx) or not np.isfinite(Z[nxt]):
                break
            cur = nxt
    return visited


def fit_block(A, S, mask=None):
    ok = np.isfinite(A) & np.isfinite(S) & (A > 0) & (S > 0)
    if mask is not None:
        ok &= mask
    f = ols_loglog_bootstrap(A[ok], S[ok])
    f["n_used"] = int(ok.sum())
    return f


def r5(f, A, S, mask=None):
    ok = np.isfinite(A) & np.isfinite(S) & (S > 0) & (A > 0)
    if mask is not None:
        ok &= mask
    ratio = A[ok] / (MD_EQ10["a_m2"] * S[ok] ** (-MD_EQ10["b"]))
    med = float(np.median(ratio))
    b_in = bool(f["b_ci95"][0] <= MD_EQ10["b"] <= f["b_ci95"][1])
    med_in = bool(R5_RATIO_RANGE[0] <= med <= R5_RATIO_RANGE[1])
    return {"b_1.65_inside_ci95": b_in, "median_ratio_to_eq10": med, "median_in_range": med_in, "consistent": bool(b_in and med_in), "ratio_pct": pct(ratio),
            "frac_in_0.5_2": float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))}


def qc_pages(Z, Ac, cell_area, tr, dx, dy, rows, a19_by_id, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    os.makedirs(fig_dir, exist_ok=True)
    half = int(round(QC_BOX_M / 2 / dx))
    pages = []
    for p0 in range(0, len(rows), QC_PER_PAGE):
        chunk = rows[p0:p0 + QC_PER_PAGE]
        fig, axes = plt.subplots(4, 4, figsize=(15, 15.5), layout="constrained")
        for ax in axes.ravel():
            ax.axis("off")
        for ax, q in zip(axes.ravel(), chunk, strict=False):
            ax.axis("on")
            r0, c0 = q["row"], q["col"]
            rs, re_, cs, ce = r0 - half - 1, r0 + half + 2, c0 - half - 1, c0 + half + 2
            z = Z[rs:re_, cs:ce].astype(float)
            hs = hillshade(z, dx)[1:-1, 1:-1]
            a = Ac[rs + 1:re_ - 1, cs + 1:ce - 1].astype(float) * cell_area
            ext = (-half - 0.5, half + 0.5, half + 0.5, -half - 0.5)
            ax.imshow(np.nan_to_num(hs, nan=1.0), cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
            am = np.where(a >= QC_A_MIN_M2, a, np.nan)
            im = ax.imshow(am, cmap="viridis", norm=LogNorm(vmin=QC_A_MIN_M2, vmax=max(1e5, np.nanmax(a) if np.isfinite(a).any() else 1e5)), extent=ext, alpha=0.75, interpolation="nearest")
            pr = np.array([x[0] for x in q["_path"]]) - r0
            pc = np.array([x[1] for x in q["_path"]]) - c0
            inb = (np.abs(pr) <= half) & (np.abs(pc) <= half)
            ax.plot(pc[inb], pr[inb], "-", color="#D55E00", lw=1.4)
            ax.plot(0, 0, "^", color="#D55E00", mec="k", ms=10, label="pick")
            for d, mk in ((5, "s"), (10, "o"), (20, "D")):
                rr, cc = q[f"A_P{d}_row"] - r0, q[f"A_P{d}_col"] - c0
                if abs(rr) <= half and abs(cc) <= half:
                    ax.plot(cc, rr, mk, mfc="none", mec="#E69F00" if d != 10 else "#CC79A7", mew=2, ms=9, label=f"{d} m of path (A {q[f'A_P{d}_m2']:,.0f} m²)")
            old = a19_by_id.get(q["pick_id"])
            if old:
                for key, mk, col in (("A_3", "x", "#0072B2"), ("A_10", "+", "#56B4E9")):
                    rr, cc = int(old[f"{key}_row"]) - r0, int(old[f"{key}_col"]) - c0
                    if abs(rr) <= half and abs(cc) <= half:
                        ax.plot(cc, rr, mk, color=col, mew=2.5, ms=11, label=f"a19 {key} ({float(old[f'{key}_m2']):,.0f} m²)")
            ax.set_xlim(-half - 0.5, half + 0.5)
            ax.set_ylim(half + 0.5, -half - 0.5)
            ae = "none" if q["axis_entry_m"] is None else f"{q['axis_entry_m']:.1f} m"
            ax.set_title(f"pick {q['pick_id']} ({q['tile']})\nA_P10 {q['A_P10_m2']:,.0f} m², axis entry {ae}", fontsize=8.5)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="lower left", framealpha=0.85)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.35, label="D8 area (m²), cells with A ≥ 100 m²")
        fig.suptitle(f"Addendum 20 snapped heads, picks {chunk[0]['pick_id']}-{chunk[-1]['pick_id']}: 40 m x 40 m, 1 m hillshade (north up, metres from the pick); "
                     "orange line = D8 path from the pick", fontsize=11)
        path = os.path.join(fig_dir, f"heads_qc_page{p0 // QC_PER_PAGE + 1}.png")
        for sc in (1.0, 0.8, 0.6):
            fig.savefig(path, dpi=int(130 * sc), facecolor="white")
            if os.path.getsize(path) <= 5_000_000:
                break
        plt.close(fig)
        pages.append({"path": os.path.relpath(path, os.path.dirname(os.path.dirname(FIG_DIR))), "bytes": os.path.getsize(path), "picks": [q["pick_id"] for q in chunk]})
    return pages


def main() -> None:
    import argparse
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-only", action="store_true", help="redraw the QC pages from the stored attributes (heads.json untouched)")
    qc_only = ap.parse_args().qc_only
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("heads", "heads.json")))
    pk = picks()
    inside, d_line = in_polygon([q["x"] for q in pk], [q["y"] for q in pk])
    Z, FD, Ac, tr, crs, dx, dy = read_full()
    cell_area = dx * dy
    print(f"full grid {Z.shape}, dx {dx}, dy {dy}, cell area {cell_area} m2 [{time.time() - t0:.0f} s]", flush=True)
    bounds = {t: tile_bounds(t) for t in TTV_TILES_ALL}
    with open(a19_path("heads", "head_attributes.csv")) as f:
        a19 = {int(r["pick_id"]): r for r in csv.DictReader(f)}
    rows = []
    for q, ins, dl in zip(pk, inside, d_line, strict=True):
        c = int(math.floor((q["x"] - tr.c) / dx))
        r = int(math.floor((tr.f - q["y"]) / dy))
        assert (r, c) == (int(a19[q["pick_id"]]["row"]), int(a19[q["pick_id"]]["col"])), "pick cell differs from addendum 19"
        cells, L = trace_path(FD, Z, r, c, dx, dy)
        A_path = np.array([float(Ac[x]) * cell_area for x in cells])
        rec = dict(q) | {"row": r, "col": c, "tile": tile_of(q["x"], q["y"], bounds), "inside_md_polygon": ins, "dist_to_traced_channel_m": dl,
                         "A_cell_m2": A_path[0], "path_length_traced_m": float(L[-1]), "path_short": bool(L[-1] < max(PATH_DISTANCES_M))}
        rec["primary_tile"] = rec["tile"] in TTV_PRIMARY
        for d in PATH_DISTANCES_M:
            j = int(np.searchsorted(L, d - 1e-9))
            j = min(j, len(cells) - 1)
            rec[f"A_P{d:g}_m2"] = float(A_path[j])
            rec[f"A_P{d:g}_row"], rec[f"A_P{d:g}_col"] = int(cells[j][0]), int(cells[j][1])
            rec[f"A_P{d:g}_L_m"] = float(L[j])
            rec[f"A_P{d:g}_reached"] = bool(L[j] >= d - 1e-9)
            rec[f"A_P{d:g}_offset_m"] = float(math.hypot((cells[j][0] - r) * dy, (cells[j][1] - c) * dx))
        je = next((j for j in range(len(cells)) if A_path[j] >= AXIS_ENTRY_FACTOR * A_path[0]), None)
        rec["axis_entry_m"] = float(L[je]) if je is not None else None
        rec["axis_entry_A_m2"] = float(A_path[je]) if je is not None else None
        for d in (10, 5, 20):
            rr, cc = rec[f"A_P{d}_row"], rec[f"A_P{d}_col"]
            for reach in (S_UP_REACHES if d == 10 else (10.0,)):
                su = slope_up(Z, FD, Ac, dx, dy, rr, cc, reach)
                tag = f"S_up_P{d}" + ("" if reach == 10.0 else f"_{reach:g}m")
                rec[tag], rec[tag + "_L"], rec[tag + "_short"] = su["S"], su["L"], su["short"]
            sd = slope_down(Z, FD, dx, dy, rr, cc, 10.0)
            rec[f"S_down_P{d}"], rec[f"S_down_P{d}_L"], rec[f"S_down_P{d}_left"] = sd["S"], sd["L"], sd["left"]
        rec["A_3_a19_m2"] = float(a19[q["pick_id"]]["A_3_m2"])
        rec["A_10_a19_m2"] = float(a19[q["pick_id"]]["A_10_m2"])
        rec["_path"] = cells[: int(np.searchsorted(L, 40.0)) + 1]
        rows.append(rec)
    print(f"attributes done [{time.time() - t0:.0f} s]", flush=True)
    if qc_only:
        pages = qc_pages(Z, Ac, cell_area, tr, dx, dy, rows, a19, os.path.join(FIG_DIR, "heads"))
        dump({"qc_pages": pages}, out_path("heads", "qc_pages.json"))
        print(f"QC pages redrawn: {len(pages)} [{time.time() - t0:.0f} s]", flush=True)
        return
    # snapped-head network (decision 3) below the A_P10 cells
    net = downstream_union(FD, Z, [(x["A_P10_row"], x["A_P10_col"]) for x in rows])
    rr_, cc_ = np.nonzero(net)
    np.savez_compressed(out_path("heads", "snapped_network.npz"), rows=rr_.astype(np.int32), cols=cc_.astype(np.int32), transform=np.array(list(tr)[:6]),
                        head_rows=np.array([x["A_P10_row"] for x in rows]), head_cols=np.array([x["A_P10_col"] for x in rows]))
    del net
    # QC pages
    pages = qc_pages(Z, Ac, cell_area, tr, dx, dy, rows, a19, os.path.join(FIG_DIR, "heads"))
    print(f"QC pages {len(pages)} [{time.time() - t0:.0f} s]", flush=True)
    for x in rows:
        x.pop("_path")
    with open(out_path("heads", "head_attributes_snapped.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # fits
    t1 = md_table1()
    A1, S1 = np.array([x["A"] for x in t1]), np.array([x["S"] for x in t1])
    g = lambda k: np.array([x[k] for x in rows], dtype=float)  # noqa: E731
    AP10, Sup, Sdn = g("A_P10_m2"), g("S_up_P10"), g("S_down_P10")
    not49 = np.array([x["pick_id"] != 49 for x in rows])
    fits = {"i_table1": fit_block(A1, S1), "ii_AP10_Sup": fit_block(AP10, Sup), "iii_AP10_Sdown": fit_block(AP10, Sdn), "iv_AP10_Sup_without_49": fit_block(AP10, Sup, not49),
            "AP5_Sup": fit_block(g("A_P5_m2"), g("S_up_P5")), "AP20_Sup": fit_block(g("A_P20_m2"), g("S_up_P20")),
            "AP5_Sdown": fit_block(g("A_P5_m2"), g("S_down_P5")), "AP20_Sdown": fit_block(g("A_P20_m2"), g("S_down_P20"))}
    a19h = load_json(a19_path("heads", "heads.json"))
    fi = fits["i_table1"]
    r4 = {"a19": a19h["rules"]["R4"], "recomputed": {"a": fi["a"], "b": fi["b"], "R2": fi["R2"]},
          "identical_to_a19": bool(abs(fi["a"] / a19h["rules"]["R4"]["a"] - 1) < 1e-12 and abs(fi["b"] - a19h["rules"]["R4"]["b"]) < 1e-12)}
    rules = {"R4": r4, "R5": {"ii": r5(fits["ii_AP10_Sup"], AP10, Sup), "iii_report_only": r5(fits["iii_AP10_Sdown"], AP10, Sdn),
                              "iv_report_only": r5(fits["iv_AP10_Sup_without_49"], AP10, Sup, not49),
                              "AP5_report_only": r5(fits["AP5_Sup"], g("A_P5_m2"), g("S_up_P5")), "AP20_report_only": r5(fits["AP20_Sup"], g("A_P20_m2"), g("S_up_P20"))}}
    rules["R5"]["verdict"] = rules["R5"]["ii"]["consistent"]
    as2 = {"picks_AP10_Sup": pct(AP10 * Sup ** 2), "picks_AP10_Sdown": pct(AP10 * Sdn ** 2), "table1": pct(A1 * S1 ** 2), "MD1992_range_m2": [500.0, 4000.0],
           "picks_AP10_Sup_in_MD_range_frac": float(np.mean((AP10 * Sup ** 2 >= 500) & (AP10 * Sup ** 2 <= 4000))),
           "picks_AP10_Sdown_in_MD_range_frac": float(np.mean((AP10 * Sdn ** 2 >= 500) & (AP10 * Sdn ** 2 <= 4000))),
           "picks_AP5_Sup": pct(g("A_P5_m2") * g("S_up_P5") ** 2), "picks_AP20_Sup": pct(g("A_P20_m2") * g("S_up_P20") ** 2)}
    dists = {f"A_P{d:g}": pct(g(f"A_P{d:g}_m2")) for d in PATH_DISTANCES_M} | {
        "A_cell": pct(g("A_cell_m2")), "A_3_a19": pct(g("A_3_a19_m2")), "A_10_a19": pct(g("A_10_a19_m2")), "table1_area": pct(A1), "A_P10_without_49": pct(AP10[not49]),
        "S_up_P10": pct(Sup), "S_down_P10": pct(Sdn), "table1_slope": pct(S1), "axis_entry_m": pct([x["axis_entry_m"] for x in rows if x["axis_entry_m"] is not None]),
        "axis_entry_A_m2": pct([x["axis_entry_A_m2"] for x in rows if x["axis_entry_A_m2"] is not None]), "n_no_axis_entry_within_200m": int(sum(x["axis_entry_m"] is None for x in rows)),
        "n_path_short_of_30m": int(sum(x["path_short"] for x in rows)), "n_S_up_P10_short": int(sum(x["S_up_P10_short"] for x in rows)), "n_S_down_P10_left": int(sum(x["S_down_P10_left"] for x in rows)),
        "A_P10_over_A_3": pct(AP10 / g("A_3_a19_m2")), "A_P10_over_A_10": pct(AP10 / g("A_10_a19_m2")),
        "tiles": {t: int(sum(x["tile"] == t for x in rows)) for t in sorted({x["tile"] for x in rows})}}
    b_hat = fits["ii_AP10_Sup"]["b"]
    alpha_b = float(f"{round(b_hat / 0.05) * 0.05:.2f}")
    run_bhat = bool(min(abs(alpha_b - a) for a in ALPHAS) > ALPHA_BHAT_MIN_SEP)
    alphas_sf = sorted(set(ALPHAS))
    field = {"A_F_m2": float(np.median(AP10)), "A_F_without_49_m2": float(np.median(AP10[not49])), "A_F_table1_m2": MD_TABLE1_MEDIAN_M2, "b_hat": b_hat, "alpha_b_hat": alpha_b,
             "run_bhat_in_SJ": run_bhat, "alphas_SF": alphas_sf,
             "C_F": {f"{al:g}": float(np.median(AP10 * np.where(Sdn > 0, Sdn, np.nan) ** al)) if al > 0 else float(np.median(AP10)) for al in alphas_sf + ([alpha_b] if run_bhat else [])},
             "C_F_without_49": {f"{al:g}": float(np.nanmedian((AP10 * np.where(Sdn > 0, Sdn, np.nan) ** al)[not49])) if al > 0 else float(np.median(AP10[not49])) for al in alphas_sf},
             "n_S_down_nonpositive": int(np.sum(~(Sdn > 0))),
             "C_F_definition": "median over the 64 picks of A_P10 * S_down^alpha (S_down the 10 m D8 reach slope below the A_P10 cell); picks with S_down <= 0 drop out at alpha > 0"}
    out = {"csv": {"path": HEADS_CSV, "sha256": sha256_file(HEADS_CSV)}, "grid": {"shape": list(Z.shape), "dx": dx, "dy": dy, "cell_area_m2": cell_area, "crs": str(crs)},
           "picks": rows, "fits": fits, "rules": rules, "AS2": as2, "distributions": dists, "field_values": field, "md_eq10": MD_EQ10, "qc_pages": pages,
           "snapped_network_cells": int(rr_.size), "wall_s": time.time() - t0}
    dump(out, out_path("heads", "heads.json"))
    print("fits:", {k: (round(v["a"]), round(v["b"], 3), round(v["R2"], 3)) for k, v in fits.items()}, flush=True)
    print("R5:", rules["R5"]["ii"], "\nfield:", field, f"[{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
