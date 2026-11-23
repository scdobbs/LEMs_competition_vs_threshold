#!/usr/bin/env python3
"""Addendum 21, Part A (decisions 1.2 and 1.3): the map-scale head measurement and the criterion networks on the observed surface.

On the stage B full-site 1 m grid: block means at 12 m (and 10 m), pit-filled, D8-routed (`border='outlets'`); D-infinity at 12 m.
Per pick (decision 1.2): the largest A in the 3 x 3 block around the pick's cell, S_12 up the largest-area donor chain over one cell
(two cells as the sensitivity); fits log10 A = log10 a - b log10 S (OLS on log A, 10,000-draw bootstrap, seed 20261123) against M&D eq. 10;
A_F = median A_12, C_F = median A_12 S_12^2 (and alpha = 1.65); everything also without pick 49. Rule H0b.
The criterion on the 12 m grid (decision 1.3): T = A_12 S_12^alpha, its upstream maximum M and the donors' maximum Mdon for alpha = 0,
1.65 and 2; for each TTV primary tile the candidate 12 m cells (those draining into the tile box with a 36 m margin), their 1 m seed
cell (largest full-grid area in the block) and the seed path's entry cell into the tile; the same for the 12 m cells of the 64 picks
(the mapped-head network). Outputs <run>/grid12/grid12.json, <run>/grid12/<tile>_seeds.npz, figure reports/v2/figures/a21/heads_area_slope_12m.png.
usage: a21_grid12.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import (  # noqa: E402
    ALPHA_PRIMARY,
    ALPHA_SENS,
    BLOCK_M,
    BLOCK_SENS_M,
    FIG_DIR,
    FULL_ORIGIN,
    H0B_RATIO,
    MARGIN_CELLS,
    MD_EQ10,
    N_BOOT_FITS,
    PICK_EXCLUDED_SENS,
    SEED,
    TTV_FULL,
    TTV_PRIMARY,
    block_view,
    coarse_grid,
    donor_slope,
    drains_to,
    dump,
    head_statistic_coarse,
    json_ready,
    md_table1,
    ols_loglog_bootstrap,
    out_path,
    pct,
    pick_area_cell,
    pick_cells,
    picks,
    slurm_record,
    tile_bounds,
    trace_paths,
)

ALPHAS = (0.0, ALPHA_SENS, ALPHA_PRIMARY)


def read_full():
    import rasterio
    with rasterio.open(TTV_FULL["filled"]) as ds:
        Z = ds.read(1).astype(np.float32)
        Z[Z == ds.nodata] = np.nan
        tr = ds.transform
    with rasterio.open(TTV_FULL["fd"]) as ds:
        FD = ds.read(1)
    with rasterio.open(TTV_FULL["area_cells"]) as ds:
        Ac = ds.read(1)
    assert abs(tr.c - FULL_ORIGIN[0]) < 1e-6 and abs(tr.f - FULL_ORIGIN[1]) < 1e-6 and abs(tr.a) == 1.0 and abs(tr.e) == 1.0, tr
    return Z, FD, Ac


def pick_table(g, S1, S2, pk, b, label, Ad=None):
    x0, y1 = FULL_ORIGIN
    rows = []
    r_all, c_all = pick_cells(np.array([q["x"] for q in pk]), np.array([q["y"] for q in pk]), x0, y1, b)
    for q, r, c in zip(pk, r_all, c_all, strict=True):
        rr, cc, a = pick_area_cell(g["A"], int(r), int(c))
        rec = {"pick_id": q["pick_id"], "row": int(r), "col": int(c), "A_cell": float(g["A"][r, c]), "area_row": rr, "area_col": cc, "A": a,
               "S1": float(S1["S"][rr, cc]), "S1_L": float(S1["L"][rr, cc]), "S2": float(S2["S"][rr, cc]), "S2_L": float(S2["L"][rr, cc]), "S2_short": bool(S2["short"][rr, cc]),
               "z_f": float(g["Zf"][rr, cc]), "edge_dist_cells": int(min(r, c, g["A"].shape[0] - 1 - r, g["A"].shape[1] - 1 - c))}
        if Ad is not None:
            r2, c2, a2 = pick_area_cell(Ad, int(r), int(c))
            rec.update({"A_dinf": a2, "A_dinf_row": r2, "A_dinf_col": c2})
        rows.append(rec)
    return rows


def fit_block(A, S, mask=None):
    ok = np.isfinite(A) & np.isfinite(S) & (A > 0) & (S > 0)
    if mask is not None:
        ok &= mask
    f = ols_loglog_bootstrap(A[ok], S[ok], n_boot=N_BOOT_FITS, seed=SEED)
    f["n_used"] = int(ok.sum())
    ratio = A[ok] / (MD_EQ10["a_m2"] * S[ok] ** (-MD_EQ10["b"]))
    med = float(np.median(ratio))
    f["ratio_to_eq10"] = pct(ratio) | {"median": med, "frac_in_0.5_2": float(np.mean((ratio >= H0B_RATIO[0]) & (ratio <= H0B_RATIO[1])))}
    f["H0b"] = {"b_1.65_inside_ci95": bool(f["b_ci95"][0] <= MD_EQ10["b"] <= f["b_ci95"][1]), "median_ratio_in_0.5_2": bool(H0B_RATIO[0] <= med <= H0B_RATIO[1])}
    f["H0b"]["consistent"] = bool(f["H0b"]["b_1.65_inside_ci95"] and f["H0b"]["median_ratio_in_0.5_2"])
    return f


def field_values(rows, key_a="A", key_s="S1"):
    A = np.array([r[key_a] for r in rows], dtype=float)
    S = np.array([r[key_s] for r in rows], dtype=float)
    not49 = np.array([r["pick_id"] != PICK_EXCLUDED_SENS for r in rows])
    Spos = np.where(np.isfinite(S) & (S > 0), S, np.nan)
    out = {"A_F": float(np.nanmedian(A)), "A_F_without_49": float(np.nanmedian(A[not49])), "n_S_nonpositive_or_nan": int(np.sum(~np.isfinite(Spos)))}
    for al in (ALPHA_PRIMARY, ALPHA_SENS):
        v = A * Spos ** al
        out[f"C_F_a{al:g}"] = float(np.nanmedian(v))
        out[f"C_F_a{al:g}_without_49"] = float(np.nanmedian(v[not49]))
        out[f"AS{al:g}_pct"] = pct(v)
    out["C_F"] = out[f"C_F_a{ALPHA_PRIMARY:g}"]
    out["C_F_without_49"] = out[f"C_F_a{ALPHA_PRIMARY:g}_without_49"]
    out["A_pct"], out["S_pct"] = pct(A), pct(S)
    return out


def tile_seeds(tile, b, g, S1, stats, Z1, FD1, Ac1, pk_rows, t0):
    """Candidate 12 m cells of a tile, their 1 m seeds and entry cells; the same for the picks' 12 m cells."""
    x0, y1 = FULL_ORIGIN
    bd = tile_bounds(tile)
    tr0, tc0 = int(round(y1 - bd["top"])), int(round(bd["left"] - x0))          # the tile's first 1 m row / col on the full grid
    tr1, tc1 = tr0 + bd["height"], tc0 + bd["width"]
    R0, R1 = int(math.floor((y1 - bd["top"]) / b)), int(math.ceil((y1 - bd["bottom"]) / b))
    C0, C1 = int(math.floor((bd["left"] - x0) / b)), int(math.ceil((bd["right"] - x0) / b))
    ny12, nx12 = g["A"].shape
    mR0, mR1, mC0, mC1 = max(R0 - MARGIN_CELLS, 0), min(R1 + MARGIN_CELLS, ny12), max(C0 - MARGIN_CELLS, 0), min(C1 + MARGIN_CELLS, nx12)
    box = np.zeros((ny12, nx12), dtype=bool)
    box[mR0:mR1, mC0:mC1] = True
    dr, n_it = drains_to(g["FD"], box)
    cR, cC = np.nonzero(dr & np.isfinite(g["A"]))
    # seeds: the largest-area 1 m cell of each candidate block (full-grid area)
    Av = block_view(Ac1, b)                                                     # view [R, C, i, j], no copy
    blocks = Av[cR, cC].reshape(cR.size, b * b)
    k = np.argmax(blocks, axis=1)
    sr, sc = cR * b + k // b, cC * b + k % b
    seed_A = blocks[np.arange(cR.size), k].astype(float)
    tp = trace_paths(FD1, sr, sc, target_box=(tr0, tr1, tc0, tc1))
    er = np.where(tp["reached"], tp["end_row"] - tr0, -1)
    ec = np.where(tp["reached"], tp["end_col"] - tc0, -1)
    rec = {"tile": tile, "b": b, "box12": [R0, R1, C0, C1], "box12_margin": [mR0, mR1, mC0, mC1], "tile_1m_full": [tr0, tr1, tc0, tc1], "drains_iters": n_it,
           "n_candidates": int(cR.size), "n_reach_tile": int(tp["reached"].sum()), "n_unfinished": int(tp["unfinished"].sum()), "max_steps": int(tp["steps"].max()) if cR.size else 0,
           "seed_A_pct": pct(seed_A)}
    arrays = {"cand_R": cR.astype(np.int32), "cand_C": cC.astype(np.int32), "seed_r": sr.astype(np.int32), "seed_c": sc.astype(np.int32), "seed_A": seed_A,
              "entry_r": er.astype(np.int32), "entry_c": ec.astype(np.int32), "cand_A12": g["A"][cR, cC], "cand_S12": S1["S"][cR, cC]}
    for al in ALPHAS:
        arrays[f"cand_M_a{al:g}"] = stats[al]["M"][cR, cC]
        arrays[f"cand_Mdon_a{al:g}"] = stats[al]["Mdon"][cR, cC]
        arrays[f"box_M_a{al:g}"] = stats[al]["M"][mR0:mR1, mC0:mC1]
        arrays[f"box_Mdon_a{al:g}"] = stats[al]["Mdon"][mR0:mR1, mC0:mC1]
    arrays.update({"box_Zf": g["Zf"][mR0:mR1, mC0:mC1], "box_A": g["A"][mR0:mR1, mC0:mC1], "box_S1": S1["S"][mR0:mR1, mC0:mC1], "box_FD": g["FD"][mR0:mR1, mC0:mC1],
                   "box12": np.array([R0, R1, C0, C1]), "box12_margin": np.array([mR0, mR1, mC0, mC1]), "tile_1m_full": np.array([tr0, tr1, tc0, tc1])})
    # the picks' 12 m cells (the area cell of decision 1.2) seeded the same way: the mapped-head network of decision 1.3
    pR = np.array([q["area_row"] for q in pk_rows])
    pC = np.array([q["area_col"] for q in pk_rows])
    pb = Av[pR, pC].reshape(pR.size, b * b)
    kp = np.argmax(pb, axis=1)
    psr, psc = pR * b + kp // b, pC * b + kp % b
    tpp = trace_paths(FD1, psr, psc, target_box=(tr0, tr1, tc0, tc1))
    arrays.update({"pick_id": np.array([q["pick_id"] for q in pk_rows], dtype=np.int32), "pick_R": pR.astype(np.int32), "pick_C": pC.astype(np.int32),
                   "pick_seed_r": psr.astype(np.int32), "pick_seed_c": psc.astype(np.int32), "pick_seed_A": pb[np.arange(pR.size), kp].astype(float),
                   "pick_entry_r": np.where(tpp["reached"], tpp["end_row"] - tr0, -1).astype(np.int32), "pick_entry_c": np.where(tpp["reached"], tpp["end_col"] - tc0, -1).astype(np.int32)})
    rec["n_pick_seeds_reaching_tile"] = int(tpp["reached"].sum())
    rec["pick_seed_inside_tile"] = int(((psr >= tr0) & (psr < tr1) & (psc >= tc0) & (psc < tc1)).sum())
    np.savez_compressed(out_path("grid12", f"{tile}_seeds.npz"), **arrays)
    print(f"{tile}: box {R0}-{R1} x {C0}-{C1}; candidates {cR.size} (drains in {n_it} it.), reach the tile {rec['n_reach_tile']} (unfinished {rec['n_unfinished']}, "
          f"max steps {rec['max_steps']}); pick seeds reaching {rec['n_pick_seeds_reaching_tile']} [{time.time() - t0:.0f} s]", flush=True)
    return rec


def figure(rows10, rows12, fits, t1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    A1, S1 = np.array([x["A"] for x in t1]), np.array([x["S"] for x in t1])
    ax.loglog(S1, A1, "s", color="0.6", ms=4, label="M&D (1989) Table 1 (63)")
    A12, S12 = np.array([q["A"] for q in rows12]), np.array([q["S1"] for q in rows12])
    ax.loglog(S12, A12, "^", color="#D55E00", mec="k", mew=0.4, ms=6, label="picks, 12 m grid (64)")
    A10, S10 = np.array([q["A"] for q in rows10]), np.array([q["S1"] for q in rows10])
    ax.loglog(S10, A10, "o", mfc="none", mec="#0072B2", ms=5, label="picks, 10 m grid")
    s = np.logspace(-0.9, 0.1, 50)
    ax.loglog(s, MD_EQ10["a_m2"] * s ** (-MD_EQ10["b"]), "k-", lw=1.2, label=f"M&D eq. 10: A = {MD_EQ10['a_m2']:.0f} S^-{MD_EQ10['b']}")
    f = fits["picks_12m_S1"]
    ax.loglog(s, f["a"] * s ** (-f["b"]), "-", color="#D55E00", lw=1.2, label=f"fit 12 m: a {f['a']:,.0f}, b {f['b']:.2f} [{f['b_ci95'][0]:.2f}, {f['b_ci95'][1]:.2f}]")
    f2 = fits["picks_12m_S2"]
    ax.loglog(s, f2["a"] * s ** (-f2["b"]), "--", color="#D55E00", lw=0.9, label=f"fit 12 m, two-cell S: b {f2['b']:.2f}")
    ax.set_xlabel("local valley slope above the head S (m/m)")
    ax.set_ylabel("source area A (m²)")
    ax.set_title("Addendum 21: map-scale source areas and slopes at Steve's picks", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    p = os.path.join(FIG_DIR, "heads_area_slope_12m.png")
    fig.tight_layout()
    fig.savefig(p, dpi=160, facecolor="white")
    plt.close(fig)
    return {"path": os.path.relpath(p, os.path.dirname(os.path.dirname(FIG_DIR))), "bytes": os.path.getsize(p)}


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("grid12", "grid12.json")))
    pk = picks()
    Z1, FD1, Ac1 = read_full()
    print(f"full grid {Z1.shape} [{time.time() - t0:.0f} s]", flush=True)
    out: dict = {"grid": {"shape": list(Z1.shape), "origin": list(FULL_ORIGIN)}, "scales": {}, "fits": {}, "field_values": {}, "tiles": {}}
    stats12 = {}
    g12 = S1_12 = None
    rows12 = rows10 = None
    for b in (BLOCK_M, BLOCK_SENS_M):
        t1 = time.perf_counter()
        g = coarse_grid(Z1, b, 1.0)
        S1 = donor_slope(g["Zf"], g["A"], g["FD"], float(b), 1)
        S2 = donor_slope(g["Zf"], g["A"], g["FD"], float(b), 2)
        Ad = None
        if b == BLOCK_M:
            from lemcvt.hydro.dinf import compute_dinf_area, compute_dinf_flow_direction
            ang, rf, fi = compute_dinf_flow_direction(g["Zf"], float(b), FD_d8=g["FD"])
            Ad = compute_dinf_area(g["Zf"], float(b), ang, rf, fi)
            Ad[~g["valid"]] = np.nan
        rows = pick_table(g, S1, S2, pk, b, f"{b}m", Ad)
        sc = {"b": b, "shape": list(g["A"].shape), "n_valid": int(g["valid"].sum()), "wall_s": time.perf_counter() - t1, "picks": rows,
              "A_pct": pct([q["A"] for q in rows]), "A_cell_pct": pct([q["A_cell"] for q in rows]), "S1_pct": pct([q["S1"] for q in rows]), "S2_pct": pct([q["S2"] for q in rows]),
              "n_S2_short": int(sum(q["S2_short"] for q in rows)), "n_S1_nan": int(sum(not np.isfinite(q["S1"]) for q in rows))}
        t1_tab = md_table1()
        At1 = np.array([x["A"] for x in t1_tab])
        sc["frac_within_table1_5_95"] = float(np.mean((np.array([q["A"] for q in rows]) >= np.percentile(At1, 5)) & (np.array([q["A"] for q in rows]) <= np.percentile(At1, 95))))
        sc["median_ratio_to_table1_median"] = float(np.median([q["A"] for q in rows]) / np.median(At1))
        if Ad is not None:
            sc["A_dinf_pct"] = pct([q["A_dinf"] for q in rows])
        out["scales"][str(b)] = sc
        print(f"b {b} m: A median {sc['A_pct']['p50']:.0f} m2 (Table 1 7,400; within 5-95 % {sc['frac_within_table1_5_95']:.2f}); S1 median {sc['S1_pct']['p50']:.3f}, S2 {sc['S2_pct']['p50']:.3f} [{time.time() - t0:.0f} s]", flush=True)
        if b == BLOCK_M:
            g12, S1_12, rows12 = g, S1, rows
            for al in ALPHAS:
                stats12[al] = head_statistic_coarse(g["A"], S1["S"], g["FD"], al)
            del Ad
        else:
            rows10 = rows
    assert g12 is not None and rows12 is not None and rows10 is not None and S1_12 is not None
    # fits and field values
    A12 = np.array([q["A"] for q in rows12])
    S1a = np.array([q["S1"] for q in rows12])
    S2a = np.array([q["S2"] for q in rows12])
    not49 = np.array([q["pick_id"] != PICK_EXCLUDED_SENS for q in rows12])
    Adn = np.array([q["A_dinf"] for q in rows12])
    A10 = np.array([q["A"] for q in rows10])
    S10 = np.array([q["S1"] for q in rows10])
    t1_tab = md_table1()
    out["fits"] = {"table1": fit_block(np.array([x["A"] for x in t1_tab]), np.array([x["S"] for x in t1_tab])),
                   "picks_12m_S1": fit_block(A12, S1a), "picks_12m_S1_without_49": fit_block(A12, S1a, not49), "picks_12m_S2": fit_block(A12, S2a),
                   "picks_12m_dinf_S1": fit_block(Adn, S1a), "picks_10m_S1": fit_block(A10, S10)}
    out["field_values"] = {"12m": field_values(rows12, "A", "S1"), "12m_S2": field_values(rows12, "A", "S2"), "12m_dinf": field_values(rows12, "A_dinf", "S1"), "10m": field_values(rows10, "A", "S1")}
    fv = out["field_values"]["12m"]
    out["rule_H0b"] = {"primary": out["fits"]["picks_12m_S1"]["H0b"], "without_49": out["fits"]["picks_12m_S1_without_49"]["H0b"], "two_cell": out["fits"]["picks_12m_S2"]["H0b"],
                       "dinf": out["fits"]["picks_12m_dinf_S1"]["H0b"], "10m": out["fits"]["picks_10m_S1"]["H0b"]}
    print("fits:", {k: (round(v["a"]), round(v["b"], 3), [round(x, 2) for x in v["b_ci95"]], round(v["ratio_to_eq10"]["median"], 2)) for k, v in out["fits"].items()}, flush=True)
    print(f"field values (12 m): A_F {fv['A_F']:.0f} m2, C_F {fv['C_F']:.1f} m2 (alpha 2), C_F,1.65 {fv['C_F_a1.65']:.1f}; without 49: {fv['A_F_without_49']:.0f}, {fv['C_F_without_49']:.1f}", flush=True)
    # the criterion on the whole 12 m grid at the field values (counts, for the record)
    out["criterion_full_grid"] = {}
    for al in ALPHAS:
        C = fv["A_F"] if al == 0 else fv[f"C_F_a{al:g}"]
        M, Md = stats12[al]["M"], stats12[al]["Mdon"]
        ch = np.isfinite(M) & (M >= C)
        out["criterion_full_grid"][f"a{al:g}"] = {"C": C, "n_channel_cells": int(ch.sum()), "n_heads": int((ch & ~(Md >= C)).sum()), "n_unique_M": int(np.unique(M[np.isfinite(M)]).size)}
    # per-tile seeds and entries
    for tile in TTV_PRIMARY:
        out["tiles"][tile] = tile_seeds(tile, BLOCK_M, g12, S1_12, stats12, Z1, FD1, Ac1, rows12, t0)
    out["figure"] = figure(rows10, rows12, out["fits"], t1_tab)
    out["wall_s"] = time.time() - t0
    dump(json_ready(out), out_path("grid12", "grid12.json"))
    np.savez_compressed(out_path("grid12", "grid12_arrays.npz"), Zf=g12["Zf"].astype(np.float32), A=g12["A"].astype(np.float32), FD=g12["FD"], S1=S1_12["S"].astype(np.float32),
                        **{f"M_a{al:g}": stats12[al]["M"] for al in ALPHAS}, **{f"Mdon_a{al:g}": stats12[al]["Mdon"] for al in ALPHAS})
    print(f"done [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
