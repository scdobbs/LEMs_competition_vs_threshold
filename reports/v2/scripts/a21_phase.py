#!/usr/bin/env python3
"""Addendum 21, a sensitivity added after Part A (stated in the report): the 12 m block lattice has a phase (the full-site lattice of
`a21_grid12.py` starts at the grid origin; the addendum 20 window started at x = 538,800, y = 4,192,400, 7 m east and 5 m south of a
full-site block boundary), and A_12 at a pick depends on which 12 m blocks the pick's hollow falls into. The decision 1.2 measurement
is repeated on 16 lattice phases (offsets 0, 3, 6, 9 m in x and y) and on the addendum 20 window phase: A_F, C_F (alpha 2), the
log A - log S fit. Output <run>/grid12/phase.json.
usage: a21_phase.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import (  # noqa: E402
    BLOCK_M,
    FULL_ORIGIN,
    coarse_grid,
    donor_slope,
    dump,
    json_ready,
    out_path,
    pct,
    pick_area_cell,
    picks,
    slurm_record,
)
from a21_grid12 import field_values, fit_block, read_full  # noqa: E402

A20_WINDOW = (538800.0, 4192400.0)


def measure(Z1, pk, dr: int, dc: int, b: int) -> dict:
    """The decision 1.2 measurement on the lattice whose blocks start at (dr, dc) of the full grid."""
    x0, y1 = FULL_ORIGIN[0] + dc, FULL_ORIGIN[1] - dr
    g = coarse_grid(Z1[dr:, dc:], b, 1.0)
    S1 = donor_slope(g["Zf"], g["A"], g["FD"], float(b), 1)
    rows = []
    for q in pk:
        r, c = int(np.floor((y1 - q["y"]) / b)), int(np.floor((q["x"] - x0) / b))
        rr, cc, a = pick_area_cell(g["A"], r, c)
        rows.append({"pick_id": q["pick_id"], "A": a, "S1": float(S1["S"][rr, cc])})
    fv = field_values(rows, "A", "S1")
    fit = fit_block(np.array([x["A"] for x in rows]), np.array([x["S1"] for x in rows]))
    return {"offset_rc": [dr, dc], "A_F": fv["A_F"], "C_F": fv["C_F"], "C_F_a1.65": fv["C_F_a1.65"], "A_pct": pct([x["A"] for x in rows]), "b": fit["b"], "b_ci95": fit["b_ci95"], "a": fit["a"],
            "median_ratio_to_eq10": fit["ratio_to_eq10"]["median"], "H0b": fit["H0b"]["consistent"]}


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("grid12", "phase.json")))
    pk = picks()
    Z1, _FD, _Ac = read_full()
    out: dict = {"phases": [], "b": BLOCK_M}
    for dr in (0, 3, 6, 9):
        for dc in (0, 3, 6, 9):
            m = measure(Z1, pk, dr, dc, BLOCK_M)
            out["phases"].append(m)
            print(f"phase ({dr}, {dc}): A_F {m['A_F']:.0f}, C_F {m['C_F']:.0f}, b {m['b']:.2f} {m['b_ci95']}, ratio {m['median_ratio_to_eq10']:.2f} [{time.time() - t0:.0f} s]", flush=True)
    dc20 = int(round((A20_WINDOW[0] - FULL_ORIGIN[0]) % BLOCK_M))
    dr20 = int(round((FULL_ORIGIN[1] - A20_WINDOW[1]) % BLOCK_M))
    out["a20_window_phase"] = measure(Z1, pk, dr20, dc20, BLOCK_M) | {"note": "the addendum 20 T2 window lattice (x 538,800, y 4,192,400)"}
    print(f"a20 window phase ({dr20}, {dc20}): A_F {out['a20_window_phase']['A_F']:.0f}, C_F {out['a20_window_phase']['C_F']:.0f}", flush=True)
    af = np.array([m["A_F"] for m in out["phases"]])
    cf = np.array([m["C_F"] for m in out["phases"]])
    out["summary"] = {"A_F": pct(af) | {"min": float(af.min()), "max": float(af.max())}, "C_F": pct(cf) | {"min": float(cf.min()), "max": float(cf.max())},
                      "n_H0b_consistent": int(sum(m["H0b"] for m in out["phases"])), "n_phases": len(out["phases"]), "b_range": [float(min(m["b"] for m in out["phases"])), float(max(m["b"] for m in out["phases"]))]}
    out["wall_s"] = time.time() - t0
    dump(json_ready(out), out_path("grid12", "phase.json"))
    print(out["summary"], flush=True)


if __name__ == "__main__":
    main()
