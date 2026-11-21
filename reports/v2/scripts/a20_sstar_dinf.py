#!/usr/bin/env python3
"""Addendum 20 supplement, T3 variant: Perron's S* regression with D-infinity area (Tarboton 1997, the package's routine) in place of D8,
on the NCALM Gabilan Mesa tiles (Perron's survey; check against his D/K = 124, m = 0.35) and on TTV1-TTV3 inside the M&D polygon.
All other choices as `a20_ttv_cal.sstar_fit` (7 m disk, mean of log S* in 0.1-decade bins, A ranges 1e1/1e2/1e3 to 1e5 m2; all cells and
the D8 stream-profile cells). Writes <run>/ttv_cal/sstar_dinf.json.
usage: a20_sstar_dinf.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a19_hilltops import md_polygon_mask  # noqa: E402
from a20_common import GM_NCALM, dump, load_json, metre_tile, out_path, slurm_record  # noqa: E402
from a20_perron import block_index, disk_quadratic  # noqa: E402
from a20_ttv_cal import PERRON_GM, SITE_M, profile_cells, sstar_fit  # noqa: E402

from lemcvt.hydro.dinf import compute_dinf_area, compute_dinf_flow_direction  # noqa: E402


def stack(tiles, poly_fn=None):
    parts = {k: [] for k in ("Ad", "Ad_prof", "A8", "g", "l", "b")}
    off = 0
    for t in tiles:
        td = metre_tile(t)
        q = disk_quadratic(td.Z, 7.0, td.dx, td.filled.dy)
        ang, rf, fi = compute_dinf_flow_direction(td.Z, td.dx, FD_d8=td.FD)
        Ad = compute_dinf_area(td.Z.astype(float), td.dx, ang, rf, fi)
        bl, _ = block_index(td.Z.shape, td.dx, td.filled.dy)
        keep = np.ones(td.Z.shape, dtype=bool) if poly_fn is None else poly_fn(td)
        prof = profile_cells(td.A, td.FD)
        parts["Ad"].append(np.where(keep, Ad, np.nan).ravel())
        parts["Ad_prof"].append(np.where(keep & prof, Ad, np.nan).ravel())
        parts["A8"].append(np.where(keep, td.A, np.nan).ravel())
        parts["g"].append(q["grad"].ravel())
        parts["l"].append(q["lap"].ravel())
        parts["b"].append(bl.ravel() + off)
        off += int(bl.max()) + 1
        print(f"{t}: D-infinity done", flush=True)
    return {k: np.concatenate(v) for k, v in parts.items()}


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("ttv_cal", "sstar_dinf.json")))
    cal = load_json(out_path("ttv_cal", "ttv_cal.json"))
    out: dict = {"published_gm": PERRON_GM, "gm": {}, "ttv": {}}
    S = stack(GM_NCALM)
    gm_lh = {"perron_lap_h": PERRON_GM["lap_h"], "our_crest_disk7": cal["T3"]["gm_check"]["our_crest_disk7_lap_h"]}
    for tag, lh in gm_lh.items():
        for rng_ in ((10.0, 1e5), (1e2, 1e5), (1e3, 1e5)):
            for cells, A in (("all", S["Ad"]), ("profiles", S["Ad_prof"])):
                k = f"{tag}:A{rng_[0]:g}-{rng_[1]:g}:{cells}"
                out["gm"][k] = sstar_fit(A, S["g"], S["l"], lh, S["b"], rng_)
                print(f"GM D-inf {k}: D/K {out['gm'][k].get('D_over_K')} m {out['gm'][k].get('m')} (Perron 124, 0.35) [{time.time() - t0:.0f} s]", flush=True)
    T = stack(["TTV1", "TTV2", "TTV3"], poly_fn=md_polygon_mask)
    lh = cal["T3"]["lap_h_crest"]["disk7_median"]
    for rng_ in ((10.0, 1e5), (1e2, 1e5), (1e3, 1e5)):
        for cells, A in (("all", T["Ad"]), ("profiles", T["Ad_prof"])):
            k = f"lap_h_disk7:A{rng_[0]:g}-{rng_[1]:g}:{cells}"
            out["ttv"][k] = sstar_fit(A, T["g"], T["l"], lh, T["b"], rng_)
            out["ttv"][k + ":m0.33"] = sstar_fit(A, T["g"], T["l"], lh, T["b"], rng_, m_fixed=SITE_M)
            print(f"TTV D-inf {k}: D/K {out['ttv'][k].get('D_over_K')} m {out['ttv'][k].get('m')} [{time.time() - t0:.0f} s]", flush=True)
    out["wall_s"] = time.time() - t0
    dump(out, out_path("ttv_cal", "sstar_dinf.json"))


if __name__ == "__main__":
    main()
