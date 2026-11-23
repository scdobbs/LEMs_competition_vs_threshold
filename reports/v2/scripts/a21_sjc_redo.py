#!/usr/bin/env python3
"""Addendum 21: redo the decision 1.4 extraction of the SJ-C surface with the corrected spur pruning of the Perron sensitivity (the SJ-C
jobs had imported the earlier code). The surface is the stored residual plus the observed grid (float32, ~1e-5 m); the criterion sets are
recomputed too and must equal the stored ones (checked). usage: a21_sjc_redo.py --tile TTV2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import (  # noqa: E402
    dump,
    extract_and_measure,
    json_ready,
    load_json,
    md_lines_mask,
    md_polygon_mask,
    metre_tile,
    out_path,
    picks_on_tile,
    slurm_record,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    a = p.parse_args()
    tile = a.tile
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("sjc", f"{tile}.json")))
    rec = load_json(out_path("sjc", f"{tile}.json"))
    td = metre_tile(tile)
    Z, dx = td.Z, td.dx
    Zs = Z + np.load(out_path("sjc", f"{tile}_residual.npz"))["SJ_C"].astype(float)
    fv = load_json(out_path("grid12", "grid12.json"))["field_values"]["12m"]
    poly = md_polygon_mask(td)
    lines = md_lines_mask(td) if poly.any() else np.zeros(Z.shape, dtype=bool)
    pick_rc, pick_ids = picks_on_tile(td)
    e_rec, arrs = extract_and_measure(Zs, dx, fv, pick_rc, pick_ids, poly, lines)
    old = rec["extraction"]
    same = all(old[k]["n_channel_12m"] == e_rec[k]["n_channel_12m"] and old[k]["n_heads_12m"] == e_rec[k]["n_heads_12m"] for k in ("a0", "a2"))
    rec["extraction_perron_old"] = old.get("perron")
    old_hz = dict(np.load(out_path("sjc", f"{tile}_heads.npz")))
    for k in ("a0", "a2"):
        e_rec[k] = old[k]                       # the stored criterion sets stand (same surface to float32; the counts are checked equal)
    e_rec["n_holes_1m"] = old.get("n_holes_1m")
    e_rec["residual"] = old.get("residual")
    e_rec["criterion_counts_equal_after_float32_roundtrip"] = bool(same)
    rec["extraction"] = e_rec
    rec["perron_redone"] = {"wall_s": time.time() - t0, "note": "spur pruning by crossing number (the first run pruned at staircase corners)"}
    for k in ("perron_skeleton", "perron_heads_rc"):
        old_hz[f"SJ_C_{k}"] = arrs[k]
    np.savez_compressed(out_path("sjc", f"{tile}_heads.npz"), **old_hz)
    dump(json_ready(rec), out_path("sjc", f"{tile}.json"))
    print(f"{tile}: criterion counts equal {same}; perron heads {old.get('perron', {}).get('n_heads')} -> {e_rec['perron']['n_heads']} [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
