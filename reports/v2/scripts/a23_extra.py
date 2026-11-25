#!/usr/bin/env python3
"""Addendum 23, a sensitivity for section 6 of the report (the tile-edge ring): the Addendum 22 per-tile parameters of the two M&D-rule
split models and of the nothing-fitted model, scored on the domain (polygon misfit and whole-domain misfit), from one reduction per
network, beside the domain fits; and the count of fixed cells each way (tile: the 3,996 edge cells; domain: the outlets).
Outputs <run>/fits/tile_params_on_domain.json. usage: a23_extra.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a23_common import (  # noqa: E402
    TILES,
    U,
    a22_path,
    domain_profiler,
    dump,
    json_ready,
    load_domain,
    load_json,
    mse_at,
    out_path,
    reduce_mask,
    slurm_record,
    terms_of,
)
from a23_fit import keep_masks, networks_for_fit  # noqa: E402


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("fits", "x")))
    dom = load_domain()
    prof = domain_profiler(dom)
    keep = keep_masks(prof, dom)
    nets, ninfo = networks_for_fit(dom)
    fx = load_json(out_path("fits", "fixed.json"))
    out: dict[str, Any] = {"networks": ninfo, "fixed_cells": {"domain_outlets": int(dom["outlets"].sum()), "tile_edge_ring": 3996, "domain_cells": int(prof.n)}, "models": {}}
    for net, name in (("C", "PF_C"), ("A", "PF_A")):
        red = reduce_mask(prof, nets[net], dom["rec"]["measured"]["m"] if "m" in dom["rec"]["measured"] else 0.33)
        T = {k: terms_of(red, prof.Z_obs_flat, v) for k, v in keep.items()}
        rec: dict[str, Any] = {"domain_fit": {k: fx["models"][name][k] for k in ("K", "D", "mse_poly", "mse_all_at_poly_fit")}, "tile_params_on_domain": {}}
        for t in TILES:
            a22 = load_json(a22_path("tiles", f"{t}.json"))
            m = a22["models"][name]
            rec["tile_params_on_domain"][t] = {"K": m["K"], "D": m["D"], "tile_mse_poly": m["mse_poly"], "domain_mse_poly": mse_at(T["poly"], U, m["K"], m["D"]), "domain_mse_all": mse_at(T["all"], U, m["K"], m["D"])}
            if net == "C":
                f2 = a22["models"]["F2_relief"]
                rec.setdefault("F2_relief_tile_params_on_domain", {})[t] = {"K": f2["K"], "D": f2["D"], "tile_mse_poly": f2["mse_poly"], "domain_mse_poly": mse_at(T["poly"], U, f2["K"], f2["D"])}
        # the D that the domain prefers at the tile's K (K fixed), for the record
        out["models"][name] = rec
        print(f"{name}: " + "; ".join(f"{t}: tile (K {v['K']:.3e}, D {v['D']:.4f}) -> domain polygon misfit {v['domain_mse_poly']:.1f} (tile {v['tile_mse_poly']:.1f}; domain fit {rec['domain_fit']['mse_poly']:.1f})" for t, v in rec["tile_params_on_domain"].items()) + f" [{time.time() - t0:.0f} s]", flush=True)
        del red, T
    out["wall_s"] = time.time() - t0
    dump(json_ready(out), out_path("fits", "tile_params_on_domain.json"))


if __name__ == "__main__":
    main()
