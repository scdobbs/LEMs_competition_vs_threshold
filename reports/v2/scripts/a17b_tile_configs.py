#!/usr/bin/env python3
"""Addendum 17 Part B: tile configs for every registered tile that has none (71 of 81), and the TTV site config.

A config carries what `lemcvt.config.load_tile_config` requires: the three grids (relative to $LEMCVT_DATA) with their manifest
sha256, shape, cell size and CRS; units (metres); a `measured_v1` block holding the site's selection reference (the loader needs a
measured block; these are not measured tile values and are labelled so); the site's model block (bounds, A_crit grid, the v2 site m
as a placeholder exponent, K0/D0 expressions resolved from the reference) so that `tile run --stages m_profile` has starting values
(the exponent profile takes the exact box-constrained (K, D) optimum at every point: K0/D0 only set the reported stage-2 MSE).
The `measured` block (site m, k_s, C_ht with provenance) is for addendum 18 to write from this addendum's run.
Also reports every new tile's elevation range and median pixel slope against the site sanity ranges.
usage: a17b_tile_configs.py --out DIR [--data $LEMCVT_DATA] [--dry-run]
"""
import argparse
import os
import sys
import time

import numpy as np
import rasterio
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a17b_common import (  # noqa: E402
    E_M_PER_YR,
    MEASURED,
    REPO,
    SITE_REFERENCE_V1,
    V2_SITE_M,
    dump,
    site_of,
    slurm_record,
    tile_registry,
)

MODEL_TEMPLATE = {"GM": "GM1", "AP": "AP1", "TTV": "GM1"}     # metre bounds of the v2 configs (TTV: the GM block, generic wide bounds)
TTV_SITE = {"site": "TTV", "name": "Tennessee Valley (Marin County, CA)", "U": E_M_PER_YR["TTV"], "n": 1.0, "S_c": 1.27,
            "z_range_m": [0.0, 800.0], "max_median_slope": 1.0, "crs_epsg": 6339,
            "regional": {"source": "addendum 16 section 2.2: the clipped M&D (1989) polygon on the 10 m NED, the selection reference; E from Heimsath et al. (1997), 77 +- 15 m/Myr",
                         "theta": 0.442, "ks": 26.86, "hilltop_curvature": 1.79e-2, "U": E_M_PER_YR["TTV"]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=os.environ.get("LEMCVT_DATA"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slurm_record(a.out)
    t0 = time.time()
    reg = tile_registry()
    cdir = os.path.join(REPO, "configs", "tiles")
    sdir = os.path.join(REPO, "configs", "sites")
    sites = {s: yaml.safe_load(open(os.path.join(sdir, f"{s}.yaml"))) for s in ("GM", "AP")}
    ttv_path = os.path.join(sdir, "TTV.yaml")
    if not os.path.exists(ttv_path) and not a.dry_run:
        with open(ttv_path, "w") as fh:
            fh.write("# Tennessee Valley (Marin County, CA): Marin 2019 QL1 lidar gridded at 1 m on NAD83(2011) / UTM 10N (EPSG:6339), z in metres\n"
                     "# (NAVD88; addendum 14 rule 5.2). Written by reports/v2/scripts/a17b_tile_configs.py (addendum 17 Part B). The sanity range\n"
                     "# is the decision-1 unit window (0-800 m); max_median_slope 1.0 admits the Marin Headlands' steep tiles (a units error would\n"
                     "# give slopes 3.3 x larger; rule 5.2 of addendum 14 established the metre).\n")
            yaml.safe_dump(TTV_SITE, fh, sort_keys=False, width=120)
    sites["TTV"] = TTV_SITE
    templates = {s: yaml.safe_load(open(os.path.join(cdir, f"{t}.yaml")))["model"] for s, t in MODEL_TEMPLATE.items()}
    rec = {"written": [], "existing": [], "sanity": {}}
    for t, r in sorted(reg.items()):
        path = os.path.join(cdir, f"{t}.yaml")
        site = site_of(t)
        if os.path.exists(path):
            rec["existing"].append(t)
            continue
        if len(r["sha256"]) != 3:
            raise KeyError(f"{t}: manifest rows missing for {set(('filled', 'area', 'fd')) - set(r['sha256'])}")
        with rasterio.open(os.path.join(a.data, r["grids"]["filled"])) as ds:
            z = ds.read(1).astype(np.float64)
            z[z == ds.nodata] = np.nan
            shape, dx, dy = [ds.height, ds.width], float(ds.transform.a), float(-ds.transform.e)
        gy, gx = np.gradient(z, dy, dx)
        med_slope = float(np.nanmedian(np.hypot(gx, gy)))
        zr = sites[site]["z_range_m"]
        san = {"z_min": float(np.nanmin(z)), "z_max": float(np.nanmax(z)), "median_slope": med_slope,
               "in_site_range": bool(zr[0] <= np.nanmin(z) and np.nanmax(z) <= zr[1] and med_slope <= sites[site]["max_median_slope"])}
        rec["sanity"][t] = san
        model = dict(templates[site])
        model["m"] = V2_SITE_M.get(site, SITE_REFERENCE_V1["TTV"]["theta"])
        model.pop("D0_B", None)
        ref = SITE_REFERENCE_V1[site]
        cfg = {"tile": t, "site": site,
               "grids": {**r["grids"], "sha256": r["sha256"], "shape": shape, "dx": dx, "dy": dy, "crs_epsg": r["crs_epsg"]},
               "units": {"z_units": "m", "xy_units": "m", "reproduce_v1_units": False},
               "measured_v1": {"theta": ref["theta"], "ks": ref["ks"], "hilltop_curvature": ref["hilltop_curvature"], "units_note": "z in metres",
                               "source": ref["source"]},
               "model": model,
               "registration": {"version": r["version"], "survey": r["survey"], "registered": "addendum 17 Part A (TTV-x, GM-FEMA2018-x, GM-FEMA2018-ctrl) or addendum 15 stage B (GM-x, AP-3DEP-x)",
                                "area_source": "full-grid routing of the survey grid (role area)", "measured_in": "addendum 17 Part B" if t in MEASURED else "not measured (Tier 1 exclusion, go-ahead 1.1)"}}
        header = (f"# Tile {t} ({r['version']}, {r['survey']}): written by reports/v2/scripts/a17b_tile_configs.py (addendum 17 Part B).\n"
                  "# `measured_v1` holds the SITE SELECTION REFERENCE (not a v1 tile value): the loader needs a measured block and the m-profile\n"
                  "# stage needs starting K0 = U/ks and D0 = U/C_ht; the exponent profile takes the exact box-constrained (K, D) optimum at every\n"
                  "# (m, A_crit), so these values do not affect it. `model.m` is a placeholder (the v2 site m; the TTV reference theta).\n"
                  "# The `measured` block (site m from S(m), k_s, C_ht with provenance) is written by addendum 18 from the Part B run.\n")
        if not a.dry_run:
            with open(path, "w") as fh:
                fh.write(header)
                yaml.safe_dump(cfg, fh, sort_keys=False, width=120)
        rec["written"].append(t)
        print(t, site, r["version"], san, flush=True)
    rec["n_written"] = len(rec["written"])
    rec["n_existing"] = len(rec["existing"])
    rec["sanity_failures"] = [t for t, s in rec["sanity"].items() if not s["in_site_range"]]
    rec["seconds"] = time.time() - t0
    dump(rec, os.path.join(a.out, "tile_configs.json"))
    print("written", rec["n_written"], "existing", rec["n_existing"], "sanity failures", rec["sanity_failures"], flush=True)


if __name__ == "__main__":
    main()
