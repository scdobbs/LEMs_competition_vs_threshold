"""Addendum 17 Part B (per-tile measurement): shared constants for the a17b_* scripts.

Go-ahead `reports/directives/2026-11-18T1630PST_v2_addendum_17_partB_goahead.md` (Steve, 2026-11-18, after the Part A figures and
before any measurement): Tier 1 exclusions (not measured, except AP1_3dep for the survey comparison), Tier 2 (kept; a clean-subset
sensitivity), the FEMA 2018 survey as the primary GM survey (GM1_fema-GM7_fema + GM8-GM30; NCALM GM1-GM7 continuity only), two
screens written before the primary set is fixed (bedrock >= 90 %; gap-filled measurement pixels <= 20 %), the expected primary set.
Measurement methods: docs/measurement_methods.md section 0 (unchanged), through the lemcvt functions of measure/phase6_inputs.py,
measure/audits.py and analyses/exponent_site.py. E per site: a17_common.E_M_PER_YR (decision 6 of addendum 17).
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import GROUP_HOME, RESULTS_ROOT, SCRATCH, SOURCE, dump, sha256_file, slurm_record  # noqa: E402,F401
from a17_common import A16, E_M_PER_YR, E_SOURCE, FIG_DIR, REPO, SURVEY  # noqa: E402,F401

A17A = os.path.join(RESULTS_ROOT, "20261118T225314Z_a17_register_figures")
GEOLOGY_DIR = os.path.join(SOURCE, "Geology_Monterey_Rosenberg_cm427jp1187")       # copy of record (PURL cm427jp1187, public domain)
GEOLOGY_SHP = os.path.join(GEOLOGY_DIR, "GEOLOGY.shp")
GEOLOGY_PURL = "https://purl.stanford.edu/cm427jp1187"

# ---------------------------------------------------------------- go-ahead section 1: tiers
TIER1 = {"GM10": "benched or hummocky ground over the whole tile (terraced ground or an old landslide complex)",
         "GM23": "a large flat area in the middle of the tile (alluvial fill or a mesa remnant), not hillslope-and-channel topography",
         "AP1_3dep": "the main channel follows a road",
         "AP12": "a flat, apparently man-made surface (pad or fill)",
         "AP15": "the channel network follows a road",
         "AP16": "a large graded pad and access roads on the ridge",
         "AP17": "a divided highway with cut and embankment across about a third of the tile",
         "AP21": "man-made flat tops (pads) on the ridges",
         "AP22": "a pad and a rectangular road grid; the network follows man-made features",
         "AP26": "a man-made flat top (pad) and its access road",
         "TTV10": "the network follows roads",
         "TTV12": "graded ground and switchback roads (Marin Headlands batteries); the network follows roads",
         "TTV13": "roads and lot-scale terracing (the urban edge)",
         "TTV14": "the network follows switchback roads"}
TIER2 = {"GM12": "a road along the divide", "GM29": "a road along the divide", "GM26": "switchback roads and terracing in the lower left",
         "AP25": "small pads on the ridges", "TTV11": "fire roads"}
# ---------------------------------------------------------------- go-ahead section 2 and 4: the sets
GM_FEMA_PRIMARY = [f"GM{n}_fema" for n in range(1, 8)] + [f"GM{n}" for n in range(8, 31) if f"GM{n}" not in TIER1]
AP_PRIMARY = [f"AP{n}_3dep" for n in range(2, 6)] + [f"AP{n}" for n in range(6, 30) if f"AP{n}" not in TIER1]
TTV_PRIMARY = [f"TTV{n}" for n in range(1, 16) if f"TTV{n}" not in TIER1]
PRIMARY_EXPECTED = {"GM": GM_FEMA_PRIMARY, "AP": AP_PRIMARY, "TTV": TTV_PRIMARY}          # before the screens of section 3
GM_NCALM = [f"GM{n}" for n in range(1, 8)]                                                  # continuity only (GM1-GM5 area_model = old grids, the existing configs)
AP_PAMAP = [f"AP{n}" for n in range(1, 6)]                                                  # rule 6.3, the existing configs (feet -> metres at load)
CONTINUITY = GM_NCALM + ["AP1_3dep"]
MEASURED = GM_FEMA_PRIMARY + AP_PRIMARY + TTV_PRIMARY + CONTINUITY + AP_PAMAP                # 73 tiles
M_PROFILE_NEW = GM_FEMA_PRIMARY + AP_PRIMARY + TTV_PRIMARY + ["GM6", "GM7", "AP1_3dep"]     # 63 new m-profile runs (GM1-GM5, AP1-AP5 have the addendum 4 runs)
V2_SITE_M = {"GM": 0.31, "AP": 0.47}                                                        # rule 6.2 reference (AP on PAMAP)
TRUNK_LIST_A18 = ["GM2_fema", "GM4_fema", "GM5_fema", "AP3_3dep", "AP4_3dep"]              # go-ahead 2.5
# ---------------------------------------------------------------- go-ahead section 3: the screens (written before any result)
BEDROCK_PASS_FRAC = 0.90
GAPFILL_MAX_FRAC = 0.20
# Quaternary surficial units of the map (PERIOD == "Quaternary"): counted with the reference unit when they are alluvium, flood-plain,
# stream-channel, alluvial-fan, fluvial-terrace, colluvium or basin deposits (by the map's general label GEN_LAB); not counted:
# landslide deposits (Qls), artificial fill (Qfl), eolian, dune, beach, coastal-terrace and Aromas Sand units, Qscp. Queried labels
# ("Qal?") count with their general label. "Quaternary-Tertiary" units (QTp, the Paso Robles Formation) are bedrock for this rule.
QUATERNARY_COUNTED_GEN_LAB = {"Qal", "Qfp", "Qsc", "Qhf", "Qfpl", "Qfpm", "Qfu", "Qt", "Qtmp", "Qtlp", "Qc", "Qb"}
QUATERNARY_LANDSLIDE_GEN_LAB = {"Qls"}
# ---------------------------------------------------------------- measurement settings (section 5; measurement_methods.md section 0)
A_C_KS = 1e3                      # k_s network (used for K0); 1e4 reported
A_C_THETA = (1e3, 1e4)            # site-pooled theta
L_F_M = 11.0
WINDOW_TEST_PASS_FRAC = 0.8       # rule 6.1: addendum 6 section 1 "at least 4 of 5 tiles" read as >= 80 % of the site's tiles (set before the results; the counts are reported too)
RULE_64_MEDIAN_TOL = 0.15
RULE_64_SINGLE_TOL = 0.30
RULE_62_TOL = 0.05
SITE_REFERENCE_V1 = {           # `measured_v1` for the new tile configs: the site selection references (addendum 14 decision 6; TTV addendum 16 section 2.2)
    "GM": {"theta": 0.292, "ks": 1.9, "hilltop_curvature": 1.28e-2, "source": "legacy TAYLOR_REF (addendum 14, a14_common.LEGACY_REF): the site selection reference, not a v1 tile value"},
    "AP": {"theta": 0.54, "ks": 47.2, "hilltop_curvature": 5.93e-3, "source": "legacy TAYLOR_REF (addendum 14, a14_common.LEGACY_REF): the site selection reference, not a v1 tile value"},
    "TTV": {"theta": 0.442, "ks": 26.86, "hilltop_curvature": 1.79e-2, "source": "the clipped M&D polygon on the 10 m NED (addendum 16 section 2.2): the site selection reference"},
}


def run_dir(name: str | None = None) -> str:
    rid = os.environ.get("A17B_RUN") or name
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a17b_measurement"
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def tile_registry() -> dict[str, dict]:
    """Every registered tile: version, site, survey, grid paths (relative to $LEMCVT_DATA) and sha256 from data/manifest.csv."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "data.yaml")))
    man = {r["path"]: r for r in csv.DictReader(open(os.path.join(REPO, "data", "manifest.csv")))}
    out = {}
    for v, blk in cfg.get("versions", {}).items():
        site = blk.get("site_label", blk["site"])
        for t in blk["tiles"]:
            g = {k: f"{v}/{t}/{t}_1m_{k}.tif" for k in ("filled", "area", "fd")}
            masks = {k: f"{v}/{t}/{t}_1m_{k}.tif" for k in ("nnfill_mask", "gapfill_mask", "class2_count")}
            out[t] = {"tile": t, "version": v, "site": site, "survey": SURVEY.get(v, v), "crs_epsg": blk["crs_epsg"], "grids": g,
                      "sha256": {k: man[p]["sha256"] for k, p in g.items() if p in man}, "masks": {k: p for k, p in masks.items() if p in man},
                      "area_model": (f"{v}/{t}/{t}_1m_area_model.tif" if f"{v}/{t}/{t}_1m_area_model.tif" in man else None)}
    return out


def site_of(tile: str) -> str:
    if tile.startswith("TTV"):
        return "TTV"
    return tile[:2]
