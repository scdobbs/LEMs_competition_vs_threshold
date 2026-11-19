"""Addendum 18 (model fits of the expanded tile set): shared constants for the a18_* scripts.

Directive `reports/directives/2026-11-19T0615PST_v2_addendum_18_expanded_fits.md`. Decisions of section 1 (planner, 2026-11-19,
under Steve's delegation), fixed before any fit:
  1. bedrock screen read at formation level (QTp + QTcl + QTg + QTm + QTpc): GM8, GM9, GM21 return; GM primary = 22, clean = 19;
     off-mesa GM14, GM18, GM19, GM25, GM28, GM30; the GM site m stays 0.29;
  2. D0 carried at both ends (D0_A at L_f = 11 m, D0_B at L_w) everywhere; NCALM-twin D0 as a labelled sensitivity on GM1_fema-GM7_fema;
  3. the same-pixel curvature diagnostic runs in Part A (report only);
  4. tiles whose option (iii) set has fewer than 500 pixels carry no D0 (left out of every D statement; P1, P2, A2 not run on them);
  5. TTV fitted at the site m = 0.33, plus a per-tile m sensitivity (six profiles at the grid edge m = 0.05 fitted at 0.05 and flagged);
  6. site m for the fits: GM 0.29, AP 0.47, TTV 0.33 (every group of a site; the regression gate excepted);
  7. E (m/yr): GM 1.47e-4, AP 5.5e-5, TTV 7.7e-5;
  8. trunk cells = A >= 1 km2 on the tile's area grid, held at the observed elevation and left out of the MSE, on every primary or
     continuity tile whose largest A exceeds 5 km2;
  9. the `measured` blocks of the new tile configs are written from the Part B run;
  11. budget <= 900 core-hours.
Rules of section 6 (80 % of N, rounded up; lopsided split < 30 %; trunk material at 25 %; m immaterial at 25 %).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import GROUP_HOME, RESULTS_ROOT, SCRATCH, SOURCE, dump, sha256_file, slurm_record  # noqa: E402,F401
from a17_common import E_M_PER_YR, HILLSHADE, SURVEY, hillshade  # noqa: E402,F401
from a17b_common import GEOLOGY_SHP, TIER1, TIER2, site_of, tile_registry  # noqa: E402,F401

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DIRECTIVE = "reports/directives/2026-11-19T0615PST_v2_addendum_18_expanded_fits.md"
PARTB = os.path.join(RESULTS_ROOT, "20261119T002830Z_a17b_measurement")           # addendum 17 Part B run directory
PARTB_MEASUREMENT = os.path.join(PARTB, "measure", "measurement.json")
PARTB_PER_TILE = os.path.join(PARTB, "measure", "per_tile.csv")
PARTB_REPORT = "reports/v2/20261118T2045PST_measurement.md"
A15 = os.path.join(RESULTS_ROOT, "20261118T160326Z_a15_stage_b")
A16 = os.path.join(RESULTS_ROOT, "20261118T185206Z_a16_tile_finish")
MD_HEADS_GPKG = os.path.join(A15, "md_georef", "md1989_fig3.gpkg")                 # the 77 digitised heads (layer channel_heads)
MD_TABLE1 = os.path.join(A16, "md_georef", "md1989_table1.csv")                     # the 63 source areas of M&D (1989) Table 1
TTV_FULL_AREA = os.path.join(A15, "grid_TTV", "TTV_1m_area_cells.tif")              # Marin 2019 full-grid routing (cells at 1 m = m2)
FIG_DIR = os.path.join(REPO, "reports", "v2", "figures", "a18")
QC_DIR = os.path.join(FIG_DIR, "hilltop_qc")
V2_BUILD = os.path.join(RESULTS_ROOT, "v2.0.0")
JOB_DIR = os.path.join(SCRATCH, "lemcvt_jobs", "a18")

# ---------------------------------------------------------------- decisions
SITE_M = {"GM": 0.29, "AP": 0.47, "TTV": 0.33}                                      # decision 6
V2_SITE_M = {"GM": 0.31, "AP": 0.47}                                                 # the v2.0.0 settings (the 4.1 gate, 5.5)
D0_MIN_PIXELS = 500                                                                  # decision 4
TRUNK_A_M2 = 1.0e6                                                                   # decision 8: trunk cells A >= 1 km2
TRUNK_LIST_MIN_A_M2 = 5.0e6                                                          # decision 8: trunk fit where max A > 5 km2
TRUNK_EXPECTED = ["GM2_fema", "GM4_fema", "GM5_fema", "AP3_3dep", "AP4_3dep"]
TTV_M_T = {"TTV1": 0.05, "TTV2": 0.45, "TTV3": 0.40, "TTV4": 0.42, "TTV5": 0.41, "TTV6": 0.05, "TTV7": 0.05, "TTV8": 0.06, "TTV9": 0.35,
           "TTV11": 0.05, "TTV15": 0.05}                                             # decision 5 (Part B Appendix B5)
TTV_M_T_EDGE = ["TTV1", "TTV6", "TTV7", "TTV8", "TTV11", "TTV15"]                     # profile optimum at the grid edge m = 0.05: flagged
M_GRID_EDGE = 0.05
BUDGET_CORE_HOURS = 900.0                                                            # decision 11 / 4.2
MAX_CONCURRENT_CPUS = 300
MAX_SUBMISSION_LIMIT_CORE_HOURS = 300.0
GEOLOGY_FORMATION_LABELS = ["QTp", "QTcl", "QTg", "QTm", "QTpc"]                     # decision 1: the Paso Robles Formation at formation level
# ---------------------------------------------------------------- groups (section 2)
GM_PRIMARY = [f"GM{n}_fema" for n in range(1, 8)] + ["GM8", "GM9", "GM11", "GM12", "GM13", "GM15", "GM16", "GM17", "GM20", "GM21", "GM22", "GM24",
                                                     "GM26", "GM27", "GM29"]
GM_CLEAN = [t for t in GM_PRIMARY if t not in ("GM12", "GM26", "GM29")]
GM_OFFMESA = ["GM14", "GM18", "GM19", "GM25", "GM28", "GM30"]
GM_NCALM = [f"GM{n}" for n in range(1, 8)]
AP_PRIMARY = [f"AP{n}_3dep" for n in range(2, 6)] + ["AP6", "AP7", "AP8", "AP9", "AP10", "AP11", "AP13", "AP14", "AP18", "AP19", "AP20", "AP23",
                                                      "AP24", "AP25", "AP27", "AP28", "AP29"]
AP_CLEAN = [t for t in AP_PRIMARY if t != "AP25"]
AP_PAMAP = [f"AP{n}" for n in range(1, 6)]
AP1_3DEP = ["AP1_3dep"]
TTV_PRIMARY = ["TTV1", "TTV2", "TTV3", "TTV4", "TTV5", "TTV6", "TTV7", "TTV8", "TTV9", "TTV11", "TTV15"]
TTV_CLEAN = [t for t in TTV_PRIMARY if t != "TTV11"]
PRIMARY = {"GM": GM_PRIMARY, "AP": AP_PRIMARY, "TTV": TTV_PRIMARY}
CLEAN = {"GM": GM_CLEAN, "AP": AP_CLEAN, "TTV": TTV_CLEAN}
GROUPS = {("GM", "primary"): GM_PRIMARY, ("GM", "clean"): GM_CLEAN, ("GM", "offmesa"): GM_OFFMESA, ("GM", "ncalm"): GM_NCALM,
          ("AP", "primary"): AP_PRIMARY, ("AP", "clean"): AP_CLEAN, ("AP", "pamap"): AP_PAMAP, ("AP", "ap1_3dep"): AP1_3DEP,
          ("TTV", "primary"): TTV_PRIMARY, ("TTV", "clean"): TTV_CLEAN}
CONTINUITY = GM_NCALM + AP_PAMAP + AP1_3DEP
FITTED_AT_SITE_M = GM_PRIMARY + GM_OFFMESA + GM_NCALM + AP_PRIMARY + AP_PAMAP + AP1_3DEP + TTV_PRIMARY        # every tile fitted at its site m
NO_D0 = ["AP2_3dep", "AP9", "AP10", "AP11", "AP14", "AP18", "AP27", "AP28", "TTV3", "TTV4"]                 # decision 4 (Part B counts)
OLD_CONFIG_TILES = [f"GM{n}" for n in range(1, 6)] + AP_PAMAP                       # v2.0.0 measured blocks: not rewritten
SURVEY_PAIRS = {"GM": [(f"GM{n}", f"GM{n}_fema") for n in range(1, 8)], "AP": [(f"AP{n}", f"AP{n}_3dep") for n in range(1, 6)]}   # (old, new)
# ---------------------------------------------------------------- the fits (section 5) and the rules (section 6)
ACRIT_RANGE = (1e2, 1e5)
THIN_N = 2000
FILL_LEVEL = 0.01
FILL_MAX = 1500
REFINE_MAX = 1000
AD_N_GRID = 200
HOLDOUT_A_C = 1e3
HOLD_FRAC = 0.80                      # a rule holds on >= 80 % of N, rounded up
LOPSIDED_FRAC = 0.30                  # 5.3: smaller half < 30 % of the cells
TRUNK_MATERIAL_REL = 0.25             # 6.8
M_IMMATERIAL_REL = 0.25               # 6.9
F4_LOG10 = 1.0                        # 6.4: AD median |log10(D/D0)| >= 1 at both ends
CHANNEL_HEAD_SEARCH_M = 3.0           # 5.7: the largest A within 3 m of a digitised head
GATE_REL_TOL = 1e-9                   # 4.1
SAME_PIXEL_TOL = 1e-3                 # 3.1: (a) and (d) within 0.1 % of Part B
SAME_PIXEL_RULES = {"surface": {"S_max": 0.85, "P_range": (0.90, 1.10)}, "selection": {"S_range": (0.90, 1.10), "P_max": 0.85}}
L_F_M = 11.0
W_F = 5
FIT_STEPS = {"part": "a18_part", "ad": "a18_ad"}


def hold_n(n: int) -> int:
    """The count a rule needs on N tiles: 80 % rounded up."""
    return int(math.ceil(HOLD_FRAC * n - 1e-9)) if n > 0 else 0


def run_dir(name: str | None = None) -> str:
    rid = os.environ.get("A18_RUN") or name
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a18_expanded_fits"
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def partb_measurement() -> dict:
    return json.load(open(PARTB_MEASUREMENT))


def partb_per_tile() -> dict[str, dict]:
    out = {}
    with open(PARTB_PER_TILE) as f:
        for r in csv.DictReader(f):
            out[r["tile"]] = r
    return out


def fnum(v) -> float | None:
    if v is None or v == "" or v == "None":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def group_of(tile: str) -> str:
    """The fit group a tile belongs to at its site m (section 2)."""
    for (_site, g), tiles in GROUPS.items():
        if g in ("clean",):
            continue
        if tile in tiles:
            return g
    raise KeyError(tile)


def spec_key(tile: str, group: str) -> str:
    return f"{tile}:{group}"
