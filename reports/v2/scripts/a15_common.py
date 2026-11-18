"""Addendum 15 (tile expansion, stage B): shared constants for the a15_* scripts.

Decisions of 2026-11-18 (addendum 15, section 1), fixed before any result:
  screens = threshold set B (reports/v2/scripts/a14_thresholds_B.json); theta dropped from the selection metrics;
  k_s and |C_ht| within +-25 % of the site reference, widened once to +-50 % at a site with fewer than 25 eligible
  windows (TTV: fewer than 30), counted after the decision-8 gap exclusion and before the draw;
  references GM/AP = legacy TAYLOR_REF, TTV = the georeferenced M&D polygon on the 10 m NED (a15_ttv_reference.py);
  draw seed 20261118; targets GM/AP 25 new, TTV 30 (M&D windows pre-placed).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import GROUP_HOME, LEGACY_REF, RESULTS_ROOT, SCRATCH, SOURCE, dump, sha256_file, slurm_record, write_tif  # noqa: E402,F401

STAGE_A = os.path.join(RESULTS_ROOT, "20261118T002225Z_a14_stage_a")
SEED = 20261118
THRESHOLDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a14_thresholds_B.json")
TOL_REL = 0.25
TOL_REL_WIDE = 0.50
MIN_ELIGIBLE = {"GM": 25, "AP": 25, "TTV": 30}
TARGET_NEW = {"GM": 25, "AP": 25, "TTV": 30}
A_NET_1M = 1000.0          # channel network on the 1 m grid for the decision-8 gap exclusion (m^2 = cells at 1 m)
TTV_E = {"E_m_per_Myr": 77.0, "sd": 15.0, "range": [64.0, 102.0], "source": "mean of Heimsath et al. (1997) Creek1a 64, Creek1b 66, Creek2 102"}
TILE_EPSG = {"GM": 32610, "AP": 6346, "TTV": 6339}
Z_RANGE = {"GM": (100.0, 500.0), "AP": (230.0, 490.0), "TTV": (0.0, 320.0)}   # section 6.4 unit check (metres)
NODATA = -9999.0
FIG3 = {"paper": "Montgomery & Dietrich (1989), WRR 25(8), 1907-1918", "pdf_page": 3, "page_pixels": [2455, 3379], "dpi": 300}


def run_dir(name: str | None = None) -> str:
    rid = os.environ.get("A15_RUN") or name
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a15_stage_b"
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None
