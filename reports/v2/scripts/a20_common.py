#!/usr/bin/env python3
"""Addendum 20 (Perron's hilltop estimator, the D bracket without the segment filter, snapped TTV heads, Part B): shared constants and
helpers for the a20_* scripts.

Directive `reports/directives/2026-11-20T1050PST_v2_addendum_20_perron_D0_heads_partB.md`. Decisions of its section 1 as constants, rules of
its section 6 as thresholds; all fixed before any result. Package code is not changed. The addendum 19 helpers are imported from
`a19_common` (whose `out_path` reads the addendum 19 run directory through A19_RUN); everything this addendum writes goes through
`a20_out_path` into the addendum 20 run directory (A20_RUN).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("A19_RUN", "20261120T160830Z_a19_hilltops_heads_slope")
from a19_common import (  # noqa: E402,F401
    A18,
    A18_SPECS,
    A18_SUMMARY,
    A_NET_M2,
    AP1_3DEP,
    AP_PAMAP,
    AP_PRIMARY,
    CLEAN,
    D0_MIN_PIXELS,
    E_SITE,
    E_STAR_BANDS,
    GM_NCALM,
    GM_OFFMESA,
    GM_PRIMARY,
    GROUPS,
    L_F_M,
    MD_EQ10,
    MD_GPKG,
    MD_TABLE1_MEDIAN_M2,
    MD_TILES,
    MEASURED_TILES,
    PERRON_2012,
    PRIMARY,
    RESULTS_ROOT,
    ROAD_BUFFER_M,
    SCRATCH,
    SEED,
    SLOPE_MAX,
    SURVEY_PAIRS,
    TTV_ANCHOR,
    TTV_FULL,
    TTV_PRIMARY,
    W_F,
    a18_specs,
    channel_heads,
    dump,
    fnum,
    hillshade,
    hold_n,
    load_json,
    md_table1,
    metre_tile,
    ols_loglog_bootstrap,
    pct,
    ratio_interval,
    segment_bootstrap_median,
    segments_of,
    sha256_file,
    site_of,
    slurm_record,
    tile_bounds,
    trace_to_mask,
)
from a19_common import out_path as a19_out_path  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DIRECTIVE = "reports/directives/2026-11-20T1050PST_v2_addendum_20_perron_D0_heads_partB.md"
A19_DIR = os.path.join(RESULTS_ROOT, os.environ["A19_RUN"])
A19_REPORT = "reports/v2/20261120T1030PST_hilltops_heads_slope.md"
FIG_DIR = os.path.join(REPO, "reports", "v2", "figures", "a20")
JOB_DIR = os.path.join(SCRATCH, "lemcvt_jobs", "a20")
HEADS_CSV = os.path.join(REPO, "data", "ttv", "ttv_heads_manual_steve_20261119.csv")
HEADS_SHA256 = "ac87de3d4ca1f479e6eb2ceda2fedb033740b3d8f37186fd267329d5e4e702b6"

# ----------------------------------------------------------------------------- decision 1 (D0_N without the segment filter)
FRAGMENT_SENS_PX = 20                 # the 20-pixel fragment filter, reported beside D0_N
D0N_HALF_WIDTHS = (1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25)     # decision 4: L = 3 ... 51 m (the addendum 19 test and scan windows up to 51 m)
D0N_ALL_HALF_WIDTHS = tuple(sorted(set(D0N_HALF_WIDTHS) | set(range(1, 16))))
N_BOOT = 2000

# ----------------------------------------------------------------------------- decision 2 (Perron's estimator)
PERRON_RADII_M = (3.0, 5.0, 7.0, 10.0, 15.0)
PERRON_R_PRIMARY = 7.0
BIN_EDGES = 10.0 ** np.arange(-3.0, 5.0 + 1e-9, 0.25)        # 4 per decade, 1e-3 ... 1e5 m^2 (33 edges, 32 bins)
PLATEAU_MIN_CELLS = 200
PLATEAU_MIN_FRAC_CONVEX = 0.6
PLATEAU_TOL = 0.20
PLATEAU_MIN_BINS = 3
FIXED_RANGES_M2 = (1.0, 3.0)
BLOCK_M = 100.0
LAP_ZERO = 1e-9                 # m^-1: |curvature| below this is a numerical zero (FFT rounding on exactly planar, e.g. pit-filled, cells; verified error <= 1.6e-13), not convex
N_BOOT_BLOCK = 2000
DINF_MAX_CORE_H = 20.0
PERRON_2009_SI = "Perron, Kirchner & Dietrich (2009), Nature 460, Supplementary Information section 2.1"

# ----------------------------------------------------------------------------- decision 5 (snapped heads)
PATH_DISTANCES_M = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
A_P_PRIMARY_M = 10.0
AXIS_ENTRY_FACTOR = 10.0
QC_BOX_M = 40.0
QC_PER_PAGE = 16
QC_A_MIN_M2 = 100.0

# ----------------------------------------------------------------------------- decision 6 (Part B)
PF_VALUES = ("A_F", "A_F_table1", "A_F_A10", "A_F_A3")        # A_P10 median, 7,400, 2,848 (A_10), 611 (A_3)
A19_A10_MEDIAN_M2 = 2848.0
A19_A3_MEDIAN_M2 = 611.0
ALPHAS = (0.0, 1.65, 2.0)
ALPHA_BHAT_MIN_SEP = 0.1
RECON_TOL = 1e-8

# ----------------------------------------------------------------------------- rules (section 6)
R1_MIN_FRAC_TILES = 0.90
R13_MIN_FRAC_TILES = 0.90
R11_RANGE = (0.67, 1.5)
R5_RATIO_RANGE = (0.5, 2.0)
R6_MIN = 9
R7_IMPROVEMENT, R7_MIN = 0.02, 9
R8_MEDIAN_DIST_M, R8_RATIO_RANGE = 25.0, (0.67, 1.5)
R10_FRAC = 0.80
R14_FRAC = 0.80
F4_LOG10 = 1.0


def a20_run_dir() -> str:
    rid = os.environ.get("A20_RUN")
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a20_perron_D0_heads_partB"
        os.environ["A20_RUN"] = rid
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def out_path(*parts: str) -> str:
    """A path in the addendum 20 run directory."""
    p = os.path.join(a20_run_dir(), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def a19_path(*parts: str) -> str:
    return os.path.join(A19_DIR, *parts)


def hms(h: float) -> str:
    s = int(round(h * 3600))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


_ = a19_out_path
