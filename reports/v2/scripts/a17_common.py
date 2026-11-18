"""Addendum 17 (register the tile set, tile figures, per-tile measurement): shared constants for the a17_* scripts.

Decisions of 2026-11-18 (addendum 17, section 1), fixed before any result:
  1. The unit-check windows were a units test, not a selection rule: Z_RANGE (a15_common) = TTV 0-800 m, GM-FEMA 0-750 m
     (and the same 0-750 m window for the FEMA control grid GMC); the parked TTV and GM-extension tiles are registered as
     cut, no re-draw.
  2. GM1-GM5 modelling area = the old tile area grids (role area_model); the addendum 16 rule grids stay beside them as
     role area_model_rule, unused.
  3. The GM landscape polygon of addendum 16 (a16_common.GM_LANDSCAPE_POLY_UTM) is accepted.
  4. GM survey control: FEMA 2018 versions of GM1-GM7 (GM<n>_fema) at the NCALM footprints on the FEMA lattice, registered
     as version GM-FEMA2018-ctrl.
  5. Visual review before measurement (Part A figures); exclusions are Steve's, recorded in the Part B report.
  6. Erosion rates E (m/yr): GM 1.47e-4 (Perron et al. 2012, every GM tile), AP 5.5e-5, TTV 7.7e-5 (Heimsath et al. 1997).
  7. Addendum 18: trunk sensitivity fit for every tile whose largest drainage area inside exceeds 5 km2.
  8. Budget: Part A <= 20 core-hours, Part B <= 150 core-hours.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import GROUP_HOME, RESULTS_ROOT, SCRATCH, SOURCE, dump, sha256_file, slurm_record, write_tif  # noqa: E402,F401
from a15_common import NODATA, STAGE_A, TILE_EPSG, Z_RANGE, rss_gb  # noqa: E402,F401
from a16_common import GM_GATE_CROP_UTM, GM_GATE_SURVEYS, STAGE_B  # noqa: E402,F401

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
A16 = os.path.join(RESULTS_ROOT, "20261118T185206Z_a16_tile_finish")
FIG_DIR = os.path.join(REPO, "reports", "v2", "figures", "a17")
Z_RANGE_A15 = {"TTV": (0.0, 320.0), "GMF": (68.0, 468.0)}        # the addendum 15/16 windows, for the record (decision 1 replaced them)
E_M_PER_YR = {"GM": 1.47e-4, "AP": 5.5e-5, "TTV": 7.7e-5}       # decision 6; carried over to every tile of the site
E_SOURCE = {"GM": "Perron et al. (2012), applied to every GM tile (NCALM 2003 and FEMA 2018)", "AP": "5.5e-5 m/yr (the v1/v2 value)",
            "TTV": "77 +- 15 m/Myr, Heimsath et al. (1997)"}
TRUNK_SENSITIVITY_KM2 = 5.0                                       # decision 7: tiles whose largest A inside exceeds this get a trunk-sensitivity fit in addendum 18
CTRL_VERSION = "GM-FEMA2018-ctrl"                                  # decision 4
CTRL_TILES = ["GM1", "GM2", "GM3", "GM4", "GM5", "GM6", "GM7"]
CTRL_BUFFER_M = 500.0                                              # section 2.2.1: tiles + upstream catchments on the NED box routing, plus 500 m
FEMA_SURVEY = GM_GATE_SURVEYS[0]                                   # CA_FEMA_Z4_B2_2018
FEMA_RECORD = os.path.join(SOURCE, "GM_FEMA2018", "ept_class2_laz_utm10n")
FEMA_EPT = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_FEMA_Z4_B2_2018/ept.json"
FEMA_VERTICAL_EVIDENCE = ("USGS WESM work unit CA_FEMA_Z4_B2_2018 (project CA_AZ_FEMA_R9_Lidar_2017_D18): vertical CRS EPSG:6360 (NAVD88 height, ftUS) "
                          "GEOID12B on the source; the Entwine copy carries EPSG:3857 and no vertical CRS; its Z is in metres by the rule 5.3 / NCALM checks "
                          "of a16_gmf_check.py (addendum 16 section 3.3.3; re-run on the control grid in addendum 17 section 2.2.2)")
# section 3: figure conventions
HILLSHADE = {"azimuth_deg": 315.0, "altitude_deg": 45.0, "z_factor": 1.0}
OVERVIEW_BLOCK_M = {"GM": 5, "AP": 10, "TTV": 5}
A_NET_FIG_M2 = 1000.0
KIND_COLORS = {"original": "#d62728", "drawn": "#1f77b4", "M&D": "#2ca02c", "control": "#ff7f0e"}
KIND_LABELS = {"original": "original", "drawn": "drawn", "M&D": "M&D by design", "control": "control (FEMA 2018)"}
SURVEY = {"GM-x": "NCALM 2003", "GM-FEMA2018-x": "FEMA 2018", "GM-FEMA2018-ctrl": "FEMA 2018", "AP-3DEP-x": "3DEP 2019-20", "TTV-x": "Marin 2019"}
VERSIONS_BY_SITE = {"GM": ["GM-x", "GM-FEMA2018-x", "GM-FEMA2018-ctrl"], "AP": ["AP-3DEP-x"], "TTV": ["TTV-x"]}


def run_dir(name: str | None = None) -> str:
    rid = os.environ.get("A17_RUN") or name
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a17_register_figures"
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path


def hillshade(z, dx: float, azimuth: float = HILLSHADE["azimuth_deg"], altitude: float = HILLSHADE["altitude_deg"], z_factor: float = HILLSHADE["z_factor"]):
    """Horn hillshade in [0, 1] (NaN where z is NaN or at the border of the array); no vertical exaggeration by default."""
    import numpy as np
    z = np.asarray(z, dtype=np.float64)
    out = np.full(z.shape, np.nan)
    if z.shape[0] < 3 or z.shape[1] < 3:
        return out
    c = z[1:-1, 1:-1]
    dzdx = ((z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])) / (8 * dx)
    dzdy = ((z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])) / (8 * dx)
    dzdx *= z_factor
    dzdy *= z_factor
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdy, dzdx)        # rows increase southward: -dzdy is the northward gradient component
    zen = np.deg2rad(90.0 - altitude)
    az = np.deg2rad(360.0 - azimuth + 90.0)
    hs = np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(az - aspect)
    hs = np.clip(hs, 0, 1)
    hs[~np.isfinite(c)] = np.nan
    out[1:-1, 1:-1] = hs
    return out
