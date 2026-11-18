"""Addendum 16 (tile set finish): shared constants for the a16_* scripts.

Decisions of 2026-11-18 (addendum 16, section 1), fixed before any result:
  1. TTV study-area polygon = digitised M&D polygon  ∩ {z >= 61 m on the TTV 1 m grid}, largest connected component,
     holes filled; rule 2.2 of addendum 15 re-run (area 2.1 km2 +- 15 % binding; z_min passes by construction).
  2. TTV k_s reference = the pooled through-origin chi-z slope over the channel points (A >= 1e4 m2) of the clipped polygon
     on the 10 m NED at the polygon's legacy-chi theta; the per-basin median is reported beside it. |C_ht| by the stage A
     method. GM and AP keep the legacy references.
  3. TTV re-draw under the addendum 15 rules, seed 20261118, from scratch; widening once; 30 tiles in all including the
     M&D windows.
  4. GM drainage-area rule: GM1-GM5 modelling area = the existing tile grids if the entry check passes (old - new area
     against the coarse-grid inflow within 10 %), else new area + coarse inflow at the entry cells; registered as role
     area_model. New GM, TTV and AP tiles use the full-grid areas.
  5. GM lidar extension gate: a survey must add >= 10 eligible windows (addendum 15 rules, 10 m NED) not overlapping
     GM1-GM7; draw seed 20261119 if the gate passes.
  6. D8 border guard in lemcvt (no v2 result may change).
  8. Budget: <= 30 core-hours expected, <= 60 GB new storage.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a14_common import GROUP_HOME, LEGACY_REF, RESULTS_ROOT, SCRATCH, SOURCE, dump, sha256_file, slurm_record, write_tif  # noqa: E402,F401
from a15_common import (  # noqa: E402,F401
    A_NET_1M,
    MIN_ELIGIBLE,
    NODATA,
    STAGE_A,
    TARGET_NEW,
    THRESHOLDS,
    TILE_EPSG,
    TOL_REL,
    TOL_REL_WIDE,
    Z_RANGE,
    rss_gb,
)

STAGE_B = os.path.join(RESULTS_ROOT, "20261118T160326Z_a15_stage_b")
SEED_TTV = 20261118          # decision 3: the TTV re-draw keeps the addendum 15 seed, from scratch
SEED_GM_EXT = 20261119       # decision 5 / section 3.3.5: the GM extension draw, written before any draw
TTV_CLIP_Z = 61.0            # decision 1: the paper's lower bound (200 ft contour of the quadrangle)
TTV_TARGET_TOTAL = 30        # decision 3: 30 TTV tiles in all, M&D windows included
GM_GATE_MIN_ELIGIBLE = 10    # decision 5 / section 3.2
GM_AREA_CHECK_TOL = 0.10     # decision 4 / section 4: |old - new - coarse inflow| / coarse inflow <= 10 %
MD_TABLE1_PAGE = {"pdf_page": 4, "journal_page": 1910, "rows_expected": 63, "note": "68 channelled hollows minus the 5 trail-affected heads omitted from the analysis"}
GM_LANDSCAPE_BOX_UTM = (675000.0, 3955000.0, 725000.0, 4005000.0)   # x0, y0, x1, y1 (EPSG:32610) of the NED working box around the GM tiles
NED_CELLS_GM = {"n36w121": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n36w121/USGS_13_n36w121.tif",
                "n37w121": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n37w121/USGS_13_n37w121.tif"}
ENTWINE_RESOURCES = "https://usgs.entwine.io/boundaries/resources.geojson"
WESM_CSV = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WESM.csv"
GM_SURVEYS = ["CA_FEMA_Z4_B2_2018", "CA_FEMAR9Estrella_2019", "CA_CaliforniaGaps_1_B23", "CA_CaliforniaGaps_3_B23",
              "ARRA-CA_CentralCoast-Z4_2010", "USGS_LPC_CA_WestCoastElNinoUTM10_2016_LAS_2017"]
NOAA_TILEINDEX = {"9269": "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/9269/tileindex_2018_ca_fema_region9_m9269.gpkg",
                  "9270": "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/9270/tileindex_2018_ca_fema_Ql1_m9270.gpkg",
                  "9263": "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/9263/tileindex_2019_FEMA_ca_estrella_m9263.gpkg"}
NOAA_INPORT = {"9269": "64466 (2018 FEMA Lidar: Region 9, CA; QL2, NPS 0.7 m; 2018-01-22 to 2018-04-21)",
               "9270": "64465 (2018 FEMA Lidar: Region 9, CA (QL1), Salinas AOI 12.6 sq mi; NPS 0.35 m; 2018-02-16 to 02-18)",
               "9263": "64364 (2019 FEMA Lidar: Estrella, CA; QL2, NPS 0.71 m; 2019-12-15 to 2020-01-04)"}


def run_dir(name: str | None = None) -> str:
    rid = os.environ.get("A16_RUN") or name
    if rid is None:
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_a16_tile_finish"
    path = os.path.join(RESULTS_ROOT, rid)
    os.makedirs(path, exist_ok=True)
    return path

# Section 3.2 (written before the gate was computed, after the footprint map of section 3.1 was inspected): the "Gabilan Mesa
# landscape" of the directive = the dissected mesa surface east of the Salinas River around the GM tiles, taken as the
# hand-drawn polygon below (EPSG:32610; south-west side along the edge of the Salinas valley floor as seen on the NED
# hillshade, north-east side along the trend of the mesa about 12-16 km from the tiles' axis) intersected with the disc of
# radius 15 km about the GM tiles' centroid. The gate count is also reported for the disc alone and the polygon alone.
GM_LANDSCAPE_POLY_UTM = [(690000, 4005000), (692500, 3996000), (693500, 3988000), (693000, 3980000), (693500, 3975000), (696500, 3970000),
                         (701000, 3965000), (705000, 3960000), (715000, 3968000), (716000, 3980000), (712000, 3992000), (706000, 4002000),
                         (700000, 4005000), (690000, 4005000)]
GM_LANDSCAPE_RADIUS_M = 15000.0
GM_GATE_SURVEYS = ["CA_FEMA_Z4_B2_2018", "ARRA-CA_CentralCoast-Z4_2010"]   # the two that reach the landscape (section 3.1)
GM_GATE_CROP_UTM = (683000.0, 3958000.0, 718000.0, 4005000.0)              # x0, y0, x1, y1: the NED crop routed for the gate (top = the NED box top; a first run used 4,006,000 and read a clipped array against an unclipped transform, a 1 km shift)
